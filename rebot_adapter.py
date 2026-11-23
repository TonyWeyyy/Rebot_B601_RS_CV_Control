#!/usr/bin/env python3
"""
reBot Arm B601-RS 机械臂适配层
替代原项目的 Panthera_lib.Panthera，对外暴露几乎相同的接口，
使得视觉伺服主程序（palm_follow_grasp.py）里的控制数学可以原样保留。

设计：
  ┌─ 运动下发：rebot_control（LAN-GER/rebot_control，MIT 位置控制 + 关节限速）
  │     原项目的 robot.Joint_Vel(关节速度) 在这里实现为：
  │     每个控制周期把关节速度积分成位置目标 q_target += dq*dt，
  │     再调 set_joint_angles()，由库内部 200Hz 平滑斜坡执行。
  │     效果等价于速度伺服，且天然带每关节速度上限保护。
  │
  ├─ 夹爪（J7）：通过 set_joint_angles 的第 7 个通道下发（单位：度），
  │     用 set_gripper_deg() 设置目标角。与 6 关节共用同一 CAN 主机，
  │     不存在第二个 motorbridge 连接争抢电机 7 的问题。
  │
  └─ 运动学/动力学：Pinocchio 直接加载 reBotArm_control_py 仓库里的 URDF
        （原项目 Panthera 也是这么做的，FK/雅可比/重力项全部对齐）

依赖：
    git clone https://github.com/LAN-GER/rebot_control.git
    pip install motorbridge pynput pyyaml pinocchio numpy
    URDF 来自 https://github.com/Seeed-Projects/reBotArm_control_py 的 urdf/ 目录

⚠️ 使用前先核对：
    1. 你机器上 rebot_control 的实际类名/方法名（不同版本可能有出入，
       本文件所有对机械臂 SDK 的调用都集中在 _ArmIO 类里，只需改这一个类）
    2. URDF 路径与关节名称（B601_RS_JOINT_NAMES）
    3. 单位：rebot_control 用【度】，Pinocchio 用【弧度】，本适配层对外一律用弧度

⚠️ 安全停机约定（2026-09 事故后修订）：
    stop() 必须走 rebot_control 的「缓慢回零 → 到位后失能」完整流程
    （return_to_zero=True, wait=True）。
    严禁在任何非零位/非重力平衡位姿下直接失能电机——
    屈肘姿态下失能 = 机械臂自由下落（已发生过真实事故）。
"""

import os
import sys
import threading
import time
import numpy as np

try:
    import pinocchio as pin
except ImportError:
    raise ImportError("pip install pinocchio")

# ─────────────────────────────────────────────────────────────
# 需要你按实际环境修改的配置
# ─────────────────────────────────────────────────────────────
# rebot_control 仓库（git clone 使用，非 pip 包，需把仓库根目录加入 sys.path）
REBOT_CONTROL_PATH = os.path.expanduser(
    "~/wey/rebot_developer/MIT_control/rebot_control")

URDF_PATH = "~/wey/rebot_developer/reBotArm_control_py/urdf/RS/urdf/ReBot_Arm_RS.urdf"

# B601 RS 的 6 个关节在 URDF 里的名称（按 1→6 顺序，已在 ReBot_Arm_RS.urdf 中核对）
B601_RS_JOINT_NAMES = ["joint1", "joint2", "joint3",
                       "joint4", "joint5", "joint6"]

# 软件关节限位（弧度）。rebot_control 明确声明【不含软件限位】，必须在这里加。
# 先用 URDF 里的 limit 再向内收 10° 作为保守值，跑顺后再放宽。
JOINT_LIMIT_MARGIN = np.deg2rad(10.0)

# ── 关节零位修正（2026-09-30 手眼+零位联合标定实测）──────────────
# SDK 上报的关节角零位与 URDF 的关节零位存在固定小偏差
# （实测约 J2=0.75° J3=0.46° J4=-2.06° J5=-2.21°）。
# 不加修正时 FK 平移与实物系统性偏差 ~30%，手眼标定/视觉伺服全挂。
# 修正只作用于 FK/雅可比（_build_q_full），不影响下发给 SDK 的角度。
import json as _json
_JOINT_OFFSET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "config", "joint_offset.json")
if os.path.exists(_JOINT_OFFSET_FILE):
    with open(_JOINT_OFFSET_FILE) as _f:
        JOINT_OFFSET = np.array(_json.load(_f)["joint_offset_rad"],
                                dtype=np.float64)
    print(f"[RebotAdapter] 已加载关节零位修正 (度): "
          f"{np.rad2deg(JOINT_OFFSET).round(2).tolist()}")
else:
    JOINT_OFFSET = np.zeros(6)
    print("[RebotAdapter] ⚠️ 未找到 joint_offset.json，FK 将不带零位修正。"
          "请先运行 handeye_joint_calib.py")

# 每关节最大速度（弧度/秒）——双保险之一（控制层也有限幅）
# 默认全轴保守为 2.0 rad/s；开启 L2 分关节差异化限速时，由控制层在初始化时通过
# set_joint_speed_limits() 下发，但一律强制钳位在 HARD_MAX_JOINT_SPEED 安全包络内：
# - 承重抗重力关节 J2/J3 严禁超过 2.0 rad/s（防堵转、防重臂过冲）；
# - 水平偏航底座 J1（无重力负载）硬上限 2.5 rad/s；
# - 轻负载手腕 RS00 电机 J4/J5（额定 ~10.5 rad/s）硬上限 3.0 rad/s，J6 硬上限 2.5 rad/s。
MAX_JOINT_SPEED = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
HARD_MAX_JOINT_SPEED = np.array([2.5, 2.0, 2.0, 3.0, 3.0, 2.5], dtype=np.float64)

# ── 夹爪（J7）参数 ──────────────────────────────────────────
# J7 与 6 个关节共用同一条 CAN 总线、同一个 rebot_control 控制器，
# 夹爪目标角通过 set_joint_angles 的第 7 个通道下发（单位：度）。
# 捏合跟随需要夹爪有足够响应速度（RS00 参考 vlim≈3 rad/s ≈ 172°/s）。
GRIPPER_SPEED_DEG_S = 500.0                 # 夹爪最大速度（度/秒）
GRIPPER_SOFT_LIMIT_DEG = (-100.0, 400.0)    # 角度硬限幅（兜底保护；
                                            # 正常开合范围由 config/grasp.yaml 标定值决定，
                                            # 且应落在本范围内）

# 控制周期（位置目标积分步长）
CTRL_DT = 0.02   # 仅作 dt 钳制参照；实际积分用墙上时钟，主循环 100Hz 下行为不变


# ─────────────────────────────────────────────────────────────
# 底层 IO：所有与 rebot_control SDK 相关的调用只出现在这个类里
# ─────────────────────────────────────────────────────────────
class _ArmIO:
    def __init__(self):
        # rebot_control 是源码仓库（无 pip 包），真实包名为 rebot，
        # 控制器类为 ReBotRSMITController（已在本地仓库核对：
        # set_joint_angles 收 7 个【度】(6关节+夹爪)、set_max_speeds 单位度/秒、
        # read_joint_angles 是同步 CAN 读、stop(return_to_zero, wait)）
        if REBOT_CONTROL_PATH not in sys.path:
            sys.path.insert(0, REBOT_CONTROL_PATH)
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from rebot import ReBotRSMITController
        # v13 方案 A：本仓库内子类化扩展（CAN 瞬时故障容错 + 200Hz 反馈缓存），
        # 上游源码不改；版本/结构校验不通过时 build_controller 自动回退上游原版。
        from mit_controller_ext import build_controller  # noqa: E402
        self.arm, self.ext_info = build_controller(ReBotRSMITController)
        self.ext_cfg = self.ext_info["cfg"]
        # enable_esc=False：关闭 SDK 的 Esc 监听（与主程序 Esc/q 退出路径竞争）。
        # 注意：install_signal_handlers 保持上游默认 True —— SIGINT/SIGTERM 仍由 SDK
        # 接管：第一次 Ctrl+C = 后台缓慢回零后失能；回零途中第二次 Ctrl+C = 立即中止
        # 回零并失能（用户已确认保留此操作习惯，手边有机械臂电源开关）。
        # 正常退出走 palm_follow_grasp 的 q/Esc 路径，最终调用本类 stop()（回零→保持→失能）。
        self.arm.start(enable_esc=False)        # start 内部自动 connect
        # 安全速度包络（度/秒）；第 7 项为夹爪
        self.arm.set_max_speeds(np.rad2deg(MAX_JOINT_SPEED).tolist()
                                + [GRIPPER_SPEED_DEG_S])
        # 夹爪（J7）目标角：初始化时读一次实际角度作为初始目标（防止使能瞬间
        # 夹爪猛动），之后由 set_gripper_deg() 更新。
        # 同步 CAN 读（read_joint_angles 是阻塞式 CAN 往返，官方建议 ≤30Hz）
        try:
            self.gripper_deg = float(self.arm.read_joint_angles()[6])
        except Exception:
            self.gripper_deg = 0.0

    def send_joint_positions_deg(self, q_deg_7):
        self.arm.set_joint_angles(q_deg_7)      # 7 个数：6 关节 + 夹爪保持（纯位置，dq_des=0）

    def send_joint_mit_targets_deg(self, q_deg_7, dq_deg_s_7):
        """P1：透传 7 路目标角与前馈角速度（扩展不可用时自动退化为 set_joint_angles）。"""
        setter = getattr(self.arm, "set_joint_mit_targets", None)
        if setter is not None:
            setter(q_deg_7, dq_deg_s_7)
        else:
            self.arm.set_joint_angles(q_deg_7)

    def read_joint_positions_deg(self):
        return np.array(self.arm.read_joint_angles()[:6], dtype=np.float64)

    def feedback_snapshot(self):
        """扩展控制器 200Hz 反馈缓存（内存读取，无 CAN I/O）；不可用时返回 None。"""
        getter = getattr(self.arm, "get_feedback_snapshot", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def loop_stats(self):
        getter = getattr(self.arm, "get_loop_stats", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def stop(self):
        """
        安全停机：缓慢回零 → 确认到位 → 才失能电机。

        ★★★ 绝对禁止改回 return_to_zero=False ★★★
        return_to_zero=False 会在当前位姿（可能是屈肘抬起等需要力矩
        对抗重力的姿态）直接失能电机，导致机械臂自由下落。

        wait=True 同样不可省略：shutdown 是守护线程，不等它跑完主线程退出时，
        解释器会把「失能电机/关闭 CAN」杀在半路上。
        """
        try:
            self.arm.stop(return_to_zero=True, wait=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# 对外适配器：接口对齐 Panthera
# ─────────────────────────────────────────────────────────────
class RebotAdapter:
    def __init__(self, urdf_path=URDF_PATH, joint_names=B601_RS_JOINT_NAMES):
        self.io = _ArmIO()

        # Pinocchio 模型（运动学 + 重力项）
        self.model = pin.buildModelFromUrdf(os.path.expanduser(urdf_path))
        self.data = self.model.createData()
        self.joint_names = joint_names
        self.motor_count = 6

        # P3 M2：独立 Pinocchio 重力动力学模型与数据（绝不与伺服 FK 共享 self.data）
        # 可附加腕部 Gemini 2 相机 + 支架等效质点，精确覆盖全工作区 J2/J3/J4 重力矩。
        self._grav_lock = threading.Lock()
        self._grav_model = pin.buildModelFromUrdf(os.path.expanduser(urdf_path))
        ext_cfg = getattr(self.io, "ext_cfg", {}) or {}
        payload_m = float(ext_cfg.get("gravity_payload_mass_kg", 0.0))
        payload_com = np.asarray(
            ext_cfg.get("gravity_payload_com_link6_m", [0.0, 0.0, 0.0]),
            dtype=np.float64)
        if payload_m > 1e-6:
            jid6 = self._grav_model.getJointId(self.joint_names[-1])
            inertia = pin.Inertia(payload_m, payload_com, np.zeros((3, 3), dtype=np.float64))
            self._grav_model.appendBodyToJoint(jid6, inertia, pin.SE3.Identity())
        self._grav_data = self._grav_model.createData()
        self._grav_idx_q = [
            self._grav_model.joints[self._grav_model.getJointId(n)].idx_q
            for n in self.joint_names]
        self._grav_idx_v = [
            self._grav_model.joints[self._grav_model.getJointId(n)].idx_v
            for n in self.joint_names]
        self._grav_q_buf = np.zeros(self._grav_model.nq, dtype=np.float64)

        # 关节限位（从 URDF 读取并内收 margin）
        self.q_min = np.zeros(6)
        self.q_max = np.zeros(6)
        for i, name in enumerate(joint_names):
            jid = self.model.getJointId(name)
            qi = self.model.joints[jid].idx_q
            self.q_min[i] = self.model.lowerPositionLimit[qi] + JOINT_LIMIT_MARGIN
            self.q_max[i] = self.model.upperPositionLimit[qi] - JOINT_LIMIT_MARGIN

        # 速度伺服的状态：当前位置目标（初始 = 当前实际位置，防止使能瞬间跳动）
        q_now = self.get_current_pos()
        self._q_target = q_now.copy()
        self._last_cmd_time = time.time()
        self._max_joint_speed = MAX_JOINT_SPEED.copy()

        # 向 MIT 扩展控制器注册 200Hz 重力前馈回调（近零位支架自动淡出至 0，安全无跳动）
        setter = getattr(self.io.arm, "set_gravity_torque_fn", None)
        if setter is not None:
            setter(self._gravity_torque_callback)

    def set_joint_speed_limits(self, limits_rad):
        """设置分关节速度上限（弧度/秒），强制钳位至 HARD_MAX_JOINT_SPEED 硬包络。"""
        arr = np.asarray(limits_rad, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(6, float(arr), dtype=np.float64)
        if arr.shape != (6,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("分关节速度上限必须为 6 维有限正数向量")
        clamped = np.clip(arr, 0.1, HARD_MAX_JOINT_SPEED)
        self._max_joint_speed = clamped.copy()
        self.io.arm.set_max_speeds(
            np.rad2deg(self._max_joint_speed).tolist() + [GRIPPER_SPEED_DEG_S])
        return self._max_joint_speed.copy()

    # ── 状态读取 ─────────────────────────────────────────────
    def get_current_pos(self):
        """
        返回 6 关节实际角度（弧度）—— 对应 Panthera.get_current_pos()
        ⚠️ 这是 7 路串行同步 CAN 读（持有 io_lock，单次约数 ms~数十 ms），
        会挤占 200Hz MIT 下发线程。只允许低频使用（初始化、标定采样、
        到位等待），50Hz 伺服热循环里请用 get_command_pos()。
        """
        return np.deg2rad(self.io.read_joint_positions_deg())

    def get_command_pos(self):
        """
        返回 6 关节"当前实际下发值"（弧度）：rebot_control 内部 200Hz 斜坡
        正在发送的角度，纯内存读取、无 CAN I/O，可在伺服热循环中每周期调用。
        伺服稳态下与实际位置差异很小；需要真值时低频调用 get_current_pos() 校准。
        """
        return np.deg2rad(np.array(self.io.arm.get_command_angles()[:6],
                                   dtype=np.float64))

    # ── v13 MIT 扩展：200Hz 反馈缓存（内存读取，无 CAN I/O，可在伺服热循环调用）──
    def mit_ext_info(self):
        """扩展控制器状态：{'active','phase','reason','upstream_sha256','cfg'}。"""
        return dict(getattr(self.io, "ext_info", {"active": False}))

    def get_mit_loop_stats(self):
        return self.io.loop_stats()

    def get_mit_gains(self):
        """上游配置中的 6 关节 MIT (kp, kd)，供探针/辨识换算预期下垂与阻尼比。"""
        motors = self.io.arm.config.motors[:6]
        return (np.array([float(m.kp) for m in motors], dtype=np.float64),
                np.array([float(m.kd) for m in motors], dtype=np.float64))

    def get_feedback_state(self, max_age_s=None):
        """
        最近一拍 MIT 应答缓存（6 关节，弧度 / 弧度每秒 / Nm）。
        返回 dict：q, dq, tau, q_cmd, t_mos(7), age_s, seq, valid；扩展不可用返回 None。
        valid=True 表示 6 关节数据齐全、有限且年龄 ≤ max_age_s。
        """
        snap = self.io.feedback_snapshot()
        if snap is None:
            return None
        if max_age_s is None:
            max_age_s = float(self.io.ext_cfg.get("feedback_max_age_s", 0.05))
        q = np.asarray(snap["q"][:6], dtype=np.float64)
        dq = np.asarray(snap["dq"][:6], dtype=np.float64)
        tau = np.asarray(snap["tau"][:6], dtype=np.float64)
        q_cmd = np.asarray(snap["cmd"][:6], dtype=np.float64)
        valid = bool(all(snap["valid"][:6])
                     and np.all(np.isfinite(q)) and np.all(np.isfinite(dq))
                     and np.all(np.isfinite(tau))
                     and 0.0 <= snap["age_s"] <= max_age_s)
        return {"q": q, "dq": dq, "tau": tau, "q_cmd": q_cmd,
                "t_mos": np.asarray(snap["t_mos"], dtype=np.float64),
                "age_s": float(snap["age_s"]), "seq": int(snap["seq"]),
                "valid": valid}

    def get_current_pos_cached(self, max_age_s=None):
        """
        非阻塞实际角（弧度）。仅当 mit_ext.feedback_trusted=true（实机探针验证通过）
        且缓存新鲜时返回，否则返回 None，调用方必须回退阻塞式 get_current_pos()。
        """
        if not self.io.ext_cfg.get("feedback_trusted", False):
            return None
        state = self.get_feedback_state(max_age_s)
        if state is None or not state["valid"]:
            return None
        return state["q"].copy()

    def start_feedback_recording(self):
        starter = getattr(self.io.arm, "start_recording", None)
        if starter is None:
            raise RuntimeError("MIT 扩展未启用，无法录波")
        starter()

    def stop_feedback_recording(self):
        """返回录波 numpy 字典（6 关节）：t, q_cmd, q, dq, tau。"""
        rows = self.io.arm.stop_recording()
        if not rows:
            empty = np.zeros((0, 6))
            return {"t": np.zeros(0), "q_cmd": empty, "q": empty,
                    "dq": empty, "tau": empty}
        t = np.array([r[0] for r in rows], dtype=np.float64)
        return {"t": t - t[0],
                "q_cmd": np.array([r[1][:6] for r in rows], dtype=np.float64),
                "q": np.array([r[2][:6] for r in rows], dtype=np.float64),
                "dq": np.array([r[3][:6] for r in rows], dtype=np.float64),
                "tau": np.array([r[4][:6] for r in rows], dtype=np.float64)}

    # ── 标定/辨识工具专用：直接位置目标与临时限速（伺服主程序不使用）──
    def set_joint_targets_direct(self, q_rad, dq_rad=None):
        """直接设定 6 关节位置目标（弧度，钳位软限位）；可选传入前馈角速度 dq_rad（rad/s）。"""
        q = np.asarray(q_rad, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("关节目标必须为 6 维有限向量")
        q_clipped = np.clip(q, self.q_min, self.q_max)
        self._q_target = q_clipped
        self._last_cmd_time = time.time()
        q_deg = np.rad2deg(self._q_target).tolist()
        q_deg.append(self._gripper_keep_deg())
        if dq_rad is None:
            self.io.send_joint_positions_deg(q_deg)
        else:
            dq = np.asarray(dq_rad, dtype=np.float64)
            if dq.shape != (6,) or not np.all(np.isfinite(dq)):
                raise ValueError("前馈速度必须为 6 维有限向量")
            dq = np.clip(dq, -HARD_MAX_JOINT_SPEED, HARD_MAX_JOINT_SPEED)
            hit_lo = (q <= self.q_min) & (dq < 0.0)
            hit_hi = (q >= self.q_max) & (dq > 0.0)
            dq = np.where(hit_lo | hit_hi, 0.0, dq)
            dq_deg_s = np.rad2deg(dq).tolist() + [0.0]
            self.io.send_joint_mit_targets_deg(q_deg, dq_deg_s)
        return self._q_target.copy()

    def override_joint_speed_limits(self, limits_rad):
        """临时下发 200Hz 斜坡限速（弧度/秒），强制钳位 HARD_MAX_JOINT_SPEED；
        不修改伺服用 _max_joint_speed，结束后必须调用 restore_joint_speed_limits()。"""
        arr = np.asarray(limits_rad, dtype=np.float64)
        if arr.shape != (6,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("限速必须为 6 维有限正数向量")
        clamped = np.clip(arr, 0.01, HARD_MAX_JOINT_SPEED)
        self.io.arm.set_max_speeds(np.rad2deg(clamped).tolist() + [GRIPPER_SPEED_DEG_S])
        return clamped

    def restore_joint_speed_limits(self):
        self.io.arm.set_max_speeds(np.rad2deg(self._max_joint_speed).tolist()
                                   + [GRIPPER_SPEED_DEG_S])

    # ── 运动指令 ─────────────────────────────────────────────
    def joint_vel_cmd(self, dq_rad):
        """
        对应 Panthera.Joint_Vel(dq)：关节速度指令。
        实现方式：积分成位置目标，并在 P1 扩展启用时同步下发速度前馈 dq_des（夹爪 J7 恒为 0）。
        dq_rad: 6 维关节速度（rad/s），None 或全零 = 保持
        """
        now = time.time()
        # dt 上限钳制到 3 个控制周期：hold()/move_to_joints() 之后或循环卡顿
        # 时若不钳制，积分项 dq*dt 会让目标位置跳变一大步（安全隐患）
        dt = min(max(now - self._last_cmd_time, 1e-3), 3 * CTRL_DT)
        self._last_cmd_time = now

        dq_send = np.zeros(6, dtype=np.float64)
        if dq_rad is not None:
            dq = np.asarray(dq_rad, dtype=np.float64)
            lim = self._max_joint_speed
            ratios = np.abs(dq) / lim
            max_ratio = float(np.max(ratios)) if dq.size else 0.0
            if max_ratio > 1.0:
                dq = dq / max_ratio
            dq = np.clip(dq, -lim, lim)
            q_next = self._q_target + dq * dt
            # 软限位保护：撞到边界就把目标钉在边界上，并立即清零该轴速度前馈
            hit_lo = (q_next <= self.q_min) & (dq < 0.0)
            hit_hi = (q_next >= self.q_max) & (dq > 0.0)
            dq_send = np.where(hit_lo | hit_hi, 0.0, dq)
            self._q_target = np.clip(q_next, self.q_min, self.q_max)

        q_deg = np.rad2deg(self._q_target).tolist()
        q_deg.append(self._gripper_keep_deg())     # 夹爪保持当前值
        dq_deg_s = np.rad2deg(dq_send).tolist()
        dq_deg_s.append(0.0)                       # J7 夹爪速度前馈恒为 0
        self.io.send_joint_mit_targets_deg(q_deg, dq_deg_s)

    def hold(self):
        """保持当前位置（丢手/暂停时调用）。用内部下发值重锚定，无 CAN I/O。"""
        self._q_target = self.get_command_pos().copy()
        self._last_cmd_time = time.time()   # 重置积分时钟，防止恢复后 dt 过大
        q_deg = np.rad2deg(self._q_target).tolist()
        q_deg.append(self._gripper_keep_deg())
        self.io.send_joint_positions_deg(q_deg)

    def move_to_joints(self, q_rad, speed_deg_s=20.0, wait=True, timeout=15.0):
        """对应 Panthera.Joint_Pos_Vel：回零/回初始位用"""
        self._q_target = np.clip(np.asarray(q_rad, dtype=np.float64),
                                 self.q_min, self.q_max)
        self.io.arm.set_max_speeds([speed_deg_s] * 6 + [GRIPPER_SPEED_DEG_S])
        q_deg = np.rad2deg(self._q_target).tolist()
        q_deg.append(self._gripper_keep_deg())
        self.io.send_joint_positions_deg(q_deg)
        if wait:
            t0 = time.time()
            while time.time() - t0 < timeout:
                # 到位判断用内部下发值（斜坡走完即到位），低频真值复核一次。
                # S3：优先用已验证的 200Hz 反馈缓存（不占 io_lock、不阻塞 MIT 帧），
                # 缓存未验证/过期时才回退阻塞式 0x7019 串行读取。
                if np.max(np.abs(self.get_command_pos() - self._q_target)) < np.deg2rad(1.0):
                    q_act = self.get_current_pos_cached()
                    if q_act is None:
                        q_act = self.get_current_pos()
                    if np.max(np.abs(q_act - self._q_target)) < np.deg2rad(2.0):
                        break
                time.sleep(0.1)
        # 恢复伺服速度包络
        self.io.arm.set_max_speeds(np.rad2deg(self._max_joint_speed).tolist()
                                   + [GRIPPER_SPEED_DEG_S])
        self._last_cmd_time = time.time()   # 重置积分时钟，防止之后 dt 过大

    def stop(self):
        self.io.stop()

    def _gripper_keep_deg(self):
        # 夹爪（J7）目标角：初始化时读取并缓存，之后由 set_gripper_deg() 更新。
        # 用缓存值随每个控制周期下发，避免每周期多发一次同步 CAN 读。
        return self.io.gripper_deg

    # ── 夹爪（J7）控制 ─────────────────────────────────────
    def set_gripper_deg(self, deg):
        """设定夹爪（J7）目标角度（度），随下一次 50Hz 指令下发生效。

        带硬限幅兜底。正常开合范围请先运行 gripper_j7_calibrate.py
        实测标定（结果写入 config/grasp.yaml），不要凭猜填写。
        """
        lo, hi = GRIPPER_SOFT_LIMIT_DEG
        self.io.gripper_deg = float(np.clip(deg, lo, hi))

    def get_gripper_deg(self):
        """当前夹爪目标角度（度）"""
        return float(self.io.gripper_deg)

    # ── 运动学（Pinocchio，单位弧度/米）────────────────────
    def _build_q_full(self, joint_positions):
        """SDK 关节角 → pinocchio q_full。★ 加 JOINT_OFFSET 零位修正 ★"""
        q_full = np.zeros(self.model.nq)
        for i, name in enumerate(self.joint_names):
            jid = self.model.getJointId(name)
            q_full[self.model.joints[jid].idx_q] = \
                joint_positions[i] + JOINT_OFFSET[i]
        return q_full

    def forward_kinematics(self):
        """对应 Panthera.forward_kinematics() → {'position','rotation'}"""
        q = self._build_q_full(self.get_current_pos())
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jid = self.model.getJointId(self.joint_names[-1])
        oMi = self.data.oMi[jid]
        return {'position': oMi.translation.copy(),
                'rotation': oMi.rotation.copy()}

    def gravity_torque(self, q_rad, include_payload=False):
        """对应 Panthera.get_Gravity(q)：广义重力项。
        include_payload=False：原始 URDF（兼容历史标定脚本）；
        include_payload=True：含腕部 Gemini 2 相机附加质量（P3 重力前馈用，线程安全独立 Data）。
        """
        if include_payload:
            return np.asarray(self._gravity_torque_callback(q_rad), dtype=np.float64)
        q_full = self._build_q_full(q_rad)
        g_full = pin.computeGeneralizedGravity(self.model, self.data, q_full)
        cols = [self.model.joints[self.model.getJointId(n)].idx_v
                for n in self.joint_names]
        return g_full[cols]

    def _gravity_torque_callback(self, q_rad):
        """200Hz MIT 线程专用静态重力矩回调（使用独立 _grav_model/_grav_data，耗时 ~0.02ms）。"""
        with self._grav_lock:
            self._grav_q_buf[:] = 0.0
            for i in range(6):
                self._grav_q_buf[self._grav_idx_q[i]] = float(q_rad[i]) + float(JOINT_OFFSET[i])
            g_full = pin.computeGeneralizedGravity(
                self._grav_model, self._grav_data, self._grav_q_buf)
            return [float(g_full[idx]) for idx in self._grav_idx_v]

    def set_reach_radius(self, reach_r_m):
        """P3 M3：把当前 TCP 水平伸展半径 (m) 通知给 200Hz 阻尼调度器。"""
        setter = getattr(self.io.arm, "set_reach_radius", None)
        if setter is not None:
            setter(reach_r_m)

    def get_safety_status(self):
        """P4 M6：读取 200Hz 主动安全监督层状态（无锁阻塞）；扩展未启用返回 None。"""
        getter = getattr(self.io.arm, "get_safety_status", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def clear_safety_fault(self):
        """P4 M6：用户排查后按 g 重新启动跟随时清除软停机锁存。"""
        clearer = getattr(self.io.arm, "clear_safety_fault", None)
        if clearer is not None:
            clearer()
