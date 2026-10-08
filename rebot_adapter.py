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
# 左右跟随的速度上限 ≈ J1 限速 × 臂展，这是横向跟随的物理瓶颈。
# 已从保守值 0.6 提到 0.8；RS06 额定转速远高于此，若想更快可逐步
# 试 1.0 → 1.2（每次只提一档，观察电机温度与跟随稳定性）。
MAX_JOINT_SPEED = np.array([0.8, 0.8, 0.8, 1.0, 1.2, 1.2])

# ── 夹爪（J7）参数 ──────────────────────────────────────────
# J7 与 6 个关节共用同一条 CAN 总线、同一个 rebot_control 控制器，
# 夹爪目标角通过 set_joint_angles 的第 7 个通道下发（单位：度）。
# 捏合跟随需要夹爪有足够响应速度（RS00 参考 vlim≈3 rad/s ≈ 172°/s）。
GRIPPER_SPEED_DEG_S = 150.0                 # 夹爪最大速度（度/秒）
GRIPPER_SOFT_LIMIT_DEG = (-100.0, 400.0)    # 角度硬限幅（兜底保护；
                                            # 正常开合范围由 config/grasp.yaml 标定值决定，
                                            # 且应落在本范围内）

# 控制周期（位置目标积分步长）
CTRL_DT = 0.02   # 50Hz 对跟随足够；rebot_control 内部以 200Hz 平滑执行


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
        from rebot import ReBotRSMITController
        self.arm = ReBotRSMITController()
        self.arm.start(enable_esc=True)         # start 内部自动 connect；Esc 触发平滑回零
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
        self.arm.set_joint_angles(q_deg_7)      # 7 个数：6 关节 + 夹爪保持

    def read_joint_positions_deg(self):
        return np.array(self.arm.read_joint_angles()[:6], dtype=np.float64)

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

    # ── 运动指令 ─────────────────────────────────────────────
    def joint_vel_cmd(self, dq_rad):
        """
        对应 Panthera.Joint_Vel(dq)：关节速度指令。
        实现方式：积分成位置目标后下发（50Hz）。
        dq_rad: 6 维关节速度（rad/s），None 或全零 = 保持
        """
        now = time.time()
        # dt 上限钳制到 3 个控制周期：hold()/move_to_joints() 之后或循环卡顿
        # 时若不钳制，积分项 dq*dt 会让目标位置跳变一大步（安全隐患）
        dt = min(max(now - self._last_cmd_time, 1e-3), 3 * CTRL_DT)
        self._last_cmd_time = now

        if dq_rad is not None:
            dq = np.clip(np.asarray(dq_rad, dtype=np.float64),
                         -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
            self._q_target = self._q_target + dq * dt
            # 软限位：撞到边界就把目标钉在边界上，防止积分饱和
            self._q_target = np.clip(self._q_target, self.q_min, self.q_max)

        q_deg = np.rad2deg(self._q_target).tolist()
        q_deg.append(self._gripper_keep_deg())     # 夹爪保持当前值
        self.io.send_joint_positions_deg(q_deg)

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
                # 到位判断用内部下发值（斜坡走完即到位），低频真值复核一次
                if np.max(np.abs(self.get_command_pos() - self._q_target)) < np.deg2rad(1.0):
                    if np.max(np.abs(self.get_current_pos() - self._q_target)) < np.deg2rad(2.0):
                        break
                time.sleep(0.1)
        # 恢复伺服速度包络
        self.io.arm.set_max_speeds(np.rad2deg(MAX_JOINT_SPEED).tolist()
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

    def gravity_torque(self, q_rad):
        """对应 Panthera.get_Gravity(q)：广义重力项（标定拖拽模式用）"""
        q_full = self._build_q_full(q_rad)
        g_full = pin.computeGeneralizedGravity(self.model, self.data, q_full)
        cols = [self.model.joints[self.model.getJointId(n)].idx_v
                for n in self.joint_names]
        return g_full[cols]
