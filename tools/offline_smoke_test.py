#!/usr/bin/env python3
"""完整主程序离线冒烟测试：假相机/检测器/机械臂替身架构，不访问任何硬件。

验证范围：
1. **替身注入架构**：在动态加载 ``palm_follow_grasp.py`` 前，通过 ``sys.modules`` 替换
   ``gemini2_camera``、``palm_detection``、``rebot_adapter`` 与 ``pynput``，并替换 FK/雅可比
   函数，实现完全脱离 Orbbec SDK、MediaPipe、Pinocchio 与 CAN 硬件的端到端执行。
2. **多版本配置与越界拒绝校验**：验证默认 ``follow.yaml``、历史版本 ``follow_v10~v13.yaml``、
   回退配置 ``follow_v9/v8/v7.1_fallback.yaml`` 与 ``mit_ext.yaml`` 均可加载并正确生效，
   同时验证超 70% 臂展与 40+ 项越界/危险配置会被 ``validate_follow_cfg()`` 明确拒绝。
3. **4 场景主程序 ``main()`` 端到端冒烟**：
   - 场景一（正常闭环）：验证起步跟随、回中心停手保持、硬件时间戳校准与异常回退、
     2Hz 实际反馈采样、200Hz MIT 缓存入 CSV 日志，以及 ``feedback_close -> safe_stop`` 退出顺序；
   - 场景二（CAN 反馈故障）：连续注入反馈异常只能关闭诊断采样，不得污染控制误差或破坏安全回零；
   - 场景三（``--no-gripper`` 夹爪避让）：验证启动时主动打开夹爪至 ``open_deg`` 防镜头遮挡；
   - 场景四（Eye-in-Hand 静止手掌）：模拟相机随机械臂平移带来的像素反向运动，验证静止手掌
     不触发自运动极限环且保持稳定锁存。
"""

import copy
import csv
import importlib.util
import os
import sys
import threading
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeCamera:
    """Orbbec Gemini 2 相机替身：60fps 输出恒定深度帧并在指定帧注入时间戳异常。"""

    max_frames = 100

    def __init__(self, *args, **kwargs):
        """初始化假相机内参（640x480, fx=fy=500）与帧计数状态。"""
        self.fx = self.fy = 500.0
        self.cx, self.cy = 320.0, 240.0
        self.width, self.height = 640, 480
        self.dropped_frames = 0
        self.frames = 0
        self.stopped = False
        self.previous_raw_s = None

    def get_frames(self):
        """按 60fps 节拍生成假 RGB/Depth 帧；超出 ``max_frames`` 抛出 ``KeyboardInterrupt`` 触发正常退出。"""
        self.frames += 1
        if self.frames > self.max_frames:
            raise KeyboardInterrupt  # 走主程序统一退出路径
        time.sleep(1.0 / 60.0)
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 350, dtype=np.uint16)
        return color, depth, 0.001

    def get_last_frame_timing(self):
        """返回模拟硬件时间戳（秒）；在第 40 帧注入重复时间戳、第 55 帧注入 +10s 跳变以测试回退逻辑。"""
        # 与主机同钟、固定20ms传输延迟，模拟 latest-only 无队列积压。
        raw_s = time.monotonic() - 0.020
        if self.frames == 40:
            raw_s = self.previous_raw_s  # 重复时间戳：必须回退主机时刻
        elif self.frames == 55:
            raw_s += 10.0                # 大跳变：必须回退主机时刻
        self.previous_raw_s = raw_s
        return raw_s, "fake_seconds"

    def stop(self):
        """标记相机已释放。"""
        self.stopped = True


class FakeDetector:
    """手掌检测器替身：输出“静止 → 向右移动 → 回中心停手”的合成像素轨迹。"""

    def __init__(self, *args, **kwargs):
        """初始化帧计数器。"""
        self.i = 0

    def detect(self, image):
        """返回当前帧假掌心像素 ``(u, v, None)``，覆盖起步解锁、巡航跟随与停手收敛全阶段。"""
        self.i += 1
        # 静止→同向移动→回中心停手，覆盖起步、巡航和停振全过程。
        u = (320 if self.i < 15 or self.i > 50
             else min(390, 320 + (self.i - 15) * 5))
        return u, 240, None

    def close(self):
        """兼容真实检测器释放接口的空实现。"""
        pass


class FakeTracker:
    """单目标手掌跟踪器替身：直接透传检测坐标并维护 ``lost=0`` 状态字典。"""

    def __init__(self):
        """初始化最近一次跟踪状态。"""
        self.track = None

    def update(self, detection):
        """根据当前帧检测结果更新并返回跟踪状态字典。"""
        if detection is None:
            return None
        u, v = detection
        self.track = {"u": float(u), "v": float(v), "lost": 0,
                      "x3d": 0.0, "y3d": 0.0, "z3d": 0.0}
        return self.track


class FakeOneEuro:
    """一欧元滤波器简化替身：以固定 0.5 系数执行一阶指数平滑。"""

    def __init__(self, *args, **kwargs):
        """初始化内部滤波状态。"""
        self.x = None

    def __call__(self, x, t=None):
        """更新并返回平滑后的标量值。"""
        self.x = float(x) if self.x is None else self.x + 0.5 * (x - self.x)
        return self.x


class FakeBridge:
    """LK 光流桥接器替身：无光流计算，直接将检测结果标记为 ``det`` 透传。"""

    def __init__(self, *args, **kwargs):
        """兼容真实光流桥接器初始化签名。"""
        pass

    def update(self, image, det):
        """将非空检测坐标包装为 ``(u, v, "det")`` 返回。"""
        return (*det, "det") if det is not None else (None, None, None)


class FakeRobot:
    """``RebotAdapter`` 机械臂替身：模拟 100Hz 速度积分、200Hz MIT 反馈缓存与生命周期事件记录。"""

    instances = []
    fail_feedback = False

    def __init__(self):
        """初始化 6 轴关节角、指令记录列表、夹爪状态及事件时序列表，并登记实例便于断言。"""
        self.q = np.zeros(6)
        self.last_dq = np.zeros(6)
        self.commands = []
        self.stopped = False
        self.gripper = 0.0
        self.feedback_calls = 0
        self.events = []
        FakeRobot.instances.append(self)

    def move_to_joints(self, *args, **kwargs):
        """模拟回 HOME 或移至指定关节角的阻塞调用（空操作）。"""
        pass

    def forward_kinematics(self):
        """返回固定安全区内的假 TCP 位姿（单位: m，单位姿态矩阵）。"""
        return {"position": np.array([0.10, 0.0, 0.30]),
                "rotation": np.eye(3)}

    def get_gripper_deg(self):
        """返回当前夹爪角度（deg）。"""
        return self.gripper

    def set_gripper_deg(self, deg):
        """记录 J7 夹爪目标角度（deg）。"""
        self.gripper = float(deg)

    def get_command_pos(self):
        """返回当前 6 轴积分指令角副本（rad）。"""
        return self.q.copy()

    def get_current_pos(self):
        """模拟同步 ``0x7019`` 反馈读取；``fail_feedback=True`` 时抛异常模拟 CAN 故障。"""
        self.feedback_calls += 1
        if FakeRobot.fail_feedback:
            raise RuntimeError("注入实际反馈 CAN 故障")
        return self.q.copy() - 0.005  # 固定小跟踪误差，仅用于日志验证

    def get_feedback_state(self):
        """返回模拟 v13 200Hz MIT 反馈缓存快照（含轻微滞后角、力矩、MOS 温度）。"""
        return {
            "q": self.q.copy() - 0.003,
            "dq": self.last_dq.copy(),
            "tau": np.array([0.0, 4.0, -2.0, 0.3, 0.1, 0.0]),
            "q_cmd": self.q.copy(),
            "t_mos": np.full(7, 38.0),
            "age_s": 0.002,
            "seq": len(self.commands) + 1,
            "valid": True,
        }

    def get_mit_loop_stats(self):
        """返回模拟 200Hz MIT 环路统计（周期 p99、CAN 异常数、P1/P3 运行指标）。"""
        return {
            "period_p99_ms": 5.2, "can_fail_total": 0,
            "dq_send_max_rad_s": float(np.max(np.abs(self.last_dq))),
            "tau_ff_nm": [0.0, 0.5, 6.2, 1.8, 0.0, 0.0],
            "cur_kd": [4.6, 13.5, 14.0, 5.0, 4.0, 4.0],
        }

    def get_safety_status(self):
        """返回正常的 P4 安全监控状态字典。"""
        return {"enabled": True, "fault": False, "reason": "", "thermal_scale": 1.0}

    def clear_safety_fault(self):
        """兼容 P4 故障清除接口。"""
        pass

    def set_reach_radius(self, reach_r_m):
        """记录上层下发的水平臂展半径（m）。"""
        self.last_reach_r = float(reach_r_m)

    def joint_vel_cmd(self, dq):
        """接收 100Hz 关节速度指令（rad/s）并按固定 10ms 步长积分更新 ``self.q``。"""
        dq = np.asarray(dq, dtype=float)
        self.last_dq = dq.copy()
        self.commands.append(dq.copy())
        self.q += dq * 0.01

    def set_joint_speed_limits(self, limits_rad):
        """记录并返回逐轴关节速度上限（rad/s）。"""
        self.max_joint_speed = np.asarray(limits_rad, dtype=float).copy()
        return self.max_joint_speed.copy()

    def hold(self):
        """模拟安全停机前的保持当前位姿动作。"""
        pass

    def stop(self):
        """记录 ``safe_stop`` 生命周期事件并标记机械臂已安全停机。"""
        self.events.append("safe_stop")
        self.stopped = True


class FakeListener:
    """``pynput.keyboard.Listener`` 替身：启动 0.10s 后自动触发一次 ``g`` 键开启跟随。"""

    def __init__(self, on_press):
        """保存按键回调并初始化定时器引用。"""
        self.on_press = on_press
        self.timer = None

    def start(self):
        """在后台线程延时 0.10s 模拟用户按下 ``g`` 键激活视觉伺服。"""
        def press_g():
            self.on_press(types.SimpleNamespace(char="g"))
        self.timer = threading.Timer(0.10, press_g)
        self.timer.daemon = True
        self.timer.start()

    def stop(self):
        """取消尚未触发的模拟按键定时器。"""
        if self.timer is not None:
            self.timer.cancel()


# 在导入主程序前替换所有硬件相关模块（gemini2_camera / palm_detection / rebot_adapter / pynput）。
camera_module = types.ModuleType("gemini2_camera")
camera_module.Gemini2Camera = FakeCamera
sys.modules["gemini2_camera"] = camera_module

palm_module = types.ModuleType("palm_detection")
palm_module.PalmDetector = FakeDetector
palm_module.PalmTracker = FakeTracker
palm_module.OneEuroFilter = FakeOneEuro
palm_module.LKPalmBridge = FakeBridge
palm_module.get_median_depth = lambda depth, scale, u, v, *args, **kwargs: 0.35
palm_module.pixel_to_3d = lambda u, v, z, fx, fy, cx, cy: (
    (u - cx) * z / fx, (v - cy) * z / fy, z)
palm_module.draw_palm = lambda *args, **kwargs: None
palm_module.draw_landmarks = lambda *args, **kwargs: None
palm_module.pinch_distance_px = lambda *args, **kwargs: (0.0, (0, 0), (0, 0))
sys.modules["palm_detection"] = palm_module

adapter_module = types.ModuleType("rebot_adapter")
adapter_module.RebotAdapter = FakeRobot
sys.modules["rebot_adapter"] = adapter_module

pynput_module = types.ModuleType("pynput")
pynput_module.keyboard = types.SimpleNamespace(
    Listener=FakeListener,
    Key=types.SimpleNamespace(space="space", esc="esc"))
sys.modules["pynput"] = pynput_module

spec = importlib.util.spec_from_file_location(
    "palm_follow_grasp_offline", os.path.join(ROOT, "palm_follow_grasp.py"))
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)


class TrackedFeedbackSampler(main_module.ActualFeedbackSampler):
    """带退出时序埋点的反馈采样器子类：用于断言 ``feedback_close`` 必须早于 ``safe_stop``。"""

    def close(self):
        """在真实关闭前向 ``robot.events`` 写入 ``feedback_close`` 事件标记。"""
        self.robot.events.append("feedback_close")
        super().close()


main_module.ActualFeedbackSampler = TrackedFeedbackSampler

# ── 配置回归校验：默认交付配置、历史版本、回退配置及 mit_ext.yaml 必须在硬件初始化前全部通过 ──
valid_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow.yaml"))
main_module.validate_follow_cfg(valid_cfg)
FAKE_DEPTH = float(valid_cfg["follow"]["distance"])
palm_module.get_median_depth = lambda depth, scale, u, v, *args, **kwargs: FAKE_DEPTH
main_module.get_median_depth = palm_module.get_median_depth
v10_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow_v10.yaml"))
main_module.validate_follow_cfg(v10_cfg)
v11_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow_v11.yaml"))
main_module.validate_follow_cfg(v11_cfg)
v12_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow_v12.yaml"))
main_module.validate_follow_cfg(v12_cfg)
main_module.apply_follow_cfg(v12_cfg)
assert not main_module.MIT_ACTUAL_FEEDBACK_EST_ENABLE
v13_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow_v13.yaml"))
main_module.validate_follow_cfg(v13_cfg)
main_module.apply_follow_cfg(v13_cfg)
assert main_module.MIT_ACTUAL_FEEDBACK_EST_ENABLE
from mit_controller_ext import load_mit_ext_cfg
load_mit_ext_cfg(os.path.join(ROOT, "config", "mit_ext.yaml"))
fallback_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow_v7.1_fallback.yaml"))
main_module.validate_follow_cfg(fallback_cfg)
main_module.apply_follow_cfg(fallback_cfg)
assert not main_module.KF_SLOW_FOLLOW_ENABLE
assert not main_module.CAMERA_HW_TIMESTAMP_ENABLE
assert not main_module.DIAG_ENABLE
assert main_module.KF_R_XY == main_module.KF_R_Z == 0.004
assert not main_module.KF_AXIS_ADAPT_ENABLE
assert not main_module.FF_ADAPTIVE_TAPER_ENABLE
assert not main_module.STATIC_VEL_HOLD_ENABLE
assert main_module.DIAG_ACTUAL_FEEDBACK_HZ == 0.0
v8_fallback_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow_v8_fallback.yaml"))
main_module.validate_follow_cfg(v8_fallback_cfg)
main_module.apply_follow_cfg(v8_fallback_cfg)
assert not main_module.KF_AXIS_ADAPT_ENABLE
assert not main_module.KF_ADAPT_ENABLE
assert not main_module.FF_ADAPTIVE_TAPER_ENABLE
assert main_module.DIAG_ACTUAL_FEEDBACK_HZ == 0.0
assert main_module.DIAG_ENABLE  # v8 回退仍保留原高频诊断
assert not main_module.STATIC_VEL_HOLD_ENABLE
v9_fallback_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow_v9_fallback.yaml"))
main_module.validate_follow_cfg(v9_fallback_cfg)
main_module.apply_follow_cfg(v9_fallback_cfg)
assert main_module.KF_AXIS_ADAPT_ENABLE
assert main_module.FF_ADAPTIVE_TAPER_ENABLE
assert not main_module.KF_AXIS_ADAPT_STATIC_GUARD
assert not main_module.STATIC_VEL_HOLD_ENABLE
main_module.apply_follow_cfg(valid_cfg)
assert main_module.KF_AXIS_ADAPT_ENABLE
assert main_module.FF_ADAPTIVE_TAPER_ENABLE
assert main_module.KF_AXIS_ADAPT_STATIC_GUARD
assert main_module.STATIC_VEL_HOLD_ENABLE
assert main_module.STATIC_VEL_HOLD_FRAMES == 3
assert main_module.DIAG_ACTUAL_FEEDBACK_HZ == 0.0

# 负向安全边界校验：超出 70% 臂展（0.42m）与各类越界伺服/速度上限配置必须抛出 ValueError
bad_cfg = copy.deepcopy(valid_cfg)
bad_cfg["workspace"]["max_radius"] = 0.43
try:
    main_module.validate_follow_cfg(bad_cfg)
except ValueError:
    pass
else:
    raise AssertionError("超过70%臂展的工作空间未被配置校验拒绝")
for key, value in (("kf_r_xy", 0.0), ("kf_slow_follow_speed", 0.07),
                   ("camera_hw_max_transport_s", 0.30),
                   ("diag_actual_feedback_hz", 2.1),
                   ("ff_taper_max_dist", 0.06),
                   ("ff_taper_speed_ref", 0.0),
                   ("kf_axis_adapt_ref_xy", 0.0),
                   ("kf_axis_adapt_max_xy", 2.1),
                   ("kf_axis_adapt_max_z", 1.6),
                   ("kf_axis_adapt_confirm", 0),
                   ("kf_static_q_scale", 1.1),
                   ("reversal_residual_scale", 1.1),
                   ("static_velocity_hold_speed", 0.0),
                   ("static_velocity_hold_speed", 0.09),
                   ("static_velocity_hold_delay_s", 0.11),
                   ("self_motion_delay_s", 0.30),
                   ("self_motion_lpf_tau_s", -0.01),
                   ("self_motion_local_window_s", 0.15),
                   ("ff_cruise_support", 0.95),
                   ("post_brake_settle_s", 0.60),
                   ("post_brake_ext_extra_s", 0.40),
                   ("kf_ff_stop_delay_s", 0.20),
                   ("osc_err_thresh", 0.001),
                   ("osc_min_half_s", 0.50),
                   ("osc_gain_cut", 0.90),
                   ("osc_decay_tau_s", 0.02),
                   ("reach_ff_min_scale", 0.10),
                   ("reach_lead_min_scale", 1.20),
                   ("reach_accel_min_scale", 0.10),
                   ("reach_deadband_on_boost", 1.50),
                   ("reach_fast_start_boost", -0.10),
                   ("reach_release_tau", 0.005),
                   ("far_gain_boost", 0.80),
                   ("osc_confirm_reversals", 1),
                   ("osc_large_err_release", 0.05),
                   ("world_kf_pose_delay_s", 0.30),
                   ("world_kf_pose_lpf_tau_s", 0.25),
                   ("world_kf_smith_blend", 1.2),
                   ("world_kf_ff_gain", 0.95),
                   ("ori_cruise_weight_min", 0.10),
                   ("ori_soft_max_deg", 25.0),
                   ("roi_crop_scale", 0.30),
                   ("roi_crop_scale", 0.95),
                   ("mit_actual_vel_lpf_tau_s", -0.01),
                   ("mit_actual_vel_lpf_tau_s", 0.20),
                   ("max_vel", 0.90),
                   ("max_vel_joints", [2.4, 2.5, 2.0, 2.8, 2.8, 2.2]),
                   ("max_vel_joints", [2.4, 2.0, 2.0, 3.5, 2.8, 2.2]),
                   ("max_vel_joints", [2.0, 2.0, 2.0])):
    invalid = copy.deepcopy(valid_cfg)
    invalid["follow"][key] = value
    try:
        main_module.validate_follow_cfg(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"危险/无效新配置未被拒绝: {key}={value}")

# 用单位雅可比验证完整数据流，不加载 Pinocchio/URDF。
main_module.compute_tcp_fk_and_jacobian = lambda robot, q: (
    np.array([0.10, 0.0, 0.30]), np.eye(3), np.eye(6))

#  yaml 默认保持主机时刻基线；冒烟显式开启以覆盖硬件时间校准和异常回退。
original_load_follow_cfg = main_module.load_follow_cfg


def load_with_fake_hw_time(path):
    """包装配置加载器：冒烟期显式开启硬件时间戳校准与 2Hz 低频反馈采样以覆盖两条链路。"""
    cfg = original_load_follow_cfg(path)
    cfg["follow"]["camera_hw_timestamp_enable"] = True
    cfg["follow"]["diag_actual_feedback_hz"] = 2.0
    return cfg


main_module.load_follow_cfg = load_with_fake_hw_time
log_dir = os.path.join(ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)


def run_complete_smoke(extra_args=()):
    """运行一次无头主程序 ``main()`` 并执行通用生命周期、静止保持与诊断日志断言。

    参数:
        extra_args (tuple[str, ...]): 追加传给 ``palm_follow_grasp.py`` 的额外命令行参数
            （如 ``("--no-gripper",)``）。

    返回:
        tuple: ``(robot, rows, log_path, nonzero_count, slow_count)`` 供各场景继续做专项断言。
    """
    logs_before = {os.path.join(log_dir, name) for name in os.listdir(log_dir)
                   if name.endswith(".csv")}
    sys.argv = ["palm_follow_grasp.py", "--headless"] + list(extra_args)
    main_module.main()

    robot = FakeRobot.instances[-1]
    nonzero = [q for q in robot.commands if np.linalg.norm(q) > 1e-6]
    assert nonzero, "跟随已启动但没有产生任何非零关节指令"
    final_tail = robot.commands[-15:]
    assert len(final_tail) == 15, "控制周期数量不足，无法验证停手保持"
    assert all(np.linalg.norm(q) < 1e-6 for q in final_tail), \
        "假目标回中心停手后，末段仍存在非零关节指令"
    # 安全停机顺序红线：实际反馈线程必须在机械臂安全回零前先关闭，杜绝退出期 CAN 并发竞争
    assert robot.stopped, "主程序退出时没有调用统一安全停止"
    assert "feedback_close" in robot.events, "实际反馈线程未经过统一关闭"
    assert robot.events.index("feedback_close") < robot.events.index("safe_stop"), \
        "实际反馈线程必须在机械臂安全回零前关闭"
    new_logs = [os.path.join(log_dir, name) for name in os.listdir(log_dir)
                if name.endswith(".csv") and os.path.join(log_dir, name) not in logs_before]
    assert len(new_logs) == 1, "高频诊断日志没有按一次运行一个文件生成"
    with open(new_logs[0], newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) > 100, "诊断日志控制周期样本不足"
    # 慢拖通道现在在世界系目标上工作，避免相机自运动误触发；
    # 激活时必须仍处于静止状态且没有额外 lead。
    slow_rows = [row for row in rows if row["slow_active"] == "1"]
    # 位置死区内不再启动慢拖；该场景可能没有死区外慢拖样本。
    assert all(row["hold_latch"] == "1" and float(row["lead_s"]) == 0.0
               for row in slow_rows), "慢拖通道破坏了静止状态或额外 lead 互斥"
    static_rows = [row for row in rows if row["static_hold"] == "1"]
    assert static_rows, "静止锁存死区内没有进入低速残余状态保持"
    assert all(row["ff_moving"] == "0" and row["hold_latch"] == "1"
               and float(row["joint_speed"]) == 0.0 for row in static_rows), \
        "静止保持不得解锁目标或输出非零关节速度"
    stationary_rows = [row for row in rows
                       if row["ff_moving"] == "0" and row["kf_valid"] == "1"]
    assert all(float(row[f"kf_q_scale_{axis}"]) == main_module.KF_STATIC_Q_SCALE
               for row in stationary_rows for axis in "xyz"), \
        "静止观测必须使用配置的静止 Q 倍率"
    assert any(row["hw_time_used"] == "1" for row in rows), "有效硬件时间未进入校准链"
    assert any(row["hw_time_used"] == "0" for row in rows), "异常硬件时间未回退主机时刻"
    assert any(row["fbc_valid"] == "1" and row["qa_2"] and row["tau_2"]
               for row in rows), "MIT 200Hz 反馈缓存未写入诊断日志"
    assert all(not row["error"] for row in rows), f"伺服日志出现运行错误: {new_logs[0]}"
    return robot, rows, new_logs[0], len(nonzero), len(slow_rows)


# 场景一：反馈正常，验证同点实际/指令、读取耗时和跟踪误差进入日志。
FakeRobot.fail_feedback = False
robot, rows, log_path, nonzero_count, slow_count = run_complete_smoke()
feedback_rows = [row for row in rows if row["fb_valid"] == "1"]
feedback_times = {row["fb_time"] for row in feedback_rows}
assert len(feedback_times) >= 2, "2Hz 实际反馈没有产生独立样本"
assert robot.feedback_calls >= 3, "实际反馈采样线程未按预期运行"
for row in feedback_rows:
    assert row["fb_age_ms"] and row["fb_read_ms"] and row["tracking_err_max"]
    assert all(row[f"q_act_{joint}"] and row[f"q_fb_cmd_{joint}"]
               for joint in range(1, 7)), "实际角或采样点指令角日志不完整"
print(f"OFFLINE_FEEDBACK_OK: samples={len(feedback_times)}, "
      f"valid_rows={len(feedback_rows)}, calls={robot.feedback_calls}, "
      f"commands={len(robot.commands)}, nonzero={nonzero_count}, "
      f"final_static_zero=15, slow_rows={slow_count}, log={log_path}")

# 场景二：连续 CAN 反馈故障只能停用诊断，不得污染控制误差或安全退出。
FakeRobot.fail_feedback = True
fault_robot, fault_rows, fault_log, fault_nonzero, _ = run_complete_smoke()
assert fault_robot.feedback_calls >= 3, "反馈故障注入未实际执行"
assert not any(row["fb_valid"] == "1" for row in fault_rows), \
    "连续故障后旧实际反馈仍被标为有效"
assert all(not row["fb_time"] and not row["fb_read_ms"]
           and not row["tracking_err_max"] for row in fault_rows), \
    "无效反馈必须只记为空"
print(f"OFFLINE_FEEDBACK_FAILURE_OK: calls={fault_robot.feedback_calls}, "
      f"commands={len(fault_robot.commands)}, nonzero={fault_nonzero}, "
      f"final_static_zero=15, safe_stop={fault_robot.stopped}, log={fault_log}")

# 场景三：--no-gripper 必须把夹爪主动打开（减少遮挡），而不是保持当前角度。
FakeRobot.fail_feedback = False
open_robot, open_rows, open_log, open_nonzero, _ = run_complete_smoke(
    ("--no-gripper",))
open_cfg = main_module.load_grasp_cfg(
    os.path.join(ROOT, "config", "grasp.yaml"))
open_target = float(open_cfg["gripper"]["open_deg"])
assert open_nonzero > 0, "--no-gripper 场景没有产生跟随运动"
assert abs(open_robot.gripper - open_target) < 1e-6, \
    "--no-gripper 没有保持夹爪打开"
print(f"OFFLINE_NO_GRIPPER_OPEN_OK: gripper={open_robot.gripper:.1f}°, "
      f"nonzero={open_nonzero}, log={open_log}")

# 场景四：真正的 eye-in-hand 静止手掌。相机随假机械臂移动，画面里的
# 静止手掌会因相机自运动而移动；验证不会重新进入运动/锁存极限环。
class EyeInHandDetector:
    """眼在手上耦合几何替身：将固定世界系掌心按当前机械臂 X 位移反投影至相机像素。"""

    world_x = 0.10
    world_z = FAKE_DEPTH

    def __init__(self, *args, **kwargs):
        """兼容真实检测器初始化签名。"""
        pass

    def detect(self, image):
        """根据当前假机械臂 ``q[0]`` 实时计算静止世界掌心在相机画面中的投影坐标 ``u``。"""
        robot = FakeRobot.instances[-1] if FakeRobot.instances else None
        camera_x = float(robot.q[0]) if robot is not None else 0.0
        x_camera = self.world_x - camera_x
        u = 320.0 + x_camera * 500.0 / self.world_z
        return u, 240.0, None

    def close(self):
        """兼容真实检测器释放接口。"""
        pass


original_detector = main_module.PalmDetector
original_fk = main_module.compute_tcp_fk_and_jacobian
original_json_load = main_module.json.load


def eye_in_hand_fk(robot, joint_positions):
    """简化运动学替身：令 ``q[0]`` 直接映射到 TCP 世界 X 坐标，雅可比为单位阵。"""
    return (np.array([float(joint_positions[0]), 0.0, 0.30]),
            np.eye(3), np.eye(6))


def identity_hand_eye_load(stream, *args, **kwargs):
    """拦截手眼标定 JSON 加载并将 ``T_tcp_camera`` 设为单位阵，便于精确闭环验证。"""
    data = original_json_load(stream, *args, **kwargs)
    if isinstance(data, dict) and "T_tcp_camera" in data:
        data["T_tcp_camera"] = np.eye(4).tolist()
    return data


FakeRobot.fail_feedback = False
FakeCamera.max_frames = 400
main_module.PalmDetector = EyeInHandDetector
main_module.compute_tcp_fk_and_jacobian = eye_in_hand_fk
main_module.json.load = identity_hand_eye_load
try:
    logs_before = {os.path.join(log_dir, name) for name in os.listdir(log_dir)
                   if name.endswith(".csv")}
    sys.argv = ["palm_follow_grasp.py", "--headless"]
    main_module.main()
    eye_robot = FakeRobot.instances[-1]
    new_logs = [os.path.join(log_dir, name) for name in os.listdir(log_dir)
                if name.endswith(".csv") and os.path.join(log_dir, name) not in logs_before]
    assert len(new_logs) == 1, "eye-in-hand 场景没有生成唯一诊断日志"
    with open(new_logs[0], newline="", encoding="utf-8") as stream:
        eye_rows = list(csv.DictReader(stream))
    assert eye_rows, "eye-in-hand 场景日志为空"
    tail = eye_rows[-80:]
    # 断言末段 80 拍（0.8s）内机械臂无跟跑、无反复解锁、目标世界坐标无漂移
    assert abs(float(eye_robot.q[0])) < 0.02, \
        "静止保持位姿时机械臂仍发生跟随运动"
    assert all(row["ff_moving"] == "0" for row in tail), \
        "相机自运动被误判为手掌运动，静止状态反复解锁"
    assert all(row["hold_latch"] == "1" for row in tail), \
        "静止状态没有稳定保持锁存标记"
    assert all(float(row["joint_speed"]) < 0.05 for row in tail), \
        "静止末端仍持续输出关节速度"
    assert abs(float(tail[-1]["target_x"]) - float(tail[0]["target_x"])) < 0.005, \
        "静止保持位姿时目标仍在漂移"
    print(f"OFFLINE_EYE_IN_HAND_OK: q0={eye_robot.q[0]:.4f}, "
          f"tail_rows={len(tail)}, log={new_logs[0]}")
finally:
    main_module.PalmDetector = original_detector
    main_module.compute_tcp_fk_and_jacobian = original_fk
    main_module.json.load = original_json_load
    FakeCamera.max_frames = 100

print("OFFLINE_SMOKE_OK: v9 默认配置、v8/v7.1 回退、反馈正常/故障、eye-in-hand 静止及安全生命周期全部通过")
