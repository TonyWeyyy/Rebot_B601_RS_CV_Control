#!/usr/bin/env python3
"""完整主程序离线冒烟测试：假相机/检测器/机械臂，不访问任何硬件。"""

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
    max_frames = 100

    def __init__(self, *args, **kwargs):
        self.fx = self.fy = 500.0
        self.cx, self.cy = 320.0, 240.0
        self.width, self.height = 640, 480
        self.dropped_frames = 0
        self.frames = 0
        self.stopped = False
        self.previous_raw_s = None

    def get_frames(self):
        self.frames += 1
        if self.frames > self.max_frames:
            raise KeyboardInterrupt  # 走主程序统一退出路径
        time.sleep(1.0 / 60.0)
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 350, dtype=np.uint16)
        return color, depth, 0.001

    def get_last_frame_timing(self):
        # 与主机同钟、固定20ms传输延迟，模拟 latest-only 无队列积压。
        raw_s = time.monotonic() - 0.020
        if self.frames == 40:
            raw_s = self.previous_raw_s  # 重复时间戳：必须回退主机时刻
        elif self.frames == 55:
            raw_s += 10.0                # 大跳变：必须回退主机时刻
        self.previous_raw_s = raw_s
        return raw_s, "fake_seconds"

    def stop(self):
        self.stopped = True


class FakeDetector:
    def __init__(self, *args, **kwargs):
        self.i = 0

    def detect(self, image):
        self.i += 1
        # 静止→同向移动→回中心停手，覆盖起步、巡航和停振全过程。
        u = (320 if self.i < 15 or self.i > 50
             else min(390, 320 + (self.i - 15) * 5))
        return u, 240, None

    def close(self):
        pass


class FakeTracker:
    def __init__(self):
        self.track = None

    def update(self, detection):
        if detection is None:
            return None
        u, v = detection
        self.track = {"u": float(u), "v": float(v), "lost": 0,
                      "x3d": 0.0, "y3d": 0.0, "z3d": 0.0}
        return self.track


class FakeOneEuro:
    def __init__(self, *args, **kwargs):
        self.x = None

    def __call__(self, x, t=None):
        self.x = float(x) if self.x is None else self.x + 0.5 * (x - self.x)
        return self.x


class FakeBridge:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, image, det):
        return (*det, "det") if det is not None else (None, None, None)


class FakeRobot:
    instances = []
    fail_feedback = False

    def __init__(self):
        self.q = np.zeros(6)
        self.commands = []
        self.stopped = False
        self.gripper = 0.0
        self.feedback_calls = 0
        self.events = []
        FakeRobot.instances.append(self)

    def move_to_joints(self, *args, **kwargs):
        pass

    def forward_kinematics(self):
        return {"position": np.array([0.10, 0.0, 0.30]),
                "rotation": np.eye(3)}

    def get_gripper_deg(self):
        return self.gripper

    def set_gripper_deg(self, deg):
        self.gripper = float(deg)

    def get_command_pos(self):
        return self.q.copy()

    def get_current_pos(self):
        self.feedback_calls += 1
        if FakeRobot.fail_feedback:
            raise RuntimeError("注入实际反馈 CAN 故障")
        return self.q.copy() - 0.005  # 固定小跟踪误差，仅用于日志验证

    def joint_vel_cmd(self, dq):
        dq = np.asarray(dq, dtype=float)
        self.commands.append(dq.copy())
        self.q += dq * 0.01

    def hold(self):
        pass

    def stop(self):
        self.events.append("safe_stop")
        self.stopped = True


class FakeListener:
    def __init__(self, on_press):
        self.on_press = on_press
        self.timer = None

    def start(self):
        def press_g():
            self.on_press(types.SimpleNamespace(char="g"))
        self.timer = threading.Timer(0.10, press_g)
        self.timer.daemon = True
        self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.timer.cancel()


# 在导入主程序前替换所有硬件相关模块。
camera_module = types.ModuleType("gemini2_camera")
camera_module.Gemini2Camera = FakeCamera
sys.modules["gemini2_camera"] = camera_module

palm_module = types.ModuleType("palm_detection")
palm_module.PalmDetector = FakeDetector
palm_module.PalmTracker = FakeTracker
palm_module.OneEuroFilter = FakeOneEuro
palm_module.LKPalmBridge = FakeBridge
palm_module.get_median_depth = lambda depth, scale, u, v: 0.35
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
    def close(self):
        self.robot.events.append("feedback_close")
        super().close()


main_module.ActualFeedbackSampler = TrackedFeedbackSampler

# 默认交付配置必须在任何硬件初始化前通过校验；危险半径必须被拒绝。
valid_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow.yaml"))
main_module.validate_follow_cfg(valid_cfg)
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
                   ("static_velocity_hold_delay_s", 0.11)):
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
    cfg = original_load_follow_cfg(path)
    cfg["follow"]["camera_hw_timestamp_enable"] = True
    cfg["follow"]["diag_actual_feedback_hz"] = 2.0
    return cfg
main_module.load_follow_cfg = load_with_fake_hw_time
log_dir = os.path.join(ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)


def run_complete_smoke(extra_args=()):
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

# --no-gripper 必须把夹爪主动打开（减少遮挡），而不是保持当前角度。
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

# 场景三：真正的 eye-in-hand 静止手掌。相机随假机械臂移动，画面里的
# 静止手掌会因相机自运动而移动；验证不会重新进入运动/锁存极限环。
class EyeInHandDetector:
    world_x = 0.10
    world_z = 0.35

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        robot = FakeRobot.instances[-1] if FakeRobot.instances else None
        camera_x = float(robot.q[0]) if robot is not None else 0.0
        x_camera = self.world_x - camera_x
        u = 320.0 + x_camera * 500.0 / self.world_z
        return u, 240.0, None

    def close(self):
        pass


original_detector = main_module.PalmDetector
original_fk = main_module.compute_tcp_fk_and_jacobian
original_json_load = main_module.json.load


def eye_in_hand_fk(robot, joint_positions):
    return (np.array([float(joint_positions[0]), 0.0, 0.30]),
            np.eye(3), np.eye(6))


def identity_hand_eye_load(stream, *args, **kwargs):
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
