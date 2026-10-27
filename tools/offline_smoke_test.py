#!/usr/bin/env python3
"""完整主程序离线冒烟测试：假相机/检测器/机械臂，不访问任何硬件。"""

import copy
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
    def __init__(self, *args, **kwargs):
        self.fx = self.fy = 500.0
        self.cx, self.cy = 320.0, 240.0
        self.width, self.height = 640, 480
        self.dropped_frames = 0
        self.frames = 0
        self.stopped = False

    def get_frames(self):
        self.frames += 1
        if self.frames > 100:
            raise KeyboardInterrupt  # 走主程序统一退出路径
        time.sleep(1.0 / 60.0)
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 350, dtype=np.uint16)
        return color, depth, 0.001

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

    def __init__(self):
        self.q = np.zeros(6)
        self.commands = []
        self.stopped = False
        self.gripper = 0.0
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

    def joint_vel_cmd(self, dq):
        dq = np.asarray(dq, dtype=float)
        self.commands.append(dq.copy())
        self.q += dq * 0.01

    def hold(self):
        pass

    def stop(self):
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

# 默认交付配置必须在任何硬件初始化前通过校验；危险半径必须被拒绝。
valid_cfg = main_module.load_follow_cfg(
    os.path.join(ROOT, "config", "follow.yaml"))
main_module.validate_follow_cfg(valid_cfg)
bad_cfg = copy.deepcopy(valid_cfg)
bad_cfg["workspace"]["max_radius"] = 0.43
try:
    main_module.validate_follow_cfg(bad_cfg)
except ValueError:
    pass
else:
    raise AssertionError("超过70%臂展的工作空间未被配置校验拒绝")

# 用单位雅可比验证完整数据流，不加载 Pinocchio/URDF。
main_module.compute_tcp_fk_and_jacobian = lambda robot, q: (
    np.array([0.10, 0.0, 0.30]), np.eye(3), np.eye(6))

sys.argv = ["palm_follow_grasp.py", "--headless"]
main_module.main()

robot = FakeRobot.instances[-1]
nonzero = [q for q in robot.commands if np.linalg.norm(q) > 1e-6]
assert nonzero, "跟随已启动但没有产生任何非零关节指令"
final_tail = robot.commands[-15:]
assert len(final_tail) == 15, "控制周期数量不足，无法验证停手保持"
assert all(np.linalg.norm(q) < 1e-6 for q in final_tail), \
    "假目标回中心停手后，末段仍存在非零关节指令"
assert robot.stopped, "主程序退出时没有调用统一安全停止"
print(f"OFFLINE_SMOKE_OK: frames=100, commands={len(robot.commands)}, "
      f"nonzero={len(nonzero)}, final_static_zero={len(final_tail)}, "
      f"safe_stop={robot.stopped}")
