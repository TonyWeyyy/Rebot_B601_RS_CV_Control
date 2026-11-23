#!/usr/bin/env python3
"""
v14.0 手掌跟随 + 捏合抓取主程序（四层解耦架构 + 统一世界系 PBVS + 数字孪生）
===========================================================================
相比 v13.0 单文件 3,200+ 行与全局变量架构，本入口仅作为薄运行壳层（~360 行），
将全部控制逻辑清晰委托给 `servo_v14/` 四层模块：

    1. 视觉/主线程（30/60fps）：Gemini2Camera + PalmDetector(ROI/LK) + DepthProtectionChain(4 层深度保护)
       + PinchGripperMapper(2D 捏合映射与运动冻结) + OpenCV HUD
    2. 100Hz 伺服线程：`ServoV14Pipeline`
       Layer 1 (WorldTargetEstimator) → Layer 2 (ServoFSM) → Layer 3 (WorldPBVSController)
       → Layer 4 (SafetyShield) → RobotBackend (HardwareRobotBackend | DigitalTwinBackend)
    3. 键盘监听线程（`pynput`）：`g`/空格 启停跟随（同步清除 M6 安全锁存）、`q`/Esc 统一安全退出

运行模式：
    python3 palm_follow_v14.py                       # 真机完整模式（跟随 + 夹爪）
    python3 palm_follow_v14.py --sim                 # 离线数字孪生模式（真相机 + 仿真机械臂）
    python3 palm_follow_v14.py --no-gripper          # 仅跟随（启动主动打开夹爪防遮挡）
    python3 palm_follow_v14.py --no-follow           # 仅捏合夹爪
    python3 palm_follow_v14.py --no-arm              # 纯视觉演练（不连硬件）
    python3 palm_follow_v14.py --headless            # 无窗口模式（SSH 场景）

硬件安全红线（与 AGENTS.md 100% 一致）：
    - 所有退出路径均走统一安全停机链：停 100Hz 伺服线程 → 停键盘 → 关相机 → 刷盘日志
      → `backend.hold()` → `backend.stop()`（缓慢回零到位后再失能，绝不半空失能）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gemini2_camera import Gemini2Camera
from palm_detection import (
    LKPalmBridge,
    OneEuroFilter,
    PalmDetector,
    PalmTracker,
    draw_landmarks,
    draw_palm,
    pinch_distance_px,
    pixel_to_3d,
)
from servo_diagnostics import DiagnosticRecorder
from servo_timing import CameraTimestampCalibrator
from servo_v14.config import (
    V14Config,
    load_v14_config,
    validate_v14_config,
)
from servo_v14.depth_guard import DepthProtectionChain
from servo_v14.pipeline import ServoV14Pipeline
from servo_v14.robot_backend import DigitalTwinBackend, HardwareRobotBackend, RobotBackend
from servo_v14.types import FSMMode, Observation

DEFAULT_HOME_JOINTS = np.array([0.0, 1.2, 1.05, 0.0, 0.0, 0.0], dtype=np.float64)


class PinchGripperMapper:
    """2D 捏合像素距离 → J7 夹爪目标角（deg）映射器，支持手臂运动自动冻结。"""

    def __init__(self, grasp_cfg: dict) -> None:
        pinch = grasp_cfg.get("pinch") or {}
        grip = grasp_cfg.get("gripper") or {}
        self.enabled: bool = bool(grip.get("enabled", True))
        self.hold_while_moving: bool = bool(grip.get("hold_while_moving", True))
        self.move_speed_threshold: float = float(grip.get("move_speed_threshold", 0.05))
        self.closed_px: float = float(pinch.get("closed_px", 20.0))
        self.open_px: float = float(pinch.get("open_px", 200.0))
        self.closed_deg: float = float(grip.get("closed_deg", 5.2))
        self.open_deg: float = float(grip.get("open_deg", 285.2))
        self.deadband_deg: float = float(pinch.get("deadband_deg", 2.0))
        self.filter_type: str = str(pinch.get("filter_type", "one_euro"))
        self.smoothing: float = float(pinch.get("smoothing", 0.35))
        self._oe = OneEuroFilter(
            min_cutoff=float(pinch.get("min_cutoff_hz", 2.0)),
            beta=float(pinch.get("beta", 0.01)),
            d_cutoff=float(pinch.get("dcutoff_hz", 1.0)),
        )
        self._last_deg: Optional[float] = None
        self.frozen_by_motion: bool = False

    def seed(self, current_deg: float) -> None:
        """以当前电机夹爪角度作为平滑起点，避免启动跳变。"""
        self._last_deg = float(current_deg)
        self._oe.reset()

    def update(
        self,
        pinch_px: Optional[float],
        timestamp_s: float,
        arm_joint_speed: float,
    ) -> Optional[float]:
        """根据新鲜 2D 捏合像素距离计算新的 J7 目标角度 (deg)，冻结或死区内返回 None。"""
        if not self.enabled or pinch_px is None:
            return None
        if self.hold_while_moving and arm_joint_speed > self.move_speed_threshold:
            self.frozen_by_motion = True
            return None
        self.frozen_by_motion = False

        span_px = max(1.0, self.open_px - self.closed_px)
        ratio = float(np.clip((pinch_px - self.closed_px) / span_px, 0.0, 1.0))
        raw_deg = self.closed_deg + ratio * (self.open_deg - self.closed_deg)

        if self.filter_type == "one_euro":
            filt_deg = float(self._oe(raw_deg, timestamp_s))
        else:
            prev = raw_deg if self._last_deg is None else self._last_deg
            filt_deg = prev + self.smoothing * (raw_deg - prev)

        if self._last_deg is not None and abs(filt_deg - self._last_deg) < self.deadband_deg:
            return None
        self._last_deg = filt_deg
        return filt_deg


def load_home_joints() -> np.ndarray:
    """优先加载 `config/home_pose.json` 中的示教 HOME 位姿，否则回退默认安全 HOME。"""
    home_file = os.path.join(ROOT, "config", "home_pose.json")
    if os.path.exists(home_file):
        with open(home_file, encoding="utf-8") as stream:
            data = json.load(stream)
        q = np.asarray(data["home_joint_pos"], dtype=np.float64)
        print(f"[HOME] 使用示教 HOME: {np.round(q, 3).tolist()}")
        return q
    return DEFAULT_HOME_JOINTS.copy()


def load_hand_eye_matrix() -> np.ndarray:
    """加载 `config/hand_eye_calibration.json` 4×4 手眼外参矩阵 `T_tcp_camera`。"""
    path = os.path.join(ROOT, "config", "hand_eye_calibration.json")
    with open(path, encoding="utf-8") as stream:
        raw = json.load(stream)
    T = np.asarray(raw["T_tcp_camera"], dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("hand_eye_calibration.json 中 T_tcp_camera 矩阵格式无效")
    return T


def load_grasp_cfg(path: str) -> dict:
    """读取夹爪捏合映射配置 `config/grasp.yaml`。"""
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数（优先级：CLI > YAML > dataclass 默认值）。"""
    p = argparse.ArgumentParser(description="v14.0 四层解耦视觉伺服 + 捏合抓取主程序")
    p.add_argument("--v14-cfg", default=os.path.join(ROOT, "config", "v14.yaml"),
                   help="v14 配置文件路径（也兼容旧版 follow.yaml）")
    p.add_argument("--grasp-cfg", default=os.path.join(ROOT, "config", "grasp.yaml"),
                   help="夹爪捏合配置文件路径")
    p.add_argument("--no-arm", action="store_true", help="纯视觉演练（不连接机械臂）")
    p.add_argument("--sim", action="store_true", help="启用离线数字孪生机械臂后端（不碰真机 CAN）")
    p.add_argument("--no-follow", action="store_true", help="仅启用夹爪捏合，禁用手臂跟随")
    p.add_argument("--no-gripper", action="store_true", help="仅启用跟随（启动主动打开夹爪防遮挡）")
    p.add_argument("--headless", action="store_true", help="无 OpenCV 窗口模式")
    p.add_argument("--distance", type=float, default=None, help="覆盖期望跟随距离 (m)")
    p.add_argument("--gain", type=float, default=None, help="覆盖竖直/深度增益")
    p.add_argument("--gain-x", type=float, default=None, help="覆盖水平偏航增益")
    p.add_argument("--max-vel", type=float, default=None, help="覆盖最大末端线速度 (m/s, ≤0.85)")
    p.add_argument("--fps", type=int, default=None, help="覆盖相机帧率 (15/30/60)")
    return p.parse_args(argv)


def apply_cli_overrides(cfg: V14Config, args: argparse.Namespace) -> V14Config:
    """以不可变方式应用 CLI 覆盖参数并重新执行安全红线校验。"""
    pbvs = cfg.pbvs
    safety = cfg.safety
    cam = cfg.camera
    if args.distance is not None:
        pbvs = replace(pbvs, distance=args.distance)
    if args.gain is not None:
        pbvs = replace(pbvs, gain=args.gain)
    if args.gain_x is not None:
        pbvs = replace(pbvs, gain_x=args.gain_x)
    if args.max_vel is not None:
        safety = replace(safety, max_vel=args.max_vel)
    if args.fps is not None:
        cam = replace(cam, fps=args.fps)
    return validate_v14_config(replace(cfg, pbvs=pbvs, safety=safety, camera=cam))


def main(argv=None) -> int:
    """v14.0 主运行入口：装配感知、四层控制流水线、后端与统一安全停机链。"""
    args = parse_args(argv)
    cfg = apply_cli_overrides(load_v14_config(args.v14_cfg), args)
    grasp_cfg = load_grasp_cfg(args.grasp_cfg)
    gripper_mapper = PinchGripperMapper(grasp_cfg)
    enable_follow = (not args.no_arm) and (not args.no_follow)
    enable_gripper = (not args.no_arm) and (not args.no_gripper) and gripper_mapper.enabled

    print("=" * 64)
    print("reBot Arm B601-RS 手掌跟随 + 捏合抓取 [v14.0 四层解耦架构]")
    print("=" * 64)
    mode_tag = "SIM数字孪生" if args.sim else ("纯视觉" if args.no_arm else "真机MIT")
    print(f"后端: {mode_tag} | follow={'ON' if enable_follow else 'OFF'} "
          f"| gripper={'ON' if enable_gripper else 'OFF'} | fps={cfg.camera.fps}")
    print(f"参数: distance={cfg.pbvs.distance:.2f}m gain={cfg.pbvs.gain:.2f}/{cfg.pbvs.gain_x:.2f} "
          f"max_vel={cfg.safety.max_vel:.2f}m/s max_r={cfg.safety.workspace_max_radius:.2f}m")
    print("操作: g/空格 启停跟随 | q/Esc/Ctrl+C 统一安全退出并回零")
    print("=" * 64)

    backend: Optional[RobotBackend] = None
    pipeline: Optional[ServoV14Pipeline] = None
    camera: Optional[Gemini2Camera] = None
    detector: Optional[PalmDetector] = None
    diag: Optional[DiagnosticRecorder] = None
    kb_listener = None

    stop_event = threading.Event()
    state_lock = threading.Lock()
    shared = {
        "following": False,
        "obs": Observation(
            seq=0, meas_time_s=0.0, host_time_s=0.0,
            visible=False, frozen=False, p_cam=None,
        ),
        "joint_speed": 0.0,
        "mode": FSMMode.IDLE,
        "err_norm": 0.0,
        "last_toggle_s": 0.0,
    }
    home_joints = load_home_joints()
    T_tcp_cam = load_hand_eye_matrix() if enable_follow else np.eye(4, dtype=np.float64)

    def toggle_following() -> None:
        now = time.monotonic()
        with state_lock:
            if now - shared["last_toggle_s"] < 0.25:
                return
            shared["last_toggle_s"] = now
            new_state = not shared["following"]
            shared["following"] = new_state
        if new_state:
            if backend is not None:
                cleared = backend.clear_safety_fault()
                if cleared:
                    print("[v14] 已清除 M6 安全监督软停机锁存")
            if pipeline is not None:
                pipeline.reset(keep_pose_history=True)
                pipeline.lock_current_orientation()
            print("[v14] 跟随启动")
        else:
            if backend is not None:
                backend.hold()
            print("[v14] 跟随暂停（保持当前位姿）")

    # ── 1. 初始化机械臂后端（真机或数字孪生） ────────────────────
    try:
        if not args.no_arm:
            if args.sim:
                offset_path = os.path.join(ROOT, "config", "joint_offset.json")
                backend = DigitalTwinBackend(
                    sim_cfg=cfg.sim,
                    q_init=home_joints,
                    use_pinocchio=True,
                    joint_offset_path=offset_path,
                )
                print(f"[SIM] 数字孪生就绪（Pinocchio={'ON' if backend.using_pinocchio else '解析回退'}）")
            else:
                backend = HardwareRobotBackend()
                print("[HW] 缓慢移动至示教 HOME 位姿...")
                backend.move_to_joints(home_joints, speed_deg_s=20.0, wait=True)
                # 同步 L2 分关节速度上限 [2.5, 2.0, 2.0, 3.0, 3.0, 2.4] 至底层 200Hz MIT 斜坡
                backend.set_joint_speed_limits(np.asarray(cfg.safety.max_vel_joints, dtype=np.float64))

            # --no-gripper 硬件安全规则：主动打开夹爪防止遮挡手腕相机
            if args.no_gripper:
                backend.set_gripper_deg(gripper_mapper.open_deg)
                print(f"[夹爪] --no-gripper 模式：已主动打开至 {gripper_mapper.open_deg:.1f}°")
            else:
                gripper_mapper.seed(backend.get_gripper_deg())

            if enable_follow:
                pipeline = ServoV14Pipeline(cfg=cfg, backend=backend, T_tcp_camera=T_tcp_cam)
                pipeline.lock_current_orientation(home_joints)

        # ── 2. 初始化相机与视觉组件 ──────────────────────────────
        camera = Gemini2Camera(
            fps=cfg.camera.fps,
            color_exposure_ms=cfg.camera.color_exposure_ms,
        )
        detector = PalmDetector(
            roi_enable=cfg.camera.roi_crop_enable,
            roi_scale=cfg.camera.roi_crop_scale,
        )
        tracker = PalmTracker()
        lk_bridge = LKPalmBridge(
            enable=cfg.camera.hybrid_track_enable,
            max_frames=cfg.camera.hybrid_max_frames,
            box_half=cfg.camera.hybrid_box_half,
            max_step_px=cfg.camera.hybrid_max_step_px,
        )
        depth_guard = DepthProtectionChain(cfg.depth, target_distance_m=cfg.pbvs.distance)
        ts_cal = CameraTimestampCalibrator(
            max_transport_s=cfg.camera.hw_max_transport_s
        )

        # ── 3. 初始化诊断日志 ────────────────────────────────────
        if cfg.camera.diag_enable and enable_follow:
            log_dir = os.path.join(ROOT, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(
                log_dir, f"servo_v14_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            diag = DiagnosticRecorder(log_path, queue_size=cfg.camera.diag_queue_size)
            diag.start()
            print(f"[诊断] v14 高频日志: {log_path}")

        # ── 4. 启动 100Hz 伺服线程 ───────────────────────────────
        servo_thread: Optional[threading.Thread] = None
        if enable_follow and pipeline is not None and backend is not None:
            def servo_loop() -> None:
                dt_nom = cfg.ctrl_dt_s
                next_tick = time.perf_counter()
                last_t = time.monotonic()
                while not stop_event.is_set():
                    now = time.monotonic()
                    dt = max(1e-4, min(0.05, now - last_t))
                    last_t = now
                    with state_lock:
                        obs = shared["obs"]
                        following = shared["following"]

                    robot_state = backend.get_state(now)
                    shielded, telem = pipeline.step(
                        obs=obs,
                        robot_state=robot_state,
                        dt_s=dt,
                        following_enabled=following,
                        apply_to_backend=True,
                    )

                    if shielded.trigger_return_home:
                        print("[安全护盾] 越出硬工作空间 >5cm，缓慢自动回 HOME！")
                        with state_lock:
                            shared["following"] = False
                        backend.hold()
                        backend.move_to_joints(home_joints, speed_deg_s=15.0, wait=True)
                        pipeline.reset(keep_pose_history=False)
                        pipeline.lock_current_orientation(home_joints)

                    with state_lock:
                        shared["joint_speed"] = telem.joint_speed_max
                        shared["mode"] = telem.mode
                        shared["err_norm"] = telem.error_norm_m

                    if diag is not None:
                        diag.log(pipeline.telemetry_to_diag_row(telem, robot_state, following))

                    next_tick += dt_nom
                    sleep_s = next_tick - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        next_tick = time.perf_counter()

            servo_thread = threading.Thread(target=servo_loop, name="v14-servo-100hz", daemon=True)
            servo_thread.start()

        # ── 5. 启动全局键盘监听 ──────────────────────────────────
        try:
            from pynput import keyboard as pynput_kb

            def on_key_press(key):
                if key == pynput_kb.Key.esc:
                    stop_event.set()
                    return False
                if key == pynput_kb.Key.space:
                    if enable_follow:
                        toggle_following()
                    return None
                ch = getattr(key, "char", None)
                if ch in ("g", "G") and enable_follow:
                    toggle_following()
                elif ch in ("q", "Q"):
                    stop_event.set()
                    return False
                return None

            kb_listener = pynput_kb.Listener(on_press=on_key_press)
            kb_listener.start()
        except Exception:
            kb_listener = None

        # ── 6. 视觉主循环（60fps） ───────────────────────────────
        seq = 0
        lost_streak = 0
        while not stop_event.is_set():
            t0 = time.perf_counter()
            color_img, depth_img, depth_scale = camera.get_frames(latest_only=True)
            host_t = time.monotonic()
            if color_img is None:
                time.sleep(0.002)
                continue

            hw_ts_raw, _ = camera.get_last_frame_timing()
            hw_ts = hw_ts_raw if cfg.camera.hw_timestamp_enable else None
            meas_t, hw_used = ts_cal.update(hw_ts, host_t)

            u_det, v_det, landmarks = detector.detect(color_img)
            proc_s = time.perf_counter() - t0

            det_uv = (u_det, v_det) if u_det is not None else None
            out_u, out_v, src = lk_bridge.update(color_img, det_uv)
            track = tracker.update((out_u, out_v) if out_u is not None else None)

            # 夹爪通道：仅要求 2D 新鲜关键点（不依赖深度，LK 桥接帧无 landmarks 自动跳过）
            if enable_gripper and backend is not None and landmarks is not None and src == "det":
                p_px, _, _ = pinch_distance_px(landmarks, color_img.shape)
                with state_lock:
                    cur_js = shared["joint_speed"]
                g_deg = gripper_mapper.update(p_px, host_t, cur_js)
                if g_deg is not None:
                    backend.set_gripper_deg(g_deg)

            # 4 层深度保护链 + 3D 反投影
            if track is not None and track["lost"] == 0:
                lost_streak = 0
                u, v = int(track["u"]), int(track["v"])
                eff_scale = depth_scale if depth_scale is not None else (camera.depth_scale or 0.001)
                raw_z = depth_guard.extract_raw_depth(depth_img, u, v, eff_scale)
                d_res = depth_guard.update(raw_z, host_t)
                if d_res.valid:
                    p_cam = np.asarray(
                        pixel_to_3d(u, v, d_res.z_m, camera.fx, camera.fy, camera.cx, camera.cy),
                        dtype=np.float64,
                    )
                    track["x3d"], track["y3d"], track["z3d"] = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
                    seq += 1
                    obs_new = Observation(
                        seq=seq,
                        meas_time_s=meas_t,
                        host_time_s=host_t,
                        visible=True,
                        frozen=d_res.frozen,
                        p_cam=p_cam,
                        processing_s=proc_s,
                        hw_time_used=hw_used,
                    )
                else:
                    obs_new = Observation(
                        seq=seq, meas_time_s=meas_t, host_time_s=host_t,
                        visible=False, frozen=False, p_cam=None,
                        processing_s=proc_s, hw_time_used=hw_used,
                    )
            else:
                lost_streak += 1
                if lost_streak >= 30:
                    depth_guard.reset()
                obs_new = Observation(
                    seq=seq, meas_time_s=meas_t, host_time_s=host_t,
                    visible=False, frozen=False, p_cam=None,
                    processing_s=proc_s, hw_time_used=hw_used,
                )

            with state_lock:
                shared["obs"] = obs_new
                cur_mode = shared["mode"]
                cur_err = shared["err_norm"]

            # 可视化 HUD
            if not args.headless:
                vis = color_img.copy()
                if landmarks is not None:
                    draw_landmarks(vis, landmarks)
                if track is not None and track["lost"] == 0:
                    draw_palm(vis, track, int(camera.cx), int(camera.cy))
                hud = f"v14 [{cur_mode.value}] err={cur_err*100:.1f}cm"
                cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("reBot v14 Palm Follow", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_event.set()
                elif key in (ord("g"), ord(" ")) and enable_follow:
                    toggle_following()

    except KeyboardInterrupt:
        print("\n[v14] 收到 Ctrl+C，进入统一安全退出链...")
    finally:
        # ── 统一安全退出链（严禁半空失能） ───────────────────────
        stop_event.set()
        if "servo_thread" in locals() and servo_thread is not None and servo_thread.is_alive():
            servo_thread.join(timeout=2.0)
        if kb_listener is not None:
            try:
                kb_listener.stop()
            except Exception:
                pass
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if diag is not None:
            diag.close()
        if backend is not None:
            print("[v14] 机械臂保持并安全缓慢回零...")
            backend.stop()
        print("[v14] 安全退出完成。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
