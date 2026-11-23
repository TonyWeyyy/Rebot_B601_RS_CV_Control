#!/usr/bin/env python3
"""
v15.0 MuJoCo Studio ↔ 真实机械臂双向数字孪生遥操作桥接主程序
================================================================
实现「在 MuJoCo 3D 窗口中拖动关节滑块或 3D 目标球（实时 IK）→ 外部真实 B601-RS
机械臂低速安全同步运动，同时真机实测关节角 / 力矩 / 温度实时回传 MuJoCo」的双向闭环：

  - 通道 A（udp://127.0.0.1:19870，本程序 → MuJoCo Studio，100Hz）：
      实时上传真机 200Hz `q_act / q_cmd / dq_act / tau_act / J7 / MOS / M6` 与武装回执；
  - 通道 B（udp://127.0.0.1:19871，MuJoCo Studio → 本程序，50-100Hz）：
      接收 MuJoCo 滑块或 3D 目标球 IK 解算目标 `q_des`，经 5 道安全防线过滤后平滑执行。

运行方式（先启动 `./tools/launch_mujoco_twin.sh --native` 打开本地 3D 窗口）：
    python3 mujoco_teleop_v15.py --sim                 # 离线数字孪生演练（不连真机 CAN）
    python3 mujoco_teleop_v15.py --max-speed 20        # 真机低速安全双向联动（推荐首次验收）
    python3 mujoco_teleop_v15.py --no-gripper          # 禁用夹爪（启动主动打开至 285.2°）

快捷按键：
    Space / h        真机侧紧急解除武装并原地保持（覆盖 MuJoCo 端请求）
    g                清除 200Hz M6 安全监督软停机锁存
    q / Esc / Ctrl+C 统一安全退出链：保持 → 缓慢回零到位 → 失能（严禁半空失能）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servo_v14.config import load_v14_config
from servo_v14.mujoco_bridge import (
    DEFAULT_MUJOCO_PORT,
    DEFAULT_TELEOP_CMD_PORT,
    MuJoCoTeleopCmdReceiver,
    MuJoCoTwinPublisher,
)
from servo_v14.robot_backend import DigitalTwinBackend, HardwareRobotBackend, RobotBackend
from servo_v14.teleop_bridge import (
    HARD_TELEOP_JOINT_SPEED,
    SafeTeleopConfig,
    SafeTeleopController,
    parse_teleop_packet,
)

DEFAULT_HOME_JOINTS = np.array([-0.4313, 0.3445, 0.3691, 0.0938, 0.0018, -0.0015], dtype=np.float64)


def load_home_joints() -> np.ndarray:
    """加载 `config/home_pose.json` 的示教 HOME 关节角，缺失时回退安全默认值。"""
    home_file = os.path.join(ROOT, "config", "home_pose.json")
    if os.path.exists(home_file):
        with open(home_file, encoding="utf-8") as stream:
            data = json.load(stream)
        q = np.asarray(data["home_joint_pos"], dtype=np.float64)
        print(f"[HOME] 使用示教 HOME: {np.round(q, 3).tolist()}")
        return q
    return DEFAULT_HOME_JOINTS.copy()


def load_gripper_bounds() -> tuple[float, float]:
    """从 `config/grasp.yaml` 读取 J7 夹爪标定闭合/全开角度 (deg)。"""
    grasp_file = os.path.join(ROOT, "config", "grasp.yaml")
    if os.path.exists(grasp_file):
        with open(grasp_file, encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        g = raw.get("gripper") or {}
        return float(g.get("closed_deg", 5.2)), float(g.get("open_deg", 285.2))
    return 5.2, 285.2


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v15.0 MuJoCo Studio ↔ reBot Arm B601-RS 双向数字孪生遥操作桥接"
    )
    p.add_argument("--sim", action="store_true", help="使用离线 200Hz 数字孪生后端演练（不碰真机 CAN）")
    p.add_argument("--no-gripper", action="store_true", help="禁用夹爪遥操作（启动主动打开至 285.2°）")
    p.add_argument(
        "--max-speed",
        type=float,
        default=35.0,
        help="真机侧允许的绝对单轴限速上限 (°/s, 范围 5~60，默认 35)",
    )
    p.add_argument(
        "--align-tol",
        type=float,
        default=8.0,
        help="解锁武装前允许的虚实最大单轴误差 (°，默认 8.0)",
    )
    p.add_argument(
        "--state-port",
        type=int,
        default=DEFAULT_MUJOCO_PORT,
        help=f"状态上传至 MuJoCo 的 UDP 端口（默认 {DEFAULT_MUJOCO_PORT}）",
    )
    p.add_argument(
        "--cmd-port",
        type=int,
        default=DEFAULT_TELEOP_CMD_PORT,
        help=f"接收 MuJoCo 遥操作指令的 UDP 端口（默认 {DEFAULT_TELEOP_CMD_PORT}）",
    )
    p.add_argument(
        "--watchdog-ms",
        type=float,
        default=250.0,
        help="MuJoCo 心跳失联自动保持超时 (ms，默认 250)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    closed_deg, open_deg = load_gripper_bounds()
    hard_speed_cap = float(np.clip(args.max_speed, 5.0, 60.0))

    teleop_cfg = SafeTeleopConfig(
        align_tol_deg=float(args.align_tol),
        hard_max_speed_deg_s=hard_speed_cap,
        gripper_min_deg=closed_deg,
        gripper_max_deg=open_deg,
    ).validate()

    print("=" * 68)
    print("reBot Arm B601-RS 双向数字孪生遥操作桥接 [v15.0 MuJoCo <-> 真机]")
    print("=" * 68)
    print(
        f"后端: {'SIM数字孪生' if args.sim else '真机MIT'} | "
        f"硬限速上限: {hard_speed_cap:.1f}°/s | 对齐门限: {teleop_cfg.align_tol_deg:.1f}° | "
        f"夹爪: {'OFF(全开)' if args.no_gripper else 'ON'}"
    )
    print(
        f"UDP 双向通道: 状态 -> :{args.state_port} | 指令 <- :{args.cmd_port} "
        f"(看门狗 {args.watchdog_ms:.0f}ms)"
    )
    print("按键: Space/h 急停解除武装 | g 清除 M6 锁存 | q/Esc/Ctrl+C 安全回零退出")
    print("=" * 68)

    backend: Optional[RobotBackend] = None
    state_pub: Optional[MuJoCoTwinPublisher] = None
    cmd_rx: Optional[MuJoCoTeleopCmdReceiver] = None
    kb_listener = None

    stop_event = threading.Event()
    hw_estop_lock = threading.Lock()
    hw_state = {
        "local_disarm": False,   # 终端按 Space/h 强制本地解除武装，需在 MuJoCo 先 Disarm 再重新 Arm 才解除
        "clear_fault_req": False,
    }
    home_joints = load_home_joints()

    try:
        # ── 1. 初始化机械臂后端（SIM 或真机） ─────────────────────
        if args.sim:
            v14_cfg = load_v14_config(os.path.join(ROOT, "config", "v14.yaml"))
            backend = DigitalTwinBackend(
                sim_cfg=v14_cfg.sim,
                q_init=home_joints,
                use_pinocchio=True,
                joint_offset_path=os.path.join(ROOT, "config", "joint_offset.json"),
            )
            print(f"[v15 SIM] 数字孪生后端就绪（Pinocchio={'ON' if backend.using_pinocchio else '解析回退'}）")
        else:
            backend = HardwareRobotBackend()
            print("[v15 HW] 缓慢移动至示教 HOME 位姿...")
            backend.move_to_joints(home_joints, speed_deg_s=20.0, wait=True)
            backend.set_joint_speed_limits(np.asarray(HARD_TELEOP_JOINT_SPEED, dtype=np.float64))

        if args.no_gripper:
            backend.set_gripper_deg(open_deg)
            print(f"[夹爪] --no-gripper 模式：已主动打开至 {open_deg:.1f}°")

        # ── 2. 初始化双向 UDP 桥接与 100Hz 安全遥操作控制器 ────────
        state_pub = MuJoCoTwinPublisher(port=args.state_port, max_hz=100.0, enabled=True)
        cmd_rx = MuJoCoTeleopCmdReceiver(
            port=args.cmd_port,
            watchdog_s=max(0.05, float(args.watchdog_ms) * 1e-3),
        )
        if cmd_rx.bind_error:
            raise RuntimeError(f"无法绑定 MuJoCo 指令端口 :{args.cmd_port} ({cmd_rx.bind_error})")

        controller = SafeTeleopController(
            cfg=teleop_cfg,
            backend=backend,
            enable_gripper=not args.no_gripper,
        )

        # ── 3. 启动全局安全热键监听（pynput） ─────────────────────
        try:
            from pynput import keyboard as pynput_kb

            def on_key_press(key):
                if key == pynput_kb.Key.esc:
                    stop_event.set()
                    return False
                if key == pynput_kb.Key.space:
                    with hw_estop_lock:
                        hw_state["local_disarm"] = True
                    print("\n[v15 安全] 已按 Space 本地急停解除武装！请在 MuJoCo 先 Disarm 后再重新解锁。")
                    return None
                ch = getattr(key, "char", None)
                if ch in ("q", "Q"):
                    stop_event.set()
                    return False
                if ch in ("h", "H"):
                    with hw_estop_lock:
                        hw_state["local_disarm"] = True
                    print("\n[v15 安全] 已按 h 本地急停解除武装！")
                elif ch in ("g", "G"):
                    with hw_estop_lock:
                        hw_state["clear_fault_req"] = True
                return None

            kb_listener = pynput_kb.Listener(on_press=on_key_press)
            kb_listener.start()
        except Exception:
            kb_listener = None

        # ── 4. 100Hz 双向同步主循环 ───────────────────────────────
        dt_nom = 0.01
        next_tick = time.perf_counter()
        last_t = time.monotonic()
        last_print_t = 0.0
        source_tag = "sim_teleop" if args.sim else "hw_teleop"

        while not stop_event.is_set():
            now = time.monotonic()
            dt = max(1e-4, min(0.05, now - last_t))
            last_t = now

            with hw_estop_lock:
                local_disarm = hw_state["local_disarm"]
                clear_fault_req = hw_state["clear_fault_req"]
                hw_state["clear_fault_req"] = False

            if clear_fault_req:
                if backend.clear_safety_fault():
                    print("[v15] 已清除 M6 安全监督软停机锁存")

            raw_pkt = cmd_rx.poll_latest(now_mono=now)
            cmd = parse_teleop_packet(raw_pkt)

            # 若触发了本地 Space/h 急停，必须等 MuJoCo 侧显式切回 Disarm 后才允许再次解锁
            if local_disarm:
                if cmd is not None and not cmd.armed:
                    with hw_estop_lock:
                        hw_state["local_disarm"] = False
                    local_disarm = False
                elif cmd is not None and cmd.armed:
                    cmd = parse_teleop_packet({**raw_pkt, "armed": False, "mode": "HOLD"})

            robot_state = backend.get_state(now)
            res = controller.step(cmd, robot_state, dt_s=dt, apply_to_backend=True)

            if res.trigger_return_home:
                print(f"[v15 安全护盾] 越出硬工作空间 ({res.reason})，自动缓慢回 HOME！")
                backend.hold()
                backend.move_to_joints(home_joints, speed_deg_s=15.0, wait=True)
                controller.reset()

            reason_str = "LOCAL_ESTOP_HOLD" if local_disarm else res.reason

            state_pub.publish(
                q_act=robot_state.q_act,
                q_cmd=res.q_cmd_next,
                dq_act=robot_state.dq_act,
                tau_act=robot_state.tau_act,
                gripper_deg=backend.get_gripper_deg(),
                visible=False,
                following=res.armed,
                mode=res.mode,
                err_norm_m=0.0,
                track_err_deg=res.tracking_err_max_deg,
                tmos_max=robot_state.t_mos_max_c,
                m6_fault=robot_state.supervisor_latched,
                source=source_tag,
                teleop_armed=res.armed,
                teleop_mode=res.mode,
                teleop_reason=reason_str,
                teleop_speed_deg_s=res.effective_speed_deg_s,
                cmd_seq_ack=res.cmd_seq_ack,
            )

            if now - last_print_t >= 1.0:
                last_print_t = now
                print(
                    f"[v15] mode={res.mode:12s} armed={'ON ' if res.armed else 'OFF'} "
                    f"reason={reason_str:24s} align={res.align_err_deg:5.1f}° "
                    f"trk={res.tracking_err_max_deg:4.2f}° r={res.reach_radius_m:.2f}m "
                    f"MOS={robot_state.t_mos_max_c:4.1f}°C cmd_hz={cmd_rx.rx_hz:4.0f}"
                )

            next_tick += dt_nom
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[v15] 收到 Ctrl+C，进入统一安全退出链...")
    finally:
        stop_event.set()
        if kb_listener is not None:
            try:
                kb_listener.stop()
            except Exception:
                pass
        if cmd_rx is not None:
            cmd_rx.close()
        if state_pub is not None:
            state_pub.close()
        if backend is not None:
            print("[v15] 机械臂保持并安全缓慢回零...")
            backend.hold()
            backend.stop()
        print("[v15] 安全退出完成。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
