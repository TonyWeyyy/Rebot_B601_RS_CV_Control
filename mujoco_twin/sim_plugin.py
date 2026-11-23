"""
MuJoCo Studio 仿真侧数字孪生插件（支持热重载）
==============================================
核心模式（自动优先级切换）：
  1. LIVE_HW / LIVE_SIM：本机非阻塞监听 UDP 127.0.0.1:19870，只要 `palm_follow_v14.py`
     （真机或 `--sim`）运行，虚拟 B601-RS 机械臂、夹爪、3D 手掌球与指令 TCP 标记即刻以
     60-100Hz 与真实机械臂完全同步运动！
  2. REPLAY：未连接实时流时，可在右侧面板一键加载并回放 `logs/servo_v14_*.csv` /
     `logs/servo_*.csv` 真实硬件运行日志，在 MuJoCo 中 1:1 复现真机与手掌轨迹。
  3. MANUAL：面板开启手动测试模式后，可直接拖动 6 轴与夹爪滑块校验运动学。
  4. STANDBY：空闲待命在示教 HOME 位姿，随时等待真机上线自动接管。
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import pathlib
import sys
import time
from typing import Any

import messages as user_messages
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

# 确保工作区在 sys.path，并通过文件路径直载 mujoco_bridge.py（避免触发 servo_v14/__init__.py 依赖 yaml）
TWIN_DIR = pathlib.Path(os.path.abspath(__file__)).parent
PROJECT_ROOT = TWIN_DIR.parent
if str(TWIN_DIR) not in sys.path:
    sys.path.insert(0, str(TWIN_DIR))

import scene_builder  # noqa: E402


def _load_bridge_module():
    bridge_path = PROJECT_ROOT / "servo_v14" / "mujoco_bridge.py"
    spec = importlib.util.spec_from_file_location("_rebot_mujoco_bridge", str(bridge_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载桥接模块: {bridge_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BRIDGE = _load_bridge_module()
DEFAULT_MUJOCO_PORT = _BRIDGE.DEFAULT_MUJOCO_PORT
MuJoCoTwinReceiver = _BRIDGE.MuJoCoTwinReceiver

HIDDEN_POS = np.array([0.0, 0.0, -10.0], dtype=np.float64)
WORKSPACE_RING_Z = 0.004


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
        return val if math.isfinite(val) else default
    except ValueError:
        return default


def _load_csv_replay(csv_path: pathlib.Path) -> list[dict[str, Any]]:
    """解析 100Hz 实机诊断 CSV 日志为轻量帧列表，供 MuJoCo 实时回放。"""
    frames: list[dict[str, Any]] = []
    if not csv_path.is_file():
        return frames
    t0: float | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_abs = _safe_float(row, "host_time", 0.0)
            if t0 is None:
                t0 = t_abs
            t_rel = max(0.0, t_abs - t0)

            q_cmd = [_safe_float(row, f"q_cmd_{i}", 0.0) for i in range(1, 7)]
            # 优先使用 200Hz MIT 实测关节角 qa_1..6，若旧日志无此列则回退 q_cmd
            fbc_valid = _safe_float(row, "fbc_valid", 0.0) > 0.5
            has_qa = row.get("qa_1") not in (None, "")
            if fbc_valid or has_qa:
                q_act = [
                    _safe_float(row, f"qa_{i}", q_cmd[i - 1])
                    for i in range(1, 7)
                ]
            else:
                q_act = list(q_cmd)
            dq_act = [_safe_float(row, f"dqa_{i}", 0.0) for i in range(1, 7)]
            tau_act = [_safe_float(row, f"tau_{i}", 0.0) for i in range(1, 7)]

            vis = _safe_float(row, "visible", 0.0) > 0.5
            following = _safe_float(row, "following", 0.0) > 0.5
            static_hold = _safe_float(row, "static_hold", 0.0) > 0.5
            err_norm = _safe_float(row, "error_norm", 0.0)
            tmos = _safe_float(row, "tmos_max", 25.0)

            tx_raw = row.get("target_x", "")
            ty_raw = row.get("target_y", "")
            tz_raw = row.get("target_z", "")
            p_cam = None
            if vis and tx_raw != "" and ty_raw != "" and tz_raw != "":
                p_cam = [
                    _safe_float(row, "target_x", 0.0),
                    _safe_float(row, "target_y", 0.0),
                    _safe_float(row, "target_z", 0.4),
                ]

            if not following:
                mode = "IDLE"
            elif static_hold:
                mode = "HOLDING"
            elif vis:
                mode = "TRACKING"
            else:
                mode = "LOST"

            track_err_deg = max(
                abs(math.degrees(q_cmd[i] - q_act[i])) for i in range(6)
            )
            frames.append({
                "t": t_rel,
                "q_act": q_act,
                "q_cmd": q_cmd,
                "dq_act": dq_act,
                "tau_act": tau_act,
                "p_cam": p_cam,
                "visible": vis,
                "following": following,
                "mode": mode,
                "err_norm_m": err_norm,
                "track_err_deg": track_err_deg,
                "tmos_max": tmos,
            })
    return frames


class SimPlugin(plugin.SimPlugin):
    """B601-RS 实时数字孪生仿真插件（监听 UDP 实时流 + 内置实机日志回放）。"""

    # 类级默认值（保障热重载后旧实例不报 AttributeError）
    udp_port: int = DEFAULT_MUJOCO_PORT
    show_palm: bool = True
    show_cmd_marker: bool = True
    show_workspace_ring: bool = True
    show_camera_ray: bool = True
    manual_mode: bool = False
    manual_q_deg: tuple[float, ...] = (-24.7, 19.7, 21.1, 5.4, 0.1, -0.1)
    manual_gripper_deg: float = 285.2

    _receiver: MuJoCoTwinReceiver | None = None
    _joint_offset_rad: np.ndarray | None = None
    _home_q_rad: np.ndarray | None = None
    _cmd_data: mujoco.MjData | None = None
    _last_telem_mono: float = 0.0
    _last_step_wall_mono: float = 0.0

    # 回放状态
    _available_logs: tuple[str, ...] = ()
    _replay_file: str = ""
    _replay_frames: list[dict[str, Any]] | None = None
    _replay_time_s: float = 0.0
    _replay_idx: int = 0
    _replay_playing: bool = False
    _replay_speed: float = 1.0

    def _ensure_initialized(self) -> None:
        if self._joint_offset_rad is None:
            self._joint_offset_rad = scene_builder.load_joint_offset_rad()
        if self._home_q_rad is None:
            self._home_q_rad = scene_builder.load_home_q_rad()
            self.manual_q_deg = tuple(round(math.degrees(x), 1) for x in self._home_q_rad)
        if self._receiver is None:
            self._receiver = MuJoCoTwinReceiver(port=self.udp_port, stale_timeout_s=0.6)
        if not self._available_logs:
            self._scan_logs()
        if self._replay_frames is None:
            self._replay_frames = []

    def _scan_logs(self) -> None:
        log_dir = PROJECT_ROOT / "logs"
        if not log_dir.is_dir():
            self._available_logs = ()
            return
        files = sorted(
            [p.name for p in log_dir.glob("servo_*.csv") if p.is_file()],
            reverse=True,
        )
        self._available_logs = tuple(files[:30])
        if not self._replay_file and self._available_logs:
            self._replay_file = self._available_logs[0]

    def _load_replay_by_name(self, name: str) -> None:
        log_path = PROJECT_ROOT / "logs" / name
        self._replay_frames = _load_csv_replay(log_path)
        self._replay_file = name if self._replay_frames else ""
        self._replay_idx = 0
        self._replay_time_s = 0.0

    def _set_mocap_pos(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
        pos_xyz: np.ndarray | list[float],
    ) -> None:
        try:
            mid = int(model.body(body_name).mocapid[0])
            if 0 <= mid < model.nmocap:
                data.mocap_pos[mid] = pos_xyz
        except KeyError:
            pass

    def _apply_arm_joints(
        self,
        data: mujoco.MjData,
        q_sdk: list[float] | np.ndarray,
        dq_sdk: list[float] | np.ndarray | None,
        gripper_deg: float,
    ) -> None:
        offset = self._joint_offset_rad if self._joint_offset_rad is not None else np.zeros(6)
        for i, jname in enumerate(scene_builder.ARM_JOINT_NAMES):
            data.joint(jname).qpos[0] = float(q_sdk[i]) + float(offset[i])
            if dq_sdk is not None:
                data.joint(jname).qvel[0] = float(dq_sdk[i])
            else:
                data.joint(jname).qvel[0] = 0.0
        stroke = scene_builder.gripper_deg_to_prismatic_m(gripper_deg)
        for gname in scene_builder.GRIPPER_JOINT_NAMES:
            data.joint(gname).qpos[0] = stroke
            data.joint(gname).qvel[0] = 0.0

    def _compute_cmd_tcp_world(
        self,
        model: mujoco.MjModel,
        q_cmd_sdk: list[float] | np.ndarray,
    ) -> np.ndarray:
        if self._cmd_data is None or self._cmd_data.qpos.shape[0] != model.nq:
            self._cmd_data = mujoco.MjData(model)
        offset = self._joint_offset_rad if self._joint_offset_rad is not None else np.zeros(6)
        for i, jname in enumerate(scene_builder.ARM_JOINT_NAMES):
            self._cmd_data.joint(jname).qpos[0] = float(q_cmd_sdk[i]) + float(offset[i])
        mujoco.mj_kinematics(model, self._cmd_data)
        return self._cmd_data.body("link6").xpos.copy()

    def _cam_to_world(self, data: mujoco.MjData, p_cam: list[float] | np.ndarray) -> np.ndarray:
        cam_site = data.site("gemini2_cam")
        R_wc = cam_site.xmat.reshape(3, 3)
        p_wc = cam_site.xpos
        return p_wc + R_wc @ np.asarray(p_cam, dtype=np.float64)

    def _apply_pose_to_model(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        q_act: list[float] | np.ndarray,
        q_cmd: list[float] | np.ndarray,
        dq_act: list[float] | np.ndarray | None,
        gripper_deg: float,
        p_world: list[float] | np.ndarray | None,
        p_cam: list[float] | np.ndarray | None,
        visible: bool,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
        self._apply_arm_joints(data, q_act, dq_act, gripper_deg)
        mujoco.mj_forward(model, data)

        # 优先使用世界系掌心估计，否则按相机站点外参即时投影 p_cam -> world
        palm_w: np.ndarray | None = None
        if visible:
            if p_world is not None:
                palm_w = np.asarray(p_world, dtype=np.float64)
            elif p_cam is not None:
                palm_w = self._cam_to_world(data, p_cam)

        if self.show_palm and palm_w is not None and np.all(np.isfinite(palm_w)):
            self._set_mocap_pos(model, data, "palm_target", palm_w)
        else:
            self._set_mocap_pos(model, data, "palm_target", HIDDEN_POS)

        if self.show_cmd_marker:
            cmd_tcp = self._compute_cmd_tcp_world(model, q_cmd)
            self._set_mocap_pos(model, data, "cmd_tcp_marker", cmd_tcp)
        else:
            self._set_mocap_pos(model, data, "cmd_tcp_marker", HIDDEN_POS)

        ring_pos = [0.0, 0.0, WORKSPACE_RING_Z] if self.show_workspace_ring else HIDDEN_POS
        self._set_mocap_pos(model, data, "workspace_ring", ring_pos)

        mujoco.mj_forward(model, data)
        tcp = data.body("link6").xpos
        tcp_xyz = (float(tcp[0]), float(tcp[1]), float(tcp[2]))
        palm_xyz = (
            (float(palm_w[0]), float(palm_w[1]), float(palm_w[2]))
            if palm_w is not None and np.all(np.isfinite(palm_w))
            else None
        )
        return tcp_xyz, palm_xyz

    @messages.handler
    def on_load_model(self, event: messages.LoadModelEvent) -> bool:
        if event.path:
            return False
        self._ensure_initialized()
        spec = scene_builder.build_b601_spec(
            show_camera_ray=self.show_camera_ray,
            home_q_sdk=self._home_q_rad,
            joint_offset_rad=self._joint_offset_rad,
        )
        self.handle.set_model(spec, watch=[scene_builder.URDF_PATH])
        return True

    @messages.handler
    def on_post_model(self, event: messages.PostModelEvent) -> None:
        self._ensure_initialized()
        self._cmd_data = mujoco.MjData(event.model)
        q_home = self._home_q_rad if self._home_q_rad is not None else scene_builder.DEFAULT_HOME_Q
        self._apply_pose_to_model(
            event.model,
            event.data,
            q_act=q_home,
            q_cmd=q_home,
            dq_act=None,
            gripper_deg=285.2,
            p_world=None,
            p_cam=None,
            visible=False,
        )

    @messages.handler
    def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
        if self in event.plugins:
            self.handle.reload()

    @messages.handler
    def on_display_config(self, snapshot: user_messages.TwinDisplayConfig) -> None:
        ray_changed = snapshot.show_camera_ray != self.show_camera_ray
        self.show_palm = snapshot.show_palm
        self.show_cmd_marker = snapshot.show_cmd_marker
        self.show_workspace_ring = snapshot.show_workspace_ring
        self.show_camera_ray = snapshot.show_camera_ray
        self.manual_mode = snapshot.manual_mode
        self.manual_q_deg = snapshot.manual_q_deg
        self.manual_gripper_deg = snapshot.manual_gripper_deg
        if ray_changed:
            self.handle.reload()

    @messages.handler
    def on_action(self, event: user_messages.TwinActionEvent) -> None:
        self._ensure_initialized()
        act = event.action
        if act == "refresh_logs":
            self._scan_logs()
        elif act == "replay_load":
            if event.log_name:
                self._load_replay_by_name(event.log_name)
                self._replay_playing = False
        elif act == "replay_play":
            if not self._replay_frames and self._replay_file:
                self._load_replay_by_name(self._replay_file)
            if self._replay_frames:
                if self._replay_idx >= len(self._replay_frames) - 1:
                    self._replay_idx = 0
                    self._replay_time_s = 0.0
                self._replay_playing = True
                self.manual_mode = False
                self._last_step_wall_mono = time.monotonic()
        elif act == "replay_pause":
            self._replay_playing = False
        elif act == "replay_stop":
            self._replay_playing = False
            self._replay_idx = 0
            self._replay_time_s = 0.0
        elif act == "replay_seek":
            if self._replay_frames:
                frac = float(np.clip(event.seek_frac, 0.0, 1.0))
                idx = int(round(frac * (len(self._replay_frames) - 1)))
                self._replay_idx = max(0, min(len(self._replay_frames) - 1, idx))
                self._replay_time_s = float(self._replay_frames[self._replay_idx]["t"])
        elif act == "replay_speed":
            self._replay_speed = max(0.1, min(4.0, float(event.speed)))
        elif act == "reset_home":
            self._replay_playing = False
            self._replay_frames = []
            self.manual_mode = False
            if self._home_q_rad is not None:
                self.manual_q_deg = tuple(round(math.degrees(x), 1) for x in self._home_q_rad)
            self.manual_gripper_deg = 285.2

    @messages.handler
    def on_post_step(self, event: messages.PostStepEvent) -> None:
        self._ensure_initialized()
        now = time.monotonic()
        dt_wall = (
            min(0.1, max(0.0, now - self._last_step_wall_mono))
            if self._last_step_wall_mono > 0.0
            else 0.01
        )
        self._last_step_wall_mono = now

        model = event.model
        data = event.data

        # 1) 优先检查真机 / SIM 实时 UDP 流
        pkt = self._receiver.poll_latest() if self._receiver is not None else None
        if pkt is not None:
            src = str(pkt.get("source", "hw"))
            link_status = "LIVE_SIM" if src == "sim" else "LIVE_HW"
            q_act = pkt.get("q_act", [0.0] * 6)
            q_cmd = pkt.get("q_cmd", q_act)
            dq_act = pkt.get("dq_act", [0.0] * 6)
            tau_act = pkt.get("tau_act", [0.0] * 6)
            gripper_deg = float(pkt.get("gripper_deg", 285.2))
            p_world = pkt.get("p_world")
            p_cam = pkt.get("p_cam")
            visible = bool(pkt.get("visible", False))
            following = bool(pkt.get("following", False))
            fsm_mode = str(pkt.get("mode", "IDLE"))
            err_norm_cm = float(pkt.get("err_norm_m", 0.0)) * 100.0
            track_err_deg = float(pkt.get("track_err_deg", 0.0))
            tmos_max = float(pkt.get("tmos_max", 25.0))
            m6_fault = bool(pkt.get("m6_fault", False))
        elif self._replay_frames:
            # 2) CSV 真实硬件日志回放模式
            frames = self._replay_frames
            duration = float(frames[-1]["t"]) if frames else 0.0
            if self._replay_playing and duration > 0.0:
                self._replay_time_s += dt_wall * self._replay_speed
                if self._replay_time_s >= duration:
                    self._replay_time_s = duration
                    self._replay_idx = len(frames) - 1
                    self._replay_playing = False
                else:
                    while (
                        self._replay_idx + 1 < len(frames)
                        and frames[self._replay_idx + 1]["t"] <= self._replay_time_s
                    ):
                        self._replay_idx += 1
            fr = frames[self._replay_idx]
            link_status = "REPLAY"
            src = "csv_log"
            q_act = fr["q_act"]
            q_cmd = fr["q_cmd"]
            dq_act = fr["dq_act"]
            tau_act = fr["tau_act"]
            gripper_deg = 285.2
            p_world = None
            p_cam = fr["p_cam"]
            visible = fr["visible"]
            following = fr["following"]
            fsm_mode = fr["mode"]
            err_norm_cm = fr["err_norm_m"] * 100.0
            track_err_deg = fr["track_err_deg"]
            tmos_max = fr["tmos_max"]
            m6_fault = False
        elif self.manual_mode:
            # 3) 面板手动滑块测试模式
            link_status = "MANUAL"
            src = "ui"
            q_act = [math.radians(x) for x in self.manual_q_deg]
            q_cmd = list(q_act)
            dq_act = [0.0] * 6
            tau_act = [0.0] * 6
            gripper_deg = float(self.manual_gripper_deg)
            p_world = None
            p_cam = None
            visible = False
            following = False
            fsm_mode = "MANUAL"
            err_norm_cm = 0.0
            track_err_deg = 0.0
            tmos_max = 25.0
            m6_fault = False
        else:
            # 4) 待机模式：保持示教 HOME，随时等待真机 UDP 接入
            link_status = "STANDBY"
            src = "none"
            q_home = self._home_q_rad if self._home_q_rad is not None else scene_builder.DEFAULT_HOME_Q
            q_act = q_home.tolist()
            q_cmd = list(q_act)
            dq_act = [0.0] * 6
            tau_act = [0.0] * 6
            gripper_deg = 285.2
            p_world = None
            p_cam = None
            visible = False
            following = False
            fsm_mode = "IDLE"
            err_norm_cm = 0.0
            track_err_deg = 0.0
            tmos_max = 25.0
            m6_fault = False

        tcp_xyz, palm_xyz = self._apply_pose_to_model(
            model,
            data,
            q_act=q_act,
            q_cmd=q_cmd,
            dq_act=dq_act,
            gripper_deg=gripper_deg,
            p_world=p_world,
            p_cam=p_cam,
            visible=visible,
        )

        # 30Hz 节流推送遥测到 UI 面板，减少 WebSocket 负载
        if now - self._last_telem_mono >= 0.033:
            self._last_telem_mono = now
            frames = self._replay_frames or []
            duration_s = float(frames[-1]["t"]) if frames else 0.0
            self.handle.send_to_viewer(
                user_messages.TwinTelemetry(
                    link_status=link_status,
                    source=src,
                    udp_port=self.udp_port,
                    rx_hz=self._receiver.rx_hz if self._receiver is not None else 0.0,
                    bind_error=(self._receiver.bind_error or "") if self._receiver is not None else "",
                    fsm_mode=fsm_mode,
                    following=following,
                    visible=visible,
                    m6_fault=m6_fault,
                    q_act_deg=tuple(math.degrees(x) for x in q_act),
                    q_cmd_deg=tuple(math.degrees(x) for x in q_cmd),
                    dq_act_deg_s=tuple(math.degrees(x) for x in dq_act),
                    tau_act=tuple(float(x) for x in tau_act),
                    gripper_deg=gripper_deg,
                    err_norm_cm=err_norm_cm,
                    track_err_deg=track_err_deg,
                    tmos_max=tmos_max,
                    tcp_xyz=tcp_xyz,
                    palm_xyz=palm_xyz,
                    replay_file=self._replay_file,
                    replay_idx=self._replay_idx,
                    replay_total=len(frames),
                    replay_time_s=self._replay_time_s,
                    replay_duration_s=duration_s,
                    replay_playing=self._replay_playing,
                    replay_speed=self._replay_speed,
                    available_logs=self._available_logs,
                )
            )

    @messages.handler
    def on_exit(self, event: messages.ExitEvent) -> bool:
        del event
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
        return False  # 绝不拦截 ExitEvent，确保所有插件正常退出
