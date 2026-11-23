"""
MuJoCo Studio 仿真侧双向数字孪生插件（v15.0，支持热重载）
=========================================================
支持 4 种工作形态：
  1. MONITOR（v14 只读镜像，默认安全态）：
     - 实时监听 UDP `:19870`，与 `palm_follow_v14.py`（真机/SIM）100Hz 同步运动；
     - 未连接实时流时，可在面板一键回放 `logs/servo_*.csv` 实机日志。
  2. JOINT（v15 关节滑块控真机 / 离线预览）：
     - 在右侧面板拖动 J1~J6 与 J7 夹爪滑块；
     - 未解锁时预览青色目标 TCP；点击 `ARM Real Robot` 后先自动对齐真机当前姿态，
       再经 UDP `:19871` 以 60Hz 发送至 `mujoco_teleop_v15.py` 平滑驱动真实机械臂。
  3. IK_3D（v15 3D 视口拖拽目标球实时 IK 控真机 / 离线预览）：
     - 显示橙金色 `ik_target` 3D 目标球（双击选中后 `Ctrl+右键` 平移、`Ctrl+左键` 旋转，
       亦可在面板微调 X/Y/Z 滑块）；
     - 内置基于 MuJoCo 解析雅可比 `mj_jacBody` 的阻尼最小二乘（DLS）IK 求解器
       （工作空间 r≤0.42m、桌面 Z≥0.06m 与 URDF 关节软限位约束），实时解算 6 轴关节角；
     - 解锁后平滑驱动真机同步追球，同时回显真机实测 `q_act / tau_act / MOS`。
  4. STANDBY：空闲待命在示教 HOME 位姿。
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

# 确保工作区在 sys.path，并通过文件路径直载 mujoco_bridge.py（不依赖 yaml）
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
DEFAULT_TELEOP_CMD_PORT = _BRIDGE.DEFAULT_TELEOP_CMD_PORT
MuJoCoTwinReceiver = _BRIDGE.MuJoCoTwinReceiver
MuJoCoTeleopCmdPublisher = _BRIDGE.MuJoCoTeleopCmdPublisher

HIDDEN_POS = np.array([0.0, 0.0, -10.0], dtype=np.float64)
WORKSPACE_RING_Z = 0.004

# B601-RS URDF 安全软限位 (rad)
_Q_MIN_SDK = np.array([-2.80, -0.15, -0.25, -1.75, -1.57, -2.80], dtype=np.float64) + 0.03
_Q_MAX_SDK = np.array([2.80, 3.05, 3.05, 1.75, 1.57, 2.80], dtype=np.float64) - 0.03
_IK_MAX_RADIUS_M = 0.42
_IK_Z_MIN_M = 0.06
_IK_Z_MAX_M = 0.62


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
        return val if math.isfinite(val) else default
    except ValueError:
        return default


def _clamp_ik_target_xyz(pos: np.ndarray) -> np.ndarray:
    """将 3D IK 目标位置裁剪到水平 0.42m 柱面与 [0.06, 0.62]m 安全高度包络内。"""
    out = np.asarray(pos, dtype=np.float64).copy()
    r = math.hypot(float(out[0]), float(out[1]))
    if r > _IK_MAX_RADIUS_M and r > 1e-9:
        scale = _IK_MAX_RADIUS_M / r
        out[0] *= scale
        out[1] *= scale
    out[2] = float(np.clip(out[2], _IK_Z_MIN_M, _IK_Z_MAX_M))
    return out


def _quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q / n)
    return mat.reshape(3, 3)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    n = float(np.linalg.norm(q))
    return (q / n) if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


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
            fbc_valid = _safe_float(row, "fbc_valid", 0.0) > 0.5
            has_qa = row.get("qa_1") not in (None, "")
            if fbc_valid or has_qa:
                q_act = [_safe_float(row, f"qa_{i}", q_cmd[i - 1]) for i in range(1, 7)]
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

            track_err_deg = max(abs(math.degrees(q_cmd[i] - q_act[i])) for i in range(6))
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
    """B601-RS v15.0 双向数字孪生仿真插件（实时镜像 + 日志回放 + 关节滑块/3D IK 控真机）。"""

    # 类级默认值（保障热重载后旧实例不报 AttributeError）
    udp_port: int = DEFAULT_MUJOCO_PORT
    cmd_port: int = DEFAULT_TELEOP_CMD_PORT

    show_palm: bool = True
    show_cmd_marker: bool = True
    show_workspace_ring: bool = True
    show_camera_ray: bool = True

    # v15.0 遥操作状态
    teleop_mode: str = "MONITOR"  # MONITOR | JOINT | IK_3D
    teleop_max_speed_deg_s: float = 25.0
    ik_lock_orientation: bool = True
    manual_q_deg: tuple[float, ...] = (-24.7, 19.7, 21.1, 5.4, 0.1, -0.1)
    manual_gripper_deg: float = 285.2

    _receiver: MuJoCoTwinReceiver | None = None
    _cmd_pub: MuJoCoTeleopCmdPublisher | None = None
    _joint_offset_rad: np.ndarray | None = None
    _home_q_rad: np.ndarray | None = None
    _cmd_data: mujoco.MjData | None = None
    _ik_data: mujoco.MjData | None = None
    _arm_dof_ids: list[int] | None = None
    _last_telem_mono: float = 0.0
    _last_step_wall_mono: float = 0.0

    # v15.0 内部实时状态
    _teleop_armed_req: bool = False
    _q_des_sdk: np.ndarray | None = None
    _last_real_q_sdk: np.ndarray | None = None
    _last_real_gripper_deg: float = 285.2
    _ik_ref_quat: np.ndarray | None = None
    _ik_sync_pending: bool = False
    _ik_set_xyz_pending: np.ndarray | None = None
    _ik_last_xyz: np.ndarray | None = None
    _ik_residual_mm: float = 0.0
    _clear_fault_pulse: bool = False

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
        if self._q_des_sdk is None:
            self._q_des_sdk = self._home_q_rad.copy()
        if self._last_real_q_sdk is None:
            self._last_real_q_sdk = self._home_q_rad.copy()
        if self._ik_ref_quat is None:
            self._ik_ref_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if self._ik_last_xyz is None:
            self._ik_last_xyz = np.array([0.135, 0.061, 0.314], dtype=np.float64)
        if self._receiver is None:
            self._receiver = MuJoCoTwinReceiver(port=self.udp_port, stale_timeout_s=0.6)
        if self._cmd_pub is None:
            self._cmd_pub = MuJoCoTeleopCmdPublisher(port=self.cmd_port, max_hz=60.0, enabled=True)
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

    def _get_mocap_id(self, model: mujoco.MjModel, body_name: str) -> int:
        try:
            mid = int(model.body(body_name).mocapid[0])
            return mid if 0 <= mid < model.nmocap else -1
        except KeyError:
            return -1

    def _set_mocap_pos(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
        pos_xyz: np.ndarray | list[float],
    ) -> None:
        mid = self._get_mocap_id(model, body_name)
        if mid >= 0:
            data.mocap_pos[mid] = pos_xyz

    def _set_mocap_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
        pos_xyz: np.ndarray | list[float],
        quat_wxyz: np.ndarray | list[float],
    ) -> None:
        mid = self._get_mocap_id(model, body_name)
        if mid >= 0:
            data.mocap_pos[mid] = pos_xyz
            data.mocap_quat[mid] = quat_wxyz

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
            data.joint(jname).qvel[0] = float(dq_sdk[i]) if dq_sdk is not None else 0.0
        stroke = scene_builder.gripper_deg_to_prismatic_m(gripper_deg)
        for gname in scene_builder.GRIPPER_JOINT_NAMES:
            data.joint(gname).qpos[0] = stroke
            data.joint(gname).qvel[0] = 0.0

    def _fk_link6(
        self,
        model: mujoco.MjModel,
        q_sdk: list[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """使用独立 scratch MjData 计算任意 SDK 关节角下 link6 的世界系 (pos, quat_wxyz)。"""
        if self._cmd_data is None or self._cmd_data.qpos.shape[0] != model.nq:
            self._cmd_data = mujoco.MjData(model)
        offset = self._joint_offset_rad if self._joint_offset_rad is not None else np.zeros(6)
        for i, jname in enumerate(scene_builder.ARM_JOINT_NAMES):
            self._cmd_data.joint(jname).qpos[0] = float(q_sdk[i]) + float(offset[i])
        mujoco.mj_kinematics(model, self._cmd_data)
        body = self._cmd_data.body("link6")
        return body.xpos.copy(), _rotmat_to_quat(body.xmat.reshape(3, 3))

    def _solve_ik_dls(
        self,
        model: mujoco.MjModel,
        q_seed_sdk: np.ndarray,
        target_pos_raw: np.ndarray,
        target_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """基于 MuJoCo 解析雅可比 mj_jacBody 的快速阻尼最小二乘（DLS）IK 求解器。"""
        if self._ik_data is None or self._ik_data.qpos.shape[0] != model.nq:
            self._ik_data = mujoco.MjData(model)
            self._arm_dof_ids = None
        if self._arm_dof_ids is None:
            self._arm_dof_ids = [int(model.joint(n).dofadr[0]) for n in scene_builder.ARM_JOINT_NAMES]

        offset = self._joint_offset_rad if self._joint_offset_rad is not None else np.zeros(6)
        target_pos = _clamp_ik_target_xyz(target_pos_raw)
        desired_quat = self._ik_ref_quat if self.ik_lock_orientation else target_quat_wxyz
        R_tgt = _quat_to_rotmat(desired_quat)

        q = np.clip(np.asarray(q_seed_sdk, dtype=np.float64).copy(), _Q_MIN_SDK, _Q_MAX_SDK)
        link6_id = int(model.body("link6").id)
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        dof_idx = self._arm_dof_ids

        damp = 0.045
        ori_weight = 0.45
        res_m = 0.0

        for _ in range(16):
            for i, jname in enumerate(scene_builder.ARM_JOINT_NAMES):
                self._ik_data.joint(jname).qpos[0] = float(q[i]) + float(offset[i])
            mujoco.mj_kinematics(model, self._ik_data)
            mujoco.mj_comPos(model, self._ik_data)

            body = self._ik_data.body("link6")
            p_cur = body.xpos
            R_cur = body.xmat.reshape(3, 3)

            e_pos = target_pos - p_cur
            res_m = float(np.linalg.norm(e_pos))
            # SO(3) 姿态误差（旋转向量近似）
            e_ori = 0.5 * (
                np.cross(R_cur[:, 0], R_tgt[:, 0])
                + np.cross(R_cur[:, 1], R_tgt[:, 1])
                + np.cross(R_cur[:, 2], R_tgt[:, 2])
            )
            if res_m < 5e-4 and float(np.linalg.norm(e_ori)) < 4e-3:
                break

            mujoco.mj_jacBody(model, self._ik_data, jacp, jacr, link6_id)
            Jp = jacp[:, dof_idx]
            Jr = jacr[:, dof_idx] * ori_weight
            J = np.vstack([Jp, Jr])
            err6 = np.concatenate([e_pos, e_ori * ori_weight])

            # 阻尼最小二乘：dq = J^T (J J^T + λ² I)^-1 e
            A = J @ J.T + (damp * damp) * np.eye(6, dtype=np.float64)
            dq = J.T @ np.linalg.solve(A, err6)
            max_step = float(np.max(np.abs(dq)))
            if max_step > 0.18:
                dq *= 0.18 / max_step
            q = np.clip(q + dq, _Q_MIN_SDK, _Q_MAX_SDK)

        return q, res_m * 1000.0, target_pos

    def _cam_to_world(self, data: mujoco.MjData, p_cam: list[float] | np.ndarray) -> np.ndarray:
        cam_site = data.site("gemini2_cam")
        R_wc = cam_site.xmat.reshape(3, 3)
        p_wc = cam_site.xpos
        return p_wc + R_wc @ np.asarray(p_cam, dtype=np.float64)

    def _sync_virtual_to_q(self, q_sdk: np.ndarray, gripper_deg: float) -> None:
        q_clean = np.clip(np.asarray(q_sdk, dtype=np.float64), _Q_MIN_SDK, _Q_MAX_SDK)
        self._q_des_sdk = q_clean.copy()
        self.manual_q_deg = tuple(round(math.degrees(x), 1) for x in q_clean)
        self.manual_gripper_deg = float(np.clip(gripper_deg, 5.2, 285.2))
        self._ik_sync_pending = True

    def _apply_pose_to_model(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        q_display: list[float] | np.ndarray,
        q_marker: list[float] | np.ndarray,
        dq_display: list[float] | np.ndarray | None,
        gripper_deg: float,
        p_world: list[float] | np.ndarray | None,
        p_cam: list[float] | np.ndarray | None,
        visible: bool,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
        self._apply_arm_joints(data, q_display, dq_display, gripper_deg)
        mujoco.mj_forward(model, data)

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
            cmd_tcp, _ = self._fk_link6(model, q_marker)
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
        self._ik_data = mujoco.MjData(event.model)
        self._arm_dof_ids = None
        q_home = self._home_q_rad if self._home_q_rad is not None else scene_builder.DEFAULT_HOME_Q
        self._apply_pose_to_model(
            event.model,
            event.data,
            q_display=q_home,
            q_marker=q_home,
            dq_display=None,
            gripper_deg=285.2,
            p_world=None,
            p_cam=None,
            visible=False,
        )
        self._set_mocap_pos(event.model, event.data, "ik_target", HIDDEN_POS)
        tcp_pos, tcp_quat = self._fk_link6(event.model, q_home)
        self._ik_ref_quat = tcp_quat
        self._ik_last_xyz = tcp_pos

    @messages.handler
    def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
        if self in event.plugins:
            self.handle.reload()

    @messages.handler
    def on_display_config(self, snapshot: user_messages.TwinDisplayConfig) -> None:
        self._ensure_initialized()
        ray_changed = snapshot.show_camera_ray != self.show_camera_ray
        mode_changed = snapshot.teleop_mode != self.teleop_mode

        self.show_palm = snapshot.show_palm
        self.show_cmd_marker = snapshot.show_cmd_marker
        self.show_workspace_ring = snapshot.show_workspace_ring
        self.show_camera_ray = snapshot.show_camera_ray
        self.teleop_max_speed_deg_s = float(np.clip(snapshot.teleop_max_speed_deg_s, 5.0, 60.0))
        self.ik_lock_orientation = snapshot.ik_lock_orientation

        if mode_changed:
            # 切换控制模式时强制解除武装，并把新模式起点同步到当前真实姿态，杜绝跳变
            self._teleop_armed_req = False
            self.teleop_mode = snapshot.teleop_mode
            if self.teleop_mode in ("JOINT", "IK_3D"):
                self._replay_playing = False
                self._replay_frames = []
                ref_q = self._last_real_q_sdk if self._last_real_q_sdk is not None else self._home_q_rad
                self._sync_virtual_to_q(ref_q, self._last_real_gripper_deg)
        else:
            if self.teleop_mode == "JOINT":
                self.manual_q_deg = snapshot.manual_q_deg
                self._q_des_sdk = np.clip(
                    np.radians(np.asarray(snapshot.manual_q_deg, dtype=np.float64)),
                    _Q_MIN_SDK,
                    _Q_MAX_SDK,
                )
            self.manual_gripper_deg = float(np.clip(snapshot.manual_gripper_deg, 5.2, 285.2))

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
                self._teleop_armed_req = False
                self.teleop_mode = "MONITOR"
                self._load_replay_by_name(event.log_name)
                self._replay_playing = False
        elif act == "replay_play":
            self._teleop_armed_req = False
            self.teleop_mode = "MONITOR"
            if not self._replay_frames and self._replay_file:
                self._load_replay_by_name(self._replay_file)
            if self._replay_frames:
                if self._replay_idx >= len(self._replay_frames) - 1:
                    self._replay_idx = 0
                    self._replay_time_s = 0.0
                self._replay_playing = True
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
            self._teleop_armed_req = False
            if self._home_q_rad is not None:
                self._sync_virtual_to_q(self._home_q_rad, 285.2)
        elif act == "teleop_sync_to_real":
            ref_q = self._last_real_q_sdk if self._last_real_q_sdk is not None else self._home_q_rad
            self._sync_virtual_to_q(ref_q, self._last_real_gripper_deg)
        elif act == "teleop_arm":
            # 解锁前强制先把虚拟目标对齐到真实位姿，确保零跳变通过真机对齐门控
            if self.teleop_mode in ("JOINT", "IK_3D"):
                ref_q = self._last_real_q_sdk if self._last_real_q_sdk is not None else self._home_q_rad
                self._sync_virtual_to_q(ref_q, self._last_real_gripper_deg)
                self._teleop_armed_req = True
        elif act == "teleop_disarm":
            self._teleop_armed_req = False
            if self._cmd_pub is not None and self._q_des_sdk is not None:
                self._cmd_pub.publish(
                    armed=False,
                    mode="HOLD",
                    q_des=self._q_des_sdk,
                    gripper_deg=self.manual_gripper_deg,
                    max_speed_deg_s=self.teleop_max_speed_deg_s,
                    force=True,
                )
        elif act == "teleop_set_ik_pos":
            self._ik_set_xyz_pending = _clamp_ik_target_xyz(np.asarray(event.ik_xyz, dtype=np.float64))
        elif act == "teleop_clear_fault":
            self._teleop_armed_req = False
            self._clear_fault_pulse = True

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

        # 1) 轮询通道 A（真机/SIM -> MuJoCo 状态流）
        pkt = self._receiver.poll_latest() if self._receiver is not None else None
        hw_connected = pkt is not None
        hw_teleop_connected = False
        hw_teleop_armed = False
        hw_teleop_reason = "NO_TELEOP_BRIDGE"

        if pkt is not None:
            src = str(pkt.get("source", "hw"))
            hw_teleop_connected = src in ("hw_teleop", "sim_teleop")
            hw_teleop_armed = bool(pkt.get("teleop_armed", False))
            hw_teleop_reason = str(pkt.get("teleop_reason", "")) or (
                "PALM_FOLLOW_ACTIVE" if not hw_teleop_connected else "OK"
            )
            self._last_real_q_sdk = np.asarray(pkt.get("q_act", [0.0] * 6), dtype=np.float64)
            self._last_real_gripper_deg = float(pkt.get("gripper_deg", 285.2))
            if bool(pkt.get("m6_fault", False)):
                self._teleop_armed_req = False
        else:
            src = "none"

        # 2) 处理 3D IK 目标球位置与实时 DLS IK 求解
        ik_mid = self._get_mocap_id(model, "ik_target")
        if self.teleop_mode == "IK_3D":
            if self._ik_sync_pending:
                self._ik_sync_pending = False
                ref_q = self._q_des_sdk if self._q_des_sdk is not None else self._home_q_rad
                tcp_pos, tcp_quat = self._fk_link6(model, ref_q)
                self._ik_ref_quat = tcp_quat
                self._ik_last_xyz = tcp_pos
                self._set_mocap_pose(model, data, "ik_target", tcp_pos, tcp_quat)
            elif self._ik_set_xyz_pending is not None:
                tgt = self._ik_set_xyz_pending
                self._ik_set_xyz_pending = None
                self._set_mocap_pos(model, data, "ik_target", tgt)

            if ik_mid >= 0:
                ik_pos = data.mocap_pos[ik_mid].copy()
                ik_quat = data.mocap_quat[ik_mid].copy()
                # 若仍在地下隐藏位，自动拉回当前末端位置
                if float(ik_pos[2]) < -1.0:
                    ref_q = self._last_real_q_sdk if hw_connected else self._q_des_sdk
                    tcp_pos, tcp_quat = self._fk_link6(model, ref_q)
                    self._ik_ref_quat = tcp_quat
                    self._set_mocap_pose(model, data, "ik_target", tcp_pos, tcp_quat)
                    ik_pos, ik_quat = tcp_pos, tcp_quat

                seed_q = self._q_des_sdk if self._q_des_sdk is not None else self._last_real_q_sdk
                q_sol, res_mm, clamped_pos = self._solve_ik_dls(model, seed_q, ik_pos, ik_quat)
                self._q_des_sdk = q_sol
                self._ik_residual_mm = res_mm
                self._ik_last_xyz = clamped_pos
                self.manual_q_deg = tuple(round(math.degrees(x), 1) for x in q_sol)
        else:
            # 非 IK_3D 模式下隐藏橙色拖拽目标球，保持 v14 视图纯净
            self._set_mocap_pos(model, data, "ik_target", HIDDEN_POS)
            self._ik_residual_mm = 0.0

        # 3) 通道 B：向真机遥操作桥接进程发送 60Hz 心跳与目标指令
        if self._cmd_pub is not None and self._q_des_sdk is not None:
            send_armed = bool(
                self._teleop_armed_req
                and self.teleop_mode in ("JOINT", "IK_3D")
                and hw_teleop_connected
            )
            send_mode = self.teleop_mode if self.teleop_mode in ("JOINT", "IK_3D") else "HOLD"
            clear_f = self._clear_fault_pulse
            self._clear_fault_pulse = False
            self._cmd_pub.publish(
                armed=send_armed,
                mode=send_mode,
                q_des=self._q_des_sdk,
                gripper_deg=self.manual_gripper_deg,
                max_speed_deg_s=self.teleop_max_speed_deg_s,
                ik_target_xyz=self._ik_last_xyz.tolist() if self.teleop_mode == "IK_3D" else None,
                clear_fault=clear_f,
                force=clear_f,
            )

        # 4) 决定 3D 机械臂展示姿态（真机连线时实体永远忠实显示真机 q_act，青球显示目标 q_des/q_cmd）
        if pkt is not None:
            if src == "hw_teleop":
                link_status = "LIVE_HW_TELEOP"
            elif src == "sim_teleop":
                link_status = "LIVE_SIM_TELEOP"
            elif src == "sim":
                link_status = "LIVE_SIM"
            else:
                link_status = "LIVE_HW"
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
            q_display = q_act
            q_marker = self._q_des_sdk if self.teleop_mode in ("JOINT", "IK_3D") else q_cmd
        elif self.teleop_mode in ("JOINT", "IK_3D"):
            # 无真机连接时的本地离线交互测试：实体臂直接跟随滑块 / 3D 目标球运动
            link_status = "TELEOP_PREVIEW"
            q_act = self._q_des_sdk.tolist()
            q_cmd = list(q_act)
            dq_act = [0.0] * 6
            tau_act = [0.0] * 6
            gripper_deg = float(self.manual_gripper_deg)
            p_world = None
            p_cam = None
            visible = False
            following = False
            fsm_mode = f"PREVIEW_{self.teleop_mode}"
            err_norm_cm = self._ik_residual_mm * 0.1
            track_err_deg = 0.0
            tmos_max = 25.0
            m6_fault = False
            q_display = q_act
            q_marker = q_act
        elif self._replay_frames:
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
            q_display = q_act
            q_marker = q_cmd
        else:
            link_status = "STANDBY"
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
            q_display = q_act
            q_marker = q_act

        tcp_xyz, palm_xyz = self._apply_pose_to_model(
            model,
            data,
            q_display=q_display,
            q_marker=q_marker,
            dq_display=dq_act,
            gripper_deg=gripper_deg,
            p_world=p_world,
            p_cam=p_cam,
            visible=visible,
        )

        align_err_deg = float(
            np.max(np.abs(np.degrees(self._q_des_sdk - np.asarray(q_act, dtype=np.float64))))
        )

        # 30Hz 节流推送遥测到 Dear ImGui 面板
        if now - self._last_telem_mono >= 0.033:
            self._last_telem_mono = now
            frames = self._replay_frames or []
            duration_s = float(frames[-1]["t"]) if frames else 0.0
            ik_xyz = tuple(float(x) for x in self._ik_last_xyz[:3])
            self.handle.send_to_viewer(
                user_messages.TwinTelemetry(
                    link_status=link_status,
                    source=src,
                    udp_port=self.udp_port,
                    cmd_port=self.cmd_port,
                    rx_hz=self._receiver.rx_hz if self._receiver is not None else 0.0,
                    bind_error=(self._receiver.bind_error or "") if self._receiver is not None else "",
                    fsm_mode=fsm_mode,
                    following=following,
                    visible=visible,
                    m6_fault=m6_fault,
                    q_act_deg=tuple(math.degrees(x) for x in q_act),
                    q_cmd_deg=tuple(math.degrees(x) for x in q_cmd),
                    q_des_deg=tuple(math.degrees(x) for x in self._q_des_sdk),
                    dq_act_deg_s=tuple(math.degrees(x) for x in dq_act),
                    tau_act=tuple(float(x) for x in tau_act),
                    gripper_deg=gripper_deg,
                    des_gripper_deg=self.manual_gripper_deg,
                    err_norm_cm=err_norm_cm,
                    track_err_deg=track_err_deg,
                    tmos_max=tmos_max,
                    tcp_xyz=tcp_xyz,
                    palm_xyz=palm_xyz,
                    teleop_mode=self.teleop_mode,
                    teleop_armed_req=self._teleop_armed_req,
                    hw_teleop_connected=hw_teleop_connected,
                    hw_teleop_armed=hw_teleop_armed,
                    hw_teleop_reason=hw_teleop_reason,
                    teleop_max_speed_deg_s=self.teleop_max_speed_deg_s,
                    ik_lock_orientation=self.ik_lock_orientation,
                    ik_target_xyz=ik_xyz,
                    ik_residual_mm=self._ik_residual_mm,
                    align_err_deg=align_err_deg,
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
        if self._cmd_pub is not None:
            if self._q_des_sdk is not None:
                self._cmd_pub.publish(
                    armed=False,
                    mode="HOLD",
                    q_des=self._q_des_sdk,
                    gripper_deg=self.manual_gripper_deg,
                    max_speed_deg_s=self.teleop_max_speed_deg_s,
                    force=True,
                )
            self._cmd_pub.close()
            self._cmd_pub = None
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
        return False
