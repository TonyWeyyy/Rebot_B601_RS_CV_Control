"""Shared Studio message contracts between mujoco_twin/sim_plugin.py and viewer_plugin.py (v15.0)."""

from __future__ import annotations

import dataclasses
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class TwinTelemetry(messages.Snapshot):
    """Sim -> UI: latest-wins real-time digital twin & bidirectional teleop telemetry."""

    link_status: str = "STANDBY"  # LIVE_HW | LIVE_SIM | LIVE_HW_TELEOP | LIVE_SIM_TELEOP | REPLAY | TELEOP_PREVIEW | STANDBY
    source: str = "none"
    udp_port: int = 19870
    cmd_port: int = 19871
    rx_hz: float = 0.0
    bind_error: str = ""

    fsm_mode: str = "IDLE"
    following: bool = False
    visible: bool = False
    m6_fault: bool = False

    q_act_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q_cmd_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q_des_deg: tuple[float, ...] = (-24.7, 19.7, 21.1, 5.4, 0.1, -0.1)
    dq_act_deg_s: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tau_act: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gripper_deg: float = 285.2
    des_gripper_deg: float = 285.2

    err_norm_cm: float = 0.0
    track_err_deg: float = 0.0
    tmos_max: float = 25.0

    tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    palm_xyz: tuple[float, float, float] | None = None

    # v15.0 双向遥操作与 3D 目标球 IK 状态
    teleop_mode: str = "MONITOR"  # MONITOR | JOINT | IK_3D
    teleop_armed_req: bool = False
    hw_teleop_connected: bool = False
    hw_teleop_armed: bool = False
    hw_teleop_reason: str = "NO_TELEOP_BRIDGE"
    teleop_max_speed_deg_s: float = 25.0
    ik_lock_orientation: bool = True
    ik_target_xyz: tuple[float, float, float] = (0.135, 0.061, 0.314)
    ik_residual_mm: float = 0.0
    align_err_deg: float = 0.0

    # CSV 日志回放状态
    replay_file: str = ""
    replay_idx: int = 0
    replay_total: int = 0
    replay_time_s: float = 0.0
    replay_duration_s: float = 0.0
    replay_playing: bool = False
    replay_speed: float = 1.0
    available_logs: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class TwinDisplayConfig(messages.Snapshot):
    """UI -> Sim: visual overlays, teleop mode, and joint/IK parameters."""

    show_palm: bool = True
    show_cmd_marker: bool = True
    show_workspace_ring: bool = True
    show_camera_ray: bool = True

    teleop_mode: str = "MONITOR"  # MONITOR | JOINT | IK_3D
    teleop_max_speed_deg_s: float = 25.0
    ik_lock_orientation: bool = True
    manual_q_deg: tuple[float, ...] = (-24.7, 19.7, 21.1, 5.4, 0.1, -0.1)
    manual_gripper_deg: float = 285.2


@dataclasses.dataclass(frozen=True)
class TwinActionEvent(messages.Event):
    """UI -> Sim: discrete replay, teleop arming, sync, and 3D IK target actions."""

    action: str = ""
    log_name: str = ""
    seek_frac: float = 0.0
    speed: float = 1.0
    ik_xyz: tuple[float, float, float] = (0.135, 0.061, 0.314)
