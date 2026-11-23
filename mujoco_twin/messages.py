"""Shared Studio message contracts between mujoco_twin/sim_plugin.py and viewer_plugin.py."""

from __future__ import annotations

import dataclasses
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class TwinTelemetry(messages.Snapshot):
    """Sim -> UI: latest-wins real-time digital twin & telemetry snapshot."""

    link_status: str = "STANDBY"  # LIVE_HW | LIVE_SIM | REPLAY | MANUAL | STANDBY
    source: str = "none"
    udp_port: int = 19870
    rx_hz: float = 0.0
    bind_error: str = ""

    fsm_mode: str = "IDLE"
    following: bool = False
    visible: bool = False
    m6_fault: bool = False

    q_act_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q_cmd_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dq_act_deg_s: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tau_act: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gripper_deg: float = 285.2

    err_norm_cm: float = 0.0
    track_err_deg: float = 0.0
    tmos_max: float = 25.0

    tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    palm_xyz: tuple[float, float, float] | None = None

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
    """UI -> Sim: visual overlay toggles and optional manual joint pose testing."""

    show_palm: bool = True
    show_cmd_marker: bool = True
    show_workspace_ring: bool = True
    show_camera_ray: bool = True
    manual_mode: bool = False
    manual_q_deg: tuple[float, ...] = (-24.7, 19.7, 21.1, 5.4, 0.1, -0.1)
    manual_gripper_deg: float = 285.2


@dataclasses.dataclass(frozen=True)
class TwinActionEvent(messages.Event):
    """UI -> Sim: discrete replay and pose actions."""

    action: str = ""  # replay_load | replay_play | replay_pause | replay_stop | replay_seek | replay_speed | reset_home | refresh_logs
    log_name: str = ""
    seek_frac: float = 0.0
    speed: float = 1.0
