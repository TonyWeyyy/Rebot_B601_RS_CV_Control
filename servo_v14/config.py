"""
v14.0 不可变配置数据类与硬安全红线校验（servo_v14/config.py）
=============================================================
彻底取代旧版模块级 `global` 变量体系，所有参数通过不可变 `@dataclass(frozen=True)`
传递，并在加载时一次性完成 NaN/Inf 检查与硬件安全包络校验。

强制锁死的硬件安全红线（与 AGENTS.md 严格一致）：
    1. workspace_max_radius <= 0.42 m（70% 臂展，防 J2 堵转）
    2. max_vel <= 0.85 m/s
    3. max_vel_joints 逐轴 <= HARD_MAX_JOINT_SPEED = [2.5, 2.0, 2.0, 3.0, 3.0, 2.5]
       （承重抗重力轴 J2/J3 严守 <= 2.0 rad/s）
    4. distance in [0.18, 0.60] m
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence, Tuple

import yaml

# 与 rebot_adapter.py 保持完全一致的硬件关节速度绝对红线 (rad/s)
HARD_MAX_JOINT_SPEED: Tuple[float, ...] = (2.5, 2.0, 2.0, 3.0, 3.0, 2.5)


def _finite_float(val: Any, name: str) -> float:
    """将配置项转为有限浮点数，若为 NaN/Inf/非数值则抛出 ValueError。"""
    try:
        out = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 {name} 不是有效数值: {val!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"配置项 {name} 不能为 NaN/Inf: {val!r}")
    return out


def _float_tuple(vals: Sequence[Any], length: int, name: str) -> Tuple[float, ...]:
    """将序列转为定长有限浮点元组。"""
    if not isinstance(vals, (list, tuple)) or len(vals) != length:
        raise ValueError(f"配置项 {name} 必须是长度为 {length} 的列表")
    return tuple(_finite_float(v, f"{name}[{i}]") for i, v in enumerate(vals))


def _check_range(val: float, lo: float, hi: float, name: str) -> None:
    """校验有限标量是否落在 [lo, hi] 闭区间。"""
    if not (math.isfinite(val) and lo <= val <= hi):
        raise ValueError(f"配置项 {name}={val} 超出允许范围 [{lo}, {hi}]")


@dataclass(frozen=True)
class EstimatorConfig:
    """层 1：统一世界系目标估计器配置。"""

    pose_delay_s: float = 0.025
    pose_lpf_tau_s: float = 0.012
    kf_q: float = 1.80
    kf_r_xy: float = 0.0023
    kf_r_z: float = 0.0038
    kf_resync_frames: int = 3
    kf_static_q_scale: float = 0.65
    axis_adapt_enable: bool = True
    axis_adapt_ref_xy: float = 0.032
    axis_adapt_ref_z: float = 0.080
    axis_adapt_max_xy: float = 1.75
    axis_adapt_max_z: float = 1.00
    axis_adapt_confirm: int = 2
    axis_adapt_attack_tau: float = 0.030
    axis_adapt_release_tau: float = 0.100
    target_stale_s: float = 0.12
    lead_base_ms: float = 8.0
    lead_max_ms: float = 15.0
    lead_near_dist: float = 0.025
    lead_full_dist: float = 0.046
    lead_dir_full_cos: float = 0.45
    lead_attack_tau: float = 0.028
    lead_release_tau: float = 0.012


@dataclass(frozen=True)
class FSMConfig:
    """层 2：显式有限状态机配置。"""

    deadband_in_m: float = 0.020
    deadband_hysteresis: float = 1.70
    vel_deadband_off: float = 0.060
    vel_deadband_on: float = 0.130
    exit_confirm_s: float = 0.050
    stop_confirm_s: float = 0.080
    docking_dist_m: float = 0.080
    fast_start_enable: bool = True
    fast_start_frames: int = 2
    fast_start_dist: float = 0.028
    fast_start_dir_cos: float = 0.80
    post_brake_settle_s: float = 0.16
    post_brake_ext_extra_s: float = 0.06
    coast_window_s: float = 0.12
    lost_hold_s: float = 0.30


@dataclass(frozen=True)
class PBVSConfig:
    """层 3：统一世界系 PBVS 控制器配置。"""

    distance: float = 0.40
    servo_direction: float = 1.0
    gain: float = 2.75
    gain_x: float = 2.20
    far_gain_boost: float = 0.40
    world_ff_gain: float = 0.62
    ff_max_vel: float = 0.95
    ff_taper_dist: float = 0.080
    ff_taper_max_dist: float = 0.130
    ff_taper_speed_ref: float = 0.35
    smooth_alpha: float = 0.50
    max_accel: float = 4.25
    reversal_accel_boost: float = 2.00
    reversal_residual_scale: float = 0.18
    coast_decay_per20ms: float = 0.90
    coast_max_vel: float = 0.35
    static_vel_hold_enable: bool = True
    static_vel_hold_speed: float = 0.060
    static_vel_hold_delay_s: float = 0.030
    reach_soft_start: float = 0.24
    reach_soft_end: float = 0.38
    reach_min_scale: float = 0.70
    reach_dock_dist: float = 0.25
    reach_ff_min_scale: float = 0.65
    reach_lead_min_scale: float = 0.30
    reach_accel_min_scale: float = 0.75
    reach_att_tau: float = 0.15
    reach_release_tau: float = 0.04
    k_orientation: float = 1.60
    orientation_deadband_deg: float = 2.0
    ori_cruise_weight_min: float = 0.80
    ori_soft_max_deg: float = 10.0
    osc_damp_enable: bool = True
    osc_err_thresh: float = 0.018
    osc_min_half_s: float = 0.14
    osc_max_half_s: float = 0.42
    osc_gain_cut: float = 0.30
    osc_decay_tau_s: float = 0.55
    osc_confirm_reversals: int = 3
    osc_large_err_release: float = 0.15


@dataclass(frozen=True)
class SafetyConfig:
    """层 4：独立安全护盾配置（受硬件硬包络二次校验）。"""

    max_vel: float = 0.85
    max_vel_angular: float = 2.00
    max_vel_joint_dock: float = 2.00
    max_vel_joints: Tuple[float, ...] = (2.50, 2.00, 2.00, 3.00, 3.00, 2.40)
    joint_vel_grouped_cruise: bool = True
    joint_limit_cushion_rad: float = 0.08
    damping_base: float = 0.010
    damping_adaptive: bool = True
    manipulability_threshold: float = 0.004
    min_manip_speed_scale: float = 0.35
    hard_wall_margin_m: float = 0.05
    workspace_min: Tuple[float, ...] = (-0.40, -0.10, 0.03)
    workspace_max: Tuple[float, ...] = (0.40, 0.40, 0.60)
    workspace_max_radius: float = 0.42


@dataclass(frozen=True)
class DepthGuardConfig:
    """4 层深度保护链配置。"""

    min_valid_m: float = 0.15
    max_valid_m: float = 1.50
    max_jump_m: float = 0.08
    max_rate_m: float = 0.05
    reacq_frames: int = 5
    reacq_tol_m: float = 0.06
    reacq_bg_margin_m: float = 0.25
    anti_lock_frames: int = 15
    z_freeze_max_s: float = 0.20
    oe_min_cutoff: float = 0.85
    oe_beta: float = 0.45


@dataclass(frozen=True)
class CameraConfig:
    """相机与检测混合跟踪配置。"""

    fps: int = 60
    color_exposure_ms: float = 6.0
    hw_timestamp_enable: bool = False
    hw_max_transport_s: float = 0.20
    roi_crop_enable: bool = True
    roi_crop_scale: float = 0.68
    hybrid_track_enable: bool = True
    hybrid_max_frames: int = 15
    hybrid_box_half: int = 52
    hybrid_max_step_px: int = 115
    diag_enable: bool = True
    diag_queue_size: int = 8192


@dataclass(frozen=True)
class SimConfig:
    """离线数字孪生 200Hz MIT 辨识动力学参数配置。"""

    mit_hz: float = 200.0
    kp: Tuple[float, ...] = (72.0, 216.0, 216.0, 50.0, 50.0, 50.0)
    kd_near: Tuple[float, ...] = (4.6, 13.5, 14.0, 5.0, 4.0, 4.0)
    kd_far: Tuple[float, ...] = (6.4, 14.5, 15.0, 5.0, 4.0, 4.0)
    j_eff: Tuple[float, ...] = (0.166, 0.341, 0.498, 0.094, 0.020, 0.048)
    j1_ext_coeff: float = 0.85
    vel_ff_scale: Tuple[float, ...] = (0.70, 0.85, 0.85, 0.90, 0.90, 0.90)
    gravity_ff_scale: Tuple[float, ...] = (0.0, 0.65, 0.80, 0.80, 0.0, 0.0)
    viscous_friction: Tuple[float, ...] = (0.45, 1.40, 1.60, 0.35, 0.20, 0.20)
    transport_delay_s: float = 0.005
    dq_noise_std: float = 0.045
    camera_delay_s: float = 0.025
    tau_limit_nm: Tuple[float, ...] = (18.0, 28.0, 28.0, 10.0, 10.0, 10.0)
    track_err_limit_deg: float = 12.0


@dataclass(frozen=True)
class V14Config:
    """v14 完整不可变根配置。"""

    ctrl_rate_hz: float = 100.0
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    pbvs: PBVSConfig = field(default_factory=PBVSConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    depth: DepthGuardConfig = field(default_factory=DepthGuardConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @property
    def ctrl_dt_s(self) -> float:
        """返回标称控制周期 (s)。"""
        return 1.0 / self.ctrl_rate_hz


def validate_v14_config(cfg: V14Config) -> V14Config:
    """
    校验 `V14Config` 全部字段的数值合法性与硬件安全红线。

    Args:
        cfg: 待校验的 v14 根配置对象。

    Returns:
        校验通过的原配置对象。

    Raises:
        ValueError: 任何字段越界、非法或触犯硬件安全红线时抛出。
    """
    _check_range(cfg.ctrl_rate_hz, 20.0, 250.0, "ctrl_rate_hz")

    # 1. 估计器校验
    e = cfg.estimator
    _check_range(e.pose_delay_s, 0.0, 0.25, "estimator.pose_delay_s")
    _check_range(e.pose_lpf_tau_s, 0.0, 0.20, "estimator.pose_lpf_tau_s")
    _check_range(e.kf_q, 0.01, 50.0, "estimator.kf_q")
    _check_range(e.kf_r_xy, 1e-5, 0.5, "estimator.kf_r_xy")
    _check_range(e.kf_r_z, 1e-5, 0.5, "estimator.kf_r_z")
    _check_range(e.kf_static_q_scale, 0.10, 1.0, "estimator.kf_static_q_scale")
    _check_range(e.axis_adapt_max_xy, 1.0, 2.0, "estimator.axis_adapt_max_xy")
    _check_range(e.axis_adapt_max_z, 1.0, 1.5, "estimator.axis_adapt_max_z")
    if e.axis_adapt_confirm < 1:
        raise ValueError("estimator.axis_adapt_confirm 必须 >= 1")
    _check_range(e.lead_base_ms, 0.0, 40.0, "estimator.lead_base_ms")
    _check_range(e.lead_max_ms, e.lead_base_ms, 60.0, "estimator.lead_max_ms")
    _check_range(e.target_stale_s, 0.02, 1.0, "estimator.target_stale_s")

    # 2. FSM 校验
    f = cfg.fsm
    _check_range(f.deadband_in_m, 0.005, 0.10, "fsm.deadband_in_m")
    _check_range(f.deadband_hysteresis, 1.0, 3.0, "fsm.deadband_hysteresis")
    _check_range(f.vel_deadband_off, 0.005, 0.30, "fsm.vel_deadband_off")
    _check_range(f.vel_deadband_on, f.vel_deadband_off, 0.50, "fsm.vel_deadband_on")
    _check_range(f.exit_confirm_s, 0.0, 0.30, "fsm.exit_confirm_s")
    _check_range(f.stop_confirm_s, 0.0, 0.25, "fsm.stop_confirm_s")
    _check_range(f.docking_dist_m, f.deadband_in_m, 0.30, "fsm.docking_dist_m")
    _check_range(f.post_brake_settle_s, 0.0, 0.50, "fsm.post_brake_settle_s")
    _check_range(f.post_brake_ext_extra_s, 0.0, 0.30, "fsm.post_brake_ext_extra_s")
    _check_range(f.coast_window_s, 0.0, 0.50, "fsm.coast_window_s")
    _check_range(f.lost_hold_s, f.coast_window_s, 2.0, "fsm.lost_hold_s")

    # 3. PBVS 控制器校验
    p = cfg.pbvs
    _check_range(p.distance, 0.18, 0.60, "pbvs.distance")
    if p.servo_direction not in (1.0, -1.0):
        raise ValueError("pbvs.servo_direction 必须为 1.0 或 -1.0")
    _check_range(p.gain, 0.1, 6.0, "pbvs.gain")
    _check_range(p.gain_x, 0.1, 6.0, "pbvs.gain_x")
    _check_range(p.far_gain_boost, 0.0, 0.70, "pbvs.far_gain_boost")
    _check_range(p.world_ff_gain, 0.0, 0.90, "pbvs.world_ff_gain")
    _check_range(p.ff_max_vel, 0.0, 1.5, "pbvs.ff_max_vel")
    _check_range(p.ff_taper_dist, 0.01, 0.30, "pbvs.ff_taper_dist")
    _check_range(p.ff_taper_max_dist, p.ff_taper_dist, 0.40, "pbvs.ff_taper_max_dist")
    _check_range(p.smooth_alpha, 0.05, 1.0, "pbvs.smooth_alpha")
    _check_range(p.max_accel, 0.5, 8.0, "pbvs.max_accel")
    _check_range(p.reversal_accel_boost, 1.0, 4.0, "pbvs.reversal_accel_boost")
    _check_range(p.reversal_residual_scale, 0.0, 1.0, "pbvs.reversal_residual_scale")
    _check_range(p.reach_soft_start, 0.05, 0.40, "pbvs.reach_soft_start")
    _check_range(p.reach_soft_end, p.reach_soft_start + 0.02, 0.42, "pbvs.reach_soft_end")
    _check_range(p.reach_min_scale, 0.25, 1.0, "pbvs.reach_min_scale")
    _check_range(p.reach_ff_min_scale, 0.20, 1.0, "pbvs.reach_ff_min_scale")
    _check_range(p.reach_lead_min_scale, 0.10, 1.0, "pbvs.reach_lead_min_scale")
    _check_range(p.reach_accel_min_scale, 0.25, 1.0, "pbvs.reach_accel_min_scale")
    _check_range(p.reach_att_tau, 0.01, 1.0, "pbvs.reach_att_tau")
    _check_range(p.reach_release_tau, 0.01, 1.0, "pbvs.reach_release_tau")
    _check_range(p.ori_cruise_weight_min, 0.30, 1.0, "pbvs.ori_cruise_weight_min")
    _check_range(p.ori_soft_max_deg, 2.0, 20.0, "pbvs.ori_soft_max_deg")
    _check_range(p.osc_gain_cut, 0.0, 0.75, "pbvs.osc_gain_cut")
    if p.osc_min_half_s >= p.osc_max_half_s:
        raise ValueError("pbvs.osc_min_half_s 必须小于 osc_max_half_s")

    # 4. 安全护盾红线校验（严防越界）
    s = cfg.safety
    _check_range(s.max_vel, 0.05, 0.85, "safety.max_vel")
    _check_range(s.max_vel_angular, 0.1, 3.0, "safety.max_vel_angular")
    _check_range(s.max_vel_joint_dock, 0.2, 2.0, "safety.max_vel_joint_dock")
    if len(s.max_vel_joints) != 6:
        raise ValueError("safety.max_vel_joints 必须为 6 维向量")
    for idx, (val, hard) in enumerate(zip(s.max_vel_joints, HARD_MAX_JOINT_SPEED)):
        if not (math.isfinite(val) and 0.2 <= val <= hard + 1e-9):
            raise ValueError(
                f"safety.max_vel_joints[J{idx + 1}]={val} 超出硬件安全红线 [0.2, {hard}]"
            )
    _check_range(s.joint_limit_cushion_rad, 0.02, 0.30, "safety.joint_limit_cushion_rad")
    _check_range(s.workspace_max_radius, 0.15, 0.42, "safety.workspace_max_radius")
    if len(s.workspace_min) != 3 or len(s.workspace_max) != 3:
        raise ValueError("workspace_min / workspace_max 必须为 3 维坐标")
    for axis in range(3):
        if s.workspace_min[axis] >= s.workspace_max[axis]:
            raise ValueError(f"workspace_min[{axis}] 必须小于 workspace_max[{axis}]")

    # 5. 深度保护链校验
    d = cfg.depth
    _check_range(d.min_valid_m, 0.05, 0.40, "depth.min_valid_m")
    _check_range(d.max_valid_m, d.min_valid_m + 0.20, 3.0, "depth.max_valid_m")
    _check_range(d.max_jump_m, 0.02, 0.30, "depth.max_jump_m")
    _check_range(d.max_rate_m, 0.01, 0.20, "depth.max_rate_m")
    if d.reacq_frames < 2:
        raise ValueError("depth.reacq_frames 必须 >= 2")

    # 6. 相机校验
    c = cfg.camera
    if c.fps not in (15, 30, 60):
        raise ValueError(f"camera.fps={c.fps} 不在支持列表 (15, 30, 60)")
    _check_range(c.roi_crop_scale, 0.40, 0.90, "camera.roi_crop_scale")

    # 7. 数字孪生校验
    sim = cfg.sim
    _check_range(sim.mit_hz, 50.0, 1000.0, "sim.mit_hz")
    for field_name in ("kp", "kd_near", "kd_far", "j_eff", "vel_ff_scale",
                       "gravity_ff_scale", "viscous_friction", "tau_limit_nm"):
        vec = getattr(sim, field_name)
        if len(vec) != 6 or any(not math.isfinite(x) for x in vec):
            raise ValueError(f"sim.{field_name} 必须是 6 维有限向量")
    if any(x <= 0.0 for x in sim.j_eff):
        raise ValueError("sim.j_eff 各轴转动惯量必须 > 0")

    return cfg


def _merge_dataclass(cls, raw: Mapping[str, Any]):
    """按目标 dataclass 字段名从 YAML dict 安全构造不可变实例。"""
    default_obj = cls()
    if not raw:
        return default_obj
    kwargs = {}
    for field_name in default_obj.__dataclass_fields__:
        if field_name not in raw:
            continue
        default_val = getattr(default_obj, field_name)
        val = raw[field_name]
        if isinstance(default_val, bool):
            kwargs[field_name] = bool(val)
        elif isinstance(default_val, int):
            kwargs[field_name] = int(val)
        elif isinstance(default_val, float):
            kwargs[field_name] = _finite_float(val, f"{cls.__name__}.{field_name}")
        elif isinstance(default_val, tuple):
            kwargs[field_name] = _float_tuple(
                val, len(default_val), f"{cls.__name__}.{field_name}"
            )
        else:
            kwargs[field_name] = val
    return replace(default_obj, **kwargs)


def load_v14_config(path: str) -> V14Config:
    """
    从 `config/v14.yaml`（或兼容的 `config/follow.yaml`）加载并校验 `V14Config`。

    Args:
        path: YAML 配置文件路径。

    Returns:
        通过硬件安全红线校验的不可变 `V14Config`。
    """
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if "v14" in raw:
        root = raw["v14"] or {}
        cfg = V14Config(
            ctrl_rate_hz=_finite_float(root.get("ctrl_rate_hz", 100.0), "ctrl_rate_hz"),
            estimator=_merge_dataclass(EstimatorConfig, root.get("estimator") or {}),
            fsm=_merge_dataclass(FSMConfig, root.get("fsm") or {}),
            pbvs=_merge_dataclass(PBVSConfig, root.get("pbvs") or {}),
            safety=_merge_dataclass(SafetyConfig, root.get("safety") or {}),
            depth=_merge_dataclass(DepthGuardConfig, root.get("depth") or {}),
            camera=_merge_dataclass(CameraConfig, root.get("camera") or {}),
            sim=_merge_dataclass(SimConfig, root.get("sim") or {}),
        )
        return validate_v14_config(cfg)

    if "follow" in raw:
        return v14_config_from_follow_dict(raw)

    raise ValueError(f"无法识别的配置文件格式（缺少 v14 或 follow 根节点）: {path}")


def v14_config_from_follow_dict(raw: Mapping[str, Any]) -> V14Config:
    """
    兼容适配器：将旧版 `config/follow.yaml` 字典映射为 `V14Config` 并执行红线校验。

    便于直接用历史配置（如 `follow_v13.yaml`）对比测试 v14 架构。
    """
    fl = raw.get("follow") or {}
    ws = raw.get("workspace") or {}

    ctrl_hz = 100.0
    exit_frames = int(fl.get("kf_ff_exit_frames", 5))

    est = replace(
        EstimatorConfig(),
        pose_delay_s=_finite_float(fl.get("world_kf_pose_delay_s", 0.025), "world_kf_pose_delay_s"),
        pose_lpf_tau_s=_finite_float(fl.get("world_kf_pose_lpf_tau_s", 0.012), "world_kf_pose_lpf_tau_s"),
        kf_q=_finite_float(fl.get("kf_q", 1.80), "kf_q"),
        kf_r_xy=_finite_float(fl.get("kf_r_xy", fl.get("kf_r", 0.0023)), "kf_r_xy"),
        kf_r_z=_finite_float(fl.get("kf_r_z", fl.get("kf_r", 0.0038)), "kf_r_z"),
        kf_resync_frames=int(fl.get("kf_resync_frames", 3)),
        kf_static_q_scale=_finite_float(fl.get("kf_static_q_scale", 0.65), "kf_static_q_scale"),
        axis_adapt_enable=bool(fl.get("kf_axis_adapt_enable", True)),
        axis_adapt_ref_xy=_finite_float(fl.get("kf_axis_adapt_ref_xy", 0.032), "kf_axis_adapt_ref_xy"),
        axis_adapt_ref_z=_finite_float(fl.get("kf_axis_adapt_ref_z", 0.080), "kf_axis_adapt_ref_z"),
        axis_adapt_max_xy=_finite_float(fl.get("kf_axis_adapt_max_xy", 1.75), "kf_axis_adapt_max_xy"),
        axis_adapt_max_z=_finite_float(fl.get("kf_axis_adapt_max_z", 1.00), "kf_axis_adapt_max_z"),
        axis_adapt_confirm=int(fl.get("kf_axis_adapt_confirm", 2)),
        target_stale_s=_finite_float(fl.get("target_stale_s", 0.12), "target_stale_s"),
        lead_base_ms=_finite_float(fl.get("kf_lead_ms", 8.0), "kf_lead_ms"),
        lead_max_ms=_finite_float(fl.get("kf_lead_max_ms", 15.0), "kf_lead_max_ms"),
    )
    fsm = replace(
        FSMConfig(),
        deadband_in_m=_finite_float(fl.get("position_deadband", 0.020), "position_deadband"),
        deadband_hysteresis=_finite_float(fl.get("deadband_hysteresis", 1.70), "deadband_hysteresis"),
        vel_deadband_off=_finite_float(fl.get("kf_ff_deadband_off", 0.060), "kf_ff_deadband_off"),
        vel_deadband_on=_finite_float(fl.get("kf_ff_deadband_on", 0.130), "kf_ff_deadband_on"),
        exit_confirm_s=max(0.0, exit_frames / ctrl_hz),
        stop_confirm_s=_finite_float(fl.get("kf_ff_stop_delay_s", 0.080), "kf_ff_stop_delay_s"),
        docking_dist_m=_finite_float(fl.get("ff_taper_dist", 0.080), "ff_taper_dist"),
        fast_start_enable=bool(fl.get("kf_fast_start_enable", True)),
        fast_start_frames=int(fl.get("kf_fast_start_frames", 2)),
        fast_start_dist=_finite_float(fl.get("kf_fast_start_dist", 0.028), "kf_fast_start_dist"),
        fast_start_dir_cos=_finite_float(fl.get("kf_fast_start_dir_cos", 0.80), "kf_fast_start_dir_cos"),
        post_brake_settle_s=_finite_float(fl.get("post_brake_settle_s", 0.16), "post_brake_settle_s"),
        post_brake_ext_extra_s=_finite_float(fl.get("post_brake_ext_extra_s", 0.06), "post_brake_ext_extra_s"),
        coast_window_s=_finite_float(fl.get("coast_window_s", 0.12), "coast_window_s"),
        lost_hold_s=_finite_float(fl.get("lost_hold_s", 0.30), "lost_hold_s"),
    )
    pbvs = replace(
        PBVSConfig(),
        distance=_finite_float(fl.get("distance", 0.40), "distance"),
        servo_direction=_finite_float(fl.get("servo_direction", 1.0), "servo_direction"),
        gain=_finite_float(fl.get("gain", 2.75), "gain"),
        gain_x=_finite_float(fl.get("gain_x", 2.20), "gain_x"),
        far_gain_boost=_finite_float(fl.get("far_gain_boost", 0.40), "far_gain_boost"),
        world_ff_gain=_finite_float(fl.get("world_kf_ff_gain", 0.62), "world_kf_ff_gain"),
        ff_max_vel=_finite_float(fl.get("ff_max_vel", 0.95), "ff_max_vel"),
        ff_taper_dist=_finite_float(fl.get("ff_taper_dist", 0.080), "ff_taper_dist"),
        ff_taper_max_dist=_finite_float(fl.get("ff_taper_max_dist", 0.130), "ff_taper_max_dist"),
        smooth_alpha=_finite_float(fl.get("smooth_alpha", 0.50), "smooth_alpha"),
        max_accel=_finite_float(fl.get("max_accel", 4.25), "max_accel"),
        reversal_accel_boost=_finite_float(fl.get("reversal_accel_boost", 2.00), "reversal_accel_boost"),
        reversal_residual_scale=_finite_float(fl.get("reversal_residual_scale", 0.18), "reversal_residual_scale"),
        reach_soft_start=_finite_float(fl.get("reach_soft_start", 0.24), "reach_soft_start"),
        reach_soft_end=_finite_float(fl.get("reach_soft_end", 0.38), "reach_soft_end"),
        reach_min_scale=_finite_float(fl.get("reach_min_scale", 0.70), "reach_min_scale"),
        reach_ff_min_scale=_finite_float(fl.get("reach_ff_min_scale", 0.65), "reach_ff_min_scale"),
        reach_lead_min_scale=_finite_float(fl.get("reach_lead_min_scale", 0.30), "reach_lead_min_scale"),
        reach_accel_min_scale=_finite_float(fl.get("reach_accel_min_scale", 0.75), "reach_accel_min_scale"),
        k_orientation=_finite_float(fl.get("k_orientation", 1.60), "k_orientation"),
        ori_cruise_weight_min=_finite_float(fl.get("ori_cruise_weight_min", 0.80), "ori_cruise_weight_min"),
        ori_soft_max_deg=_finite_float(fl.get("ori_soft_max_deg", 10.0), "ori_soft_max_deg"),
        osc_damp_enable=bool(fl.get("osc_damp_enable", True)),
        osc_gain_cut=_finite_float(fl.get("osc_gain_cut", 0.30), "osc_gain_cut"),
    )
    mvj = fl.get("max_vel_joints")
    if mvj is None:
        mv_scalar = _finite_float(fl.get("max_vel_joint", 2.0), "max_vel_joint")
        mvj_tuple = tuple(mv_scalar for _ in range(6))
    else:
        mvj_tuple = _float_tuple(mvj, 6, "max_vel_joints")
    safety = replace(
        SafetyConfig(),
        max_vel=_finite_float(fl.get("max_vel", 0.85), "max_vel"),
        max_vel_angular=_finite_float(fl.get("max_vel_angular", 2.00), "max_vel_angular"),
        max_vel_joint_dock=_finite_float(fl.get("max_vel_joint", 2.00), "max_vel_joint"),
        max_vel_joints=mvj_tuple,
        joint_vel_grouped_cruise=bool(fl.get("joint_vel_grouped_cruise", True)),
        workspace_min=_float_tuple(ws.get("min", [-0.40, -0.10, 0.03]), 3, "workspace.min"),
        workspace_max=_float_tuple(ws.get("max", [0.40, 0.40, 0.60]), 3, "workspace.max"),
        workspace_max_radius=_finite_float(ws.get("max_radius", 0.42), "workspace.max_radius"),
    )
    depth = replace(
        DepthGuardConfig(),
        max_jump_m=_finite_float(fl.get("z3d_max_jump", 0.08), "z3d_max_jump"),
        max_rate_m=_finite_float(fl.get("z3d_max_rate", 0.05), "z3d_max_rate"),
        z_freeze_max_s=_finite_float(fl.get("z_freeze_max", 0.20), "z_freeze_max"),
        oe_min_cutoff=_finite_float(fl.get("depth_oe_min_cutoff", 0.85), "depth_oe_min_cutoff"),
        oe_beta=_finite_float(fl.get("depth_oe_beta", 0.45), "depth_oe_beta"),
    )
    cam = replace(
        CameraConfig(),
        fps=int(fl.get("fps", 60)),
        color_exposure_ms=_finite_float(fl.get("color_exposure_ms", 6.0), "color_exposure_ms"),
        roi_crop_enable=bool(fl.get("roi_crop_enable", True)),
        roi_crop_scale=_finite_float(fl.get("roi_crop_scale", 0.68), "roi_crop_scale"),
    )
    return validate_v14_config(V14Config(
        ctrl_rate_hz=ctrl_hz,
        estimator=est,
        fsm=fsm,
        pbvs=pbvs,
        safety=safety,
        depth=depth,
        camera=cam,
    ))
