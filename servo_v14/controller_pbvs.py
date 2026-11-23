"""
层 3：统一世界系 PBVS 控制器（servo_v14/controller_pbvs.py）
==========================================================
纯数学视觉伺服控制模块——零硬件 I/O、零全局变量、零线程锁。

核心几何原理（统一基座世界系 Position-Based Visual Servoing）：
  1. 严格以控制指令角 `q_cmd` 计算当前 TCP 位姿 `(tcp_pos_cmd, tcp_rot_cmd)` 与
     6×6 雅可比 `J_cmd`（绝不混入 `q_act`，杜绝内外环反馈振荡）；
  2. 当前相机世界位姿：`cam_pos_cmd = tcp_pos_cmd + tcp_rot_cmd @ t_tcp_cam`，
     `cam_rot_cmd = tcp_rot_cmd @ R_tcp_cam`；
  3. 给定目标世界位置 `p_world_lead`（来自层 1）与期望距离 `d`（默认 0.40 m），
     在保持固定初始姿态 `R_cam_init` 的前提下，期望相机世界位置为：
         p*_cam_w = p_world_lead - d * z_cam_init
     世界系位置误差即为 `e_w = p*_cam_w - cam_pos_cmd`（等价于相机系误差 `R_cam.T @ e_w`）；
  4. 分方向比例增益（水平 X 偏航用 `gain_x`，竖直/深度用 `gain`）+ 大误差同向非线性追赶
     + 世界系目标速度前馈（DOCKING 区间平滑退出至 0，HOLDING 严格为 0）
     + 伸展度联动调度 + 振荡阻尼安全网 + 换向制动与加速度限幅；
  5. 通过精确刚体杠杆臂伴随变换将相机世界速度转到 TCP，结合姿态误差求解加权阻尼伪逆，
     输出未经安全护盾裁剪的 `ControlCommand` 交由层 4（`SafetyShield`）审定。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from servo_timing import (
    OscillationDamper,
    StaticVelocityHold,
    adaptive_feedforward_scale,
    apply_axis_reversal_damping,
    compute_orientation_cruise_weight,
    compute_reach_schedule,
    compute_weighted_damped_pseudoinverse,
    scaled_alpha,
    scaled_decay,
)
from .config import PBVSConfig, SafetyConfig
from .types import ControlCommand, FSMDecision, FSMMode, TargetEstimate


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """将 3×3 旋转矩阵转换为轴角向量 `axis * angle` (rad)。"""
    trace = float(np.trace(R))
    cos_a = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_a)
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float64)
    if abs(angle - math.pi) < 1e-4:
        diag = np.maximum(0.0, (np.diag(R) + 1.0) * 0.5)
        axis = np.sqrt(diag)
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] = -axis[2]
        n = float(np.linalg.norm(axis))
        return (axis / n) * angle if n > 1e-9 else np.zeros(3, dtype=np.float64)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2.0 * math.sin(angle))
    return axis * angle


def compute_orientation_error_world(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """计算基座世界系下从 `R_cur` 指向 `R_des` 的旋转误差轴角向量 (rad)。"""
    return rotation_matrix_to_axis_angle(R_des @ R_cur.T)


class WorldPBVSController:
    """统一基座世界系 PBVS 控制器（层 3）。"""

    def __init__(
        self,
        pbvs_cfg: PBVSConfig,
        safety_cfg: SafetyConfig,
        T_tcp_camera: np.ndarray,
        ctrl_dt_s: float = 0.01,
    ) -> None:
        """
        初始化统一世界系 PBVS 控制器。

        Args:
            pbvs_cfg: 控制律参数（`PBVSConfig`）。
            safety_cfg: 引用其中基础线速度/角速度与阻尼基准（硬裁剪仍由层 4 独立兜底）。
            T_tcp_camera: 4×4 手眼标定外参矩阵。
            ctrl_dt_s: 标称控制周期 (s)。
        """
        self.cfg = pbvs_cfg
        self.safety_cfg = safety_cfg
        self.ctrl_dt_s = float(ctrl_dt_s)
        self.R_tcp_cam = np.asarray(T_tcp_camera[:3, :3], dtype=np.float64).copy()
        self.t_tcp_cam = np.asarray(T_tcp_camera[:3, 3], dtype=np.float64).copy()

        self._R_tcp_init: Optional[np.ndarray] = None
        self._v_ema_state = np.zeros(3, dtype=np.float64)
        self._v_cam_w_prev = np.zeros(3, dtype=np.float64)
        self._reach_att_smoothed: float = 1.0
        self._static_vel_hold = StaticVelocityHold(
            speed_limit=pbvs_cfg.static_vel_hold_speed,
            confirm_frames=max(1, round(pbvs_cfg.static_vel_hold_delay_s / max(ctrl_dt_s, 1e-4))),
        )
        self._osc_damper = OscillationDamper(
            err_thresh=pbvs_cfg.osc_err_thresh,
            min_half_s=pbvs_cfg.osc_min_half_s,
            max_half_s=pbvs_cfg.osc_max_half_s,
            decay_tau_s=pbvs_cfg.osc_decay_tau_s,
            confirm_reversals=pbvs_cfg.osc_confirm_reversals,
            large_err_release=pbvs_cfg.osc_large_err_release,
            enabled=pbvs_cfg.osc_damp_enable,
        )
        self.last_osc_active: bool = False
        self.last_reach_lead_scale: float = 1.0

    def set_reference_orientation(self, R_tcp_init: np.ndarray) -> None:
        """锁定姿态保持的基准 TCP 旋转矩阵 `R_tcp_init`（启动跟随时记录）。"""
        self._R_tcp_init = np.asarray(R_tcp_init, dtype=np.float64).copy()

    def reset(self, clear_reference_ori: bool = False) -> None:
        """复位内部速度平滑、伸展平滑、振荡阻尼与尾速保持状态。"""
        self._v_ema_state[:] = 0.0
        self._v_cam_w_prev[:] = 0.0
        self._reach_att_smoothed = 1.0
        self._static_vel_hold.reset()
        self._osc_damper.reset()
        self.last_osc_active = False
        self.last_reach_lead_scale = 1.0
        if clear_reference_ori:
            self._R_tcp_init = None

    def compute_world_position_error(
        self,
        p_world_target: np.ndarray,
        tcp_pos_cmd: np.ndarray,
        tcp_rot_cmd: np.ndarray,
    ) -> np.ndarray:
        """
        计算世界系下期望相机位置误差 `e_w = p*_cam_w - p_cam_cmd` (m)。

        关键几何稳定性约束：
            期望相机位置必须沿**固定参考光轴 `z_cam_init = (R_tcp_init @ R_tcp_cam)[:, 2]`**
            退后 `cfg.distance` 米，绝不使用当前瞬时姿态 `cam_rot[:, 2]`！
            否则一旦手腕 J4 触及 87° 限位导致相机仰角偏斜，`-d * z_cam_cur` 会将期望点
            压低、迫使 J2 进一步下压并加剧 J4 仰角，形成正反馈俯冲失稳。
        """
        R_ref = self._R_tcp_init if self._R_tcp_init is not None else tcp_rot_cmd
        cam_rot_ref = R_ref @ self.R_tcp_cam
        cam_pos = tcp_pos_cmd + tcp_rot_cmd @ self.t_tcp_cam
        z_axis_ref = cam_rot_ref[:, 2]
        p_cam_des_w = np.asarray(p_world_target, dtype=np.float64) - self.cfg.distance * z_axis_ref
        return p_cam_des_w - cam_pos

    def step(
        self,
        estimate: TargetEstimate,
        decision: FSMDecision,
        tcp_pos_cmd: np.ndarray,
        tcp_rot_cmd: np.ndarray,
        jacobian_cmd: np.ndarray,
        now_s: float,
        dt_s: float,
    ) -> ControlCommand:
        """
        执行单拍（100Hz）统一世界系 PBVS 速度控制律计算。

        Args:
            estimate: 层 1 输出的统一世界系目标估计。
            decision: 层 2 输出的显式状态机决策。
            tcp_pos_cmd: 基于 `q_cmd` 的 TCP 世界位置 (3,) m。
            tcp_rot_cmd: 基于 `q_cmd` 的 TCP 世界旋转矩阵 (3, 3)。
            jacobian_cmd: 基于 `q_cmd` 的 6×6 空间雅可比矩阵（LOCAL_WORLD_ALIGNED）。
            now_s: 当前单调时间戳 (s)。
            dt_s: 本拍真实时间步长 (s)。

        Returns:
            `ControlCommand`：期望 TCP 速度、关节速度与运动学上下文（交由层 4 护盾审定）。
        """
        dt = max(1e-4, min(0.10, float(dt_s) if math.isfinite(dt_s) else self.ctrl_dt_s))
        tcp_p = np.asarray(tcp_pos_cmd, dtype=np.float64)
        tcp_r = np.asarray(tcp_rot_cmd, dtype=np.float64)
        J = np.asarray(jacobian_cmd, dtype=np.float64)

        if self._R_tcp_init is None:
            self._R_tcp_init = tcp_r.copy()

        reach_r = float(math.hypot(tcp_p[0], tcp_p[1]))
        zero3 = np.zeros(3, dtype=np.float64)
        zero6 = np.zeros(6, dtype=np.float64)

        # ── 非运动模式（IDLE / HOLDING / LOST / FAULT_HOLD / RETURN_HOME）：零速度输出 ──
        if not decision.allow_motion or not estimate.valid:
            # 静止期间让振荡阻尼器平滑衰减而非清零，防止回弹重启盲区
            self._osc_damper.update(zero3, now_s, dt, active=False)
            self._v_ema_state[:] = 0.0
            self._v_cam_w_prev[:] = 0.0
            self._static_vel_hold.reset()
            self.last_osc_active = False
            return ControlCommand(
                v_tcp_world=zero3,
                w_tcp_world=zero3,
                dq_des=zero6,
                tcp_pos_cmd=tcp_p,
                tcp_rot_cmd=tcp_r,
                jacobian_cmd=J,
                reach_radius_m=reach_r,
                ori_weight=1.0,
                osc_active=False,
                docking=decision.docking,
                ff_scale=0.0,
                static_hold_active=(decision.mode == FSMMode.HOLDING),
            )

        cam_rot = tcp_r @ self.R_tcp_cam
        cam_pos = tcp_p + tcp_r @ self.t_tcp_cam

        # ── COASTING 短时丢手滑行模式：指数平滑衰减上拍速度 ──────
        if decision.mode == FSMMode.COASTING:
            decay = scaled_decay(self.cfg.coast_decay_per20ms, dt, nominal_dt=0.02)
            v_cam_w = self._v_cam_w_prev * decay
            speed = float(np.linalg.norm(v_cam_w))
            if speed > self.cfg.coast_max_vel:
                v_cam_w = v_cam_w * (self.cfg.coast_max_vel / speed)
            self._v_cam_w_prev = v_cam_w.copy()
            return self._solve_ik_command(
                v_cam_w=v_cam_w,
                tcp_p=tcp_p,
                tcp_r=tcp_r,
                J=J,
                reach_r=reach_r,
                err_norm=decision.error_norm_m,
                osc_active=False,
                docking=True,
                ff_scale=0.0,
                static_hold_active=False,
            )

        # ── TRACKING / DOCKING：统一世界系 PBVS 位置 + 速度前馈控制 ──
        err_w = self.compute_world_position_error(estimate.p_world_lead, tcp_p, tcp_r)
        err_cam = cam_rot.T @ err_w  # 仅用于分方向增益（水平 X 偏航 vs 竖直/深度 YZ）与分轴振荡检测
        err_norm = float(np.linalg.norm(err_w))

        # 1. 伸展度联动调度（近中区 1.0 满效，深伸展平滑收紧，拉远按 release_tau 快速释放）
        rs = compute_reach_schedule(
            reach_r=reach_r,
            pos_err_norm=err_norm,
            soft_start=self.cfg.reach_soft_start,
            soft_end=self.cfg.reach_soft_end,
            min_scale=self.cfg.reach_min_scale,
            dock_dist=self.cfg.reach_dock_dist,
            ff_min_scale=self.cfg.reach_ff_min_scale,
            lead_min_scale=self.cfg.reach_lead_min_scale,
            accel_min_scale=self.cfg.reach_accel_min_scale,
        )
        target_att = float(rs["vel_att_target"])
        tau_att = (
            self.cfg.reach_release_tau
            if target_att > self._reach_att_smoothed
            else self.cfg.reach_att_tau
        )
        alpha_att = 1.0 - math.exp(-dt / max(tau_att, 1e-3))
        self._reach_att_smoothed += alpha_att * (target_att - self._reach_att_smoothed)
        self.last_reach_lead_scale = float(rs["lead_scale"])

        # 2. 分轴闭环自激振荡阻尼安全网（在相机三轴上检测 1.2~2.5Hz 极限环）
        osc_vec = self._osc_damper.update(err_cam, now_s, dt, active=True)
        osc_active = bool(np.any(osc_vec > 0.05))
        self.last_osc_active = osc_active

        # 3. 位置比例项（分方向 gain_x / gain + 大误差同向非线性追赶 + tanh 软饱和）
        gains_cam = np.array([self.cfg.gain_x, self.cfg.gain, self.cfg.gain], dtype=np.float64)
        # 振荡活跃轴削减比例增益
        gains_cam = gains_cam * (1.0 - self.cfg.osc_gain_cut * osc_vec)

        # >8cm 同向追赶补偿 tanh 压缩（深伸展或振荡时关闭）
        v_w = estimate.v_world
        if err_norm > 0.08 and (not osc_active) and float(rs["ext_ratio"]) < 0.70:
            dot_same = float(np.dot(err_w, v_w))
            if dot_same >= 0.0:
                boost = self.cfg.far_gain_boost * float(np.clip((err_norm - 0.08) / 0.14, 0.0, 1.0))
                gains_cam = gains_cam * (1.0 + boost)

        max_v_eff = self.safety_cfg.max_vel * self._reach_att_smoothed
        v_p_cam = gains_cam * err_cam * self.cfg.servo_direction
        v_p_norm = float(np.linalg.norm(v_p_cam))
        if v_p_norm > 1e-6 and max_v_eff > 1e-6:
            v_p_cam = v_p_cam * (max_v_eff * math.tanh(v_p_norm / max_v_eff) / v_p_norm)
        v_p_world = cam_rot @ v_p_cam

        # 4. 世界系目标速度前馈（与 v13.0 实机验收曲线一致：≤5.5cm 为 0，5.5~11cm 线性切入，避免双重衰减）
        ff_scale = 0.0
        v_ff_world = zero3
        if decision.allow_ff and (not osc_active):
            dock_ramp = float(np.clip((err_norm - 0.055) / 0.055, 0.0, 1.0))
            ff_scale = dock_ramp * float(rs["ff_scale"])
            v_ff_world = v_w * (self.cfg.world_ff_gain * ff_scale)
            ff_norm = float(np.linalg.norm(v_ff_world))
            if ff_norm > self.cfg.ff_max_vel:
                v_ff_world = v_ff_world * (self.cfg.ff_max_vel / ff_norm)

        v_raw_w = v_p_world + v_ff_world
        raw_norm = float(np.linalg.norm(v_raw_w))
        if raw_norm > max_v_eff and raw_norm > 1e-9:
            v_raw_w = v_raw_w * (max_v_eff / raw_norm)

        # 5. 分轴换向残速收缩 + EMA 平滑 + 加速度限幅
        # 注意：apply_axis_reversal_damping 接收 (v_target=期望速度, actual_vel=EMA滤波状态, prev_cmd_vel=上拍指令速度)
        ema_damped, prev_damped, reversing = apply_axis_reversal_damping(
            v_target=v_raw_w,
            actual_vel=self._v_ema_state,
            prev_cmd_vel=self._v_cam_w_prev,
            residual_scale=self.cfg.reversal_residual_scale,
            enabled=True,
        )
        alpha = scaled_alpha(self.cfg.smooth_alpha, dt, nominal_dt=self.ctrl_dt_s)
        v_smooth = ema_damped + alpha * (v_raw_w - ema_damped)
        self._v_ema_state = v_smooth.copy()

        # 加速度限幅（以收缩后的 prev_damped 为基准，换向时启用加速制动倍率）
        dv = v_smooth - prev_damped
        accel_lim = self.cfg.max_accel * float(rs["accel_scale"]) * (
            self.cfg.reversal_accel_boost if reversing else 1.0
        )
        max_dv = accel_lim * dt
        dv_norm = float(np.linalg.norm(dv))
        if dv_norm > max_dv and dv_norm > 1e-9:
            v_cam_w = prev_damped + dv * (max_dv / dv_norm)
        else:
            v_cam_w = v_smooth

        # 6. 静止残余尾速归零保持（传入 6D 速度包以匹配 StaticVelocityHold 契约）
        v6_in = np.concatenate([v_cam_w, zero3])
        v6_out, static_hold_active = self._static_vel_hold.update(
            velocity=v6_in,
            stationary=(not decision.allow_ff),
            in_deadband=(err_norm <= 0.025),
            enabled=self.cfg.static_vel_hold_enable,
        )
        v_cam_w = v6_out[:3]
        self._v_cam_w_prev = v_cam_w.copy()

        return self._solve_ik_command(
            v_cam_w=v_cam_w,
            tcp_p=tcp_p,
            tcp_r=tcp_r,
            J=J,
            reach_r=reach_r,
            err_norm=err_norm,
            osc_active=osc_active,
            docking=decision.docking,
            ff_scale=ff_scale,
            static_hold_active=static_hold_active,
        )

    def _solve_ik_command(
        self,
        *,
        v_cam_w: np.ndarray,
        tcp_p: np.ndarray,
        tcp_r: np.ndarray,
        J: np.ndarray,
        reach_r: float,
        err_norm: float,
        osc_active: bool,
        docking: bool,
        ff_scale: float,
        static_hold_active: bool,
    ) -> ControlCommand:
        """将世界系相机线速度与固定姿态角速度经伴随变换与加权 DLS 映射为关节速度。"""
        # 姿态保持角速度（世界系）
        ori_err_w = compute_orientation_error_world(tcp_r, self._R_tcp_init)
        ori_err_norm = float(np.linalg.norm(ori_err_w))
        ori_deadband_rad = math.radians(self.cfg.orientation_deadband_deg)
        if ori_err_norm <= ori_deadband_rad:
            w_tcp_w = np.zeros(3, dtype=np.float64)
        else:
            w_tcp_w = self.cfg.k_orientation * ori_err_w
            w_norm = float(np.linalg.norm(w_tcp_w))
            if w_norm > self.safety_cfg.max_vel_angular:
                w_tcp_w = w_tcp_w * (self.safety_cfg.max_vel_angular / w_norm)

        # 精确刚体杠杆臂伴随变换：v_cam_w = v_tcp_w + w_tcp_w × r_tcp_to_cam_w
        # ⇒ v_tcp_w = v_cam_w - w_tcp_w × (tcp_r @ t_tcp_cam)
        r_lever_w = tcp_r @ self.t_tcp_cam
        v_tcp_w = v_cam_w - np.cross(w_tcp_w, r_lever_w)

        # 加权阻尼最小二乘（大误差巡航适度放松姿态权重至 0.80，≤10° 软护栏自动收回 1.0）
        ori_weight = compute_orientation_cruise_weight(
            pos_err_norm=err_norm,
            ori_err_norm=ori_err_norm,
            min_weight=self.cfg.ori_cruise_weight_min,
            ori_soft_max_rad=math.radians(self.cfg.ori_soft_max_deg),
            active=(not docking and not osc_active),
        )
        row_weights = np.array([1.0, 1.0, 1.0, ori_weight, ori_weight, ori_weight], dtype=np.float64)
        twist_w = np.concatenate([v_tcp_w, w_tcp_w])
        J_pinv = compute_weighted_damped_pseudoinverse(
            J, row_weights, damping=self.safety_cfg.damping_base
        )
        dq_des = J_pinv @ twist_w

        return ControlCommand(
            v_tcp_world=v_tcp_w,
            w_tcp_world=w_tcp_w,
            dq_des=dq_des,
            tcp_pos_cmd=tcp_p,
            tcp_rot_cmd=tcp_r,
            jacobian_cmd=J,
            reach_radius_m=reach_r,
            ori_weight=float(ori_weight),
            osc_active=osc_active,
            docking=docking,
            ff_scale=float(ff_scale),
            static_hold_active=static_hold_active,
        )
