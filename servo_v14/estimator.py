"""
层 1：统一世界系目标估计器（servo_v14/estimator.py）
==================================================
v14 架构的核心变革之一：**基座世界系作为唯一真相源（Single Source of Truth）**。

相比 v12/v13 并行运行“相机系 KF + DelayedTwistCompensator + 世界系 KF + 双共识投票”的
历史过渡架构，v14 直接利用 200Hz 真实物理关节反馈 `q_act`（滞后已由 P1~P3 压至 20~30ms，
下垂降 85%+）：
  1. 每拍将真实 TCP 位姿与手眼外参存入 `CameraPoseHistory`（SO(3) 测地插值）；
  2. 收到新帧观测 `obs.p_cam` 时，按纯相机曝光+传输延迟 `pose_delay_s=25ms` 查表得历史相机位姿，
     一次反投影至基座世界系，从几何根源彻底消除 eye-in-hand 自运动干扰；
  3. 单一 6 维世界系恒速卡尔曼滤波（`KalmanTarget3D`）维护目标真实位置与速度；
  4. 深度冻结期（`obs.frozen=True`）严格只做相机光轴正交 XY 平面子空间更新（`update_partial`），
     绝不让冻结占位深度污染世界系深度方向估计；
  5. 输出带观测年龄外推与自适应前瞻的 `TargetEstimate` 给层 2（FSM）和层 3（PBVS）。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from kalman_target import KalmanTarget3D
from servo_timing import (
    AdaptiveLeadGate,
    AxisProcessNoiseGate,
    CameraPoseHistory,
    dynamic_lead_seconds,
    measurement_age,
)
from .config import EstimatorConfig
from .types import Observation, RobotState, TargetEstimate


class WorldTargetEstimator:
    """单一世界系目标位置/速度估计器（层 1）。"""

    def __init__(
        self,
        cfg: EstimatorConfig,
        T_tcp_camera: np.ndarray,
        ctrl_dt_s: float = 0.01,
        vel_deadband_off: float = 0.06,
        vel_deadband_on: float = 0.13,
    ) -> None:
        """
        初始化统一世界系估计器。

        Args:
            cfg: 估计器配置（`EstimatorConfig`）。
            T_tcp_camera: 4×4 手眼标定外参矩阵（相机系到 TCP 系）。
            ctrl_dt_s: 上层标称控制周期 (s)，默认 0.01s (100Hz)。
            vel_deadband_off: 速度静止下沿 (m/s)，用于 `AdaptiveLeadGate`。
            vel_deadband_on: 速度运动上沿 (m/s)，用于 `AdaptiveLeadGate`。
        """
        self.cfg = cfg
        self.ctrl_dt_s = float(ctrl_dt_s)
        self.T_tcp_camera = np.asarray(T_tcp_camera, dtype=np.float64).copy()
        self.R_tcp_cam = self.T_tcp_camera[:3, :3].copy()
        self.t_tcp_cam = self.T_tcp_camera[:3, 3].copy()

        self._kf = KalmanTarget3D(
            q=cfg.kf_q,
            r_xy=cfg.kf_r_xy,
            r_z=cfg.kf_r_z,
        )
        self._pose_history = CameraPoseHistory(
            lpf_tau_s=cfg.pose_lpf_tau_s,
            max_history_s=0.60,
        )
        self._axis_q_gate = AxisProcessNoiseGate(
            ref_xy=cfg.axis_adapt_ref_xy,
            ref_z=cfg.axis_adapt_ref_z,
            max_xy=cfg.axis_adapt_max_xy,
            max_z=cfg.axis_adapt_max_z,
            confirm_frames=cfg.axis_adapt_confirm,
            attack_tau=cfg.axis_adapt_attack_tau,
            release_tau=cfg.axis_adapt_release_tau,
        )
        self._lead_gate = AdaptiveLeadGate(
            speed_off=vel_deadband_off,
            speed_on=vel_deadband_on,
            near_dist=cfg.lead_near_dist,
            full_dist=cfg.lead_full_dist,
            direction_full_cos=cfg.lead_dir_full_cos,
            attack_tau=cfg.lead_attack_tau,
            release_tau=cfg.lead_release_tau,
        )
        self._last_seq: int = -1
        self._q_scale = np.ones(3, dtype=np.float64)
        self._cam_pos_act = np.zeros(3, dtype=np.float64)
        self._cam_rot_act = np.eye(3, dtype=np.float64)

    def reset(self, keep_pose_history: bool = True) -> None:
        """
        重置卡尔曼状态与自适应门控（丢手、停止跟随或切回 IDLE 时调用）。

        Args:
            keep_pose_history: True 时保留相机位姿历史缓冲，避免恢复瞬间无滞后位姿可查。
        """
        self._kf.reset()
        self._axis_q_gate.reset()
        self._lead_gate.reset()
        self._last_seq = -1
        self._q_scale = np.ones(3, dtype=np.float64)
        if not keep_pose_history:
            self._pose_history.reset()

    def clear_residual_velocity(self) -> None:
        """刹车冷静窗 65% 处调用：清零世界系残余速度并重置自适应 Q/前瞻，消除残速拖尾。"""
        if self._kf.initialized:
            self._kf.x[3:6] = 0.0
        self._axis_q_gate.reset()
        self._lead_gate.reset()

    def step(
        self,
        obs: Observation,
        robot_state: RobotState,
        tcp_pos_act: np.ndarray,
        tcp_rot_act: np.ndarray,
        dt_s: float,
        *,
        is_holding: bool = False,
        osc_active: bool = False,
        reach_lead_scale: float = 1.0,
        error_world_for_lead: Optional[np.ndarray] = None,
    ) -> TargetEstimate:
        """
        执行单拍（100Hz）统一世界系状态估计。

        Args:
            obs: 当前最新视觉观测。
            robot_state: 机械臂状态快照（仅使用其单调时间戳）。
            tcp_pos_act: 基于真实 `q_act`（缓存不可用时退回 `q_cmd`）算出的 TCP 世界位置 (3,) m。
            tcp_rot_act: 对应的 TCP 世界旋转矩阵 (3, 3)。
            dt_s: 本拍真实时间步长 (s)。
            is_holding: 当前 FSM 是否处于 HOLDING 静止保持态（抑制过程噪声与前瞻）。
            osc_active: 当前是否处于振荡阻尼态（封锁 Q 放大与前瞻）。
            reach_lead_scale: 伸展度前瞻缩放倍率 [0.0, 1.0]。
            error_world_for_lead: 用于自适应前瞻方向门控的世界系误差向量 (3,) m。

        Returns:
            `TargetEstimate`：统一世界系位置、速度与前瞻预测。
        """
        dt = max(1e-4, min(0.10, float(dt_s) if math.isfinite(dt_s) else self.ctrl_dt_s))
        t_now = float(robot_state.timestamp_s)

        # 1. 更新真实物理相机位姿历史缓冲（SO(3) 测地插值）
        tcp_p = np.asarray(tcp_pos_act, dtype=np.float64)
        tcp_r = np.asarray(tcp_rot_act, dtype=np.float64)
        self._pose_history.push(t_now, tcp_p, tcp_r, self.T_tcp_camera, dt=dt)
        cam_rot_act = tcp_r @ self.R_tcp_cam
        cam_pos_act = tcp_p + tcp_r @ self.t_tcp_cam
        self._cam_pos_act = cam_pos_act
        self._cam_rot_act = cam_rot_act

        # 2. 判断本拍是否收到新的有效视觉观测
        new_obs = bool(
            obs.visible
            and obs.p_cam is not None
            and obs.seq != self._last_seq
            and np.all(np.isfinite(obs.p_cam))
        )

        # 3. 设定当前拍过程噪声倍率（静止或振荡时压低过程噪声，稳定输出）
        if is_holding:
            self._kf.q_scale = np.full(3, self.cfg.kf_static_q_scale, dtype=np.float64)
        elif osc_active:
            self._kf.q_scale = np.ones(3, dtype=np.float64)
        else:
            self._kf.q_scale = self._q_scale

        # 4. 若有新观测，按纯相机曝光延迟查表反投影至世界系并执行预测+更新；否则仅 predict
        if new_obs:
            self._last_seq = obs.seq

            t_lookup = float(obs.meas_time_s) - self.cfg.pose_delay_s
            looked = self._pose_history.lookup(t_lookup, use_lpf=True)
            if looked is not None:
                cam_pos_d, cam_rot_d = looked
            else:
                cam_pos_d, cam_rot_d = cam_pos_act, cam_rot_act

            p_cam = np.asarray(obs.p_cam, dtype=np.float64)
            z_world = cam_pos_d + cam_rot_d @ p_cam

            if not obs.frozen:
                # 完整 3D 新观测
                if not self._kf.initialized:
                    self._kf.reset()
                    self._kf.update(z_world, dt)
                    self._axis_q_gate.reset()
                    self._q_scale = np.ones(3, dtype=np.float64)
                else:
                    # 将世界系先验新息转回相机系以区分 XY 与深度 Z 轴的自适应 Q 阈值
                    innov_world = z_world - (self._kf.position() + self._kf.velocity() * dt)
                    innov_cam = cam_rot_d.T @ innov_world
                    if (not self.cfg.axis_adapt_enable) or is_holding or osc_active:
                        self._axis_q_gate.reset()
                        q_cam_scale = np.ones(3, dtype=np.float64)
                    else:
                        q_cam_scale = self._axis_q_gate.update(
                            innov_cam, dt, valid=True
                        )
                    self._q_scale = np.clip(
                        np.abs(cam_rot_d) @ q_cam_scale,
                        1.0,
                        self.cfg.axis_adapt_max_xy,
                    )
                    if not is_holding and not osc_active:
                        self._kf.q_scale = self._q_scale

                    # 各向异性观测噪声：沿相机光轴用 kf_r_z，正交平面用 kf_r_xy
                    R_cam_diag = np.diag([self.cfg.kf_r_xy, self.cfg.kf_r_xy, self.cfg.kf_r_z])
                    self._kf.r_vector = np.diag(cam_rot_d @ R_cam_diag @ cam_rot_d.T).copy()
                    self._kf.update(z_world, dt)
            else:
                # 深度冻结期（2D 新鲜、深度无效）：严格只更新相机光轴正交 XY 子空间，绝不污染 Z！
                if self._kf.initialized:
                    H_xy_world = cam_rot_d[:, :2].T  # (2, 3) 世界系 → 相机 XY 正交平面
                    z_xy_world = H_xy_world @ z_world
                    self._kf.update_partial(
                        z_xy_world,
                        H_xy_world,
                        dt,
                        r_diag=np.array([self.cfg.kf_r_xy, self.cfg.kf_r_xy], dtype=np.float64),
                    )
        elif self._kf.initialized:
            self._kf.predict(dt)

        # 5. 若估计器尚未就绪，返回无效占位
        if not self._kf.initialized:
            zero3 = np.zeros(3, dtype=np.float64)
            return TargetEstimate(
                valid=False,
                new_obs=new_obs,
                visible=obs.visible,
                frozen=obs.frozen,
                p_world=zero3,
                v_world=zero3,
                p_world_lead=zero3,
                obs_age_s=0.0,
                lead_s=0.0,
                lead_weight=0.0,
                q_scale=self._q_scale.copy(),
                cam_pos_act=cam_pos_act,
                cam_rot_act=cam_rot_act,
            )

        # 6. 观测年龄外推 + 自适应动态前瞻（HOLDING / 振荡时严格归零）
        p_w = self._kf.position()
        v_w = self._kf.velocity()
        obs_age = measurement_age(
            t_now,
            obs.meas_time_s,
            max_age=self.cfg.target_stale_s,
        )
        lead_raw_s = dynamic_lead_seconds(
            base_lead_ms=self.cfg.lead_base_ms,
            processing_s=obs.processing_s,
            max_lead_ms=self.cfg.lead_max_ms,
        )
        err_vec = (
            np.asarray(error_world_for_lead, dtype=np.float64)
            if error_world_for_lead is not None
            else np.zeros(3, dtype=np.float64)
        )
        if is_holding or osc_active or not obs.visible:
            self._lead_gate.reset()
            lead_weight = 0.0
            lead_eff_s = 0.0
        else:
            lead_weight, _ = self._lead_gate.update(
                velocity=v_w,
                position_error=err_vec,
                dt=dt,
                moving=True,
                valid=True,
            )
            lead_eff_s = lead_raw_s * lead_weight * float(np.clip(reach_lead_scale, 0.0, 1.0))

        total_extrap_s = 0.0 if is_holding else (obs_age + lead_eff_s)
        p_world_lead = p_w + v_w * total_extrap_s

        return TargetEstimate(
            valid=True,
            new_obs=new_obs,
            visible=obs.visible,
            frozen=obs.frozen,
            p_world=p_w,
            v_world=v_w,
            p_world_lead=p_world_lead,
            obs_age_s=obs_age,
            lead_s=lead_eff_s,
            lead_weight=lead_weight,
            q_scale=self._q_scale.copy(),
            cam_pos_act=cam_pos_act,
            cam_rot_act=cam_rot_act,
        )
