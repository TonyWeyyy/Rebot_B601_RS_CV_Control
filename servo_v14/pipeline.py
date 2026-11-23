"""
v14 四层流水线总编排器（servo_v14/pipeline.py）
=============================================
将 Layer 1（`WorldTargetEstimator`）→ Layer 2（`ServoFSM`）→ Layer 3（`WorldPBVSController`）
→ Layer 4（`SafetyShield`）串成单一、无全局状态、确定性的 `step()` 调用。

设计价值：
  - **真机 100Hz 线程与离线数字孪生回放 100% 共享同一 `ServoV14Pipeline` 类**；
  - 每次 `step()` 返回 `(ShieldedCommand, ServoTickTelemetry)`，同时产出兼容现有
    `tools/analyze_servo_log.py` 的高频诊断字典，实现 v13 vs v14 同口径定量 A/B 对照。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import V14Config
from .controller_pbvs import WorldPBVSController
from .estimator import WorldTargetEstimator
from .robot_backend import RobotBackend
from .safety_shield import SafetyShield
from .state_machine import ServoFSM
from .types import (
    FSMDecision,
    FSMMode,
    Observation,
    RobotState,
    ServoTickTelemetry,
    ShieldedCommand,
    TargetEstimate,
)


class ServoV14Pipeline:
    """v14 四层解耦视觉伺服流水线编排器。"""

    def __init__(
        self,
        cfg: V14Config,
        backend: RobotBackend,
        T_tcp_camera: np.ndarray,
    ) -> None:
        """
        初始化四层解耦视觉伺服流水线。

        Args:
            cfg: 已校验的不可变 `V14Config`。
            backend: 真机后端 `HardwareRobotBackend` 或离线数字孪生 `DigitalTwinBackend`。
            T_tcp_camera: 4×4 手眼标定外参矩阵。
        """
        self.cfg = cfg
        self.backend = backend
        self.T_tcp_camera = np.asarray(T_tcp_camera, dtype=np.float64).copy()

        # 四层组件各自独立实例化，零共享全局变量
        self.estimator = WorldTargetEstimator(
            cfg=cfg.estimator,
            T_tcp_camera=self.T_tcp_camera,
            ctrl_dt_s=cfg.ctrl_dt_s,
            vel_deadband_off=cfg.fsm.vel_deadband_off,
            vel_deadband_on=cfg.fsm.vel_deadband_on,
        )
        self.fsm = ServoFSM(cfg=cfg.fsm)
        self.controller = WorldPBVSController(
            pbvs_cfg=cfg.pbvs,
            safety_cfg=cfg.safety,
            T_tcp_camera=self.T_tcp_camera,
            ctrl_dt_s=cfg.ctrl_dt_s,
        )
        self.shield = SafetyShield(
            cfg=cfg.safety,
            q_min=backend.q_min,
            q_max=backend.q_max,
        )

        self._last_joint_speed: float = 0.0
        self._last_err_world = np.zeros(3, dtype=np.float64)
        self._last_decision: Optional[FSMDecision] = None
        self._last_estimate: Optional[TargetEstimate] = None

    def reset(self, *, keep_pose_history: bool = True, clear_reference_ori: bool = False) -> None:
        """复位四层状态（启停跟随、回 HOME 或重跑离线仿真时调用）。"""
        self.estimator.reset(keep_pose_history=keep_pose_history)
        self.fsm.reset(FSMMode.IDLE)
        self.controller.reset(clear_reference_ori=clear_reference_ori)
        self._last_joint_speed = 0.0
        self._last_err_world[:] = 0.0
        self._last_decision = None
        self._last_estimate = None

    def lock_current_orientation(self, q_cmd: Optional[np.ndarray] = None) -> None:
        """将当前控制位姿对应的 TCP 旋转矩阵锁定为姿态保持参考 `R_tcp_init`。"""
        if q_cmd is None:
            q_cmd = self.backend.get_state(time.monotonic()).q_cmd
        _, tcp_rot = self.backend.fk_only(q_cmd)
        self.controller.set_reference_orientation(tcp_rot)

    def step(
        self,
        obs: Observation,
        robot_state: RobotState,
        dt_s: float,
        *,
        following_enabled: bool,
        apply_to_backend: bool = True,
    ) -> Tuple[ShieldedCommand, ServoTickTelemetry]:
        """
        执行单拍（100Hz）完整的四层视觉伺服闭环：
        Estimator → FSM → PBVS Controller → SafetyShield (→ Backend)。

        Args:
            obs: 当前视觉观测包。
            robot_state: 当前机械臂状态快照。
            dt_s: 本拍真实时间步长 (s)。
            following_enabled: 是否启用跟随。
            apply_to_backend: True 时自动将审定后的安全指令下发至后端。

        Returns:
            `(ShieldedCommand, ServoTickTelemetry)`。
        """
        t_work_start = time.perf_counter()
        dt = max(1e-4, min(0.05, float(dt_s) if math.isfinite(dt_s) else self.cfg.ctrl_dt_s))

        # ── 运动学计算严格双轨分离 ────────────────────────────────
        # 控制轨道（严格 q_cmd）：用于 PBVS 雅可比、工作空间软/硬墙、关节积分
        tcp_pos_cmd, tcp_rot_cmd, J_cmd = self.backend.fk_and_jacobian(robot_state.q_cmd)
        # 感知轨道（优先 200Hz 真实 q_act）：仅用于世界系相机历史与目标重建
        if robot_state.feedback_valid:
            tcp_pos_act, tcp_rot_act = self.backend.fk_only(robot_state.q_act)
        else:
            tcp_pos_act, tcp_rot_act = tcp_pos_cmd, tcp_rot_cmd

        reach_r = float(math.hypot(tcp_pos_cmd[0], tcp_pos_cmd[1]))
        ext_ratio = float(np.clip(
            (reach_r - self.cfg.pbvs.reach_soft_start)
            / max(self.cfg.pbvs.reach_soft_end - self.cfg.pbvs.reach_soft_start, 1e-4),
            0.0,
            1.0,
        ))
        hard_breach = self.shield.check_hard_workspace_breach(tcp_pos_cmd)

        # ── Layer 1：统一世界系目标估计器 ─────────────────────────
        is_holding_prev = self.fsm.mode == FSMMode.HOLDING
        estimate = self.estimator.step(
            obs=obs,
            robot_state=robot_state,
            tcp_pos_act=tcp_pos_act,
            tcp_rot_act=tcp_rot_act,
            dt_s=dt,
            is_holding=is_holding_prev,
            osc_active=self.controller.last_osc_active,
            reach_lead_scale=self.controller.last_reach_lead_scale,
            error_world_for_lead=self._last_err_world,
        )

        # 计算当前世界系误差（基于估计位置与当前指令相机位姿）
        if estimate.valid:
            err_world = self.controller.compute_world_position_error(
                estimate.p_world, tcp_pos_cmd, tcp_rot_cmd
            )
            err_norm = float(np.linalg.norm(err_world))
            self._last_err_world = err_world.copy()
        else:
            err_world = np.zeros(3, dtype=np.float64)
            err_norm = 0.0
            self._last_err_world[:] = 0.0

        # ── Layer 2：显式有限状态机 ───────────────────────────────
        decision = self.fsm.step(
            following_enabled=following_enabled,
            estimate=estimate,
            error_norm_m=err_norm,
            joint_speed_rad_s=self._last_joint_speed,
            dt_s=dt,
            ext_ratio=ext_ratio,
            supervisor_latched=robot_state.supervisor_latched,
            hard_wall_breached=hard_breach,
        )

        # 状态机事件回调：长丢手复位估计器 / 冷静窗 65% 清零残速
        if decision.mode == FSMMode.LOST and decision.prev_mode != FSMMode.LOST:
            self.estimator.reset(keep_pose_history=True)
        elif decision.kf_reanchor_event:
            self.estimator.clear_residual_velocity()

        # ── Layer 3：统一世界系 PBVS 控制器 ───────────────────────
        ctrl_cmd = self.controller.step(
            estimate=estimate,
            decision=decision,
            tcp_pos_cmd=tcp_pos_cmd,
            tcp_rot_cmd=tcp_rot_cmd,
            jacobian_cmd=J_cmd,
            now_s=robot_state.timestamp_s,
            dt_s=dt,
        )

        # ── Layer 4：不可绕过安全护盾 ─────────────────────────────
        shielded = self.shield.filter(ctrl_cmd, robot_state, dt)
        self._last_joint_speed = float(np.max(np.abs(shielded.dq_cmd_safe)))
        self._last_decision = decision
        self._last_estimate = estimate

        if apply_to_backend:
            self.backend.apply_command(shielded, dt)

        # 组装本拍诊断快照（含相机系投影误差以便与 v13 日志同口径对比）
        cam_rot_act = estimate.cam_rot_act
        err_cam = cam_rot_act.T @ err_world if estimate.valid else np.zeros(3, dtype=np.float64)
        track_err_deg = float(np.max(np.abs(np.rad2deg(robot_state.q_cmd - robot_state.q_act))))
        work_ms = (time.perf_counter() - t_work_start) * 1000.0

        telemetry = ServoTickTelemetry(
            host_time_s=robot_state.timestamp_s,
            dt_s=dt,
            seq=obs.seq,
            new_obs=estimate.new_obs,
            mode=decision.mode,
            visible=obs.visible,
            frozen=obs.frozen,
            kf_valid=estimate.valid,
            error_cam=err_cam,
            error_norm_m=err_norm,
            p_world_est=estimate.p_world,
            v_world_est=estimate.v_world,
            q_cmd=shielded.q_cmd_next,
            dq_cmd=shielded.dq_cmd_safe,
            q_act=robot_state.q_act,
            dq_act=robot_state.dq_act,
            tau_act=robot_state.tau_act,
            tracking_err_max_deg=track_err_deg,
            joint_speed_max=self._last_joint_speed,
            obs_age_ms=estimate.obs_age_s * 1000.0,
            lead_ms=estimate.lead_s * 1000.0,
            ctrl_work_ms=work_ms,
            shield_flags=shielded.shield_reason,
        )
        return shielded, telemetry

    def telemetry_to_diag_row(
        self,
        telem: ServoTickTelemetry,
        robot_state: RobotState,
        following_enabled: bool,
    ) -> Dict[str, Any]:
        """将 `ServoTickTelemetry` 转为与 `servo_diagnostics.DIAG_FIELDS` 100% 兼容的 CSV 行。"""
        is_moving = telem.mode in (FSMMode.TRACKING, FSMMode.DOCKING)
        is_holding = telem.mode == FSMMode.HOLDING
        target_cam = (
            telem.error_cam + np.array([0.0, 0.0, self.cfg.pbvs.distance])
            if telem.kf_valid
            else None
        )
        row: Dict[str, Any] = {
            "host_time": telem.host_time_s,
            "dt_s": telem.dt_s,
            "seq": telem.seq,
            "new_obs": int(telem.new_obs),
            "following": int(following_enabled),
            "visible": int(telem.visible),
            "frozen": int(telem.frozen),
            "kf_valid": int(telem.kf_valid),
            "ff_moving": int(is_moving),
            "hold_latch": int(is_holding),
            "slow_active": 0,
            "static_hold": int(is_holding),
            "lead_weight": (
                self._last_estimate.lead_weight if self._last_estimate is not None else 0.0
            ),
            "lead_raw_s": self.cfg.estimator.lead_base_ms * 1e-3,
            "lead_s": telem.lead_ms * 1e-3,
            "obs_age_ms": telem.obs_age_ms,
            "processing_ms": 0.0,
            "ctrl_work_ms": telem.ctrl_work_ms,
            "overrun": int(telem.ctrl_work_ms > self.cfg.ctrl_dt_s * 1000.0),
            "error_norm": telem.error_norm_m if telem.kf_valid else "",
            "target_x": float(target_cam[0]) if target_cam is not None else "",
            "target_y": float(target_cam[1]) if target_cam is not None else "",
            "target_z": float(target_cam[2]) if target_cam is not None else "",
            "kv_x": float(telem.v_world_est[0]),
            "kv_y": float(telem.v_world_est[1]),
            "kv_z": float(telem.v_world_est[2]),
            "joint_speed": telem.joint_speed_max,
            "fbc_valid": int(robot_state.feedback_valid),
            "fbc_age_ms": robot_state.feedback_age_ms,
            "tmos_max": robot_state.t_mos_max_c,
            "mit_period_p99_ms": robot_state.mit_period_p99_ms,
            "can_fail_total": robot_state.can_fail_total,
            "mit_dq_ff_max": float(np.max(np.abs(telem.dq_cmd))),
            "tau_ff_2": float(robot_state.tau_ff_nm[1]),
            "tau_ff_3": float(robot_state.tau_ff_nm[2]),
            "tau_ff_4": float(robot_state.tau_ff_nm[3]),
            "kd_j1": robot_state.kd_j1,
        }
        for i in range(6):
            row[f"q_cmd_{i + 1}"] = float(telem.q_cmd[i])
            row[f"qa_{i + 1}"] = float(telem.q_act[i])
            row[f"dqa_{i + 1}"] = float(telem.dq_act[i])
            row[f"tau_{i + 1}"] = float(telem.tau_act[i])
        return row
