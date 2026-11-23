"""
层 2：显式视觉伺服有限状态机（servo_v14/state_machine.py）
=========================================================
将旧版散落在 `ff_moving`、`hold_latch`、`slow_follow`、`settled`、`post_brake`、
`lost_frames`、`supervisor_latched` 等十余个隐式标志位中的状态转移逻辑，
收束为一张显式、确定、可单测的状态机：

    IDLE ──(启动跟随)──► TRACKING ◄──(距离/速度迟滞)──► DOCKING
       ▲                   │  ▲                           │  ▲
       │                   │  └──(快启/超上沿/出死区)──────┴──┤
       │                   ▼                                 ▼
       │               COASTING ──(超时)──► LOST          HOLDING
       │                                                  (静止锁存)
       └────(停止跟随)──── FAULT_HOLD ◄──(M6 安全监督触发)────┘
                           RETURN_HOME ◄──(硬工作空间越界 >5cm)

核心行为约定（与 v13.0 实机验收保持一致）：
  - `HOLDING` 态严格保持当前物理位姿，绝不因微弱深度噪声产生慢速漂移；
  - 从运动进入 `HOLDING` 时触发 `PostBrakeSettleGuard` 冷静窗，并在 65% 窗长处发出
    一次性 `kf_reanchor_event=True` 清理 KF 残速，彻底杜绝悬臂弹性回弹二次误唤醒；
  - 死区外快速折返过零需连续低速满 `stop_confirm_s=80ms` 才入静止，消除中途假停顿。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from servo_timing import FastStartGate, PostBrakeSettleGuard
from .config import FSMConfig
from .types import FSMDecision, FSMMode, TargetEstimate


class ServoFSM:
    """显式 8 态视觉伺服有限状态机（层 2）。"""

    def __init__(self, cfg: FSMConfig) -> None:
        """
        初始化有限状态机。

        Args:
            cfg: 状态机配置（`FSMConfig`）。
        """
        self.cfg = cfg
        self.mode: FSMMode = FSMMode.IDLE
        self._fast_start = FastStartGate(
            enable=cfg.fast_start_enable,
            frames=cfg.fast_start_frames,
            distance=cfg.fast_start_dist,
            direction_cos=cfg.fast_start_dir_cos,
        )
        self._brake_guard = PostBrakeSettleGuard(
            base_settle_s=cfg.post_brake_settle_s,
            ext_extra_s=cfg.post_brake_ext_extra_s,
        )
        self._exit_timer_s: float = 0.0
        self._stop_timer_s: float = 0.0
        self._lost_timer_s: float = 0.0
        self._hold_anchor_world: Optional[np.ndarray] = None

    def reset(self, mode: FSMMode = FSMMode.IDLE) -> None:
        """复位全部状态机计时器、门控与静止锚点。"""
        self.mode = mode
        self._fast_start.reset()
        self._brake_guard.reset()
        self._exit_timer_s = 0.0
        self._stop_timer_s = 0.0
        self._lost_timer_s = 0.0
        self._hold_anchor_world = None

    def step(
        self,
        *,
        following_enabled: bool,
        estimate: TargetEstimate,
        error_norm_m: float,
        joint_speed_rad_s: float,
        dt_s: float,
        ext_ratio: float = 0.0,
        supervisor_latched: bool = False,
        hard_wall_breached: bool = False,
    ) -> FSMDecision:
        """
        根据当前观测估计、误差模长、安全状态推进状态机一拍。

        Args:
            following_enabled: 用户是否已开启跟随（按 `g`/空格）。
            estimate: 层 1 输出的统一世界系目标估计。
            error_norm_m: 世界系相机期望位置误差模长 ||p*_cam - p_cam|| (m)。
            joint_speed_rad_s: 当前最大单轴指令角速度 (rad/s)，驱动冷静窗状态。
            dt_s: 本拍真实时间步长 (s)。
            ext_ratio: 当前伸展度 [0.0, 1.0]，用于冷静窗时长与唤醒门槛加权。
            supervisor_latched: M6 200Hz 安全监督是否处于软停机锁存态。
            hard_wall_breached: 安全护盾是否检测到工作空间硬越界 >5cm。

        Returns:
            `FSMDecision`：当前状态、转移原因与下游控制使能标志。
        """
        dt = max(1e-4, min(0.10, float(dt_s) if math.isfinite(dt_s) else 0.01))
        err_norm = float(error_norm_m) if math.isfinite(error_norm_m) else 0.0
        ext = float(np.clip(ext_ratio, 0.0, 1.0)) if math.isfinite(ext_ratio) else 0.0

        prev_mode = self.mode
        reason = ""

        # 始终推进刹停冷静窗
        self._brake_guard.update(dt, float(joint_speed_rad_s))
        in_cooldown = self._brake_guard.in_cooldown(ext_ratio=ext)
        kf_reanchor = self._brake_guard.consume_kf_reanchor(ext_ratio=ext)

        # ── 最高优先级 1：硬工作空间越界 → RETURN_HOME ────────────
        if hard_wall_breached:
            if self.mode != FSMMode.RETURN_HOME:
                reason = "hard_wall_breach"
            self.mode = FSMMode.RETURN_HOME
            return self._decision(prev_mode, reason, in_cooldown, False, err_norm)

        # ── 最高优先级 2：M6 主动安全监督软停机 → FAULT_HOLD ───────
        if supervisor_latched:
            if self.mode != FSMMode.FAULT_HOLD:
                reason = "m6_supervisor_latch"
            self.mode = FSMMode.FAULT_HOLD
            self._exit_timer_s = 0.0
            self._stop_timer_s = 0.0
            return self._decision(prev_mode, reason, in_cooldown, False, err_norm)

        # ── 优先级 3：未开启跟随 → IDLE ──────────────────────────
        if not following_enabled:
            if self.mode != FSMMode.IDLE:
                reason = "follow_disabled"
            self.reset(FSMMode.IDLE)
            return self._decision(prev_mode, reason, False, False, 0.0)

        # 从 IDLE / FAULT_HOLD / RETURN_HOME 恢复跟随
        if self.mode in (FSMMode.IDLE, FSMMode.FAULT_HOLD, FSMMode.RETURN_HOME):
            if estimate.valid and estimate.visible:
                self.mode = (
                    FSMMode.DOCKING
                    if err_norm <= self.cfg.docking_dist_m
                    else FSMMode.TRACKING
                )
                reason = "follow_started"
            else:
                self.mode = FSMMode.LOST
                reason = "waiting_target"

        # ── 优先级 4：目标不可见 → COASTING → LOST ───────────────
        if not estimate.visible or not estimate.valid:
            self._lost_timer_s += dt
            self._exit_timer_s = 0.0
            self._stop_timer_s = 0.0
            if self.mode in (FSMMode.TRACKING, FSMMode.DOCKING, FSMMode.COASTING):
                if self._lost_timer_s <= self.cfg.coast_window_s and estimate.valid:
                    if self.mode != FSMMode.COASTING:
                        reason = "brief_loss_coast"
                    self.mode = FSMMode.COASTING
                elif self._lost_timer_s >= self.cfg.lost_hold_s:
                    if self.mode != FSMMode.LOST:
                        reason = "lost_timeout"
                    self.mode = FSMMode.LOST
                    self._hold_anchor_world = None
                else:
                    # 超过滑行窗但未到重置窗：原地保持等待重获
                    if self.mode != FSMMode.HOLDING:
                        reason = "coast_ended_hold"
                    self.mode = FSMMode.HOLDING
            else:
                if self._lost_timer_s >= self.cfg.lost_hold_s and self.mode != FSMMode.LOST:
                    reason = "lost_timeout"
                    self.mode = FSMMode.LOST
                    self._hold_anchor_world = None
            return self._decision(prev_mode, reason, in_cooldown, kf_reanchor, err_norm)

        # 目标可见：清空丢失计时器
        self._lost_timer_s = 0.0
        speed_w = float(np.linalg.norm(estimate.v_world))
        deadband_in = self.cfg.deadband_in_m
        deadband_out = self.cfg.deadband_in_m * self.cfg.deadband_hysteresis

        # ── 从 LOST / COASTING 重获目标 ──────────────────────────
        if self.mode in (FSMMode.LOST, FSMMode.COASTING):
            if err_norm <= deadband_in and speed_w <= self.cfg.vel_deadband_off:
                self.mode = FSMMode.HOLDING
                self._hold_anchor_world = estimate.p_world.copy()
                self._fast_start.reset(origin=self._hold_anchor_world)
                reason = "reacquired_static"
            elif err_norm <= self.cfg.docking_dist_m:
                self.mode = FSMMode.DOCKING
                reason = "reacquired_docking"
            else:
                self.mode = FSMMode.TRACKING
                reason = "reacquired_tracking"
            return self._decision(prev_mode, reason, in_cooldown, kf_reanchor, err_norm)

        # ── HOLDING 静止保持态唤醒判据 ───────────────────────────
        if self.mode == FSMMode.HOLDING:
            if self._hold_anchor_world is None:
                self._hold_anchor_world = estimate.p_world.copy()
                self._fast_start.reset(origin=self._hold_anchor_world)

            # 冷静窗与深伸展自动抬高唤醒门槛，防止悬臂弹性回弹误触发
            eff_vel_on = self._brake_guard.effective_unlock_threshold(
                self.cfg.vel_deadband_on, ext_ratio=ext
            )
            eff_fs_dist = self._brake_guard.effective_fast_start_dist(
                self.cfg.fast_start_dist, ext_ratio=ext
            )
            self._fast_start.distance = eff_fs_dist

            # 通道 A：快启门控（连续 2 帧同向真实位移）
            fast_unlocked = False
            if estimate.new_obs:
                fast_unlocked = bool(self._fast_start.update(estimate.p_world))

            # 通道 B：速度持续越过上沿满 exit_confirm_s
            if speed_w >= eff_vel_on:
                self._exit_timer_s += dt
            else:
                self._exit_timer_s = 0.0
            vel_unlocked = self._exit_timer_s >= self.cfg.exit_confirm_s

            # 通道 C：大幅离开静止锚点或大幅超出迟滞死区（真实大幅移手直通）
            anchor_disp = float(np.linalg.norm(estimate.p_world - self._hold_anchor_world))
            large_disp_unlock = (
                (not in_cooldown and err_norm >= deadband_out and speed_w >= self.cfg.vel_deadband_off)
                or (anchor_disp >= max(deadband_out, 0.055))
            )

            if fast_unlocked or vel_unlocked or large_disp_unlock:
                self._exit_timer_s = 0.0
                self._stop_timer_s = 0.0
                self._hold_anchor_world = None
                self.mode = (
                    FSMMode.DOCKING
                    if err_norm <= self.cfg.docking_dist_m
                    else FSMMode.TRACKING
                )
                reason = (
                    "fast_start_unlock"
                    if fast_unlocked
                    else ("vel_unlock" if vel_unlocked else "disp_unlock")
                )
            return self._decision(prev_mode, reason, in_cooldown, kf_reanchor, err_norm)

        # ── TRACKING / DOCKING 运动态停稳与切换判据 ──────────────
        # 1) 进入死区且手掌静止 → 立即进入 HOLDING
        if err_norm <= deadband_in and speed_w <= self.cfg.vel_deadband_off:
            self.mode = FSMMode.HOLDING
            self._hold_anchor_world = estimate.p_world.copy()
            self._fast_start.reset(origin=self._hold_anchor_world)
            self._exit_timer_s = 0.0
            self._stop_timer_s = 0.0
            reason = "settled_in_deadband"
            return self._decision(prev_mode, reason, in_cooldown, kf_reanchor, err_norm)

        # 2) 连续低速满 stop_confirm_s（80ms 防折返过零假停顿）→ 无条件进入 HOLDING 保持当前位姿
        # （与 v13.0 静止保持行为一致：不要求误差必须 <= deadband_out，避免手停后在 3.5~6cm 持续蠕动）
        if speed_w <= self.cfg.vel_deadband_off:
            self._stop_timer_s += dt
            if self._stop_timer_s >= self.cfg.stop_confirm_s:
                self.mode = FSMMode.HOLDING
                self._hold_anchor_world = estimate.p_world.copy()
                self._fast_start.reset(origin=self._hold_anchor_world)
                self._stop_timer_s = 0.0
                reason = "target_stopped_hold"
                return self._decision(prev_mode, reason, in_cooldown, kf_reanchor, err_norm)
        else:
            self._stop_timer_s = 0.0

        # 3) TRACKING ↔ DOCKING 距离迟滞切换
        dock_enter = self.cfg.docking_dist_m
        dock_exit = self.cfg.docking_dist_m * 1.25
        if self.mode == FSMMode.TRACKING and err_norm <= dock_enter:
            self.mode = FSMMode.DOCKING
            reason = "enter_docking"
        elif self.mode == FSMMode.DOCKING and err_norm >= dock_exit:
            self.mode = FSMMode.TRACKING
            reason = "exit_docking"

        return self._decision(prev_mode, reason, in_cooldown, kf_reanchor, err_norm)

    def _decision(
        self,
        prev_mode: FSMMode,
        reason: str,
        in_cooldown: bool,
        kf_reanchor: bool,
        err_norm: float,
    ) -> FSMDecision:
        """按当前生效模式打包标准 `FSMDecision` 输出。"""
        allow_motion = self.mode in (FSMMode.TRACKING, FSMMode.DOCKING, FSMMode.COASTING)
        allow_ff = self.mode in (FSMMode.TRACKING, FSMMode.DOCKING)
        docking = self.mode == FSMMode.DOCKING
        return FSMDecision(
            mode=self.mode,
            prev_mode=prev_mode,
            transition_reason=reason,
            allow_motion=allow_motion,
            allow_ff=allow_ff,
            docking=docking,
            in_cooldown=in_cooldown,
            kf_reanchor_event=kf_reanchor,
            hold_anchor_world=(
                self._hold_anchor_world.copy()
                if self._hold_anchor_world is not None
                else None
            ),
            error_norm_m=err_norm,
        )
