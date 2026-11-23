"""
层 4：不可绕过独立安全护盾（servo_v14/safety_shield.py）
========================================================
位于任何上层控制律（PBVS / 规划器 / 未来 RL 或 VLA 策略）与底层执行器之间的
**统一硬安全拦截层**。所有 `AGENTS.md` 中的物理安全红线全部收敛于此，单一审计入口：

  1. NaN / Inf 输入检测：任何非有限值立刻 fail-safe 归零；
  2. M6 200Hz 主动安全监督锁存：触发碰撞/堵转/超差时强行置零速度（维持重力补偿，绝不半空失能）；
  3. 工作空间硬墙检测：越出 `[min, max]` 或 `max_radius` 超过 `hard_wall_margin_m=5cm`
     → 输出 `trigger_return_home=True` 并清零当前拍速度；
  4. 工作空间软墙裁剪：
     - XYZ 三轴盒式单向软墙（只削外推分量，绝不锁死回退）；
     - 水平径向圆柱软墙 `horizontal_radial_soft_wall(max_radius <= 0.42m)`
       （只限水平 `hypot(X, Y)`，完整保留向内回收、切向滑动与竖直 Z 运动）；
  5. 可操作度奇异保护：低可操作度时按自适应阻尼重解伪逆并平滑降速至 `min_manip_speed_scale`；
  6. M6 温升前置线性降额（MOS 90~110°C → `thermal_speed_scale ∈ [0.40, 1.0]`）；
  7. 关节软限位线性缓冲 `apply_joint_limit_cushion()`（近 `q_min/q_max` ~4.6° 收紧外推速度，
     并强制将受缓冲轴的 MIT 速度前馈 `dq_ff_safe` 置 0）；
  8. L2 分关节硬限速 `HARD_MAX_JOINT_SPEED = [2.5, 2.0, 2.0, 3.0, 3.0, 2.5]`
     （承重 `J2/J3 <= 2.0 rad/s` 绝对不可突破；停靠/深伸展/振荡自动收回 `<= 2.0 rad/s`）
     与安全指令积分 `q_cmd_next = clip(q_cmd + dq * dt, q_min, q_max)`。
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from servo_timing import (
    apply_joint_limit_cushion,
    compute_weighted_damped_pseudoinverse,
    horizontal_radial_soft_wall,
    scale_joint_velocities,
)
from .config import HARD_MAX_JOINT_SPEED, SafetyConfig
from .types import ControlCommand, RobotState, ShieldedCommand


def compute_manipulability(J: np.ndarray) -> float:
    """计算 6×6 雅可比的可操作度 `sqrt(max(det(J @ J.T), 0))`。"""
    det = float(np.linalg.det(J @ J.T))
    return math.sqrt(max(det, 0.0)) if math.isfinite(det) else 0.0


class SafetyShield:
    """不可绕过的物理安全拦截护盾（层 4）。"""

    def __init__(
        self,
        cfg: SafetyConfig,
        q_min: np.ndarray,
        q_max: np.ndarray,
    ) -> None:
        """
        初始化安全护盾。

        Args:
            cfg: 安全护盾配置（已通过 `validate_v14_config` 校验）。
            q_min: 6 轴关节下限 (rad)。
            q_max: 6 轴关节上限 (rad)。
        """
        self.cfg = cfg
        self.q_min = np.asarray(q_min, dtype=np.float64).copy()
        self.q_max = np.asarray(q_max, dtype=np.float64).copy()
        self.ws_min = np.asarray(cfg.workspace_min, dtype=np.float64)
        self.ws_max = np.asarray(cfg.workspace_max, dtype=np.float64)
        # 无论配置如何，逐轴硬上限再与 HARD_MAX_JOINT_SPEED 取 min 双保险锁死
        self.hard_joint_limits = np.minimum(
            np.asarray(cfg.max_vel_joints, dtype=np.float64),
            np.asarray(HARD_MAX_JOINT_SPEED, dtype=np.float64),
        )
        self.dock_joint_limits = np.full(
            6, min(float(cfg.max_vel_joint_dock), 2.0), dtype=np.float64
        )

    def check_hard_workspace_breach(self, tcp_pos: np.ndarray) -> bool:
        """检查 TCP 位置是否越出工作空间硬边界（超出软墙 `hard_wall_margin_m=5cm`）。"""
        if not np.all(np.isfinite(tcp_pos)):
            return True
        m = self.cfg.hard_wall_margin_m
        if np.any(tcp_pos < self.ws_min - m) or np.any(tcp_pos > self.ws_max + m):
            return True
        horiz_r = float(math.hypot(tcp_pos[0], tcp_pos[1]))
        if horiz_r > self.cfg.workspace_max_radius + m:
            return True
        return False

    def filter(
        self,
        cmd: ControlCommand,
        robot_state: RobotState,
        dt_s: float,
    ) -> ShieldedCommand:
        """
        对上层 `ControlCommand` 执行全套安全拦截与指令积分。

        Args:
            cmd: 层 3（或任何策略）输出的期望控制指令。
            robot_state: 当前机械臂状态（使用 `q_cmd` 做运动学/积分，使用 M6 监督状态做限速/保持）。
            dt_s: 本拍真实时间步长 (s)。

        Returns:
            `ShieldedCommand`：100% 满足全部硬件安全红线的可执行关节指令。
        """
        dt = max(1e-4, min(0.05, float(dt_s) if math.isfinite(dt_s) else 0.01))
        q_cmd = np.asarray(robot_state.q_cmd, dtype=np.float64)
        zero6 = np.zeros(6, dtype=np.float64)
        reasons: List[str] = []

        # ── 1. 非有限值 fail-safe ─────────────────────────────────
        if (
            not np.all(np.isfinite(q_cmd))
            or not np.all(np.isfinite(cmd.v_tcp_world))
            or not np.all(np.isfinite(cmd.w_tcp_world))
            or not np.all(np.isfinite(cmd.tcp_pos_cmd))
            or not np.all(np.isfinite(cmd.jacobian_cmd))
        ):
            q_safe = np.clip(np.nan_to_num(q_cmd, nan=0.0, posinf=0.0, neginf=0.0), self.q_min, self.q_max)
            return ShieldedCommand(
                q_cmd_next=q_safe,
                dq_cmd_safe=zero6,
                dq_ff_safe=zero6,
                reach_radius_m=0.0,
                trigger_return_home=False,
                soft_wall_active=False,
                limit_cushion_active=False,
                singularity_active=False,
                speed_clamped=False,
                manipulability=0.0,
                shield_reason="nonfinite_input",
            )

        tcp_p = cmd.tcp_pos_cmd
        reach_r = float(math.hypot(tcp_p[0], tcp_p[1]))
        manip = compute_manipulability(cmd.jacobian_cmd)

        # ── 2. M6 主动安全监督软停机：强制速度为 0，保持当前 q_cmd ──
        if robot_state.supervisor_latched:
            return ShieldedCommand(
                q_cmd_next=np.clip(q_cmd, self.q_min, self.q_max),
                dq_cmd_safe=zero6,
                dq_ff_safe=zero6,
                reach_radius_m=reach_r,
                trigger_return_home=False,
                soft_wall_active=False,
                limit_cushion_active=False,
                singularity_active=False,
                speed_clamped=False,
                manipulability=manip,
                shield_reason=f"m6_hold:{robot_state.supervisor_reason}",
            )

        # ── 3. 工作空间硬墙越界 >5cm：请求自动安全回 HOME ─────────
        if self.check_hard_workspace_breach(tcp_p):
            return ShieldedCommand(
                q_cmd_next=np.clip(q_cmd, self.q_min, self.q_max),
                dq_cmd_safe=zero6,
                dq_ff_safe=zero6,
                reach_radius_m=reach_r,
                trigger_return_home=True,
                soft_wall_active=True,
                limit_cushion_active=False,
                singularity_active=False,
                speed_clamped=False,
                manipulability=manip,
                shield_reason="hard_wall_breach",
            )

        # ── 4. 工作空间软墙裁剪（盒式 XYZ + 70% 水平柱面径向） ─────
        v_tcp = cmd.v_tcp_world.copy()
        w_tcp = cmd.w_tcp_world.copy()
        soft_wall_active = False

        # 4a) 末端线速度硬上限（防止任何外部策略突破 0.85 m/s）
        v_norm = float(np.linalg.norm(v_tcp))
        if v_norm > self.cfg.max_vel:
            v_tcp = v_tcp * (self.cfg.max_vel / v_norm)
            reasons.append("max_vel_cap")

        w_norm = float(np.linalg.norm(w_tcp))
        if w_norm > self.cfg.max_vel_angular:
            w_tcp = w_tcp * (self.cfg.max_vel_angular / w_norm)

        # 4b) XYZ 单向盒式软墙（只削外推分量，保留回退）
        for axis in range(3):
            if tcp_p[axis] <= self.ws_min[axis] and v_tcp[axis] < 0.0:
                v_tcp[axis] = 0.0
                soft_wall_active = True
            elif tcp_p[axis] >= self.ws_max[axis] and v_tcp[axis] > 0.0:
                v_tcp[axis] = 0.0
                soft_wall_active = True

        # 4c) 水平径向 70% 臂展软墙（仅作用于 hypot(X, Y)，完整保留 Z 运动）
        v_after_rad, rad_clipped = horizontal_radial_soft_wall(
            tcp_p, v_tcp, self.cfg.workspace_max_radius
        )
        if rad_clipped:
            v_tcp = v_after_rad
            soft_wall_active = True
        if soft_wall_active:
            reasons.append("soft_wall")

        # ── 5. 可操作度奇异保护与安全逆解（与 v13.0 实机验收公式完全一致） ──
        singularity_active = False
        damping = self.cfg.damping_base
        manip_scale = 1.0
        if self.cfg.damping_adaptive and manip < self.cfg.manipulability_threshold:
            singularity_active = True
            damping = self.cfg.damping_base * (
                1.0 + (self.cfg.manipulability_threshold - manip) * 100.0
            )
            manip_scale = max(
                self.cfg.min_manip_speed_scale,
                manip / max(self.cfg.manipulability_threshold, 1e-9),
            )
            reasons.append("singularity")

        # 若软墙修改了速度或处于奇异区，则按护盾后的安全 twist 重新求伪逆；否则直接采用 dq_des
        if soft_wall_active or singularity_active or not np.all(np.isfinite(cmd.dq_des)):
            row_w = np.array(
                [1.0, 1.0, 1.0, cmd.ori_weight, cmd.ori_weight, cmd.ori_weight],
                dtype=np.float64,
            )
            J_pinv = compute_weighted_damped_pseudoinverse(
                cmd.jacobian_cmd, row_w, damping=damping
            )
            dq = (J_pinv @ np.concatenate([v_tcp, w_tcp])) * manip_scale
        else:
            dq = cmd.dq_des.copy()

        # ── 6. M6 高温前置速度降额（MOS 90~110°C） ─────────────────
        thermal_scale = float(np.clip(robot_state.thermal_speed_scale, 0.20, 1.0))
        if thermal_scale < 0.999:
            dq = dq * thermal_scale
            reasons.append(f"thermal_{thermal_scale:.2f}")

        # ── 7. 关节软限位线性缓冲 + 活跃集列屏蔽重解（防限位关节被截断后引发其余轴偏航/俯冲）──
        dq_before_cushion = dq.copy()
        dq, limit_cushion_active = apply_joint_limit_cushion(
            q_now=q_cmd,
            joint_vel=dq,
            q_min=self.q_min,
            q_max=self.q_max,
            margin_rad=self.cfg.joint_limit_cushion_rad,
        )
        cushioned_mask = np.abs(dq - dq_before_cushion) > 1e-6
        at_limit_outward = (
            ((q_cmd <= self.q_min + 0.015) & (dq_before_cushion < -1e-4))
            | ((q_cmd >= self.q_max - 0.015) & (dq_before_cushion > 1e-4))
        )
        if np.any(at_limit_outward):
            # 被限位锁死的关节不能参与逆解补偿：屏蔽对应雅可比列并重解其余自由关节，
            # 同时在限位锁死时降低姿态权重以优先保障位置不偏移、不压穿桌面软墙。
            J_masked = cmd.jacobian_cmd.copy()
            J_masked[:, at_limit_outward] = 0.0
            ori_w_eff = min(cmd.ori_weight, 0.35)
            row_w_m = np.array(
                [1.0, 1.0, 1.0, ori_w_eff, ori_w_eff, ori_w_eff], dtype=np.float64
            )
            J_pinv_m = compute_weighted_damped_pseudoinverse(
                J_masked, row_w_m, damping=max(damping, 0.025)
            )
            dq_resolved = (J_pinv_m @ np.concatenate([v_tcp, w_tcp])) * manip_scale * thermal_scale
            dq, _ = apply_joint_limit_cushion(
                q_now=q_cmd,
                joint_vel=dq_resolved,
                q_min=self.q_min,
                q_max=self.q_max,
                margin_rad=self.cfg.joint_limit_cushion_rad,
            )
            dq[at_limit_outward] = 0.0
            cushioned_mask = cushioned_mask | at_limit_outward
            limit_cushion_active = True

        if limit_cushion_active:
            reasons.append("limit_cushion")

        # ── 8. L2 分关节速度硬包络（停靠/深伸展/振荡自动收回 ≤ 2.0 rad/s）──
        deep_ext = reach_r >= 0.34
        taper_to_dock = cmd.docking or cmd.osc_active or deep_ext
        active_limits = self.dock_joint_limits if taper_to_dock else self.hard_joint_limits
        use_grouped = bool(self.cfg.joint_vel_grouped_cruise and (not taper_to_dock))

        dq_scaled = scale_joint_velocities(dq, active_limits, grouped=use_grouped)
        # 逐轴硬裁剪绝对兜底（J2/J3 永不超过 2.0 rad/s）
        dq_safe = np.clip(dq_scaled, -self.hard_joint_limits, self.hard_joint_limits)
        speed_clamped = bool(np.any(np.abs(dq_safe - dq) > 1e-5))
        if speed_clamped:
            reasons.append("speed_clamp")

        # ── 8b. 逆解后工作空间软墙最终兜底校验（确保阻尼伪逆/限位裁剪后绝不外泄穿墙）──
        v_pred = (cmd.jacobian_cmd @ dq_safe)[:3]
        wall_leak = False
        for axis in range(3):
            if tcp_p[axis] <= self.ws_min[axis] + 0.005 and v_pred[axis] < -1e-4:
                wall_leak = True
            elif tcp_p[axis] >= self.ws_max[axis] - 0.005 and v_pred[axis] > 1e-4:
                wall_leak = True
        if reach_r >= self.cfg.workspace_max_radius - 0.005:
            rad_dir = np.array([tcp_p[0], tcp_p[1]], dtype=np.float64) / max(reach_r, 1e-6)
            if float(np.dot(v_pred[:2], rad_dir)) > 1e-4:
                wall_leak = True
        if wall_leak:
            # 若仍有穿墙残余分量，直接对本拍速度归零，绝不允许跨入硬墙越界区
            dq_safe = zero6.copy()
            soft_wall_active = True
            if "soft_wall" not in reasons:
                reasons.append("soft_wall")

        # 构造送入 MIT 底层的前馈速度：受软限位缓冲或已在软限位边界的轴强制清零 dq_ff
        dq_ff_safe = dq_safe.copy()
        at_limit = (q_cmd <= self.q_min + 1e-3) | (q_cmd >= self.q_max - 1e-3)
        dq_ff_safe[cushioned_mask | at_limit] = 0.0

        # ── 9. 安全指令积分（仅积分 q_cmd，绝不混入 q_act） ────────
        q_cmd_next = np.clip(q_cmd + dq_safe * dt, self.q_min, self.q_max)

        return ShieldedCommand(
            q_cmd_next=q_cmd_next,
            dq_cmd_safe=dq_safe,
            dq_ff_safe=dq_ff_safe,
            reach_radius_m=reach_r,
            trigger_return_home=False,
            soft_wall_active=soft_wall_active,
            limit_cushion_active=limit_cushion_active,
            singularity_active=singularity_active,
            speed_clamped=speed_clamped,
            manipulability=manip,
            shield_reason=",".join(reasons),
        )
