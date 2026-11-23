"""
v15.0 MuJoCo → 真机双向遥操作安全控制器（servo_v14/teleop_bridge.py）
=======================================================================
负责将 MuJoCo Studio 发来的关节目标 `q_des`（来自关节滑块或 3D 拖拽目标球 IK）
安全、平滑地转化为 100Hz MIT 关节位置+速度前馈指令，供真机或仿真后端执行。

强制 5 道不可绕过的硬件安全防线：
  1. **显式解锁 + 姿态对齐防跳变门控**：
     从 DISARMED 切换为 ARMED 时，要求 `max_i |q_des[i] - q_act[i]| <= align_tol_deg`
     （默认 8.0°），否则拒绝解锁并报告 `ALIGN_REQUIRED`，杜绝误解锁瞬间甩臂。
  2. **分轴速度包络 + 平滑加减速斜坡（无超调着陆）**：
     硬包络 `HARD_TELEOP_JOINT_SPEED = (1.5, 1.2, 1.2, 1.8, 1.8, 1.5) rad/s`
     （远严于整机红线，承重 J2/J3 严格 ≤ 1.2 rad/s），叠加用户限速滑块
     `[5°, 60°]/s`、热降额 `thermal_speed_scale`、加速度斜坡与 `sqrt(2·a·|e|)` 制动曲线。
  3. **URDF 软限位 + 0.42m 水平柱面软/硬墙 + 桌面 Z 安全净空**：
     每拍执行前先用 `backend.fk_only(q_next)` 预检末端位置：
       - `hypot(X, Y) > 0.42m` 且向外推进 → 拦截外推，允许回缩；
       - `Z < 0.05m` 且向下推进 → 拦截撞桌，允许抬升；
       - 实际位姿越出硬墙 `r > 0.47m` → 触发 `trigger_return_home` 缓慢回 HOME。
  4. **250ms 失联看门狗 + 200Hz M6 安全监督联锁**：
     超时无心跳或底层 M6 锁存立即解除武装、速度归零并 `hold()`（绝不半空失能）。
  5. **单 CAN 主机夹爪限速通道**：
     J7 夹爪经死区与速率限幅后通过 `backend.set_gripper_deg()` 下发。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .robot_backend import RobotBackend
from .types import RobotState, ShieldedCommand

# 遥操作专属关节硬限速 (rad/s)：承重 J2/J3 严守 <= 1.2 rad/s，显著低于自动跟随极限 [2.5,2.0,2.0,3.0,3.0,2.5]
HARD_TELEOP_JOINT_SPEED = (1.5, 1.2, 1.2, 1.8, 1.8, 1.5)

# Seeed reBot Arm B601-RS URDF 关节硬限位 (rad)
B601_JOINT_LOWER = (-2.80, -0.15, -0.25, -1.75, -1.57, -2.80)
B601_JOINT_UPPER = (2.80, 3.05, 3.05, 1.75, 1.57, 2.80)


@dataclass(frozen=True)
class SafeTeleopConfig:
    """v15.0 MuJoCo 双向遥操作安全参数（不可变配置）。"""

    align_tol_deg: float = 8.0                 # 解锁时允许的虚实最大单轴偏差 (°)
    default_max_speed_deg_s: float = 25.0      # 默认单轴最高跟随速度 (°/s)
    hard_max_speed_deg_s: float = 60.0         # 允许面板调高的绝对上限 (°/s, ~1.05 rad/s)
    max_joint_accel_rad_s2: float = 2.5        # 单轴平滑加/减速上限 (rad/s²)
    joint_limit_margin_rad: float = 0.03       # 距离 URDF 限位的软件缓冲边距 (rad)
    workspace_max_radius_m: float = 0.42       # 70% 臂展水平柱面软墙红线 (m)
    workspace_hard_overshoot_m: float = 0.05   # 超出软墙 >5cm 自动安全回 HOME
    workspace_z_min_m: float = 0.05            # 防撞桌面最小 TCP 高度 (m)
    workspace_z_max_m: float = 0.65            # 最大 TCP 高度 (m)
    vel_ff_scale: float = 0.85                 # 送入底层 MIT 的速度前馈缩放比例
    gripper_min_deg: float = 5.2               # J7 闭合标定角 (deg)
    gripper_max_deg: float = 285.2             # J7 全开标定角 (deg)
    gripper_deadband_deg: float = 1.5          # 夹爪防微抖死区 (deg)
    gripper_max_rate_deg_s: float = 240.0      # 夹爪平滑速率限制 (deg/s)

    def validate(self) -> "SafeTeleopConfig":
        if not (0.5 <= self.align_tol_deg <= 20.0):
            raise ValueError(f"align_tol_deg 超出安全范围 [0.5, 20.0]: {self.align_tol_deg}")
        if not (5.0 <= self.hard_max_speed_deg_s <= 60.0):
            raise ValueError(f"hard_max_speed_deg_s 超出安全范围 [5, 60]: {self.hard_max_speed_deg_s}")
        if not (0.10 <= self.workspace_max_radius_m <= 0.42):
            raise ValueError(f"workspace_max_radius_m 违反 70% 臂展红线 (≤0.42m): {self.workspace_max_radius_m}")
        if self.workspace_z_min_m < 0.02:
            raise ValueError(f"workspace_z_min_m 过低存在撞桌风险: {self.workspace_z_min_m}")
        return self


@dataclass(frozen=True)
class TeleopCommand:
    """从 MuJoCo Studio 通道 B 解码后的单帧遥操作目标。"""

    seq: int
    armed: bool
    mode: str                                  # JOINT | IK_3D | HOLD
    q_des: np.ndarray                          # (6,) rad
    gripper_deg: float = 285.2
    max_speed_deg_s: float = 25.0
    clear_fault: bool = False


@dataclass(frozen=True)
class TeleopStepResult:
    """单拍 100Hz 遥操作安全控制器输出。"""

    armed: bool
    mode: str                                  # DISARMED | TELEOP_JOINT | TELEOP_IK | HOLD | FAULT_HOLD | RETURN_HOME
    reason: str                                # 状态/拦截原因（OK / WATCHDOG_TIMEOUT / ALIGN_REQUIRED / SOFT_WALL / Z_FLOOR / M6_FAULT）
    q_cmd_next: np.ndarray                     # (6,) rad
    dq_cmd_safe: np.ndarray                    # (6,) rad/s
    dq_ff_safe: np.ndarray                     # (6,) rad/s
    reach_radius_m: float
    effective_speed_deg_s: float
    align_err_deg: float
    tracking_err_max_deg: float
    soft_wall_active: bool = False
    z_guard_active: bool = False
    limit_clamped: bool = False
    trigger_return_home: bool = False
    gripper_deg_cmd: Optional[float] = None
    cmd_seq_ack: int = 0


def parse_teleop_packet(pkt: Optional[dict]) -> Optional[TeleopCommand]:
    """将 UDP 字典安全解析为强类型 `TeleopCommand`，异常/NaN 返回 None。"""
    if pkt is None or not isinstance(pkt, dict):
        return None
    q_raw = pkt.get("q_des")
    if not isinstance(q_raw, (list, tuple)) or len(q_raw) != 6:
        return None
    q_arr = np.asarray(q_raw, dtype=np.float64)
    if not np.all(np.isfinite(q_arr)):
        return None
    g_deg = float(pkt.get("gripper_deg", 285.2))
    spd = float(pkt.get("max_speed_deg_s", 25.0))
    if not (math.isfinite(g_deg) and math.isfinite(spd)):
        return None
    return TeleopCommand(
        seq=int(pkt.get("seq", 0)),
        armed=bool(pkt.get("armed", False)),
        mode=str(pkt.get("mode", "HOLD")),
        q_des=q_arr,
        gripper_deg=g_deg,
        max_speed_deg_s=spd,
        clear_fault=bool(pkt.get("clear_fault", False)),
    )


class SafeTeleopController:
    """v15.0 100Hz MuJoCo → 真机安全遥操作控制器。"""

    def __init__(
        self,
        cfg: SafeTeleopConfig,
        backend: RobotBackend,
        *,
        enable_gripper: bool = True,
    ) -> None:
        self.cfg = cfg.validate()
        self.backend = backend
        self.enable_gripper = bool(enable_gripper)

        self._q_min = np.asarray(backend.q_min, dtype=np.float64) + self.cfg.joint_limit_margin_rad
        self._q_max = np.asarray(backend.q_max, dtype=np.float64) - self.cfg.joint_limit_margin_rad
        self._hard_vel = np.asarray(HARD_TELEOP_JOINT_SPEED, dtype=np.float64)

        self._armed: bool = False
        self._dq_prev = np.zeros(6, dtype=np.float64)
        self._last_gripper_deg: Optional[float] = None
        self._last_cmd_seq: int = 0
        self._last_clear_fault: bool = False

    def reset(self) -> None:
        """强制解除武装并清零速度斜坡记忆。"""
        self._armed = False
        self._dq_prev[:] = 0.0

    @property
    def is_armed(self) -> bool:
        return self._armed

    def _clamp_joints(self, q: np.ndarray) -> Tuple[np.ndarray, bool]:
        q_c = np.clip(q, self._q_min, self._q_max)
        clamped = bool(np.any(np.abs(q_c - q) > 1e-9))
        return q_c, clamped

    def _update_gripper(self, target_deg: float, dt_s: float) -> Optional[float]:
        if not self.enable_gripper:
            return None
        tgt = float(np.clip(target_deg, self.cfg.gripper_min_deg, self.cfg.gripper_max_deg))
        cur = self._last_gripper_deg
        if cur is None:
            cur = float(self.backend.get_gripper_deg())
            self._last_gripper_deg = cur
        diff = tgt - cur
        if abs(diff) < self.cfg.gripper_deadband_deg:
            return None
        max_step = self.cfg.gripper_max_rate_deg_s * max(1e-4, dt_s)
        nxt = cur + float(np.clip(diff, -max_step, max_step))
        self._last_gripper_deg = nxt
        return nxt

    def step(
        self,
        cmd: Optional[TeleopCommand],
        robot_state: RobotState,
        dt_s: float,
        *,
        apply_to_backend: bool = True,
    ) -> TeleopStepResult:
        """执行单拍 100Hz 安全遥操作闭环。"""
        dt = max(1e-4, min(0.05, float(dt_s)))
        q_cmd = np.asarray(robot_state.q_cmd, dtype=np.float64)
        q_act = np.asarray(robot_state.q_act, dtype=np.float64)
        tcp_curr, _ = self.backend.fk_only(q_cmd)
        r_curr = float(math.hypot(float(tcp_curr[0]), float(tcp_curr[1])))
        z_curr = float(tcp_curr[2])
        track_err_deg = float(np.max(np.abs(np.degrees(q_cmd - q_act))))

        # ── 1) 处理一次性的 M6 故障清除脉冲（仅非武装时生效） ─────────
        if cmd is not None and cmd.clear_fault and not self._last_clear_fault and not cmd.armed:
            self.backend.clear_safety_fault()
        self._last_clear_fault = bool(cmd.clear_fault) if cmd is not None else False

        # ── 2) 硬工作空间越界 >5cm 最高优先级抢占 ─────────────────────
        if r_curr > self.cfg.workspace_max_radius_m + self.cfg.workspace_hard_overshoot_m:
            self._armed = False
            self._dq_prev[:] = 0.0
            if apply_to_backend:
                self.backend.hold()
            return TeleopStepResult(
                armed=False,
                mode="RETURN_HOME",
                reason=f"HARD_WALL_r={r_curr:.3f}m",
                q_cmd_next=q_cmd.copy(),
                dq_cmd_safe=np.zeros(6),
                dq_ff_safe=np.zeros(6),
                reach_radius_m=r_curr,
                effective_speed_deg_s=0.0,
                align_err_deg=0.0,
                tracking_err_max_deg=track_err_deg,
                trigger_return_home=True,
                cmd_seq_ack=self._last_cmd_seq,
            )

        # ── 3) M6 底层安全监督锁存拦截 ────────────────────────────────
        if robot_state.supervisor_latched:
            self._armed = False
            self._dq_prev[:] = 0.0
            if apply_to_backend:
                self.backend.hold()
            return TeleopStepResult(
                armed=False,
                mode="FAULT_HOLD",
                reason=f"M6_FAULT:{robot_state.supervisor_reason or 'latched'}",
                q_cmd_next=q_cmd.copy(),
                dq_cmd_safe=np.zeros(6),
                dq_ff_safe=np.zeros(6),
                reach_radius_m=r_curr,
                effective_speed_deg_s=0.0,
                align_err_deg=0.0,
                tracking_err_max_deg=track_err_deg,
                cmd_seq_ack=self._last_cmd_seq,
            )

        # ── 4) 250ms 失联看门狗 ───────────────────────────────────────
        if cmd is None:
            was_armed = self._armed
            self._armed = False
            self._dq_prev[:] = 0.0
            if apply_to_backend and was_armed:
                self.backend.hold()
            return TeleopStepResult(
                armed=False,
                mode="DISARMED",
                reason="WATCHDOG_TIMEOUT",
                q_cmd_next=q_cmd.copy(),
                dq_cmd_safe=np.zeros(6),
                dq_ff_safe=np.zeros(6),
                reach_radius_m=r_curr,
                effective_speed_deg_s=0.0,
                align_err_deg=0.0,
                tracking_err_max_deg=track_err_deg,
                cmd_seq_ack=self._last_cmd_seq,
            )

        self._last_cmd_seq = cmd.seq
        q_des_clamped, limit_clamped = self._clamp_joints(cmd.q_des)
        align_err_deg = float(np.max(np.abs(np.degrees(q_des_clamped - q_act))))

        # ── 5) 显式解锁与姿态对齐防跳变门控 ───────────────────────────
        if not cmd.armed or cmd.mode == "HOLD":
            was_armed = self._armed
            self._armed = False
            self._dq_prev[:] = 0.0
            if apply_to_backend and was_armed:
                self.backend.hold()
            return TeleopStepResult(
                armed=False,
                mode="DISARMED",
                reason="STANDBY_DISARMED",
                q_cmd_next=q_cmd.copy(),
                dq_cmd_safe=np.zeros(6),
                dq_ff_safe=np.zeros(6),
                reach_radius_m=r_curr,
                effective_speed_deg_s=0.0,
                align_err_deg=align_err_deg,
                tracking_err_max_deg=track_err_deg,
                limit_clamped=limit_clamped,
                cmd_seq_ack=cmd.seq,
            )

        if not self._armed:
            # 刚请求武装：必须满足虚实姿态偏差 <= align_tol_deg，严防瞬间甩臂
            if align_err_deg > self.cfg.align_tol_deg:
                self._dq_prev[:] = 0.0
                return TeleopStepResult(
                    armed=False,
                    mode="DISARMED",
                    reason=f"ALIGN_REQUIRED({align_err_deg:.1f}>{self.cfg.align_tol_deg:.0f}deg)",
                    q_cmd_next=q_cmd.copy(),
                    dq_cmd_safe=np.zeros(6),
                    dq_ff_safe=np.zeros(6),
                    reach_radius_m=r_curr,
                    effective_speed_deg_s=0.0,
                    align_err_deg=align_err_deg,
                    tracking_err_max_deg=track_err_deg,
                    limit_clamped=limit_clamped,
                    cmd_seq_ack=cmd.seq,
                )
            # 对齐校验通过：锚定当前位姿，正式解锁武装
            self._armed = True
            self._dq_prev[:] = 0.0
            if apply_to_backend:
                self.backend.hold()

        # ── 6) 平滑速度/加速度斜坡 + 制动曲线（无超调着陆） ────────────
        req_spd_deg_s = float(np.clip(cmd.max_speed_deg_s, 5.0, self.cfg.hard_max_speed_deg_s))
        thermal = float(np.clip(robot_state.thermal_speed_scale, 0.4, 1.0))
        eff_spd_deg_s = req_spd_deg_s * thermal
        eff_spd_rad_s = math.radians(eff_spd_deg_s)
        per_axis_vmax = np.minimum(self._hard_vel, eff_spd_rad_s)

        err = q_des_clamped - q_cmd
        a_max = self.cfg.max_joint_accel_rad_s2
        # 时间最优制动包络（留 20% 离散步长裕量）：靠近目标时平滑减速至 0，严格零超调无振荡
        v_brake = np.sqrt(2.0 * (0.80 * a_max) * np.abs(err))
        v_step = np.abs(err) / dt
        v_mag = np.minimum(per_axis_vmax, np.minimum(v_brake, v_step))
        dq_target = np.sign(err) * v_mag

        # 加速度斜坡限幅
        max_dv = a_max * dt
        dq_ramped = self._dq_prev + np.clip(dq_target - self._dq_prev, -max_dv, max_dv)
        dq_ramped = np.clip(dq_ramped, -per_axis_vmax, per_axis_vmax)

        # 微小死区内干净归零，避免静抖
        small_mask = (np.abs(err) < 5e-4) & (np.abs(dq_ramped) < 5e-3)
        dq_ramped[small_mask] = 0.0

        q_next_raw = q_cmd + dq_ramped * dt
        q_next, next_clamped = self._clamp_joints(q_next_raw)
        limit_clamped = limit_clamped or next_clamped
        dq_safe = (q_next - q_cmd) / dt

        # ── 7) 末端工作空间柱面软墙 & 桌面高度净空预检 ────────────────
        tcp_next, _ = self.backend.fk_only(q_next)
        r_next = float(math.hypot(float(tcp_next[0]), float(tcp_next[1])))
        z_next = float(tcp_next[2])

        soft_wall_active = False
        z_guard_active = False
        reason = "OK"

        # 水平柱面 0.42m 软墙：禁止向外突破，允许向内回撤
        if r_next > self.cfg.workspace_max_radius_m and r_next > r_curr + 1e-5:
            soft_wall_active = True
            q_next = q_cmd.copy()
            dq_safe = np.zeros(6, dtype=np.float64)
            reason = f"SOFT_WALL_r={r_next:.3f}m"
        # 桌面 Z 下界与上界保护：禁止向下撞桌或向上超限
        elif z_next < self.cfg.workspace_z_min_m and z_next < z_curr - 1e-5:
            z_guard_active = True
            q_next = q_cmd.copy()
            dq_safe = np.zeros(6, dtype=np.float64)
            reason = f"Z_FLOOR_z={z_next:.3f}m"
        elif z_next > self.cfg.workspace_z_max_m and z_next > z_curr + 1e-5:
            z_guard_active = True
            q_next = q_cmd.copy()
            dq_safe = np.zeros(6, dtype=np.float64)
            reason = f"Z_CEIL_z={z_next:.3f}m"

        self._dq_prev = dq_safe.copy()

        # 限位接触轴强制速度前馈为 0
        at_lim = (q_next <= self._q_min + 1e-4) | (q_next >= self._q_max - 1e-4)
        dq_ff_safe = dq_safe * self.cfg.vel_ff_scale
        dq_ff_safe[at_lim] = 0.0

        # ── 8) 下发底层 100Hz 指令 + 单 CAN 夹爪通道 ───────────────────
        g_cmd = self._update_gripper(cmd.gripper_deg, dt)
        if apply_to_backend:
            shield_cmd = ShieldedCommand(
                q_cmd_next=q_next,
                dq_cmd_safe=dq_safe,
                dq_ff_safe=dq_ff_safe,
                reach_radius_m=r_curr,
                trigger_return_home=False,
                soft_wall_active=soft_wall_active,
                limit_cushion_active=limit_clamped,
                singularity_active=False,
                speed_clamped=False,
                manipulability=1.0,
                shield_reason=reason,
            )
            self.backend.apply_command(shield_cmd, dt)
            if g_cmd is not None:
                self.backend.set_gripper_deg(g_cmd)

        out_mode = "TELEOP_IK" if cmd.mode == "IK_3D" else "TELEOP_JOINT"
        return TeleopStepResult(
            armed=True,
            mode=out_mode,
            reason=reason,
            q_cmd_next=q_next,
            dq_cmd_safe=dq_safe,
            dq_ff_safe=dq_ff_safe,
            reach_radius_m=r_curr,
            effective_speed_deg_s=eff_spd_deg_s,
            align_err_deg=align_err_deg,
            tracking_err_max_deg=track_err_deg,
            soft_wall_active=soft_wall_active,
            z_guard_active=z_guard_active,
            limit_clamped=limit_clamped,
            trigger_return_home=False,
            gripper_deg_cmd=g_cmd,
            cmd_seq_ack=cmd.seq,
        )
