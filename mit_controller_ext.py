#!/usr/bin/env python3
"""
reBot B601-RS MIT 底层扩展控制器（v13 · 方案 A：本仓库内子类化，上游 rebot_control 一行不改）

当前阶段：P1（P0 安全加固 + MIT 速度前馈 dq_des + 200Hz 一阶保持插值 + 看门狗）

1. S1 CAN 瞬时故障容错：
   单次/短时异常只记录并继续下一拍；连续失败持续 >= can_fault_window_s 才判定通信故障，
   并默认走「缓慢回零 → 失能」。安全停机阶段（回零）的 MIT 发送带有限次立即重试。
2. 200Hz 零额外 CAN 请求的真实反馈缓存：
   每拍发送 MIT 帧后读取 motorbridge `get_state()` 缓存（pos/vel/torq/t_mos），
   异常仅计诊断并自动降级，绝不误触发 CAN 故障。
3. P1 速度前馈 `dq_des` + 200Hz 一阶保持（FOH）连续插值 + 看门狗：
   - 新增 `set_joint_mit_targets(angles_deg, velocities_deg_s)`；老接口 `set_joint_angles()`
     100% 保持原样（自动退出速度前馈模式并令 dq_des=0，兼容回零、保持与所有标定脚本）。
   - 200Hz 连续插值 `q_interp = q_target - dq * max(0, dt_upper - elapsed)`，在任意线程相位下
     彻底消除 100Hz→200Hz 阶梯纹波，且绝不外推越过 q_target 或软限位。
   - 40ms 看门狗指数衰减（tau=30ms）：上层卡顿超时自动将 dq_des 衰减回 0。
   - 硬安全边界：J7 夹爪恒为 dq_des=0；每轴 |dq_des| 硬钳制在 max_speeds_rad_s 内；
     方向冲突自动清零；回零阶段强制 dq_des=0。
   - 一键回退：`vel_ff_enable: false` 退回 P0；`enable: false` 退回上游原版（v12 行为）。

不改动：三级温度保护、安全回零轨迹、失能/关闭流程、Esc/信号处理（含二次 Ctrl+C 行为）。
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import threading
import time
from collections import deque

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境已依赖 pyyaml
    yaml = None


# 上游 rebot_control/rebot/controller.py 的 SHA256（2026-11-23 实机所用版本）。
# 上游更新后必须重新审阅 _control_loop/_send_mit_positions 再更新此值。
PINNED_UPSTREAM_SHA256 = (
    "0d5087872eddb3ac8372ed0cb2d1aeb8dac580a47e372446e41c7913b340f845")

MIT_EXT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "mit_ext.yaml")

DEFAULT_MIT_EXT_CFG = {
    "enable": True,
    "require_pinned_upstream": True,
    "can_fault_window_s": 0.03,
    "can_fault_action": "return_to_zero",
    "can_error_log_interval_s": 1.0,
    "feedback_cache": True,
    "feedback_poll_once": False,
    "feedback_trusted": False,
    "feedback_max_age_s": 0.05,
    "record_max_s": 30.0,
    # P1：MIT 速度前馈 + 200Hz 一阶保持插值 + 看门狗
    "vel_ff_enable": False,
    "vel_ff_scale": [0.70, 0.85, 0.85, 0.90, 0.90, 0.90],
    "foh_interp_enable": True,
    "foh_upper_dt_s": 0.010,
    "vel_ff_watchdog_s": 0.040,
    "vel_ff_watchdog_tau_s": 0.030,
    "vel_ff_lpf_tau_s": 0.008,
    # P3 M2：Pinocchio + 相机附加质量静态重力补偿 tau_ff（默认关闭以保持向后兼容）
    "gravity_ff_enable": False,
    "gravity_ff_scale": [0.0, 0.35, 0.35, 0.35, 0.0, 0.0],
    "gravity_ff_max_nm": [0.0, 12.0, 12.0, 2.5, 2.0, 1.0],
    "gravity_payload_mass_kg": 0.32,
    "gravity_payload_com_link6_m": [0.056, -0.011, 0.095],
    "gravity_ff_ramp_in_s": 0.50,
    "gravity_ff_slew_nm_s": 30.0,
    # P3 M3：伸展度自适应阻尼调度 kp/kd（默认关闭以保持向后兼容）
    "gain_sched_enable": False,
    "gain_sched_r_min_m": 0.18,
    "gain_sched_r_max_m": 0.36,
    "gain_sched_kd_near": [4.6, 13.5, 14.0, 5.0, 4.0, 4.0],
    "gain_sched_kd_far": [6.4, 14.5, 15.0, 5.0, 4.0, 4.0],
    "gain_sched_kp_scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "gain_sched_slew_per_s": 10.0,
    # P4 M6：200Hz 主动安全监督层（跟踪误差碰撞/堵转保护、力矩预算保护、温升前置降速）
    "supervisor_enable": True,
    "track_err_limit_deg": 12.0,
    "track_err_window_s": 0.08,
    "tau_limit_nm": [18.0, 28.0, 28.0, 10.0, 10.0, 10.0],
    "tau_limit_window_s": 0.08,
    "thermal_derate_start_c": 90.0,
    "thermal_derate_full_c": 110.0,
    "thermal_derate_min_scale": 0.40,
}

# 代码级绝对力矩硬天花板（Nm），即使改坏 yaml 也不得突破
HARD_MAX_GRAVITY_TAU_NM = (0.0, 14.0, 14.0, 3.5, 2.5, 1.5)
# 电机峰值力矩硬件上限（RS06=36Nm, RS00=14Nm）
HARD_MOTOR_PEAK_TAU_NM = (36.0, 36.0, 36.0, 14.0, 14.0, 14.0)

_CAN_FAULT_ACTIONS = ("return_to_zero", "emergency")

# 上游类必须具备的成员（扩展依赖的私有实现细节）。
_REQUIRED_UPSTREAM_MEMBERS = (
    "_control_loop", "_send_mit_positions", "_io_lock_guard",
    "_read_positions_rad", "request_stop", "set_joint_angles",
    "get_command_angles", "set_max_speeds", "start", "stop",
)


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────
def validate_mit_ext_cfg(cfg):
    """校验并返回完整配置（缺省项补默认值）。任何越界/类型错误抛 ValueError。"""
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("mit_ext 配置必须是字典")
    unknown = set(cfg) - set(DEFAULT_MIT_EXT_CFG)
    if unknown:
        raise ValueError(f"mit_ext 存在未知配置项: {sorted(unknown)}")
    merged = dict(DEFAULT_MIT_EXT_CFG)
    merged.update(cfg)

    for key in ("enable", "require_pinned_upstream", "feedback_cache",
                "feedback_poll_once", "feedback_trusted",
                "vel_ff_enable", "foh_interp_enable",
                "gravity_ff_enable", "gain_sched_enable",
                "supervisor_enable"):
        if not isinstance(merged[key], bool):
            raise ValueError(f"mit_ext.{key} 必须是 true/false")

    def _bounded(key, lo, hi):
        value = merged[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"mit_ext.{key} 必须是数值")
        value = float(value)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"mit_ext.{key}={value} 超出安全范围 [{lo}, {hi}]")
        merged[key] = value

    def _vec(key, length, lo, hi):
        raw = merged[key]
        if not isinstance(raw, (list, tuple)) or len(raw) != length:
            raise ValueError(f"mit_ext.{key} 必须是 {length} 维列表")
        out = []
        for idx, val in enumerate(raw):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"mit_ext.{key}[{idx}] 必须是数值")
            fv = float(val)
            lo_i = lo[idx] if isinstance(lo, (list, tuple)) else lo
            hi_i = hi[idx] if isinstance(hi, (list, tuple)) else hi
            if not math.isfinite(fv) or not lo_i <= fv <= hi_i:
                raise ValueError(
                    f"mit_ext.{key}[{idx}]={fv} 超出安全范围 [{lo_i}, {hi_i}]")
            out.append(fv)
        merged[key] = out

    # 20ms 以下等同上游“一次即判死”；超过 100ms 则真实断线时保持过久。
    _bounded("can_fault_window_s", 0.01, 0.10)
    _bounded("can_error_log_interval_s", 0.1, 10.0)
    _bounded("feedback_max_age_s", 0.01, 0.20)
    _bounded("record_max_s", 1.0, 120.0)
    _bounded("foh_upper_dt_s", 0.005, 0.050)
    _bounded("vel_ff_watchdog_s", 0.015, 0.150)
    _bounded("vel_ff_watchdog_tau_s", 0.005, 0.100)
    _bounded("vel_ff_lpf_tau_s", 0.0, 0.050)
    _vec("vel_ff_scale", 6, 0.0, 1.0)

    # P3 M2 重力补偿校验
    _vec("gravity_ff_scale", 6, 0.0, [0.0, 0.95, 0.95, 0.95, 0.95, 0.95])
    _vec("gravity_ff_max_nm", 6, 0.0, HARD_MAX_GRAVITY_TAU_NM)
    _bounded("gravity_payload_mass_kg", 0.0, 1.0)
    _vec("gravity_payload_com_link6_m", 3, -0.30, 0.30)
    _bounded("gravity_ff_ramp_in_s", 0.05, 3.0)
    _bounded("gravity_ff_slew_nm_s", 2.0, 80.0)

    # P3 M3 增益调度校验
    _bounded("gain_sched_r_min_m", 0.05, 0.35)
    _bounded("gain_sched_r_max_m", 0.15, 0.45)
    if merged["gain_sched_r_max_m"] <= merged["gain_sched_r_min_m"]:
        raise ValueError("mit_ext.gain_sched_r_max_m 必须大于 gain_sched_r_min_m")
    _vec("gain_sched_kd_near", 6, 0.5, [8.0, 18.0, 18.0, 5.0, 5.0, 5.0])
    _vec("gain_sched_kd_far", 6, 0.5, [8.0, 18.0, 18.0, 5.0, 5.0, 5.0])
    _vec("gain_sched_kp_scale", 6, 0.8, 1.5)
    _bounded("gain_sched_slew_per_s", 1.0, 40.0)

    # P4 M6 主动安全监督层校验
    _bounded("track_err_limit_deg", 4.0, 25.0)
    _bounded("track_err_window_s", 0.02, 0.50)
    _vec("tau_limit_nm", 6, 2.0, HARD_MOTOR_PEAK_TAU_NM)
    _bounded("tau_limit_window_s", 0.02, 0.50)
    _bounded("thermal_derate_start_c", 50.0, 105.0)
    _bounded("thermal_derate_full_c", 60.0, 120.0)
    if merged["thermal_derate_full_c"] <= merged["thermal_derate_start_c"]:
        raise ValueError("mit_ext.thermal_derate_full_c 必须大于 thermal_derate_start_c")
    _bounded("thermal_derate_min_scale", 0.20, 1.0)

    if merged["can_fault_action"] not in _CAN_FAULT_ACTIONS:
        raise ValueError(
            f"mit_ext.can_fault_action 必须是 {_CAN_FAULT_ACTIONS} 之一")
    if merged["feedback_trusted"] and not merged["feedback_cache"]:
        raise ValueError("feedback_trusted=true 要求 feedback_cache=true")
    return merged


def load_mit_ext_cfg(path=None):
    """读取 config/mit_ext.yaml；文件不存在时使用默认值。"""
    path = path or MIT_EXT_CONFIG_PATH
    raw = {}
    if os.path.exists(path):
        if yaml is None:
            raise ImportError("读取 mit_ext.yaml 需要 pyyaml")
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} 顶层必须是字典")
        raw = data.get("mit_ext", {}) or {}
    return validate_mit_ext_cfg(raw)


# ─────────────────────────────────────────────────────────────
# 纯逻辑组件（可离线单测）
# ─────────────────────────────────────────────────────────────
class CanFaultMonitor:
    """连续发送失败计数器：时长由控制频率推导（禁止硬编码 tick 数）。"""

    def __init__(self, window_s, control_hz):
        rate = float(control_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("control_hz 必须为正数")
        self.threshold = max(2, int(round(float(window_s) * rate)))
        self.consecutive = 0
        self.max_consecutive = 0
        self.total_failures = 0
        self.total_success = 0

    def record_success(self):
        self.consecutive = 0
        self.total_success += 1

    def record_failure(self):
        """记录一次失败；返回 True 表示已达到故障判定阈值。"""
        self.consecutive += 1
        self.total_failures += 1
        self.max_consecutive = max(self.max_consecutive, self.consecutive)
        return self.consecutive >= self.threshold


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def compute_p1_mit_step(
    target, command, dq_target, dq_prev_send, max_speed, period, elapsed, *,
    vel_ff_enable, vel_ff_scale, foh_enable, foh_upper_dt_s,
    watchdog_s, watchdog_tau_s, lpf_tau_s, mit_vel_mode,
):
    """
    单关节 200Hz MIT 步进与速度前馈计算（纯函数，可独立离线单测）。
    返回 (new_command, new_dq_send, watchdog_active)。

    安全与平滑特性：
    - 非速度前馈模式 / 关闭开关时，逐行等价于上游原版限速斜坡，dq_send = 0。
    - 一阶保持（FOH）连续插值：q_interp = target - dq * max(0, upper_dt - elapsed)，
      在任意 100Hz/200Hz 线程相位下彻底消除阶梯纹波，且绝不外推越过 target/软限位。
    - 看门狗：上层超过 watchdog_s 未更新，dq 指数衰减至 0（时间常数 watchdog_tau_s）。
    - 方向冲突保护：dq 与 (target - command) 反向超过 0.2° 时，前馈速度立即归零。
    - 硬钳制：步长严格 ≤ max_speed * period，|dq_send| 严格 ≤ max_speed。
    """
    max_step = max_speed * period
    if not mit_vel_mode or not vel_ff_enable:
        error = target - command
        step = max(-max_step, min(max_step, error))
        return command + step, 0.0, False

    elapsed_s = max(0.0, float(elapsed))
    if elapsed_s <= watchdog_s:
        wd = 1.0
        wd_active = False
    else:
        wd = math.exp(-(elapsed_s - watchdog_s) / max(watchdog_tau_s, 1e-3))
        if wd < 1e-3:
            wd = 0.0
        wd_active = True

    dq_clamped = max(-max_speed, min(max_speed, float(dq_target))) * wd
    if foh_enable:
        rem_s = max(0.0, foh_upper_dt_s - elapsed_s)
        q_interp = target - dq_clamped * rem_s
    else:
        q_interp = target

    step = max(-max_step, min(max_step, q_interp - command))
    new_cmd = command + step

    error_full = target - command
    # 方向冲突、零速目标、或上层已停更超 1.5 个上层周期且已到目标：前馈清零
    if (abs(error_full) > 0.0035 and dq_clamped * error_full < 0.0):
        dq_raw = 0.0
    elif abs(dq_clamped) < 1e-4:
        dq_raw = 0.0
    elif abs(error_full) < 1e-4 and elapsed_s > 1.5 * foh_upper_dt_s:
        dq_raw = 0.0
    else:
        dq_raw = dq_clamped * float(vel_ff_scale)

    if lpf_tau_s > 0.0:
        alpha = period / (lpf_tau_s + period)
        new_dq_send = float(dq_prev_send) + alpha * (dq_raw - float(dq_prev_send))
    else:
        new_dq_send = dq_raw

    if dq_raw == 0.0 and abs(new_dq_send) < 1e-3:
        new_dq_send = 0.0
    new_dq_send = max(-max_speed, min(max_speed, new_dq_send))
    return new_cmd, new_dq_send, wd_active


def _smoothstep01(x):
    u = max(0.0, min(1.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


def cradle_zero_fade(q_rad_6):
    """
    零位支架淡出因子 [0, 1]：
    - 机械臂停靠在零位机械支撑上时（max(|J2|, |J3|) ≤ 1.7°），返回 0.0（绝不推顶支架，回零失能前平滑归零）；
    - 抬臂离开零位（≥ 7.5°，如 HOME 位姿 J2≈19.6°/J3≈19.1°），平滑过渡至 1.0 全量重力补偿。
    """
    lift = max(abs(float(q_rad_6[1])), abs(float(q_rad_6[2])))
    return _smoothstep01((lift - 0.03) / 0.10)


def compute_p3_gravity_tau(
    q_eval_6, prev_tau_6, gravity_fn, period, *,
    enabled, scales, max_nm, ramp_factor, slew_nm_s,
):
    """
    计算 6 轴静态重力补偿前馈力矩（纯函数，带斜坡淡入、零位淡出、硬限幅与变化率限制）。
    返回 (tau_out_6, error_flag)。J1 恒为 0.0 Nm。
    """
    target = [0.0] * 6
    error_flag = False
    zfade = cradle_zero_fade(q_eval_6) if len(q_eval_6) >= 6 else 0.0
    eff_ramp = max(0.0, min(1.0, float(ramp_factor))) * zfade
    if enabled and gravity_fn is not None and eff_ramp > 1e-4:
        try:
            raw = gravity_fn(q_eval_6)
            if len(raw) != 6 or not all(math.isfinite(float(v)) for v in raw):
                error_flag = True
            else:
                for i in range(6):
                    if i == 0:
                        target[0] = 0.0
                        continue
                    cap = min(float(max_nm[i]), float(HARD_MAX_GRAVITY_TAU_NM[i]))
                    val = float(raw[i]) * float(scales[i]) * eff_ramp
                    target[i] = max(-cap, min(cap, val))
        except Exception:  # noqa: BLE001
            error_flag = True

    max_step = max(1e-3, float(slew_nm_s) * float(period))
    out = [0.0] * 6
    for i in range(6):
        prev = float(prev_tau_6[i])
        delta = max(-max_step, min(max_step, target[i] - prev))
        v = prev + delta
        if target[i] == 0.0 and abs(v) < 1e-3:
            v = 0.0
        out[i] = 0.0 if i == 0 else v
    return out, error_flag


def compute_p3_gains(
    base_kp_6, base_kd_6, prev_kp_6, prev_kd_6, reach_r, period, *,
    enabled, r_min, r_max, kd_near, kd_far, kp_scale, slew_per_s,
):
    """
    计算 6 轴伸展度自适应 (kp, kd)（纯函数，带 0.8x~1.8x 硬安全区间与斜率限制）。
    返回 (kp_out_6, kd_out_6)。
    """
    s = _smoothstep01((float(reach_r) - float(r_min))
                      / max(float(r_max) - float(r_min), 1e-3))
    max_step = max(1e-3, float(slew_per_s) * float(period))
    kp_out = [0.0] * 6
    kd_out = [0.0] * 6
    for i in range(6):
        bkp = float(base_kp_6[i])
        bkd = float(base_kd_6[i])
        if enabled:
            kp_lo, kp_hi = 0.8 * bkp, 1.5 * bkp
            kd_hi_hw = 5.0 if i >= 3 else 18.0
            kd_lo, kd_hi = 0.8 * bkd, min(1.8 * bkd, kd_hi_hw)
            t_kp = max(kp_lo, min(kp_hi, bkp * float(kp_scale[i])))
            raw_kd = float(kd_near[i]) + (float(kd_far[i]) - float(kd_near[i])) * s
            t_kd = max(kd_lo, min(kd_hi, raw_kd))
        else:
            t_kp, t_kd = bkp, bkd
        pkp = float(prev_kp_6[i])
        pkd = float(prev_kd_6[i])
        kp_out[i] = pkp + max(-max_step, min(max_step, t_kp - pkp))
        kd_out[i] = pkd + max(-max_step, min(max_step, t_kd - pkd))
    return kp_out, kd_out


class SafetySupervisor:
    """
    P4 M6 200Hz 主动安全监督器（纯逻辑，可独立离线单测）：
    1. 跟踪误差碰撞/堵转检测：|q_cmd - q_act| 超限持续 >= track_err_window_s → 锁存软停机；
    2. 力矩预算堵转检测：|tau_act| 超限持续 >= tau_limit_window_s → 锁存软停机；
    3. 温升前置降速：MOS 温度在 [start_c, full_c] 区间线性下调速度倍率至 min_scale。
    绝不直接失能电机，始终以「软停机保持当前位姿 + 重力补偿维持」响应。
    """

    def __init__(self, cfg, control_hz):
        rate = float(control_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("control_hz 必须为正数")
        self.enabled = bool(cfg["supervisor_enable"])
        self.track_err_limit_deg = float(cfg["track_err_limit_deg"])
        self.track_thresh_ticks = max(2, int(round(float(cfg["track_err_window_s"]) * rate)))
        self.track_window_ms = float(cfg["track_err_window_s"]) * 1000.0
        self.tau_limit_nm = [float(v) for v in cfg["tau_limit_nm"]]
        self.tau_thresh_ticks = max(2, int(round(float(cfg["tau_limit_window_s"]) * rate)))
        self.tau_window_ms = float(cfg["tau_limit_window_s"]) * 1000.0
        self.thermal_start_c = float(cfg["thermal_derate_start_c"])
        self.thermal_full_c = float(cfg["thermal_derate_full_c"])
        self.thermal_min_scale = float(cfg["thermal_derate_min_scale"])
        self._track_over = 0
        self._tau_over = 0
        self.fault = False
        self.fault_reason = ""
        self.thermal_scale = 1.0
        self.last_track_err_deg = 0.0
        self.last_tau_ratio = 0.0
        self.last_tmos_max = math.nan

    def clear_fault(self):
        self.fault = False
        self.fault_reason = ""
        self._track_over = 0
        self._tau_over = 0

    def update(self, q_cmd_6, q_act_6, tau_act_6, t_mos_all, valid_6):
        """
        每拍更新监督状态，返回新触发标志 newly_faulted（True 表示本拍首次锁存故障）。
        """
        # 温升前置降速：只要有有效 MOS 温度即实时计算
        finite_temps = [float(t) for t in t_mos_all
                        if t is not None and math.isfinite(float(t))]
        if finite_temps:
            tmax = max(finite_temps)
            self.last_tmos_max = tmax
            if tmax <= self.thermal_start_c:
                self.thermal_scale = 1.0
            elif tmax >= self.thermal_full_c:
                self.thermal_scale = self.thermal_min_scale
            else:
                frac = ((tmax - self.thermal_start_c)
                        / max(self.thermal_full_c - self.thermal_start_c, 1e-3))
                self.thermal_scale = 1.0 - frac * (1.0 - self.thermal_min_scale)

        if not self.enabled or not all(valid_6[:6]):
            self._track_over = 0
            self._tau_over = 0
            return False

        # 1) 跟踪误差监测
        worst_err = 0.0
        worst_err_j = 1
        for i in range(6):
            err = abs(math.degrees(float(q_cmd_6[i]) - float(q_act_6[i])))
            if math.isfinite(err) and err > worst_err:
                worst_err = err
                worst_err_j = i + 1
        self.last_track_err_deg = worst_err
        if worst_err > self.track_err_limit_deg:
            self._track_over += 1
        else:
            self._track_over = 0

        # 2) 力矩预算监测
        worst_ratio = 0.0
        worst_tau = 0.0
        worst_tau_lim = 1.0
        worst_tau_j = 1
        for i in range(6):
            tv = abs(float(tau_act_6[i]))
            lim = self.tau_limit_nm[i]
            ratio = tv / max(lim, 1e-6)
            if math.isfinite(ratio) and ratio > worst_ratio:
                worst_ratio = ratio
                worst_tau = tv
                worst_tau_lim = lim
                worst_tau_j = i + 1
        self.last_tau_ratio = worst_ratio
        if worst_ratio > 1.0:
            self._tau_over += 1
        else:
            self._tau_over = 0

        if self.fault:
            return False

        if self._track_over >= self.track_thresh_ticks:
            self.fault = True
            self.fault_reason = (
                f"安全监督: J{worst_err_j} 跟踪误差 {worst_err:.1f}° > "
                f"{self.track_err_limit_deg:.1f}° (≥{self.track_window_ms:.0f}ms)")
            return True
        if self._tau_over >= self.tau_thresh_ticks:
            self.fault = True
            self.fault_reason = (
                f"安全监督: J{worst_tau_j} 力矩 {worst_tau:.1f}Nm > "
                f"{worst_tau_lim:.1f}Nm (≥{self.tau_window_ms:.0f}ms)")
            return True
        return False

    def status(self):
        return {
            "enabled": self.enabled,
            "fault": self.fault,
            "reason": self.fault_reason,
            "thermal_scale": float(self.thermal_scale),
            "track_err_max_deg": float(self.last_track_err_deg),
            "tau_max_ratio": float(self.last_tau_ratio),
            "tmos_max_c": (None if not math.isfinite(self.last_tmos_max)
                           else float(self.last_tmos_max)),
        }


def upstream_source_sha256(base_cls):
    path = inspect.getsourcefile(base_cls)
    if not path or not os.path.exists(path):
        return None, path
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest(), path


def check_upstream_compat(base_cls, pinned_sha256=PINNED_UPSTREAM_SHA256,
                          require_pinned=True):
    """返回 (ok, reason, sha256)。结构或版本不符时 ok=False，调用方须回退上游原版。"""
    missing = [name for name in _REQUIRED_UPSTREAM_MEMBERS
               if not callable(getattr(base_cls, name, None))]
    if missing:
        return False, f"上游控制器缺少成员 {missing}", None
    try:
        params = inspect.signature(base_cls._send_mit_positions).parameters
    except (TypeError, ValueError):
        return False, "无法解析上游 _send_mit_positions 签名", None
    if "lock_timeout" not in params:
        return False, "上游 _send_mit_positions 签名已变化（缺少 lock_timeout）", None
    sha, path = upstream_source_sha256(base_cls)
    if require_pinned:
        if sha is None:
            return False, f"无法读取上游源码 {path} 校验版本", None
        if sha != pinned_sha256:
            return False, (f"上游 controller.py 版本与钉死版本不一致"
                           f"（{sha[:12]}… ≠ {pinned_sha256[:12]}…），"
                           "需重新审阅后更新 PINNED_UPSTREAM_SHA256"), sha
    return True, "ok", sha


# ─────────────────────────────────────────────────────────────
# 扩展控制器工厂（运行时才绑定上游基类，便于离线以假基类测试）
# ─────────────────────────────────────────────────────────────
def make_extended_controller_class(base_cls):
    class ReBotMITExtController(base_cls):
        """上游 ReBotRSMITController 的 P4 扩展（v13 完整版：S1/S3 + 200Hz 反馈 + 速度前馈 + 重力补偿 + 阻尼调度 + 安全监督）。"""

        EXT_PHASE = "P4"
        # 连续 1s 读取缓存异常即停用缓存（时长由 control_hz 推导）。
        _FEEDBACK_ERROR_DISABLE_S = 1.0
        # 安全停机阶段单帧发送的立即重试窗口。
        _SHUTDOWN_SEND_RETRY_S = 0.02

        def __init__(self, *args, ext_cfg=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.ext_cfg = validate_mit_ext_cfg(ext_cfg)
            rate = float(self.control_hz)
            self._fault_monitor = CanFaultMonitor(
                self.ext_cfg["can_fault_window_s"], rate)
            self._supervisor = SafetySupervisor(self.ext_cfg, rate)
            self._supervisor_lock = threading.Lock()
            # P1 速度前馈状态（与 target_positions 共用 target_lock）
            self._target_velocities = [0.0 for _ in range(self.motor_count)]
            self._command_velocities = [0.0 for _ in range(self.motor_count)]
            self._mit_vel_mode = False
            self._last_mit_update_t = -math.inf
            self._watchdog_trips = 0
            self._watchdog_was_active = False
            self._last_dq_send_max = 0.0
            # P3 M2/M3 状态（重力前馈回调、斜坡淡入、伸展半径与当前生效增益/力矩）
            self._gravity_fn = None
            self._gravity_start_t = None
            self._gravity_errors = 0
            self._gravity_error_logged = False
            self._tau_ff_6 = [0.0] * 6
            self._reach_r = 0.15
            self._base_kp_6 = [float(self.config.motors[i].kp) for i in range(6)]
            self._base_kd_6 = [float(self.config.motors[i].kd) for i in range(6)]
            self._cur_kp_6 = list(self._base_kp_6)
            self._cur_kd_6 = list(self._base_kd_6)
            self._fb_lock = threading.Lock()
            self._fb_snapshot = None
            self._fb_seq = 0
            self._fb_enabled = bool(self.ext_cfg["feedback_cache"])
            self._fb_errors = 0
            self._fb_consecutive_errors = 0
            self._fb_disable_threshold = max(
                2, int(round(self._FEEDBACK_ERROR_DISABLE_S * rate)))
            self._fb_last_values = None
            self._fb_change_ticks = 0
            self._fb_ticks = 0
            self._shutdown_retries = max(
                1, int(round(self._SHUTDOWN_SEND_RETRY_S * rate)))
            window = max(16, int(round(2.0 * rate)))  # 最近 2s
            self._stats_lock = threading.Lock()
            self._period_hist = deque(maxlen=window)
            self._send_hist = deque(maxlen=window)
            self._sample_hist = deque(maxlen=window)
            self._ticks = 0
            self._last_can_log = -math.inf
            self._suppressed_can_logs = 0
            self._can_fault_triggered = False
            self._rec_lock = threading.Lock()
            self._recording = None
            self._rec_max_rows = int(round(self.ext_cfg["record_max_s"] * rate))

        # ── P1 扩展目标下发接口（旧接口 100% 兼容）────────────
        def set_joint_angles(self, angles_deg):
            """兼容老接口：纯位置模式，立即退出速度前馈模式（dq_des = 0）。"""
            with self.target_lock:
                super().set_joint_angles(angles_deg)
                self._mit_vel_mode = False
                for i in range(self.motor_count):
                    self._target_velocities[i] = 0.0

        def set_joint_angle(self, joint_id, angle_deg):
            """兼容老接口：单关节纯位置修改，清零该轴速度前馈。"""
            with self.target_lock:
                super().set_joint_angle(joint_id, angle_deg)
                self._target_velocities[joint_id - 1] = 0.0

        def set_joint_mit_targets(self, angles_deg, velocities_deg_s=None):
            """
            P1 扩展接口：同步下发 7 路目标角度（度）与速度前馈（度/秒）。
            - J7 夹爪（索引 6）强制 dq_des = 0.0，严禁透传前馈速度。
            - 任何 NaN/Inf 立即抛 ValueError，绝不进入 200Hz 循环。
            """
            if self.shutdown_started:
                return
            with self._supervisor_lock:
                if self._supervisor.fault:
                    return
            if len(angles_deg) != self.motor_count:
                raise ValueError(
                    f"Must provide {self.motor_count} angles, received {len(angles_deg)}")
            positions = []
            for idx, angle in enumerate(angles_deg):
                val = float(angle)
                if not math.isfinite(val):
                    raise ValueError(f"Invalid angle for joint {idx + 1}")
                positions.append(math.radians(val))

            if velocities_deg_s is None:
                velocities = [0.0] * self.motor_count
                vel_mode = False
            else:
                if len(velocities_deg_s) != self.motor_count:
                    raise ValueError(
                        f"Must provide {self.motor_count} velocities, "
                        f"received {len(velocities_deg_s)}")
                velocities = []
                for idx, vel in enumerate(velocities_deg_s):
                    val = float(vel)
                    if not math.isfinite(val):
                        raise ValueError(f"Invalid velocity for joint {idx + 1}")
                    velocities.append(math.radians(val))
                # 夹爪通道（最后一个电机，motor_id=7）恒为纯位置模式
                if self.motor_count >= 7:
                    velocities[6] = 0.0
                vel_mode = True

            now = time.monotonic()
            with self.target_lock:
                self.target_positions[:] = positions
                self._target_velocities[:] = velocities
                self._mit_vel_mode = vel_mode
                self._last_mit_update_t = now

        def set_vel_ff_enable(self, enabled):
            """运行时开关速度前馈（供辨识工具 A/B 对比）。"""
            with self.target_lock:
                self.ext_cfg["vel_ff_enable"] = bool(enabled)
                if not enabled:
                    for i in range(self.motor_count):
                        self._command_velocities[i] = 0.0

        def set_gravity_torque_fn(self, fn):
            """注册纯数学静态重力矩计算回调 fn(q_rad_6) -> 6 维 Nm（由 RebotAdapter 注入）。"""
            with self.target_lock:
                self._gravity_fn = fn
                self._gravity_start_t = time.monotonic() if fn is not None else None
                self._gravity_error_logged = False

        def set_gravity_ff_runtime(self, enabled=None, scale=None):
            """运行时调节重力补偿开关与分轴比例（供 p3_gravity_verify 渐进验证，严格受限）。"""
            with self.target_lock:
                if enabled is not None:
                    self.ext_cfg["gravity_ff_enable"] = bool(enabled)
                    if enabled and self._gravity_start_t is None:
                        self._gravity_start_t = time.monotonic()
                if scale is not None:
                    if len(scale) != 6:
                        raise ValueError("gravity_ff_scale 必须为 6 维")
                    clean = []
                    for i, v in enumerate(scale):
                        fv = float(v)
                        hi = 0.0 if i == 0 else 0.95
                        if not math.isfinite(fv) or not 0.0 <= fv <= hi:
                            raise ValueError(f"gravity_ff_scale[{i}]={fv} 越界 [0, {hi}]")
                        clean.append(fv)
                    self.ext_cfg["gravity_ff_scale"] = clean

        def set_gain_sched_enable(self, enabled):
            """运行时开关伸展阻尼调度（变化经斜率限速平滑过渡，无阶跃）。"""
            with self.target_lock:
                self.ext_cfg["gain_sched_enable"] = bool(enabled)

        def set_reach_radius(self, reach_r_m):
            """更新当前水平伸展半径 r (m)，驱动 200Hz 平滑阻尼调度。"""
            val = float(reach_r_m)
            if math.isfinite(val) and 0.0 <= val <= 0.60:
                with self.target_lock:
                    self._reach_r = val

        def _compute_p3_step_locked(self, commands, now_mono, period, shutdown_mode=False):
            """在 target_lock 内计算本拍 6 轴重力前馈力矩与 (kp, kd) 增益。"""
            # 1) 重力补偿：运行期优先取新鲜真实 q_act；安全回零期控制线程已停，
            #    必须直接用回零 smoothstep 轨迹 commands[:6]，确保靠零位时 100% 平滑淡出至 0。
            q_eval = list(commands[:6])
            if not shutdown_mode:
                with self._fb_lock:
                    snap = self._fb_snapshot
                if (snap is not None and all(snap["valid"][:6])
                        and (now_mono - snap["time"]) <= self.ext_cfg["feedback_max_age_s"]):
                    q_eval = list(snap["q"][:6])
            if self._gravity_start_t is None:
                ramp = 0.0
            else:
                ramp = min(1.0, max(0.0, now_mono - self._gravity_start_t)
                           / max(float(self.ext_cfg["gravity_ff_ramp_in_s"]), 1e-3))
            tau_6, g_err = compute_p3_gravity_tau(
                q_eval, self._tau_ff_6, self._gravity_fn, period,
                enabled=bool(self.ext_cfg["gravity_ff_enable"]),
                scales=self.ext_cfg["gravity_ff_scale"],
                max_nm=self.ext_cfg["gravity_ff_max_nm"],
                ramp_factor=ramp,
                slew_nm_s=self.ext_cfg["gravity_ff_slew_nm_s"],
            )
            self._tau_ff_6 = tau_6
            if g_err:
                self._gravity_errors += 1
                if not self._gravity_error_logged:
                    self._gravity_error_logged = True
                    print("\n[MIT扩展] ⚠️ 重力补偿回调异常，已平滑归零 tau_ff（不影响位置控制）")

            # 2) 增益调度：安全停机/回零阶段平滑回基准增益
            kp_6, kd_6 = compute_p3_gains(
                self._base_kp_6, self._base_kd_6,
                self._cur_kp_6, self._cur_kd_6,
                self._reach_r, period,
                enabled=(bool(self.ext_cfg["gain_sched_enable"]) and not shutdown_mode),
                r_min=self.ext_cfg["gain_sched_r_min_m"],
                r_max=self.ext_cfg["gain_sched_r_max_m"],
                kd_near=self.ext_cfg["gain_sched_kd_near"],
                kd_far=self.ext_cfg["gain_sched_kd_far"],
                kp_scale=self.ext_cfg["gain_sched_kp_scale"],
                slew_per_s=self.ext_cfg["gain_sched_slew_per_s"],
            )
            self._cur_kp_6 = kp_6
            self._cur_kd_6 = kd_6
            return list(tau_6), list(kp_6), list(kd_6)

        # ── 200Hz MIT 主循环（P3：FOH + 速度前馈 + 重力补偿 + 阻尼调度 + S1 容错）──
        def _control_loop(self):
            period = 1.0 / self.control_hz
            next_tick = time.perf_counter()
            last_tick = None

            while not self.worker_stop_event.is_set():
                tick_start = time.perf_counter()
                now_mono = time.monotonic()
                with self.target_lock:
                    vel_ff_on = bool(self.ext_cfg["vel_ff_enable"])
                    foh_on = bool(self.ext_cfg["foh_interp_enable"])
                    foh_dt = float(self.ext_cfg["foh_upper_dt_s"])
                    wd_s = float(self.ext_cfg["vel_ff_watchdog_s"])
                    wd_tau = float(self.ext_cfg["vel_ff_watchdog_tau_s"])
                    lpf_tau = float(self.ext_cfg["vel_ff_lpf_tau_s"])
                    scales = self.ext_cfg["vel_ff_scale"]
                    elapsed = (now_mono - self._last_mit_update_t
                               if math.isfinite(self._last_mit_update_t) else math.inf)
                    any_wd = False
                    for index in range(self.motor_count):
                        is_arm_joint = (index < 6)
                        new_cmd, new_dq, wd_act = compute_p1_mit_step(
                            self.target_positions[index],
                            self.command_positions[index],
                            self._target_velocities[index] if is_arm_joint else 0.0,
                            self._command_velocities[index] if is_arm_joint else 0.0,
                            self.max_speeds_rad_s[index],
                            period,
                            elapsed,
                            vel_ff_enable=(vel_ff_on and is_arm_joint),
                            vel_ff_scale=(scales[index] if is_arm_joint else 0.0),
                            foh_enable=(foh_on and is_arm_joint),
                            foh_upper_dt_s=foh_dt,
                            watchdog_s=wd_s,
                            watchdog_tau_s=wd_tau,
                            lpf_tau_s=lpf_tau,
                            mit_vel_mode=(self._mit_vel_mode and is_arm_joint),
                        )
                        self.command_positions[index] = new_cmd
                        self._command_velocities[index] = new_dq if is_arm_joint else 0.0
                        any_wd = any_wd or wd_act
                    if any_wd and not self._watchdog_was_active:
                        self._watchdog_trips += 1
                    self._watchdog_was_active = any_wd
                    commands = self.command_positions.copy()
                    velocities = self._command_velocities.copy()
                    self._last_dq_send_max = (
                        max(abs(v) for v in velocities[:6]) if len(velocities) >= 6 else 0.0)
                    tau_6, kp_6, kd_6 = self._compute_p3_step_locked(
                        commands, now_mono, period, shutdown_mode=False)

                try:
                    send_s, sample_s = self._send_and_sample(
                        commands, velocities, tau_6=tau_6, kp_6=kp_6, kd_6=kd_6)
                except Exception as error:  # noqa: BLE001 - 任何 CAN 异常都需计数
                    self.last_error = error
                    fault = self._fault_monitor.record_failure()
                    self._log_can_error(error, fault)
                    if fault:
                        self._trigger_can_fault(error)
                        return
                else:
                    self._fault_monitor.record_success()
                    with self._stats_lock:
                        self._send_hist.append(send_s)
                        if sample_s is not None:
                            self._sample_hist.append(sample_s)

                with self._stats_lock:
                    if last_tick is not None:
                        self._period_hist.append(tick_start - last_tick)
                    self._ticks += 1
                last_tick = tick_start

                next_tick += period
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

        def _send_mit_custom_locked(self, commands, velocities, tau_6, kp_6, kd_6):
            for index, motor in enumerate(self.motors):
                mc = self.config.motors[index]
                if index < 6:
                    dq_i = float(velocities[index]) if velocities is not None else 0.0
                    kp_i = float(kp_6[index]) if kp_6 is not None else float(mc.kp)
                    kd_i = float(kd_6[index]) if kd_6 is not None else float(mc.kd)
                    tau_i = float(tau_6[index]) if tau_6 is not None else 0.0
                else:
                    # J7 夹爪恒为纯位置基线 MIT，严禁注入动力学前馈或改增益
                    dq_i, kp_i, kd_i, tau_i = 0.0, float(mc.kp), float(mc.kd), 0.0
                motor.send_mit(float(commands[index]), dq_i, kp_i, kd_i, tau_i)

        def _needs_custom_mit(self, velocities, tau_6, kp_6, kd_6):
            if velocities is not None and any(v != 0.0 for v in velocities[:6]):
                return True
            if tau_6 is not None and any(t != 0.0 for t in tau_6):
                return True
            if kp_6 is not None and any(abs(kp_6[i] - self._base_kp_6[i]) > 1e-6 for i in range(6)):
                return True
            if kd_6 is not None and any(abs(kd_6[i] - self._base_kd_6[i]) > 1e-6 for i in range(6)):
                return True
            return False

        def _send_and_sample(self, commands, velocities=None, tau_6=None, kp_6=None, kd_6=None):
            """在同一把 io_lock 内发送 MIT 帧并读取应答缓存（不发额外 CAN 请求）。"""
            with self._io_lock_guard():
                t0 = time.perf_counter()
                if self._needs_custom_mit(velocities, tau_6, kp_6, kd_6):
                    self._send_mit_custom_locked(commands, velocities, tau_6, kp_6, kd_6)
                else:
                    base_cls._send_mit_positions(self, commands)
                t1 = time.perf_counter()
                sample_s = None
                if self._fb_enabled:
                    self._sample_feedback_locked(commands)
                    sample_s = time.perf_counter() - t1
            return t1 - t0, sample_s

        def _send_mit_positions(self, positions_rad, *, lock_timeout=None):
            """安全停机阶段（回零）：强制 dq_des=0、基线 kp/kd，平滑保持重力补偿并在近零位自动淡出至 0，带瞬时重试。"""
            if not self.shutdown_started:
                return base_cls._send_mit_positions(
                    self, positions_rad, lock_timeout=lock_timeout)
            period = 1.0 / self.control_hz
            now_mono = time.monotonic()
            with self.target_lock:
                tau_6, kp_6, kd_6 = self._compute_p3_step_locked(
                    positions_rad, now_mono, period, shutdown_mode=True)
            use_custom = self._needs_custom_mit(None, tau_6, kp_6, kd_6)
            last_error = None
            for _attempt in range(self._shutdown_retries):
                try:
                    if use_custom:
                        with self._io_lock_guard(lock_timeout):
                            self._send_mit_custom_locked(
                                positions_rad, None, tau_6, kp_6, kd_6)
                        return None
                    return base_cls._send_mit_positions(
                        self, positions_rad, lock_timeout=lock_timeout)
                except TimeoutError:
                    raise  # io_lock 超时由上游回零逻辑处理
                except Exception as error:  # noqa: BLE001
                    last_error = error
                    time.sleep(0.001)
            raise last_error

        # ── 反馈缓存 ─────────────────────────────────────────
        def _sample_feedback_locked(self, commands):
            if self.ext_cfg.get("feedback_poll_once", False) and self.controller is not None:
                try:
                    self.controller.poll_feedback_once()
                except Exception:  # noqa: BLE001 - 仅记录，不触发 CAN 故障
                    pass
            n = len(self.motors)
            q = [math.nan] * n
            dq = [math.nan] * n
            tau = [math.nan] * n
            t_mos = [math.nan] * n
            t_rotor = [math.nan] * n
            status = [-1] * n
            valid = [False] * n
            error_seen = False
            for index, motor in enumerate(self.motors):
                try:
                    state = motor.get_state()
                except Exception:  # noqa: BLE001
                    error_seen = True
                    continue
                if state is None:
                    continue
                try:
                    values = (float(state.pos), float(state.vel),
                              float(state.torq), float(state.t_mos),
                              float(getattr(state, "t_rotor", math.nan)))
                except (TypeError, ValueError, AttributeError):
                    error_seen = True
                    continue
                if not all(math.isfinite(v) for v in values[:3]):
                    continue
                q[index], dq[index], tau[index], t_mos[index], t_rotor[index] = values
                status[index] = int(getattr(state, "status_code", -1))
                valid[index] = True

            if error_seen:
                self._fb_errors += 1
                self._fb_consecutive_errors += 1
                if self._fb_consecutive_errors >= self._fb_disable_threshold:
                    self._fb_enabled = False
                    print("\n[MIT扩展] 反馈缓存连续读取异常超过 1s，已自动停用"
                          "（不影响 MIT 控制与温度保护）")
            else:
                self._fb_consecutive_errors = 0

            current = (tuple(q), tuple(dq), tuple(tau))
            self._fb_ticks += 1
            if self._fb_last_values is not None and current != self._fb_last_values:
                self._fb_change_ticks += 1
            self._fb_last_values = current

            now = time.monotonic()
            self._fb_seq += 1
            snapshot = {
                "time": now, "seq": self._fb_seq,
                "q": q, "dq": dq, "tau": tau, "t_mos": t_mos,
                "t_rotor": t_rotor, "status": status, "valid": valid,
                "cmd": list(commands),
            }
            with self._fb_lock:
                self._fb_snapshot = snapshot
            # P4 M6：200Hz 主动安全监督（碰撞/堵转/温升降速）；触发时软停机保持，绝不失能
            if not self.shutdown_started:
                with self._supervisor_lock:
                    newly_faulted = self._supervisor.update(
                        commands[:6], q[:6], tau[:6], t_mos, valid[:6])
                    fault_reason = self._supervisor.fault_reason
                if newly_faulted:
                    with self.target_lock:
                        for idx in range(min(6, self.motor_count)):
                            self.target_positions[idx] = self.command_positions[idx]
                            self._target_velocities[idx] = 0.0
                            self._command_velocities[idx] = 0.0
                        self._mit_vel_mode = False
                    print(f"\n[MIT扩展] ⚠️ {fault_reason} —— 已立即软停机保持当前位姿（重力补偿维持，按 g 可恢复）")
            if self._recording is not None:
                with self._rec_lock:
                    rec = self._recording
                    if rec is not None and len(rec) < self._rec_max_rows:
                        rec.append((now, list(commands), q, dq, tau))

        def get_feedback_snapshot(self):
            """最近一拍反馈缓存的拷贝（弧度/弧度每秒/Nm，7 路含夹爪）；无数据返回 None。"""
            with self._fb_lock:
                snap = self._fb_snapshot
            if snap is None:
                return None
            out = dict(snap)
            for key in ("q", "dq", "tau", "t_mos", "t_rotor", "status",
                        "valid", "cmd"):
                out[key] = list(snap[key])
            out["age_s"] = time.monotonic() - snap["time"]
            return out

        def get_safety_status(self):
            """返回 P4 M6 安全监督器当前状态快照（无锁阻塞）。"""
            with self._supervisor_lock:
                return self._supervisor.status()

        def clear_safety_fault(self):
            """清除安全监督故障锁存（由用户检查安全并重新按 g 启动跟随时调用）。"""
            with self._supervisor_lock:
                self._supervisor.clear_fault()

        def set_feedback_poll_once(self, enabled):
            """运行时切换是否在 MIT 发送后调用 poll_feedback_once()（供探针 A/B 对比）。"""
            self.ext_cfg["feedback_poll_once"] = bool(enabled)

        def reset_feedback_change_stats(self):
            """重置反馈刷新率与计时窗口，用于探针分阶段采样。"""
            with self._stats_lock:
                self._period_hist.clear()
                self._send_hist.clear()
                self._sample_hist.clear()
            self._fb_last_values = None
            self._fb_change_ticks = 0
            self._fb_ticks = 0

        # ── 录波 ─────────────────────────────────────────────
        def start_recording(self):
            with self._rec_lock:
                self._recording = []

        def stop_recording(self):
            """返回录波行列表：(t_monotonic, cmd[7], q[7], dq[7], tau[7])。"""
            with self._rec_lock:
                rows = self._recording or []
                self._recording = None
            return rows

        # ── 统计与故障处理 ───────────────────────────────────
        def get_loop_stats(self):
            with self._stats_lock:
                periods = list(self._period_hist)
                sends = list(self._send_hist)
                samples = list(self._sample_hist)
                ticks = self._ticks
            ms = 1000.0

            def _p(values, q):
                value = _percentile(values, q)
                return None if value is None else value * ms

            return {
                "phase": self.EXT_PHASE,
                "ticks": ticks,
                "period_p50_ms": _p(periods, 0.50),
                "period_p99_ms": _p(periods, 0.99),
                "period_max_ms": (max(periods) * ms) if periods else None,
                "send_p99_ms": _p(sends, 0.99),
                "sample_p99_ms": _p(samples, 0.99),
                "can_fail_total": self._fault_monitor.total_failures,
                "can_fail_max_consecutive": self._fault_monitor.max_consecutive,
                "can_fault_threshold_ticks": self._fault_monitor.threshold,
                "can_fault_triggered": self._can_fault_triggered,
                "feedback_enabled": self._fb_enabled,
                "feedback_errors": self._fb_errors,
                "feedback_change_ratio": (self._fb_change_ticks / max(1, self._fb_ticks - 1)
                                          if self._fb_ticks > 1 else None),
                "vel_ff_enable": bool(self.ext_cfg["vel_ff_enable"]),
                "dq_send_max_rad_s": float(self._last_dq_send_max),
                "watchdog_trips": int(self._watchdog_trips),
                "gravity_ff_enable": bool(self.ext_cfg["gravity_ff_enable"]),
                "gravity_errors": int(self._gravity_errors),
                "tau_ff_nm": list(self._tau_ff_6),
                "gain_sched_enable": bool(self.ext_cfg["gain_sched_enable"]),
                "cur_kp": list(self._cur_kp_6),
                "cur_kd": list(self._cur_kd_6),
                "safety": self.get_safety_status(),
            }

        def _log_can_error(self, error, fault):
            now = time.monotonic()
            interval = self.ext_cfg["can_error_log_interval_s"]
            if fault or now - self._last_can_log >= interval:
                extra = (f"（期间另有 {self._suppressed_can_logs} 次未打印）"
                         if self._suppressed_can_logs else "")
                print(f"\n[MIT扩展] CAN 发送异常 连续{self._fault_monitor.consecutive}/"
                      f"{self._fault_monitor.threshold} 次: {error}{extra}")
                self._last_can_log = now
                self._suppressed_can_logs = 0
            else:
                self._suppressed_can_logs += 1

        def _trigger_can_fault(self, error):
            self._can_fault_triggered = True
            action = self.ext_cfg["can_fault_action"]
            window_ms = self._fault_monitor.threshold * 1000.0 / self.control_hz
            print(f"\n[MIT扩展] CAN 通信连续失败 ≥{window_ms:.0f}ms，判定通信故障；"
                  + ("尝试缓慢回零后失能" if action == "return_to_zero"
                     else "按上游原逻辑立即失能"))
            self.request_stop(
                reason=(f"CAN communication fault ({self._fault_monitor.consecutive} "
                        f"consecutive failures): {error} / CAN 通信连续失败"),
                return_to_zero=(action == "return_to_zero"),
                emergency=(action == "emergency"),
                wait=False,
            )

    ReBotMITExtController.__name__ = "ReBotMITExtController"
    ReBotMITExtController.__qualname__ = "ReBotMITExtController"
    return ReBotMITExtController


def build_controller(base_cls, cfg=None, cfg_path=None):
    """
    构造控制器实例：扩展可用时返回扩展版，否则返回上游原版（v12 行为）。
    返回 (controller, info)。info 记录是否启用扩展及原因，供日志与工具判断。
    """
    if cfg is None:
        cfg = load_mit_ext_cfg(cfg_path)
    else:
        cfg = validate_mit_ext_cfg(cfg)
    info = {"active": False, "phase": None, "reason": "", "upstream_sha256": None,
            "cfg": dict(cfg)}
    if not cfg["enable"]:
        info["reason"] = "mit_ext.enable=false"
        print("[MIT扩展] 已在配置中关闭，使用上游原版控制器（v12 行为）")
        return base_cls(), info
    ok, reason, sha = check_upstream_compat(
        base_cls, require_pinned=cfg["require_pinned_upstream"])
    info["upstream_sha256"] = sha
    if not ok:
        info["reason"] = reason
        print("\n" + "!" * 70)
        print(f"[MIT扩展] ⚠️ 未启用：{reason}")
        print("[MIT扩展] 已回退上游原版控制器（v12 行为），CAN 容错与 200Hz 反馈缓存不可用")
        print("!" * 70 + "\n")
        return base_cls(), info
    cls = make_extended_controller_class(base_cls)
    controller = cls(ext_cfg=cfg)
    info.update(active=True, phase=cls.EXT_PHASE, reason="ok")
    print(f"[MIT扩展] 已启用 {cls.EXT_PHASE}："
          f"CAN 容错 {cfg['can_fault_window_s'] * 1000:.0f}ms → {cfg['can_fault_action']}，"
          f"反馈缓存={'开' if cfg['feedback_cache'] else '关'}(trusted={cfg['feedback_trusted']})，"
          f"速度前馈={'开' if cfg['vel_ff_enable'] else '关'}，"
          f"重力补偿={'开' if cfg['gravity_ff_enable'] else '关'}"
          + (f"(scale={cfg['gravity_ff_scale']}, payload={cfg['gravity_payload_mass_kg']}kg)"
             if cfg["gravity_ff_enable"] else "")
          + f"，阻尼调度={'开' if cfg['gain_sched_enable'] else '关'}")
    return controller, info
