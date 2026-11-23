#!/usr/bin/env python3
"""
reBot B601-RS MIT 底层扩展控制器（v13 · 方案 A：本仓库内运行时子类化，上游 rebot_control 一行不改）

设计定位（P0 ~ P4 完整版）：
    通过 `make_extended_controller_class(base_cls)` 在运行时动态继承上游
    `rebot.ReBotRSMITController`，并在加载前强制校验上游方法签名与源码 SHA256
    （`PINNED_UPSTREAM_SHA256`）。若校验失败或配置 `mit_ext.enable: false`，自动
    回退上游原版控制器（v12 行为），保证零侵入、可一键回退。

核心能力分层：
1. P0 · S1 CAN 瞬时故障容错（`CanFaultMonitor`）：
   - 单次/短时 CAN 发送异常只计数、限频打印并继续下一拍，避免单帧偶发丢包导致
     机械臂在半空直接失能坠落；
   - 仅当连续失败持续 >= `can_fault_window_s`（默认 30ms，按 `control_hz` 推导
     tick 阈值，绝不硬编码）才判定通信故障，并默认走「缓慢回零 → 失能」；
   - 安全停机回零阶段的 MIT 发送带有限次立即重试（20ms 窗口内 1ms 退避）。
2. P0 · 200Hz 零额外 CAN 请求的真实反馈缓存：
   - 每拍发送 MIT 控制帧后，在同一把 `_io_lock_guard()` 内直接读取 motorbridge
     应答帧缓存 `motor.get_state()`（pos/vel/torq/t_mos/t_rotor/status）；
   - 缓存读取异常仅计入诊断计数，连续 1s 异常后自动停用缓存，绝不误触发 CAN 故障。
3. P1 · 速度前馈 `dq_des` + 200Hz 一阶保持（FOH）插值 + 看门狗（`compute_p1_mit_step`）：
   - 新增 `set_joint_mit_targets(angles_deg, velocities_deg_s)`；老接口
     `set_joint_angles()` / `set_joint_angle()` 100% 兼容并立即退出前馈模式（`dq_des=0`）；
   - FOH 向后插值 `q_interp = target - dq * max(0, foh_upper_dt_s - elapsed)`，在任意
     100Hz/200Hz 线程相位差下消除阶梯纹波，且永不外推越过目标位或软限位；
   - 上层超过 `vel_ff_watchdog_s`（默认 40ms）未刷新时，按指数时间常数 `watchdog_tau_s`
     平滑将 `dq_des` 衰减归零；叠加一阶低通滤波与反向冲突保护；
   - 硬隔离：J7 夹爪（索引 6）以及回零/保持/标定路径恒强制 `dq_des = 0`。
4. P3 · 静态重力前馈 `tau_ff`（`compute_p3_gravity_tau`）与伸展自适应阻尼（`compute_p3_gains`）：
   - 由 `RebotAdapter` 注入隔离 Pinocchio 重力回调（含 0.32kg 相机载荷），叠加启动斜坡
     淡入、零位机械支架平滑淡出（`cradle_zero_fade`）、双重硬封顶与斜率限速（Nm/s）；
   - 安全回零期（`shutdown_mode=True`）强制用回零指令轨迹 `commands[:6]` 计算重力淡出，
     避免 200Hz 线程停止后反馈缓存过期导致近零位顶撞支架；
   - 基于水平伸展半径 `reach_r` 平滑调度 J1~J6 的 `(kp, kd)`，抑制悬臂长伸展共振。
5. P4 · 200Hz 主动安全监督层（`SafetySupervisor`）：
   - 持续监测 6 轴跟踪误差（碰撞/堵转，默认 >12° 持续 ≥80ms）与实测力矩预算
     （默认超分轴阈值持续 ≥80ms），触发后锁存「软停机保持当前位姿 + 重力维持」，
     绝不空中直接失能电机，检查安全后按 `g` 可清除锁存恢复；
   - MOS 温度进入 `[90°C, 110°C]` 区间时线性前置下调速度倍率至 `0.40`。

保持上游原样、绝不改动：
    三级过温硬件保护、安全回零 smoothstep 轨迹、失能/关闭流程、Esc/信号处理。
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
# 配置校验与加载
# ─────────────────────────────────────────────────────────────
def validate_mit_ext_cfg(cfg):
    """
    校验并补全 MIT 扩展配置字典（缺省项自动填充 `DEFAULT_MIT_EXT_CFG`）。

    约束与安全检查：
        - 禁止未知键（防止 yaml 拼写错误静默失效）；
        - 所有布尔开关强制要求精确 `bool` 类型；
        - 所有数值/向量均检查有限性（非 NaN/Inf）并落入硬件安全边界内，
          其中 `gravity_ff_max_nm` 绝不允许突破 `HARD_MAX_GRAVITY_TAU_NM`，
          `tau_limit_nm` 绝不允许突破 `HARD_MOTOR_PEAK_TAU_NM`；
        - J1 为竖直回转轴不受重力影响，强制 `gravity_ff_scale[0] == 0.0`。

    参数:
        cfg (dict | None): 来自 `config/mit_ext.yaml` 的 `mit_ext` 子字典或调用方字典；
            传 `None` 时返回全默认值拷贝。

    返回:
        dict: 类型规范化、经安全边界校验的完整配置字典。

    异常:
        ValueError: 当类型错误、存在未知项、数值为 NaN/Inf、超出安全范围或逻辑冲突
            （如 `feedback_trusted=True` 但 `feedback_cache=False`）时抛出。
    """
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
        """校验标量配置项为有限浮点数且满足 `lo <= value <= hi`（排斥 bool）。"""
        value = merged[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"mit_ext.{key} 必须是数值")
        value = float(value)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"mit_ext.{key}={value} 超出安全范围 [{lo}, {hi}]")
        merged[key] = value

    def _vec(key, length, lo, hi):
        """校验定长数值向量（支持统一标量边界或逐轴独立边界列表）。"""
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

    # P0/P1 时序与容错边界：
    # can_fault_window_s 在 20ms 以下几乎等同上游“一次即判死”，超过 100ms 则真实断线时悬停过久。
    _bounded("can_fault_window_s", 0.01, 0.10)
    _bounded("can_error_log_interval_s", 0.1, 10.0)
    _bounded("feedback_max_age_s", 0.01, 0.20)
    _bounded("record_max_s", 1.0, 120.0)
    _bounded("foh_upper_dt_s", 0.005, 0.050)
    _bounded("vel_ff_watchdog_s", 0.015, 0.150)
    _bounded("vel_ff_watchdog_tau_s", 0.005, 0.100)
    _bounded("vel_ff_lpf_tau_s", 0.0, 0.050)
    _vec("vel_ff_scale", 6, 0.0, 1.0)

    # P3 M2 重力补偿校验：J1 竖直轴强制为 0；分轴上限同时受代码绝对硬天花板约束
    _vec("gravity_ff_scale", 6, 0.0, [0.0, 0.95, 0.95, 0.95, 0.95, 0.95])
    _vec("gravity_ff_max_nm", 6, 0.0, HARD_MAX_GRAVITY_TAU_NM)
    _bounded("gravity_payload_mass_kg", 0.0, 1.0)
    _vec("gravity_payload_com_link6_m", 3, -0.30, 0.30)
    _bounded("gravity_ff_ramp_in_s", 0.05, 3.0)
    _bounded("gravity_ff_slew_nm_s", 2.0, 80.0)

    # P3 M3 增益调度校验：RS00 腕部电机（J4~J6）硬件阻尼上限 5.0，RS06（J2/J3）上限 18.0
    _bounded("gain_sched_r_min_m", 0.05, 0.35)
    _bounded("gain_sched_r_max_m", 0.15, 0.45)
    if merged["gain_sched_r_max_m"] <= merged["gain_sched_r_min_m"]:
        raise ValueError("mit_ext.gain_sched_r_max_m 必须大于 gain_sched_r_min_m")
    _vec("gain_sched_kd_near", 6, 0.5, [8.0, 18.0, 18.0, 5.0, 5.0, 5.0])
    _vec("gain_sched_kd_far", 6, 0.5, [8.0, 18.0, 18.0, 5.0, 5.0, 5.0])
    _vec("gain_sched_kp_scale", 6, 0.8, 1.5)
    _bounded("gain_sched_slew_per_s", 1.0, 40.0)

    # P4 M6 主动安全监督层校验（跟踪误差度、力矩预算 Nm、MOS 温升前置降速 °C）
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
    """
    从 YAML 文件读取并校验 `mit_ext` 配置；文件不存在时直接返回校验后的默认配置。

    参数:
        path (str | None): 配置文件路径，默认为仓库内 `config/mit_ext.yaml`。

    返回:
        dict: 经 `validate_mit_ext_cfg()` 校验后的完整配置字典。

    异常:
        ImportError: 文件存在但环境未安装 `pyyaml`。
        ValueError: YAML 顶层不是字典或内部参数未通过安全校验。
    """
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
# 纯逻辑组件（无硬件依赖，可独立离线单元测试）
# ─────────────────────────────────────────────────────────────
class CanFaultMonitor:
    """
    S1 CAN 连续发送失败监测器（纯状态机，阈值由控制频率推导，禁止硬编码 tick 数）。

    设计目的：
        替代上游原版「单次发送异常即直接紧急失能（导致半空坠臂）」的激进策略。
        偶发 CAN 总线抖动在连续成功一拍后即清零计数；仅当连续失败达到
        `max(2, round(window_s * control_hz))` 拍（例如 30ms @ 200Hz → 6 拍）时，
        才判定为持续通信故障，交由上层触发「缓慢回零 → 失能」。
    """

    def __init__(self, window_s, control_hz):
        """
        初始化故障监测器。

        参数:
            window_s (float): 判定通信故障所需的最短连续失败时长（秒）。
            control_hz (float): MIT 控制循环频率（Hz），必须为有限正数。
        """
        rate = float(control_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("control_hz 必须为正数")
        # 至少要求连续失败 2 拍，彻底杜绝单帧偶发丢包误触发停机
        self.threshold = max(2, int(round(float(window_s) * rate)))
        self.consecutive = 0
        self.max_consecutive = 0
        self.total_failures = 0
        self.total_success = 0

    def record_success(self):
        """记录一次 MIT 帧发送成功，清零当前连续失败计数。"""
        self.consecutive = 0
        self.total_success += 1

    def record_failure(self):
        """
        记录一次 MIT 帧发送失败并检查是否触及故障阈值。

        返回:
            bool: 若连续失败拍数达到或超过 `self.threshold`，返回 `True`（触发故障）；
                否则返回 `False`（按瞬时抖动容忍）。
        """
        self.consecutive += 1
        self.total_failures += 1
        self.max_consecutive = max(self.max_consecutive, self.consecutive)
        return self.consecutive >= self.threshold


def _percentile(values, q):
    """
    计算数值序列的近似分位数（最近秩法），供 200Hz 循环耗时 P50/P99 诊断使用。

    参数:
        values (Sequence[float]): 采样序列。
        q (float): 分位点，范围 `[0.0, 1.0]`（如 0.50、0.99）。

    返回:
        float | None: 对应分位数值；输入为空时返回 `None`。
    """
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
    单关节 200Hz MIT 位置步进与速度前馈 `dq_des` 计算（纯函数，可独立离线单测）。

    核心算法与安全特性：
    1. 纯位置回退：当 `mit_vel_mode=False` 或 `vel_ff_enable=False`（如回零、保持、
       老接口 `set_joint_angles()` 或 J7 夹爪）时，完全等价于上游原版限速斜坡，
       且强制返回 `dq_send = 0.0`。
    2. 看门狗指数衰减：若上层 100Hz 线程距上次下发已超时（`elapsed > watchdog_s`），
       对 `dq_target` 施加衰减因子 `exp(-(elapsed - watchdog_s) / watchdog_tau_s)`，
       低于 `1e-3` 直接截断为 `0.0`，防止上层卡死时电机持续按旧速度前冲。
    3. 一阶保持（FOH）向后连续插值：
       `q_interp = target - dq_clamped * max(0, foh_upper_dt_s - elapsed)`。
       因为 `target` 是上层面向下一个 100Hz 周期末的积分位置，向后回溯剩余时间
       可在任意 100Hz/200Hz 相位差下平滑连接两拍轨迹、消除阶梯纹波，且绝不会
       外推越过 `target` 或软限位。
    4. 方向冲突与停更保护：若前馈速度与总位置误差 `(target - command)` 反向超过
       `0.0035 rad`（≈0.2°）、目标速度极小或停更超过 `1.5 * foh_upper_dt_s` 且已到位，
       立即清零本拍原始前馈 `dq_raw = 0.0`。
    5. 一阶低通平滑与硬钳制：经时间常数 `lpf_tau_s` 平滑 `dq_send` 后，严格保证
       单拍位置步长 `<= max_speed * period` 且 `|new_dq_send| <= max_speed`。

    参数:
        target (float): 上层下发的目标关节角（rad）。
        command (float): 上一 200Hz 拍实际下发的指令关节角（rad）。
        dq_target (float): 上层下发的目标前馈角速度（rad/s）。
        dq_prev_send (float): 上一 200Hz 拍经低通滤波后的前馈角速度（rad/s）。
        max_speed (float): 该关节允许的最大角速度（rad/s，来自控制器限速表）。
        period (float): 底层 MIT 控制周期（s，200Hz 下为 0.005）。
        elapsed (float): 距上一次 `set_joint_mit_targets()` 更新的耗时（s）。
        vel_ff_enable (bool): 速度前馈总开关（J7 通道调用方恒传 False）。
        vel_ff_scale (float): 分轴速度前馈缩放系数 `[0.0, 1.0]`。
        foh_enable (bool): 是否启用一阶保持连续插值。
        foh_upper_dt_s (float): 上层控制周期标称时长（s，100Hz 下为 0.010）。
        watchdog_s (float): 速度前馈看门狗触发阈值（s，默认 0.040）。
        watchdog_tau_s (float): 看门狗触发后的指数衰减时间常数（s，默认 0.030）。
        lpf_tau_s (float): 前馈角速度一阶低通滤波时间常数（s，0 表示直通）。
        mit_vel_mode (bool): 当前是否处于速度前馈下发模式。

    返回:
        tuple[float, float, bool]: `(new_command, new_dq_send, watchdog_active)`，分别为
            本拍下发位置（rad）、本拍下发速度前馈（rad/s）及本拍看门狗是否激活。
    """
    max_step = max_speed * period
    # 非前馈模式或功能关闭：逐行保持与上游原版一致的纯位置限速斜坡，dq_des 恒为 0
    if not mit_vel_mode or not vel_ff_enable:
        error = target - command
        step = max(-max_step, min(max_step, error))
        return command + step, 0.0, False

    # 1) 看门狗判定：上层超过 watchdog_s 未刷新目标，指数衰减前馈速度至 0
    elapsed_s = max(0.0, float(elapsed))
    if elapsed_s <= watchdog_s:
        wd = 1.0
        wd_active = False
    else:
        wd = math.exp(-(elapsed_s - watchdog_s) / max(watchdog_tau_s, 1e-3))
        if wd < 1e-3:
            wd = 0.0
        wd_active = True

    # 2) 先将前馈速度硬钳制在关节限速内并乘看门狗因子，再执行 FOH 向后插值
    dq_clamped = max(-max_speed, min(max_speed, float(dq_target))) * wd
    if foh_enable:
        # 仅向后回溯未走完的上层周期时间（rem_s >= 0），绝不向前外推越过 target
        rem_s = max(0.0, foh_upper_dt_s - elapsed_s)
        q_interp = target - dq_clamped * rem_s
    else:
        q_interp = target

    # 3) 位置指令按单拍最大步长限速推进
    step = max(-max_step, min(max_step, q_interp - command))
    new_cmd = command + step

    # 4) 构造前馈速度原始值：方向冲突（反向且误差 > 0.2° ≈ 0.0035 rad）、
    #    极小速度、或上层已停更超 1.5 个上层周期且已到目标位置时，前馈清零
    error_full = target - command
    if (abs(error_full) > 0.0035 and dq_clamped * error_full < 0.0):
        dq_raw = 0.0
    elif abs(dq_clamped) < 1e-4:
        dq_raw = 0.0
    elif abs(error_full) < 1e-4 and elapsed_s > 1.5 * foh_upper_dt_s:
        dq_raw = 0.0
    else:
        dq_raw = dq_clamped * float(vel_ff_scale)

    # 5) 一阶低通滤波平滑前馈速度变化，抑制 100Hz 速度指令毛刺引发的力矩尖峰
    if lpf_tau_s > 0.0:
        alpha = period / (lpf_tau_s + period)
        new_dq_send = float(dq_prev_send) + alpha * (dq_raw - float(dq_prev_send))
    else:
        new_dq_send = dq_raw

    # 拖尾死区：当目标前馈已归零且滤波残留 < 1e-3 rad/s 时直接置零，避免无限衰减长尾
    if dq_raw == 0.0 and abs(new_dq_send) < 1e-3:
        new_dq_send = 0.0
    new_dq_send = max(-max_speed, min(max_speed, new_dq_send))
    return new_cmd, new_dq_send, wd_active


def _smoothstep01(x):
    """
    三次 Hermite 平滑阶跃函数 $S(u) = 3u^2 - 2u^3$（输入自动夹紧至 `[0, 1]`）。

    在端点 `0` 与 `1` 处一阶导数为 0（$C^1$ 连续），用于重力淡出与增益调度，
    避免力矩或阻尼在过渡区边界出现斜率折点。
    """
    u = max(0.0, min(1.0, float(x)))
    return u * u * (3.0 - 2.0 * u)


def cradle_zero_fade(q_rad_6):
    """
    计算零位机械支架重力补偿淡出因子，范围 `[0.0, 1.0]`。

    设计原理：
        B601-RS 回零停靠时大臂/小臂落在零位机械限位支架上，若仍输出向下或向上
        重力前馈会导致电机顶撞支架或在失能瞬间产生弹跳。
        - 以 `lift = max(|J2|, |J3|)`（单位 rad）衡量离开支架的抬起程度；
        - 当 `lift <= 0.03 rad`（约 1.72°）时返回 `0.0`，确保停靠/失能前完全卸载；
        - 当 `lift >= 0.13 rad`（约 7.45°，如 HOME 姿态 J2≈19.6°/J3≈19.1°）时返回
          `1.0` 全额重力补偿；中间区间通过三次 `_smoothstep01` 平滑过渡。

    参数:
        q_rad_6 (Sequence[float]): 6 轴关节角（rad，索引 1/2 对应 J2/J3）。

    返回:
        float: 平滑淡出系数，范围 `[0.0, 1.0]`。
    """
    lift = max(abs(float(q_rad_6[1])), abs(float(q_rad_6[2])))
    return _smoothstep01((lift - 0.03) / 0.10)


def compute_p3_gravity_tau(
    q_eval_6, prev_tau_6, gravity_fn, period, *,
    enabled, scales, max_nm, ramp_factor, slew_nm_s,
):
    """
    计算 6 轴静态重力补偿前馈力矩 `tau_ff`（纯函数，含淡入、零位淡出、硬封顶与斜率限速）。

    安全机制：
        1. J1（索引 0）为竖直基座旋转轴，不受重力矩影响，恒强制输出 `0.0 Nm`；
        2. 有效倍率 `eff_ramp = clip(ramp_factor, 0, 1) * cradle_zero_fade(q_eval_6)`，
           同时保证上电启动平滑加载（默认 0.5s）与回零贴近支架前平滑卸载至 0；
        3. 双重硬封顶：每轴取 `min(max_nm[i], HARD_MAX_GRAVITY_TAU_NM[i])`，即使 YAML
           配置误写也不会越过代码绝对天花板；
        4. 异常静默降级：若 `gravity_fn` 抛异常或返回非 6 维/含 NaN/Inf，目标力矩自动
           回落 `0.0` 并置位 `error_flag=True`，且依然经 `slew_nm_s` 限速平滑卸载，
           避免前馈瞬间跌零引起机械臂下坠抖动。

    参数:
        q_eval_6 (Sequence[float]): 用于求解重力矩的 6 轴关节角（rad）。
        prev_tau_6 (Sequence[float]): 上一 200Hz 拍实际下发的 6 轴重力前馈（Nm）。
        gravity_fn (Callable[[Sequence[float]], Sequence[float]] | None): 静态重力矩
            回调函数，输入 6 轴弧度角，返回 6 轴重力矩（Nm）。
        period (float): 底层控制周期（s，200Hz 下为 0.005）。
        enabled (bool): 重力前馈使能开关；为 False 时平滑斜坡归零。
        scales (Sequence[float]): 6 轴补偿比例 `[0.0, 0.95]`（J1 恒 0）。
        max_nm (Sequence[float]): 6 轴配置力矩上限（Nm）。
        ramp_factor (float): 启动斜坡淡入进度 `[0.0, 1.0]`。
        slew_nm_s (float): 力矩最大变化率上限（Nm/s，防止阶跃冲击）。

    返回:
        tuple[list[float], bool]: `(tau_out_6, error_flag)`，分别为本拍 6 轴下发力矩（Nm）
            以及本拍回调是否发生异常/非法值。
    """
    target = [0.0] * 6
    error_flag = False
    # 复合生效系数 = 启动斜坡淡入 × 零位支架淡出；回零近支架时自动把 target 压向 0
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
                    # 双重硬上限：同时受 yaml 配置上限与代码级绝对天花板限制
                    cap = min(float(max_nm[i]), float(HARD_MAX_GRAVITY_TAU_NM[i]))
                    val = float(raw[i]) * float(scales[i]) * eff_ramp
                    target[i] = max(-cap, min(cap, val))
        except Exception:  # noqa: BLE001
            error_flag = True

    # 斜率限速推进：无论是启用、关闭还是回调异常，均按 slew_nm_s 平滑过渡，杜绝阶跃
    max_step = max(1e-3, float(slew_nm_s) * float(period))
    out = [0.0] * 6
    for i in range(6):
        prev = float(prev_tau_6[i])
        delta = max(-max_step, min(max_step, target[i] - prev))
        v = prev + delta
        # 归零拖尾死区（< 1mNm 立即清零），便于 _needs_custom_mit() 回到零开销基线发送
        if target[i] == 0.0 and abs(v) < 1e-3:
            v = 0.0
        out[i] = 0.0 if i == 0 else v
    return out, error_flag


def compute_p3_gains(
    base_kp_6, base_kd_6, prev_kp_6, prev_kd_6, reach_r, period, *,
    enabled, r_min, r_max, kd_near, kd_far, kp_scale, slew_per_s,
):
    """
    计算 6 轴水平伸展度自适应 MIT 增益 `(kp, kd)`（纯函数，含安全夹紧与斜率限速）。

    设计目的：
        机械臂水平前伸越远（`reach_r` 越大），大臂/悬臂等效转动惯量越大，J1~J3 越容易
        出现低频欠阻尼振荡。通过 `_smoothstep01` 将 `reach_r` 在 `[r_min, r_max]` 映射为
        `s ∈ [0, 1]`，平滑插值 `kd_near → kd_far` 增大阻尼；同时严格限制：
        - `kp` 限制在 `[0.8, 1.5] * base_kp`；
        - `kd` 限制在 `[0.8, 1.8] * base_kd`，且腕部 RS00 电机（J4~J6）硬封顶 `<= 5.0`，
          基座/大臂/小臂 RS06 电机（J1~J3）硬封顶 `<= 18.0`；
        - 任意增益变化（包括关闭调度或进入回零停机）均按 `slew_per_s * period` 限速平滑过渡。

    参数:
        base_kp_6 (Sequence[float]): 上游配置的 6 轴基线刚度 `kp`（Nm/rad）。
        base_kd_6 (Sequence[float]): 上游配置的 6 轴基线阻尼 `kd`（Nm·s/rad）。
        prev_kp_6 (Sequence[float]): 上一拍实际生效的 6 轴 `kp`。
        prev_kd_6 (Sequence[float]): 上一拍实际生效的 6 轴 `kd`。
        reach_r (float): 当前末端水平伸展半径 `hypot(X, Y)`（m）。
        period (float): 底层控制周期（s）。
        enabled (bool): 是否启用增益调度；为 False（含安全回零期）时平滑回归基线增益。
        r_min (float): 调度下边界半径（m），`<= r_min` 时取 `kd_near`。
        r_max (float): 调度上边界半径（m），`>= r_max` 时取 `kd_far`。
        kd_near (Sequence[float]): 近距收缩姿态的 6 轴目标阻尼。
        kd_far (Sequence[float]): 远距伸展姿态的 6 轴目标阻尼。
        kp_scale (Sequence[float]): 6 轴刚度缩放倍率。
        slew_per_s (float): `kp`/`kd` 每秒最大允许变化速率（单位/s）。

    返回:
        tuple[list[float], list[float]]: `(kp_out_6, kd_out_6)`，本拍下发的 6 轴增益。
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
            # RS00 腕部电机（索引 3~5）阻尼上限 5.0；RS06 基座/肩肘（索引 0~2）上限 18.0
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
    P4 M6 200Hz 主动安全监督器（纯状态机，无硬件依赖，可独立离线单元测试）。

    三重保护职责：
        1. 跟踪误差碰撞/堵转检测：任一臂关节 `|q_cmd - q_act| > track_err_limit_deg`
           （默认 12°）连续持续 `>= track_err_window_s`（默认 80ms）→ 锁存软停机故障；
        2. 力矩预算堵转检测：任一臂关节实测 `|tau_act| > tau_limit_nm[i]`
           （默认 `[18, 28, 28, 10, 10, 10]` Nm）连续持续 `>= tau_limit_window_s`
           （默认 80ms）→ 锁存软停机故障；
        3. 温升前置降速：只要 7 路中存在有效 MOS 温度，即按最高温 `tmax` 在
           `[thermal_derate_start_c, thermal_derate_full_c]`（默认 90~110°C）区间
           线性将速度倍率 `thermal_scale` 从 `1.0` 下调至 `thermal_derate_min_scale`
           （默认 `0.40`），在上游三级硬件过温急停前主动给系统降温留余量。

    安全红线：
        触发故障后绝不直接失能电机（避免空中坠臂），由控制器在当前位置执行
        「软停机保持 + 维持重力补偿」，待人工确认安全并按 `g` 调用 `clear_fault()` 后恢复。
    """

    def __init__(self, cfg, control_hz):
        """
        初始化安全监督器（连续超限拍数阈值由 `control_hz` 推导，禁止硬编码）。

        参数:
            cfg (dict): 经 `validate_mit_ext_cfg()` 校验的扩展配置字典。
            control_hz (float): MIT 控制循环频率（Hz），必须为有限正数。
        """
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
        """清除故障锁存及连续超限计数器（供用户确认环境安全后重新启动跟随）。"""
        self.fault = False
        self.fault_reason = ""
        self._track_over = 0
        self._tau_over = 0

    def update(self, q_cmd_6, q_act_6, tau_act_6, t_mos_all, valid_6):
        """
        每拍（200Hz）更新温升降速系数及碰撞/堵转故障判定。

        参数:
            q_cmd_6 (Sequence[float]): 本拍 6 轴指令关节角（rad）。
            q_act_6 (Sequence[float]): 本拍 6 轴实测关节角（rad）。
            tau_act_6 (Sequence[float]): 本拍 6 轴实测力矩（Nm）。
            t_mos_all (Sequence[float | None]): 全部电机（含 J7）MOS 温度（°C）。
            valid_6 (Sequence[bool]): 6 轴反馈有效性标志；任一轴无效时清零超限计数防误报。

        返回:
            bool: 仅当本拍**首次**由正常状态跃迁至故障锁存时返回 `True`（用于触发一次性
                软停机动作与日志打印）；已锁存或正常时均返回 `False`。
        """
        # 温升前置降速：不依赖 supervisor_enable 与 6 轴全有效，只要有有限 MOS 温度即实时计算
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

        # 未启用碰撞/堵转监督或本拍 6 轴反馈不全有效：清零计数避免陈旧/残缺数据误触发
        if not self.enabled or not all(valid_6[:6]):
            self._track_over = 0
            self._tau_over = 0
            return False

        # 1) 跟踪误差监测（rad 转 deg 后与阈值比对，记录最严重轴号 1..6）
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

        # 2) 力矩预算监测（按分轴上限归一化 ratio = |tau| / tau_limit_nm[i]）
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

        # 已处于故障锁存状态时仅更新监测指标，不再重复返回 newly_faulted=True
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
        """
        返回监督器当前状态快照字典，供上层 100Hz 控制降速、HUD 显示及诊断日志使用。

        返回:
            dict: 包含 `enabled`, `fault`, `reason`, `thermal_scale`,
                `track_err_max_deg`, `tau_max_ratio`, `tmos_max_c`（无有效温度时为 None）。
        """
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
    """
    计算上游控制器基类所在源文件的 SHA256 摘要，用于方案 A 运行时版本钉死校验。

    参数:
        base_cls (type): 上游控制器类（如 `rebot.ReBotRSMITController`）。

    返回:
        tuple[str | None, str | None]: `(hex_sha256, source_path)`；无法定位源文件时
            摘要返回 `None`。
    """
    path = inspect.getsourcefile(base_cls)
    if not path or not os.path.exists(path):
        return None, path
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest(), path


def check_upstream_compat(base_cls, pinned_sha256=PINNED_UPSTREAM_SHA256,
                          require_pinned=True):
    """
    检验上游控制器基类与本扩展的结构及源码版本兼容性（方案 A 安全门控）。

    校验三项内容：
        1. `_REQUIRED_UPSTREAM_MEMBERS` 中所有私有/公有依赖方法均存在且可调用；
        2. `_send_mit_positions` 签名保留关键字参数 `lock_timeout`（回零阶段依赖）；
        3. 若 `require_pinned=True`，上游 `controller.py` 文件 SHA256 必须与
           `pinned_sha256` 完全一致，防止上游静默变更私有字段语义引发隐蔽故障。

    参数:
        base_cls (type): 上游控制器类。
        pinned_sha256 (str): 期望的上游源码 SHA256 十六进制串。
        require_pinned (bool): 是否强制校验源码哈希（离线假基类测试可置 False）。

    返回:
        tuple[bool, str, str | None]: `(ok, reason, sha256)`。`ok=False` 时调用方必须
            回退上游原版控制器，绝不强行挂载扩展子类。
    """
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
    """
    动态构造继承自 `base_cls` 的 `ReBotMITExtController` 子类（方案 A 运行时子类化）。

    设计说明：
        延迟在运行时绑定 `base_cls`，既能零修改复用外部仓库 `rebot_control` 中的
        `ReBotRSMITController`，也允许离线单元测试注入轻量级假基类验证 200Hz 线程、
        CAN 容错、FOH 插值、重力补偿、增益调度与安全监督全链路。

    参数:
        base_cls (type): 上游 MIT 控制器基类（需预先通过 `check_upstream_compat()`）。

    返回:
        type: 动态生成的 `ReBotMITExtController` 类。
    """
    class ReBotMITExtController(base_cls):
        """
        上游 `ReBotRSMITController` 的 P4 完整扩展子类（v13 方案 A）。

        功能覆盖：
            - P0: S1 CAN 30ms 容错 + 200Hz 零开销反馈缓存；
            - P1: `set_joint_mit_targets()` 速度前馈 + FOH 连续插值 + 40ms 看门狗；
            - P3: Pinocchio 静态重力前馈（含回零平滑淡出）+ 水平伸展自适应阻尼调度；
            - P4: 200Hz 主动安全监督层（碰撞/堵转软停机保持 + 90~110°C 线性热降额）。

        并发锁划分与防死锁顺序约定：
            - `self.target_lock`（上游定义）：保护目标位置、指令位置、速度前馈状态、
              P3 重力回调/当前生效力矩与增益；
            - `self._io_lock_guard()`（上游定义）：独占底层 CAN I/O 发送与同拍缓存读取；
            - `self._fb_lock`：仅保护 `_fb_snapshot` 引用读写（持锁时间极短，不嵌套他锁）；
            - `self._supervisor_lock`：保护 `_supervisor` 状态机（持锁内绝不申请 `target_lock`）；
            - `self._stats_lock` / `self._rec_lock`：叶节点锁，分别保护耗时统计队列与录波缓存。
            锁顺序约束：`target_lock` 可在内部短暂获取 `_fb_lock`；`_io_lock_guard` 内部
            调用 `_sample_feedback_locked()` 时依次独立获取 `_fb_lock`、`_supervisor_lock`、
            `target_lock`（仅首次故障软停机时）、`_rec_lock`，绝不与 `target_lock → _io_lock_guard`
            形成交叉循环等待。
        """

        EXT_PHASE = "P4"
        # 连续 1s 读取缓存异常即停用缓存（具体 tick 阈值由 control_hz 动态推导）。
        _FEEDBACK_ERROR_DISABLE_S = 1.0
        # 安全停机回零阶段单帧发送的立即重试时间窗口（s）。
        _SHUTDOWN_SEND_RETRY_S = 0.02

        def __init__(self, *args, ext_cfg=None, **kwargs):
            """
            初始化扩展控制器：先完成上游基类构造，再依据 `self.control_hz` 推导所有时序阈值。

            参数:
                *args, **kwargs: 透传给上游 `base_cls.__init__`。
                ext_cfg (dict | None): `mit_ext` 配置字典；缺省项由 `validate_mit_ext_cfg()` 补全。
            """
            super().__init__(*args, **kwargs)
            self.ext_cfg = validate_mit_ext_cfg(ext_cfg)
            rate = float(self.control_hz)
            # P0/P4 状态机：窗口阈值均按 control_hz 推导，严禁硬编码拍数
            self._fault_monitor = CanFaultMonitor(
                self.ext_cfg["can_fault_window_s"], rate)
            self._supervisor = SafetySupervisor(self.ext_cfg, rate)
            self._supervisor_lock = threading.Lock()
            # P1 速度前馈状态（与上游 target_positions/command_positions 共用 target_lock）
            self._target_velocities = [0.0 for _ in range(self.motor_count)]
            self._command_velocities = [0.0 for _ in range(self.motor_count)]
            self._mit_vel_mode = False
            self._last_mit_update_t = -math.inf
            self._watchdog_trips = 0
            self._watchdog_was_active = False
            self._last_dq_send_max = 0.0
            # P3 M2/M3 状态（重力前馈回调、斜坡淡入起点、伸展半径与当前生效力矩/增益）
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
            # P0 200Hz 反馈缓存状态（_fb_lock 仅保护 _fb_snapshot 指针切换与拷贝）
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
            # 循环抖动与 CAN 发送耗时滑动统计窗口（最近 2s）
            window = max(16, int(round(2.0 * rate)))
            self._stats_lock = threading.Lock()
            self._period_hist = deque(maxlen=window)
            self._send_hist = deque(maxlen=window)
            self._sample_hist = deque(maxlen=window)
            self._ticks = 0
            self._last_can_log = -math.inf
            self._suppressed_can_logs = 0
            self._can_fault_triggered = False
            # 200Hz 高频录波缓冲（供 sysid / 探针脚本按需启停）
            self._rec_lock = threading.Lock()
            self._recording = None
            self._rec_max_rows = int(round(self.ext_cfg["record_max_s"] * rate))

        # ── P1 扩展目标下发接口（旧接口 100% 兼容）────────────
        def set_joint_angles(self, angles_deg):
            """
            兼容上游老接口：下发纯位置目标（度），同时立即退出速度前馈模式并清零 `dq_des`。

            说明：
                用于 `hold()`、`move_to_joints()`、标定脚本与回零逻辑，确保纯位置场景
                绝不残留上一阶段的旧前馈速度。
            """
            with self.target_lock:
                super().set_joint_angles(angles_deg)
                self._mit_vel_mode = False
                for i in range(self.motor_count):
                    self._target_velocities[i] = 0.0

        def set_joint_angle(self, joint_id, angle_deg):
            """兼容上游老接口：修改单个关节纯位置目标（度），同时清零该轴速度前馈。"""
            with self.target_lock:
                super().set_joint_angle(joint_id, angle_deg)
                self._target_velocities[joint_id - 1] = 0.0

        def set_joint_mit_targets(self, angles_deg, velocities_deg_s=None):
            """
            P1 扩展接口：向 200Hz 底层同步下发 7 路目标角度（度）与速度前馈（度/秒）。

            安全拦截与硬约束：
                - 若已进入安全停机（`shutdown_started=True`）或 P4 安全监督已锁存故障
                  （`supervisor.fault=True`），静默丢弃指令，保持当前安全姿态不动；
                - 任意通道出现 NaN/Inf 立即抛 `ValueError`，绝不污染 200Hz 实时循环；
                - J7 夹爪（索引 6）硬隔离：强制 `velocities[6] = 0.0`，恒为纯位置模式；
                - 内部统一转换为弧度（rad / rad/s）并在单把 `target_lock` 内原子写入，
                  同时刷新看门狗时间戳 `_last_mit_update_t`。

            参数:
                angles_deg (Sequence[float]): `motor_count` 维目标关节角（度）。
                velocities_deg_s (Sequence[float] | None): `motor_count` 维目标角速度
                    前馈（度/秒）；传 `None` 时退化为纯位置模式（`vel_mode=False`）。
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
                # 夹爪通道（最后一个电机，motor_id=7，索引 6）恒为纯位置模式，严禁前馈
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
            """
            运行时动态开关速度前馈（供 `tools/joint_sysid.py` 做开/关 A/B 辨识对比）。

            参数:
                enabled (bool): `False` 时立即清零内部低通滤波前馈速度缓存 `_command_velocities`。
            """
            with self.target_lock:
                self.ext_cfg["vel_ff_enable"] = bool(enabled)
                if not enabled:
                    for i in range(self.motor_count):
                        self._command_velocities[i] = 0.0

        def set_gravity_torque_fn(self, fn):
            """
            注册纯数学静态重力矩计算回调（由 `RebotAdapter` 注入隔离 Pinocchio 模型）。

            参数:
                fn (Callable[[Sequence[float]], Sequence[float]] | None): 输入 6 轴弧度角，
                    返回 6 轴重力矩（Nm）；注册非空回调时重置斜坡淡入起点 `_gravity_start_t`。
            """
            with self.target_lock:
                self._gravity_fn = fn
                self._gravity_start_t = time.monotonic() if fn is not None else None
                self._gravity_error_logged = False

        def set_gravity_ff_runtime(self, enabled=None, scale=None):
            """
            运行时调节重力补偿开关与分轴补偿比例（供 `tools/p3_gravity_verify.py` 分级验证）。

            参数:
                enabled (bool | None): 非 None 时切换 `gravity_ff_enable`；首次开启时初始化淡入计时。
                scale (Sequence[float] | None): 非 None 时校验并更新 6 轴比例（J1 强制 0.0，
                    其余轴硬限制在 `[0.0, 0.95]`）。
            """
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
            """运行时开关伸展阻尼调度（内部经 `gain_sched_slew_per_s` 限速平滑过渡，无阶跃）。"""
            with self.target_lock:
                self.ext_cfg["gain_sched_enable"] = bool(enabled)

        def set_reach_radius(self, reach_r_m):
            """
            更新当前末端水平伸展半径 `hypot(X, Y)`，驱动 200Hz 自适应阻尼平滑调度。

            参数:
                reach_r_m (float): 水平伸展半径（m），仅接受 `[0.0, 0.60]` 内的有限值。
            """
            val = float(reach_r_m)
            if math.isfinite(val) and 0.0 <= val <= 0.60:
                with self.target_lock:
                    self._reach_r = val

        def _compute_p3_step_locked(self, commands, now_mono, period, shutdown_mode=False):
            """
            在持有 `self.target_lock` 的前提下计算本拍 6 轴重力前馈 `tau_ff` 与增益 `(kp, kd)`。

            关键安全设计（`shutdown_mode` 区分）：
                - 正常运行期（`shutdown_mode=False`）：优先取未过期（`<= feedback_max_age_s`）
                  的 200Hz 实测关节角 `q_act` 计算重力矩，贴合真实连杆姿态；缓存过期时回退
                  至指令角 `commands[:6]`；
                - 安全回零期（`shutdown_mode=True`）：200Hz 控制线程已停止，反馈缓存不再
                  更新；此时**必须**直接使用上游回零 smoothstep 轨迹 `commands[:6]` 评估
                  `cradle_zero_fade()`，确保臂接近零位支架时重力前馈 100% 平滑淡出至 0 Nm，
                  绝不因缓存过期卡在旧抬臂角度而推顶零位支架；同时关闭增益调度令 `(kp, kd)`
                  平滑回归上游基线。

            参数:
                commands (Sequence[float]): 当前拍关节指令角（rad，至少前 6 维）。
                now_mono (float): 当前单调时钟时间戳（s）。
                period (float): 底层控制周期（s）。
                shutdown_mode (bool): 是否处于安全回零停机阶段。

            返回:
                tuple[list[float], list[float], list[float]]: `(tau_6, kp_6, kd_6)`。
            """
            # 1) 重力补偿：运行期优先取新鲜真实 q_act；安全回零期控制线程已停，
            #    必须直接用回零 smoothstep 轨迹 commands[:6]，确保靠零位时 100% 平滑淡出至 0。
            q_eval = list(commands[:6])
            if not shutdown_mode:
                with self._fb_lock:
                    snap = self._fb_snapshot
                if (snap is not None and all(snap["valid"][:6])
                        and (now_mono - snap["time"]) <= self.ext_cfg["feedback_max_age_s"]):
                    q_eval = list(snap["q"][:6])
            # 启动斜坡淡入进度 [0, 1]，避免开启重力补偿瞬间产生力矩阶跃
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

            # 2) 增益调度：安全停机/回零阶段置 enabled=False，经斜率限速平滑回基准增益
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

        # ── 200Hz MIT 主循环（P3/P4：FOH + 速度前馈 + 重力补偿 + 阻尼调度 + S1 容错 + 反馈监督）──
        def _control_loop(self):
            """
            200Hz MIT 控制主循环（覆盖上游 `base_cls._control_loop`）。

            单拍执行流水线：
                1. 在 `target_lock` 内逐轴计算 P1 步进（`compute_p1_mit_step`），J7（索引 >= 6）
                   强制隔离为纯位置模式（`vel_ff=0, foh=False`）；
                2. 计算 P3 静态重力前馈 `tau_6` 与伸展自适应增益 `(kp_6, kd_6)`；
                3. 释放 `target_lock` 后调用 `_send_and_sample()`，在同一把 `_io_lock_guard` 内
                   发送 MIT 帧并零开销读取 `motor.get_state()` 反馈 + 触发 P4 安全监督；
                4. 若发送异常，交由 `CanFaultMonitor` 做 S1 30ms 连续窗口容错；未超阈值则
                   限频打印并继续下一拍，超阈值后调用 `_trigger_can_fault()`；
                5. 记录周期/发送/采样统计并按绝对时间戳 `next_tick` 补偿睡眠防累积漂移。
            """
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
                        # 硬隔离：仅 J1~J6（index < 6）允许速度前馈与 FOH 插值，J7 夹爪恒为纯位置
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
                    # 仅在看门狗上升沿累计触发次数，便于离线日志区分单次卡顿与持续超时
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

                # 绝对节拍调度：落后超过整拍时重锚 next_tick，避免突发补拍打爆 CAN 总线
                next_tick += period
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

        def _send_mit_custom_locked(self, commands, velocities, tau_6, kp_6, kd_6):
            """
            在持有 `_io_lock_guard()` 的前提下逐轴下发带扩展参数的 MIT 控制帧。

            约定：
                - J1~J6（索引 0..5）按需注入 `dq_des`、`kp`、`kd`、`tau_ff`；
                - J7 夹爪（索引 >= 6）始终硬编码为 `(dq=0, base_kp, base_kd, tau=0)`，
                  绝不透传机械臂动力学前馈或改写夹爪增益。
            """
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
            """
            判断本拍是否存在非基线 MIT 参数（任一轴有非零速度/力矩前馈或增益偏离基线）。

            返回:
                bool: `False` 时可直接回退上游 `base_cls._send_mit_positions()`，保持
                    与上游原版 100% 一致的发送行为。
            """
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
            """
            在同一把 `_io_lock_guard()` 内发送 MIT 帧并立即读取应答缓存（零额外 CAN 请求）。

            返回:
                tuple[float, float | None]: `(send_s, sample_s)`，分别为本拍 CAN 发送耗时（s）
                    与反馈采样+监督耗时（s，缓存停用时为 `None`）。
            """
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
            """
            覆盖上游纯位置发送方法，为安全停机回零阶段提供平滑重力淡出与瞬时重试保护。

            行为区分：
                - 非停机阶段（`shutdown_started=False`）：原样委托上游 `base_cls._send_mit_positions`；
                - 安全停机回零阶段（`shutdown_started=True`）：
                  1. 强制 `dq_des = 0.0`、以 `shutdown_mode=True` 调用 `_compute_p3_step_locked()`，
                     令 `(kp, kd)` 平滑退回基线，并用回零 smoothstep 轨迹 `positions_rad` 驱动
                     `cradle_zero_fade()`，保证高抬位回零全程有重力托举且抵近零位支架时完全卸载；
                  2. 对非 `TimeoutError` 的 CAN 瞬时异常在 `_shutdown_retries` 次数内以 1ms 间隔
                     立即重试，提高回零可靠性；`TimeoutError`（`io_lock` 超时）原样抛出交上游处理。
            """
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

        # ── 200Hz 反馈缓存与主动安全监督 ─────────────────────
        def _sample_feedback_locked(self, commands):
            """
            在 `_io_lock_guard()` 内零额外 CAN 请求读取全部电机应答缓存，并执行 P4 安全监督。

            处理流程：
                1. 可选调用 `poll_feedback_once()`（仅当 `feedback_poll_once=True` 时用于探针测试）；
                2. 逐电机调用 `motor.get_state()` 提取 `(pos, vel, torq, t_mos, t_rotor, status)`；
                   任何读取/解析异常只累计诊断错误，连续 1s 异常后自动置 `_fb_enabled=False`
                   降级，绝不抛出打断 200Hz 控制或误报 CAN 故障；
                3. 统计连续帧数值变化率（用于检测底层是否真实刷新缓存）并原子发布 `_fb_snapshot`；
                4. 非停机状态下运行 `SafetySupervisor.update()`：首次锁存碰撞/堵转故障时，
                   立即在 `target_lock` 内将前 6 轴目标钉死在当前 `command_positions` 并清零
                   速度前馈，实现「软停机保持 + 重力补偿维持」（绝不空中失能）；
                5. 若已开启 200Hz 录波且未达上限行数，追加一行 `(t, cmd, q, dq, tau)`。
            """
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
                # 仅当核心动力学状态 pos/vel/torq 均为有限值时标记该轴本拍有效
                if not all(math.isfinite(v) for v in values[:3]):
                    continue
                q[index], dq[index], tau[index], t_mos[index], t_rotor[index] = values
                status[index] = int(getattr(state, "status_code", -1))
                valid[index] = True

            # 连续异常保护：超过 1s 连续报错自动停用缓存降级，不影响 MIT 下发与硬件温保
            if error_seen:
                self._fb_errors += 1
                self._fb_consecutive_errors += 1
                if self._fb_consecutive_errors >= self._fb_disable_threshold:
                    self._fb_enabled = False
                    print("\n[MIT扩展] 反馈缓存连续读取异常超过 1s，已自动停用"
                          "（不影响 MIT 控制与温度保护）")
            else:
                self._fb_consecutive_errors = 0

            # 统计反馈帧实际刷新比例（用于 P0 探针验证 get_state() 是否随 200Hz MIT 帧更新）
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
                    # 先释放 _supervisor_lock 再拿 target_lock，遵守锁序约定；钉死当前指令位姿
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
            """
            获取最近一拍 200Hz 反馈缓存的深拷贝快照（零 CAN I/O，线程安全）。

            返回:
                dict | None: 若尚未采样返回 `None`；否则返回包含 `time`（monotonic s）、
                    `age_s`（距今耗时 s，供调用方按 `feedback_max_age_s` 判过期）、`seq`、
                    `q`（rad）、`dq`（rad/s）、`tau`（Nm）、`t_mos`（°C）、`t_rotor`（°C）、
                    `status`、`valid`、`cmd`（rad）的 7 路拷贝字典。
            """
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
            """返回 P4 M6 安全监督器当前状态快照字典（持叶锁瞬时拷贝，无 I/O 阻塞）。"""
            with self._supervisor_lock:
                return self._supervisor.status()

        def clear_safety_fault(self):
            """清除安全监督故障锁存（由用户检查现场安全并重新按 `g` 启动跟随时调用）。"""
            with self._supervisor_lock:
                self._supervisor.clear_fault()

        def set_feedback_poll_once(self, enabled):
            """运行时切换是否在 MIT 发送后调用 `poll_feedback_once()`（供 P0 探针 A/B 对比）。"""
            self.ext_cfg["feedback_poll_once"] = bool(enabled)

        def reset_feedback_change_stats(self):
            """重置反馈刷新率计数与最近 2s 循环计时窗口，用于探针分阶段采样。"""
            with self._stats_lock:
                self._period_hist.clear()
                self._send_hist.clear()
                self._sample_hist.clear()
            self._fb_last_values = None
            self._fb_change_ticks = 0
            self._fb_ticks = 0

        # ── 200Hz 高频录波（供辨识与标定工具） ──────────────
        def start_recording(self):
            """开启 200Hz 内存录波（最大录制时长受 `record_max_s` 限制，防止内存无限增长）。"""
            with self._rec_lock:
                self._recording = []

        def stop_recording(self):
            """
            停止 200Hz 录波并返回采样行列表。

            返回:
                list[tuple]: 每行格式为 `(t_monotonic, cmd[7], q[7], dq[7], tau[7])`，
                    角度/角速度单位为 rad / rad/s，力矩单位为 Nm。
            """
            with self._rec_lock:
                rows = self._recording or []
                self._recording = None
            return rows

        # ── 运行统计与 S1 CAN 故障处理 ───────────────────────
        def get_loop_stats(self):
            """
            汇总最近 2s 的 200Hz 控制循环性能、CAN 容错、反馈缓存、P1/P3 状态及 P4 安全摘要。

            返回:
                dict: 含周期/发送/采样耗时分位数（ms）、CAN 故障统计、前馈/重力/增益现状及
                    `safety` 子字典，供 `servo_diagnostics` 与 CLI 工具记录分析。
            """
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
            """限频打印 CAN 发送异常（每 `can_error_log_interval_s` 最多一行；触发故障必打）。"""
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
            """
            S1 持续 CAN 通信故障响应：按 `can_fault_action` 异步请求停机。

            默认 `can_fault_action="return_to_zero"`：先尝试缓慢回零再失能，避免半空断电坠臂；
            `wait=False` 防止 200Hz 工作线程在自身内部死等自己退出。
            """
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
    构造 MIT 底层控制器实例：配置开启且上游兼容时返回扩展版，否则自动回退上游原版。

    降级路径（保证安全与向后兼容）：
        1. `mit_ext.enable == False` → 直接实例化 `base_cls()`（v12 原版行为）；
        2. `check_upstream_compat()` 不通过（缺少成员、签名变化或 SHA256 与钉死版本不符）
           → 打印醒目告警并回退 `base_cls()`；
        3. 全部校验通过 → 通过 `make_extended_controller_class(base_cls)` 构造并返回扩展实例。

    参数:
        base_cls (type): 上游控制器类（实机为 `rebot.ReBotRSMITController`）。
        cfg (dict | None): 显式传入的 `mit_ext` 配置字典；为 `None` 时从 `cfg_path` 加载。
        cfg_path (str | None): 可选 YAML 配置路径，默认 `config/mit_ext.yaml`。

    返回:
        tuple[object, dict]: `(controller, info)`，其中 `info` 记录 `active`（bool）、
            `phase`（如 `"P4"` 或 `None`）、`reason`、`upstream_sha256` 与生效 `cfg`。
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
