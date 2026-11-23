#!/usr/bin/env python3
"""
v13 P0 阶段单关节安全小幅辨识工具（阶跃 + 离散正弦扫频）。

安全设计：
- 仅在安全位姿下逐轴激励（其余 5 轴保持不动），默认 HOME 位姿（可选安全中伸展位姿测 J1 共振）。
- 硬限幅：阶跃 ≤ 4.0°（默认 2.5°）、扫频幅值 ≤ 3.0°（默认 2.0°）、峰值速度 ≤ 0.80 rad/s、
  距离软限位至少留 5° 余量。
- 实时安全看护：MOS 温度 ≥ 95°C 或跟踪偏移超 amp+6° 立即中止并回 HOME。
- 无论正常结束还是 Ctrl+C，finally 中严格执行恢复限速 → hold() → stop(return_to_zero=True, wait=True)。

辨识输出（同时保存 logs/sysid_<timestamp>.json 与原始 200Hz 波形 .npz）：
1. 静态重力下垂与 Pinocchio 重力矩符号/比例核验（为 P3 M2 重力补偿标定）
2. 阶跃响应：延迟、上升时间、超调量、二阶拟合 (delay_ms, wn_hz, zeta, J_eff, 推荐 kd)
3. 扫频频率响应：各频点幅值比 |H(jw)|、相位滞后（度）、等效延迟（ms），直接定位共振峰

支持离线自检模式：`python3 tools/joint_sysid.py --simulate`（不访问任何硬件）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 安全硬上限（防止辨识激励触发过流、机械碰撞或超速）
MAX_STEP_DEG = 4.0                # 单次阶跃最大允许幅值（deg）
MAX_SINE_AMP_DEG = 3.0            # 正弦扫频最大允许幅值（deg）
MAX_SINE_PEAK_VEL_RAD_S = 0.80    # 正弦扫频峰值角速度硬上限 w*A <= 0.80 rad/s
LIMIT_SAFETY_MARGIN_DEG = 5.0     # 被测轴距软件限位 q_min/q_max 最小安全余量（deg）
MAX_MOS_TEMP_C = 95.0             # 实时看护 MOS 温度硬熔断上限（°C）
DEFAULT_FREQS_HZ = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # 覆盖臂展悬臂共振峰 (1~2Hz) 至带宽滚降区


# ─────────────────────────────────────────────────────────────
# 纯数学辨识函数（可独立离线单测）
# ─────────────────────────────────────────────────────────────
def second_order_step_model(t, delay_s, wn_rad_s, zeta, gain=1.0):
    """标准延迟二阶系统单位阶跃响应 y(t)（支持欠阻尼/临界阻尼/过阻尼）。

    对应连续传递函数：
        G(s) = gain * wn^2 / (s^2 + 2 * zeta * wn * s + wn^2) * exp(-s * delay_s)

    参数:
        t (array-like): 时间采样序列（单位: s）。
        delay_s (float): 纯传输/执行延迟 Td（单位: s）。
        wn_rad_s (float): 无阻尼自然角频率 wn（单位: rad/s，内部下限钳位 1e-3）。
        zeta (float): 阻尼比（无量纲，内部下限钳位 0.02）。
        gain (float): 直流稳态增益（默认 1.0）。

    返回:
        np.ndarray: 与 ``t`` 同形状的阶跃响应序列（tau <= 0 段严格为 0）。
    """
    t_arr = np.asarray(t, dtype=np.float64)
    tau = t_arr - float(delay_s)
    y = np.zeros_like(t_arr, dtype=np.float64)
    mask = tau > 0.0
    if not np.any(mask):
        return y
    tm = tau[mask]
    wn = max(float(wn_rad_s), 1e-3)
    z = max(float(zeta), 0.02)
    if z < 0.999:
        # 欠阻尼解析解（有阻尼振荡频率 wd = wn * sqrt(1 - zeta^2)）
        wd = wn * math.sqrt(1.0 - z * z)
        env = np.exp(-z * wn * tm)
        ym = 1.0 - env * (np.cos(wd * tm) + (z / math.sqrt(1.0 - z * z)) * np.sin(wd * tm))
    elif z <= 1.001:
        # 临界阻尼解析解（避免分母 sqrt(|1 - zeta^2|) 数值奇异）
        env = np.exp(-wn * tm)
        ym = 1.0 - env * (1.0 + wn * tm)
    else:
        # 过阻尼双实极点解析解（极点 -r1, -r2）
        s = math.sqrt(z * z - 1.0)
        r1 = wn * (z - s)
        r2 = wn * (z + s)
        ym = 1.0 - (r2 * np.exp(-r1 * tm) - r1 * np.exp(-r2 * tm)) / (r2 - r1)
    y[mask] = float(gain) * ym
    return y


def _simulate_filtered_step(t, u_norm, delay_s, wn_rad_s, zeta, gain=1.0):
    """用输入斜坡 u_norm(t) 与延迟二阶阶跃响应差分卷积，精确兼容 200Hz 有限斜率指令。

    因为 MIT 底层对目标角有斜率限速，实际下发的 ``q_cmd`` 是约 15~30ms 的斜坡而非理想阶跃。
    根据线性时不变系统叠加原理（离散杜哈梅积分）：
        y[k] = sum_{m=0..k} du[m] * s[k - m]
    其中 ``du[m] = u[m] - u[m-1]`` 为每拍指令增量，``s`` 为单位阶跃响应。
    这样拟合出的 ``delay_s`` 反映真实执行纯滞后，不会把指令斜坡时长误算作延迟。

    参数:
        t (array-like): 均匀采样时间序列（单位: s）。
        u_norm (array-like): 归一化后的实际下发指令波形（约从 0 过渡到 1）。
        delay_s (float): 纯延迟 Td（单位: s）。
        wn_rad_s (float): 自然角频率 wn（单位: rad/s）。
        zeta (float): 阻尼比。
        gain (float): 稳态增益。

    返回:
        np.ndarray: 卷积预测的归一化关节实际响应序列。
    """
    t_arr = np.asarray(t, dtype=np.float64)
    u = np.asarray(u_norm, dtype=np.float64)
    if t_arr.size < 4:
        return np.zeros_like(t_arr)
    dt = float(np.median(np.diff(t_arr)))
    t_grid = np.arange(t_arr.size, dtype=np.float64) * dt
    step_ir = second_order_step_model(t_grid, delay_s, wn_rad_s, zeta, gain=gain)
    du = np.diff(u, prepend=u[0])
    y = np.convolve(du, step_ir)[:t_arr.size]
    return y


def analyze_step_response(t, q_cmd, q_act, kp=None, pre_s=0.15):
    """分析单次阶跃段（包含阶跃前稳态与阶跃后至少 0.6s 响应）并拟合延迟二阶模型。

    物理参数换算关系（忽略摩擦近似刚体关节）：
        - 等效转动惯量: ``J_eff = kp / wn^2``（单位: kg·m²）
        - 目标阻尼比 ``zeta = 0.70`` 推荐 MIT 微分阻尼: ``kd_0.7 = 2 * 0.70 * sqrt(kp * J_eff)``

    参数:
        t (array-like): 采样时间戳序列（单位: s）。
        q_cmd (array-like): 底层下发指令关节角序列（单位: rad）。
        q_act (array-like): 200Hz 实测关节角序列（单位: rad）。
        kp (float | None): 当前关节 MIT 位置刚度（单位: Nm/rad）；提供时推算 ``J_eff`` 与 ``kd_for_zeta_07``。
        pre_s (float): 阶跃前基线稳态窗口时长（单位: s，默认 0.15s）。

    返回:
        dict: 包含 ``ok`` 标志、指令/实际阶跃幅值（deg）、稳态增益 ``ss_gain``、
            10%/50% 延迟与 10%~90% 上升时间（ms）、超调百分比、5% 稳定时间（ms）
            以及非线性最小二乘拟合结果字典 ``fit``（含 ``delay_ms, wn_hz, zeta, gain, r2, j_eff_kgm2, kd_for_zeta_07``）。
    """
    t = np.asarray(t, dtype=np.float64)
    qc = np.asarray(q_cmd, dtype=np.float64)
    qa = np.asarray(q_act, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(qc) & np.isfinite(qa)
    t, qc, qa = t[valid], qc[valid], qa[valid]
    if t.size < 20:
        return {"ok": False, "reason": "样本不足"}

    # 用段首 pre_s 与段尾 20% 的中位数估计阶跃前后稳态角，免疫偶发噪点
    t = t - t[0]
    n_pre = max(5, int(np.searchsorted(t, pre_s)))
    if n_pre >= t.size - 10:
        n_pre = max(5, t.size // 6)
    qc0 = float(np.median(qc[:n_pre]))
    qa0 = float(np.median(qa[:n_pre]))
    n_tail = max(5, t.size // 5)
    qc1 = float(np.median(qc[-n_tail:]))
    qa1 = float(np.median(qa[-n_tail:]))
    cmd_delta = qc1 - qc0
    act_delta = qa1 - qa0
    if abs(cmd_delta) < np.deg2rad(0.3):
        return {"ok": False, "reason": "指令阶跃幅值过小"}

    # u_norm/y_norm 以指令幅值归一化（用于拟合稳态增益），y_ss 以实际稳态幅值归一化（用于时域指标）
    u_norm = (qc - qc0) / cmd_delta
    y_norm = (qa - qa0) / cmd_delta
    ss_gain = float(act_delta / cmd_delta)
    y_ss = (qa - qa0) / act_delta if abs(act_delta) > 1e-6 else y_norm

    # 指令与实际穿过 10% / 50% / 90% 的时刻（线性插值消除 5ms 采样栅格量化误差）
    def _first_cross(sig, level):
        """线性插值求信号首次上穿给定阈值 level 的精确时刻（s）。"""
        idx = np.where(sig >= level)[0]
        if idx.size == 0:
            return None
        i = int(idx[0])
        if i == 0:
            return float(t[0])
        s0, s1 = float(sig[i - 1]), float(sig[i])
        frac = (level - s0) / (s1 - s0) if abs(s1 - s0) > 1e-9 else 0.0
        return float(t[i - 1] + np.clip(frac, 0.0, 1.0) * (t[i] - t[i - 1]))

    t_cmd_10 = _first_cross(u_norm, 0.10) or 0.0
    t_cmd_50 = _first_cross(u_norm, 0.50) or t_cmd_10
    t_act_10 = _first_cross(y_ss, 0.10)
    t_act_50 = _first_cross(y_ss, 0.50)
    t_act_90 = _first_cross(y_ss, 0.90)

    delay_10_ms = (None if t_act_10 is None else max(0.0, (t_act_10 - t_cmd_10) * 1000.0))
    delay_50_ms = (None if t_act_50 is None else max(0.0, (t_act_50 - t_cmd_50) * 1000.0))
    rise_ms = (None if (t_act_10 is None or t_act_90 is None)
               else max(0.0, (t_act_90 - t_act_10) * 1000.0))

    post_mask = t >= t_cmd_10
    peak_val = float(np.max(y_ss[post_mask])) if np.any(post_mask) else 1.0
    overshoot_pct = max(0.0, (peak_val - 1.0) * 100.0)

    # 5% 稳定时间（从指令 10% 时刻起算，最后一次离开 [0.95, 1.05] 的时间）
    outside = np.where(post_mask & (np.abs(y_ss - 1.0) > 0.05))[0]
    settle_ms = (float(max(0.0, (t[int(outside[-1])] - t_cmd_10) * 1000.0))
                 if outside.size > 0 else 0.0)

    # 延迟二阶系统拟合（使用实际指令斜坡卷积，避免把斜坡本身当成延迟）
    fit = {"ok": False}
    try:
        from scipy.optimize import least_squares

        t_fit = t - t[0]

        def _resid(p):
            """非线性最小二乘残差：p = [delay_s, wn_rad_s, zeta, gain]。"""
            pred = _simulate_filtered_step(t_fit, u_norm, p[0], p[1], p[2], gain=p[3])
            return pred - y_norm

        p0 = [max(0.005, (delay_10_ms or 15.0) / 1000.0 * 0.6), 15.0, 0.6,
              float(np.clip(ss_gain, 0.5, 1.5))]
        # 参数物理上下界：delay∈[0, 0.25]s, wn∈[3, 120]rad/s, zeta∈[0.08, 5.0], gain∈[0.5, 1.5]
        bounds = ([0.0, 3.0, 0.08, 0.5], [0.25, 120.0, 5.0, 1.5])
        res = least_squares(_resid, p0, bounds=bounds, max_nfev=400)
        d_s, wn_rad, zeta, gain_f = [float(v) for v in res.x]
        pred = _simulate_filtered_step(t_fit, u_norm, d_s, wn_rad, zeta, gain=gain_f)
        ss_res = float(np.sum((y_norm - pred) ** 2))
        ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        wn_hz = wn_rad / (2.0 * math.pi)
        # 由刚度 kp 与自然频率 wn 反推等效惯量 J_eff，并给出 zeta=0.70 临界偏欠阻尼推荐 kd
        j_eff = (float(kp) / (wn_rad * wn_rad)
                 if (kp is not None and kp > 0.0 and wn_rad > 1e-3) else None)
        kd_zeta07 = (2.0 * 0.70 * math.sqrt(float(kp) * j_eff)
                     if (j_eff is not None and kp is not None) else None)
        fit = {
            "ok": True,
            "delay_ms": round(d_s * 1000.0, 2),
            "wn_hz": round(wn_hz, 3),
            "zeta": round(zeta, 3),
            "gain": round(gain_f, 4),
            "r2": round(r2, 4),
            "j_eff_kgm2": (None if j_eff is None else round(j_eff, 5)),
            "kd_for_zeta_07": (None if kd_zeta07 is None else round(kd_zeta07, 3)),
        }
    except Exception as exc:  # pragma: no cover
        fit = {"ok": False, "reason": str(exc)}

    return {
        "ok": True,
        "cmd_step_deg": round(float(np.rad2deg(cmd_delta)), 3),
        "act_step_deg": round(float(np.rad2deg(act_delta)), 3),
        "ss_gain": round(ss_gain, 4),
        "delay_10_ms": (None if delay_10_ms is None else round(delay_10_ms, 2)),
        "delay_50_ms": (None if delay_50_ms is None else round(delay_50_ms, 2)),
        "rise_ms": (None if rise_ms is None else round(rise_ms, 2)),
        "overshoot_pct": round(overshoot_pct, 2),
        "settle_5pct_ms": round(settle_ms, 2),
        "fit": fit,
    }


def analyze_sine_segment(t, q_cmd, q_act, freq_hz, discard_cycles=1.0):
    """锁相（正交最小二乘正弦投影）提取单频点的幅值比 |H(jw)|、相位滞后（度）与等效延迟（ms）。

    将指令与实测波形分别投影到基函数 ``[1, cos(wt), sin(wt)]`` 上：
    - 常数项 ``c0`` 自动吸收直流重力静差与中心角偏置；
    - 复相量比 ``H(jw) = z_act / z_cmd`` 对宽带测量噪声与高次谐波天然免疫；
    - 丢弃段首 ``discard_cycles`` 个周期以跳过起振暂态与淡入包络。

    参数:
        t (array-like): 采样时间戳序列（单位: s）。
        q_cmd (array-like): 指令关节角序列（单位: rad）。
        q_act (array-like): 实测关节角序列（单位: rad）。
        freq_hz (float): 当前正弦激励频率（单位: Hz）。
        discard_cycles (float): 段首丢弃的暂态周期数（默认 1.0）。

    返回:
        dict: 包含 ``ok``、频点 ``freq_hz``、指令/实际基波幅值（deg）、幅值比 ``mag_ratio``、
            增益 ``mag_db``（dB）、展开后的相位滞后 ``phase_deg``（deg，负值表示滞后）、
            等效延迟 ``equiv_delay_ms``（ms）及基波拟合优度 ``r2``。
    """
    t = np.asarray(t, dtype=np.float64)
    qc = np.asarray(q_cmd, dtype=np.float64)
    qa = np.asarray(q_act, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(qc) & np.isfinite(qa)
    t, qc, qa = t[valid], qc[valid], qa[valid]
    if t.size < 20 or freq_hz <= 0.0:
        return {"ok": False, "freq_hz": float(freq_hz), "reason": "样本不足"}

    t = t - t[0]
    t_cut = float(discard_cycles) / float(freq_hz)
    mask = t >= t_cut
    if int(np.sum(mask)) < 16:
        mask = np.ones_like(t, dtype=bool)
    tm, qcm, qam = t[mask], qc[mask], qa[mask]

    w = 2.0 * math.pi * float(freq_hz)
    # 拟合 c0 + a*cos(wt) + b*sin(wt)，其中 c0 吸收静态重力下垂等直流偏置
    A = np.column_stack([np.ones_like(tm), np.cos(w * tm), np.sin(w * tm)])
    coef_c, *_ = np.linalg.lstsq(A, qcm, rcond=None)
    coef_a, *_ = np.linalg.lstsq(A, qam, rcond=None)
    # 复相量：a*cos(wt) + b*sin(wt) = Re((a - j*b) * e^{jwt})
    z_cmd = complex(float(coef_c[1]), -float(coef_c[2]))
    z_act = complex(float(coef_a[1]), -float(coef_a[2]))
    amp_cmd = abs(z_cmd)
    amp_act = abs(z_act)
    if amp_cmd < np.deg2rad(0.15):
        return {"ok": False, "freq_hz": float(freq_hz), "reason": "激励幅值过小"}

    H = z_act / z_cmd
    mag = abs(H)
    phase_rad = math.atan2(H.imag, H.real)
    # 相位滞后展开到 (-2*pi, 0.35]（允许高频滞后超过 180°，同时容忍轻微超前误差）
    if phase_rad > 0.35:
        phase_rad -= 2.0 * math.pi
    phase_deg = math.degrees(phase_rad)
    equiv_delay_ms = (-phase_rad / w) * 1000.0
    # 拟合质量（基波占比）：R2 越接近 1 说明非线性/摩擦畸变越小
    pred_a = A @ coef_a
    ss_res = float(np.sum((qam - pred_a) ** 2))
    ss_tot = float(np.sum((qam - np.mean(qam)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "ok": True,
        "freq_hz": round(float(freq_hz), 3),
        "cmd_amp_deg": round(float(np.rad2deg(amp_cmd)), 3),
        "act_amp_deg": round(float(np.rad2deg(amp_act)), 3),
        "mag_ratio": round(float(mag), 4),
        "mag_db": round(float(20.0 * math.log10(max(mag, 1e-6))), 2),
        "phase_deg": round(float(phase_deg), 2),
        "equiv_delay_ms": round(float(equiv_delay_ms), 2),
        "r2": round(float(r2), 4),
    }


def safe_sine_amp_rad(amp_deg, freq_hz):
    """按硬幅值上限与峰值速度上限 0.80 rad/s 计算安全扫频幅值（弧度）。

    高频段自动按 ``A <= MAX_SINE_PEAK_VEL_RAD_S / (2*pi*f)`` 收缩幅值，防止高速正弦
    引发过流或触发底层限速截断（削波会破坏正弦基波假设）。

    参数:
        amp_deg (float): 期望扫频幅值（单位: deg）。
        freq_hz (float): 激励频率（单位: Hz）。

    返回:
        float: 钳位后的安全正弦幅值（单位: rad）。
    """
    amp = np.deg2rad(min(float(amp_deg), MAX_SINE_AMP_DEG))
    vel_cap = MAX_SINE_PEAK_VEL_RAD_S / max(2.0 * math.pi * float(freq_hz), 1e-6)
    return float(min(amp, vel_cap))


# ─────────────────────────────────────────────────────────────
# 离线仿真自检（不访问硬件）
# ─────────────────────────────────────────────────────────────
def run_offline_simulation():
    """用已知参数的延迟二阶合成系统验证阶跃拟合与扫频锁相算法精度（零硬件依赖）。

    合成真值：``delay = 15.0 ms``、``wn = 1.80 Hz``、``zeta = 0.35``、``kp = 72.0 Nm/rad``，
    并在正弦响应中叠加直流重力偏置验证锁相抗直流偏移能力。

    返回:
        int: 固定返回 0。
    """
    dt = 0.005
    t = np.arange(0.0, 1.4, dt)
    u = np.clip((t - 0.20) / 0.02, 0.0, 1.0)
    step_rad = np.deg2rad(2.5)
    kp_j1 = 72.0
    true_delay = 0.015
    true_wn = 2.0 * math.pi * 1.80
    true_zeta = 0.35
    q_cmd = u * step_rad
    q_act = _simulate_filtered_step(t, u, true_delay, true_wn, true_zeta, gain=1.0) * step_rad
    step_res = analyze_step_response(t, q_cmd, q_act, kp=kp_j1, pre_s=0.15)

    sine_results = []
    for f in DEFAULT_FREQS_HZ:
        cycles = max(3.5, min(6.0, f * 2.0))
        ts = np.arange(0.0, cycles / f, dt)
        amp = safe_sine_amp_rad(2.0, f)
        w = 2.0 * math.pi * f
        # 二阶传递函数 H(jw) = wn^2 / (wn^2 - w^2 + j*2*zeta*wn*w) * exp(-j*w*delay)
        H = (true_wn ** 2 / ((true_wn ** 2 - w ** 2) + 1j * 2.0 * true_zeta * true_wn * w)
             * np.exp(-1j * w * true_delay))
        qc = amp * np.sin(w * ts)
        qa = amp * abs(H) * np.sin(w * ts + math.atan2(H.imag, H.real)) + 0.015  # 带直流偏置
        sine_results.append(analyze_sine_segment(ts, qc, qa, f))

    print("[离线仿真自检] 真值: delay=15.0ms, wn=1.80Hz, zeta=0.35, kp=72.0")
    print("  阶跃拟合:", json.dumps(step_res, ensure_ascii=False))
    print("  扫频结果:")
    for sr in sine_results:
        print(f"    f={sr['freq_hz']:4.1f}Hz: |H|={sr['mag_ratio']:.3f} ({sr['mag_db']:+5.2f}dB), "
              f"phase={sr['phase_deg']:+7.2f}°, eq_delay={sr['equiv_delay_ms']:5.1f}ms")
    return 0


# ─────────────────────────────────────────────────────────────
# 实机安全激励流程
# ─────────────────────────────────────────────────────────────
def _check_safety_guard(robot, q_center, joint_idx, max_dev_rad):
    """实机激励期高频安全看护：监测 MOS 过温与被测关节异常偏离。

    参数:
        robot (RebotAdapter): 机械臂适配器。
        q_center (np.ndarray): 激励基准中心角（6 轴，单位: rad）。
        joint_idx (int): 当前被测关节 0-based 索引（0~5）。
        max_dev_rad (float): 允许偏离中心角的最大绝对值（单位: rad，通常为幅值 + 6°）。

    异常:
        RuntimeError: 当任意电机 MOS 温度 ``>= MAX_MOS_TEMP_C`` 或被测关节偏离超出
            ``max_dev_rad`` 时抛出，触发外层 ``finally`` 恢复限速并安全回零。
    """
    fb = robot.get_feedback_state(max_age_s=0.15)
    if fb is None or not fb["valid"]:
        return
    t_mos = fb["t_mos"][np.isfinite(fb["t_mos"])]
    if t_mos.size and float(np.max(t_mos)) >= MAX_MOS_TEMP_C:
        raise RuntimeError(f"MOS 温度过高 ({float(np.max(t_mos)):.1f}°C ≥ {MAX_MOS_TEMP_C}°C)")
    dev = abs(float(fb["q"][joint_idx]) - float(q_center[joint_idx]))
    if dev > max_dev_rad:
        raise RuntimeError(
            f"J{joint_idx + 1} 偏离中心角过大 ({math.degrees(dev):.2f}° > "
            f"{math.degrees(max_dev_rad):.2f}°)")


def _sleep_guarded(robot, duration_s, q_center, joint_idx, max_dev_rad):
    """带 50Hz 安全看护轮询的等待函数，替代裸 ``time.sleep``。

    参数:
        robot (RebotAdapter): 机械臂适配器。
        duration_s (float): 期望等待时长（单位: s）。
        q_center (np.ndarray): 激励基准中心角（6 轴，单位: rad）。
        joint_idx (int): 当前被测关节 0-based 索引（0~5）。
        max_dev_rad (float): 允许最大角度偏离（单位: rad）。
    """
    t_end = time.monotonic() + float(duration_s)
    while time.monotonic() < t_end:
        _check_safety_guard(robot, q_center, joint_idx, max_dev_rad)
        time.sleep(min(0.02, max(0.0, t_end - time.monotonic())))


def _run_joint_step(robot, q_center, joint_idx, step_rad, kp, hold_s=1.0,
                    step_speed_rad_s=1.8):
    """对单个关节依次执行正向（``pos``）与负向（``neg``）小幅阶跃并录波辨识。

    其余 5 个非被测轴保持在 ``q_center`` 且限速压至 0.5 rad/s；被测轴临时放宽至
    ``min(step_speed_rad_s, 1.8) rad/s`` 使斜坡在 ~20ms 内完成。无论成败 ``finally``
    都会回归中心位并恢复原有限速。

    参数:
        robot (RebotAdapter): 机械臂适配器。
        q_center (np.ndarray): 6 轴中心位姿（单位: rad）。
        joint_idx (int): 被测关节索引（0~5）。
        step_rad (float): 阶跃幅值（单位: rad）。
        kp (float): 被测轴当前 MIT 位置刚度（单位: Nm/rad）。
        hold_s (float): 阶跃后稳态保持采样时长（单位: s，默认 1.0s）。
        step_speed_rad_s (float): 被测轴阶跃斜坡限速上限（单位: rad/s，默认 1.8）。

    返回:
        tuple[dict, dict]: ``(results, raw_segments)``，分别包含 ``pos/neg`` 两向分析指标
            与原始 200Hz 波形字典（供 ``.npz`` 归档）。
    """
    max_dev = abs(step_rad) + np.deg2rad(6.0)
    # 临时提高被测轴斜坡限速，使指令阶跃本身在 ~20ms 内完成（仍远低于 HARD_MAX_JOINT_SPEED）
    limits = np.full(6, 0.5, dtype=np.float64)
    limits[joint_idx] = min(float(step_speed_rad_s), 1.8)
    robot.override_joint_speed_limits(limits)

    results = {}
    raw_segments = {}
    try:
        for tag, sign in (("pos", +1.0), ("neg", -1.0)):
            q_step = q_center.copy()
            q_step[joint_idx] = q_center[joint_idx] + sign * step_rad
            robot.set_joint_targets_direct(q_center)
            _sleep_guarded(robot, 0.35, q_center, joint_idx, max_dev)

            # 录制 0.20s 阶跃前基线 + hold_s 阶跃响应
            robot.start_feedback_recording()
            _sleep_guarded(robot, 0.20, q_center, joint_idx, max_dev)
            robot.set_joint_targets_direct(q_step)
            _sleep_guarded(robot, hold_s, q_center, joint_idx, max_dev)
            rec = robot.stop_feedback_recording()

            # 平滑回中心位
            robot.set_joint_targets_direct(q_center)
            _sleep_guarded(robot, 0.50, q_center, joint_idx, max_dev)

            results[tag] = analyze_step_response(
                rec["t"], rec["q_cmd"][:, joint_idx], rec["q"][:, joint_idx],
                kp=kp, pre_s=0.15)
            raw_segments[tag] = {
                "t": rec["t"],
                "q_cmd": rec["q_cmd"][:, joint_idx],
                "q_act": rec["q"][:, joint_idx],
                "dq_act": rec["dq"][:, joint_idx],
                "tau_act": rec["tau"][:, joint_idx],
            }
    finally:
        robot.set_joint_targets_direct(q_center)
        robot.restore_joint_speed_limits()
    return results, raw_segments


def _run_joint_sine(robot, q_center, joint_idx, amp_deg, freqs_hz,
                    with_vel_ff=False):
    """对单个关节按离散频点列表执行 100Hz 正弦扫频激励并提取幅相频响。

    支持两种激励模式用于 P0 vs P1 A/B 对比：
    - ``with_vel_ff=False``（P0 基线）：仅发送位置参考 ``q_des``，底层 ``dq_des = 0``；
    - ``with_vel_ff=True``（P1 速度前馈）：按 100Hz 积分器约定将 ``q_des`` 对齐到区间末端
      ``elapsed + dt``、解析速度 ``dq_des`` 取区间中点 ``elapsed + 0.5*dt``，与 200Hz FOH
      插值器无相位失配地配合。

    参数:
        robot (RebotAdapter): 机械臂适配器。
        q_center (np.ndarray): 6 轴中心位姿（单位: rad）。
        joint_idx (int): 被测关节索引（0~5）。
        amp_deg (float): 目标扫频幅值（单位: deg，内部经 ``safe_sine_amp_rad`` 速度限幅）。
        freqs_hz (Sequence[float]): 扫频离散频率列表（单位: Hz）。
        with_vel_ff (bool): 是否启用 P1 解析速度前馈透传。

    返回:
        tuple[list[dict], dict]: ``(freq_results, raw_sines)``，分别为各频点锁相辨识结果
            与原始 200Hz 波形字典。
    """
    limits = np.full(6, 0.5, dtype=np.float64)
    limits[joint_idx] = 1.5
    robot.override_joint_speed_limits(limits)

    freq_results = []
    raw_sines = {}
    try:
        for f in freqs_hz:
            amp_rad = safe_sine_amp_rad(amp_deg, f)
            max_dev = amp_rad + np.deg2rad(6.0)
            # 低频保证至少 3.5 个周期（扣 1 周期暂态后仍余 >=2.5 周期），高频封顶 6 周期控温
            cycles = max(3.5, min(6.0, float(f) * 2.0))
            duration_s = cycles / float(f)
            robot.set_joint_targets_direct(q_center)
            _sleep_guarded(robot, 0.25, q_center, joint_idx, max_dev)

            robot.start_feedback_recording()
            t0 = time.monotonic()
            next_t = t0
            dt = 0.01  # 100Hz 喂正弦参考（模拟视觉伺服控制线程更新节拍）
            w = 2.0 * math.pi * float(f)
            while True:
                now = time.monotonic()
                elapsed = now - t0
                if elapsed >= duration_s:
                    break
                _check_safety_guard(robot, q_center, joint_idx, max_dev)
                if with_vel_ff:
                    # 对齐 100Hz 积分器约定：target 落在区间末端 elapsed+dt，供 200Hz FOH 连续插值
                    t_end = min(duration_s, elapsed + dt)
                    t_mid = min(duration_s, elapsed + 0.5 * dt)
                    ramp_e = min(1.0, t_end / 0.15, (duration_s - t_end) / 0.15)
                    ramp_m = min(1.0, t_mid / 0.15, (duration_s - t_mid) / 0.15)
                    q_des = q_center.copy()
                    q_des[joint_idx] = (q_center[joint_idx]
                                        + ramp_e * amp_rad * math.sin(w * t_end))
                    dq_des = np.zeros(6, dtype=np.float64)
                    dq_des[joint_idx] = ramp_m * amp_rad * w * math.cos(w * t_mid)
                    robot.set_joint_targets_direct(q_des, dq_rad=dq_des)
                else:
                    # 淡入淡出包络（首尾 0.15s），避免起停冲击
                    ramp = min(1.0, elapsed / 0.15, (duration_s - elapsed) / 0.15)
                    q_des = q_center.copy()
                    q_des[joint_idx] = (q_center[joint_idx]
                                        + ramp * amp_rad * math.sin(w * elapsed))
                    robot.set_joint_targets_direct(q_des)
                next_t += dt
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
            rec = robot.stop_feedback_recording()
            robot.set_joint_targets_direct(q_center)
            _sleep_guarded(robot, 0.30, q_center, joint_idx, max_dev)

            seg_res = analyze_sine_segment(
                rec["t"], rec["q_cmd"][:, joint_idx], rec["q"][:, joint_idx], f,
                discard_cycles=1.0)
            freq_results.append(seg_res)
            raw_sines[f"{f:.2f}Hz"] = {
                "t": rec["t"],
                "q_cmd": rec["q_cmd"][:, joint_idx],
                "q_act": rec["q"][:, joint_idx],
                "dq_act": rec["dq"][:, joint_idx],
                "tau_act": rec["tau"][:, joint_idx],
            }
    finally:
        robot.set_joint_targets_direct(q_center)
        robot.restore_joint_speed_limits()
    return freq_results, raw_sines


def main():
    """命令行入口：执行离线仿真自检或实机逐轴小幅阶跃/正弦扫频系统辨识。

    安全红线:
        - 参数校验严格限制阶跃 ``<= 4.0°``、扫频幅值 ``<= 3.0°``；
        - 实机模式默认要求交互式输入 ``yes`` 二次确认，并校验基准位姿水平半径 ``<= 0.38m``、
          高度 ``z >= 0.08m`` 及各被测轴距软件限位至少保留 5° 余量；
        - 退出路径（含 Ctrl+C 或看护异常）始终先恢复关节限速，再执行 ``hold() -> stop()``
          缓慢回零后失能。

    返回:
        int: 进程退出码（0 表示成功或用户取消，1 表示参数非法、环境/位姿不满足安全条件）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true",
                        help="离线仿真自检（不访问硬件）")
    parser.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                        help="要辨识的关节列表（1~6），默认全部 6 轴")
    parser.add_argument("--pose", choices=("home", "extended", "current"), default="home",
                        help="激励基准位姿：home=config/home_pose.json（默认），"
                             "extended=安全中伸展位姿（r≈0.31m，专测 J1 伸展共振），current=保持当前姿态")
    parser.add_argument("--step-deg", type=float, default=2.5,
                        help=f"阶跃幅值 (度)，硬上限 {MAX_STEP_DEG}°")
    parser.add_argument("--sine-deg", type=float, default=2.0,
                        help=f"扫频幅值 (度)，硬上限 {MAX_SINE_AMP_DEG}°")
    parser.add_argument("--no-step", action="store_true",
                        help="跳过阶跃辨识，仅做静态重力 + 正弦扫频")
    parser.add_argument("--no-sine", action="store_true",
                        help="仅做静态重力 + 阶跃辨识，跳过正弦扫频")
    parser.add_argument("--with-vel-ff", action="store_true",
                        help="P1 对比模式：正弦扫频同步透传解析速度前馈 dq_des（直接复测相位滞后下降）")
    parser.add_argument("--freqs", type=float, nargs="+", default=list(DEFAULT_FREQS_HZ),
                        help="扫频频率列表 (Hz)")
    parser.add_argument("--confirm", action="store_true",
                        help="跳过交互式回车确认（确认机械臂周围安全后使用）")
    args = parser.parse_args()

    if args.simulate:
        return run_offline_simulation()

    if any(j < 1 or j > 6 for j in args.joints):
        print("❌ --joints 必须在 1~6 范围内")
        return 1
    if not (0.5 <= args.step_deg <= MAX_STEP_DEG):
        print(f"❌ --step-deg 必须在 [0.5, {MAX_STEP_DEG}] 范围内")
        return 1
    if not (0.5 <= args.sine_deg <= MAX_SINE_AMP_DEG):
        print(f"❌ --sine-deg 必须在 [0.5, {MAX_SINE_AMP_DEG}] 范围内")
        return 1

    print("=" * 70)
    print(f"reBot B601-RS 单关节小幅辨识工具 ({'v13 P1 vel_ff' if args.with_vel_ff else 'v13 P0 基线'})")
    print(f"  测试位姿: {args.pose} | 关节: {args.joints}")
    print(f"  阶跃幅值: {'跳过' if args.no_step else f'±{args.step_deg:.1f}°'} | "
          f"扫频幅值: {'跳过' if args.no_sine else f'±{args.sine_deg:.1f}° {args.freqs}'}")
    print("  ⚠️ 请确认机械臂工作区无障碍物、手边有电源开关；随时可按 Ctrl+C 安全回零")
    print("=" * 70)
    if not args.confirm:
        ans = input("输入 yes 确认开始实机辨识，其他任意键退出: ").strip().lower()
        if ans != "yes":
            print("已取消。")
            return 0

    from rebot_adapter import RebotAdapter

    robot = None
    try:
        robot = RebotAdapter()
        ext_info = robot.mit_ext_info()
        if not ext_info.get("active"):
            print("❌ MIT 扩展未启用，无法录制 200Hz 真实反馈。请先确认 config/mit_ext.yaml")
            return 1

        # 确认反馈有效（如果默认 poll_once=False 下无效，提示先跑 mit_feedback_probe）
        time.sleep(0.2)
        fb0 = robot.get_feedback_state(max_age_s=0.15)
        if fb0 is None or not fb0["valid"]:
            print("❌ 200Hz 反馈缓存当前无有效数据！请先运行 tools/mit_feedback_probe.py "
                  "确认是否需要开启 feedback_poll_once: true")
            return 1

        home_file = os.path.join(ROOT, "config", "home_pose.json")
        with open(home_file, encoding="utf-8") as stream:
            q_home = np.array(json.load(stream)["home_joint_pos"], dtype=np.float64)

        if args.pose == "current":
            q_center = robot.get_current_pos()
        else:
            q_center = q_home.copy()
            if args.pose == "extended":
                # 从 HOME 温和前伸（收小肘屈/抬肩），先用 FK 校验水平半径 ≤ 0.34m 且 z ≥ 0.15m
                q_try = q_home.copy()
                q_try[1] = min(float(robot.q_max[1]) - np.deg2rad(10.0), float(q_home[1]) + np.deg2rad(20.0))
                q_try[2] = max(float(robot.q_min[2]) + np.deg2rad(10.0), float(q_home[2]) + np.deg2rad(12.0))
                q_try[3] = float(np.clip(q_home[3] - np.deg2rad(15.0),
                                         robot.q_min[3] + np.deg2rad(10.0),
                                         robot.q_max[3] - np.deg2rad(10.0)))
                q_center = q_try
            print(f"[辨识] 正在以 15°/s 移至目标基准位姿 {np.rad2deg(q_center).round(1).tolist()}° ...")
            robot.move_to_joints(q_center, speed_deg_s=15.0, wait=True, timeout=20.0)
            time.sleep(0.5)

        # 读取真实基准位姿与重力核验，并同步水平伸展半径供 P3 阻尼调度生效
        fk = robot.forward_kinematics()
        tcp_r = float(np.hypot(fk["position"][0], fk["position"][1]))
        if hasattr(robot, "set_reach_radius"):
            robot.set_reach_radius(tcp_r)
        print(f"[辨识] 基准 TCP={fk['position'].round(3).tolist()} m (水平半径 r={tcp_r:.3f}m)")
        if tcp_r > 0.38 or float(fk["position"][2]) < 0.08:
            print(f"❌ 基准位姿过于靠近工作区边界 (r={tcp_r:.3f}m, z={fk['position'][2]:.3f}m)，中止！")
            return 1

        q_0x7019 = robot.get_current_pos()
        q_cmd0 = robot.get_command_pos()
        fb_static = robot.get_feedback_state(max_age_s=0.15)
        tau_pin = robot.gravity_torque(q_0x7019)
        kp, kd = robot.get_mit_gains()

        static_summary = {
            "tcp_m": fk["position"].round(4).tolist(),
            "tcp_radius_m": round(tcp_r, 4),
            "q_cmd_deg": np.rad2deg(q_cmd0).round(3).tolist(),
            "q_act_deg": np.rad2deg(q_0x7019).round(3).tolist(),
            "sag_deg": np.rad2deg(q_cmd0 - q_0x7019).round(3).tolist(),
            "tau_pin_nm": np.round(tau_pin, 4).tolist(),
            "tau_act_nm": (np.round(fb_static["tau"], 4).tolist()
                           if (fb_static and fb_static["valid"]) else None),
        }

        step_rad = np.deg2rad(args.step_deg)
        margin_rad = np.deg2rad(LIMIT_SAFETY_MARGIN_DEG)
        joint_reports = {}
        npz_payload = {
            "q_center_rad": q_center,
            "kp": kp,
            "kd": kd,
            "tau_pin_nm": tau_pin,
        }

        if hasattr(robot.io.arm, "set_vel_ff_enable"):
            robot.io.arm.set_vel_ff_enable(bool(args.with_vel_ff))

        # 逐轴执行阶跃与正弦扫频，每轴激励前均检查距软件限位的双向安全余量
        for j_num in args.joints:
            ji = j_num - 1
            need = max(step_rad, np.deg2rad(args.sine_deg)) + margin_rad
            if (q_center[ji] - need < robot.q_min[ji]
                    or q_center[ji] + need > robot.q_max[ji]):
                print(f"[辨识] ⚠️ 跳过 J{j_num}：距软件限位余量不足 {math.degrees(need):.1f}°")
                joint_reports[f"J{j_num}"] = {"skipped": True, "reason": "限位余量不足"}
                continue

            step_res = {}
            if not args.no_step:
                print(f"\n[辨识] ── J{j_num}（kp={kp[ji]:.1f}, kd={kd[ji]:.1f}）阶跃 ±{args.step_deg:.1f}° ──")
                step_res, raw_step = _run_joint_step(
                    robot, q_center, ji, step_rad, kp=float(kp[ji]))
                for tag in ("pos", "neg"):
                    sr = step_res[tag]
                    if sr.get("ok"):
                        f = sr["fit"]
                        print(f"  step({tag}): d10={sr['delay_10_ms']}ms, rise={sr['rise_ms']}ms, "
                              f"ov={sr['overshoot_pct']}%, settle={sr['settle_5pct_ms']}ms | "
                              f"fit: delay={f.get('delay_ms')}ms, wn={f.get('wn_hz')}Hz, "
                              f"zeta={f.get('zeta')}, J_eff={f.get('j_eff_kgm2')}, "
                              f"kd@0.7={f.get('kd_for_zeta_07')} (R2={f.get('r2')})")
                    else:
                        print(f"  step({tag}) 失败: {sr}")
                for tag, seg in raw_step.items():
                    for k, arr in seg.items():
                        npz_payload[f"j{j_num}_step_{tag}_{k}"] = arr

            sine_res = []
            if not args.no_sine:
                mode_tag = " [P1 vel_ff=ON]" if args.with_vel_ff else " [P0 vel_ff=OFF]"
                print(f"[辨识] ── J{j_num} 离散正弦扫频 {args.freqs} Hz{mode_tag} ──")
                sine_res, raw_sine = _run_joint_sine(
                    robot, q_center, ji, args.sine_deg, args.freqs,
                    with_vel_ff=bool(args.with_vel_ff))
                for sr in sine_res:
                    if sr.get("ok"):
                        print(f"  f={sr['freq_hz']:4.1f}Hz: |H|={sr['mag_ratio']:.3f} "
                              f"({sr['mag_db']:+5.2f}dB), phase={sr['phase_deg']:+7.2f}°, "
                              f"eq_delay={sr['equiv_delay_ms']:5.1f}ms (R2={sr['r2']:.3f})")
                for ftag, seg in raw_sine.items():
                    clean_tag = ftag.replace(".", "p")
                    for k, arr in seg.items():
                        npz_payload[f"j{j_num}_sine_{clean_tag}_{k}"] = arr

            joint_reports[f"J{j_num}"] = {
                "kp": float(kp[ji]),
                "kd": float(kd[ji]),
                "step": step_res,
                "sine": sine_res,
            }

        stats = robot.get_mit_loop_stats() or {}
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_vff" if args.with_vel_ff else ""
        json_path = os.path.join(ROOT, "logs", f"sysid_{args.pose}{suffix}_{stamp}.json")
        npz_path = os.path.join(ROOT, "logs", f"sysid_{args.pose}{suffix}_{stamp}.npz")
        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pose": args.pose,
            "with_vel_ff": bool(args.with_vel_ff),
            "vel_ff_scale": list(ext_info["cfg"].get("vel_ff_scale", [])),
            "step_deg": args.step_deg,
            "sine_deg": args.sine_deg,
            "freqs_hz": ([] if args.no_sine else list(args.freqs)),
            "static": static_summary,
            "mit_loop_stats": stats,
            "joints": joint_reports,
        }
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        np.savez_compressed(npz_path, **npz_payload)

        print("\n" + "=" * 70)
        print(f"单关节辨识完成！报告: {json_path}")
        print(f"原始 200Hz 波形:      {npz_path}")
        print(f"MIT 200Hz 环路统计:   period_p99={stats.get('period_p99_ms')}ms, "
              f"can_fail_total={stats.get('can_fail_total')}")
        print("=" * 70)
        return 0
    finally:
        # 硬件安全红线：先还原被测轴临时限速，再 hold 稳住姿态并通过 stop() 安全回零失能
        if robot is not None:
            print("[辨识] 执行安全停机（恢复限速 → 保持 → 缓慢回零 → 失能）...")
            try:
                robot.restore_joint_speed_limits()
            except Exception:
                pass
            try:
                robot.hold()
                time.sleep(0.2)
            finally:
                robot.stop()


if __name__ == "__main__":
    sys.exit(main())
