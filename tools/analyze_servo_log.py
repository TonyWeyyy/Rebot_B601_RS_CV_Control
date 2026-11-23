#!/usr/bin/env python3
"""离线分析 100Hz servo_diagnostics CSV 日志，不连接相机、CAN 或机械臂。

汇总口径说明：
1. 运动与稳态静止拆分：
   - 仅统计 ``following=1`` 且 ``visible=1`` 的有效伺服段；
   - 静止抖动指标扣除停手后前 0.30s 停靠/刹车过渡窗，且要求 ``hold_latch=1``、
     非慢拖（``slow_active=0``）、非运动（``ff_moving=0``）且 KF 残速 ``< 0.06 m/s``。
2. 停手稳定时间（``stop_settle``）：
   - 捕捉 ``ff_moving`` 由 1 跳变至 0 的每次停手事件，寻找连续 20 拍（0.20s）关节速度
     ``< 0.05 rad/s`` 的耗时；
   - 若期间 ``ff_moving`` 重新置 1（快挥折返），立即中止当前扫描，防止跨波次错算至下一次停手。
3. 低频同步反馈（旧 ``ActualFeedbackSampler``）按采样时间戳去重统计跟踪误差与 CAN 读耗时。
4. v13 MIT 200Hz 反馈缓存指标（兼容老日志，缺列时优雅返回 ``None``）：
   - 统计缓存有效率、数据新鲜度 ``fbc_age_ms`` p95、六轴最大跟踪误差/静态下垂 ``|q_cmd - qa|``（deg）、
     逐轴峰值力矩（Nm）、最高 MOS 温度（°C）、200Hz 环路周期 p99 最大值（ms）与累计 CAN 异常数。
"""

import argparse
import csv
import json
import math
from pathlib import Path


def number(row, key, default=0.0):
    """安全解析 CSV 单元格为有限浮点数；缺失、非法或 NaN/Inf 时返回 ``default``。"""
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def flag(row, key):
    """将 CSV 布尔/整型标记列（``1``/``true``/``yes``，忽略大小写）解析为 ``bool``。"""
    return str(row.get(key, "")).strip().lower() in ("1", "true", "yes")


def percentile(values, fraction):
    """计算列表的最近邻分位数（``fraction`` 取 ``0.0~1.0``）；空列表返回 ``None``。"""
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def summarize(path):
    """读取单个 100Hz 伺服诊断 CSV 并返回全量离线评估指标字典。

    参数:
        path (str | Path): ``DiagnosticRecorder`` 生成的 ``logs/servo_*.csv`` 文件路径。

    返回:
        dict: 包含样本数、时长（s）、运动误差 median/p95（m）、扣除 0.30s 停靠窗后的
            静止关节速度 p95（rad/s）及非零比例、停手稳定时间 median（s）、观测/处理/控制
            延迟 p95（ms）、超时率、低频同步反馈指标以及 v13 200Hz MIT 缓存汇总指标。

    异常:
        ValueError: CSV 文件没有任何数据行时抛出。
    """
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("日志没有数据行")
    active = [row for row in rows if flag(row, "following") and flag(row, "visible")]
    moving = [row for row in active if flag(row, "ff_moving")]
    # 慢拖和停手后的短暂停靠都是刻意运动；静止指标只取连续静止 0.30s 后
    # 且 KF 残速低于下沿的样本，停靠过程由 stop_settle 指标单独衡量。
    static = []
    static_since = None
    for row in active:
        kf_speed = math.sqrt(sum(number(row, f"kv_{axis}") ** 2 for axis in "xyz"))
        eligible = (flag(row, "hold_latch") and not flag(row, "slow_active")
                    and not flag(row, "ff_moving") and kf_speed < 0.06)
        if not eligible:
            static_since = None
            continue
        now = number(row, "host_time")
        if static_since is None:
            static_since = now
        if now - static_since >= 0.30:
            static.append(row)
    moving_errors = [number(row, "error_norm") for row in moving]
    static_speeds = [number(row, "joint_speed") for row in static]

    # 停手稳定时间统计：检测 ff_moving 1 -> 0 下降沿，寻找连续 20 帧（0.20s）低速停稳窗口
    settle_times = []
    for index in range(1, len(active)):
        if not flag(active[index - 1], "ff_moving") or flag(active[index], "ff_moving"):
            continue
        # 若停顿不足 0.20s 手掌即重新运动（快挥折返），不跨段扫描到十几秒后的下一次停手。
        for count in range(1, len(active) - index + 1):
            cur = active[index + count - 1]
            if flag(cur, "ff_moving"):
                break
            if count < 20:
                continue
            window = active[index:index + count]
            if all(number(row, "joint_speed") < 0.05 for row in window[-20:]):
                settle_times.append(number(window[-1], "host_time") -
                                    number(active[index], "host_time"))
                break

    # 同一低频反馈会被 100Hz 控制日志重复引用；按采样时刻去重后再统计。
    feedback_rows = []
    feedback_seen = set()
    feedback_valid_rows = 0
    for row in rows:
        if not flag(row, "fb_valid"):
            continue
        feedback_valid_rows += 1
        stamp = str(row.get("fb_time", "")).strip()
        if stamp:
            key = ("time", stamp)
        else:
            key = ("values", tuple(row.get(f"q_act_{joint}", "") for joint in range(1, 7)),
                   tuple(row.get(f"q_fb_cmd_{joint}", "") for joint in range(1, 7)))
        if key not in feedback_seen:
            feedback_seen.add(key)
            feedback_rows.append(row)
    feedback_reads = [number(row, "fb_read_ms") for row in feedback_rows]
    tracking_errors = [number(row, "tracking_err_max") for row in feedback_rows]

    # v13 P0：MIT 扩展 200Hz 应答缓存指标（老版本日志无此列时优雅返回 None）
    fbc_rows = [row for row in rows if flag(row, "fbc_valid")]
    fbc_active_rows = [row for row in active if flag(row, "fbc_valid")]
    fbc_ages = [number(row, "fbc_age_ms", math.nan) for row in fbc_rows]
    fbc_ages = [v for v in fbc_ages if math.isfinite(v)]
    # 统计伺服激活期每拍六轴最大指令-实际跟踪误差（deg），反映动态跟随滞后与静态重力下垂
    fbc_track_deg = []
    for row in fbc_active_rows:
        diffs = []
        for joint in range(1, 7):
            qc = number(row, f"q_cmd_{joint}", math.nan)
            qa = number(row, f"qa_{joint}", math.nan)
            if math.isfinite(qc) and math.isfinite(qa):
                diffs.append(abs(math.degrees(qc - qa)))
        if len(diffs) == 6:
            fbc_track_deg.append(max(diffs))
    # 逐关节提取全程峰值绝对力矩（Nm），用于检查是否接近过载/堵转阈值
    tau_abs_max = None
    if fbc_rows:
        peaks = []
        for joint in range(1, 7):
            vals = [abs(number(row, f"tau_{joint}", math.nan)) for row in fbc_rows]
            vals = [v for v in vals if math.isfinite(v)]
            peaks.append(round(max(vals), 3) if vals else None)
        tau_abs_max = peaks
    # 提取全日志最高 MOS 温度（°C）、200Hz 环路周期 p99 最大值（ms）与 CAN 累计失败次数
    tmos_vals = [number(row, "tmos_max", math.nan) for row in rows]
    tmos_vals = [v for v in tmos_vals if math.isfinite(v)]
    mit_p99_vals = [number(row, "mit_period_p99_ms", math.nan) for row in rows]
    mit_p99_vals = [v for v in mit_p99_vals if math.isfinite(v)]
    can_fail_vals = [number(row, "can_fail_total", math.nan) for row in rows]
    can_fail_vals = [int(v) for v in can_fail_vals if math.isfinite(v)]
    duration = number(rows[-1], "host_time") - number(rows[0], "host_time")
    result = {
        "file": str(path), "samples": len(rows), "duration_s": duration,
        "active_samples": len(active), "moving_samples": len(moving),
        "static_samples": len(static),
        "moving_error_median_m": percentile(moving_errors, 0.50),
        "moving_error_p95_m": percentile(moving_errors, 0.95),
        "static_joint_speed_p95_rad_s": percentile(static_speeds, 0.95),
        "static_nonzero_ratio": (sum(value > 0.05 for value in static_speeds) /
                                  len(static_speeds) if static_speeds else None),
        "stop_settle_median_s": percentile(settle_times, 0.50),
        "stop_settle_events": len(settle_times),
        "obs_age_p95_ms": percentile([number(row, "obs_age_ms") for row in active], 0.95),
        "processing_p95_ms": percentile([number(row, "processing_ms") for row in active], 0.95),
        "control_work_p95_ms": percentile([number(row, "ctrl_work_ms") for row in rows], 0.95),
        "overrun_ratio": sum(flag(row, "overrun") for row in rows) / len(rows),
        "hardware_time_ratio": (sum(flag(row, "hw_time_used") for row in active) /
                                 len(active) if active else None),
        "slow_follow_samples": sum(flag(row, "slow_active") for row in active),
        "static_hold_samples": sum(flag(row, "static_hold") for row in active),
        "feedback_valid_rows": feedback_valid_rows,
        "feedback_samples": len(feedback_rows),
        "feedback_read_p95_ms": percentile(feedback_reads, 0.95),
        "feedback_tracking_median_rad": percentile(tracking_errors, 0.50),
        "feedback_tracking_p95_rad": percentile(tracking_errors, 0.95),
        "feedback_tracking_max_rad": max(tracking_errors) if tracking_errors else None,
        "mit_fbc_valid_ratio": (len(fbc_rows) / len(rows) if "fbc_valid" in rows[0] else None),
        "mit_fbc_age_p95_ms": percentile(fbc_ages, 0.95),
        "mit_tracking_err_median_deg": percentile(fbc_track_deg, 0.50),
        "mit_tracking_err_p95_deg": percentile(fbc_track_deg, 0.95),
        "mit_tracking_err_max_deg": max(fbc_track_deg) if fbc_track_deg else None,
        "mit_tau_abs_max_nm": tau_abs_max,
        "mit_tmos_max_c": max(tmos_vals) if tmos_vals else None,
        "mit_period_p99_ms_max": max(mit_p99_vals) if mit_p99_vals else None,
        "mit_can_fail_total": max(can_fail_vals) if can_fail_vals else None,
        "errors": sorted({row.get("error", "") for row in rows if row.get("error", "")}),
    }
    return result


def main():
    """命令行入口：批量汇总一个或多个伺服诊断 CSV，支持人类可读文本或 ``--json`` 输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="+", help="一个或多个 servo CSV 日志文件")
    parser.add_argument("--json", action="store_true", help="输出 JSON 便于 A/B 脚本处理")
    args = parser.parse_args()
    results = [summarize(p) for p in args.log]
    if args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for idx, result in enumerate(results):
        if idx > 0:
            print("-" * 60)
        print(f"日志: {result['file']}")
        print(f"样本: {result['samples']}，时长: {result['duration_s']:.2f}s，"
              f"跟随/运动/锁存: {result['active_samples']}/{result['moving_samples']}/{result['static_samples']}")
        print(f"运动误差 median/p95: {result['moving_error_median_m']} / {result['moving_error_p95_m']} m")
        print(f"稳态静止(扣除0.30s停靠)关节速度 p95: {result['static_joint_speed_p95_rad_s']} rad/s，"
              f">0.05 比例: {result['static_nonzero_ratio']}")
        print(f"停手稳定时间 median: {result['stop_settle_median_s']} s（{result['stop_settle_events']} 次）")
        print(f"观测年龄/处理/控制计算 p95: {result['obs_age_p95_ms']} / "
              f"{result['processing_p95_ms']} / {result['control_work_p95_ms']} ms")
        print(f"控制超时比例: {result['overrun_ratio']:.4f}，慢速通道样本: {result['slow_follow_samples']}，"
              f"静止保持样本: {result['static_hold_samples']}")
        print(f"硬件时间使用比例: {result['hardware_time_ratio']}，错误: {result['errors']}")
        print(f"实际反馈: 有效日志行/独立样本 {result['feedback_valid_rows']}/"
              f"{result['feedback_samples']}，同步读取 p95: {result['feedback_read_p95_ms']} ms")
        print(f"采样点六轴最大跟踪误差 median/p95/max: "
              f"{result['feedback_tracking_median_rad']} / "
              f"{result['feedback_tracking_p95_rad']} / "
              f"{result['feedback_tracking_max_rad']} rad")
        if result["feedback_samples"]:
            print("注意：低频反馈用于观察跟踪误差趋势，不能据此精确测量执行延迟。")
        if result.get("mit_fbc_valid_ratio") is not None:
            print(f"MIT 200Hz 缓存: 有效率 {result['mit_fbc_valid_ratio']*100:.1f}%，"
                  f"age p95={result['mit_fbc_age_p95_ms']} ms，"
                  f"六轴最大跟踪误差 median/p95/max="
                  f"{result['mit_tracking_err_median_deg']}/"
                  f"{result['mit_tracking_err_p95_deg']}/"
                  f"{result['mit_tracking_err_max_deg']}°")
            print(f"MIT 峰值力矩(Nm): {result['mit_tau_abs_max_nm']}，"
                  f"最高 MOS 温度: {result['mit_tmos_max_c']}°C，"
                  f"200Hz 周期 p99 最大值: {result['mit_period_p99_ms_max']} ms，"
                  f"CAN 累计异常: {result['mit_can_fail_total']}")


if __name__ == "__main__":
    main()
