#!/usr/bin/env python3
"""离线分析 servo_diagnostics CSV，不连接相机、CAN 或机械臂。"""

import argparse
import csv
import json
import math
from pathlib import Path


def number(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def flag(row, key):
    return str(row.get(key, "")).strip().lower() in ("1", "true", "yes")


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def summarize(path):
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
    settle_times = []
    for index in range(1, len(active)):
        if not flag(active[index - 1], "ff_moving") or flag(active[index], "ff_moving"):
            continue
        for count in range(20, len(active) - index + 1):
            window = active[index:index + count]
            if len(window) < 20:
                break
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
        "errors": sorted({row.get("error", "") for row in rows if row.get("error", "")}),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", action="store_true", help="输出 JSON 便于 A/B 脚本处理")
    args = parser.parse_args()
    result = summarize(args.log)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
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


if __name__ == "__main__":
    main()
