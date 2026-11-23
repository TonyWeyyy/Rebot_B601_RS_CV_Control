#!/usr/bin/env python3
"""
v14.0 支柱三：离线数字孪生闭环回放与 A/B 评测工具（tools/replay_digital_twin.py）
================================================================================
将真实硬件运行采集的高频 CSV 日志（如 `logs/servo_20261123_171409.csv`）还原为
**基座世界系下的真实人手 3D 运动轨迹**，然后喂给「v14 四层控制流水线 + 200Hz 辨识
B601-RS 数字孪生机械臂」，在无硬件条件下执行 100% 闭环仿真并输出与 v13.0 实机日志
同口径的量化对比报告！

工作原理：
  1. 轨迹重建：用真实 Pinocchio FK（基于日志里 200Hz 实测关节角 `qa_1..6`）与手眼外参
     `T_tcp_camera`，把每帧日志中的 `target_x/y/z` 经 25ms 相机曝光滞后对齐后反投影回
     基座世界系，恢复出真实人手的空间轨迹 `p_world_hand(t)` 与可见/冻结序列；
  2. 闭环重演：仿真机械臂从日志初始关节位姿出发，每拍由 `DigitalTwinBackend` 以 200Hz
     积分辨识机电动力学（含速度前馈、静重力补偿、伸展 `kd` 调度、5ms 总线延迟与 M6 监督），
     相机按 25ms 真实延迟和日志原始观测节拍重新观测 `p_world_hand(t)` 得到新的相机系观测；
  3. 同口径验收：统计与 `tools/analyze_servo_log.py` 一致的跟手误差、停手稳定耗时、
     静止速度 P95、6 轴跟踪误差、峰值力矩与 FSM 状态分布，并支持 `--out-csv` 导出。

用法示例：
    python3 tools/replay_digital_twin.py logs/servo_20261123_171409.csv
    python3 tools/replay_digital_twin.py logs/servo_20261123_171409.csv --v14-cfg config/v14.yaml --json
    python3 tools/replay_digital_twin.py logs/servo_20261123_171409.csv --sweep gain=2.55,2.75,3.00
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servo_diagnostics import DIAG_FIELDS
from servo_timing import CameraPoseHistory
from servo_v14.config import V14Config, load_v14_config, validate_v14_config
from servo_v14.pipeline import ServoV14Pipeline
from servo_v14.robot_backend import DigitalTwinBackend
from servo_v14.types import FSMMode, Observation
from tools.analyze_servo_log import summarize as summarize_csv_rows


@dataclass(frozen=True)
class ReconstructedSample:
    """从真实日志重建出的单拍世界系人手真值样本。"""

    t_s: float
    dt_s: float
    seq: int
    new_obs: bool
    following: bool
    visible: bool
    frozen: bool
    p_world_hand: Optional[np.ndarray]


def _num(row: dict, key: str) -> Optional[float]:
    val = row.get(key)
    if val in (None, ""):
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _flag(row: dict, key: str) -> bool:
    x = _num(row, key)
    return bool(x is not None and x > 0.5)


def load_hand_eye_matrix() -> np.ndarray:
    """读取 `config/hand_eye_calibration.json` 中的 4×4 `T_tcp_camera`。"""
    path = os.path.join(ROOT, "config", "hand_eye_calibration.json")
    with open(path, encoding="utf-8") as stream:
        return np.asarray(json.load(stream)["T_tcp_camera"], dtype=np.float64)


def reconstruct_hand_trajectory(
    csv_path: str,
    backend: DigitalTwinBackend,
    T_tcp_camera: np.ndarray,
    camera_delay_s: float = 0.025,
) -> Tuple[List[dict], np.ndarray, List[ReconstructedSample]]:
    """
    从实机 100Hz CSV 日志重建真实人手基座世界系轨迹。

    Returns:
        `(raw_rows, q_init, samples)`：原始日志行、首帧机械臂关节角、重建世界轨迹。
    """
    with open(csv_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"日志文件为空: {csv_path}")

    # 找首帧有效实测关节角作为仿真起点
    q_init = None
    for row in rows:
        qa = [_num(row, f"qa_{i + 1}") for i in range(6)]
        if all(v is not None for v in qa):
            q_init = np.asarray(qa, dtype=np.float64)
            break
    if q_init is None:
        q_init = np.array([-0.4313, 0.3445, 0.3691, 0.0938, 0.0018, -0.0015], dtype=np.float64)

    pose_hist = CameraPoseHistory(lpf_tau_s=0.0, max_history_s=0.60)
    t0 = _num(rows[0], "host_time") or 0.0
    samples: List[ReconstructedSample] = []
    last_q = q_init.copy()
    last_hand_w: Optional[np.ndarray] = None

    for row in rows:
        t = (_num(row, "host_time") or t0) - t0
        dt = _num(row, "dt_s") or 0.01
        qa = [_num(row, f"qa_{i + 1}") for i in range(6)]
        if all(v is not None for v in qa):
            last_q = np.asarray(qa, dtype=np.float64)
        tcp_p, tcp_r = backend.fk_only(last_q)
        pose_hist.push(t, tcp_p, tcp_r, T_tcp_camera, dt=dt)

        seq = int(_num(row, "seq") or 0)
        new_obs = _flag(row, "new_obs")
        following = _flag(row, "following")
        visible = _flag(row, "visible")
        frozen = _flag(row, "frozen")
        tx, ty, tz = _num(row, "target_x"), _num(row, "target_y"), _num(row, "target_z")

        p_w: Optional[np.ndarray] = None
        if visible and tx is not None and ty is not None and tz is not None:
            looked = pose_hist.lookup(t - camera_delay_s, use_lpf=False)
            if looked is not None:
                cam_p, cam_r = looked
            else:
                cam_r = tcp_r @ T_tcp_camera[:3, :3]
                cam_p = tcp_p + tcp_r @ T_tcp_camera[:3, 3]
            p_w = cam_p + cam_r @ np.array([tx, ty, tz], dtype=np.float64)
            last_hand_w = p_w.copy()
        elif visible and last_hand_w is not None:
            p_w = last_hand_w.copy()

        samples.append(ReconstructedSample(
            t_s=t,
            dt_s=dt,
            seq=seq,
            new_obs=new_obs,
            following=following,
            visible=visible,
            frozen=frozen,
            p_world_hand=p_w,
        ))

    return rows, q_init, samples


def run_closed_loop_replay(
    samples: List[ReconstructedSample],
    q_init: np.ndarray,
    cfg: V14Config,
    T_tcp_camera: np.ndarray,
    *,
    use_pinocchio: bool = True,
    rng_seed: Optional[int] = 42,
    out_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    在离线数字孪生后端上闭环重演重建的人手轨迹，返回完整量化评测报告。
    """
    offset_path = os.path.join(ROOT, "config", "joint_offset.json")
    backend = DigitalTwinBackend(
        sim_cfg=cfg.sim,
        q_init=q_init,
        use_pinocchio=use_pinocchio,
        joint_offset_path=offset_path,
        rng_seed=rng_seed,
    )
    pipeline = ServoV14Pipeline(cfg=cfg, backend=backend, T_tcp_camera=T_tcp_camera)
    pipeline.lock_current_orientation(q_init)

    # 仿真相机真实位姿历史（用于按 camera_delay_s=25ms 渲染滞后相机观测）
    sim_cam_hist = CameraPoseHistory(lpf_tau_s=0.0, max_history_s=0.60)
    hand_hist: Deque[Tuple[float, np.ndarray]] = deque(maxlen=128)

    replay_rows: List[dict] = []
    mode_counts: Counter = Counter()
    shield_counts: Counter = Counter()
    prev_following = False
    last_obs = Observation(
        seq=0, meas_time_s=0.0, host_time_s=0.0,
        visible=False, frozen=False, p_cam=None,
    )
    last_p_cam: Optional[np.ndarray] = None
    last_valid_z: float = cfg.pbvs.distance

    for s in samples:
        # 1. 记录仿真机械臂当前真实相机位姿（来自 200Hz 辨识动力学后的 q_act）
        robot_state = backend.get_state(s.t_s)
        tcp_p_act, tcp_r_act = backend.fk_only(robot_state.q_act)
        sim_cam_hist.push(s.t_s, tcp_p_act, tcp_r_act, T_tcp_camera, dt=s.dt_s)

        if s.p_world_hand is not None:
            hand_hist.append((s.t_s, s.p_world_hand))

        # 2. 跟随开关边沿检测（与真机按 g 行为一致）
        if s.following and not prev_following:
            pipeline.reset(keep_pose_history=True)
            pipeline.lock_current_orientation(robot_state.q_cmd)
        prev_following = s.following

        # 3. 按真实相机延迟 (25ms) 与原始日志 60fps 帧节拍生成仿真观测
        t_meas = max(0.0, s.t_s - cfg.sim.camera_delay_s)
        if s.visible and s.p_world_hand is not None:
            if s.new_obs or last_p_cam is None:
                # 查 t_meas 时刻仿真相机位姿与真实人手位置
                looked = sim_cam_hist.lookup(t_meas, use_lpf=False)
                if looked is not None:
                    cam_p_m, cam_r_m = looked
                else:
                    cam_r_m = tcp_r_act @ T_tcp_camera[:3, :3]
                    cam_p_m = tcp_p_act + tcp_r_act @ T_tcp_camera[:3, 3]

                # 查离 t_meas 最近的真实人手世界位置
                p_hand_m = s.p_world_hand
                for ht, hp in reversed(hand_hist):
                    if ht <= t_meas:
                        p_hand_m = hp
                        break
                p_cam_now = cam_r_m.T @ (p_hand_m - cam_p_m)
                if s.frozen:
                    # 模拟深度冻结：保留真实 2D 射线方向但沿用旧深度 Z
                    scale_z = last_valid_z / max(float(p_cam_now[2]), 0.10)
                    p_cam_now = np.array([p_cam_now[0] * scale_z, p_cam_now[1] * scale_z, last_valid_z])
                else:
                    last_valid_z = float(p_cam_now[2])
                last_p_cam = p_cam_now
                last_obs = Observation(
                    seq=s.seq,
                    meas_time_s=t_meas,
                    host_time_s=s.t_s,
                    visible=True,
                    frozen=s.frozen,
                    p_cam=last_p_cam,
                    processing_s=0.006,
                )
            else:
                # 非新帧拍：沿用上一帧观测但刷新 host_time
                last_obs = replace(last_obs, host_time_s=s.t_s)
        else:
            last_obs = Observation(
                seq=s.seq,
                meas_time_s=t_meas,
                host_time_s=s.t_s,
                visible=False,
                frozen=False,
                p_cam=None,
                processing_s=0.006,
            )

        # 4. 运行与真机 100% 相同的 v14 四层流水线 + 数字孪生动力学步进
        shielded, telem = pipeline.step(
            obs=last_obs,
            robot_state=robot_state,
            dt_s=s.dt_s,
            following_enabled=s.following,
            apply_to_backend=True,
        )
        mode_counts[telem.mode.value] += 1
        if shielded.shield_reason:
            for tag in shielded.shield_reason.split(","):
                shield_counts[tag] += 1

        row = pipeline.telemetry_to_diag_row(telem, robot_state, s.following)
        # 转为字符串风格字典以直接复用 analyze_servo_log.summarize()
        replay_rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=DIAG_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in replay_rows:
                writer.writerow(r)
        summary = summarize_csv_rows(out_csv)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", newline="", encoding="utf-8", delete=True) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=DIAG_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in replay_rows:
                writer.writerow(r)
            tmp.flush()
            summary = summarize_csv_rows(tmp.name)

    summary["fsm_mode_counts"] = dict(mode_counts)
    summary["shield_counts"] = dict(shield_counts)
    summary["using_pinocchio"] = backend.using_pinocchio
    summary["supervisor_latched"] = backend.supervisor_latched
    summary["supervisor_reason"] = backend.supervisor_reason
    return summary


def _fmt_cm(val: Optional[float]) -> str:
    return f"{val * 100:.2f}cm" if val is not None else "-"


def _fmt_f(val: Optional[float], unit: str = "", prec: int = 3) -> str:
    return f"{val:.{prec}f}{unit}" if val is not None else "-"


def _apply_override(cfg: V14Config, key: str, value: float) -> V14Config:
    """支持 `--sweep key=v1,v2,...` 对 PBVS/FSM 常用参数做离线扫描。"""
    if hasattr(cfg.pbvs, key):
        return validate_v14_config(replace(cfg, pbvs=replace(cfg.pbvs, **{key: value})))
    if hasattr(cfg.fsm, key):
        return validate_v14_config(replace(cfg, fsm=replace(cfg.fsm, **{key: value})))
    if hasattr(cfg.estimator, key):
        return validate_v14_config(replace(cfg, estimator=replace(cfg.estimator, **{key: value})))
    raise ValueError(f"未知的 sweep 参数名: {key}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v14.0 离线数字孪生闭环回放与 A/B 对比工具")
    parser.add_argument("csv_path", help="真实硬件伺服 CSV 日志路径（如 logs/servo_20261123_171409.csv）")
    parser.add_argument("--v14-cfg", default=os.path.join(ROOT, "config", "v14.yaml"),
                        help="v14 配置文件路径")
    parser.add_argument("--out-csv", default=None, help="可选：导出回放 CSV 供 analyze_servo_log.py 深入分析")
    parser.add_argument("--sweep", default=None, help="可选参数扫描，例如 gain=2.55,2.75,3.00")
    parser.add_argument("--no-pinocchio", action="store_true", help="强制使用解析运动学回退（测试用）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出对比结果")
    args = parser.parse_args(argv)

    cfg = load_v14_config(args.v14_cfg)
    T_tcp_cam = load_hand_eye_matrix()
    offset_path = os.path.join(ROOT, "config", "joint_offset.json")

    recon_backend = DigitalTwinBackend(
        sim_cfg=cfg.sim,
        use_pinocchio=(not args.no_pinocchio),
        joint_offset_path=offset_path,
        rng_seed=None,
    )
    raw_rows, q_init, samples = reconstruct_hand_trajectory(
        args.csv_path,
        recon_backend,
        T_tcp_cam,
        camera_delay_s=cfg.sim.camera_delay_s,
    )
    v13_real = summarize_csv_rows(args.csv_path)

    if args.sweep:
        key, vals_str = args.sweep.split("=", 1)
        vals = [float(x.strip()) for x in vals_str.split(",") if x.strip()]
        results = []
        for v in vals:
            c_i = _apply_override(cfg, key.strip(), v)
            res = run_closed_loop_replay(
                samples, q_init, c_i, T_tcp_cam,
                use_pinocchio=(not args.no_pinocchio),
            )
            results.append({"param": f"{key}={v}", "summary": res})
        if args.json:
            print(json.dumps({"v13_real": v13_real, "sweep": results}, indent=2, ensure_ascii=False))
        else:
            print(f"=== 数字孪生参数扫描 ({args.sweep}) | 样本数={len(samples)} ===")
            for item in results:
                s = item["summary"]
                print(
                    f"  {item['param']:<14} | mov_med={_fmt_cm(s['moving_error_median_m'])} "
                    f"p95={_fmt_cm(s['moving_error_p95_m'])} "
                    f"| settle={_fmt_f(s['stop_settle_median_s'], 's')} "
                    f"| static_p95={_fmt_f(s['static_joint_speed_p95_rad_s'], '', 4)} "
                    f"| trk_med={_fmt_f(s['mit_tracking_err_median_deg'], '°', 2)} "
                    f"| m6_latch={s['supervisor_latched']}"
                )
        return 0

    v14_sim = run_closed_loop_replay(
        samples, q_init, cfg, T_tcp_cam,
        use_pinocchio=(not args.no_pinocchio),
        out_csv=args.out_csv,
    )

    if args.json:
        print(json.dumps({"v13_real": v13_real, "v14_digital_twin": v14_sim}, indent=2, ensure_ascii=False))
        return 0

    print("=" * 72)
    print(f"v14.0 离线数字孪生闭环回放对比 | 输入日志: {os.path.basename(args.csv_path)}")
    print(f"运动学后端: {'Pinocchio 真实 URDF' if v14_sim['using_pinocchio'] else '纯NumPy精确URDF(0.02mm)'} | 回放步数: {len(samples)}")
    print("=" * 72)
    print(f"{'指标':<28} | {'v13.0 实机日志':<18} | {'v14.0 数字孪生闭环':<18}")
    print("-" * 72)
    rows_cmp = [
        ("运动 3D 误差 median", _fmt_cm(v13_real["moving_error_median_m"]), _fmt_cm(v14_sim["moving_error_median_m"])),
        ("运动 3D 误差 p95", _fmt_cm(v13_real["moving_error_p95_m"]), _fmt_cm(v14_sim["moving_error_p95_m"])),
        ("停手稳定耗时 median", _fmt_f(v13_real["stop_settle_median_s"], "s"), _fmt_f(v14_sim["stop_settle_median_s"], "s")),
        ("静止残余速度 p95 (rad/s)", _fmt_f(v13_real["static_joint_speed_p95_rad_s"], "", 4), _fmt_f(v14_sim["static_joint_speed_p95_rad_s"], "", 4)),
        ("6轴跟踪误差 median", _fmt_f(v13_real["mit_tracking_err_median_deg"], "°", 2), _fmt_f(v14_sim["mit_tracking_err_median_deg"], "°", 2)),
        ("6轴跟踪误差 max", _fmt_f(v13_real["mit_tracking_err_max_deg"], "°", 2), _fmt_f(v14_sim["mit_tracking_err_max_deg"], "°", 2)),
        ("控制耗时 p95", _fmt_f(v13_real["control_work_p95_ms"], "ms", 2), _fmt_f(v14_sim["control_work_p95_ms"], "ms", 2)),
        ("M6 安全监督锁存", "False", str(v14_sim["supervisor_latched"])),
    ]
    for label, a, b in rows_cmp:
        print(f"{label:<28} | {a:<18} | {b:<18}")
    print("-" * 72)
    print(f"v14 FSM 状态分布: {v14_sim['fsm_mode_counts']}")
    print(f"v14 安全护盾介入: {v14_sim['shield_counts']}")
    if args.out_csv:
        print(f"回放 CSV 已导出: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
