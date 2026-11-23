#!/usr/bin/env python3
"""
v13 P3 阶段静态重力补偿渐进安全核验工具（零大幅运动）。

流程：
1. 以 15°/s 缓慢移至 HOME 悬臂位姿（可选 `--extended` 测前伸悬臂位姿）；
2. 保持目标位置不动，依次平滑切换 `gravity_ff_scale` = 0.00 → 0.35 → 0.60 → 0.80（每档稳定 2.0s）；
3. 每档实时核对：
   - 静态下垂 `q_cmd - q_act`（J3/J4 应随补偿比例单调减小，若反而增大 >0.4° 立即自动切回 0.0 并中止！）
   - 注入前馈力矩 `tau_ff` vs 电机上报总力矩 `tau_act` vs Pinocchio（裸 URDF 与含相机质点）预测值
   - MOS 温度监控（≥85°C 立即中止）与当前生效 `kd`
4. 保存报告至 `logs/p3_gravity_verify_<timestamp>.json`，并执行完整安全回零 → 失能。
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

MAX_VERIFY_MOS_C = 85.0
DEFAULT_SCALES = (0.0, 0.35, 0.60, 0.80)


def check_sag_monotonic_safety(base_sag_deg, cur_sag_deg, tol_deg=0.45):
    """核验主要承载轴（J3/J4）下垂是否改善而未反向放大。返回 (ok, reason)。"""
    for j in (2, 3):  # J3, J4
        b = abs(float(base_sag_deg[j]))
        c = abs(float(cur_sag_deg[j]))
        if b >= 0.8 and c > b + tol_deg:
            return False, (f"J{j + 1} 静态下垂反而增大 ({b:.2f}° → {c:.2f}°)，"
                           "疑似补偿符号/模型异常，立即中止！")
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", choices=("home", "extended"), default="home",
                        help="测试悬臂位姿：home（默认）或 extended（温和前伸 r≈0.19m）")
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES),
                        help="逐级测试的重力补偿比例，默认 0.0 0.35 0.60 0.80")
    parser.add_argument("--hold-s", type=float, default=2.0,
                        help="每档稳态观察时长 (s)，默认 2.0s")
    args = parser.parse_args()

    if any(s < 0.0 or s > 0.90 for s in args.scales):
        print("❌ --scales 必须在 [0.0, 0.90] 范围内")
        return 1

    from rebot_adapter import RebotAdapter

    robot = None
    try:
        robot = RebotAdapter()
        ext_info = robot.mit_ext_info()
        if not ext_info.get("active"):
            print("❌ MIT 扩展未启用，无法验证 P3 重力补偿。")
            return 1
        arm = robot.io.arm
        # 移位前先暂时归零重力补偿，保证与 P0 完全一致的基准起测
        arm.set_gravity_ff_runtime(enabled=False, scale=[0.0] * 6)

        home_file = os.path.join(ROOT, "config", "home_pose.json")
        with open(home_file, encoding="utf-8") as stream:
            q_home = np.array(json.load(stream)["home_joint_pos"], dtype=np.float64)
        q_target = q_home.copy()
        if args.pose == "extended":
            q_target[1] = min(float(robot.q_max[1]) - np.deg2rad(10.0),
                              float(q_home[1]) + np.deg2rad(20.0))
            q_target[2] = max(float(robot.q_min[2]) + np.deg2rad(10.0),
                              float(q_home[2]) + np.deg2rad(12.0))
            q_target[3] = float(np.clip(q_home[3] - np.deg2rad(15.0),
                                        robot.q_min[3] + np.deg2rad(10.0),
                                        robot.q_max[3] - np.deg2rad(10.0)))

        print(f"[P3核验] 以 15°/s 缓慢移至 {args.pose} 悬臂位姿 "
              f"{np.rad2deg(q_target).round(1).tolist()}° ...")
        robot.move_to_joints(q_target, speed_deg_s=15.0, wait=True, timeout=20.0)
        fk = robot.forward_kinematics()
        tcp_r = float(np.hypot(fk["position"][0], fk["position"][1]))
        robot.set_reach_radius(tcp_r)
        time.sleep(0.6)

        q_cmd = robot.get_command_pos()
        tau_urdf = robot.gravity_torque(q_target, include_payload=False)
        tau_full = robot.gravity_torque(q_target, include_payload=True)
        print(f"[P3核验] TCP={fk['position'].round(3).tolist()} m (r={tcp_r:.3f}m)")
        print(f"  Pinocchio 重力矩 裸URDF(Nm):   {np.round(tau_urdf, 3).tolist()}")
        print(f"  Pinocchio 重力矩 含相机(Nm):   {np.round(tau_full, 3).tolist()}")

        stages = []
        base_sag_deg = None
        aborted = False
        abort_reason = ""

        for s in args.scales:
            scale_vec = [0.0, s, s, s, 0.0, 0.0]
            arm.set_gravity_ff_runtime(enabled=(s > 0.0), scale=scale_vec)
            # 等待斜率限速与机械响应平稳
            t_wait_end = time.monotonic() + max(0.8, args.hold_s)
            samples_q = []
            samples_tau = []
            tmos_peak = 0.0
            while time.monotonic() < t_wait_end:
                time.sleep(0.05)
                fb = robot.get_feedback_state(max_age_s=0.15)
                if fb is None or not fb["valid"]:
                    continue
                tm = fb["t_mos"][np.isfinite(fb["t_mos"])]
                if tm.size:
                    tmos_peak = max(tmos_peak, float(np.max(tm)))
                    if tmos_peak >= MAX_VERIFY_MOS_C:
                        aborted = True
                        abort_reason = f"MOS 温度过高 ({tmos_peak:.1f}°C)"
                        break
                # 后半段采集稳态均值
                if t_wait_end - time.monotonic() <= args.hold_s * 0.5:
                    samples_q.append(fb["q"].copy())
                    samples_tau.append(fb["tau"].copy())
            if aborted:
                arm.set_gravity_ff_runtime(enabled=False, scale=[0.0] * 6)
                print(f"[P3核验] ❌ {abort_reason}，已切回 tau_ff=0")
                break

            q_act_med = np.median(np.array(samples_q), axis=0)
            tau_act_med = np.median(np.array(samples_tau), axis=0)
            sag_deg = np.rad2deg(q_cmd - q_act_med)
            stats = robot.get_mit_loop_stats() or {}
            tau_ff = stats.get("tau_ff_nm", [0.0] * 6)
            cur_kd = stats.get("cur_kd", [])

            if base_sag_deg is None:
                base_sag_deg = sag_deg.copy()
            else:
                ok, reason = check_sag_monotonic_safety(base_sag_deg, sag_deg)
                if not ok:
                    aborted = True
                    abort_reason = reason
                    arm.set_gravity_ff_runtime(enabled=False, scale=[0.0] * 6)
                    print(f"[P3核验] ❌ {reason}")
                    break

            stage = {
                "scale": float(s),
                "sag_deg": np.round(sag_deg, 3).tolist(),
                "tau_ff_nm": [round(float(x), 3) for x in tau_ff],
                "tau_act_nm": np.round(tau_act_med, 3).tolist(),
                "cur_kd": [round(float(x), 3) for x in cur_kd],
                "tmos_max_c": round(tmos_peak, 1),
                "period_p99_ms": stats.get("period_p99_ms"),
                "can_fail_total": stats.get("can_fail_total"),
            }
            stages.append(stage)
            print(f"  scale={s:.2f}: sag(J2/J3/J4)=[{sag_deg[1]:+.2f}°, {sag_deg[2]:+.2f}°, {sag_deg[3]:+.2f}°] | "
                  f"tau_ff=[{tau_ff[1]:+.2f}, {tau_ff[2]:+.2f}, {tau_ff[3]:+.2f}]Nm | "
                  f"tau_act=[{tau_act_med[1]:+.2f}, {tau_act_med[2]:+.2f}, {tau_act_med[3]:+.2f}]Nm | "
                  f"kd_J1={cur_kd[0] if cur_kd else 'N/A'} | T_mos={tmos_peak:.1f}°C")

        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        out_path = os.path.join(
            ROOT, "logs",
            f"p3_gravity_verify_{args.pose}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pose": args.pose,
            "tcp_m": fk["position"].round(4).tolist(),
            "tcp_radius_m": round(tcp_r, 4),
            "payload_mass_kg": ext_info["cfg"].get("gravity_payload_mass_kg"),
            "payload_com_link6_m": ext_info["cfg"].get("gravity_payload_com_link6_m"),
            "tau_pin_urdf_nm": np.round(tau_urdf, 4).tolist(),
            "tau_pin_payload_nm": np.round(tau_full, 4).tolist(),
            "aborted": aborted,
            "abort_reason": abort_reason,
            "stages": stages,
        }
        with open(out_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"P3 重力补偿静态核验{'中断' if aborted else '完成'}！报告: {out_path}")
        print("=" * 70)
        return 1 if aborted else 0
    finally:
        if robot is not None:
            print("[P3核验] 执行安全停机（保持 → 缓慢回零并自动淡出重力矩 → 失能）...")
            try:
                robot.hold()
                time.sleep(0.2)
            finally:
                robot.stop()


if __name__ == "__main__":
    sys.exit(main())
