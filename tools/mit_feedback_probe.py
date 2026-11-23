#!/usr/bin/env python3
"""
v13 P0 阶段只读硬件探针：验证 motorbridge 200Hz MIT 反馈缓存、精度与重力模型对齐。

默认**零运动**（保持当前姿态），绝不主动移动机械臂；可选 `--move-home` 以 15°/s
缓慢移动到 config/home_pose.json（用于测非零位姿下的真实重力下垂与 Pinocchio 符号对齐）。
退出时始终执行完整安全回零 → 保持 → 失能（robot.hold() + robot.stop()）。

实测内容：
1. Mode A（仅 get_state()，不调 poll_feedback_once）vs Mode B（每拍调 poll_feedback_once）：
   - 200Hz 环路周期 p50/p99/max、单拍 send_mit 与采样耗时 p99
   - 反馈有效率、数值变化率（判断 motorbridge 后台线程是否自动刷新，还是需 poll_feedback_once）
2. 精度核对：缓存位置 q_cache vs 同步 0x7019（read_joint_angles）六轴偏差（度）
3. 静态重力核对（含符号检验！为 P3 重力前馈打底）：
   - 对比 Pinocchio gravity_torque(q) 与电机上报力矩 tau_act 的符号与幅值比
   - 对比实测静态下垂 (q_cmd - q_act) 与理论下垂 g(q) / kp
4. 自动输出结论：是否可以开启 mit_ext.feedback_trusted / feedback_poll_once，并保存 JSON 报告。
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


def evaluate_probe_metrics(mode_a, mode_b, q_cache_rad, q_0x7019_rad,
                           q_cmd_rad, tau_act, tau_pin, kp):
    """纯数学评估函数（可离线单测）：核对位置精度、刷新率与重力矩符号/比例。

    参数:
        mode_a (dict): Mode A（关闭 ``poll_feedback_once``）采样统计字典，含
            ``valid_ratio``、``period_p99_ms``、``change_ratio`` 等字段。
        mode_b (dict): Mode B（每拍调用 ``poll_feedback_once``）采样统计字典。
        q_cache_rad (array-like): 200Hz 反馈缓存中位关节角（6 轴，单位: rad）。
        q_0x7019_rad (array-like): 同步 ``0x7019`` 寄存器读取的参考关节角（6 轴，单位: rad）。
        q_cmd_rad (array-like): 控制器当前下发的目标指令关节角（6 轴，单位: rad）。
        tau_act (array-like): 电机上报的静态实测力矩中位数（6 轴，单位: Nm）。
        tau_pin (array-like): Pinocchio 动力学模型预测的静态重力力矩（6 轴，单位: Nm）。
        kp (array-like): 6 轴 MIT 位置刚度系数（单位: Nm/rad）。

    返回:
        dict: 包含位置偏差（deg）、实测/理论下垂角（deg）、逐轴重力符号/比例核对列表、
            推荐配置标志（``recommend_feedback_poll_once`` / ``recommend_feedback_trusted``）
            以及可读结论字符串 ``verdict``。

    判定准则:
        - 位置精度门限：六轴缓存与 ``0x7019`` 最大绝对误差 ``<= 0.30°``；
        - 有效性门限：``valid_ratio >= 0.95``；
        - 实时性门限：200Hz 环路周期 ``p99 <= 8.0 ms``（标称 5.0 ms）；
        - 刷新活性门限：静态数值变化率 ``change_ratio >= 0.15``。
    """
    q_cache = np.asarray(q_cache_rad, dtype=np.float64)
    q_ref = np.asarray(q_0x7019_rad, dtype=np.float64)
    q_cmd = np.asarray(q_cmd_rad, dtype=np.float64)
    tau_a = np.asarray(tau_act, dtype=np.float64)
    tau_p = np.asarray(tau_pin, dtype=np.float64)
    kp_arr = np.asarray(kp, dtype=np.float64)

    # 缓存角度 vs 0x7019 寄存器同步读取基准角度的逐轴误差（转为度便于人工判读）
    pos_diff_deg = np.rad2deg(q_cache - q_ref)
    max_pos_diff_deg = (float(np.max(np.abs(pos_diff_deg)))
                        if np.all(np.isfinite(pos_diff_deg)) else math.inf)
    # 静态重力下垂对比：
    # - 实测下垂 = 指令角 - 实际角（MIT PD 稳态下 kp*(q_cmd - q_act) ≈ tau_gravity）
    # - 理论下垂 = Pinocchio 重力矩 / kp（对 kp 做 1e-6 下界保护防除零）
    sag_meas_deg = np.rad2deg(q_cmd - q_ref)
    sag_pred_deg = np.rad2deg(tau_p / np.maximum(kp_arr, 1e-6))

    # 重力力矩符号与比例（仅对 |tau_pin| >= 0.3 Nm 的承载轴评估，避免近零除噪）
    gravity_check = []
    for j in range(6):
        tp = float(tau_p[j])
        ta = float(tau_a[j])
        if not (math.isfinite(tp) and math.isfinite(ta)):
            gravity_check.append({
                "joint": j + 1, "tau_pin_nm": None, "tau_act_nm": None,
                "ratio_act_over_pin": None, "sign_match": None,
            })
            continue
        if abs(tp) >= 0.3:
            sign_match = bool((tp * ta) > 0.0)
            ratio = ta / tp
        else:
            sign_match = None
            ratio = None
        gravity_check.append({
            "joint": j + 1,
            "tau_pin_nm": round(tp, 4),
            "tau_act_nm": round(ta, 4),
            "ratio_act_over_pin": (None if ratio is None else round(ratio, 4)),
            "sign_match": sign_match,
        })

    # 提取 Mode A / Mode B 关键健康指标，准备四分支推荐决策
    a_valid = bool(mode_a.get("valid_ratio", 0.0) >= 0.95)
    b_valid = bool(mode_b.get("valid_ratio", 0.0) >= 0.95)
    a_p99 = mode_a.get("period_p99_ms")
    b_p99 = mode_b.get("period_p99_ms")
    a_change = float(mode_a.get("change_ratio") or 0.0)
    b_change = float(mode_b.get("change_ratio") or 0.0)
    pos_ok = bool(math.isfinite(max_pos_diff_deg) and max_pos_diff_deg <= 0.30)

    # 判断推荐模式：优先不调 poll_feedback_once（零额外调用），仅当 Mode A 不刷新而 Mode B 刷新且耗时达标时推荐开启。
    a_timing_ok = a_p99 is not None and a_p99 <= 8.0
    b_timing_ok = b_p99 is not None and b_p99 <= 8.0
    if pos_ok and a_valid and a_timing_ok and a_change >= 0.15:
        rec_poll = False
        rec_trusted = True
        verdict = "PASS_MODE_A: 后台线程自动高频刷新，可直接启用 feedback_trusted=true，保持 feedback_poll_once=false"
    elif pos_ok and b_valid and b_timing_ok and b_change >= 0.15:
        rec_poll = True
        rec_trusted = True
        verdict = "PASS_MODE_B: 需启用 feedback_poll_once=true 以实时刷新，且耗时与精度达标，可启用 feedback_trusted=true"
    elif pos_ok and (a_valid or b_valid):
        # 静态下编码器与力矩噪声极小时 change_ratio 可能偏低，不直接判 FAIL，提示结合动态录波复核
        rec_poll = bool(b_change > a_change * 1.5 and b_timing_ok)
        rec_trusted = False
        verdict = ("WARN_LOW_CHANGE_RATE: 静止下数值变化率偏低（可能静止噪声极小，属正常），"
                   "请结合单关节辨识动态录波确认 200Hz 刷新率后再开启 feedback_trusted=true")
    else:
        rec_poll = False
        rec_trusted = False
        verdict = "FAIL: 反馈缓存无效、精度偏差 >0.3° 或 200Hz 周期超标，保持 feedback_trusted=false"

    return {
        "pos_diff_deg": [round(float(x), 4) if math.isfinite(x) else None for x in pos_diff_deg],
        "max_pos_diff_deg": (round(max_pos_diff_deg, 4) if math.isfinite(max_pos_diff_deg) else None),
        "pos_match_ok": pos_ok,
        "sag_meas_deg": [round(float(x), 4) for x in sag_meas_deg],
        "sag_pred_deg": [round(float(x), 4) for x in sag_pred_deg],
        "gravity_check": gravity_check,
        "recommend_feedback_poll_once": rec_poll,
        "recommend_feedback_trusted": rec_trusted,
        "verdict": verdict,
    }


def _sample_mode(robot, poll_once, duration_s):
    """在指定反馈轮询模式下执行一段 200Hz 静置录波并汇总性能/精度指标。

    参数:
        robot (RebotAdapter): 已连接硬件且启用 MIT 扩展的机械臂适配器。
        poll_once (bool): ``False`` 对应 Mode A（仅 ``motor.get_state()``），
            ``True`` 对应 Mode B（每拍附加调用 ``controller.poll_feedback_once()``）。
        duration_s (float): 录波持续时长（单位: s）。

    返回:
        dict: 包含采样点数、有效帧率、总/逐关节数值变化率、200Hz 环路周期与发送/采样耗时
            p50/p99/max（ms）、CAN 累计异常数、以及 6 轴 ``q`` 中位数（rad）、``dq`` 标准差
            （rad/s）和 ``tau`` 中位数（Nm）。
    """
    arm = robot.io.arm
    arm.set_feedback_poll_once(poll_once)
    time.sleep(0.15)  # 等当前模式稳定
    arm.reset_feedback_change_stats()
    robot.start_feedback_recording()
    time.sleep(duration_s)
    rec = robot.stop_feedback_recording()
    stats = robot.get_mit_loop_stats() or {}

    q = rec["q"]
    dq = rec["dq"]
    tau = rec["tau"]
    n = len(rec["t"])
    if n > 0:
        # 仅统计 6 轴位置与力矩全为有限值的样本，剔除尚未刷新或异常 NaN 帧
        finite_mask = np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(tau), axis=1)
        valid_ratio = float(np.mean(finite_mask))
        q_med = np.nanmedian(q, axis=0) if np.any(finite_mask) else np.full(6, math.nan)
        dq_std = np.nanstd(dq, axis=0) if np.any(finite_mask) else np.full(6, math.nan)
        tau_med = np.nanmedian(tau, axis=0) if np.any(finite_mask) else np.full(6, math.nan)
        # 统计逐关节独立变化比例，排除单轴噪声干扰
        if n > 1:
            per_joint_change = np.mean(
                (np.diff(q, axis=0) != 0.0)
                | (np.diff(dq, axis=0) != 0.0)
                | (np.diff(tau, axis=0) != 0.0),
                axis=0,
            )
            per_joint_change_list = [round(float(x), 4) for x in per_joint_change]
        else:
            per_joint_change_list = [0.0] * 6
    else:
        valid_ratio = 0.0
        q_med = np.full(6, math.nan)
        dq_std = np.full(6, math.nan)
        tau_med = np.full(6, math.nan)
        per_joint_change_list = [0.0] * 6

    return {
        "poll_once": bool(poll_once),
        "samples": int(n),
        "valid_ratio": round(valid_ratio, 4),
        "change_ratio": (None if stats.get("feedback_change_ratio") is None
                         else round(float(stats["feedback_change_ratio"]), 4)),
        "per_joint_change_ratio": per_joint_change_list,
        "period_p50_ms": stats.get("period_p50_ms"),
        "period_p99_ms": stats.get("period_p99_ms"),
        "period_max_ms": stats.get("period_max_ms"),
        "send_p99_ms": stats.get("send_p99_ms"),
        "sample_p99_ms": stats.get("sample_p99_ms"),
        "can_fail_total": stats.get("can_fail_total"),
        "feedback_errors": stats.get("feedback_errors"),
        "q_med_rad": q_med.tolist(),
        "dq_std_rad_s": dq_std.tolist(),
        "tau_med_nm": tau_med.tolist(),
    }


def main():
    """命令行入口：依次采样 Mode A/B、核对 0x7019 精度与重力模型并生成 JSON 报告。

    安全红线:
        - 默认零运动保持当前位姿，仅显式指定 ``--move-home`` 时才以 15°/s 缓慢移至 HOME；
        - 无论中途正常完成或抛出异常，``finally`` 块均严格执行 ``hold() -> stop()``
          （内部安全回零并等待到位后再失能），杜绝悬空掉臂。

    返回:
        int: 进程退出码（0 表示探针执行完成，1 表示 MIT 扩展未激活）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.5,
                        help="每种模式采样时长 (s)，默认 2.5s")
    parser.add_argument("--move-home", action="store_true",
                        help="可选：以 15°/s 移至 config/home_pose.json 再探测（用于测真实重力下垂与力矩符号）")
    parser.add_argument("--out", type=str, default=None,
                        help="报告输出路径（默认 logs/mit_probe_<timestamp>.json）")
    args = parser.parse_args()

    # 延迟导入硬件适配器，确保离线单测导入 evaluate_probe_metrics 时不依赖 Pinocchio/CAN
    from rebot_adapter import RebotAdapter

    robot = None
    try:
        robot = RebotAdapter()
        ext_info = robot.mit_ext_info()
        print(f"[探针] MIT 扩展状态: active={ext_info.get('active')} "
              f"phase={ext_info.get('phase')} reason={ext_info.get('reason')}")
        if not ext_info.get("active"):
            print("[探针] ❌ MIT 扩展未启用，无法执行 200Hz 反馈缓存探测。请检查 config/mit_ext.yaml")
            return 1

        pose_name = "current_pose"
        if args.move_home:
            home_file = os.path.join(ROOT, "config", "home_pose.json")
            with open(home_file, encoding="utf-8") as stream:
                q_home = np.array(json.load(stream)["home_joint_pos"], dtype=np.float64)
            print(f"[探针] 正在以 15°/s 缓慢移至 HOME 位姿 {np.rad2deg(q_home).round(1).tolist()}° ...")
            robot.move_to_joints(q_home, speed_deg_s=15.0, wait=True, timeout=20.0)
            time.sleep(0.5)
            pose_name = "home_pose"
        else:
            print("[探针] 零运动模式：保持当前姿态进行只读探测（可用 --move-home 测悬臂重力下垂）")
            time.sleep(0.3)

        # 依次在 Mode A（无主动 poll）与 Mode B（每拍主动 poll）下采样，对比刷新率与环路抖动
        print(f"[探针] Mode A（feedback_poll_once=False）采样 {args.duration:.1f}s ...")
        mode_a = _sample_mode(robot, poll_once=False, duration_s=args.duration)
        print(f"[探针] Mode B（feedback_poll_once=True） 采样 {args.duration:.1f}s ...")
        mode_b = _sample_mode(robot, poll_once=True, duration_s=args.duration)
        # 恢复配置原值
        robot.io.arm.set_feedback_poll_once(bool(ext_info["cfg"]["feedback_poll_once"]))

        # 同步读一次 0x7019 作为精度基准（同时测量阻塞 CAN 读取耗时用于对比 200Hz 缓存收益）
        q_cmd = robot.get_command_pos()
        t_r0 = time.perf_counter()
        q_0x7019 = robot.get_current_pos()
        read_0x7019_ms = (time.perf_counter() - t_r0) * 1000.0
        fb_now = robot.get_feedback_state(max_age_s=0.20)
        # 优先选取有效率更高的模式中位数作为缓存读数代表值
        q_cache = (np.asarray(mode_a["q_med_rad"], dtype=np.float64)
                   if mode_a["valid_ratio"] >= mode_b["valid_ratio"]
                   else np.asarray(mode_b["q_med_rad"], dtype=np.float64))
        tau_act = (np.asarray(mode_a["tau_med_nm"], dtype=np.float64)
                   if mode_a["valid_ratio"] >= mode_b["valid_ratio"]
                   else np.asarray(mode_b["tau_med_nm"], dtype=np.float64))

        # 用 0x7019 实测角计算 Pinocchio 重力矩并结合 MIT 刚度 kp 做综合评估
        tau_pin = robot.gravity_torque(q_0x7019)
        kp, kd = robot.get_mit_gains()
        eval_res = evaluate_probe_metrics(
            mode_a, mode_b, q_cache, q_0x7019, q_cmd, tau_act, tau_pin, kp)

        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pose": pose_name,
            "ext_info": {k: v for k, v in ext_info.items() if k != "cfg"},
            "mit_kp": kp.tolist(),
            "mit_kd": kd.tolist(),
            "read_0x7019_ms": round(read_0x7019_ms, 2),
            "q_cmd_deg": np.rad2deg(q_cmd).round(3).tolist(),
            "q_0x7019_deg": np.rad2deg(q_0x7019).round(3).tolist(),
            "q_cache_deg": [round(float(np.rad2deg(x)), 3) if math.isfinite(x) else None
                            for x in q_cache],
            "t_mos_c": (None if fb_now is None else
                        [round(float(x), 1) if math.isfinite(x) else None
                         for x in fb_now["t_mos"]]),
            "mode_a_no_poll": mode_a,
            "mode_b_with_poll": mode_b,
            "evaluation": eval_res,
        }

        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        out_path = args.out or os.path.join(
            ROOT, "logs", "mit_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
        with open(out_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)

        def _fmt(v):
            """将可选浮点指标格式化为保留两位小数的字符串，None 显示为 N/A。"""
            return "N/A" if v is None else f"{v:.2f}"

        print("\n" + "=" * 70)
        print(f"MIT 200Hz 反馈探针报告（姿态: {pose_name}）")
        print("-" * 70)
        for label, m in (("Mode A (无 poll)", mode_a), ("Mode B (含 poll)", mode_b)):
            print(f"{label}: samples={m['samples']}, valid={m['valid_ratio']*100:.1f}%, "
                  f"change={m['change_ratio']}, period p50/p99/max="
                  f"{_fmt(m['period_p50_ms'])}/{_fmt(m['period_p99_ms'])}/{_fmt(m['period_max_ms'])} ms, "
                  f"send_p99={_fmt(m['send_p99_ms'])}ms, sample_p99={_fmt(m['sample_p99_ms'])}ms, "
                  f"can_fail={m['can_fail_total']}")
        print(f"同步 0x7019 读取耗时: {read_0x7019_ms:.2f} ms")
        print(f"缓存 vs 0x7019 偏差(度): {eval_res['pos_diff_deg']} (max={eval_res['max_pos_diff_deg']}°)")
        print(f"静态下垂 实测(度):       {eval_res['sag_meas_deg']}")
        print(f"静态下垂 URDF预测(度):   {eval_res['sag_pred_deg']}")
        print("重力力矩核对（正负号与比例，P3 关键依据）:")
        for item in eval_res["gravity_check"]:
            print(f"  J{item['joint']}: tau_pin={item['tau_pin_nm']} Nm, "
                  f"tau_act={item['tau_act_nm']} Nm, "
                  f"act/pin={item['ratio_act_over_pin']}, sign_match={item['sign_match']}")
        print(f"MOS 温度(°C): {report['t_mos_c']}")
        print("-" * 70)
        print(f"结论: {eval_res['verdict']}")
        print(f"报告已保存: {out_path}")
        print("=" * 70)
        return 0
    finally:
        # 硬件安全红线：始终先 hold 稳住当前姿态，再经由 robot.stop() 缓慢回零后失能
        if robot is not None:
            print("[探针] 执行安全停机（保持 → 缓慢回零 → 失能）...")
            try:
                robot.hold()
                time.sleep(0.2)
            finally:
                robot.stop()


if __name__ == "__main__":
    sys.exit(main())
