#!/usr/bin/env python3
"""
手眼标定故障诊断脚本 —— 不标定，只做体检
==========================================
原理（AX=XB 的相对运动相似性）：
    机械臂两位姿间的相对变换 A_rel 和相机观测的相对变换 B_rel
    满足 A_rel = X · B_rel · X⁻¹（相似变换），因此必有：
        rotation_angle(A_rel) == rotation_angle(B_rel)
        |t(A_rel)|            == |t(B_rel)|
    任何一边数据错了，这两个等式就会破。

能定位的病灶：
    ① 静态抖动大            → 检测/相机问题
    ② 平移比值恒定且 ≠ 1     → MARKER_SIZE 写错了（比值即修正系数）
    ③ 旋转角普遍对不上       → FK 关节符号/零位与 URDF 不一致，或 PnP/内参错误
    ④ 重投影误差大          → 相机内参与图像分辨率不匹配
    ⑤ PnP 距离与深度图不符   → 深度流或对齐问题

安全：仅围绕示教位姿做单关节 ±10° 小动作，总行程 < 2 分钟；
      退出路径与标定脚本相同（保持 → 缓慢回零 → 失能）。

用法：
    python calib_diagnose.py           # 需要先跑过一次标定示教（viewpoint.json 存在）
"""

import os
import sys
import json
import time

import numpy as np
import cv2

# 项目根目录（tools/ 的上一级）加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini2_camera import Gemini2Camera
from rebot_adapter import RebotAdapter
from hand_eye_calibration_auto import (
    ArucoTracker, get_tcp_matrix, VIEWPOINT_FILE, MOVE_SPEED_DEG,
    MARKER_SIZE,
)

DELTA_DEG = 10.0          # 单关节扰动幅度
TEST_JOINTS = [0, 1, 2, 4]  # J1 J2 J3 J5（覆盖大关节 + 腕关节）
AVG_FRAMES = 12           # 每个位姿平均多少帧
SETTLE_S = 0.6


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def sample_marker(tracker, cam, n=AVG_FRAMES):
    """采 n 帧有效标记位姿，返回平均 T（位置平均，旋转用首帧——小动作下差异可忽略）。"""
    ts, Ts = [], []
    t_end = time.time() + 4.0
    while len(ts) < n and time.time() < t_end:
        color, depth, scale = cam.get_frames()
        if color is None:
            continue
        try:
            T, annotated = tracker.detect(color)
        except Exception:
            continue
        cv2.imshow('Diagnose', annotated)
        cv2.waitKey(1)
        if T is None:
            continue
        ts.append(T[:3, 3].copy())
        Ts.append(T.copy())
    if not ts:
        return None, None, None, None
    t_mean = np.mean(ts, axis=0)
    T_out = Ts[0].copy()
    T_out[:3, 3] = t_mean
    # 标记中心像素的深度中位数（用于 ⑤ 对比）
    return T_out, np.std(ts, axis=0), color, (depth, scale)


def depth_at_marker(T_m2c, depth_raw, scale, K):
    """把标记原点投影回深度图，取 11x11 窗口中位数（米）。"""
    if depth_raw is None:
        return None
    p_cam = T_m2c[:3, 3]
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    ui, vi = int(round(u)), int(round(v))
    h, w = depth_raw.shape
    if not (5 <= ui < w - 5 and 5 <= vi < h - 5):
        return None
    win = depth_raw[vi-5:vi+6, ui-5:ui+6].astype(np.float32) * scale
    win = win[(win > 0.15) & (win < 2.0)]
    if len(win) < 10:
        return None
    return float(np.median(win))


def main():
    if not os.path.exists(VIEWPOINT_FILE):
        print(f"✗ 未找到 {VIEWPOINT_FILE}\n"
              "  先运行 hand_eye_calibration_auto.py 完成一次示教（空格保存）再来诊断")
        return
    with open(VIEWPOINT_FILE) as f:
        q_view = np.array(json.load(f)['q_view'], dtype=np.float64)

    cam = Gemini2Camera()
    tracker = ArucoTracker(cam.camera_matrix, cam.dist_coeffs)
    robot = RebotAdapter()
    time.sleep(1.0)

    try:
        print("=" * 60)
        print("手眼标定故障诊断（只体检，不标定）")
        print("=" * 60)

        # ── 检查 ④：内参 vs 图像分辨率 ─────────────────────
        color, depth_raw, scale = cam.get_frames()
        while color is None:
            color, depth_raw, scale = cam.get_frames()
        H, W = color.shape[:2]
        fx, cx, cy = cam.camera_matrix[0, 0], cam.camera_matrix[0, 2], cam.camera_matrix[1, 2]
        print(f"\n[检查④ 内参 vs 分辨率]")
        print(f"  彩色图: {W}x{H}   内参: fx={fx:.1f} cx={cx:.1f} cy={cy:.1f}")
        res_ok = abs(cx - W / 2) < 0.15 * W and abs(cy - H / 2) < 0.15 * H
        print(f"  主点应≈({W/2:.0f},{H/2:.0f}) → {'✓ 匹配' if res_ok else '✗✗ 严重不匹配！内参与图像分辨率不符，PnP 全错'}")
        if depth_raw is not None:
            print(f"  深度图: {depth_raw.shape[1]}x{depth_raw.shape[0]}")

        # ── 移动到示教位姿 ─────────────────────────────────
        print("\n[准备] 移动到示教位姿...")
        robot.move_to_joints(q_view, speed_deg_s=MOVE_SPEED_DEG, wait=True, timeout=60.0)
        time.sleep(SETTLE_S)

        # ── 检查 ①②⑤：静态观测 ────────────────────────────
        print("[检查① 静态稳定性] 采 12 帧...")
        T0, t_std, color0, (d0, s0) = sample_marker(tracker, cam)
        if T0 is None:
            print("✗ 示教位姿下看不到标记！请重新示教（标记要居中、25~40cm）")
            return
        print(f"  标记距离 |t| = {np.linalg.norm(T0[:3,3])*1000:.0f} mm")
        print(f"  位置抖动 std = {t_std.max()*1000:.2f} mm  "
              f"{'✓ 好' if t_std.max() < 0.002 else '⚠ 偏大（>2mm），检测不稳定' if t_std.max() < 0.005 else '✗✗ 很差，标定必挂'}")
        T_base0 = get_tcp_matrix(robot)

        z_pnp = float(np.linalg.norm(T0[:3, 3]))
        z_depth = depth_at_marker(T0, d0, s0, cam.camera_matrix)
        if z_depth is not None:
            print(f"[检查⑤ PnP vs 深度] PnP={z_pnp*1000:.0f}mm  深度图={z_depth*1000:.0f}mm  "
                  f"差={abs(z_pnp-z_depth)*1000:.0f}mm  "
                  f"{'✓' if abs(z_pnp-z_depth) < 0.02 else '⚠ 偏差大'}")

        # 重投影误差
        corners, ids, _ = tracker.detector.detectMarkers(color0)
        if ids is not None:
            img_pts = corners[0].reshape(4, 2).astype(np.float32)
            rvec0, _ = cv2.Rodrigues(T0[:3, :3])
            proj, _ = cv2.projectPoints(tracker.obj_pts, rvec0, T0[:3, 3],
                                        cam.camera_matrix, cam.dist_coeffs)
            err = np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean()
            print(f"  PnP 重投影误差 = {err:.2f} px  "
                  f"{'✓ 好' if err < 1.0 else '⚠ 偏大' if err < 2.5 else '✗✗ 内参/分辨率/畸变有问题'}")

        # ── 检查 ②③：相对运动一致性 ────────────────────────
        print(f"\n[检查②③ 相对运动一致性] 逐关节扰动 ±{DELTA_DEG:.0f}° ...")
        print(f"{'关节':<6}{'FK旋转角':>10}{'相机旋转角':>10}{'角差':>8}"
              f"{'FK平移mm':>10}{'相机平移mm':>10}{'平移比':>8}")
        ratios, ang_errs = [], []
        for j in TEST_JOINTS:
            q = q_view.copy()
            q[j] += np.deg2rad(DELTA_DEG)
            q = np.clip(q, robot.q_min, robot.q_max)
            robot.move_to_joints(q, speed_deg_s=MOVE_SPEED_DEG, wait=True, timeout=60.0)
            time.sleep(SETTLE_S)
            Tj, _, _, _ = sample_marker(tracker, cam)
            T_base_j = get_tcp_matrix(robot)
            robot.move_to_joints(q_view, speed_deg_s=MOVE_SPEED_DEG, wait=True, timeout=60.0)
            time.sleep(SETTLE_S)
            if Tj is None:
                print(f"J{j+1:<5} 看不到标记，跳过")
                continue

            A_rel = np.linalg.inv(T_base0) @ T_base_j          # 机械臂相对运动
            B_rel = Tj @ np.linalg.inv(T0)                      # 相机观测相对运动
            ang_a, ang_b = rot_angle_deg(A_rel[:3, :3]), rot_angle_deg(B_rel[:3, :3])
            ta, tb = np.linalg.norm(A_rel[:3, 3]), np.linalg.norm(B_rel[:3, 3])
            ratio = tb / ta if ta > 1e-6 else float('nan')
            ratios.append(ratio)
            ang_errs.append(abs(ang_a - ang_b))
            print(f"J{j+1:<6}{ang_a:>9.2f}°{ang_b:>9.2f}°{abs(ang_a-ang_b):>7.2f}°"
                  f"{ta*1000:>10.1f}{tb*1000:>10.1f}{ratio:>8.3f}")

        # ── 综合裁决 ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("[诊断结论]")
        if not res_ok:
            print("✗ 病灶 = 相机内参与彩色图分辨率不匹配（检查④未过）")
            print("  → PnP 输入全错，标定必然失败。先把内参问题修掉再谈其它。")
        if ratios:
            med = float(np.median(ratios))
            if not (0.92 < med < 1.08):
                corrected = MARKER_SIZE / med
                print(f"✗ 病灶 = MARKER_SIZE 错误。平移比恒为 {med:.3f} ≠ 1")
                print(f"  → 你假设标记 {MARKER_SIZE*1000:.0f}mm，实际应为约 {corrected*1000:.1f}mm")
                print(f"  → 用尺子量黑色方块的『外边』边长，改 MARKER_SIZE 后重跑标定")
            else:
                print(f"✓ 平移比中位数 {med:.3f} ≈ 1，MARKER_SIZE 正确")
        if ang_errs:
            med_ang = float(np.median(ang_errs))
            if med_ang > 2.0:
                print(f"✗ 病灶 = 旋转不一致（中位角差 {med_ang:.2f}°）。")
                print("  → 可能原因：SDK 关节角符号/零位与 URDF 不一致（FK 不可信），")
                print("    或相机内参/畸变错误。若检查④也未过，先修④再重测。")
            else:
                print(f"✓ 旋转一致（中位角差 {med_ang:.2f}°），FK 与 PnP 旋转链可信")
        if ratios and ang_errs and (0.92 < float(np.median(ratios)) < 1.08) \
                and float(np.median(ang_errs)) <= 2.0 and res_ok \
                and t_std.max() < 0.005:
            print("\n✓✓ 数据链全部健康 → 可以直接重跑标定，这次应该能过自检")
        print("=" * 60)

    finally:
        cv2.destroyAllWindows()
        cam.stop()
        try:
            robot.hold()
        except Exception:
            pass
        robot.stop()
        print("程序退出（已安全回零）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消（机械臂将由 rebot_control 平滑回零）")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
