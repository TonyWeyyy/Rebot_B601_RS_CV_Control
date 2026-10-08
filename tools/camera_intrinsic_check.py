#!/usr/bin/env python3
"""
相机内参/畸变裁决测试 —— 纯相机，不动机械臂
==============================================
诊断已确认：PnP 重投影误差 89px（应 <2px），说明
「检测到的角点像素」与「内参 K + 畸变 dist」几何模型不自洽。

本脚本对同一帧图像用 3 种假设分别跑 solvePnP，裁决哪种是对的：
  假设 A：K + SDK 给的畸变系数（当前代码用法）
  假设 B：K + 零畸变（SDK 可能已输出过去畸变的图，或畸变字段语义不对）
  假设 C：K×2 + 零畸变（内参实际对应 1280 宽、图像却是 640 宽的情形）

判据（同时满足才算健康）：
  1. 重投影误差 < 2 px
  2. PnP 距离 ≈ 深度图中位数（差 < 15%）

用法：把 ArUco 标记（4X4_50, ID=0）放在相机正前方 20~40cm，
      python camera_intrinsic_check.py
      会自动采 30 帧取最稳定的一帧做裁决，并打印修复建议。
"""

import os
import sys
import numpy as np
import cv2

# 项目根目录（tools/ 的上一级）加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini2_camera import Gemini2Camera

ARUCO_DICT = cv2.aruco.DICT_4X4_50
EXPECTED_ID = 0
MARKER_SIZE = 0.10   # 米；尺寸错误不影响重投影误差，只影响 z 的比例，
                     # z 与深度的一致性会顺带暴露尺寸问题


def solve_and_score(obj_pts, img_pts, K, dist):
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean()
    return reproj, tvec.reshape(3)


def main():
    cam = Gemini2Camera()
    K = cam.camera_matrix.astype(np.float64)
    dist_sdk = cam.dist_coeffs.astype(np.float64)
    print(f"SDK 畸变系数 (k1,k2,p1,p2,k3) = {dist_sdk}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    s = MARKER_SIZE / 2.0
    obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                       dtype=np.float32)

    print("\n把标记放到相机正前方 20~40cm，采集中（30 帧）...")
    best = None   # (marker_px_width, img_pts, depth_raw, scale, color)
    n = 0
    while n < 30:
        color, depth, scale = cam.get_frames()
        if color is None:
            continue
        corners, ids, _ = detector.detectMarkers(color)
        cv2.imshow('intrinsic check', color)
        cv2.waitKey(1)
        if ids is None:
            continue
        ids_flat = np.asarray(ids).reshape(-1)
        m = np.where(ids_flat == EXPECTED_ID)[0]
        if len(m) == 0:
            continue
        img_pts = corners[m[0]].reshape(4, 2).astype(np.float32)
        width_px = float(np.linalg.norm(img_pts[0] - img_pts[1]))
        n += 1
        if best is None or width_px > best[0]:
            best = (width_px, img_pts, depth, scale, color)
    cv2.destroyAllWindows()

    if best is None:
        print("✗ 30 帧内没看到 ID=0 的标记，调整位置重试")
        cam.stop()
        return
    width_px, img_pts, depth_raw, scale, color = best
    print(f"标记图像宽度: {width_px:.0f} px")

    # 标记中心的深度真值（用角点质心像素，与 PnP 无关，最可靠）
    cu, cv_ = img_pts.mean(axis=0).astype(int)
    h, w = depth_raw.shape
    win = depth_raw[max(0,cv_-6):cv_+7, max(0,cu-6):cu+7].astype(np.float32) * scale
    win = win[(win > 0.15) & (win < 2.0)]
    z_depth = float(np.median(win)) if len(win) > 10 else None
    print(f"深度图测得标记距离: {z_depth*1000:.0f} mm" if z_depth
          else "深度图该位置无有效值（太近/反光）")

    K2 = K.copy(); K2[0, 0] *= 2; K2[1, 1] *= 2; K2[0, 2] *= 2; K2[1, 2] *= 2
    hypotheses = [
        ("A: K + SDK畸变（现状）", K, dist_sdk),
        ("B: K + 零畸变", K, None),
        ("C: K×2 + 零畸变（内参实为1280宽）", K2, None),
    ]

    print(f"\n{'假设':<28}{'重投影px':>10}{'PnP距离mm':>12}{'与深度差':>10}")
    results = []
    for name, Kh, dh in hypotheses:
        r = solve_and_score(obj_pts, img_pts, Kh, dh)
        if r is None:
            print(f"{name:<28}  solvePnP 失败")
            continue
        reproj, tvec = r
        z_pnp = float(np.linalg.norm(tvec))
        dz = abs(z_pnp - z_depth) / z_depth * 100 if z_depth else float('nan')
        ok = reproj < 2.0 and (z_depth is None or dz < 15)
        results.append((name, ok, reproj, z_pnp))
        print(f"{name:<28}{reproj:>10.2f}{z_pnp*1000:>12.0f}"
              f"{(f'{dz:>8.0f}%' if z_depth else '      n/a'):>10}  {'✓✓✓' if ok else ''}")

    print("\n" + "=" * 56)
    good = [r for r in results if r[1]]
    if good:
        name, _, reproj, z_pnp = good[0]
        print(f"✓ 裁决结果：{name} 是正确配置（reproj={reproj:.2f}px, z={z_pnp*1000:.0f}mm）")
        if name.startswith("B"):
            print("\n修复：gemini2_camera.py 中把 dist_coeffs 强制置零：")
            print("  self.dist_coeffs = np.zeros(5, dtype=np.float32)")
        elif name.startswith("C"):
            print("\n修复：内参需按 1280 宽读取/缩放，把 camera_intrinsic_check 结果贴给我")
    else:
        print("✗ 三种假设都不健康，把上面的表格完整贴给我，")
        print("  我再加假设（比如畸变模型是鱼眼/ Kannala-Brandt）")
    print("=" * 56)
    cam.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
