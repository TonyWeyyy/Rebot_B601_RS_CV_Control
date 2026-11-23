#!/usr/bin/env python3
"""
Orbbec Gemini 2 相机内参/畸变模型裁决工具 —— 纯相机测试，不驱动机械臂。
========================================================================
背景与用途：
    当手眼标定诊断发现 PnP 重投影误差异常（例如高达 ~89 px，正常应 <2 px）时，
    说明「图像角点像素」与「内参矩阵 K + 畸变系数 dist」所描述的成像模型不自洽。
    特别地，Orbbec SDK v2 上报的畸变参数（k1/k2 等）不适用于 OpenCV Brown 模型，
    项目最终结论是在 gemini2_camera.py 中强制 `dist_coeffs = np.zeros(5)`。

    本脚本在同一帧最清晰的 ArUco 标记图像上，用 3 种内参/畸变假设分别运行
    cv2.solvePnP（IPPE_SQUARE），并与硬件 D2C 对齐后的深度图距离交叉验证：
        假设 A：K + SDK 上报畸变系数（排查畸变模型语义冲突）
        假设 B：K + 零畸变（验证去畸变输出或禁用畸变系数）
        假设 C：K×2 + 零畸变（排查内参对应 1280 宽而彩色图实际为 640 宽的分辨率错配）

健康判据（两项同时满足才算通过）：
    1. 4 角点平均重投影误差 < 2.0 px
    2. PnP 估计距离 ‖t_cam_marker‖ 与深度图窗口中位数相对偏差 < 15%

用法：
    将 ArUco 标记（DICT_4X4_50, ID=0，边长 100mm）置于相机正前方 20~40cm：
        python tools/camera_intrinsic_check.py
    脚本自动采集 30 帧有效检测，选取标记像素宽度最大的一帧做裁决并打印修复建议。
"""

import os
import sys
import numpy as np
import cv2

# 项目根目录（tools/ 的上一级）加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini2_camera import Gemini2Camera

# ArUco 标记字典与尺寸约定（单位：m）
ARUCO_DICT = cv2.aruco.DICT_4X4_50
EXPECTED_ID = 0
MARKER_SIZE = 0.10   # 黑边外边长（米）；尺寸错误不影响重投影误差，只缩放 z，
                     # 因此 PnP z 与深度图一致性会顺带暴露尺寸写错问题


def solve_and_score(obj_pts, img_pts, K, dist):
    """在给定内参与畸变假设下求解正方形 ArUco 标记位姿并计算重投影误差。

    使用 cv2.SOLVEPNP_IPPE_SQUARE 专为共面正方形标记求解 T_camera_marker，
    随后将 4 个 3D 角点投影回像素平面并计算平均欧氏像素误差。

    参数:
        obj_pts (np.ndarray): (4, 3) 标记局部坐标系 3D 角点（单位：m，Z=0 平面）。
        img_pts (np.ndarray): (4, 2) 检测到的图像 2D 角点像素坐标（单位：px）。
        K (np.ndarray): 3×3 相机内参矩阵（float64，单位：px）。
        dist (np.ndarray | None): 畸变系数向量；传 None 表示零畸变假设。

    返回:
        tuple[float, np.ndarray] | None: 求解失败返回 None；成功返回
            (reproj_px, tvec)，其中 reproj_px 为平均重投影误差（单位：px），
            tvec 为形状 (3,) 的标记原点在相机系下的平移向量（单位：m）。
    """
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean()
    return reproj, tvec.reshape(3)


def main():
    """主流程：采集 30 帧 ArUco 图像，选出最优帧并在 3 种内参假设下完成重投影与深度一致性裁决。"""
    cam = Gemini2Camera()
    K = cam.camera_matrix.astype(np.float64)
    dist_sdk = cam.dist_coeffs.astype(np.float64)
    print(f"SDK 畸变系数 (k1,k2,p1,p2,k3) = {dist_sdk}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    # OpenCV ArUco 角点顺序为：左上 → 右上 → 右下 → 左下（标记局部系，单位：m）
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
        # 选取标记像素宽度最大的帧（视角最正/最近，角点定位信噪比最高）
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

    # 标记中心的深度真值（用角点质心像素在 13×13 邻域取有效深度中位数，与 PnP 独立）
    cu, cv_ = img_pts.mean(axis=0).astype(int)
    h, w = depth_raw.shape
    win = depth_raw[max(0,cv_-6):cv_+7, max(0,cu-6):cu+7].astype(np.float32) * scale
    win = win[(win > 0.15) & (win < 2.0)]  # 过滤无效零值及超出 Gemini 2 近距有效范围的点
    z_depth = float(np.median(win)) if len(win) > 10 else None
    print(f"深度图测得标记距离: {z_depth*1000:.0f} mm" if z_depth
          else "深度图该位置无有效值（太近/反光）")

    # 构造假设 C 所需的 2 倍缩放内参（测试 1280 vs 640 宽度错配）
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
        # 双判据联合判定：几何重投影精度 + 绝对尺度一致性
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
