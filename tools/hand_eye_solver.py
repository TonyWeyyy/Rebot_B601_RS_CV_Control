#!/usr/bin/env python3
"""
手眼标定求解器（AX = XB，Park-Martin 1994 闭式解）

为什么需要这个文件：
    OpenCV 5.0 的 pip 包移除了 cv2.calibrateHandEye（枚举常量还在，
    函数本体没了）。本模块实现等价的求解算法，纯 numpy 无额外依赖。
    若当前 cv2 有 calibrateHandEye（4.x），自动优先调用原生实现。

数学约定（与 OpenCV calibrateHandEye 完全一致）：
    输入：
        R_g2b, t_g2b : 机械臂 FK 位姿 T_base_tcp（gripper → base）
        R_t2c, t_t2c : 标记在相机系位姿 T_camera_target（target → camera，
                       solvePnP 输出）
    输出：
        T_tcp_camera（即 OpenCV 返回的 cam2gripper）

    推导：T_base_tcp · X · T_cam_target = T_base_target = const（标记不动）
        → (T_g2b_i⁻¹ T_g2b_j) X = X (T_t2c_i T_t2c_j⁻¹)   即 A X = X B
"""

import numpy as np


def _vee(R):
    """旋转矩阵 → 旋转向量（log 映射）"""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-12:
        return np.zeros(3)
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return theta * k


def _relative_motions(R_g2b, t_g2b, R_t2c, t_t2c):
    """由位姿序列构造相对运动对 (A, B)，跳过旋转过小的对"""
    pairs = []
    n = len(R_g2b)
    for i in range(n - 1):
        for j in range(i + 1, n):
            Ra = R_g2b[i].T @ R_g2b[j]
            ta = R_g2b[i].T @ (t_g2b[j] - t_g2b[i])
            Rb = R_t2c[i] @ R_t2c[j].T
            tb = t_t2c[i] - Rb @ t_t2c[j]
            # 旋转太小的对对旋转解没有约束，跳过
            if np.linalg.norm(_vee(Ra)) < 0.01 or np.linalg.norm(_vee(Rb)) < 0.01:
                continue
            pairs.append((Ra, ta, Rb, tb))
    return pairs


def solve_hand_eye_park(R_g2b, t_g2b, R_t2c, t_t2c):
    """
    Park-Martin 闭式解。输入为列表（元素为 3x3 / 3 向量），返回 4x4 T_tcp_camera。
    """
    R_g2b = [np.asarray(R, dtype=np.float64) for R in R_g2b]
    t_g2b = [np.asarray(t, dtype=np.float64).reshape(3) for t in t_g2b]
    R_t2c = [np.asarray(R, dtype=np.float64) for R in R_t2c]
    t_t2c = [np.asarray(t, dtype=np.float64).reshape(3) for t in t_t2c]

    pairs = _relative_motions(R_g2b, t_g2b, R_t2c, t_t2c)
    if len(pairs) < 2:
        raise ValueError(f"有效相对运动对不足（{len(pairs)}），"
                         "样本需要更多旋转多样性")

    # ── 旋转部分：α = Rx·β，正交 Procrustes 闭式解 ──
    # max trace(Rx·M)，M = Σ β αᵀ；SVD: M = U S Vᵀ → Rx = V diag(1,1,det) Uᵀ
    M = np.zeros((3, 3))
    for Ra, _, Rb, _ in pairs:
        alpha = _vee(Ra)
        beta = _vee(Rb)
        M += np.outer(beta, alpha)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])   # 保证 det(Rx)=+1
    Rx = Vt.T @ D @ U.T

    # ── 平移部分：(Ra − I) tx = Rx·tb − ta，堆叠最小二乘 ──
    A_rows, b_rows = [], []
    for Ra, ta, Rb, tb in pairs:
        A_rows.append(Ra - np.eye(3))
        b_rows.append(Rx @ tb - ta)
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    tx, *_ = np.linalg.lstsq(A, b, rcond=None)

    T = np.eye(4)
    T[:3, :3] = Rx
    T[:3, 3] = tx
    return T


def calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c):
    """
    统一入口：优先用 OpenCV 原生实现（4.x），没有则用 Park-Martin。
    返回 (T_tcp_camera, method_name)
    """
    import cv2
    if hasattr(cv2, "calibrateHandEye"):
        R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                    method=cv2.CALIB_HAND_EYE_TSAI)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T, "opencv_tsai"
    return solve_hand_eye_park(R_g2b, t_g2b, R_t2c, t_t2c), "park_martin"
