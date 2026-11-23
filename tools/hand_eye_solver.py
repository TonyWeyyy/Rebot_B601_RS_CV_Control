#!/usr/bin/env python3
"""
Eye-in-Hand 手眼标定求解器（AX = XB，Park-Martin 1994 闭式解 + OpenCV 4.x/5.0 兼容封装）。

为什么需要这个文件：
    OpenCV 5.0 的 pip 包移除了 cv2.calibrateHandEye（枚举常量仍在，但函数本体缺失）。
    本模块提供纯 NumPy 实现的 Park-Martin (1994) 闭式求解算法，无额外依赖；
    若运行环境为 OpenCV 4.x 且存在 cv2.calibrateHandEye，则自动优先调用原生 Tsai-Lenz 实现。

坐标系与单位约定（与 OpenCV calibrateHandEye 完全一致）：
    输入：
        R_g2b, t_g2b : 机械臂正向运动学位姿 T_base_tcp（gripper/TCP → base），
                       旋转矩阵 3×3（无量纲），平移向量长度 3（单位：m）
        R_t2c, t_t2c : 标定板/标记在相机坐标系下的位姿 T_camera_target（target → camera，
                       即 cv2.solvePnP 输出），平移单位与 t_g2b 一致（单位：m）
    输出：
        T_tcp_camera : 4×4 齐次变换矩阵（camera → TCP，即 OpenCV 返回的 cam2gripper），
                       平移分量单位：m

数学推导（Eye-in-Hand 固定标定板）：
    由于标定板相对基座静止，任意两个采样时刻 i, j 满足：
        T_base_tcp_i · X · T_cam_target_i = T_base_tcp_j · X · T_cam_target_j = T_base_target
    移项得相对运动方程：
        (T_g2b_i⁻¹ · T_g2b_j) · X = X · (T_t2c_i · T_t2c_j⁻¹)   即 A · X = X · B
"""

import numpy as np


def _vee(R):
    """SO(3) 对数映射：将 3×3 旋转矩阵转换为 3D 旋转向量（李代数 so(3) 坐标）。

    利用旋转矩阵迹性质 trace(R) = 1 + 2 cos(θ) 求转角 θ ∈ [0, π]，
    并从反对称部分 (R - Rᵀ) / (2 sin θ) 提取单位旋转轴 k，返回 θ·k。

    参数:
        R (np.ndarray): 3×3 正交旋转矩阵（det(R) = +1）。

    返回:
        np.ndarray: 形状 (3,) 的旋转向量，模长为旋转角（单位：rad）。
            当转角 θ < 1e-12 rad 时直接返回零向量以避免除零奇异。
    """
    # clamp 到 [-1, 1] 防止浮点误差导致 arccos 产生 NaN
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-12:
        return np.zeros(3)
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return theta * k


def _relative_motions(R_g2b, t_g2b, R_t2c, t_t2c):
    """由多组绝对位姿序列两两配对，构造 AX = XB 所需的相对运动对 (A, B)。

    对每一对索引 (i, j)（i < j）计算：
        A = T_g2b_i⁻¹ · T_g2b_j  （机械臂末端在 i 时刻 TCP 系下的相对运动）
        B = T_t2c_i · T_t2c_j⁻¹  （相机观测到的等价相对运动）
    若任一侧的相对旋转角小于 0.01 rad（≈0.57°），该对几乎为纯平移，
    对旋转矩阵 Rx 无约束且会放大噪声，直接跳过。

    参数:
        R_g2b (list[np.ndarray]): T_base_tcp 的 3×3 旋转矩阵列表，长度 N。
        t_g2b (list[np.ndarray]): T_base_tcp 的 (3,) 平移向量列表（单位：m），长度 N。
        R_t2c (list[np.ndarray]): T_camera_target 的 3×3 旋转矩阵列表，长度 N。
        t_t2c (list[np.ndarray]): T_camera_target 的 (3,) 平移向量列表（单位：m），长度 N。

    返回:
        list[tuple]: 有效相对运动元组列表，每项为 (Ra, ta, Rb, tb)，
            其中 Ra/Rb 为 3×3 旋转矩阵，ta/tb 为 (3,) 平移向量（单位：m）。
    """
    pairs = []
    n = len(R_g2b)
    for i in range(n - 1):
        for j in range(i + 1, n):
            # A = T_g2b[i]⁻¹ · T_g2b[j]，利用刚体逆公式 R⁻¹ = Rᵀ, t_inv = -Rᵀ t
            Ra = R_g2b[i].T @ R_g2b[j]
            ta = R_g2b[i].T @ (t_g2b[j] - t_g2b[i])
            # B = T_t2c[i] · T_t2c[j]⁻¹
            Rb = R_t2c[i] @ R_t2c[j].T
            tb = t_t2c[i] - Rb @ t_t2c[j]
            # 旋转太小的对对旋转解没有约束，跳过
            if np.linalg.norm(_vee(Ra)) < 0.01 or np.linalg.norm(_vee(Rb)) < 0.01:
                continue
            pairs.append((Ra, ta, Rb, tb))
    return pairs


def solve_hand_eye_park(R_g2b, t_g2b, R_t2c, t_t2c):
    """基于 Park-Martin (1994) 算法的 Eye-in-Hand AX = XB 两步闭式求解器。

    求解流程（解耦旋转与平移）：
        1. 旋转求解：在李代数 so(3) 上满足 α = Rx · β（α = log(Ra), β = log(Rb)），
           构造相关矩阵 M = Σ β · αᵀ，经 SVD 正交 Procrustes 问题求得全局最优旋转矩阵 Rx
           （并显式约束 det(Rx) = +1 防止反射解）。
        2. 平移求解：将已求得的 Rx 代入平移方程 (Ra - I) · tx = Rx · tb - ta，
           堆叠所有运动对形成超定线性方程组，用最小二乘求出平移向量 tx。

    参数:
        R_g2b (Sequence[array-like]): T_base_tcp 的 3×3 旋转矩阵序列。
        t_g2b (Sequence[array-like]): T_base_tcp 的平移向量序列（单位：m）。
        R_t2c (Sequence[array-like]): T_camera_target 的 3×3 旋转矩阵序列。
        t_t2c (Sequence[array-like]): T_camera_target 的平移向量序列（单位：m）。

    返回:
        np.ndarray: 4×4 齐次变换矩阵 T_tcp_camera（camera → TCP），平移单位为 m。

    异常:
        ValueError: 当有效相对运动对数量少于 2（缺少足够旋转多样性）时抛出。
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
    # 目标：min Σ ‖α - Rx·β‖²  ⇔  max trace(Rx·M)，其中 M = Σ β·αᵀ
    # 对 M 做 SVD: M = U S Vᵀ  ⇒  最优正交矩阵 Rx = V · diag(1, 1, det(V Uᵀ)) · Uᵀ
    M = np.zeros((3, 3))
    for Ra, _, Rb, _ in pairs:
        alpha = _vee(Ra)
        beta = _vee(Rb)
        M += np.outer(beta, alpha)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])   # 保证 det(Rx)=+1，排除右手系翻转
    Rx = Vt.T @ D @ U.T

    # ── 平移部分：(Ra − I) tx = Rx·tb − ta，堆叠最小二乘 ──
    # 由 Ra·tx + ta = Rx·tb + tx 整理得到关于 tx 的线性方程组
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
    """手眼标定统一入口（自动适配 OpenCV 4.x 与 OpenCV 5.0）。

    若当前 OpenCV 提供原生 cv2.calibrateHandEye（4.x），优先调用 Tsai-Lenz 方法；
    否则自动回退至本模块实现的 Park-Martin 闭式解（OpenCV 5.0 兼容路径）。

    参数:
        R_g2b (Sequence[array-like]): T_base_tcp 的 3×3 旋转矩阵序列。
        t_g2b (Sequence[array-like]): T_base_tcp 的平移向量序列（单位：m）。
        R_t2c (Sequence[array-like]): T_camera_target 的 3×3 旋转矩阵序列。
        t_t2c (Sequence[array-like]): T_camera_target 的平移向量序列（单位：m）。

    返回:
        tuple[np.ndarray, str]: (T_tcp_camera, method_name)
            - T_tcp_camera: 4×4 齐次变换矩阵（camera → TCP，平移单位：m）
            - method_name: 实际使用的算法标识（"opencv_tsai" 或 "park_martin"）
    """
    import cv2
    if hasattr(cv2, "calibrateHandEye"):
        # OpenCV 4.x 原生 Tsai-Lenz 求解器，接口约定与本模块一致
        R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                    method=cv2.CALIB_HAND_EYE_TSAI)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T, "opencv_tsai"
    # OpenCV 5.0 移除了 calibrateHandEye，回退至纯 NumPy Park-Martin 实现
    return solve_hand_eye_park(R_g2b, t_g2b, R_t2c, t_t2c), "park_martin"
