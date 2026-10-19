#!/usr/bin/env python3
"""
卡尔曼目标滤波器（恒速 CV 模型）——手掌 3D 位置 + 速度联合估计
=================================================================
替代 palm_follow_grasp.py 原来的「帧间差分 + EMA」目标速度估计：

  差分方案的问题：
    tv = (target - prev_target) / CTRL_RATE
    - 分子是带噪声的位置差，分母是 0.01s，噪声被放大 100 倍，
      必须靠 EMA 压 → 引入群延迟（匀速跟随时前馈永远慢半拍）
    - 相机 60fps / 控制 100Hz 不同步：同一帧图像被差分 1~2 次，
      速度估计带量化台阶
  卡尔曼方案：
    状态 x = [px, py, pz, vx, vy, vz]，恒速模型预测 + 观测修正。
    新相机帧到达才 update（观测修正），其余控制周期只 predict
    （纯外推）——天然适配 60fps 检测 / 100Hz 控制的异步结构。
    速度是滤波器内部状态，不经差分，无放大噪声，几乎无群延迟。

第一版只取「速度估计」喂给现有前馈（lead=0，不做延迟外推），
位置通道暂不使用（死区/增益仍用原 target_cam，行为不变）。
后续第二版在此之上加 lead 外推即可，不用改结构。

调参直觉：
  q 大 → 更信观测，跟踪快、噪声多（≈ EMA alpha 调大）
  q 小 → 更信模型，平滑好、响应慢
  r 大 → 同上反方向。先动 q，r 用深度噪声实测值（σ≈2cm → 4e-4）。
"""

import numpy as np


class KalmanTarget3D:
    """恒速模型卡尔曼滤波：掌心 3D 位置/速度估计（异步 predict/update）。"""

    def __init__(self, q=20.0, r=4e-4):
        self.q = float(q)   # 过程噪声：手的加速度能力 (m/s^2)^2（白噪声加速度谱密度）
        self.r = float(r)   # 观测噪声方差 (m^2)：单帧掌心 3D 位置噪声，σ≈2cm → 4e-4
        self.x = np.zeros(6)              # [px,py,pz, vx,vy,vz]
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0            # 速度初值不确定性给大些，收敛更快
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)        # 只观测位置
        self.q_scale = 1.0                # 自适应过程噪声倍率（外部按创新调整）
        self.initialized = False

    def reset(self):
        """跟随停止/丢手超阈值/自动回 HOME 后调用：丢弃全部状态。"""
        self.x[:] = 0.0
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self.q_scale = 1.0
        self.initialized = False

    def _FQ(self, dt):
        """恒速模型状态转移 F 与白噪声加速度过程噪声 Q。"""
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = np.zeros((6, 6))
        q = self.q * self.q_scale     # 自适应：加速段放大过程噪声，快速跟手
        for i in range(3):
            Q[i, i] = dt3 / 3.0 * q
            Q[i, i + 3] = Q[i + 3, i] = dt2 / 2.0 * q
            Q[i + 3, i + 3] = dt * q
        return F, Q

    def predict(self, dt):
        """每个控制周期都调用（无论有没有新观测）。"""
        if not self.initialized:
            return
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas, dt):
        """新相机帧到达时调用。meas = 掌心 3D（相机系，米）。"""
        meas = np.asarray(meas, dtype=np.float64)
        if not self.initialized:
            self.x[:3] = meas
            self.initialized = True
            return
        self.predict(dt)
        S = self.H @ self.P @ self.H.T + np.eye(3) * self.r
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (meas - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def position(self):
        return self.x[:3].copy()

    def velocity(self):
        return self.x[3:].copy()
