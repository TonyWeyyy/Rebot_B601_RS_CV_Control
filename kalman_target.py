#!/usr/bin/env python3
"""
卡尔曼目标滤波器（6 维恒速 CV 模型）——手掌 3D 位置 + 速度联合估计
====================================================================
替代 ``palm_follow_grasp.py`` 早期的「帧间差分 + EMA」目标速度估计：

  差分方案的问题：
    tv = (target - prev_target) / CTRL_RATE
    - 分子是带噪声的位置差，分母是 0.01s，噪声被放大 100 倍，
      必须靠 EMA 压 → 引入群延迟（匀速跟随时前馈永远慢半拍）
    - 相机 60fps / 控制 100Hz 不同步：同一帧图像被差分 1~2 次，
      速度估计带量化台阶
  卡尔曼方案：
    - 状态向量 ``x = [px, py, pz, vx, vy, vz]^T``（单位：m 与 m/s），采用连续白噪声加速度
      驱动的恒速（Constant Velocity, CV）状态空间模型；
    - 异步 predict/update：新相机观测到达时调用 ``update()`` 或 ``update_partial()`` 做预测+修正，
      其余 100Hz 控制周期仅调用 ``predict(dt)`` 做纯外推，天然适配 60fps 视觉与 100Hz 伺服异步频率；
    - 观测年龄外推：在相机观测存在固定管线延迟 ``obs_age_s`` 时，上层可直接按当前状态
      ``p_extrap = position() + velocity() * lead_s`` 线性前推补偿延迟；
    - 分轴噪声建模：观测噪声支持水平/深度各向异性 ``R = diag(r_xy, r_xy, r_z)``，过程噪声
      支持标量或 XYZ 三维自适应倍率 ``q_scale``（按分轴创新在线放大，实现变加速快跟、静止抑抖）；
    - 深度冻结期子空间更新（``update_partial``）：当 2D 关键点新鲜但深度临时无效（``frozen=1``）时，
      仅将观测投影到与相机光轴正交的 XY 二维子空间更新，绝不把冻结 Z 值喂入滤波器污染深度状态。

调参直觉：
  q 大 → 更信观测，跟踪快、噪声多（≈ EMA alpha 调大）
  q 小 → 更信模型，平滑好、响应慢
  r 大 → 同上反方向。先动 q，r 用深度噪声实测值（σ≈2cm → 4e-4 m^2）。
"""

import numpy as np


class KalmanTarget3D:
    """6 维恒速模型卡尔曼滤波器：掌心 3D 位置与速度联合估计（异步 predict/update）。

    支持在相机坐标系或世界坐标系下独立实例化（例如视觉伺服中并行运行的相机系 KF 与
    世界系时滞对齐 KF），所有长度与速度均严格使用国际单位（m、m/s、s）。

    核心属性：
        x (np.ndarray): 6 维状态向量 ``[px, py, pz, vx, vy, vz]``（m, m/s）。
        P (np.ndarray): 6×6 状态误差协方差矩阵。
        H (np.ndarray): 3×6 完整位置观测矩阵 ``[I_3, 0_3]``。
        q (float): 基准过程噪声谱密度（``m^2 / s^3``，对应白噪声加速度功率谱密度）。
        r_vector (np.ndarray): 3 维各向异性观测位置方差 ``[r_xy, r_xy, r_z]``（``m^2``）。
        q_scale (float | np.ndarray): 外部自适应调度写入的过程噪声倍率（标量或 3 维分轴向量）。
        initialized (bool): 是否已完成首帧完整 3D 位置初始化。
    """

    def __init__(self, q=20.0, r=4e-4, r_xy=None, r_z=None):
        """初始化 6 维恒速卡尔曼滤波器。

        Args:
            q (float): 基准过程噪声谱密度（``(m/s^2)^2`` 量纲对应的白噪声加速度强度），默认 20.0。
            r (float): 默认标量位置观测方差（``m^2``），默认 ``4e-4``（对应标准差 2cm），
                在未指定 ``r_xy``/``r_z`` 时作为回退值，保持旧接口兼容。
            r_xy (float | None): X/Y 平面位置观测方差（``m^2``）；``None`` 时回退为 ``r``。
            r_z (float | None): Z 轴深度方向观测方差（``m^2``）；``None`` 时回退为 ``r``。

        Raises:
            ValueError: 任一观测噪声方差非有限或 ``<= 0`` 时抛出。
        """
        self.q = float(q)   # 过程噪声：手的加速度能力 (m/s^2)^2（白噪声加速度谱密度）
        self.r = float(r)   # 标量回退参数，保留旧调用兼容性
        xy = self.r if r_xy is None else float(r_xy)
        zz = self.r if r_z is None else float(r_z)
        self.r_vector = np.array([xy, xy, zz], dtype=np.float64)
        if not np.all(np.isfinite(self.r_vector)) or np.any(self.r_vector <= 0.0):
            raise ValueError("KF 观测噪声方差必须为有限正数")
        self.x = np.zeros(6)              # 状态向量：[px, py, pz, vx, vy, vz] (m, m/s)
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0            # 速度初值不确定性给大些，令前几帧观测快速拉齐速度估计
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)        # 完整观测仅直接量测 3D 位置，速度由位置-速度互协方差隐式更新
        self.q_scale = 1.0                # 自适应过程噪声倍率（标量或 3 维向量，外部按创新在线调整）
        self.initialized = False

    def reset(self):
        """清空滤波器全部状态与协方差（跟随停止 / 长丢手超阈值 / 自动回 HOME 后调用）。"""
        self.x[:] = 0.0
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self.q_scale = 1.0
        self.initialized = False

    def _FQ(self, dt):
        """构造步长 ``dt``（秒）对应的恒速状态转移矩阵 ``F`` 与过程噪声协方差 ``Q``。

        连续白噪声加速度模型按 XYZ 三轴解耦积分，单轴 2×2 过程噪声块为：
            ``Q_i = q_i * [[dt^3 / 3, dt^2 / 2], [dt^2 / 2, dt]]``
        其中 ``q_i = self.q * q_scale_i``，使水平轴与噪声更大的深度轴可独立调大/缩窄带宽。

        Args:
            dt (float): 预测时间步长（秒）。

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(F, Q)``，均为 6×6 float64 矩阵。

        Raises:
            ValueError: ``self.q_scale`` 非有限标量或非有限 3 维向量时抛出。
        """
        F = np.eye(6)
        # 位置按当前速度恒速外推：p_{k} = p_{k-1} + v_{k-1} * dt
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = np.zeros((6, 6))
        scale = np.asarray(self.q_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(3, float(scale))
        if scale.shape != (3,) or not np.all(np.isfinite(scale)):
            raise ValueError("KF 自适应 Q 倍率必须为有限标量或三维向量")
        q = self.q * scale              # 标量旧逻辑与 XYZ 分轴新逻辑共用
        for i in range(3):
            # 位置方差、位置-速度互协方差、速度方差（解析积分连续白噪声加速度）
            Q[i, i] = dt3 / 3.0 * q[i]
            Q[i, i + 3] = Q[i + 3, i] = dt2 / 2.0 * q[i]
            Q[i + 3, i + 3] = dt * q[i]
        return F, Q

    def predict(self, dt):
        """执行一步先验状态外推与协方差传播（无新观测的控制周期直接调用）。

        在 60fps 相机 / 100Hz 伺服异步周期中，没有新观测到达时只调用本方法以恒速外推
        位置并随 ``dt`` 增大不确定性 ``P``；未初始化时为空操作。

        Args:
            dt (float): 距上一次预测/更新的时间间隔（秒）。
        """
        if not self.initialized:
            return
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas, dt):
        """新相机完整 3D 观测到达时执行「先验预测 + 卡尔曼测量修正」。

        首帧到达时仅用 ``meas`` 初始化 3D 位置（速度保持 0），不计算增益；
        后续帧先以 ``dt`` 做先验预测，再用各向异性观测协方差 ``diag(r_vector)`` 更新。

        Args:
            meas (array-like): 3 维目标位置观测 ``[px, py, pz]``（米，相机系或世界系）。
            dt (float): 距上一次预测/更新的时间间隔（秒）。
        """
        meas = np.asarray(meas, dtype=np.float64)
        if not self.initialized:
            self.x[:3] = meas
            self.initialized = True
            return
        self.predict(dt)
        # 创新协方差 S = H P H^T + R，卡尔曼增益 K = P H^T S^{-1}
        S = self.H @ self.P @ self.H.T + np.diag(self.r_vector)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (meas - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def update_partial(self, meas_sub, H_pos, dt, r_diag=None):
        """分轴/正交子空间观测更新（深度冻结期间仅用新鲜 2D 的 XY 信息更新，绝不污染 Z）。

        数学原理：
            当本帧 MediaPipe 2D 关键点新鲜但深度临时无效（``frozen=1``）时，相机系仅 X/Y
            方向有效（或在世界系下仅与相机光轴 ``z_cam`` 正交的二维平面 ``[e_x, e_y]^T`` 有效）。
            调用方构造行正交（或分轴选择）位置观测矩阵 ``H_pos ∈ R^{m×3}`` 与投影观测
            ``meas_sub = H_pos @ p_meas ∈ R^m``，本方法扩展为 6 维子空间观测矩阵
            ``H_sub = [H_pos, 0_{m×3}]`` 并执行标准卡尔曼修正：
                ``S = H_sub P H_sub^T + diag(r_diag)``
                ``K = P H_sub^T S^{-1}``
            从而只对可观子空间及其速度分量注入新息，完全屏蔽冻结 Z 轴，避免深度跳变或
            陈旧 Z 值污染位置与 Z 向速度估计。

        Args:
            meas_sub (array-like): 长度为 ``m``（``m >= 1``）的子空间位置观测向量（米）。
            H_pos (array-like): 形状为 ``(m, 3)`` 的位置子空间观测矩阵（无量纲）。
            dt (float): 距上一次预测/更新的时间间隔（秒）。
            r_diag (array-like | None): 长度为 ``m`` 的子空间观测方差对角元素（``m^2``）；
                为 ``None`` 时默认各分量取 ``self.r_vector[0]``（XY 平面方差）。

        Returns:
            bool: 更新成功返回 ``True``；若滤波器尚未经完整 3D 初始化、或输入维度/数值非法，
            则直接拒绝并返回 ``False``（不修改滤波器状态）。
        """
        if not self.initialized:
            return False
        z = np.asarray(meas_sub, dtype=np.float64)
        Hp = np.asarray(H_pos, dtype=np.float64)
        if (z.ndim != 1 or Hp.ndim != 2 or Hp.shape != (z.size, 3)
                or z.size == 0 or not np.all(np.isfinite(z))
                or not np.all(np.isfinite(Hp))):
            return False
        r_vec = (np.full(z.size, self.r_vector[0], dtype=np.float64)
                 if r_diag is None else np.asarray(r_diag, dtype=np.float64))
        if (r_vec.shape != (z.size,) or not np.all(np.isfinite(r_vec))
                or np.any(r_vec <= 0.0)):
            return False
        self.predict(dt)
        # 仅在位置子块填充 H_pos，速度子块保持全零，速度通过 P 中的位置-速度互协方差间接修正
        H_sub = np.zeros((z.size, 6), dtype=np.float64)
        H_sub[:, :3] = Hp
        S = H_sub @ self.P @ H_sub.T + np.diag(r_vec)
        K = self.P @ H_sub.T @ np.linalg.inv(S)
        self.x += K @ (z - H_sub @ self.x)
        self.P = (np.eye(6) - K @ H_sub) @ self.P
        return True

    def position(self):
        """返回当前估计的 3D 位置副本 ``[px, py, pz]``（单位：米）。"""
        return self.x[:3].copy()

    def velocity(self):
        """返回当前估计的 3D 速度副本 ``[vx, vy, vz]``（单位：米/秒）。"""
        return self.x[3:].copy()
