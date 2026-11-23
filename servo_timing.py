#!/usr/bin/env python3
"""视觉伺服时序、自运动补偿、门控状态机与运动学安全核心模块（纯计算，无硬件依赖）。

本模块为 Eye-in-Hand 6-DOF 视觉伺服控制链路提供全部无状态/轻量有状态的纯数值算法，
与具体相机 SDK、CAN 驱动或 Pinocchio 绑定完全解耦，便于 100Hz 实时伺服线程调用及离线单元测试。

主要功能分组：
1. 时间尺度与时间戳校准：
   - ``scaled_alpha`` / ``scaled_decay``：将标称周期系数换算至真实 ``dt``，消除周期抖动对群延迟的影响；
   - ``measurement_age`` / ``dynamic_lead_seconds``：观测年龄钳制与动态前瞻时间计算；
   - ``CameraTimestampCalibrator``：相机硬件时间戳到主机单调时钟的在线漂移校准。
2. 眼在手上（Eye-in-Hand）坐标变换与自运动补偿：
   - ``camera_point_to_world`` / ``world_point_to_camera``：相机系与基座世界系点变换；
   - ``camera_twist_from_tcp`` / ``compensate_palm_velocity`` / ``compensate_palm_velocity_delayed``：
     含安装偏置杠杆臂的刚体速度变换与自运动速度抵消；
   - ``DelayedTwistCompensator`` / ``ActualVelocityFilter``：延迟+低通相位匹配补偿器与 200Hz 实测角速度滤波；
   - ``CameraPoseHistory``（含 SO(3) 测地插值 ``_so3_log`` / ``_so3_exp`` / ``_so3_interp``）：
     世界系 KF 与 Smith 预估器专用的带低通相机位姿历史缓冲；
   - ``consensus_velocity``：世界系 KF 与 Twist 补偿器双估计器逐轴保守共识速度。
3. 启停门控、滤波调度与抗振荡状态机：
   - ``AdaptiveLeadGate``：速度/距离/方向三因子乘性自适应前瞻门控；
   - ``FastStartGate`` / ``SlowFollowGate``：静止锁存快速起步与慢速连续拖动门控；
   - ``StaticVelocityHold``：静止死区内低速残余指令清零保持器；
   - ``AxisProcessNoiseGate``：分轴同向创新确认的自适应过程噪声 Q 倍率器；
   - ``adaptive_feedforward_scale`` / ``apply_axis_reversal_damping``：速度感知停靠前馈衰减与分轴换向阻尼；
   - ``PostBrakeSettleGuard`` / ``compute_reach_schedule`` / ``OscillationDamper``：
     刹车物理回弹冷静窗、水平伸展半径调度与分轴闭环自激极限环阻尼器。
4. 运动学优先级与安全约束：
   - ``compute_orientation_cruise_weight`` / ``compute_weighted_damped_pseudoinverse``：
     巡航段姿态软让权与行加权阻尼最小二乘伪逆；
   - ``horizontal_radial_soft_wall`` / ``apply_joint_limit_cushion`` / ``scale_joint_velocities``：
     水平圆柱软虚拟墙、关节软限位缓冲与保持笛卡尔方向的等比例限速。
5. 视觉 ROI 辅助：
   - ``compute_roi_box`` / ``remap_roi_landmarks``：迟滞死区方形 ROI 裁剪与归一化关键点全图重映射。

单位与安全约定：
- 除非参数名显式带 ``_ms`` / ``_deg`` / ``_uv`` 后缀，所有长度均采用米（m）、时间采用秒（s）、
  角度采用弧度（rad）、线速度采用 m/s、角速度采用 rad/s；
- 所有对外接口均对 ``NaN``/``Inf``、形状不匹配或负 ``dt`` 执行 fail-safe 防护（返回零速、零权重或回退保守状态），
  绝不向 100Hz 控制回路传播非有限数值。
"""

import math
from types import SimpleNamespace

import numpy as np


def scaled_alpha(alpha, dt, nominal_dt=0.01):
    """将标称控制周期的 EMA 平滑系数换算到实际周期 ``dt``，保持一阶低通群延迟不变。

    推导原理：
        标称周期 ``nominal_dt`` 下每步保留权重为 ``(1 - alpha)``，对应连续时间衰减因子
        ``exp(-dt / tau)``。经过实际时长 ``dt`` 后等效保留权重为 ``(1 - alpha) ** (dt / nominal_dt)``，
        因此实际步长下的平滑系数为：
        ``alpha_eff = 1 - (1 - alpha) ** (dt / nominal_dt)``。

    Args:
        alpha (float): 标称周期 ``nominal_dt`` 下的 EMA 系数，范围 ``(0, 1]``（无量纲）。
        dt (float): 当前控制周期的实际步长（秒，s）。
        nominal_dt (float, optional): 标称设计周期（秒，s），默认 ``0.01``（对应 100Hz）。

    Returns:
        float: 换算到实际 ``dt`` 下的等效 EMA 系数，位于 ``(0, 1]``。

    Fail-safe 行为：
        ``alpha`` 被钳制至 ``[1e-6, 1.0]``，``dt`` 下限钳制为 ``1e-6``，避免除零或复数幂。
    """
    # 钳制 alpha 与 dt 下限，确保 (1 - alpha) 非负且幂指数严格为正
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    dt = max(float(dt), 1e-6)
    return 1.0 - (1.0 - alpha) ** (dt / nominal_dt)


def scaled_decay(decay, dt, nominal_dt=0.01):
    """将标称周期的乘法衰减系数换算到实际周期 ``dt``，保证半衰期与帧率抖动无关。

    公式：
        ``decay_eff = decay ** (max(dt, 0.0) / nominal_dt)``。

    Args:
        decay (float): 标称周期 ``nominal_dt`` 下的单步保留比例（通常位于 ``(0, 1]``，无量纲）。
        dt (float): 当前实际经过的时间步长（秒，s）。
        nominal_dt (float, optional): 标称设计周期（秒，s），默认 ``0.01``（100Hz）。

    Returns:
        float: 实际步长 ``dt`` 对应的等效衰减倍率；当 ``dt <= 0`` 时退化为 ``1.0``（不衰减）。
    """
    return float(decay) ** (max(float(dt), 0.0) / nominal_dt)


def measurement_age(now, timestamp, max_age=0.25):
    """计算基于单调时钟的视觉观测年龄，并对异常时间戳做安全钳制。

    物理意义：
        测量视觉观测从捕获/校准时刻 ``timestamp`` 到当前控制时刻 ``now`` 已经过去的时间，
        用于卡尔曼滤波器从观测时刻向前外推至当前时刻。

    Args:
        now (float): 当前控制周期单调时钟时间（秒，s，来自 ``time.monotonic()``）。
        timestamp (float): 视觉观测时间戳（秒，s，须与 ``now`` 同一单调时钟基准）。
        max_age (float, optional): 允许的最大观测年龄上限（秒，s），默认 ``0.25``，
            防止长时间丢帧或线程阻塞后发生过度线性外推。

    Returns:
        float: 位于 ``[0.0, max_age]`` 内的观测年龄（秒，s）。

    Fail-safe 行为：
        若 ``age`` 为 ``NaN``/``Inf`` 或为负（时钟倒退/未来时间戳），直接返回 ``0.0``，
        避免负时间步或无限外推污染状态估计。
    """
    age = float(now) - float(timestamp)
    if not np.isfinite(age) or age < 0.0:
        return 0.0
    return min(age, float(max_age))


def horizontal_radial_soft_wall(position, velocity, radius):
    """水平径向软虚拟墙：仅限制水平半径 ``hypot(x, y)``，完全保留竖直 Z 轴工作空间。

    设计原理（70% 臂展防 J2 堵转规则）：
        当 TCP 水平投影半径 ``r = hypot(x, y) >= radius`` 时，沿水平单位径向向量
        ``r_hat = [x/r, y/r, 0]`` 分解笛卡尔线速度，仅投影剔除朝外的正径向分量：
        ``v_xy_new = v_xy - max(0, v_xy · r_hat) * r_hat``。
        相比三维球形虚拟墙，本实现不会在 TCP 抬高时错误压缩水平可达范围，且完全保留
        向内回缩速度、水平切向巡航速度及全部 Z 轴升降速度。

    Args:
        position (array_like): 基座世界系下 TCP 当前位置 ``[x, y, z]``（米，m）。
        velocity (array_like): 基座世界系下待限幅的 TCP 期望线速度 ``[vx, vy, vz]``（m/s）。
        radius (float): 水平软墙触发半径（米，m，工程红线通常 ``<= 0.42`` m）。

    Returns:
        tuple[np.ndarray, bool]: ``(clipped_velocity, wall_active)``：
            - ``clipped_velocity``: 形状 ``(3,)`` 的处理后线速度（m/s），不修改输入数组；
            - ``wall_active``: 布尔值，若发生向外速度削减或触发异常保护则为 ``True``。

    Fail-safe 行为：
        若 ``position`` 或 ``velocity`` 形状不是 ``(3,)`` 或含 ``NaN``/``Inf``，保守返回
        ``(np.zeros(3), True)`` 立即停机并报告触墙。
    """
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    result = velocity.copy()
    # 非法形状或非有限输入：保守清零速度并置位触墙标志
    if (position.shape != (3,) or velocity.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))):
        return np.zeros(3), True
    # 仅计算水平面内的柱面半径，排除原点奇异（distance <= 1e-9）
    distance = float(np.hypot(position[0], position[1]))
    if distance >= float(radius) and distance > 1e-9:
        radial = np.array([position[0] / distance, position[1] / distance, 0.0])
        outward = float(np.dot(result[:2], radial[:2]))
        if outward > 0.0:
            # 仅削去向外径向分量，切向与 Z 分量原样保留
            result[:2] -= outward * radial[:2]
            return result, True
    return result, False


def dynamic_lead_seconds(base_lead_ms, processing_s, max_lead_ms):
    """计算额外前瞻提前量（秒），用于补偿视觉处理耗时与机电执行固定余量。

    公式：
        ``lead_s = clip(base_lead_ms + processing_s * 1000.0, 0.0, max_lead_ms) / 1000.0``。

    注意：
        ``processing_s`` 必须仅代表单帧“取帧完成→目标发布”的管线处理耗时，绝不能传入
        会随 100Hz 重复控制周期递增的帧年龄（frame age）；观测年龄外推已由 KF 预测独立完成，
        拆分后可避免同一 60fps 视觉帧被多个伺服周期复用时前瞻量越算越大。

    Args:
        base_lead_ms (float): 基础固定执行前瞻余量（毫秒，ms）。
        processing_s (float): 本帧视觉处理延迟（秒，s）。
        max_lead_ms (float): 允许的最大前瞻上限（毫秒，ms）。

    Returns:
        float: 钳制后的前瞻提前时间（秒，s），位于 ``[0.0, max_lead_ms / 1000.0]``。

    Fail-safe 行为：
        当 ``processing_s`` 为负或非有限值时按 ``0.0`` 处理，防止异常延迟拉爆外推距离。
    """
    if not np.isfinite(processing_s) or processing_s < 0.0:
        processing_s = 0.0
    lead_ms = float(base_lead_ms) + processing_s * 1000.0
    return float(np.clip(lead_ms, 0.0, max_lead_ms)) / 1000.0


def camera_twist_from_tcp(tcp_twist_world, tcp_rotation, T_tcp_camera):
    """将世界系表达的 TCP 6D 速度（twist）转换到相机光心处的相机系 6D twist。

    运动学公式（刚体速度伴随变换）：
        设世界系下 TCP 线速度与角速度为 ``(v_tcp_w, omega_w)``，相机相对 TCP 的刚体变换为
        ``(R_tcp_cam, t_tcp_cam)``，则相机光心在世界系下的安装偏置向量为
        ``r_w = R_w_tcp @ t_tcp_cam``，光心在世界系下的线速度包含杠杆臂附加项：
        ``v_cam_w = v_tcp_w + omega_w × r_w``。
        最后经旋转矩阵 ``R_cam_w = (R_w_tcp @ R_tcp_cam)^T`` 投影至相机局部坐标系：
        ``twist_cam = [R_cam_w @ v_cam_w, R_cam_w @ omega_w]``。

    为何必须计入杠杆臂项：
        手眼相机安装偏离 TCP 原点时，纯腕部旋转也会在相机光心引起显著线速度；
        若忽略 ``omega_w × r_w``，自运动补偿将残留横向伪速度，导致静止手掌误唤醒。

    Args:
        tcp_twist_world (array_like): 世界系 TCP 速度 ``[vx, vy, vz, wx, wy, wz]``（m/s, rad/s）。
        tcp_rotation (array_like): 形状 ``(3, 3)`` 的旋转矩阵 ``R_world_tcp``。
        T_tcp_camera (array_like): 形状 ``(4, 4)`` 的手眼外参齐次变换矩阵 ``T_tcp_camera``（平移单位 m）。

    Returns:
        np.ndarray: 形状 ``(6,)`` 的相机系 twist ``[vx_c, vy_c, vz_c, wx_c, wy_c, wz_c]``（m/s, rad/s）。

    Fail-safe 行为：
        任意参数类型异常、形状不符或含 ``NaN``/``Inf`` 时返回零 6D 向量，使下游补偿退化为
        原始观测，不破坏控制安全链。
    """
    zero = np.zeros(6)
    try:
        twist = np.asarray(tcp_twist_world, dtype=np.float64)
        rotation = np.asarray(tcp_rotation, dtype=np.float64)
        transform = np.asarray(T_tcp_camera, dtype=np.float64)
    except (TypeError, ValueError):
        return zero
    if (twist.shape != (6,) or rotation.shape != (3, 3)
            or transform.shape != (4, 4)
            or not np.all(np.isfinite(twist))
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(transform))):
        return zero
    R_world_tcp = rotation
    R_tcp_camera = transform[:3, :3]
    t_tcp_camera = transform[:3, 3]
    omega_world = twist[3:]
    # 将手眼安装平移偏置旋至世界系，并叠加腕部角速度产生的杠杆线速度
    camera_offset_world = R_world_tcp @ t_tcp_camera
    v_camera_world = twist[:3] + np.cross(omega_world, camera_offset_world)
    # R_camera_world = (R_world_tcp @ R_tcp_camera)^T，将世界系线/角速度投影至相机系
    R_camera_world = (R_world_tcp @ R_tcp_camera).T
    return np.concatenate((R_camera_world @ v_camera_world,
                           R_camera_world @ omega_world))


def compensate_palm_velocity(observed_velocity, camera_twist_camera,
                             palm_position_camera):
    """扣除相机自身运动引起的画面表观速度，还原手掌相对世界的真实速度（相机系表达）。

    刚体运动学推导：
        对世界系中静止的一点 ``P``，当相机以自身系线速度 ``v_c`` 与角速度 ``omega_c`` 运动时，
        该点在相机系下的坐标时间导数满足：
        ``p_dot_c = -v_c - omega_c × p_c``。
        因此当目标相对世界以 ``v_palm_w``（在当前相机系投影为 ``v_palm_c``）运动时，观测速度为
        ``v_obs = v_palm_c - v_c - omega_c × p_c``，反解得补偿公式：
        ``v_palm_c = v_obs + v_c + omega_c × p_c``。

    Args:
        observed_velocity (array_like): 相机系下观测到的手掌表观速度 ``[vx, vy, vz]``（m/s）。
        camera_twist_camera (array_like): 相机系自身 6D twist ``[v_c, omega_c]``（m/s, rad/s）。
        palm_position_camera (array_like): 相机系下当前手掌 3D 位置 ``p_c = [x, y, z]``（m）。

    Returns:
        np.ndarray: 形状 ``(3,)`` 的补偿后手掌相对世界速度（在相机系坐标轴下表达，m/s）。

    Fail-safe 行为：
        - ``observed_velocity`` 无效（类型/形状/非有限）时返回 ``np.zeros(3)``；
        - 仅 ``camera_twist_camera`` 或 ``palm_position_camera`` 无效时，退化返回 ``observed.copy()``
          （跳过自运动补偿而不丢弃有效观测）。
    """
    try:
        observed = np.asarray(observed_velocity, dtype=np.float64)
        twist = np.asarray(camera_twist_camera, dtype=np.float64)
        palm = np.asarray(palm_position_camera, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(3)
    if (observed.shape != (3,) or not np.all(np.isfinite(observed))):
        return np.zeros(3)
    if (twist.shape != (6,) or palm.shape != (3,)
            or not np.all(np.isfinite(twist))
            or not np.all(np.isfinite(palm))):
        return observed.copy()
    return observed + twist[:3] + np.cross(twist[3:], palm)


def compensate_palm_velocity_delayed(observed_velocity, camera_twist_history,
                                     palm_position_camera):
    """基于历史 twist 窗口的延迟对齐相机自运动补偿（匹配视觉+KF 30~60ms 群延迟）。

    控制背景：
        机械臂高速接近静止手掌并进入死区瞬间，当前周期的关节指令速度已降为 0，
        但视觉 KF 输出的原始速度仍反映 30~60ms 前机械臂冲刺造成的画面相对位移。
        本函数遍历近期若干控制周期的 ``camera_twist_history``，选取使补偿后残差范数
        ``||v_obs + v_c(t_k) + omega_c(t_k) × p_c||_2`` 最小的候选结果，精确抵消滞后自运动残影，
        防止停靠瞬间因相位错位无法进入静止锁存。

    Args:
        observed_velocity (array_like): 相机系下视觉/KF 观测到的手掌速度 ``(3,)``（m/s）。
        camera_twist_history (Sequence[array_like]): 近期若干控制周期的相机系 6D twist 序列（m/s, rad/s）。
        palm_position_camera (array_like): 相机系下手掌 3D 位置 ``(3,)``（m）。

    Returns:
        np.ndarray: 形状 ``(3,)`` 的残差最小补偿速度（m/s）。

    Fail-safe 行为：
        ``observed_velocity`` 无效返回 ``np.zeros(3)``；历史窗口为空时返回 ``observed.copy()``。
    """
    try:
        observed = np.asarray(observed_velocity, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(3)
    if observed.shape != (3,) or not np.all(np.isfinite(observed)):
        return np.zeros(3)
    if not camera_twist_history:
        return observed.copy()
    best_vec = None
    best_norm = float("inf")
    # 在历史窗口内穷举候选 twist，选取残余速度 L2 范数最小者
    for twist in camera_twist_history:
        candidate = compensate_palm_velocity(
            observed, twist, palm_position_camera)
        norm = float(np.linalg.norm(candidate))
        if norm < best_norm:
            best_norm = norm
            best_vec = candidate
    return best_vec if best_vec is not None else observed.copy()


def camera_point_to_world(point_camera, tcp_position, tcp_rotation,
                          T_tcp_camera):
    """将相机坐标系下的 3D 点刚体变换至机械臂基座世界坐标系。

    变换公式：
        ``p_tcp = R_tcp_cam @ p_cam + t_tcp_cam``
        ``p_world = p_tcp_world + R_world_tcp @ p_tcp``。

    Args:
        point_camera (array_like): 相机系 3D 点坐标 ``(3,)``（m）。
        tcp_position (array_like): 世界系 TCP 原点位置 ``(3,)``（m）。
        tcp_rotation (array_like): 世界系 TCP 姿态旋转矩阵 ``R_world_tcp``，形状 ``(3, 3)``。
        T_tcp_camera (array_like): 手眼外参齐次矩阵 ``T_tcp_camera``，形状 ``(4, 4)``（平移单位 m）。

    Returns:
        np.ndarray: 世界系 3D 点坐标 ``(3,)``（m）；任意输入无效时返回 ``np.zeros(3)``。
    """
    zero = np.zeros(3)
    try:
        point = np.asarray(point_camera, dtype=np.float64)
        position = np.asarray(tcp_position, dtype=np.float64)
        rotation = np.asarray(tcp_rotation, dtype=np.float64)
        transform = np.asarray(T_tcp_camera, dtype=np.float64)
    except (TypeError, ValueError):
        return zero
    if (point.shape != (3,) or position.shape != (3,)
            or rotation.shape != (3, 3) or transform.shape != (4, 4)
            or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(transform))):
        return zero
    # 先经手眼外参转至 TCP 坐标系，再由正运动学旋转和平移映射至基座世界系
    point_tcp = transform[:3, :3] @ point + transform[:3, 3]
    return position + rotation @ point_tcp


def world_point_to_camera(point_world, tcp_position, tcp_rotation,
                          T_tcp_camera):
    """将机械臂基座世界坐标系下的 3D 点逆变换至相机坐标系。

    逆变换公式（利用正交矩阵 ``R^{-1} = R^T``）：
        ``p_tcp = R_world_tcp^T @ (p_world - p_tcp_world)``
        ``p_cam = R_tcp_cam^T @ (p_tcp - t_tcp_cam)``。

    Args:
        point_world (array_like): 世界系 3D 点坐标 ``(3,)``（m）。
        tcp_position (array_like): 世界系 TCP 原点位置 ``(3,)``（m）。
        tcp_rotation (array_like): 世界系 TCP 姿态旋转矩阵 ``R_world_tcp``，形状 ``(3, 3)``。
        T_tcp_camera (array_like): 手眼外参齐次矩阵 ``T_tcp_camera``，形状 ``(4, 4)``（平移单位 m）。

    Returns:
        np.ndarray: 相机系 3D 点坐标 ``(3,)``（m）；任意输入无效时返回 ``np.zeros(3)``。
    """
    zero = np.zeros(3)
    try:
        point = np.asarray(point_world, dtype=np.float64)
        position = np.asarray(tcp_position, dtype=np.float64)
        rotation = np.asarray(tcp_rotation, dtype=np.float64)
        transform = np.asarray(T_tcp_camera, dtype=np.float64)
    except (TypeError, ValueError):
        return zero
    if (point.shape != (3,) or position.shape != (3,)
            or rotation.shape != (3, 3) or transform.shape != (4, 4)
            or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(transform))):
        return zero
    # 依次撤销世界→TCP 与 TCP→相机 刚体变换
    point_tcp = rotation.T @ (point - position)
    return transform[:3, :3].T @ (point_tcp - transform[:3, 3])


def _smoothstep(value):
    """三次 Hermite 平滑阶跃函数（``[0, 1]`` 上的 C1 连续过渡）。

    公式：``S(u) = 3u^2 - 2u^3``（``u = clip(value, 0, 1)``），满足
    ``S(0)=0, S(1)=1, S'(0)=S'(1)=0``，避免在门控/调度阈值边界产生加速度与力矩跳变。

    Args:
        value (float): 归一化输入标量（超出 ``[0, 1]`` 将被自动钳制）。

    Returns:
        float: 平滑过渡后的权重值，位于 ``[0.0, 1.0]``。
    """
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class AdaptiveLeadGate:
    """自适应前瞻（Lead）门控器：运动追赶中保留提前量，停手/近目标/反向纠偏时快速收回。

    控制物理意义：
        本类输出权重 ``weight ∈ [0, 1]`` 用于调制**额外前瞻提前量**（``dynamic_lead_seconds``），
        不作用于观测年龄外推。目标权重由三个 C1 平滑因子连乘：
        ``target = speed_w * distance_w * direction_w``
        - ``speed_w``：低速残余噪声不参与前瞻外推；
        - ``distance_w``：临近停靠目标时提前撤销前瞻，抑制过冲；
        - ``direction_w``：要求估计速度方向与未消除位置误差同向（``cos > 0``），
          若 ``cos <= 0``（已越过目标或突然换向）立即硬清零，杜绝沿旧方向多推一拍。
        权重平滑采用非对称一阶指数滤波（``attack_tau`` 慢升、``release_tau`` 快撤），
        且按真实 ``dt`` 计算系数 ``1 - exp(-dt / tau)``，防止在静止边界发生 0/满量程拍打。
    """

    def __init__(self, speed_off=0.06, speed_on=0.13,
                 near_dist=0.022, full_dist=0.045,
                 direction_full_cos=0.50,
                 attack_tau=0.025, release_tau=0.015):
        """初始化自适应前瞻门控器。

        Args:
            speed_off (float): 速度权重归零下限（m/s），低于此速 ``speed_w = 0``。
            speed_on (float): 速度权重满额上限（m/s），高于此速 ``speed_w = 1``。
            near_dist (float): 距离权重归零下限（m），误差小于此值不加额外 lead。
            full_dist (float): 距离权重满额上限（m），误差大于此值距离因子为 ``1``。
            direction_full_cos (float): 速度与误差夹角余弦达到满额权重的阈值（无量纲，钳制至 ``[0, 0.99]``）。
            attack_tau (float): 权重上升时间常数（秒，s），决定恢复 lead 的平滑快慢。
            release_tau (float): 权重回落时间常数（秒，s），通常小于 ``attack_tau`` 以快速撤权。
        """
        self.speed_off = max(0.0, float(speed_off))
        self.speed_on = max(self.speed_off + 1e-6, float(speed_on))
        self.near_dist = max(0.0, float(near_dist))
        self.full_dist = max(self.near_dist + 1e-6, float(full_dist))
        self.direction_full_cos = float(np.clip(direction_full_cos, 0.0, 0.99))
        self.attack_tau = max(1e-3, float(attack_tau))
        self.release_tau = max(1e-3, float(release_tau))
        self.weight = 0.0

    def reset(self):
        """复位内部滤波权重为 ``0.0``（立即撤销全部额外前瞻）。"""
        self.weight = 0.0

    def update(self, velocity, position_error, dt, moving, valid=True):
        """根据当前目标速度、位置误差与运行状态更新前瞻门控权重。

        Args:
            velocity (array_like): 估计目标速度向量 ``(3,)``（m/s）。
            position_error (array_like): 当前待消除的位置误差向量 ``(3,)``（m，指向目标）。
            dt (float): 本次控制步长（秒，s）。
            moving (bool): 上游是否判定目标处于运动态；``False`` 时立即清零。
            valid (bool, optional): 当前视觉/滤波观测是否有效，默认 ``True``。

        Returns:
            tuple[float, float]: ``(smoothed_weight, raw_target)``，两者均位于 ``[0.0, 1.0]``。

        Fail-safe 行为：
            非运动、观测无效、数组形状不符、含 ``NaN``/``Inf`` 或 ``cos <= 0``（反向/越界）
            均立即调用 ``reset()`` 并返回 ``(0.0, 0.0)``；非有限/负 ``dt`` 钳制为 ``0.0``。
        """
        if not moving or not valid:
            self.reset()
            return 0.0, 0.0
        velocity = np.asarray(velocity, dtype=np.float64)
        position_error = np.asarray(position_error, dtype=np.float64)
        if (velocity.shape != (3,) or position_error.shape != (3,)
                or not np.all(np.isfinite(velocity))
                or not np.all(np.isfinite(position_error))):
            self.reset()
            return 0.0, 0.0

        speed = float(np.linalg.norm(velocity))
        error = float(np.linalg.norm(position_error))
        # 三因子分别经 C1 smoothstep 映射到 [0, 1]
        speed_w = _smoothstep((speed - self.speed_off)
                              / (self.speed_on - self.speed_off))
        distance_w = _smoothstep((error - self.near_dist)
                                 / (self.full_dist - self.near_dist))
        direction_w = 0.0
        if speed > 1e-9 and error > 1e-9:
            cosine = float(np.dot(velocity, position_error) / (speed * error))
            if cosine <= 0.0:
                # 已越过目标或明确换向：不能带着旧 lead 再外推一拍。
                self.reset()
                return 0.0, 0.0
            direction_w = _smoothstep(
                cosine / max(self.direction_full_cos, 1e-6))
        target = speed_w * distance_w * direction_w

        # 非对称一阶低通：增权用 attack_tau，撤权用更快的 release_tau
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            dt = 0.0
        tau = self.attack_tau if target > self.weight else self.release_tau
        alpha = 1.0 - np.exp(-dt / tau)
        self.weight += (target - self.weight) * alpha
        # 截断极小尾部残值，防止浮点长尾持续保持非零状态
        if self.weight < 1e-4:
            self.weight = 0.0
        return float(self.weight), float(target)


class FastStartGate:
    """静止锁存状态的快速起步退出门控：连续、同向、累计位移超噪门槛即解锁。

    控制物理意义：
        当机械臂处于静止锁存态时，常规速度估计因 KF 群延迟可能需要 4~6 帧才跨过
        ``deadband_on``。本类作为独立的位移快速通道，仅在收到新的有效 3D 观测时调用：
        若目标相对静止锚点 ``origin`` 的位移连续 ``frames`` 帧超过 ``distance``，且相邻帧
        位移方向余弦 ``>= direction_cos``（排除往复随机抖动），立即触发解锁。
        本通道不改变进入静止的条件，也不放宽任何深度保护门槛。
    """

    def __init__(self, enable=True, frames=2, distance=0.030,
                 direction_cos=0.80):
        """初始化快速起步门控器。

        Args:
            enable (bool): 是否启用快速起步通道，默认 ``True``。
            frames (int): 触发解锁所需的连续同向超距帧数，最小为 ``1``。
            distance (float): 相对静止锚点 ``origin`` 的位移触发门槛（米，m）。
            direction_cos (float): 连续两帧位移向量夹角余弦下限（无量纲），用于抑制抖动。
        """
        self.enable = bool(enable)
        self.frames = max(1, int(frames))
        self.distance = float(distance)
        self.direction_cos = float(direction_cos)
        self.origin = None
        self.prev_delta = None
        self.count = 0

    def reset(self, origin=None):
        """重置快速起步状态，并可选设定新的静止锚点。

        Args:
            origin (array_like, optional): 新的 3D 静止参考点（m）；传入 ``None`` 则清空锚点。
        """
        self.origin = (None if origin is None else
                       np.asarray(origin, dtype=np.float64).copy())
        self.prev_delta = None
        self.count = 0

    def update(self, point):
        """喂入一帧新的有效 3D 掌心位置，判断是否满足快速起步解锁条件。

        Args:
            point (array_like): 当前新观测到的 3D 位置 ``(3,)``（m，须与 ``origin`` 同坐标系）。

        Returns:
            bool: 连续同向超距计数达到 ``self.frames`` 返回 ``True``，否则返回 ``False``。

        Fail-safe 行为：
            未启用、尚未设置 ``origin``，或 ``point`` 形状非 ``(3,)``/含非有限值时复位并返回 ``False``。
        """
        if not self.enable or self.origin is None:
            return False
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            self.reset(point if point.shape == (3,) else None)
            return False
        delta = point - self.origin
        norm = float(np.linalg.norm(delta))
        if self.prev_delta is not None:
            prev_norm = float(np.linalg.norm(self.prev_delta))
            if norm > 1e-6 and prev_norm > 1e-6:
                cosine = float(np.dot(delta, self.prev_delta) / (norm * prev_norm))
                if cosine < self.direction_cos:
                    # 换向不能继承旧方向的确认计数。
                    self.reset(point)
                    return False
        # 仅当当前位移超过门槛时递增计数，否则清零要求严格连续
        self.count = self.count + 1 if norm >= self.distance else 0
        self.prev_delta = delta.copy()
        return self.count >= self.frames


class SlowFollowGate:
    """静止锁存态下的慢速连续同向拖动确认门控（带迟滞释放，不解锁前馈与 lead）。

    状态机行为：
        - 锚定：首次有效观测自动记录 ``origin``；
        - 迟滞释放区（``dist < release_distance``）：若已激活则重新锚定原点并退出激活；
          若未激活则保留旧原点以累计极慢速微位移；
        - 迟滞保持区（``release_distance <= dist < distance``）：维持当前 ``active`` 状态，
          防止在阈值边界频繁抖动切换；
        - 同向确认区（``dist >= distance``）：检查与上一帧位移向量余弦 ``>= direction_cos``，
          连续满足 ``confirm_frames`` 帧后置位 ``active = True``；一旦换向则立即以当前点
          重新锚定并退出激活。
    """

    def __init__(self, distance=0.014, release_distance=0.008,
                 confirm_frames=3, direction_cos=0.80):
        """初始化慢速拖动门控器。

        Args:
            distance (float): 触发慢拖确认的位移上限阈值（m）。
            release_distance (float): 退出慢拖激活的迟滞下限阈值（m，通常小于 ``distance``）。
            confirm_frames (int): 激活慢拖所需的连续同向超距帧数，最小为 ``1``。
            direction_cos (float): 相邻帧位移方向一致性余弦下限（无量纲）。
        """
        self.distance = float(distance)
        self.release_distance = float(release_distance)
        self.confirm_frames = max(1, int(confirm_frames))
        self.direction_cos = float(direction_cos)
        self.reset()

    def reset(self, origin=None):
        """复位慢拖状态机，并可选用合法 3D 点重新锚定原点。

        Args:
            origin (array_like, optional): 可选的 3D 锚点坐标（m）；非法或含 ``NaN`` 时忽略。
        """
        self.origin = None
        if origin is not None:
            origin = np.asarray(origin, dtype=np.float64)
            if origin.shape == (3,) and np.all(np.isfinite(origin)):
                self.origin = origin.copy()
        self.previous_delta = None
        self.coherent_frames = 0
        self.active = False

    def update(self, point, enabled=True):
        """按新 3D 观测推进慢速拖动状态机。

        Args:
            point (array_like): 当前 3D 观测点 ``(3,)``（m）。
            enabled (bool, optional): 是否允许慢拖功能，默认 ``True``；``False`` 时复位并返回 ``False``。

        Returns:
            bool: 当前是否处于慢速拖动激活态（``self.active``）。

        Fail-safe 行为：
            禁用、点形状非 ``(3,)`` 或含 ``NaN``/``Inf`` 时自动复位并返回 ``False``。
        """
        point = np.asarray(point, dtype=np.float64)
        if not enabled or point.shape != (3,) or not np.all(np.isfinite(point)):
            self.reset()
            return False
        if self.origin is None:
            self.origin = point.copy()
            return False
        delta = point - self.origin
        distance = float(np.linalg.norm(delta))
        if distance < self.release_distance:
            # 未激活时必须保留原点，才能累计每帧都小于阈值的慢拖；
            # 已激活后回到原点附近才重新锚定并释放。
            if self.active:
                self.origin = point.copy()
            self.previous_delta = None
            self.coherent_frames = 0
            self.active = False
            return False
        if distance < self.distance:
            # 处于迟滞带 [release_distance, distance) 内，保持原状态不变
            return self.active
        direction_ok = True
        if self.previous_delta is not None:
            previous_norm = float(np.linalg.norm(self.previous_delta))
            if previous_norm > 1e-9 and distance > 1e-9:
                cosine = float(np.dot(delta, self.previous_delta) /
                               (distance * previous_norm))
                direction_ok = cosine >= self.direction_cos
        if direction_ok:
            self.coherent_frames += 1
        else:
            # 换向后从当前点重新确认，旧方向的锁存迁移立即停止。
            self.origin = point.copy()
            self.previous_delta = None
            self.coherent_frames = 0
            self.active = False
            return False
        self.previous_delta = delta.copy()
        if self.coherent_frames >= self.confirm_frames:
            self.active = True
        return self.active


class CameraTimestampCalibrator:
    """相机硬件时间戳到主机单调时钟（``time.monotonic()``）的在线偏移校准器。

    控制物理意义：
        Orbbec 深度相机硬件时间戳反映真实曝光/成像时刻，但不受 USB 传输抖动影响，
        与主机 ``time.monotonic()`` 之间存在固定偏置及微小晶振频偏。本类在线估计时钟偏置
        ``offset_s ≈ host_s - raw_s``，并按时间常数 ``offset_tau_s`` 低通跟踪晶振漂移：
        ``mapped_s = raw_s + offset_s``，``transport_s = host_s - mapped_s``。
        遇到时钟跳变、倒退、负传输延迟或超长传输阻塞的异常帧时，自动回退为主机到达时刻
        ``host_arrival_s``，确保下游观测年龄与位姿查表绝不被损坏时间戳污染。
    """

    def __init__(self, max_transport_s=0.20, max_raw_step_s=0.50,
                 offset_tau_s=2.0):
        """初始化相机硬件时间戳校准器。

        Args:
            max_transport_s (float): 允许的最大合理 USB/SDK 传输延迟上限（秒，s）。
            max_raw_step_s (float): 相邻两帧硬件时间戳允许的最大步进间隔（秒，s），超过则视为掉线/重置。
            offset_tau_s (float): 跟踪主机与相机晶振慢漂移的一阶指数滤波时间常数（秒，s）。
        """
        self.max_transport_s = float(max_transport_s)
        self.max_raw_step_s = float(max_raw_step_s)
        self.offset_tau_s = float(offset_tau_s)
        self.offset_s = None
        self.last_raw_s = None
        self.last_host_s = None
        self.rejected = 0
        self.used = 0

    def reset(self):
        """清空时钟偏置与上一帧参考点（保留累计统计计数 ``used``/``rejected``）。"""
        self.offset_s = None
        self.last_raw_s = None
        self.last_host_s = None

    def _fallback(self, host_s, raw_s=None, reset_origin=False):
        """异常帧统一回退处理：累加拒绝计数并返回主机到达时刻。

        Args:
            host_s (float): 主机单调时钟到达时刻（秒，s）。
            raw_s (float, optional): 若未触发完全重置，用于更新 ``last_raw_s``/``last_host_s``。
            reset_origin (bool, optional): 为 ``True`` 时彻底重置偏置原点（如发生时钟跳变或非法值）。

        Returns:
            tuple[float, bool]: ``(host_s, False)`` 表示本次未使用硬件时间戳。
        """
        self.rejected += 1
        if reset_origin:
            self.reset()
        elif raw_s is not None:
            self.last_raw_s = raw_s
            self.last_host_s = host_s
        return host_s, False

    def update(self, raw_timestamp_s, host_arrival_s):
        """将一帧相机的硬件秒级时间戳映射至主机单调时钟基准。

        Args:
            raw_timestamp_s (float or None): 相机 SDK 报告的硬件时间戳（秒，s）；``None`` 直接回退。
            host_arrival_s (float): 主机收到该帧时的 ``time.monotonic()`` 时刻（秒，s）。

        Returns:
            tuple[float, bool]: ``(calibrated_time_s, used_hardware_ts)``：
                - ``calibrated_time_s``: 校准后的单调时间戳（秒，s），异常时回退为 ``host_arrival_s``；
                - ``used_hardware_ts``: 布尔值，``True`` 表示硬件时间戳通过一致性校验并被采纳。

        Fail-safe 行为：
            输入为 ``None``/类型错误/非有限/负数、相邻帧步进非正或超大（相机重启/长时间卡顿）、
            或者映射后传输延迟落在 ``[-0.005, max_transport_s]`` 之外时，均安全回退到 ``host_arrival_s``。
        """
        host_s = float(host_arrival_s)
        if raw_timestamp_s is None:
            return host_s, False
        try:
            raw_s = float(raw_timestamp_s)
        except (TypeError, ValueError, OverflowError):
            return self._fallback(host_s, reset_origin=True)
        if not math.isfinite(raw_s) or not math.isfinite(host_s) or raw_s < 0.0:
            return self._fallback(host_s, reset_origin=True)
        if self.last_raw_s is not None:
            raw_step = raw_s - self.last_raw_s
            host_step = host_s - self.last_host_s
            # 硬件/主机时间戳必须严格单调递增且帧间隔在合理上限内，否则重置偏置
            if (raw_step <= 0.0 or raw_step > self.max_raw_step_s
                    or host_step <= 0.0 or host_step > 1.0):
                return self._fallback(host_s, reset_origin=True)
        if self.offset_s is None:
            # 首帧以当前到达时刻建立初始时钟偏置基准
            self.offset_s = host_s - raw_s
        mapped_s = raw_s + self.offset_s
        transport_s = host_s - mapped_s
        # 负传输延迟意味着倒退/异常时钟；过大延迟按陈旧主机帧处理，不送入控制链。
        if transport_s < -0.005 or transport_s > self.max_transport_s:
            return self._fallback(host_s, raw_s=raw_s)
        if self.last_host_s is not None:
            # 慢速指数跟踪晶振漂移：将当前残余传输偏置按时间常数 offset_tau_s 缓慢吸收入 offset_s
            elapsed = max(1e-4, host_s - self.last_host_s)
            alpha = 1.0 - math.exp(-elapsed / max(1e-3, self.offset_tau_s))
            self.offset_s += alpha * transport_s
        self.last_raw_s = raw_s
        self.last_host_s = host_s
        self.used += 1
        return mapped_s, True


class StaticVelocityHold:
    """静止锁存且位于位置死区内时的低速残余指令清零保持器。

    控制物理意义：
        高速停靠阶段仍完全遵循原有速度 EMA 与加速度限制以平滑减速；
        只有当系统同时满足（1）静止锁存 ``stationary=True``、（2）进入位置死区
        ``in_deadband=True``、且（3）当前指令线速度幅值 ``||v[:3]||_2 <= speed_limit``
        并连续保持 ``confirm_frames`` 个控制周期后，才将 6D 速度指令彻底清零，
        消除 EMA 指数长尾引起的微小蠕动。一旦恢复运动、离开死区或禁用立即复位，
        绝不拦截正常起步、慢拖或反向纠偏。返回值始终为副本，不修改调用方原数组。
    """

    def __init__(self, speed_limit=0.025, confirm_frames=3):
        """初始化静止残余速度清零保持器。

        Args:
            speed_limit (float): 允许触发清零的指令线速度范数上限（m/s，必须为有限正数）。
            confirm_frames (int): 触发彻底清零前需连续满足条件的控制周期数，最小为 ``1``。

        Raises:
            ValueError: 当 ``speed_limit`` 非有限或 ``<= 0`` 时抛出。
        """
        self.speed_limit = float(speed_limit)
        self.confirm_frames = max(1, int(confirm_frames))
        if (not math.isfinite(self.speed_limit) or self.speed_limit <= 0.0):
            raise ValueError("静止残余速度门槛必须为有限正数")
        self.count = 0
        self.active = False

    def reset(self):
        """清零连续确认计数并退出保持激活态。"""
        self.count = 0
        self.active = False

    def update(self, velocity, stationary, in_deadband, enabled=True):
        """根据当前 6D 指令速度与停靠状态决定是否清零低速残余指令。

        Args:
            velocity (array_like): 当前 6D 笛卡尔速度指令 ``[vx, vy, vz, wx, wy, wz]``（m/s, rad/s）。
            stationary (bool): 上游是否处于目标静止锁存态。
            in_deadband (bool): 当前位置误差是否位于静止位置死区内。
            enabled (bool, optional): 功能使能开关，默认 ``True``。

        Returns:
            tuple[np.ndarray, bool]: ``(output_velocity, hold_active)``：
                - ``output_velocity``: 形状 ``(6,)`` 的输出速度副本；激活保持时为全零；
                - ``hold_active``: 本周期是否已激活残速清零保持。

        Fail-safe 行为：
            若 ``velocity`` 形状不是 ``(6,)`` 或含 ``NaN``/``Inf``，复位并返回 ``(np.zeros(6), False)``。
        """
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
            self.reset()
            return np.zeros(6), False
        if not enabled or not stationary or not in_deadband:
            self.reset()
            return velocity.copy(), False
        if float(np.linalg.norm(velocity[:3])) > self.speed_limit:
            self.reset()
            return velocity.copy(), False
        self.count += 1
        self.active = self.count >= self.confirm_frames
        if self.active:
            return np.zeros(6), True
        return velocity.copy(), False


class AxisProcessNoiseGate:
    """基于连续同向创新确认的分轴卡尔曼过程噪声（Q）自适应倍率门控器。

    控制物理意义：
        目标突然加速或变向时，卡尔曼滤波器新息（innovation = 观测 - 预测）会在对应轴上
        连续多帧呈现同号大幅偏差；而 RGB-D 深度（Z 轴）噪声显著高于图像平面 XY。
        本类对 X/Y/Z 三轴独立计数：仅当某轴 ``|innovation[axis]| >= ref[axis]`` 且连续
        同向达到 ``confirm_frames`` 帧时，才将该轴目标 Q 倍率提升至
        ``clip(|innovation| / ref, 1.0, max_scale)``，再经非对称指数平滑（快升 ``attack_tau``、
        慢回落 ``release_tau``）输出。默认仅增强 XY（``max_xy=1.60``），Z 轴可独立封顶
        为 ``1.0`` 以防深度尖峰穿透滤波。
    """

    def __init__(self, ref_xy=0.060, ref_z=0.080, max_xy=1.60, max_z=1.0,
                 confirm_frames=2, attack_tau=0.040, release_tau=0.120):
        """初始化分轴自适应过程噪声门控器。

        Args:
            ref_xy (float): X/Y 轴触发 Q 放大的新息幅值基准（m，须为有限正数）。
            ref_z (float): Z 轴触发 Q 放大的新息幅值基准（m，须为有限正数）。
            max_xy (float): X/Y 轴允许的最大 Q 放大倍率（无量纲，须 ``>= 1.0``）。
            max_z (float): Z 轴允许的最大 Q 放大倍率（无量纲，须 ``>= 1.0``，默认 ``1.0`` 不放大）。
            confirm_frames (int): 触发放大的连续同号超阈值帧数，最小为 ``1``。
            attack_tau (float): 倍率上升时间常数（秒，s）。
            release_tau (float): 倍率回落至 ``1.0`` 的时间常数（秒，s）。

        Raises:
            ValueError: 任一基准非正、最大倍率小于 ``1.0`` 或含非有限值时抛出。
        """
        self.refs = np.array([ref_xy, ref_xy, ref_z], dtype=np.float64)
        self.max_scales = np.array([max_xy, max_xy, max_z], dtype=np.float64)
        if (np.any(self.refs <= 0.0) or np.any(self.max_scales < 1.0)
                or not np.all(np.isfinite(self.refs))
                or not np.all(np.isfinite(self.max_scales))):
            raise ValueError("分轴 Q 参数必须为有限正数且倍率不小于 1")
        self.confirm_frames = max(1, int(confirm_frames))
        self.attack_tau = max(1e-3, float(attack_tau))
        self.release_tau = max(1e-3, float(release_tau))
        self.scales = np.ones(3)
        self.counts = np.zeros(3, dtype=int)
        self.signs = np.zeros(3, dtype=int)

    def reset(self):
        """将三轴 Q 倍率复位为 ``1.0``，并清空连续同向计数与符号记忆。"""
        self.scales[:] = 1.0
        self.counts[:] = 0
        self.signs[:] = 0

    def relax(self, dt):
        """在无新观测的控制周期中，使三轴 Q 倍率按 ``release_tau`` 平滑回落至 ``1.0``。

        Args:
            dt (float): 本周期经过的时间步长（秒，s）。

        Returns:
            np.ndarray: 形状 ``(3,)`` 的当前三轴 Q 倍率副本。
        """
        dt = max(1e-4, float(dt))
        alpha = 1.0 - math.exp(-dt / self.release_tau)
        self.scales += (1.0 - self.scales) * alpha
        return self.scales.copy()

    def update(self, innovation, dt, valid=True):
        """根据新到来的 3D 卡尔曼新息向量更新各轴 Q 倍率。

        Args:
            innovation (array_like): 3D 观测新息 ``z_meas - z_pred``，形状 ``(3,)``（m）。
            dt (float): 距离上次更新的时间间隔（秒，s）。
            valid (bool, optional): 本次新息观测是否有效，默认 ``True``。

        Returns:
            np.ndarray: 形状 ``(3,)`` 的平滑后分轴 Q 倍率副本，各分量位于 ``[1.0, max_scale]``。

        Fail-safe 行为：
            当 ``valid=False``、``innovation`` 形状非法或含 ``NaN``/``Inf``、或 ``dt`` 非有限时，
            立即复位全部倍率为 ``1.0`` 并返回。
        """
        innovation = np.asarray(innovation, dtype=np.float64)
        if (not valid or innovation.shape != (3,)
                or not np.all(np.isfinite(innovation)) or not math.isfinite(dt)):
            self.reset()
            return self.scales.copy()
        magnitude = np.abs(innovation)
        # 逐轴统计连续同号超阈值的创新帧数，单帧噪声毛刺或正负交替抖动不会累积确认
        for axis in range(3):
            if magnitude[axis] < self.refs[axis]:
                self.counts[axis] = 0
                self.signs[axis] = 0
                continue
            sign = 1 if innovation[axis] >= 0.0 else -1
            if sign == self.signs[axis]:
                self.counts[axis] += 1
            else:
                self.counts[axis] = 1
            self.signs[axis] = sign
        desired = np.ones(3)
        confirmed = self.counts >= self.confirm_frames
        desired[confirmed] = np.clip(
            magnitude[confirmed] / self.refs[confirmed], 1.0,
            self.max_scales[confirmed])
        # 非对称平滑：需要提高带宽时用 attack_tau 快响应，恢复稳态时用 release_tau 缓回落
        attacking = desired > self.scales
        alpha = np.empty(3)
        alpha[attacking] = 1.0 - math.exp(-max(1e-4, dt) / self.attack_tau)
        alpha[~attacking] = 1.0 - math.exp(-max(1e-4, dt) / self.release_tau)
        self.scales += (desired - self.scales) * alpha
        self.scales = np.clip(self.scales, 1.0, self.max_scales)
        return self.scales.copy()


def adaptive_feedforward_scale(error_vector, target_velocity, base_taper_dist,
                               enabled=True, speed_ref=0.35, max_taper_dist=0.12):
    """速度感知的停靠前馈衰减比例：高速接近时提前收紧前馈以抑制停靠过冲。

    公式与物理意义：
        - 沿位置误差反方向计算接近速度（仅在目标正朝向误差缩小方向相对接近时为正）：
          ``approach_speed = max(0.0, -(target_velocity · error) / ||error||)``；
        - 当 ``approach_speed`` 超过起步门槛 ``onset = min(0.10, 0.5 * speed_ref)`` 后，
          按线性插值 ``blend ∈ [0, 1]`` 将停靠前馈衰减距离从 ``base_taper_dist`` 扩展至
          ``max_taper_dist``；
        - 最终前馈比例取 ``min(old_scale, min(1.0, ||error|| / taper))``，保证仅会比基线
          更早减前馈，绝不放大前馈。

    Args:
        error_vector (array_like): 3D 位置误差向量 ``(3,)``（m，指向目标）。
        target_velocity (array_like): 3D 目标速度向量 ``(3,)``（m/s）。
        base_taper_dist (float): 静态/低速基线的前馈线性衰减距离（m，须为有限正数）。
        enabled (bool, optional): 是否启用速度感知扩展停靠距，默认 ``True``；``False`` 退化为固定 ``base_taper_dist``。
        speed_ref (float, optional): 达到 ``max_taper_dist`` 时的参考接近速度（m/s）。
        max_taper_dist (float, optional): 高速接近时的最大停靠衰减距离上限（m）。

    Returns:
        tuple[float, float, float]: ``(ff_scale, taper_dist, approach_speed)``：
            - ``ff_scale``: 前馈缩放比例，位于 ``[0.0, 1.0]``；
            - ``taper_dist``: 本周期生效的停靠衰减距离（m）；
            - ``approach_speed``: 计算得到的接近速度标量（m/s，非负）。

    Fail-safe 行为：
        输入向量形状非法、含非有限值或 ``base_taper_dist <= 0`` 时，保守返回前馈 ``0.0``。
    """
    error = np.asarray(error_vector, dtype=np.float64)
    velocity = np.asarray(target_velocity, dtype=np.float64)
    base = float(base_taper_dist)
    if (error.shape != (3,) or velocity.shape != (3,)
            or not np.all(np.isfinite(error)) or not np.all(np.isfinite(velocity))
            or not math.isfinite(base) or base <= 0.0):
        return 0.0, max(base, 1e-3) if math.isfinite(base) else 0.07, 0.0
    error_norm = float(np.linalg.norm(error))
    old_scale = min(1.0, error_norm / base)
    if not enabled or error_norm <= 1e-9:
        return old_scale, base, 0.0
    # 误差指向目标：当速度与误差反向时，说明相对距离正在快速收拢（高速逼近停靠点）
    approach_speed = max(0.0, -float(np.dot(velocity, error)) / error_norm)
    speed_ref = max(1e-3, float(speed_ref))
    max_taper = max(base, float(max_taper_dist))
    # 慢速接近完整保留 v8.0；超过 0.10m/s 后才连续增加停靠距离。
    onset = min(0.10, 0.5 * speed_ref)
    blend = float(np.clip((approach_speed - onset)
                          / max(speed_ref - onset, 1e-6), 0.0, 1.0))
    taper = base + (max_taper - base) * blend
    return min(old_scale, min(1.0, error_norm / taper)), taper, approach_speed


def scale_joint_velocities(joint_vel, max_speed, grouped=False):
    """分关节向量限速与等比例缩放：任一轴超限时整体同比例收缩，严格保持笛卡尔运动方向。

    控制物理意义：
        对各关节计算超限比 ``ratio_i = |dq_i| / max_speed_i``。
        - 当 ``grouped=False``（默认）时，取全臂最大比 ``r_max = max(ratio_i)``，若 ``r_max > 1``
          则将整条速度向量除以 ``r_max``。这严格保持 ``J @ dq`` 的三维空间运动方向与姿态协同，
          避免单轴硬截断导致轨迹偏航，同时压制斜向高速停靠的合成总动能；
        - 当 ``grouped=True`` 且为 6 轴时，按水平偏航组 ``[J1, J5]`` 与竖直俯仰组 ``[J2, J3, J4, J6]``
          分组等比例限幅。

    Args:
        joint_vel (array_like): 期望关节角速度向量 ``dq``（rad/s，一维非空数组）。
        max_speed (float or array_like): 标量统一限速或与 ``joint_vel`` 等长的分轴限速向量（rad/s，严格为正）。
        grouped (bool, optional): 是否启用 6 轴分组限幅，默认 ``False``（推荐全臂统一缩放）。

    Returns:
        np.ndarray: 限速后的关节角速度副本（rad/s），绝不修改输入数组。

    Fail-safe 行为：
        ``joint_vel`` 维度非法/空/含 ``NaN``/``Inf``，或 ``max_speed`` 形状不匹配/含非正非有限值时，
        保守返回全零速度向量。
    """
    dq = np.asarray(joint_vel, dtype=np.float64)
    if dq.ndim != 1 or dq.size == 0 or not np.all(np.isfinite(dq)):
        return np.zeros_like(dq, dtype=np.float64) if dq.ndim == 1 else np.zeros(6)
    limit = np.asarray(max_speed, dtype=np.float64)
    if limit.ndim == 0:
        limit = np.full(dq.shape, float(limit), dtype=np.float64)
    if (limit.shape != dq.shape or not np.all(np.isfinite(limit))
            or np.any(limit <= 0.0)):
        return np.zeros_like(dq)
    ratios = np.abs(dq) / limit
    if grouped and dq.shape == (6,):
        # 分组等比缩放：水平转动耦合轴 [0, 4] 与竖直平面臂杆轴 [1, 2, 3, 5] 各自保比例
        out = dq.copy()
        for group in ([0, 4], [1, 2, 3, 5]):
            group_ratio = float(np.max(ratios[group]))
            if group_ratio > 1.0:
                out[group] /= group_ratio
        return out
    max_ratio = float(np.max(ratios))
    if max_ratio > 1.0:
        return dq / max_ratio
    return dq.copy()


def apply_axis_reversal_damping(v_target, actual_vel, prev_cmd_vel,
                                residual_scale, enabled=True):
    """分轴独立换向惯性衰减：同步收缩 EMA 状态与加速度限制器在过零前的旧方向残速。

    解决两个典型闭环问题：
        1. 三维整体点积 ``v_target · actual_vel < 0`` 在画圆/斜向变向时，单轴反向易被正交同向轴掩盖，
           或反之牵连正交巡航轴无辜减速；本函数改为 X/Y/Z 分轴独立判定 ``v_target[i] * v[i] < 0``。
        2. 若只衰减滤波状态 ``actual_vel`` 而不衰减上一拍指令 ``prev_cmd_vel``，下游 slew-rate
           加速度限制器会直接以未衰减的 ``prev_cmd_vel`` 覆盖 ``actual_vel``，导致 ``residual_scale``
           失效。此处对换向轴按 ``0.5 * (1 + scale)`` 同步收缩 ``prev_cmd_vel``，既削去 ~100ms
           换向拖尾，又确保过零后的反向起步仍严格受最大加速度约束。

    Args:
        v_target (array_like): 当前周期期望 3D 线速度 ``(3,)``（m/s）。
        actual_vel (array_like): 速度 EMA 滤波状态 ``(3,)``（m/s）。
        prev_cmd_vel (array_like): 上一控制周期下发的速度指令 ``(3,)``（m/s，用作加速度限制基准）。
        residual_scale (float): 换向轴残余速度保留比例，钳制至 ``[0.0, 1.0]``（越小制动越果断）。
        enabled (bool, optional): 是否执行衰减，默认 ``True``；``False`` 时仅计算 ``any_reversing`` 标志。

    Returns:
        tuple[np.ndarray, np.ndarray, bool]: ``(new_actual, new_prev_cmd, any_reversing)``：
            - ``new_actual`` / ``new_prev_cmd``: 形状 ``(3,)`` 的处理后速度副本（m/s）；
            - ``any_reversing``: 是否存在任一轴换向或整体速度点积为负。

    Fail-safe 行为：
        任一向量形状非 ``(3,)`` 或含 ``NaN``/``Inf`` 时返回 ``(zeros(3), zeros(3), False)``。
    """
    target = np.asarray(v_target, dtype=np.float64)
    actual = np.asarray(actual_vel, dtype=np.float64).copy()
    prev = np.asarray(prev_cmd_vel, dtype=np.float64).copy()
    if (target.shape != (3,) or actual.shape != (3,) or prev.shape != (3,)
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(prev))):
        return np.zeros(3), np.zeros(3), False
    scale = float(np.clip(residual_scale, 0.0, 1.0))
    rev_cmd_mask = (target * prev) < 0.0
    any_reversing = bool(np.any(rev_cmd_mask) or np.dot(target, prev) < 0.0)
    if not enabled:
        return actual, prev, any_reversing
    rev_act_mask = (target * actual) < 0.0
    if np.any(rev_act_mask):
        actual[rev_act_mask] *= scale
    if np.any(rev_cmd_mask):
        # 制动段将加速度限制基准向零温和收缩，避免把已衰减的 actual 重新拉回满速旧值
        prev_brake_scale = 0.5 * (1.0 + scale)
        prev[rev_cmd_mask] *= prev_brake_scale
    return actual, prev, any_reversing


class DelayedTwistCompensator:
    """延迟+低通相位匹配的眼在手上（Eye-in-Hand）相机自运动补偿器。

    控制物理意义：
        实机闭环中，指令关节速度经 200Hz 斜坡、MIT 电机跟踪、相机曝光/USB 传输与
        MediaPipe/OneEuro 产生约 25~100ms 纯传输滞后（使用 200Hz 实测 ``dq_act`` 时缩至 ~25ms），
        再经卡尔曼滤波器速度状态引入约 80~90ms 一阶观测群延迟。若直接拿当前时刻的相机 twist
        去抵消滞后的 KF 速度，在 1.5~2Hz 摆动下会出现超过 120° 的相位反转，反而将静止手掌的
        自运动残影放大近一倍。

    算法架构：
        利用线性时不变系统“先低通再延迟 == 先延迟再低通”的交换律：
        1. ``push()``：每周期先将 6D 相机 twist 经过时间常数为 ``lpf_tau_s`` 的一阶低通滤波
           （匹配视觉 KF 速度滞后）并写入带时间戳的环形缓冲；
        2. ``compensate(near_stop=False)``（巡航段）：严格在 ``now - delay_s`` 单点线性插值查表，
           相位精准对齐，不会误杀匀速移动的人手；
        3. ``compensate(near_stop=True)``（停靠/低速段）：在 ``delay_s ± local_window_s`` 窄局部窗内
           取 5 个采样点并选取补偿残差范数最小者，兼容高伸展态下 MIT 延迟略微漂移的情况。
    """

    def __init__(self, delay_s=0.10, lpf_tau_s=0.08, local_window_s=0.04,
                 max_history_s=0.50):
        """初始化延迟+低通相机自运动补偿器。

        Args:
            delay_s (float): 相机 twist 到视觉速度观测的标称纯延迟时间（秒，s，非负）。
            lpf_tau_s (float): 匹配 KF 速度状态群延迟的一阶低通时间常数（秒，s，非负；``<= 1e-4`` 表示直通）。
            local_window_s (float): 停靠段局部容错搜索半宽（秒，s，非负）。
            max_history_s (float): 历史缓冲保留时长（秒，s），内部自动保证不小于
                ``delay_s + local_window_s + 0.15``。

        Raises:
            ValueError: 任一时间参数为负或非有限值时抛出。
        """
        self.delay_s = float(delay_s)
        self.lpf_tau_s = float(lpf_tau_s)
        self.local_window_s = float(local_window_s)
        if (not math.isfinite(self.delay_s) or self.delay_s < 0.0
                or not math.isfinite(self.lpf_tau_s) or self.lpf_tau_s < 0.0
                or not math.isfinite(self.local_window_s)
                or self.local_window_s < 0.0):
            raise ValueError("自运动延迟/低通/窗口参数必须为有限非负数")
        self.max_history_s = max(
            float(max_history_s), self.delay_s + self.local_window_s + 0.15)
        self._lpf = np.zeros(6, dtype=np.float64)
        self._buf = []   # [(t_s, lpf_twist_6d), ...]
        self._last_t = None

    def reset(self):
        """清零低通滤波状态并清空历史 twist 缓冲区。"""
        self._lpf[:] = 0.0
        self._buf.clear()
        self._last_t = None

    def push(self, camera_twist, now, dt):
        """将当前控制周期的相机系 6D twist 滤波并压入带时间戳的历史缓冲。

        Args:
            camera_twist (array_like): 当前周期相机系 6D twist ``[vx, vy, vz, wx, wy, wz]``（m/s, rad/s）。
            now (float): 当前单调时钟时刻（秒，s）。
            dt (float): 本周期时间步长（秒，s）。

        Returns:
            np.ndarray: 形状 ``(6,)`` 的滤波后相机系 twist 副本（m/s, rad/s）。

        Fail-safe 行为：
            非法 twist 替换为零向量；时间戳大幅回退（``> 0.5s``）触发 ``reset()``，
            微小倒退/重复时间戳自动前推 ``max(step, 1e-4)`` 以维护缓冲单调有序。
        """
        try:
            twist = np.asarray(camera_twist, dtype=np.float64)
            t_now = float(now)
            step = float(dt)
        except (TypeError, ValueError):
            twist = np.zeros(6, dtype=np.float64)
            t_now = 0.0 if self._last_t is None else self._last_t + 0.01
            step = 0.01
        if twist.shape != (6,) or not np.all(np.isfinite(twist)):
            twist = np.zeros(6, dtype=np.float64)
        if not math.isfinite(t_now):
            t_now = 0.0 if self._last_t is None else self._last_t + 0.01
        if not math.isfinite(step) or step <= 0.0:
            step = 0.01
        # 时间基准发生大幅跳退（例如仿真/复位重启）时清空历史；否则强制严格递增
        if self._last_t is not None and t_now < self._last_t - 0.5:
            self.reset()
        elif self._last_t is not None and t_now <= self._last_t:
            t_now = self._last_t + max(step, 1e-4)

        if self.lpf_tau_s <= 1e-4:
            self._lpf = twist.copy()
        else:
            alpha = 1.0 - math.exp(-step / self.lpf_tau_s)
            self._lpf += (twist - self._lpf) * alpha

        self._buf.append((t_now, self._lpf.copy()))
        self._last_t = t_now
        # 裁剪早于 max_history_s 的过期记录，始终保留至少 2 个边界点便于插值
        cutoff = t_now - self.max_history_s
        while len(self._buf) > 2 and self._buf[1][0] < cutoff:
            self._buf.pop(0)
        return self._lpf.copy()

    def sample(self, target_time):
        """在指定历史时刻 ``target_time`` 线性插值采样滤波后的 6D 相机 twist。

        Args:
            target_time (float): 查询单调时钟时刻（秒，s）。

        Returns:
            np.ndarray: 形状 ``(6,)`` 的插值相机 twist（m/s, rad/s）。

        Fail-safe 行为：
            缓冲为空、``target_time`` 非有限或早于缓冲起点时返回零向量（机械臂启动前静止）；
            晚于最新记录时钳位返回最新样本。
        """
        if not self._buf:
            return np.zeros(6, dtype=np.float64)
        t_q = float(target_time)
        if not math.isfinite(t_q):
            return np.zeros(6, dtype=np.float64)
        # 早于复位起点时机械臂处于静止，返回零 twist 而非首帧值
        if t_q < self._buf[0][0]:
            return np.zeros(6, dtype=np.float64)
        if t_q >= self._buf[-1][0]:
            return self._buf[-1][1].copy()
        # 缓冲区很小（~40-50 点），线性倒序查找即可（查询多靠近尾部）
        for idx in range(len(self._buf) - 1, 0, -1):
            t1, v1 = self._buf[idx]
            t0, v0 = self._buf[idx - 1]
            if t0 <= t_q <= t1:
                span = t1 - t0
                if span <= 1e-9:
                    return v1.copy()
                ratio = (t_q - t0) / span
                return v0 + (v1 - v0) * ratio
        return np.zeros(6, dtype=np.float64)

    def compensate(self, observed_velocity, palm_position_camera, now=None,
                   near_stop=False):
        """使用相位对齐的历史相机 twist 扣除自运动，输出相机系表达的手掌真实速度。

        Args:
            observed_velocity (array_like): 视觉/KF 观测到的相机系手掌速度 ``(3,)``（m/s）。
            palm_position_camera (array_like): 相机系当前手掌位置 ``(3,)``（m）。
            now (float, optional): 查询基准单调时刻（秒，s）；为 ``None`` 或非有限时取 ``self._last_t``。
            near_stop (bool, optional): 是否处于停靠/低速段；为 ``True`` 且 ``local_window_s > 1e-4`` 时
                启用 5 点窄窗最小残差搜索，否则严格在 ``t_ref - delay_s`` 单点补偿。

        Returns:
            np.ndarray: 形状 ``(3,)`` 的自运动补偿后速度（m/s）。

        Fail-safe 行为：
            ``observed_velocity`` 无效返回零向量；缓冲为空时原样返回 ``observed.copy()``。
        """
        try:
            observed = np.asarray(observed_velocity, dtype=np.float64)
        except (TypeError, ValueError):
            return np.zeros(3, dtype=np.float64)
        if observed.shape != (3,) or not np.all(np.isfinite(observed)):
            return np.zeros(3, dtype=np.float64)
        if not self._buf:
            return observed.copy()
        t_ref = self._last_t if now is None else float(now)
        if not math.isfinite(t_ref):
            t_ref = self._last_t

        if not near_stop or self.local_window_s <= 1e-4:
            matched_twist = self.sample(t_ref - self.delay_s)
            return compensate_palm_velocity(
                observed, matched_twist, palm_position_camera)

        # 停靠/低速段在标称延迟 ± local_window_s 的窄窗口内取 5 个采样点容错
        offsets = (-self.local_window_s, -0.5 * self.local_window_s, 0.0,
                   0.5 * self.local_window_s, self.local_window_s)
        best_vec = None
        best_norm = float("inf")
        for off in offsets:
            d = max(0.0, self.delay_s + off)
            twist = self.sample(t_ref - d)
            candidate = compensate_palm_velocity(
                observed, twist, palm_position_camera)
            norm = float(np.linalg.norm(candidate))
            if norm < best_norm:
                best_norm = norm
                best_vec = candidate
        return best_vec if best_vec is not None else observed.copy()


class PostBrakeSettleGuard:
    """刹车停稳后的物理回弹冷静窗守卫（纯时间制，无硬编码 tick 计数）。

    控制物理意义：
        当指令关节速度降到 ``<= speed_thresh`` 时，真实 MIT 机械臂仍存在约 150~220ms 的
        残余滑移与前伸悬臂弹性回弹（伸展度越高回弹越长，按 ``base_settle_s + ext_extra_s * ext``
        延长）。冷静窗内本守卫提供三项协同保护：
        1. 将静止解锁速度门槛临时抬高 ``override_speed_mult`` 倍，屏蔽低中幅值回弹伪速度，
           杜绝刹车后约 70ms 处的周期性自解锁；同时保留高门槛直通能力，真实大幅挥手仍立即响应；
        2. 同步抬高快速起步位移门槛至 ``max(fast_start_dist * 1.5, override_min_dist)``；
        3. 在冷静窗经过 ``kf_reanchor_ratio``（默认 65%）后一次性触发 ``consume_kf_reanchor()``，
           通知上层清理 KF 残余速度并重锚静止原点。
    """

    def __init__(self, base_settle_s=0.22, ext_extra_s=0.08,
                 kf_reanchor_ratio=0.65, override_speed_mult=2.2,
                 override_min_dist=0.050, speed_thresh=0.05):
        """初始化刹车冷静窗守卫。

        Args:
            base_settle_s (float): 近/中区基础冷静窗时长（秒，s）。
            ext_extra_s (float): 满伸展态（``ext_ratio = 1``）叠加的额外冷静时长（秒，s）。
            kf_reanchor_ratio (float): 触发一次性 KF 残速重锚的冷静窗进度比例，钳制至 ``[0.1, 1.0]``。
            override_speed_mult (float): 冷静窗内解锁速度门槛的放大倍率（无量纲，最小 ``1.0``）。
            override_min_dist (float): 冷静窗内快速起步位移门槛的硬下限（米，m）。
            speed_thresh (float): 判定机械臂“仍在运动”的关节速度阈值（rad/s）。
        """
        self.base_settle_s = max(0.0, float(base_settle_s))
        self.ext_extra_s = max(0.0, float(ext_extra_s))
        self.kf_reanchor_ratio = float(np.clip(kf_reanchor_ratio, 0.1, 1.0))
        self.override_speed_mult = max(1.0, float(override_speed_mult))
        self.override_min_dist = max(0.0, float(override_min_dist))
        self.speed_thresh = max(0.0, float(speed_thresh))
        self.time_since_moving = float("inf")
        self._reanchored_this_stop = True

    def reset(self):
        """复位冷静窗为“已过冷静期”状态，保证启动或恢复追踪后的首次起步零延迟。"""
        # 初始/重新启动时不在冷静期，保证首次起步无延迟
        self.time_since_moving = float("inf")
        self._reanchored_this_stop = True

    def update(self, dt, joint_speed):
        """按当前周期步长与关节速度更新距离上次运动的静默计时。

        Args:
            dt (float): 本周期时间步长（秒，s）；非有限或 ``<= 0`` 按 ``0.0`` 处理。
            joint_speed (float): 当前机械臂关节速度标量（通常为 ``max(|dq|)``，rad/s）；非有限按 ``0.0`` 处理。
        """
        dt = float(dt) if (math.isfinite(dt) and dt > 0.0) else 0.0
        speed = float(joint_speed) if math.isfinite(joint_speed) else 0.0
        if speed > self.speed_thresh:
            # 机械臂仍在运动：清零静默计时并武装下一次停稳后的 KF 重锚事件
            self.time_since_moving = 0.0
            self._reanchored_this_stop = False
        elif math.isfinite(self.time_since_moving):
            self.time_since_moving += dt

    def settle_duration(self, ext_ratio=0.0):
        """计算当前伸展度对应的有效冷静窗总时长（秒）。

        Args:
            ext_ratio (float, optional): 归一化伸展度，位于 ``[0.0, 1.0]``（非有限值按 ``0.0``）。

        Returns:
            float: 冷静窗时长 ``base_settle_s + ext_extra_s * ext_ratio``（秒，s）。
        """
        ext = float(np.clip(ext_ratio, 0.0, 1.0)) if math.isfinite(ext_ratio) else 0.0
        return self.base_settle_s + self.ext_extra_s * ext

    def in_cooldown(self, ext_ratio=0.0):
        """判断当前是否仍处于刹车物理回弹冷静窗内。

        Args:
            ext_ratio (float, optional): 归一化伸展度 ``[0.0, 1.0]``。

        Returns:
            bool: 若冷静窗时长为正且 ``time_since_moving < settle_duration`` 返回 ``True``。
        """
        dur = self.settle_duration(ext_ratio)
        return dur > 0.0 and self.time_since_moving < dur

    def consume_kf_reanchor(self, ext_ratio=0.0):
        """消费一次性事件：刹车尾段过去约 ``kf_reanchor_ratio``（默认 65%）冷静窗时返回 ``True``。

        每次完整停靠周期至多触发一次，通知上层清除 KF 残余速度并重锚静止参考点。

        Args:
            ext_ratio (float, optional): 归一化伸展度 ``[0.0, 1.0]``。

        Returns:
            bool: 本周期是否应当执行一次性 KF 残速清理与重锚。
        """
        dur = self.settle_duration(ext_ratio)
        if dur <= 0.0 or self._reanchored_this_stop:
            return False
        if self.time_since_moving >= dur * self.kf_reanchor_ratio:
            self._reanchored_this_stop = True
            return True
        return False

    def effective_unlock_threshold(self, deadband_on, ext_ratio=0.0):
        """计算当前生效的静止解锁速度门槛（冷静窗内放大 ``override_speed_mult`` 倍）。

        Args:
            deadband_on (float): 标称静止解锁速度阈值（m/s）。
            ext_ratio (float, optional): 归一化伸展度 ``[0.0, 1.0]``。

        Returns:
            float: 当前生效的解锁速度阈值（m/s）。
        """
        base = max(0.0, float(deadband_on))
        if self.in_cooldown(ext_ratio):
            return base * self.override_speed_mult
        return base

    def effective_fast_start_dist(self, fast_start_dist, ext_ratio=0.0):
        """计算当前生效的快速起步位移门槛（冷静窗内抬高防回弹误触发）。

        Args:
            fast_start_dist (float): 标称快速起步位移阈值（m）。
            ext_ratio (float, optional): 归一化伸展度 ``[0.0, 1.0]``。

        Returns:
            float: 冷静窗内返回 ``max(fast_start_dist * 1.5, override_min_dist)``，窗外返回标称值（m）。
        """
        base = max(0.0, float(fast_start_dist))
        if self.in_cooldown(ext_ratio):
            return max(base * 1.5, self.override_min_dist)
        return base


def compute_reach_schedule(reach_r, pos_err_norm, soft_start=0.24,
                           soft_end=0.38, min_scale=0.48, dock_dist=0.25,
                           ff_min_scale=0.50, lead_min_scale=0.30,
                           accel_min_scale=0.65, deadband_on_boost=0.35,
                           fast_start_boost=0.40):
    """基于水平伸展半径的平滑稳定性调度（近/中区全为 1.0，高伸展区自动临界阻尼化）。

    控制物理意义：
        机械臂水平前伸时，J1/J2/J3 末端等效转动惯量 ``I ∝ r^2`` 迅速增大、机械刚度下降且
        MIT 电机位置跟踪滞后增加。本函数按水平半径 ``reach_r`` 经 C1 smoothstep 计算归一化
        伸展度 ``ext_s ∈ [0, 1]``，再结合当前位置误差 ``pos_err_norm`` 的停靠接近因子
        ``dock = clip(1 - pos_err_norm / dock_dist, 0, 1)``：
        - 大误差远距追赶时 ``dock ≈ 0``，``vel_att_target ≈ 1.0``，完整保留高速追手能力；
        - 临近停靠时 ``vel_att_target`` 随 ``ext_s * dock`` 平滑下探至 ``min_scale``（上层配合
          非对称 ``reach_release_tau`` 慢松刹车），同时同步收缩前馈、lead、最大加速度并适度
          抬高防误唤醒门槛，抑制高伸展悬臂弹性回弹引发的往复晃动。

    Args:
        reach_r (float): 当前 TCP 水平径向伸展距离 ``hypot(x, y)``（m）。
        pos_err_norm (float): 当前 3D 位置跟踪误差范数（m）。
        soft_start (float): 开始介入伸展调度的水平半径下限（m）。
        soft_end (float): 达到满伸展调度（``ext_ratio = 1.0``）的水平半径上限（m）。
        min_scale (float): 满伸展近距停靠时的最低速度衰减倍率（钳制至 ``[0.1, 1.0]``）。
        dock_dist (float): 速度衰减开始随距离介入的停靠作用半径（m）。
        ff_min_scale (float): 满伸展时的前馈最小保留倍率（钳制至 ``[0.1, 1.0]``）。
        lead_min_scale (float): 满伸展时的额外 lead 最小保留倍率（钳制至 ``[0.0, 1.0]``）。
        accel_min_scale (float): 满伸展时的最大加速度最小保留倍率（钳制至 ``[0.2, 1.0]``）。
        deadband_on_boost (float): 满伸展时静止解锁速度门槛的附加增幅（如 ``0.35`` 表示 ``1.35x``）。
        fast_start_boost (float): 满伸展时快速起步位移门槛的附加增幅（如 ``0.40`` 表示 ``1.40x``）。

    Returns:
        dict[str, float]: 包含以下键值的调度字典：
            - ``ext_ratio``: C1 平滑后的伸展度 ``[0.0, 1.0]``；
            - ``vel_att_target``: 目标速度衰减倍率 ``[min_scale, 1.0]``；
            - ``ff_scale`` / ``lead_scale`` / ``accel_scale``: 前馈/lead/加速度缩放倍率；
            - ``deadband_on_scale`` / ``fast_start_dist_scale``: 解锁速度与快启位移门槛倍率（``>= 1.0``）。

    Fail-safe 行为：
        任一核心标量（``reach_r``, ``pos_err_norm``, ``soft_start``, ``soft_end``）非有限时，
        退化返回全中性字典（``ext_ratio=0.0``，其余倍率均为 ``1.0``）。
    """
    if (not math.isfinite(reach_r) or not math.isfinite(pos_err_norm)
            or not math.isfinite(soft_start) or not math.isfinite(soft_end)):
        return {
            "ext_ratio": 0.0, "vel_att_target": 1.0, "ff_scale": 1.0,
            "lead_scale": 1.0, "accel_scale": 1.0,
            "deadband_on_scale": 1.0, "fast_start_dist_scale": 1.0,
        }
    span = max(float(soft_end) - float(soft_start), 1e-6)
    ext_linear = float(np.clip((float(reach_r) - float(soft_start)) / span,
                               0.0, 1.0))
    ext_s = _smoothstep(ext_linear)
    # 停靠接近因子：大误差远距追赶时 dock=0（不降速），误差趋近 0 时 dock->1
    dock = float(np.clip(1.0 - max(0.0, float(pos_err_norm))
                         / max(float(dock_dist), 1e-6), 0.0, 1.0))
    min_s = float(np.clip(min_scale, 0.1, 1.0))
    vel_att_target = 1.0 - (1.0 - min_s) * ext_s * dock
    ff_s = 1.0 - (1.0 - float(np.clip(ff_min_scale, 0.1, 1.0))) * ext_s
    lead_s = 1.0 - (1.0 - float(np.clip(lead_min_scale, 0.0, 1.0))) * ext_s
    accel_s = 1.0 - (1.0 - float(np.clip(accel_min_scale, 0.2, 1.0))) * ext_s
    db_scale = 1.0 + max(0.0, float(deadband_on_boost)) * ext_s
    fs_scale = 1.0 + max(0.0, float(fast_start_boost)) * ext_s
    return {
        "ext_ratio": ext_s,
        "vel_att_target": float(vel_att_target),
        "ff_scale": float(ff_s),
        "lead_scale": float(lead_s),
        "accel_scale": float(accel_s),
        "deadband_on_scale": float(db_scale),
        "fast_start_dist_scale": float(fs_scale),
    }


class OscillationDamper:
    """分轴闭环自激振荡检测与阻尼抑制器（专杀 1.2~2.5Hz 极限环，不伤正常快速跟随）。

    控制物理意义与判据：
        对 X/Y/Z 三轴独立跟踪位置误差过零反转的半周期 ``half_p = t_now - last_flip_t``：
        - 正常单程高速追赶（0 次换向）、单次折返（1~2 次换向 < ``confirm_reversals``）、
          大误差主动追手（``|err| >= large_err_release``）或正常人类慢挥手（``half_p > max_half_s``）
          均不会触发，``levels`` 恒保持 ``0.0``，实现零巡航速度损耗；
        - 一旦某轴在中小误差区间 ``[err_thresh, large_err_release)`` 内连续发生
          ``>= confirm_reversals`` 次半周期落于机械臂 MIT 共振带 ``[min_half_s, max_half_s]``
          （对应频率 ``1 / (2 * half_p) ≈ 1.19 ~ 3.57 Hz``）的符号反转，该轴阻尼等级
          ``levels[ax]`` 立即跃升 ``+0.70``（上限 ``1.0``），供上层即时撤除该轴前馈与 lead、
          封锁 Q 放大并削减比例增益，在 1 个半周期内压死自激极限环。
    """

    def __init__(self, err_thresh=0.018, min_half_s=0.14, max_half_s=0.42,
                 decay_tau_s=0.55, confirm_reversals=3,
                 large_err_release=0.15, enabled=True):
        """初始化分轴闭环自激振荡阻尼器。

        Args:
            err_thresh (float): 参与反转检测的最小单轴误差幅值门槛（m，屏蔽传感器底噪）。
            min_half_s (float): 判定为机械臂共振极限环的最短半周期下限（秒，s）。
            max_half_s (float): 判定为机械臂共振极限环的最长半周期上限（秒，s，须 ``> min_half_s``）。
            decay_tau_s (float): 振荡等级自然指数衰减的时间常数（秒，s）。
            confirm_reversals (int): 触发振荡阻尼所需的连续窗内反转次数（须 ``>= 2``）。
            large_err_release (float): 大误差快速释放阈值（m，须 ``> err_thresh``）；
                超过该误差视为人手主动大幅移动，立即清零连击并以 4 倍速泄放残余阻尼。
            enabled (bool): 功能使能开关，默认 ``True``。

        Raises:
            ValueError: 任一阈值、周期范围、确认次数或衰减常数不合法时抛出。
        """
        self.err_thresh = float(err_thresh)
        self.min_half_s = float(min_half_s)
        self.max_half_s = float(max_half_s)
        self.decay_tau_s = float(decay_tau_s)
        self.confirm_reversals = int(confirm_reversals)
        self.large_err_release = float(large_err_release)
        self.enabled = bool(enabled)
        if (not math.isfinite(self.err_thresh) or self.err_thresh <= 0.0
                or not math.isfinite(self.min_half_s) or self.min_half_s <= 0.0
                or not math.isfinite(self.max_half_s)
                or self.max_half_s <= self.min_half_s
                or not math.isfinite(self.decay_tau_s)
                or self.decay_tau_s <= 0.0
                or self.confirm_reversals < 2
                or not math.isfinite(self.large_err_release)
                or self.large_err_release <= self.err_thresh):
            raise ValueError("振荡阻尼器阈值/周期/确认次数/衰减常数不合法")
        self.last_sign = np.zeros(3, dtype=int)
        self.last_flip_t = np.full(3, -1e9, dtype=np.float64)
        self.streak = np.zeros(3, dtype=int)
        self.levels = np.zeros(3, dtype=np.float64)

    def reset(self):
        """彻底清空三轴的反转符号记忆、时间戳、连击次数与阻尼等级。"""
        self.last_sign[:] = 0
        self.last_flip_t[:] = -1e9
        self.streak[:] = 0
        self.levels[:] = 0.0

    def update(self, error_vec, now, dt, active=True, hard_reset=False):
        """按当前周期的 3D 跟踪误差更新三轴振荡阻尼等级。

        Args:
            error_vec (array_like): 3D 位置跟踪误差向量 ``(3,)``（m）。
            now (float): 当前单调时钟时刻（秒，s）。
            dt (float): 本周期时间步长（秒，s）。
            active (bool, optional): 当前是否处于活跃视觉伺服追踪态；``False``（如短暂静止/深度冻结）
                时仅指数衰减 ``levels`` 而不清空 ``streak`` 连击记忆，防止短暂回弹误解锁躲过检测。
            hard_reset (bool, optional): 为 ``True``（如目标彻底丢失/模式切换）时强制 ``reset()``。

        Returns:
            np.ndarray: 形状 ``(3,)`` 的当前分轴振荡阻尼等级副本，各分量位于 ``[0.0, 1.0]``。

        Fail-safe 行为：
            未启用、``hard_reset=True`` 或输入包含非法类型/形状/``NaN``/``Inf`` 时自动复位并返回全零。
        """
        if not self.enabled or hard_reset:
            self.reset()
            return self.levels.copy()
        if not active:
            # 短暂停手/锁存/深度冻结期间只做平滑衰减而不清零连击记忆：
            # 避免 100ms 短暂回弹误解锁后连击归零、被迫多晃 3~4 个半周期。
            # 若静止持续超过 max_half_s * 1.4，下一次激活时 half_p 超窗会自动清零。
            try:
                step = max(float(dt), 0.0)
            except (TypeError, ValueError):
                step = 0.0
            if math.isfinite(step) and step > 0.0:
                self.levels *= math.exp(-step / self.decay_tau_s)
                self.levels[self.levels < 0.02] = 0.0
            return self.levels.copy()
        try:
            err = np.asarray(error_vec, dtype=np.float64)
            t_now = float(now)
            step = float(dt)
        except (TypeError, ValueError):
            self.reset()
            return self.levels.copy()
        if (err.shape != (3,) or not np.all(np.isfinite(err))
                or not math.isfinite(t_now) or not math.isfinite(step)):
            self.reset()
            return self.levels.copy()

        step_pos = max(step, 0.0)
        decay = math.exp(-step_pos / self.decay_tau_s)
        fast_decay = math.exp(-step_pos / max(self.decay_tau_s * 0.25, 0.05))
        for ax in range(3):
            v = float(err[ax])
            if abs(v) >= self.large_err_release:
                # 误差拉开到大追赶区间，说明是人手主动大幅移动而非原地极限环：
                # 清零反转连击并以 4 倍速释放残留阻尼，避免拖慢主动追手。
                self.levels[ax] *= fast_decay
                self.streak[ax] = 0
                sign = 1 if v > 0.0 else -1
                if self.last_sign[ax] != sign:
                    self.last_sign[ax] = sign
                    self.last_flip_t[ax] = t_now
                continue
            self.levels[ax] *= decay
            # 小于噪声门槛的微小误差不更新符号，避免死区底噪伪造反转
            if abs(v) < self.err_thresh:
                continue
            sign = 1 if v > 0.0 else -1
            if self.last_sign[ax] == 0:
                self.last_sign[ax] = sign
                self.last_flip_t[ax] = t_now
            elif sign != self.last_sign[ax]:
                half_p = t_now - self.last_flip_t[ax]
                if self.min_half_s <= half_p <= self.max_half_s:
                    # 反转间隔命中 MIT 共振频带：累积连击，达标后立即拉起阻尼等级
                    self.streak[ax] += 1
                    if self.streak[ax] >= self.confirm_reversals:
                        self.levels[ax] = min(1.0, self.levels[ax] + 0.70)
                else:
                    self.streak[ax] = 0
                self.last_sign[ax] = sign
                self.last_flip_t[ax] = t_now
            elif t_now - self.last_flip_t[ax] > self.max_half_s * 1.4:
                # 长时间维持单边符号未再反转：退出振荡嫌疑，清零连击计数
                self.streak[ax] = 0
        # 截断极小尾部残值，避免极弱阻尼长期残留
        self.levels[self.levels < 0.02] = 0.0
        return self.levels.copy()


def _so3_log(R):
    """SO(3) 李群对数映射：将 ``3×3`` 旋转矩阵映射到 ``so(3)`` 3D 轴角向量（rad）。

    数值分段策略：
        - 小角度（``theta < 1e-6``）：使用一阶近似 ``0.5 * vee(R - R^T)``，避免 ``0 / 0`` 除零；
        - 近 ``pi`` 奇异（``pi - theta < 1e-3``）：由对称阵 ``(R + I) / 2 = a a^T`` 对角线开方
          提取旋转轴并结合非对角元素符号确定象限，避免 ``sin(theta) -> 0`` 引发数值爆炸；
        - 常规角度：``w = (theta / (2 * sin(theta))) * vee(R - R^T)``。

    Args:
        R (np.ndarray): 形状 ``(3, 3)`` 的正交旋转矩阵（``det(R) = +1``）。

    Returns:
        np.ndarray: 形状 ``(3,)`` 的轴角向量 ``w = theta * a``（弧度，rad）。
    """
    # 旋转角满足 cos(theta) = (tr(R) - 1) / 2，先钳制到 [-1, 1] 防止浮点误差触发 acos 越界
    trace = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(trace)
    if theta < 1e-6:
        return 0.5 * np.array([
            R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64)
    if math.pi - theta < 1e-3:
        # 接近 180° 翻转时 Rodrigues 反对称项趋零，改由 (R + I)/2 还原旋转轴方向
        S = (R + np.eye(3)) * 0.5
        axis = np.array([
            math.sqrt(max(float(S[0, 0]), 0.0)),
            math.sqrt(max(float(S[1, 1]), 0.0)),
            math.sqrt(max(float(S[2, 2]), 0.0))], dtype=np.float64)
        axis[1] = math.copysign(axis[1], float(R[0, 1] + R[1, 0]))
        axis[2] = math.copysign(axis[2], float(R[0, 2] + R[2, 0]))
        n = float(np.linalg.norm(axis))
        return np.zeros(3) if n < 1e-9 else theta * (axis / n)
    scale = theta / (2.0 * math.sin(theta))
    return scale * np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64)


def _so3_exp(w):
    """SO(3) 李代数指数映射：将 3D 轴角向量通过 Rodrigues 公式映射为 ``3×3`` 旋转矩阵。

    公式：
        - 常规角度：``R = I + sin(theta) [a]_× + (1 - cos(theta)) [a]_×^2``，其中 ``a = w / ||w||``；
        - 极小角度（``theta < 1e-8``）：采用二阶泰勒展开 ``I + [w]_× + 0.5 [w]_×^2``，避免除零。

    Args:
        w (np.ndarray): 形状 ``(3,)`` 的轴角向量（弧度，rad）。

    Returns:
        np.ndarray: 形状 ``(3, 3)`` 的旋转矩阵。
    """
    theta = float(np.linalg.norm(w))
    if theta < 1e-8:
        K = np.array([[0.0, -w[2], w[1]],
                      [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]], dtype=np.float64)
        return np.eye(3) + K + 0.5 * (K @ K)
    a = w / theta
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _so3_interp(R0, R1, u):
    """SO(3) 流形上的最短测地线球面插值（等价于单位四元数 SLERP）。

    公式：``R(u) = R0 @ exp(u * log(R0^T @ R1))``，满足 ``R(0) = R0``、``R(1) = R1``
    且全程严格保持正交性与 ``det(R) = +1``，旋转角速度恒定。

    Args:
        R0 (np.ndarray): 起始旋转矩阵 ``(3, 3)``（对应 ``u = 0``）。
        R1 (np.ndarray): 终止旋转矩阵 ``(3, 3)``（对应 ``u = 1``）。
        u (float): 插值系数，自动钳制至 ``[0.0, 1.0]``。

    Returns:
        np.ndarray: 插值后的 ``(3, 3)`` 旋转矩阵副本。
    """
    u = float(np.clip(u, 0.0, 1.0))
    if u <= 0.0:
        return R0.copy()
    if u >= 1.0:
        return R1.copy()
    w = _so3_log(R0.T @ R1)
    return R0 @ _so3_exp(u * w)


class CameraPoseHistory:
    """带时间戳与流形一阶低通的相机世界位姿环形缓冲（L1 世界系 KF + Smith 预估器核心组件）。

    控制物理意义：
        1. ``push()``：在每个 100Hz 控制周期记录相机在基座世界系下的实时位姿 ``(p_raw, R_raw)``
           （v13 启用实测反馈时来自 200Hz 真实 ``q_act`` 正运动学），同时在 ``R^3 × SO(3)`` 上
           维护一阶低通平滑位姿 ``(p_lpf, R_lpf)`` 以匹配机电上升响应；
        2. ``lookup(t_query, use_lpf=True)``：对位置做线性插值、对姿态做 SO(3) 测地插值，
           将相机系 3D 掌心观测对齐至真实成像时刻 ``meas_time - pose_delay_s`` 后投影到基座世界系，
           从几何定义上直接消除眼在手上相机自运动；
        3. ``blended_pose(t_now, delay_s, smith_blend)``：在滞后位姿与当前最新位姿之间插值，
           用于将世界系目标估计反投影回相机系构建 Smith 预估控制误差，既能吸收在途执行延迟，
           又能保持静态重力下垂零稳态偏差。
    """

    def __init__(self, lpf_tau_s=0.04, max_history_s=0.60):
        """初始化相机世界位姿历史缓冲。

        Args:
            lpf_tau_s (float): 位姿低通时间常数（秒，s，非负；``<= 1e-4`` 表示关闭低通）。
            max_history_s (float): 历史滑窗最大保留时长（秒，s，须 ``> 0.05``）。

        Raises:
            ValueError: 参数非有限、``lpf_tau_s < 0`` 或 ``max_history_s <= 0.05`` 时抛出。
        """
        self.lpf_tau_s = float(lpf_tau_s)
        self.max_history_s = float(max_history_s)
        if (not math.isfinite(self.lpf_tau_s) or self.lpf_tau_s < 0.0
                or not math.isfinite(self.max_history_s)
                or self.max_history_s <= 0.05):
            raise ValueError("相机位姿历史缓冲参数不合法")
        self._buf = []   # [(t, p_raw, R_raw, p_lpf, R_lpf), ...]
        self._p_lpf = None
        self._R_lpf = None
        self._last_t = None

    def reset(self):
        """清空全部历史位姿样本与低通滤波内部状态。"""
        self._buf.clear()
        self._p_lpf = None
        self._R_lpf = None
        self._last_t = None

    def has_data(self):
        """返回缓冲区是否至少包含一条有效位姿记录。"""
        return len(self._buf) > 0

    def push(self, now, tcp_pos, tcp_rot, T_tcp_camera, dt=0.01):
        """由 TCP 世界位姿与手眼外参合成相机世界位姿，经低通滤波后写入历史缓冲。

        相机世界位姿合成公式：
            ``p_cam_w = p_tcp + R_tcp @ t_tcp_cam``
            ``R_cam_w = R_tcp @ R_tcp_cam``。

        Args:
            now (float): 当前单调时钟时刻（秒，s）。
            tcp_pos (array_like): 世界系 TCP 位置 ``(3,)``（m）。
            tcp_rot (array_like): 世界系 TCP 旋转矩阵 ``(3, 3)``。
            T_tcp_camera (array_like): 手眼外参齐次矩阵 ``(4, 4)``（平移单位 m）。
            dt (float, optional): 本周期时间步长（秒，s），默认 ``0.01``。

        Returns:
            bool: 写入成功返回 ``True``；若任一输入类型/形状非法或含 ``NaN``/``Inf`` 返回 ``False``
                且不污染现有缓冲。
        """
        try:
            t_now = float(now)
            step = float(dt)
            p_tcp = np.asarray(tcp_pos, dtype=np.float64)
            R_tcp = np.asarray(tcp_rot, dtype=np.float64)
            T_tc = np.asarray(T_tcp_camera, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if (not math.isfinite(t_now) or not math.isfinite(step)
                or p_tcp.shape != (3,) or R_tcp.shape != (3, 3)
                or T_tc.shape != (4, 4)
                or not np.all(np.isfinite(p_tcp))
                or not np.all(np.isfinite(R_tcp))
                or not np.all(np.isfinite(T_tc))):
            return False
        if step <= 0.0:
            step = 0.01
        # 时间戳大幅倒退（>0.5s）视作重置；微小倒退或重复时间戳强制单调推进
        if self._last_t is not None and t_now < self._last_t - 0.5:
            self.reset()
        elif self._last_t is not None and t_now <= self._last_t:
            t_now = self._last_t + max(step, 1e-4)

        p_raw = p_tcp + R_tcp @ T_tc[:3, 3]
        R_raw = R_tcp @ T_tc[:3, :3]
        if self._p_lpf is None or self.lpf_tau_s <= 1e-4:
            self._p_lpf = p_raw.copy()
            self._R_lpf = R_raw.copy()
        else:
            # 位置做欧氏一阶低通，姿态在 SO(3) 测地线上做指数平滑
            alpha = 1.0 - math.exp(-step / self.lpf_tau_s)
            self._p_lpf = self._p_lpf + (p_raw - self._p_lpf) * alpha
            self._R_lpf = _so3_interp(self._R_lpf, R_raw, alpha)

        self._buf.append((t_now, p_raw.copy(), R_raw.copy(),
                          self._p_lpf.copy(), self._R_lpf.copy()))
        self._last_t = t_now
        cutoff = t_now - self.max_history_s
        while len(self._buf) > 2 and self._buf[1][0] < cutoff:
            self._buf.pop(0)
        return True

    def latest(self, use_lpf=False):
        """返回最新一条记录的相机世界位姿副本 ``(p_cam, R_cam)``。

        Args:
            use_lpf (bool, optional): ``True`` 返回低通滤波位姿，``False`` 返回原始位姿（默认）。

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: 位置 ``(3,)``（m）与旋转矩阵 ``(3, 3)``；
                缓冲为空时返回 ``(None, None)``。
        """
        if not self._buf:
            return None, None
        item = self._buf[-1]
        if use_lpf:
            return item[3].copy(), item[4].copy()
        return item[1].copy(), item[2].copy()

    def lookup(self, t_query, use_lpf=True):
        """按目标时刻 ``t_query`` 对相机世界位姿执行线性 + SO(3) 测地插值查表。

        Args:
            t_query (float): 待查询的单调时钟时刻（秒，s）。
            use_lpf (bool, optional): 是否从低通位姿序列查表，默认 ``True``。

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: 插值后的相机世界位置 ``(3,)``（m）与
                旋转矩阵 ``(3, 3)``；缓冲为空返回 ``(None, None)``。

        Fail-safe 行为：
            ``t_query`` 非有限时回退取最新记录；早于/晚于缓冲时间跨度时分别钳位至首/尾样本。
        """
        if not self._buf:
            return None, None
        p_idx, r_idx = (3, 4) if use_lpf else (1, 2)
        tq = float(t_query) if math.isfinite(float(t_query)) else self._buf[-1][0]
        if tq <= self._buf[0][0]:
            return self._buf[0][p_idx].copy(), self._buf[0][r_idx].copy()
        if tq >= self._buf[-1][0]:
            return self._buf[-1][p_idx].copy(), self._buf[-1][r_idx].copy()
        # 倒序扫描：视觉查表时刻通常靠近队列尾部，平均只需数次比较
        for i in range(len(self._buf) - 1, 0, -1):
            t1 = self._buf[i][0]
            t0 = self._buf[i - 1][0]
            if t0 <= tq <= t1:
                u = (tq - t0) / max(t1 - t0, 1e-9)
                p = self._buf[i - 1][p_idx] + (
                    self._buf[i][p_idx] - self._buf[i - 1][p_idx]) * u
                R = _so3_interp(self._buf[i - 1][r_idx], self._buf[i][r_idx], u)
                return p, R
        return self._buf[-1][p_idx].copy(), self._buf[-1][r_idx].copy()

    def blended_pose(self, t_now, delay_s, smith_blend):
        """在滞后低通位姿 ``(t_now - delay_s)`` 与当前最新位姿 ``(t_now)`` 之间按 ``smith_blend`` 插值。

        控制物理意义：
            - ``smith_blend = 0.0``：完全使用滞后位姿，等价于纯视觉反馈误差（无在途指令预估）；
            - ``smith_blend = 1.0``：完全使用当前最新位姿（v13 配合真实 200Hz ``q_act`` 时默认），
              等价于完整 Smith 预估器，实时吸收在途位移并消除过冲；
            - 中间值在两者之间做线性 + SO(3) 测地融合。

        Args:
            t_now (float): 当前控制时刻（秒，s）。
            delay_s (float): 滞后参考时间偏移（秒，s）。
            smith_blend (float): 融合系数，钳制至 ``[0.0, 1.0]``（非有限值回退为 ``0.0``）。

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: 融合后的相机世界位置 ``(3,)``（m）与
                旋转矩阵 ``(3, 3)``；缓冲为空返回 ``(None, None)``。
        """
        if not self._buf:
            return None, None
        blend = float(np.clip(smith_blend, 0.0, 1.0)) if math.isfinite(smith_blend) else 0.0
        p_now, R_now = self.latest(use_lpf=False)
        if blend >= 1.0 or delay_s <= 1e-4:
            return p_now, R_now
        p_del, R_del = self.lookup(float(t_now) - max(0.0, float(delay_s)), use_lpf=True)
        if blend <= 0.0:
            return p_del, R_del
        p = p_del + (p_now - p_del) * blend
        R = _so3_interp(R_del, R_now, blend)
        return p, R


def compute_orientation_cruise_weight(pos_err_norm, ori_err_norm,
                                      min_weight=0.40, start_err=0.08,
                                      full_err=0.18,
                                      ori_soft_max_rad=math.radians(10.0),
                                      active=True):
    """大误差巡航段的姿态软优先级权重（L1-B：释放被腕部 J4 姿态锁死拖慢的大臂追赶速度）。

    控制物理意义：
        6-DOF 串联机械臂在高速平移追赶时，若姿态 3 行始终与位置 3 行等权（1.0），腕部 J4
        为了实时抵消大臂 J2+J3 俯仰角变化会率先触发关节限速，从而经全臂等比例缩放把整臂速度
        一起拉低。本函数构造姿态行权重 ``w_ori ∈ [min_weight, 1.0]``：
        - 停靠/静止段（``pos_err_norm <= start_err``）或未激活时恒返 ``1.0``（完全刚性保持姿态）；
        - 大误差追赶段（``start_err < pos_err_norm < full_err``）经 C1 smoothstep 平滑下调至
          ``min_weight``，优先保证 3D 位置追赶带宽；
        - 姿态倾角护栏：当姿态误差超过 ``0.5 * ori_soft_max_rad`` 后平滑撤销让权，到达
          ``ori_soft_max_rad`` 时强制拉回 ``1.0``，确保腕部相机绝不因过度偏转丢失视野。

    Args:
        pos_err_norm (float): 当前 3D 位置误差范数（m）。
        ori_err_norm (float): 当前姿态误差轴角范数（rad）。
        min_weight (float, optional): 最大让权时的姿态权重下限（钳制至 ``[0.15, 1.0]``）。
        start_err (float, optional): 开始下调姿态权重的位置误差门槛（m）。
        full_err (float, optional): 达到最大姿态让权的位置误差上限（m）。
        ori_soft_max_rad (float, optional): 姿态偏离软上限（rad），超过即恢复满权重 ``1.0``。
        active (bool, optional): 是否启用巡航姿态让权，默认 ``True``。

    Returns:
        float: 姿态行权重标量，位于 ``[min_weight, 1.0]``；任意非法/非有限输入均保守返回 ``1.0``。
    """
    if not active:
        return 1.0
    try:
        pe = float(pos_err_norm)
        oe = float(ori_err_norm)
        mw = float(min_weight)
        se = float(start_err)
        fe = float(full_err)
        omax = float(ori_soft_max_rad)
    except (TypeError, ValueError):
        return 1.0
    if (not all(math.isfinite(x) for x in (pe, oe, mw, se, fe, omax))
            or mw >= 1.0 or omax <= 1e-4):
        return 1.0
    mw = float(np.clip(mw, 0.15, 1.0))
    if pe <= se or oe >= omax:
        return 1.0
    pos_t = float(np.clip((pe - se) / max(fe - se, 1e-6), 0.0, 1.0))
    pos_s = _smoothstep(pos_t)
    # 姿态偏离超过软上限的一半时，平滑撤销让权，保护视野
    ori_t = float(np.clip((oe - 0.5 * omax) / max(0.5 * omax, 1e-6), 0.0, 1.0))
    ori_guard = 1.0 - _smoothstep(ori_t)
    return float(1.0 - (1.0 - mw) * pos_s * ori_guard)


def compute_weighted_damped_pseudoinverse(J, row_weights, damping=0.01):
    """计算任务空间行加权阻尼最小二乘（Weighted DLS）伪逆矩阵。

    数学推导：
        求解加权阻尼最小二乘优化问题：
        ``min_{dq} || W (J dq - v) ||_2^2 + lambda^2 || dq ||_2^2``，
        其中 ``W = diag(row_weights)``，令 ``J_w = W J``，其解析右伪逆闭式解为：
        ``J_pinv_w = J_w^T (J_w J_w^T + lambda^2 I_m)^{-1} W``，
        满足 ``dq = J_pinv_w @ v``。当 ``row_weights`` 全为 ``1.0`` 时与标准阻尼伪逆
        ``J^T (J J^T + lambda^2 I)^{-1}`` 数学完全恒等。

    Args:
        J (array_like): 形状 ``(m, n)`` 的任务空间雅可比矩阵（典型为 ``(6, 6)``）。
        row_weights (array_like): 长度 ``m`` 的正行权重向量（如 ``[1, 1, 1, w_ori, w_ori, w_ori]``）。
        damping (float, optional): Tikhonov 阻尼因子 ``lambda``（无量纲，默认 ``0.01``，防奇异位形爆速）。

    Returns:
        np.ndarray: 形状 ``(n, m)`` 的行加权阻尼伪逆矩阵。

    Fail-safe 行为：
        - 若 ``J`` 非二维、权重长度不匹配、含非有限值或存在非正权重，返回零矩阵 ``(n, m)``；
        - 若 ``damping`` 非法/非正，回退使用 ``0.01``；
        - 若矩阵求逆遭遇数值奇异（``LinAlgError``），自动将阻尼增大 10 倍重试求逆。
    """
    J_arr = np.asarray(J, dtype=np.float64)
    w = np.asarray(row_weights, dtype=np.float64)
    if (J_arr.ndim != 2 or w.shape != (J_arr.shape[0],)
            or not np.all(np.isfinite(J_arr))
            or not np.all(np.isfinite(w)) or np.any(w <= 0.0)):
        return np.zeros((J_arr.shape[1], J_arr.shape[0]), dtype=np.float64)
    m = J_arr.shape[0]
    # J_w = W @ J，逐行缩放等价于对任务空间各自由度误差残差施加权重 w_i
    J_w = w[:, None] * J_arr
    JJT = J_w @ J_w.T
    d = float(damping) if (math.isfinite(float(damping)) and damping > 0.0) else 0.01
    try:
        inv_term = np.linalg.inv(JJT + (d ** 2) * np.eye(m))
    except np.linalg.LinAlgError:
        # 极端奇异保护：将阻尼系数放大 10 倍以保证矩阵严格正定可逆
        inv_term = np.linalg.inv(JJT + ((d * 10.0) ** 2) * np.eye(m))
    # 右乘对角矩阵 W：等效将列 j 乘以 w[j]
    return (J_w.T @ inv_term) * w[None, :]


def consensus_velocity(v_world_cam, v_twist_cam):
    """双估计器逐轴保守共识速度（世界系 KF 与延迟低通 Twist 补偿器的同向最小幅值交集）。

    控制物理意义与逐轴规则：
        对 X/Y/Z 每一轴独立比较两个异构速度估计器 ``va`` 与 ``vb``：
        - 若同号（``va * vb > 0``）：取 ``sign(va) * min(|va|, |vb|)``；
        - 若异号或任一为零（``va * vb <= 0``）：输出 ``0.0``。
        真实人手移动时两路估计方向一致、幅值相近，可完整保留追手响应与前馈速度；
        而在机械臂急停回弹或 1.7~2.0Hz 欠阻尼共振时，单点查表的世界系 KF 可能漏出
        0.3~0.5 m/s 假速度，带 ``±40ms`` 局部容错窗的 Twist 补偿器已将其压至近零，
        逐轴取保守交集可同时继承两套观测器的抗自运动优势，根除停手二次误解锁。

    Args:
        v_world_cam (array_like): 世界系 KF 速度投影至相机系的结果 ``(3,)``（m/s）。
        v_twist_cam (array_like): 延迟低通 Twist 补偿器输出的相机系速度 ``(3,)``（m/s）。

    Returns:
        np.ndarray: 形状 ``(3,)`` 的逐轴共识速度向量（m/s）。

    Fail-safe 行为：
        任一输入类型错误、形状非 ``(3,)`` 或含 ``NaN``/``Inf`` 时返回零向量 ``np.zeros(3)``。
    """
    try:
        a = np.asarray(v_world_cam, dtype=np.float64)
        b = np.asarray(v_twist_cam, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(3, dtype=np.float64)
    if (a.shape != (3,) or b.shape != (3,)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
        return np.zeros(3, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    for ax in range(3):
        va, vb = float(a[ax]), float(b[ax])
        if va * vb > 0.0:
            out[ax] = math.copysign(min(abs(va), abs(vb)), va)
    return out


def apply_joint_limit_cushion(q_now, joint_vel, q_min, q_max, margin_rad=0.08):
    """关节软限位线性缓冲：在距离 ``q_min`` / ``q_max`` 最后 ``margin_rad`` 内平滑收缩外推速度。

    控制物理意义：
        对每一关节 ``i``，仅当速度朝向限位边界运动（``dq[i] > 0`` 逼近 ``q_max[i]`` 或
        ``dq[i] < 0`` 逼近 ``q_min[i]``）且剩余行程 ``rem < margin_rad`` 时，按线性比例
        ``clip(rem / margin_rad, 0.0, 1.0)`` 衰减 ``dq[i]``（越界 ``rem <= 0`` 时完全归零）。
        朝离开限位方向运动的速度绝不衰减。这能防止 J3/J4 高速顶死底层硬截断
        （例如 ``q4`` 撞 ``+1.40 rad`` 软墙时引发 10° 突变俯仰冲击）。

    Args:
        q_now (array_like): 当前各关节指令角度 ``q_cmd``（rad，一维向量，典型长度 6）。
        joint_vel (array_like): 待缓冲的期望关节角速度 ``dq``（rad/s，与 ``q_now`` 等长）。
        q_min (array_like): 各关节允许下限（rad，与 ``q_now`` 等长）。
        q_max (array_like): 各关节允许上限（rad，与 ``q_now`` 等长）。
        margin_rad (float, optional): 软限位缓冲带宽度（rad，须为有限正数），默认 ``0.08``（约 4.6°）。

    Returns:
        tuple[np.ndarray, bool]: ``(cushioned_dq, cushioned_flag)``：
            - ``cushioned_dq``: 缓冲后的关节角速度副本（rad/s），不修改输入数组；
            - ``cushioned_flag``: 是否有任意关节触发了软限位减速缓冲。

    Fail-safe 行为：
        输入维度/形状不匹配、含 ``NaN``/``Inf`` 或 ``margin_rad <= 0`` 时返回全零速度与 ``False``。
    """
    try:
        q = np.asarray(q_now, dtype=np.float64)
        dq = np.asarray(joint_vel, dtype=np.float64).copy()
        lo = np.asarray(q_min, dtype=np.float64)
        hi = np.asarray(q_max, dtype=np.float64)
        margin = float(margin_rad)
    except (TypeError, ValueError):
        return np.zeros(6, dtype=np.float64), False
    if (q.ndim != 1 or dq.shape != q.shape or lo.shape != q.shape
            or hi.shape != q.shape or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(dq)) or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi)) or not math.isfinite(margin)
            or margin <= 0.0):
        return np.zeros_like(dq) if dq.ndim == 1 else np.zeros(6), False
    cushioned = False
    for i in range(q.size):
        if dq[i] > 0.0:
            # 朝正向限位运动：仅在剩余裕度不足 margin 时按比例线性刹车
            rem = hi[i] - q[i]
            if rem < margin:
                dq[i] *= float(np.clip(rem / margin, 0.0, 1.0))
                cushioned = True
        elif dq[i] < 0.0:
            # 朝负向限位运动：仅在剩余裕度不足 margin 时按比例线性刹车
            rem = q[i] - lo[i]
            if rem < margin:
                dq[i] *= float(np.clip(rem / margin, 0.0, 1.0))
                cushioned = True
    return dq, cushioned


def compute_roi_box(frame_w, frame_h, center_uv, prev_box=None,
                    scale=0.68, min_size=288, margin_ratio=0.26):
    """计算带迟滞死区的方形 ROI 裁剪框 ``(x0, y0, x1, y1)``，用于加速 MediaPipe 手部推理。

    迟滞设计原理：
        - 裁剪边长按 ``short_side * clip(scale, 0.45, 0.90)`` 与 ``min_size`` 取大并向下对齐偶数，
          若边长已接近全图短边（``>= short_side - 8``）则直接返回 ``None`` 走全图推理；
        - 当上一帧 ``prev_box`` 合法且边长一致，且当前掌心 ``(u, v)`` 仍位于 ``prev_box``
          内缩 ``margin = size * clip(margin_ratio, 0.10, 0.40)`` 的安全死区以内时，原样复用
          ``prev_box`` 不做移动，避免每帧微移裁剪框破坏 MediaPipe ``VIDEO`` 模式的跨帧追踪；
        - 掌心接近 ROI 边缘时，重新以掌心居中并将裁剪框钳位在画面 ``[0, w] × [0, h]`` 内。

    Args:
        frame_w (int): 原始 RGB 图像宽度（像素，px）。
        frame_h (int): 原始 RGB 图像高度（像素，px）。
        center_uv (Sequence[float] or None): 上一帧掌心像素中心 ``(u, v)``（px）；``None`` 返回 ``None``。
        prev_box (tuple[int, int, int, int] or None, optional): 上一帧 ROI 框 ``(x0, y0, x1, y1)``（px）。
        scale (float, optional): ROI 边长占图像短边的比例，钳制至 ``[0.45, 0.90]``。
        min_size (int, optional): ROI 最小方形边长（px），避免过度放大导致手掌出框。
        margin_ratio (float, optional): 迟滞安全边距占 ROI 边长的比例，钳制至 ``[0.10, 0.40]``。

    Returns:
        tuple[int, int, int, int] or None: 像素坐标裁剪框 ``(x0, y0, x1, y1)``；
            输入无效或无需裁剪时返回 ``None``（调用方退回全图推理）。
    """
    try:
        w = int(frame_w)
        h = int(frame_h)
        sc = float(scale)
        ms = int(min_size)
        mr = float(margin_ratio)
    except (TypeError, ValueError):
        return None
    if (w <= 0 or h <= 0 or center_uv is None
            or not math.isfinite(sc) or not math.isfinite(mr)):
        return None
    try:
        u, v = float(center_uv[0]), float(center_uv[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(u) and math.isfinite(v)):
        return None

    short_side = min(w, h)
    size = min(short_side, max(ms, int(round(short_side * float(np.clip(sc, 0.45, 0.90))))))
    # 对齐偶数边长，保证半宽划分对称且对硬件/图像缩放更友好
    size = size - (size % 2)
    if size <= 0 or size >= short_side - 8:
        return None

    if prev_box is not None and len(prev_box) == 4:
        px0, py0, px1, py1 = (int(prev_box[0]), int(prev_box[1]),
                              int(prev_box[2]), int(prev_box[3]))
        if (0 <= px0 < px1 <= w and 0 <= py0 < py1 <= h
                and (px1 - px0) == size and (py1 - py0) == size):
            margin = size * float(np.clip(mr, 0.10, 0.40))
            if (px0 + margin <= u <= px1 - margin
                    and py0 + margin <= v <= py1 - margin):
                return (px0, py0, px1, py1)

    half = size // 2
    x0 = int(round(u)) - half
    y0 = int(round(v)) - half
    # 将裁剪框左上角钳位在合法图像范围内，确保框体不越界
    x0 = max(0, min(w - size, x0))
    y0 = max(0, min(h - size, y0))
    return (x0, y0, x0 + size, y0 + size)


def remap_roi_landmarks(landmarks, roi_box, frame_w, frame_h):
    """将 ROI 局部归一化手部关键点精确重映射回全图归一化坐标系。

    映射公式：
        - ``x_full = (x0 + lm.x * roi_w) / frame_w``
        - ``y_full = (y0 + lm.y * roi_h) / frame_h``
        - ``z_full = lm.z * (roi_w / frame_w)``（MediaPipe 的相对深度 ``z`` 以输入图宽为尺度基准，
          因此必须乘以 ``roi_w / frame_w`` 才能与全图推理的深度尺度及捏合距离度量完全一致）。

    Args:
        landmarks (Sequence): MediaPipe 输出的 ROI 局部关键点列表（元素需含 ``.x``, ``.y`` 属性）。
        roi_box (tuple[int, int, int, int] or None): 对应的 ROI 裁剪框 ``(x0, y0, x1, y1)``（px）。
        frame_w (int): 全图宽度（px）。
        frame_h (int): 全图高度（px）。

    Returns:
        list[SimpleNamespace] | Sequence: 重映射后的全图归一化关键点列表；若输入为空、
            ``roi_box is None`` 或任一尺寸 ``<= 0``，原样返回 ``landmarks``。
    """
    if not landmarks or roi_box is None:
        return landmarks
    x0, y0, x1, y1 = roi_box
    roi_w = float(x1 - x0)
    roi_h = float(y1 - y0)
    fw = float(frame_w)
    fh = float(frame_h)
    if roi_w <= 0.0 or roi_h <= 0.0 or fw <= 0.0 or fh <= 0.0:
        return landmarks
    z_scale = roi_w / fw
    return [
        SimpleNamespace(
            x=(x0 + float(lm.x) * roi_w) / fw,
            y=(y0 + float(lm.y) * roi_h) / fh,
            z=float(getattr(lm, "z", 0.0)) * z_scale)
        for lm in landmarks
    ]


class ActualVelocityFilter:
    """200Hz MIT 电机实测关节角速度 ``dq_act`` 的轻量一阶低通滤波器（v13 P2）。

    控制物理意义：
        P0 硬件实测表明，直接读取的电机编码器微分角速度在静止时存在约 ``0.04~0.07 rad/s``
        的量化抖动噪声。本滤波器以小时间常数（默认 ``tau_s = 0.020s``，约 8Hz 截止频率）
        平滑 6 轴实测角速度 ``dq_act``，在滤除静态量化毛刺的同时保持真实物理速度的高相位准确度，
        供 ``DelayedTwistCompensator`` 计算真实眼在手上相机自运动。
        遇到无效反馈或异常数值时自动复位并返回 ``None``，上层据此无缝回退到 ``q_cmd`` 差分速度。
    """

    def __init__(self, tau_s=0.020, max_speed=None):
        """初始化实测关节角速度低通滤波器。

        Args:
            tau_s (float): 一阶低通时间常数（秒，s，非负有限值；``<= 1e-4`` 表示无滤波直通）。
            max_speed (float or array_like, optional): 可选的单轴角速度硬钳制上限（rad/s），
                用于在滤波前抑制异常单帧编码器尖峰。

        Raises:
            ValueError: 当 ``tau_s`` 为负或非有限值时抛出。
        """
        self.tau_s = float(tau_s)
        if not math.isfinite(self.tau_s) or self.tau_s < 0.0:
            raise ValueError("实际角速度滤波时间常数必须为非负有限值")
        self.max_speed = (None if max_speed is None
                          else np.asarray(max_speed, dtype=np.float64).copy())
        self._vel = None

    def reset(self):
        """清除滤波内部状态，下一次有效观测将直接作为无滞后初值。"""
        self._vel = None

    def update(self, dq_act, dt, valid=True):
        """喂入当前周期的 6 轴实测关节角速度并输出平滑结果。

        Args:
            dq_act (array_like or None): 6 轴实测关节角速度 ``(6,)``（rad/s）。
            dt (float): 距离上次更新的时间步长（秒，s，须为有限正数）。
            valid (bool, optional): 200Hz MIT 反馈缓存是否有效新鲜，默认 ``True``。

        Returns:
            np.ndarray or None: 滤波后的 6 轴角速度副本 ``(6,)``（rad/s）；
                输入无效时复位并返回 ``None`` 以触发上层回退逻辑。
        """
        if not valid or dq_act is None:
            self.reset()
            return None
        try:
            dq = np.asarray(dq_act, dtype=np.float64)
            step = float(dt)
        except (TypeError, ValueError):
            self.reset()
            return None
        if (dq.shape != (6,) or not np.all(np.isfinite(dq))
                or not math.isfinite(step) or step <= 0.0):
            self.reset()
            return None
        if self.max_speed is not None:
            dq = np.clip(dq, -self.max_speed, self.max_speed)
        if self._vel is None or self.tau_s <= 1e-4:
            self._vel = dq.copy()
        else:
            alpha = 1.0 - math.exp(-step / self.tau_s)
            self._vel = self._vel + (dq - self._vel) * alpha
        return self._vel.copy()







