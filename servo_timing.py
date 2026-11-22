#!/usr/bin/env python3
"""视觉伺服的时间与状态切换工具（纯计算，无硬件依赖）。"""

import math

import numpy as np


def scaled_alpha(alpha, dt, nominal_dt=0.01):
    """把标称周期的 EMA 系数换算到实际 dt，避免周期抖动改变群延迟。"""
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    dt = max(float(dt), 1e-6)
    return 1.0 - (1.0 - alpha) ** (dt / nominal_dt)


def scaled_decay(decay, dt, nominal_dt=0.01):
    """把标称周期的衰减系数换算到实际 dt。"""
    return float(decay) ** (max(float(dt), 0.0) / nominal_dt)


def measurement_age(now, timestamp, max_age=0.25):
    """单调时钟观测年龄；异常时间戳返回 0，避免负 dt 或无限外推。"""
    age = float(now) - float(timestamp)
    if not np.isfinite(age) or age < 0.0:
        return 0.0
    return min(age, float(max_age))


def horizontal_radial_soft_wall(position, velocity, radius):
    """水平径向软墙：限 hypot(x,y)，不占用/限制 Z 方向工作空间。

    仅删除水平向外的径向速度，保留水平向内、切向及全部竖直速度。
    相比三维球形墙，不会在 TCP 抬高时错误压缩水平可达范围。
    """
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    result = velocity.copy()
    if (position.shape != (3,) or velocity.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))):
        return np.zeros(3), True
    distance = float(np.hypot(position[0], position[1]))
    if distance >= float(radius) and distance > 1e-9:
        radial = np.array([position[0] / distance, position[1] / distance, 0.0])
        outward = float(np.dot(result[:2], radial[:2]))
        if outward > 0.0:
            result[:2] -= outward * radial[:2]
            return result, True
    return result, False


def dynamic_lead_seconds(base_lead_ms, processing_s, max_lead_ms):
    """观测已外推到现在后，再补偿处理延迟和固定执行余量。

    processing_s 只取本帧“取帧完成→目标发布”的处理时间，不包含会随
    重复控制周期增长的帧年龄，避免重复帧时 lead 越算越大。
    """
    if not np.isfinite(processing_s) or processing_s < 0.0:
        processing_s = 0.0
    lead_ms = float(base_lead_ms) + processing_s * 1000.0
    return float(np.clip(lead_ms, 0.0, max_lead_ms)) / 1000.0


def camera_twist_from_tcp(tcp_twist_world, tcp_rotation, T_tcp_camera):
    """把世界系 TCP twist 转成相机系 twist（线速度位于相机光心）。

    tcp_twist_world 的线速度在 TCP 原点，角速度是世界系轴角速度；
    T_tcp_camera 是相机在 TCP 系下的位姿。相机线速度必须加上 TCP
    角速度在相机安装偏移处产生的杠杆速度，否则手眼安装偏置较大时
    自运动补偿会残留横向误差。
    任何无效输入都返回零 twist，补偿退化为原始观测，不影响安全链。
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
    camera_offset_world = R_world_tcp @ t_tcp_camera
    v_camera_world = twist[:3] + np.cross(omega_world, camera_offset_world)
    # R_camera_world = (R_world_tcp @ R_tcp_camera)^T
    R_camera_world = (R_world_tcp @ R_tcp_camera).T
    return np.concatenate((R_camera_world @ v_camera_world,
                           R_camera_world @ omega_world))


def compensate_palm_velocity(observed_velocity, camera_twist_camera,
                             palm_position_camera):
    """去掉相机自身运动后，手掌相对世界的速度（仍用相机系表达）。

    静止点满足 p_dot = -v_camera - omega × p，因此补偿项为
    +v_camera + omega × p。机械臂自身的运动不会再被识别成手掌运动。
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
    """延迟对齐的相机自运动补偿：匹配视觉+KF 链路 30~60ms 群延迟。

    机械臂高速接近静止手掌并进入死区瞬间，当前周期的关节指令速度已降为 0，
    但视觉 KF 输出的 _kv_raw 仍反映 30~60ms 前机械臂冲刺引起的画面相对速度。
    在最近若干控制周期的相机 twist 历史窗口中选取残差范数最小的补偿结果，
    可精确抵消滞后的相机自运动残影，防止停靠瞬间因相位错位无法进入静止锁存。
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
    """相机系点 → 世界系点；无效输入返回零向量。"""
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
    point_tcp = transform[:3, :3] @ point + transform[:3, 3]
    return position + rotation @ point_tcp


def world_point_to_camera(point_world, tcp_position, tcp_rotation,
                          T_tcp_camera):
    """世界系点 → 相机系点；无效输入返回零向量。"""
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
    point_tcp = rotation.T @ (point - position)
    return transform[:3, :3].T @ (point_tcp - transform[:3, 3])


def _smoothstep(value):
    """0~1 的 C1 连续过渡，避免门控阈值处出现指令跳变。"""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class AdaptiveLeadGate:
    """运动中保留 lead，停手/近目标/反向纠偏时自动收回。

    lead 是额外提前量，不包含观测年龄补偿。三个权重相乘：
    速度（残速不可外推）、距离（到位不可继续推）、方向（预测必须指向
    尚未消除的误差）。无效观测直接清零；正常撤权快、恢复权慢，且都
    按真实 dt 计算，避免 0/15ms 在静止边界拍打。
    """

    def __init__(self, speed_off=0.06, speed_on=0.13,
                 near_dist=0.022, full_dist=0.045,
                 direction_full_cos=0.50,
                 attack_tau=0.025, release_tau=0.015):
        self.speed_off = max(0.0, float(speed_off))
        self.speed_on = max(self.speed_off + 1e-6, float(speed_on))
        self.near_dist = max(0.0, float(near_dist))
        self.full_dist = max(self.near_dist + 1e-6, float(full_dist))
        self.direction_full_cos = float(np.clip(direction_full_cos, 0.0, 0.99))
        self.attack_tau = max(1e-3, float(attack_tau))
        self.release_tau = max(1e-3, float(release_tau))
        self.weight = 0.0

    def reset(self):
        self.weight = 0.0

    def update(self, velocity, position_error, dt, moving, valid=True):
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

        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            dt = 0.0
        tau = self.attack_tau if target > self.weight else self.release_tau
        alpha = 1.0 - np.exp(-dt / tau)
        self.weight += (target - self.weight) * alpha
        if self.weight < 1e-4:
            self.weight = 0.0
        return float(self.weight), float(target)


class FastStartGate:
    """静止锁存的快速退出：连续、同向、累计位移超过噪声门槛。

    只在新三维观测上调用。原速度确认通道仍独立保留；本类只是额外的
    快速通道，不改变静止进入条件，也不放宽深度保护。
    """

    def __init__(self, enable=True, frames=2, distance=0.030,
                 direction_cos=0.80):
        self.enable = bool(enable)
        self.frames = max(1, int(frames))
        self.distance = float(distance)
        self.direction_cos = float(direction_cos)
        self.origin = None
        self.prev_delta = None
        self.count = 0

    def reset(self, origin=None):
        self.origin = (None if origin is None else
                       np.asarray(origin, dtype=np.float64).copy())
        self.prev_delta = None
        self.count = 0

    def update(self, point):
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
        self.count = self.count + 1 if norm >= self.distance else 0
        self.prev_delta = delta.copy()
        return self.count >= self.frames


class SlowFollowGate:
    """锁存状态下的慢速、连续、同向拖动确认；不解锁前馈或额外 lead。"""

    def __init__(self, distance=0.014, release_distance=0.008,
                 confirm_frames=3, direction_cos=0.80):
        self.distance = float(distance)
        self.release_distance = float(release_distance)
        self.confirm_frames = max(1, int(confirm_frames))
        self.direction_cos = float(direction_cos)
        self.reset()

    def reset(self, origin=None):
        self.origin = None
        if origin is not None:
            origin = np.asarray(origin, dtype=np.float64)
            if origin.shape == (3,) and np.all(np.isfinite(origin)):
                self.origin = origin.copy()
        self.previous_delta = None
        self.coherent_frames = 0
        self.active = False

    def update(self, point, enabled=True):
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
    """把相机硬件秒时间戳映射到 time.monotonic()；异常帧回退主机到达时刻。"""

    def __init__(self, max_transport_s=0.20, max_raw_step_s=0.50,
                 offset_tau_s=2.0):
        self.max_transport_s = float(max_transport_s)
        self.max_raw_step_s = float(max_raw_step_s)
        self.offset_tau_s = float(offset_tau_s)
        self.offset_s = None
        self.last_raw_s = None
        self.last_host_s = None
        self.rejected = 0
        self.used = 0

    def reset(self):
        self.offset_s = None
        self.last_raw_s = None
        self.last_host_s = None

    def _fallback(self, host_s, raw_s=None, reset_origin=False):
        self.rejected += 1
        if reset_origin:
            self.reset()
        elif raw_s is not None:
            self.last_raw_s = raw_s
            self.last_host_s = host_s
        return host_s, False

    def update(self, raw_timestamp_s, host_arrival_s):
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
            if (raw_step <= 0.0 or raw_step > self.max_raw_step_s
                    or host_step <= 0.0 or host_step > 1.0):
                return self._fallback(host_s, reset_origin=True)
        if self.offset_s is None:
            self.offset_s = host_s - raw_s
        mapped_s = raw_s + self.offset_s
        transport_s = host_s - mapped_s
        # 负传输延迟意味着倒退/异常时钟；过大延迟按陈旧主机帧处理，不送入控制链。
        if transport_s < -0.005 or transport_s > self.max_transport_s:
            return self._fallback(host_s, raw_s=raw_s)
        if self.last_host_s is not None:
            elapsed = max(1e-4, host_s - self.last_host_s)
            alpha = 1.0 - math.exp(-elapsed / max(1e-3, self.offset_tau_s))
            self.offset_s += alpha * transport_s
        self.last_raw_s = raw_s
        self.last_host_s = host_s
        self.used += 1
        return mapped_s, True


class StaticVelocityHold:
    """静止锁存且位于位置死区内时，清除低速残余指令状态。

    高速停靠仍走原有 EMA + 加速度限制；只有线速度已经低于门槛并连续
    满足若干控制周期后才清零。运动、离开死区或禁用时会立即复位确认，
    不会拦截正常起步、慢拖或反向纠偏。返回值不修改调用方原数组。
    """

    def __init__(self, speed_limit=0.025, confirm_frames=3):
        self.speed_limit = float(speed_limit)
        self.confirm_frames = max(1, int(confirm_frames))
        if (not math.isfinite(self.speed_limit) or self.speed_limit <= 0.0):
            raise ValueError("静止残余速度门槛必须为有限正数")
        self.count = 0
        self.active = False

    def reset(self):
        self.count = 0
        self.active = False

    def update(self, velocity, stationary, in_deadband, enabled=True):
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
    """连续同向创新确认的分轴 Q 倍率；默认只增强 XY，Z 可独立封顶。"""

    def __init__(self, ref_xy=0.060, ref_z=0.080, max_xy=1.60, max_z=1.0,
                 confirm_frames=2, attack_tau=0.040, release_tau=0.120):
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
        self.scales[:] = 1.0
        self.counts[:] = 0
        self.signs[:] = 0

    def relax(self, dt):
        dt = max(1e-4, float(dt))
        alpha = 1.0 - math.exp(-dt / self.release_tau)
        self.scales += (1.0 - self.scales) * alpha
        return self.scales.copy()

    def update(self, innovation, dt, valid=True):
        innovation = np.asarray(innovation, dtype=np.float64)
        if (not valid or innovation.shape != (3,)
                or not np.all(np.isfinite(innovation)) or not math.isfinite(dt)):
            self.reset()
            return self.scales.copy()
        magnitude = np.abs(innovation)
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
        attacking = desired > self.scales
        alpha = np.empty(3)
        alpha[attacking] = 1.0 - math.exp(-max(1e-4, dt) / self.attack_tau)
        alpha[~attacking] = 1.0 - math.exp(-max(1e-4, dt) / self.release_tau)
        self.scales += (desired - self.scales) * alpha
        self.scales = np.clip(self.scales, 1.0, self.max_scales)
        return self.scales.copy()


def adaptive_feedforward_scale(error_vector, target_velocity, base_taper_dist,
                               enabled=True, speed_ref=0.35, max_taper_dist=0.12):
    """返回 (前馈比例, 当前停靠距离, 接近速度)；快速接近只更早减前馈。"""
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
    """关节速度等比例限幅：任一关节超限时整体按比例缩放，保持笛卡尔运动方向。

    默认 grouped=False 对全臂 6 轴统一等比例缩放，既严格保持 J @ dq 的三维运动
    方向与手腕姿态协同，又限制斜向合成总动能，防止高速斜向停靠时过冲晃动。
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
    """分轴独立换向惯性衰减：同步收缩 EMA 与加速度限制器过零前的旧方向残速。

    解决两个问题：
    1) 三维整体点积在画圆/斜向变向时，单轴反向会被正交同向轴掩盖，或反过来
       连累正交巡航轴被无辜减速；改为 X/Y/Z 分轴独立判定 `v_target[i]*actual[i] < 0`。
    2) 原先只衰减 actual_vel 而不衰减 prev_cmd_vel，导致后续 slew-rate 限制器
       直接以未衰减的 prev_cmd_vel 覆盖 actual_vel，使 residual_scale 形同虚设。
       现在对换向轴同步将 prev_cmd_vel 在同号制动方向上收缩，既消除 ~100ms
       换向滞后，又保证过零后的反向起步仍严格受加速度上限约束。
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
    """延迟+低通匹配的眼在手上相机自运动补偿器。

    实机闭环中，指令关节速度经 200Hz 斜坡、MIT 电机跟踪、相机曝光/USB 与
    MediaPipe/OneEuro 产生 ~100ms 传输滞后，再经卡尔曼速度状态产生 ~80~90ms
    观测滞后。直接用当前周期的相机 twist 抵消滞后的 KF 速度，在 1.5~2Hz 摆动下
    会出现 >120° 相位反转，把静止手掌的自运动残影反而放大近一倍。

    利用线性时不变系统“先低通再延迟 == 先延迟再低通”的交换律：
    每周期先把相机 twist 过一阶低通（匹配 KF 速度滞后）并写入带时间戳的环形缓冲；
    - 巡航段只取 exact `now - delay_s` 单点查表（相位严格对齐，不误杀匀速移动手掌）；
    - 接近停靠/静止段在 `delay_s ± local_window_s` 窄局部窗口内取最小残差，
      兼容伸展态 MIT 延迟略增的漂移。
    """

    def __init__(self, delay_s=0.10, lpf_tau_s=0.08, local_window_s=0.04,
                 max_history_s=0.50):
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
        self._lpf[:] = 0.0
        self._buf.clear()
        self._last_t = None

    def push(self, camera_twist, now, dt):
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
        cutoff = t_now - self.max_history_s
        while len(self._buf) > 2 and self._buf[1][0] < cutoff:
            self._buf.pop(0)
        return self._lpf.copy()

    def sample(self, target_time):
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
    """刹车停稳后的物理回弹冷静窗（时间制，不依赖硬编码 tick 数）。

    指令关节速度降到 <= speed_thresh 后，真实 MIT 机械臂与视觉/KF 链路仍有
    150~220ms 的残余滑移与伸展悬臂弹性回弹。冷静窗内屏蔽低中幅值假速度，
    防止刹车后精确 70ms 自解锁；同时保留高门槛直通通道，真实大幅挥手仍能
    立即响应。
    """

    def __init__(self, base_settle_s=0.22, ext_extra_s=0.08,
                 kf_reanchor_ratio=0.65, override_speed_mult=2.2,
                 override_min_dist=0.050, speed_thresh=0.05):
        self.base_settle_s = max(0.0, float(base_settle_s))
        self.ext_extra_s = max(0.0, float(ext_extra_s))
        self.kf_reanchor_ratio = float(np.clip(kf_reanchor_ratio, 0.1, 1.0))
        self.override_speed_mult = max(1.0, float(override_speed_mult))
        self.override_min_dist = max(0.0, float(override_min_dist))
        self.speed_thresh = max(0.0, float(speed_thresh))
        self.time_since_moving = float("inf")
        self._reanchored_this_stop = True

    def reset(self):
        # 初始/重新启动时不在冷静期，保证首次起步无延迟
        self.time_since_moving = float("inf")
        self._reanchored_this_stop = True

    def update(self, dt, joint_speed):
        dt = float(dt) if (math.isfinite(dt) and dt > 0.0) else 0.0
        speed = float(joint_speed) if math.isfinite(joint_speed) else 0.0
        if speed > self.speed_thresh:
            self.time_since_moving = 0.0
            self._reanchored_this_stop = False
        elif math.isfinite(self.time_since_moving):
            self.time_since_moving += dt

    def settle_duration(self, ext_ratio=0.0):
        ext = float(np.clip(ext_ratio, 0.0, 1.0)) if math.isfinite(ext_ratio) else 0.0
        return self.base_settle_s + self.ext_extra_s * ext

    def in_cooldown(self, ext_ratio=0.0):
        dur = self.settle_duration(ext_ratio)
        return dur > 0.0 and self.time_since_moving < dur

    def consume_kf_reanchor(self, ext_ratio=0.0):
        """物理刹车尾段过去约 65% 冷静窗时触发一次 KF 残速清理与起点重锚。"""
        dur = self.settle_duration(ext_ratio)
        if dur <= 0.0 or self._reanchored_this_stop:
            return False
        if self.time_since_moving >= dur * self.kf_reanchor_ratio:
            self._reanchored_this_stop = True
            return True
        return False

    def effective_unlock_threshold(self, deadband_on, ext_ratio=0.0):
        base = max(0.0, float(deadband_on))
        if self.in_cooldown(ext_ratio):
            return base * self.override_speed_mult
        return base

    def effective_fast_start_dist(self, fast_start_dist, ext_ratio=0.0):
        base = max(0.0, float(fast_start_dist))
        if self.in_cooldown(ext_ratio):
            return max(base * 1.5, self.override_min_dist)
        return base


def compute_reach_schedule(reach_r, pos_err_norm, soft_start=0.24,
                           soft_end=0.38, min_scale=0.48, dock_dist=0.25,
                           ff_min_scale=0.50, lead_min_scale=0.30,
                           accel_min_scale=0.65, deadband_on_boost=0.35,
                           fast_start_boost=0.40):
    """基于水平伸展半径的平滑稳定调度（近/中区全为 1.0，伸展区临界阻尼化）。

    机械臂前伸时等效惯量 I ∝ r^2 增大、刚度下降且 MIT 滞后增加：
    在保留大误差远距追赶能力的同时，随伸展度平滑衰减停靠速度、前馈、lead 与
    最大加速度，并适度抬高静止防误唤醒门槛，防止伸展悬臂回弹引起往复晃动。
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
    """分轴闭环自激振荡检测与阻尼抑制器（专杀 1.2~2.5Hz 极限环，不伤正常快跟）。

    原理：单程高速追赶（0 次换向）、单次快速折返（1~2 次换向）、大误差主动
    追赶（|err| >= large_err_release）及正常人类慢挥手（半周期 > max_half_s）
    均不会误触发，osc_level 恒为 0，零速度损耗；一旦某轴中小误差连续
    ≥confirm_reversals 次在 [min_half_s, max_half_s]（机械臂 MIT 共振带）
    内过零反转，该轴 osc_level 迅速拉起至 0.7~1.0，即时撤除该轴正向前馈与
    额外 lead、封锁 Q 放大并削减比例增益，1 个半周期内将自激极限环压死。
    """

    def __init__(self, err_thresh=0.018, min_half_s=0.14, max_half_s=0.42,
                 decay_tau_s=0.55, confirm_reversals=3,
                 large_err_release=0.15, enabled=True):
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
        self.last_sign[:] = 0
        self.last_flip_t[:] = -1e9
        self.streak[:] = 0
        self.levels[:] = 0.0

    def update(self, error_vec, now, dt, active=True):
        if not self.enabled or not active:
            self.reset()
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
            if abs(v) < self.err_thresh:
                continue
            sign = 1 if v > 0.0 else -1
            if self.last_sign[ax] == 0:
                self.last_sign[ax] = sign
                self.last_flip_t[ax] = t_now
            elif sign != self.last_sign[ax]:
                half_p = t_now - self.last_flip_t[ax]
                if self.min_half_s <= half_p <= self.max_half_s:
                    self.streak[ax] += 1
                    if self.streak[ax] >= self.confirm_reversals:
                        self.levels[ax] = min(1.0, self.levels[ax] + 0.70)
                else:
                    self.streak[ax] = 0
                self.last_sign[ax] = sign
                self.last_flip_t[ax] = t_now
            elif t_now - self.last_flip_t[ax] > self.max_half_s * 1.4:
                self.streak[ax] = 0
        self.levels[self.levels < 0.02] = 0.0
        return self.levels.copy()



