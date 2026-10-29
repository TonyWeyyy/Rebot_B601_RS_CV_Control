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
