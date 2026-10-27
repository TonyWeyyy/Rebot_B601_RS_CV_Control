#!/usr/bin/env python3
"""视觉伺服的时间与状态切换工具（纯计算，无硬件依赖）。"""

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
