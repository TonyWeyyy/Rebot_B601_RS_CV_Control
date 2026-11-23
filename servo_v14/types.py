"""
v14.0 架构强类型数据契约定义（servo_v14/types.py）
=================================================
明确定义感知层、估计层、状态机层、控制层、安全护盾与执行后端之间传递的
所有结构体与物理单位，杜绝无类型 dict 拼写错误与单位混用。

单位约定：
    - 位置与距离：米 (m)
    - 线速度与加速度：m/s、m/s²
    - 关节角与角速度：弧度 (rad)、rad/s（6 轴；夹爪 J7 角度另以 deg 显式标注）
    - 力矩：牛米 (Nm)
    - 时间戳与周期：单调秒 (s)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class FSMMode(str, Enum):
    """v14 显式视觉伺服有限状态机模式枚举。"""

    IDLE = "IDLE"                # 未启动跟随，机械臂静止待命
    TRACKING = "TRACKING"        # 正常世界系 PBVS 追手/巡航
    DOCKING = "DOCKING"          # 近目标减速停靠区（平滑撤出前馈并收紧关节限速）
    HOLDING = "HOLDING"          # 目标静止锁存：严格保持当前物理位姿，零速度漂移
    COASTING = "COASTING"        # 短时丢手滑行保持窗（指数衰减速度）
    LOST = "LOST"                # 长时丢手：归零速度并复位估计器
    FAULT_HOLD = "FAULT_HOLD"    # M6 安全监督触发原地软保持（永不半空失能，按 g 恢复）
    RETURN_HOME = "RETURN_HOME"  # 硬工作空间越界 >5cm，缓慢安全回 HOME


@dataclass(frozen=True)
class Observation:
    """单帧视觉感知观测数据包（由视觉线程或离线日志回放构造）。"""

    seq: int                                   # 单调递增观测序号（仅新帧变化）
    meas_time_s: float                         # 帧测量/采集单调时间戳 (s)
    host_time_s: float                         # 主机接收单调时间戳 (s)
    visible: bool                              # 手掌是否可见（含 LK 光流桥接）
    frozen: bool                               # True=2D 有效但深度暂时冻结（仅允许 XY 子空间更新）
    p_cam: Optional[np.ndarray]                # 相机系掌心 3D 坐标 [x, y, z] (m)，不可见时为 None
    processing_s: float = 0.0                  # 视觉推理处理耗时 (s)
    hw_time_used: bool = False                 # 是否使用了相机硬件时间戳


@dataclass(frozen=True)
class RobotState:
    """机械臂当前状态快照（严格区分控制内环 q_cmd 与感知用实测 q_act）。"""

    timestamp_s: float                         # 状态采集单调时间戳 (s)
    q_cmd: np.ndarray                          # 当前控制指令关节角 (6,) rad（控制雅可比/软硬墙唯一依据）
    q_act: np.ndarray                          # 200Hz 真实物理关节角 (6,) rad（仅用于感知/估计/日志）
    dq_act: np.ndarray                         # 200Hz 真实物理关节角速度 (6,) rad/s
    tau_act: np.ndarray                        # 200Hz 实测关节力矩 (6,) Nm
    feedback_valid: bool = True                # 200Hz 实测反馈是否新鲜有效
    feedback_age_ms: float = 0.0               # 实测反馈缓存年龄 (ms)
    t_mos_max_c: float = 25.0                  # 最高 MOS 温度 (°C)
    supervisor_latched: bool = False           # M6 安全监督是否处于软停机锁存态
    supervisor_reason: str = ""                # M6 软停机触发原因
    thermal_speed_scale: float = 1.0           # M6 高温前置速度降额倍率 [0.40, 1.0]
    mit_period_p99_ms: float = 5.0             # 底层 200Hz 循环周期 P99 (ms)
    can_fail_total: int = 0                    # CAN 累计发送失败次数
    tau_ff_nm: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    kd_j1: float = 4.6                         # 当前 J1 生效阻尼系数


@dataclass(frozen=True)
class TargetEstimate:
    """层 1 输出：统一基座世界系下的目标位置与速度估计。"""

    valid: bool                                # 估计器是否已初始化有效
    new_obs: bool                              # 本拍是否消费了新视觉观测
    visible: bool                              # 当前观测是否可见
    frozen: bool                               # 当前是否为深度冻结态（仅 XY 子空间更新）
    p_world: np.ndarray                        # 滤波后目标世界系位置 (3,) m
    v_world: np.ndarray                        # 滤波后目标世界系速度 (3,) m/s
    p_world_lead: np.ndarray                   # 叠加观测年龄与自适应前瞻后的目标世界位置 (3,) m
    obs_age_s: float                           # 本拍观测年龄 (s)
    lead_s: float                              # 本拍生效的额外前瞻时间 (s)
    lead_weight: float                         # 自适应前瞻权重 [0.0, 1.0]
    q_scale: np.ndarray                        # 当前三轴过程噪声 Q 放大倍率 (3,)
    cam_pos_act: np.ndarray                    # 当前真实物理相机世界位置 (3,) m
    cam_rot_act: np.ndarray                    # 当前真实物理相机世界旋转矩阵 (3, 3)


@dataclass(frozen=True)
class FSMDecision:
    """层 2 输出：显式状态机模式决策与使能信号。"""

    mode: FSMMode                              # 当前生效状态
    prev_mode: FSMMode                         # 上一拍状态
    transition_reason: str                     # 本拍发生状态转移的原因（未转移则为空）
    allow_motion: bool                         # 是否允许输出非零速度指令
    allow_ff: bool                             # 是否允许世界系目标速度前馈
    docking: bool                              # 是否处于近目标减速停靠区
    in_cooldown: bool                          # 是否处于刹停冷静窗内
    kf_reanchor_event: bool                    # 冷静窗 65% 处触发一次性清理 KF 残速事件
    hold_anchor_world: Optional[np.ndarray]    # 静止保持时的世界系锚点 (3,) m
    error_norm_m: float                        # 当前世界系相机位置误差模长 (m)


@dataclass(frozen=True)
class ControlCommand:
    """层 3 输出：未经安全护盾拦截前的期望关节速度与运动学上下文。"""

    v_tcp_world: np.ndarray                    # 期望 TCP 世界系线速度 (3,) m/s
    w_tcp_world: np.ndarray                    # 期望 TCP 世界系角速度 (3,) rad/s
    dq_des: np.ndarray                         # 期望 6 轴关节角速度 (6,) rad/s
    tcp_pos_cmd: np.ndarray                    # 当前指令位姿下 TCP 世界位置 (3,) m
    tcp_rot_cmd: np.ndarray                    # 当前指令位姿下 TCP 世界旋转矩阵 (3, 3)
    jacobian_cmd: np.ndarray                   # 当前指令位姿下 6×6 雅可比矩阵
    reach_radius_m: float                      # 水平伸展半径 hypot(X, Y) (m)
    ori_weight: float                          # 本拍加权 DLS 姿态权重 [0.80, 1.0]
    osc_active: bool                           # 振荡阻尼安全网当前是否活跃
    docking: bool                              # 是否处于停靠区（告知护盾收紧关节限速）
    ff_scale: float                            # 本拍生效前馈比例
    static_hold_active: bool                   # 尾速归零保持是否激活


@dataclass(frozen=True)
class ShieldedCommand:
    """层 4 输出：经安全护盾裁剪与积分后的最终下发指令。"""

    q_cmd_next: np.ndarray                     # 下一拍安全关节目标角 (6,) rad
    dq_cmd_safe: np.ndarray                    # 安全限幅后关节速度 (6,) rad/s
    dq_ff_safe: np.ndarray                     # 送入底层 MIT 的安全速度前馈 (6,) rad/s（软限位轴强制 0）
    reach_radius_m: float                      # 下发给底层 P3 阻尼调度的水平伸展半径 (m)
    trigger_return_home: bool                  # 硬工作空间越界 >5cm，要求立即安全回 HOME
    soft_wall_active: bool                     # 本拍是否触及工作空间软墙
    limit_cushion_active: bool                 # 本拍是否触及关节软限位缓冲
    singularity_active: bool                   # 本拍是否触及奇异降速保护
    speed_clamped: bool                        # 本拍是否触及关节速度硬上限
    manipulability: float                      # 当前可操作度标量
    shield_reason: str = ""                    # 护盾介入原因摘要


@dataclass(frozen=True)
class ServoTickTelemetry:
    """单拍 100Hz 完整诊断快照（兼容 analyze_servo_log.py 与数字孪生回放评估）。"""

    host_time_s: float
    dt_s: float
    seq: int
    new_obs: bool
    mode: FSMMode
    visible: bool
    frozen: bool
    kf_valid: bool
    error_cam: np.ndarray                      # 相机系位置误差 [ex, ey, ez] (m)
    error_norm_m: float
    p_world_est: np.ndarray                    # 世界系目标估计位置 (3,) m
    v_world_est: np.ndarray                    # 世界系目标估计速度 (3,) m/s
    q_cmd: np.ndarray                          # (6,) rad
    dq_cmd: np.ndarray                         # (6,) rad/s
    q_act: np.ndarray                          # (6,) rad
    dq_act: np.ndarray                         # (6,) rad/s
    tau_act: np.ndarray                        # (6,) Nm
    tracking_err_max_deg: float                # 6 轴最大跟踪误差 (deg)
    joint_speed_max: float                     # 指令最大单轴速度 (rad/s)
    obs_age_ms: float
    lead_ms: float
    ctrl_work_ms: float
    shield_flags: str = ""
