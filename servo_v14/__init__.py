"""
v14.0 模块化视觉伺服架构包（servo_v14）
======================================
将原有单文件控制主循环拆解为严格解耦的四层流水线与可插拔硬件/数字孪生后端：

    Perception (Observation + DepthProtectionChain)
        ↓
    Layer 1: WorldTargetEstimator   （统一世界系目标估计器，基于 200Hz 真实 q_act）
        ↓
    Layer 2: ServoFSM               （显式有限状态机，清晰管理启停/停靠/保持/故障转移）
        ↓
    Layer 3: WorldPBVSController    （统一世界系 PBVS 控制器，严格锚定 q_cmd 雅可比）
        ↓
    Layer 4: SafetyShield           （不可绕过安全护盾：软硬墙、软限位、奇异、硬速与监督）
        ↓
    Backend: HardwareRobotBackend | DigitalTwinBackend（真机与离线数字孪生共享同一接口）

所有配置均为不可变 `@dataclass(frozen=True)`，零模块级全局变量。
"""

from .types import (
    FSMMode,
    Observation,
    RobotState,
    TargetEstimate,
    FSMDecision,
    ControlCommand,
    ShieldedCommand,
    ServoTickTelemetry,
)
from .config import (
    V14Config,
    EstimatorConfig,
    FSMConfig,
    PBVSConfig,
    SafetyConfig,
    DepthGuardConfig,
    CameraConfig,
    SimConfig,
    HARD_MAX_JOINT_SPEED,
    load_v14_config,
    validate_v14_config,
    v14_config_from_follow_dict,
)

__all__ = [
    "FSMMode",
    "Observation",
    "RobotState",
    "TargetEstimate",
    "FSMDecision",
    "ControlCommand",
    "ShieldedCommand",
    "ServoTickTelemetry",
    "V14Config",
    "EstimatorConfig",
    "FSMConfig",
    "PBVSConfig",
    "SafetyConfig",
    "DepthGuardConfig",
    "CameraConfig",
    "SimConfig",
    "HARD_MAX_JOINT_SPEED",
    "load_v14_config",
    "validate_v14_config",
    "v14_config_from_follow_dict",
]
