"""
可插拔机械臂后端接口：真机后端 + 离线数字孪生后端（servo_v14/robot_backend.py）
==============================================================================
定义统一的 `RobotBackend` 抽象接口，使 `ServoV14Pipeline` 在**真实 B601-RS 机械臂**
与**离线数字孪生仿真器**上运行 100% 同一套代码，彻底消除 Sim-to-Real 代码分叉：

1. `HardwareRobotBackend`：
   - 复用经过 v13.0 实机充分验收的 `RebotAdapter` + `mit_controller_ext.py`（P0~P4）；
   - 保持单 CAN 主机 J7 夹爪通道、独立 Pinocchio 重力模型与 `hold() → stop(return_to_zero=True, wait=True)` 安全退出链；
   - 提供线程安全 FK/雅可比计算与 200Hz 实测状态快照读取。

2. `DigitalTwinBackend`（v14 支柱三：离线数字孪生）：
   - 优先加载真实 B601-RS URDF + Pinocchio（含 `config/joint_offset.json` 与 `0.32 kg` Gemini 2 负载），
     在无 Pinocchio/URDF 的纯单测环境下自动退回解析运动学替身；
   - 内置 200Hz 辨识动力学积分器（参数来自 B601-RS P0~P3 实机扫频与重力标定）：
     - 完整仿真 MIT 五元组 $\tau = k_p(q_{\text{cmd}} - q) + k_d(r)(\dot{q}_{\text{ff}} - \dot{q}) + \tau_{\text{ff}} - b\dot{q} - g(q)$；
     - 仿真 5ms 纯总线传输延迟、J1 随伸展半径 $r^2$ 增长的悬臂惯量、实测速度量化噪声与 M6 安全监督；
   - 支持高速离线步进（100x+ 实时速度）与 `--sim` 交互式演练。
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from typing import Deque, Optional, Protocol, Tuple

import numpy as np

from .config import SimConfig
from .types import RobotState, ShieldedCommand

# 默认 B601-RS 关节软限位（rad），与 rebot_adapter.py / URDF 一致
DEFAULT_Q_MIN = np.array([-2.79, -0.05, -0.05, -1.92, -1.57, -3.14], dtype=np.float64)
DEFAULT_Q_MAX = np.array([2.79, 2.95, 2.95, 1.92, 1.57, 3.14], dtype=np.float64)

URDF_DEFAULT_PATH = os.path.expanduser(
    "~/wey/rebot_developer/reBotArm_control_py/urdf/RS/urdf/ReBot_Arm_RS.urdf"
)


class RobotBackend(Protocol):
    """真机与离线数字孪生共享的机械臂后端契约。"""

    @property
    def q_min(self) -> np.ndarray: ...

    @property
    def q_max(self) -> np.ndarray: ...

    def fk_and_jacobian(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算给定 6 轴关节角 `q` (rad) 下的 `(tcp_pos, tcp_rot, J_6x6)`。"""
        ...

    def fk_only(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """仅计算给定 6 轴关节角 `q` (rad) 下的 `(tcp_pos, tcp_rot)`。"""
        ...

    def get_state(self, now_s: float) -> RobotState:
        """获取当前机械臂状态快照（包含 `q_cmd` 与 200Hz 实测 `q_act/dq_act/tau_act`）。"""
        ...

    def apply_command(self, cmd: ShieldedCommand, dt_s: float) -> None:
        """下发经 `SafetyShield` 审定后的安全指令（并推进一步数字孪生动力学）。"""
        ...

    def set_gripper_deg(self, angle_deg: float) -> None:
        """设置 J7 夹爪目标角 (deg)，走单一 CAN 主机第 7 通道。"""
        ...

    def get_gripper_deg(self) -> float:
        """读取当前 J7 夹爪目标/估算角 (deg)。"""
        ...

    def clear_safety_fault(self) -> bool:
        """清除 M6 安全监督软停机锁存态（按 `g` 恢复跟随时调用）。"""
        ...

    def move_to_joints(self, q_target: np.ndarray, speed_deg_s: float = 20.0, wait: bool = True) -> None:
        """平滑点动至目标关节角（如启动回 HOME 或硬墙越界自动回 HOME）。"""
        ...

    def hold(self) -> None:
        """立即保持当前关节位姿（强制 `dq_des=0`）。"""
        ...

    def stop(self) -> None:
        """统一安全退出：保持 → 缓慢回零到位 → 失能（严禁半空失能）。"""
        ...


class HardwareRobotBackend:
    """真机硬件后端：薄封装经过实机验收的 `RebotAdapter`（v13.0 P0~P4）。"""

    def __init__(self, adapter=None) -> None:
        """
        初始化真机后端。

        Args:
            adapter: 已创建的 `RebotAdapter` 实例；为 None 时自动惰性导入并初始化真机连接。
        """
        if adapter is None:
            from rebot_adapter import RebotAdapter
            adapter = RebotAdapter()
        self.adapter = adapter

    @property
    def q_min(self) -> np.ndarray:
        return np.asarray(self.adapter.q_min, dtype=np.float64)

    @property
    def q_max(self) -> np.ndarray:
        return np.asarray(self.adapter.q_max, dtype=np.float64)

    def fk_and_jacobian(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """调用 Pinocchio 计算 `joint6` TCP 位姿与 6×6 LOCAL_WORLD_ALIGNED 雅可比（与 v13 100% 对齐）。"""
        import pinocchio as pin

        q_arr = np.asarray(q, dtype=np.float64)
        q_full = self.adapter._build_q_full(q_arr)
        pin.forwardKinematics(self.adapter.model, self.adapter.data, q_full)
        pin.updateFramePlacements(self.adapter.model, self.adapter.data)
        pin.computeJointJacobians(self.adapter.model, self.adapter.data, q_full)

        last_jid = self.adapter.model.getJointId(self.adapter.joint_names[-1])
        oMi = self.adapter.data.oMi[last_jid]
        pos = oMi.translation.copy()
        rot = oMi.rotation.copy()
        J_full = pin.getJointJacobian(
            self.adapter.model,
            self.adapter.data,
            last_jid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        cols = [
            self.adapter.model.joints[self.adapter.model.getJointId(n)].idx_v
            for n in self.adapter.joint_names
        ]
        return pos, rot, J_full[:, cols].copy()

    def fk_only(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """基于指定 6 轴关节角（不阻塞读 CAN）快速计算 `joint6` TCP 位姿。"""
        import pinocchio as pin

        q_full = self.adapter._build_q_full(np.asarray(q, dtype=np.float64))
        pin.forwardKinematics(self.adapter.model, self.adapter.data, q_full)
        last_jid = self.adapter.model.getJointId(self.adapter.joint_names[-1])
        oMi = self.adapter.data.oMi[last_jid]
        return oMi.translation.copy(), oMi.rotation.copy()

    def get_state(self, now_s: float) -> RobotState:
        """组合 `q_cmd`、200Hz 零开销实测反馈缓存与 M6 主动安全监督状态。"""
        q_cmd = np.asarray(self.adapter.get_command_pos(), dtype=np.float64)
        fb = self.adapter.get_feedback_state(max_age_s=0.05)
        sup = self.adapter.get_safety_status() or {}
        stats = self.adapter.get_mit_loop_stats() or {}

        if fb is not None and fb.get("valid", False):
            q_act = np.asarray(fb["q"], dtype=np.float64)
            dq_act = np.asarray(fb["dq"], dtype=np.float64)
            tau_act = np.asarray(fb["tau"], dtype=np.float64)
            fb_valid = True
            fb_age_ms = float(fb.get("age_ms", 0.0))
            t_mos = fb.get("t_mos")
            t_mos_max = float(np.nanmax(t_mos)) if (t_mos is not None and np.any(np.isfinite(t_mos))) else 25.0
            tau_ff = np.asarray(fb.get("tau_ff", np.zeros(6)), dtype=np.float64)
            kd_vec = fb.get("kd")
            kd_j1 = float(kd_vec[0]) if (kd_vec is not None and len(kd_vec) > 0) else 4.6
            period_p99 = float(stats.get("period_p99_ms", 5.0))
            can_fail = int(stats.get("can_fail_total", 0))
        else:
            q_act = q_cmd.copy()
            dq_act = np.zeros(6, dtype=np.float64)
            tau_act = np.zeros(6, dtype=np.float64)
            fb_valid = False
            fb_age_ms = 999.0
            t_mos_max = 25.0
            tau_ff = np.zeros(6, dtype=np.float64)
            kd_j1 = 4.6
            period_p99 = float(stats.get("period_p99_ms", 5.0))
            can_fail = int(stats.get("can_fail_total", 0))

        return RobotState(
            timestamp_s=float(now_s),
            q_cmd=q_cmd,
            q_act=q_act,
            dq_act=dq_act,
            tau_act=tau_act,
            feedback_valid=fb_valid,
            feedback_age_ms=fb_age_ms,
            t_mos_max_c=t_mos_max,
            supervisor_latched=bool(sup.get("latched", False)),
            supervisor_reason=str(sup.get("reason", "")),
            thermal_speed_scale=float(sup.get("thermal_speed_scale", 1.0)),
            mit_period_p99_ms=period_p99,
            can_fail_total=can_fail,
            tau_ff_nm=tau_ff,
            kd_j1=kd_j1,
        )

    def apply_command(self, cmd: ShieldedCommand, dt_s: float) -> None:
        """下发经护盾裁剪与积分后的安全关节位置+速度前馈给 `RebotAdapter`。"""
        self.adapter.set_reach_radius(cmd.reach_radius_m)
        self.adapter.set_joint_targets_direct(cmd.q_cmd_next, cmd.dq_ff_safe)

    def set_gripper_deg(self, angle_deg: float) -> None:
        self.adapter.set_gripper_deg(float(angle_deg))

    def get_gripper_deg(self) -> float:
        return float(self.adapter.get_gripper_deg())

    def clear_safety_fault(self) -> bool:
        sup = self.adapter.get_safety_status() or {}
        was = bool(sup.get("latched", False))
        self.adapter.clear_safety_fault()
        return was

    def move_to_joints(self, q_target: np.ndarray, speed_deg_s: float = 20.0, wait: bool = True) -> None:
        self.adapter.move_to_joints(np.asarray(q_target, dtype=np.float64), speed_deg_s=speed_deg_s, wait=wait)

    def hold(self) -> None:
        self.adapter.hold()

    def stop(self) -> None:
        """硬件安全退出红线：先保持再平滑回零到位后失能。"""
        self.adapter.hold()
        self.adapter.stop()


class _PinocchioKinematicsHelper:
    """加载真实 B601-RS URDF + Pinocchio（含关节零位偏置与 0.32kg 相机质点重力模型）。"""

    def __init__(self, urdf_path: str, joint_offset_path: Optional[str] = None) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.grav_model = pin.buildModelFromUrdf(urdf_path)
        # 在 link6 挂入 0.32kg Gemini 2 相机质点，与真实硬件一致
        try:
            jid = self.grav_model.getJointId("joint6")
            if 0 < jid < self.grav_model.njoints:
                com = np.array([0.056, -0.011, 0.095], dtype=np.float64)
                inertia = pin.Inertia(0.32, com, np.eye(3, dtype=np.float64) * 1e-4)
                self.grav_model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
        except Exception:
            pass
        self.grav_data = self.grav_model.createData()

        self.joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
        self.last_jid = self.model.getJointId(self.joint_names[-1])
        self._idx_q = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.joint_names]
        self._idx_v = [self.model.joints[self.model.getJointId(n)].idx_v for n in self.joint_names]
        self._grav_idx_q = [self.grav_model.joints[self.grav_model.getJointId(n)].idx_q for n in self.joint_names]
        self._grav_idx_v = [self.grav_model.joints[self.grav_model.getJointId(n)].idx_v for n in self.joint_names]

        self.joint_offset = np.zeros(6, dtype=np.float64)
        if joint_offset_path and os.path.exists(joint_offset_path):
            try:
                with open(joint_offset_path, encoding="utf-8") as stream:
                    raw = json.load(stream)
                off = raw.get("joint_offset_rad") or raw.get("offsets_rad") or raw.get("joint_offset")
                if off is not None and len(off) == 6:
                    self.joint_offset = np.asarray(off, dtype=np.float64)
            except Exception:
                pass

        margin = 0.05
        self.q_min = np.array(
            [self.model.lowerPositionLimit[iq] + margin for iq in self._idx_q],
            dtype=np.float64,
        )
        self.q_max = np.array(
            [self.model.upperPositionLimit[iq] - margin for iq in self._idx_q],
            dtype=np.float64,
        )

    def _q_full(self, q6: np.ndarray, *, grav: bool = False) -> np.ndarray:
        model = self.grav_model if grav else self.model
        idx_q = self._grav_idx_q if grav else self._idx_q
        q = np.zeros(model.nq, dtype=np.float64)
        for i, iq in enumerate(idx_q):
            q[iq] = float(q6[i]) + float(self.joint_offset[i])
        return q

    def fk_and_jacobian(self, q6: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pin = self._pin
        qf = self._q_full(q6, grav=False)
        pin.forwardKinematics(self.model, self.data, qf)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, qf)
        oMi = self.data.oMi[self.last_jid]
        J_full = pin.getJointJacobian(
            self.model, self.data, self.last_jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return oMi.translation.copy(), oMi.rotation.copy(), J_full[:, self._idx_v].copy()

    def fk_only(self, q6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pin = self._pin
        qf = self._q_full(q6, grav=False)
        pin.forwardKinematics(self.model, self.data, qf)
        oMi = self.data.oMi[self.last_jid]
        return oMi.translation.copy(), oMi.rotation.copy()

    def gravity_torque(self, q6: np.ndarray) -> np.ndarray:
        pin = self._pin
        qf = self._q_full(q6, grav=True)
        tau = pin.computeGeneralizedGravity(self.grav_model, self.grav_data, qf)
        return np.array([float(tau[iv]) for iv in self._grav_idx_v], dtype=np.float64)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将 URDF 固定轴 XYZ（roll, pitch, yaw）欧拉角转为 3×3 旋转矩阵 Rz(yaw)@Ry(pitch)@Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _axis_angle_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """罗德里格斯公式：绕单位轴 `axis` 旋转 `angle` (rad) 的 3×3 旋转矩阵。"""
    c, s = math.cos(angle), math.sin(angle)
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


class _PureNumpyURDFKinematics:
    """
    与 B601-RS `ReBot_Arm_RS.urdf` 逐参数精确一致的纯 NumPy 运动学/雅可比/静重力引擎。

    无需 Pinocchio 即可在任意 Python 环境下提供亚毫米级一致的 FK、LOCAL_WORLD_ALIGNED
    几何雅可比与含末端夹爪 + 0.32kg Gemini 2 相机的广义重力矩 `g(q)`。
    """

    # 直接摘自 ReBot_Arm_RS.urdf 的 6 关节运动链定义：(xyz, rpy, axis, lower, upper)
    _JOINTS = (
        ((-0.00034283, -0.00098683, 0.075), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), -2.80, 2.80),
        ((0.020343, 0.027237, 0.070), (-1.5708, 0.0, 0.0), (0.0, 0.0, 1.0), 0.00, 3.14),
        ((-0.236, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.00, 3.14),
        ((0.228, -0.072746, 0.0045), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), -1.57, 1.57),
        ((0.087, -0.048, -0.03075), (-1.5708, 0.0, 0.0), (0.0, 0.0, -1.0), -1.57, 1.57),
        ((0.0365, 0.0, 0.048), (0.0, 1.5708, 0.0), (0.0, 0.0, -1.0), -3.14, 3.14),
    )

    # 各连杆 (mass_kg, com_local_m) — 含 URDF link1~link6、固定夹爪与 0.32kg Gemini 2 相机质点
    _LINK_BODIES = (
        (0, 0.4655, (0.00818, 0.00123, 0.03314)),          # link1
        (1, 1.5520, (-0.11786, -0.00342, -0.02716)),       # link2
        (2, 1.2520, (0.10343, -0.06641, -0.02711)),        # link3
        (3, 0.4600, (0.04755, -0.04411, -0.03141)),        # link4
        (4, 0.2012, (0.01144, 0.00000, 0.03805)),          # link5
        (5, 0.1000, (0.00002, 0.00000, 0.00435)),          # link6
        # gripper_end（固定于 link6 [0,0,0.16621], rpy=[3.1416,-1.5708,0]）+ 左右爪合并等效到 link6 系
        (5, 0.8004, (0.00000, -0.00013, 0.05346)),
        # 外挂 Gemini 2 相机 + 支架等效质点（P0/P3 实机反推 0.32kg）
        (5, 0.3200, (0.05600, -0.01100, 0.09500)),
    )

    def __init__(self, joint_offset_path: Optional[str] = None) -> None:
        self._fixed_t = [np.array(j[0], dtype=np.float64) for j in self._JOINTS]
        self._fixed_R = [_rpy_matrix(*j[1]) for j in self._JOINTS]
        self._axes = [np.array(j[2], dtype=np.float64) for j in self._JOINTS]
        margin = 0.05
        self.q_min = np.array([j[3] + margin for j in self._JOINTS], dtype=np.float64)
        self.q_max = np.array([j[4] - margin for j in self._JOINTS], dtype=np.float64)

        self.joint_offset = np.zeros(6, dtype=np.float64)
        if joint_offset_path and os.path.exists(joint_offset_path):
            try:
                with open(joint_offset_path, encoding="utf-8") as stream:
                    raw = json.load(stream)
                off = raw.get("joint_offset_rad") or raw.get("offsets_rad") or raw.get("joint_offset")
                if off is not None and len(off) == 6:
                    self.joint_offset = np.asarray(off, dtype=np.float64)
            except Exception:
                pass

    def _chain(self, q: np.ndarray) -> Tuple[list, list, list]:
        """正向递推 6 个关节的基座系位姿 `(p_i, R_i)` 与世界系转轴 `z_i`。"""
        p_cur = np.zeros(3, dtype=np.float64)
        R_cur = np.eye(3, dtype=np.float64)
        p_list, R_list, z_list = [], [], []
        for i in range(6):
            p_cur = p_cur + R_cur @ self._fixed_t[i]
            R_joint0 = R_cur @ self._fixed_R[i]
            qi = float(q[i]) + float(self.joint_offset[i])
            R_cur = R_joint0 @ _axis_angle_rot(self._axes[i], qi)
            z_w = R_joint0 @ self._axes[i]
            p_list.append(p_cur.copy())
            R_list.append(R_cur.copy())
            z_list.append(z_w)
        return p_list, R_list, z_list

    def fk_only(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_list, R_list, _ = self._chain(np.asarray(q, dtype=np.float64))
        return p_list[-1], R_list[-1]

    def fk_and_jacobian(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_list, R_list, z_list = self._chain(np.asarray(q, dtype=np.float64))
        p_end = p_list[-1]
        R_end = R_list[-1]
        J = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            J[:3, i] = np.cross(z_list[i], p_end - p_list[i])
            J[3:, i] = z_list[i]
        return p_end, R_end, J

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """精确解析计算 6 关节广义静重力补偿力矩 g(q) (Nm)。"""
        p_list, R_list, z_list = self._chain(np.asarray(q, dtype=np.float64))
        g_vec = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        tau = np.zeros(6, dtype=np.float64)
        for link_idx, mass, com_local in self._LINK_BODIES:
            c_w = p_list[link_idx] + R_list[link_idx] @ np.asarray(com_local, dtype=np.float64)
            f_g = mass * g_vec
            for i in range(link_idx + 1):
                # 抵消重力所需驱动力矩 τ_i = -z_i · ((c_w - p_i) × m g)
                tau[i] -= float(np.dot(z_list[i], np.cross(c_w - p_list[i], f_g)))
        return tau


class DigitalTwinBackend:
    """
    B601-RS 离线数字孪生后端（v14 支柱三）。

    以 200Hz 积分真实辨识出的 6 轴 MIT 二阶机电动力学（含速度前馈、静重力补偿、
    伸展自适应阻尼、5ms 总线纯滞后、速度量化噪声与 M6 安全监督）。
    """

    def __init__(
        self,
        sim_cfg: Optional[SimConfig] = None,
        q_init: Optional[np.ndarray] = None,
        *,
        use_pinocchio: bool = True,
        urdf_path: str = URDF_DEFAULT_PATH,
        joint_offset_path: Optional[str] = None,
        rng_seed: Optional[int] = 42,
    ) -> None:
        """
        初始化离线数字孪生机械臂后端。

        Args:
            sim_cfg: 辨识动力学参数（`SimConfig`）。
            q_init: 初始 6 轴关节角 (rad)；默认使用示教 HOME。
            use_pinocchio: 优先使用真实 URDF + Pinocchio（若加载失败自动降级解析模型）。
            urdf_path: B601-RS URDF 路径。
            joint_offset_path: 关节零位偏置 JSON 路径。
            rng_seed: 速度量化噪声随机种子；传 None 则完全无噪声确定性仿真。
        """
        self.sim_cfg = sim_cfg or SimConfig()
        self.kin = None
        self.using_pinocchio = False
        if use_pinocchio and os.path.exists(urdf_path):
            try:
                self.kin = _PinocchioKinematicsHelper(urdf_path, joint_offset_path)
                self.using_pinocchio = True
            except Exception:
                self.kin = None
        if self.kin is None:
            self.kin = _PureNumpyURDFKinematics(joint_offset_path=joint_offset_path)

        self._q_min = np.asarray(self.kin.q_min, dtype=np.float64)
        self._q_max = np.asarray(self.kin.q_max, dtype=np.float64)

        if q_init is None:
            q0 = np.array([-0.4313, 0.3445, 0.3691, 0.0938, 0.0018, -0.0015], dtype=np.float64)
        else:
            q0 = np.asarray(q_init, dtype=np.float64).copy()
        q0 = np.clip(q0, self._q_min, self._q_max)

        self.q_cmd = q0.copy()
        self.q_act = q0.copy()
        self.dq_act = np.zeros(6, dtype=np.float64)
        self.tau_act = self.kin.gravity_torque(q0).copy()
        self.tau_ff_last = np.zeros(6, dtype=np.float64)
        self.kd_last = np.asarray(self.sim_cfg.kd_near, dtype=np.float64).copy()
        self.reach_radius_m: float = float(math.hypot(*self.fk_only(q0)[0][:2]))
        self.gripper_deg: float = 285.2

        # 5ms 总线纯延迟队列（存 (q_cmd, dq_ff)）
        self._delay_steps = max(0, int(round(self.sim_cfg.transport_delay_s * self.sim_cfg.mit_hz)))
        self._cmd_pipe: Deque[Tuple[np.ndarray, np.ndarray]] = deque()
        for _ in range(self._delay_steps):
            self._cmd_pipe.append((q0.copy(), np.zeros(6, dtype=np.float64)))

        self._rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        self.supervisor_latched: bool = False
        self.supervisor_reason: str = ""
        self._err_fault_s: float = 0.0
        self._tau_fault_s: float = 0.0
        self.sim_time_s: float = 0.0
        self.stopped: bool = False

    @property
    def q_min(self) -> np.ndarray:
        return self._q_min

    @property
    def q_max(self) -> np.ndarray:
        return self._q_max

    def fk_and_jacobian(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.kin.fk_and_jacobian(np.asarray(q, dtype=np.float64))

    def fk_only(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.kin.fk_only(np.asarray(q, dtype=np.float64))

    def get_state(self, now_s: float) -> RobotState:
        """返回带真实辨识动力学滞后、力矩与可选速度量化噪声的仿真机械臂状态。"""
        dq_meas = self.dq_act.copy()
        if self._rng is not None and self.sim_cfg.dq_noise_std > 0.0:
            dq_meas = dq_meas + self._rng.normal(0.0, self.sim_cfg.dq_noise_std, size=6)
        return RobotState(
            timestamp_s=float(now_s),
            q_cmd=self.q_cmd.copy(),
            q_act=self.q_act.copy(),
            dq_act=dq_meas,
            tau_act=self.tau_act.copy(),
            feedback_valid=True,
            feedback_age_ms=0.5,
            t_mos_max_c=35.0,
            supervisor_latched=self.supervisor_latched,
            supervisor_reason=self.supervisor_reason,
            thermal_speed_scale=1.0,
            mit_period_p99_ms=5.05,
            can_fail_total=0,
            tau_ff_nm=self.tau_ff_last.copy(),
            kd_j1=float(self.kd_last[0]),
        )

    def _compute_scheduled_kd(self, reach_r: float) -> np.ndarray:
        """按水平伸展半径 [0.18, 0.36]m 线性调度 J1~J3 阻尼（与 P3 M3 完全一致）。"""
        ratio = float(np.clip((reach_r - 0.18) / (0.36 - 0.18), 0.0, 1.0))
        kd_near = np.asarray(self.sim_cfg.kd_near, dtype=np.float64)
        kd_far = np.asarray(self.sim_cfg.kd_far, dtype=np.float64)
        return kd_near + ratio * (kd_far - kd_near)

    def apply_command(self, cmd: ShieldedCommand, dt_s: float) -> None:
        """以 200Hz 子步长积分 MIT 机电闭环动力学并更新 M6 安全监督状态。"""
        dt = max(1e-4, min(0.05, float(dt_s)))
        q_start = self.q_cmd.copy()
        q_end = np.clip(cmd.q_cmd_next, self._q_min, self._q_max)
        dq_ff_in = cmd.dq_ff_safe.copy()
        self.reach_radius_m = float(cmd.reach_radius_m)

        mit_dt = 1.0 / self.sim_cfg.mit_hz
        n_sub = max(1, int(round(dt / mit_dt)))
        sub_dt = dt / n_sub

        kp = np.asarray(self.sim_cfg.kp, dtype=np.float64)
        kd = self._compute_scheduled_kd(self.reach_radius_m)
        self.kd_last = kd
        vff_scale = np.asarray(self.sim_cfg.vel_ff_scale, dtype=np.float64)
        gff_scale = np.asarray(self.sim_cfg.gravity_ff_scale, dtype=np.float64)
        visc = np.asarray(self.sim_cfg.viscous_friction, dtype=np.float64)
        tau_lim = np.asarray(self.sim_cfg.tau_limit_nm, dtype=np.float64)

        # J1 转动惯量随悬臂伸展半径 r² 增长（与 P0 双位姿扫频辨识吻合）
        j_eff = np.asarray(self.sim_cfg.j_eff, dtype=np.float64).copy()
        j_eff[0] += self.sim_cfg.j1_ext_coeff * max(0.0, self.reach_radius_m ** 2 - 0.15 ** 2)

        for k in range(1, n_sub + 1):
            # 200Hz FOH 一阶保持插值
            frac = k / n_sub
            q_foh = q_start + frac * (q_end - q_start)

            if self._delay_steps > 0:
                self._cmd_pipe.append((q_foh, dq_ff_in))
                q_des, dq_ff = self._cmd_pipe.popleft()
            else:
                q_des, dq_ff = q_foh, dq_ff_in

            # 真实重力矩与 P3 静态重力前馈
            tau_grav = self.kin.gravity_torque(self.q_act)
            tau_ff = tau_grav * gff_scale
            self.tau_ff_last = tau_ff

            # 完整 MIT 五元组力矩控制律
            dq_des = dq_ff * vff_scale
            tau_motor = kp * (q_des - self.q_act) + kd * (dq_des - self.dq_act) + tau_ff
            # 物理力矩饱和（按额定峰值约 1.35 倍）
            tau_motor = np.clip(tau_motor, -tau_lim * 1.35, tau_lim * 1.35)
            self.tau_act = tau_motor

            # 半隐式欧拉积分（电机力矩 - 真实重力 - 粘滞摩擦 = J * ddq）
            ddq = (tau_motor - tau_grav - visc * self.dq_act) / j_eff
            self.dq_act = self.dq_act + ddq * sub_dt
            self.q_act = np.clip(self.q_act + self.dq_act * sub_dt, self._q_min, self._q_max)

            # 200Hz M6 主动安全监督
            track_err_deg = float(np.max(np.abs(np.rad2deg(q_des - self.q_act))))
            if track_err_deg > self.sim_cfg.track_err_limit_deg:
                self._err_fault_s += sub_dt
            else:
                self._err_fault_s = 0.0
            if np.any(np.abs(tau_motor) > tau_lim):
                self._tau_fault_s += sub_dt
            else:
                self._tau_fault_s = 0.0
            if self._err_fault_s >= 0.08:
                self.supervisor_latched = True
                self.supervisor_reason = f"track_err>{self.sim_cfg.track_err_limit_deg:.1f}deg"
            elif self._tau_fault_s >= 0.08:
                self.supervisor_latched = True
                self.supervisor_reason = "torque_limit"

        self.q_cmd = q_end.copy()
        self.sim_time_s += dt

    def set_gripper_deg(self, angle_deg: float) -> None:
        self.gripper_deg = float(np.clip(angle_deg, -100.0, 400.0))

    def get_gripper_deg(self) -> float:
        return self.gripper_deg

    def clear_safety_fault(self) -> bool:
        was = self.supervisor_latched
        self.supervisor_latched = False
        self.supervisor_reason = ""
        self._err_fault_s = 0.0
        self._tau_fault_s = 0.0
        return was

    def move_to_joints(self, q_target: np.ndarray, speed_deg_s: float = 20.0, wait: bool = True) -> None:
        qt = np.clip(np.asarray(q_target, dtype=np.float64), self._q_min, self._q_max)
        self.q_cmd = qt.copy()
        self.q_act = qt.copy()
        self.dq_act[:] = 0.0
        self.tau_act = self.kin.gravity_torque(qt).copy()
        self._cmd_pipe.clear()
        for _ in range(self._delay_steps):
            self._cmd_pipe.append((qt.copy(), np.zeros(6, dtype=np.float64)))

    def hold(self) -> None:
        self.dq_act[:] = 0.0

    def stop(self) -> None:
        self.move_to_joints(np.zeros(6, dtype=np.float64))
        self.stopped = True
