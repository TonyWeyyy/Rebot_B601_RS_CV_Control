#!/usr/bin/env python3
"""
v14.0 四层解耦架构 + 统一世界系 PBVS + 离线数字孪生 单元测试与闭环集成套件
========================================================================
无需相机或 CAN 硬件，全面覆盖：
  1. `V14Config` 不可变配置加载、旧配置适配与全部硬件安全红线负向校验
  2. `DepthProtectionChain` 4 层深度保护链（背景过滤/5 帧恢复/跳变拒收/15 帧防锁死/速率限幅/冻结超时）
  3. `WorldTargetEstimator` 眼在手上世界系不变性、深度冻结期 XY 子空间更新与静止前瞻撤回
  4. `ServoFSM` 显式 8 态转移、快启、折返防假停、冷静窗与 M6/硬墙最高优先级抢占
  5. `WorldPBVSController` 固定参考光轴防俯冲稳定性、停靠前馈撤出与滑行衰减
  6. `SafetyShield` 不可绕过硬限速、水平 0.42m 柱面软墙、活跃集限位重解与穿墙拦截
  7. `DigitalTwinBackend` 纯 NumPy B601-RS URDF 精度（<0.5mm）+ 200Hz MIT 动力学闭环与日志回放
  8. `palm_follow_v14.py` CLI 覆写校验与 `PinchGripperMapper` 运动冻结逻辑
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servo_v14.config import (
    HARD_MAX_JOINT_SPEED,
    V14Config,
    load_v14_config,
    validate_v14_config,
)
from servo_v14.controller_pbvs import WorldPBVSController
from servo_v14.depth_guard import DepthProtectionChain
from servo_v14.estimator import WorldTargetEstimator
from servo_v14.pipeline import ServoV14Pipeline
from servo_v14.robot_backend import DigitalTwinBackend, _PureNumpyURDFKinematics
from servo_v14.safety_shield import SafetyShield
from servo_v14.state_machine import ServoFSM
from servo_v14.types import ControlCommand, FSMMode, Observation, RobotState


def _make_robot_state(
    q_cmd: np.ndarray,
    q_act: np.ndarray | None = None,
    t: float = 1.0,
    **kwargs,
) -> RobotState:
    qa = q_cmd.copy() if q_act is None else np.asarray(q_act, dtype=np.float64)
    return RobotState(
        timestamp_s=t,
        q_cmd=np.asarray(q_cmd, dtype=np.float64),
        q_act=qa,
        dq_act=np.zeros(6, dtype=np.float64),
        tau_act=np.zeros(6, dtype=np.float64),
        **kwargs,
    )


class V14ConfigTests(unittest.TestCase):
    """验证 v14 配置加载、不可变性、旧版适配与硬件安全红线校验。"""

    def test_load_v14_and_compat_follow_yaml(self):
        """验证 config/v14.yaml 与 config/follow_v13.yaml 均可无错加载并通过红线校验。"""
        cfg = load_v14_config(os.path.join(ROOT, "config", "v14.yaml"))
        self.assertEqual(cfg.safety.workspace_max_radius, 0.42)
        self.assertEqual(cfg.safety.max_vel, 0.85)
        with self.assertRaises(Exception):
            cfg.pbvs.gain = 9.9  # frozen dataclass 禁止运行期篡改

        compat = load_v14_config(os.path.join(ROOT, "config", "follow_v13.yaml"))
        self.assertAlmostEqual(compat.pbvs.distance, 0.40)
        self.assertAlmostEqual(compat.estimator.pose_delay_s, 0.025)

    def test_hardware_red_lines_rejected(self):
        """验证超越 70% 臂展、max_vel>0.85、J2/J3>2.0 rad/s、非法距离与 NaN 全部被拒绝。"""
        base = load_v14_config(os.path.join(ROOT, "config", "v14.yaml"))
        bad_cases = [
            replace(base, safety=replace(base.safety, workspace_max_radius=0.43)),
            replace(base, safety=replace(base.safety, max_vel=0.86)),
            replace(base, safety=replace(base.safety, max_vel_joints=(2.5, 2.1, 2.0, 3.0, 3.0, 2.4))),
            replace(base, safety=replace(base.safety, max_vel_joints=(2.5, 2.0, 2.05, 3.0, 3.0, 2.4))),
            replace(base, pbvs=replace(base.pbvs, distance=0.15)),
            replace(base, pbvs=replace(base.pbvs, distance=0.65)),
            replace(base, pbvs=replace(base.pbvs, gain=float("nan"))),
        ]
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                validate_v14_config(bad)


class DepthProtectionChainTests(unittest.TestCase):
    """验证 4 层深度保护链独立模块的全部边界行为。"""

    def test_reacq_background_filter_and_5_frame_consistency(self):
        """验证启动期拒绝背景深度，需连续 5 帧一致深度才激活有效输出。"""
        cfg = V14Config().depth
        guard = DepthProtectionChain(cfg, target_distance_m=0.40)
        # 背景深度 (> 0.40 + 0.25 = 0.65m) 必须被拒
        res_bg = guard.update(0.95, 0.10)
        self.assertFalse(res_bg.valid)
        self.assertEqual(res_bg.reject_reason, "bg_depth_reject")

        # 前 4 帧收集中未输出有效深度，第 5 帧一致通过
        for i in range(4):
            r = guard.update(0.41 + 0.002 * i, 0.20 + 0.02 * i)
            self.assertFalse(r.valid)
        r5 = guard.update(0.415, 0.28)
        self.assertTrue(r5.valid)
        self.assertFalse(r5.frozen)
        self.assertAlmostEqual(r5.z_m, 0.414, delta=0.02)

    def test_jump_reject_anti_lock_and_rate_limit(self):
        """验证单帧大跳变进入冻结、连续 15 帧后防锁死重锚、以及单帧变化率限幅。"""
        cfg = V14Config().depth
        guard = DepthProtectionChain(cfg, target_distance_m=0.40)
        t = 0.0
        for _ in range(5):
            t += 0.016
            guard.update(0.40, t)

        # 单帧跳变 0.20m (> 0.08m) → 拒收并进入 frozen 复用旧安全深度
        t += 0.016
        r_jump = guard.update(0.60, t)
        self.assertTrue(r_jump.valid)
        self.assertTrue(r_jump.frozen)
        self.assertLess(abs(r_jump.z_m - 0.40), 0.02)

        # 正常小步长 +0.07m（< 0.08m 跳变门，但 > 0.05m 速率门）→ 被第 4 层限幅至 +0.05m
        t += 0.016
        r_rate = guard.update(0.47, t)
        self.assertTrue(r_rate.valid)
        self.assertFalse(r_rate.frozen)
        self.assertLessEqual(guard._z_ref, 0.451)

        # 连续 15 帧持续处于新深度 0.60m → 触发防锁死重锚
        r_last = None
        for _ in range(cfg.anti_lock_frames):
            t += 0.010
            r_last = guard.update(0.60, t)
        self.assertTrue(r_last.valid)
        self.assertFalse(r_last.frozen)
        self.assertEqual(r_last.reject_reason, "anti_lock_reanchor")

    def test_freeze_timeout_resets_cleanly(self):
        """验证深度持续无效超过 z_freeze_max_s 后自动退出冻结并清零复位。"""
        cfg = V14Config().depth
        guard = DepthProtectionChain(cfg, target_distance_m=0.40)
        for i in range(5):
            guard.update(0.40, 0.02 * i)
        r_ok = guard.update(0.0, 0.15)
        self.assertTrue(r_ok.valid and r_ok.frozen)
        r_exp = guard.update(0.0, 0.15 + cfg.z_freeze_max_s + 0.05)
        self.assertFalse(r_exp.valid)
        self.assertEqual(r_exp.reject_reason, "freeze_timeout")


class WorldTargetEstimatorTests(unittest.TestCase):
    """验证层 1 单一世界系估计器的不变性与深度冻结 XY 子空间更新。"""

    def test_static_world_point_invariant_under_camera_motion(self):
        """验证相机平移运动中，滞后对齐世界系估计器对静止目标估出零速度。"""
        cfg = V14Config().estimator
        est = WorldTargetEstimator(cfg, np.eye(4), ctrl_dt_s=0.01)
        p_world_true = np.array([0.50, 0.10, 0.30])
        q_dummy = np.zeros(6)

        t = 0.0
        for step_i in range(80):
            t += 0.01
            # 相机沿 X 正弦平移（模拟机械臂自运动）
            cam_pos = np.array([0.10 + 0.08 * math.sin(4.0 * t), 0.0, 0.30])
            cam_pos_delayed = np.array([
                0.10 + 0.08 * math.sin(4.0 * max(0.0, t - cfg.pose_delay_s)), 0.0, 0.30
            ])
            p_cam = p_world_true - cam_pos_delayed
            obs = Observation(
                seq=step_i + 1, meas_time_s=t, host_time_s=t,
                visible=True, frozen=False, p_cam=p_cam,
            )
            res = est.step(obs, _make_robot_state(q_dummy, t=t), cam_pos, np.eye(3), 0.01)

        self.assertTrue(res.valid)
        self.assertLess(float(np.linalg.norm(res.p_world - p_world_true)), 0.015)
        self.assertLess(float(np.linalg.norm(res.v_world)), 0.05)

    def test_frozen_depth_updates_only_xy_without_contaminating_z(self):
        """验证深度冻结期仅更新相机光轴正交 XY 子空间，绝不让占位深度污染世界 Z。"""
        cfg = V14Config().estimator
        est = WorldTargetEstimator(cfg, np.eye(4), ctrl_dt_s=0.01)
        q_dummy = np.zeros(6)
        t = 0.0
        # 稳定初始化在 [0.0, 0.0, 0.40]
        for i in range(15):
            t += 0.01
            obs = Observation(i + 1, t, t, True, False, np.array([0.0, 0.0, 0.40]))
            res = est.step(obs, _make_robot_state(q_dummy, t=t), np.zeros(3), np.eye(3), 0.01)

        # 冻结期：X 移至 0.15m，同时故意塞入错误占位深度 z=0.95m
        for i in range(15, 35):
            t += 0.01
            obs = Observation(i + 1, t, t, True, True, np.array([0.15, 0.0, 0.95]))
            res = est.step(obs, _make_robot_state(q_dummy, t=t), np.zeros(3), np.eye(3), 0.01)

        self.assertGreater(res.p_world[0], 0.10)      # X 已跟踪新位置
        self.assertAlmostEqual(res.p_world[2], 0.40, delta=0.01)  # Z 毫发无损！


class ServoFSMTests(unittest.TestCase):
    """验证层 2 显式有限状态机的全部关键转移与优先级抢占。"""

    def _est(self, p_w, v_w, *, valid=True, visible=True, new_obs=True):
        return type("Est", (), dict(
            valid=valid, visible=visible, new_obs=new_obs, frozen=False,
            p_world=np.asarray(p_w, dtype=np.float64),
            v_world=np.asarray(v_w, dtype=np.float64),
        ))()

    def test_explicit_lifecycle_and_priority_preemption(self):
        """验证 IDLE→TRACKING→DOCKING→HOLDING→FAULT_HOLD→RETURN_HOME 全路径转移。"""
        fsm = ServoFSM(V14Config().fsm)
        self.assertEqual(fsm.mode, FSMMode.IDLE)

        # 启动跟随且大误差 → TRACKING
        d = fsm.step(following_enabled=True, estimate=self._est([0, 0, 0.5], [0.2, 0, 0]),
                     error_norm_m=0.15, joint_speed_rad_s=0.5, dt_s=0.01)
        self.assertEqual(d.mode, FSMMode.TRACKING)
        self.assertTrue(d.allow_motion and d.allow_ff)

        # 接近目标 <= 0.08m → DOCKING
        d = fsm.step(following_enabled=True, estimate=self._est([0, 0, 0.5], [0.1, 0, 0]),
                     error_norm_m=0.05, joint_speed_rad_s=0.3, dt_s=0.01)
        self.assertEqual(d.mode, FSMMode.DOCKING)
        self.assertTrue(d.docking)

        # 进入死区且手掌静止 → HOLDING（速度完全禁止）
        d = fsm.step(following_enabled=True, estimate=self._est([0, 0, 0.5], [0.01, 0, 0]),
                     error_norm_m=0.010, joint_speed_rad_s=0.0, dt_s=0.01)
        self.assertEqual(d.mode, FSMMode.HOLDING)
        self.assertFalse(d.allow_motion)

        # M6 安全监督锁存抢占为 FAULT_HOLD
        d = fsm.step(following_enabled=True, estimate=self._est([0, 0, 0.5], [0.5, 0, 0]),
                     error_norm_m=0.20, joint_speed_rad_s=0.0, dt_s=0.01, supervisor_latched=True)
        self.assertEqual(d.mode, FSMMode.FAULT_HOLD)
        self.assertFalse(d.allow_motion)

        # 硬工作空间越界最高优先级抢占为 RETURN_HOME
        d = fsm.step(following_enabled=True, estimate=self._est([0, 0, 0.5], [0.5, 0, 0]),
                     error_norm_m=0.20, joint_speed_rad_s=0.0, dt_s=0.01,
                     supervisor_latched=True, hard_wall_breached=True)
        self.assertEqual(d.mode, FSMMode.RETURN_HOME)

    def test_brief_loss_coasts_then_resets_to_lost(self):
        """验证短丢手进入 COASTING 滑行，超过 lost_hold_s 转入 LOST。"""
        cfg = V14Config().fsm
        fsm = ServoFSM(cfg)
        fsm.step(following_enabled=True, estimate=self._est([0, 0, 0.5], [0.2, 0, 0]),
                 error_norm_m=0.15, joint_speed_rad_s=0.5, dt_s=0.01)
        lost_est = self._est([0, 0, 0.5], [0.2, 0, 0], valid=True, visible=False, new_obs=False)
        d_coast = fsm.step(following_enabled=True, estimate=lost_est,
                           error_norm_m=0.15, joint_speed_rad_s=0.3, dt_s=0.02)
        self.assertEqual(d_coast.mode, FSMMode.COASTING)
        d_lost = None
        for _ in range(40):
            d_lost = fsm.step(following_enabled=True, estimate=lost_est,
                              error_norm_m=0.15, joint_speed_rad_s=0.0, dt_s=0.01)
        self.assertEqual(d_lost.mode, FSMMode.LOST)
        self.assertFalse(d_lost.allow_motion)


class WorldPBVSAndShieldTests(unittest.TestCase):
    """验证层 3 固定参考光轴稳定性与层 4 安全护盾不可绕过硬约束。"""

    def test_reference_optical_axis_prevents_tilt_dive(self):
        """验证腕部俯仰偏斜 30° 时，固定参考光轴仍保持世界系期望点不变，杜绝正反馈俯冲。"""
        cfg = V14Config()
        ctrl = WorldPBVSController(cfg.pbvs, cfg.safety, np.eye(4))
        ctrl.set_reference_orientation(np.eye(3))
        p_target = np.array([0.20, 0.0, 0.70])
        tcp_p = np.array([0.20, 0.0, 0.30])
        # 模拟 J4 触限导致仰角倾斜 30°
        th = math.radians(30.0)
        R_tilted = np.array([
            [math.cos(th), 0, math.sin(th)],
            [0, 1, 0],
            [-math.sin(th), 0, math.cos(th)],
        ])
        err = ctrl.compute_world_position_error(p_target, tcp_p, R_tilted)
        self.assertAlmostEqual(float(np.linalg.norm(err)), 0.0, places=6)

    def test_safety_shield_unbypassable_limits_and_walls(self):
        """验证即使恶意上层输出超大速度，安全护盾仍严守硬限速、水平软墙、热降额与 M6 软保持。"""
        cfg = V14Config()
        kin = _PureNumpyURDFKinematics(os.path.join(ROOT, "config", "joint_offset.json"))
        shield = SafetyShield(cfg.safety, kin.q_min, kin.q_max)

        q = np.array([-0.43, 0.35, 0.37, 0.09, 0.0, 0.0])
        tcp_p, tcp_r, J = kin.fk_and_jacobian(q)
        malicious_cmd = ControlCommand(
            v_tcp_world=np.array([5.0, 5.0, 5.0]),
            w_tcp_world=np.array([10.0, 0.0, 0.0]),
            dq_des=np.full(6, 10.0),
            tcp_pos_cmd=tcp_p,
            tcp_rot_cmd=tcp_r,
            jacobian_cmd=J,
            reach_radius_m=float(math.hypot(tcp_p[0], tcp_p[1])),
            ori_weight=1.0,
            osc_active=False,
            docking=False,
            ff_scale=1.0,
            static_hold_active=False,
        )
        # 1) 正常态严格不超 HARD_MAX_JOINT_SPEED（J2/J3 <= 2.0 rad/s）
        sh = shield.filter(malicious_cmd, _make_robot_state(q), dt_s=0.01)
        self.assertTrue(np.all(np.abs(sh.dq_cmd_safe) <= np.asarray(HARD_MAX_JOINT_SPEED) + 1e-6))
        self.assertLessEqual(abs(sh.dq_cmd_safe[1]), 2.0 + 1e-6)
        self.assertLessEqual(abs(sh.dq_cmd_safe[2]), 2.0 + 1e-6)

        # 2) M6 锁存态强制零速度
        sh_m6 = shield.filter(
            malicious_cmd,
            _make_robot_state(q, supervisor_latched=True, supervisor_reason="torque"),
            dt_s=0.01,
        )
        self.assertEqual(float(np.linalg.norm(sh_m6.dq_cmd_safe)), 0.0)
        self.assertEqual(float(np.linalg.norm(sh_m6.dq_ff_safe)), 0.0)

        # 3) 越界 >5cm 触发 RETURN_HOME
        far_cmd = replace(malicious_cmd, tcp_pos_cmd=np.array([0.50, 0.0, 0.30]))
        sh_hard = shield.filter(far_cmd, _make_robot_state(q), dt_s=0.01)
        self.assertTrue(sh_hard.trigger_return_home)
        self.assertEqual(float(np.linalg.norm(sh_hard.dq_cmd_safe)), 0.0)

        # 4) NaN fail-safe 归零
        nan_cmd = replace(malicious_cmd, v_tcp_world=np.array([float("nan"), 0.0, 0.0]))
        sh_nan = shield.filter(nan_cmd, _make_robot_state(q), dt_s=0.01)
        self.assertEqual(float(np.linalg.norm(sh_nan.dq_cmd_safe)), 0.0)


class DigitalTwinAndEndToEndTests(unittest.TestCase):
    """验证数字孪生运动学精度、闭环跟踪稳停与真实日志端到端回放。"""

    def test_pure_numpy_urdf_matches_home_pose_submillimeter(self):
        """验证纯 NumPy B601-RS URDF FK 与实机示教 home_pose.json 偏差 < 0.5 mm。"""
        kin = _PureNumpyURDFKinematics(os.path.join(ROOT, "config", "joint_offset.json"))
        with open(os.path.join(ROOT, "config", "home_pose.json"), encoding="utf-8") as stream:
            hp = json.load(stream)
        p, _, J = kin.fk_and_jacobian(np.asarray(hp["home_joint_pos"], dtype=np.float64))
        err_mm = float(np.linalg.norm(p - np.asarray(hp["tcp_position"]))) * 1000.0
        self.assertLess(err_mm, 0.5)
        self.assertEqual(J.shape, (6, 6))
        tau = kin.gravity_torque(np.asarray(hp["home_joint_pos"], dtype=np.float64))
        self.assertGreater(tau[2], 6.0)   # J3 承重正力矩
        self.assertGreater(tau[3], 1.8)   # J4 承重正力矩

    def test_closed_loop_pipeline_tracks_and_settles_to_holding(self):
        """验证 v14 四层流水线 + 数字孪生闭环：跟踪手掌平移 8cm 后停手稳停至 HOLDING。"""
        cfg = load_v14_config(os.path.join(ROOT, "config", "v14.yaml"))
        with open(os.path.join(ROOT, "config", "hand_eye_calibration.json"), encoding="utf-8") as s:
            T_tc = np.asarray(json.load(s)["T_tcp_camera"], dtype=np.float64)
        with open(os.path.join(ROOT, "config", "home_pose.json"), encoding="utf-8") as s:
            q_home = np.asarray(json.load(s)["home_joint_pos"], dtype=np.float64)

        backend = DigitalTwinBackend(
            sim_cfg=cfg.sim, q_init=q_home, use_pinocchio=False,
            joint_offset_path=os.path.join(ROOT, "config", "joint_offset.json"),
            rng_seed=7,
        )
        pipe = ServoV14Pipeline(cfg, backend, T_tc)
        pipe.lock_current_orientation(q_home)

        p0, R0 = backend.fk_only(q_home)
        cam_p0 = p0 + R0 @ T_tc[:3, 3]
        cam_R0 = R0 @ T_tc[:3, :3]
        hand_start = cam_p0 + cam_R0[:, 2] * cfg.pbvs.distance
        # 手掌平滑沿世界 Z 上移 8cm，随后保持静止 1.2s
        t = 0.0
        modes = []
        last_telem = None
        for k in range(250):
            t += 0.01
            frac = min(1.0, k / 80.0)
            hand_w = hand_start + np.array([0.0, 0.0, 0.08 * frac])
            rs = backend.get_state(t)
            tp, tr = backend.fk_only(rs.q_act)
            cp = tp + tr @ T_tc[:3, 3]
            cR = tr @ T_tc[:3, :3]
            p_cam = cR.T @ (hand_w - cp)
            obs = Observation(k + 1, t, t, True, False, p_cam)
            _, telem = pipe.step(obs, rs, 0.01, following_enabled=True, apply_to_backend=True)
            modes.append(telem.mode)
            last_telem = telem

        self.assertIn(FSMMode.DOCKING, modes)
        self.assertEqual(last_telem.mode, FSMMode.HOLDING)
        self.assertEqual(last_telem.joint_speed_max, 0.0)
        self.assertLess(last_telem.error_norm_m, 0.03)
        self.assertFalse(backend.supervisor_latched)

    def test_pinch_mapper_freezes_while_moving_and_cli_overrides(self):
        """验证 `PinchGripperMapper` 运动冻结逻辑与 `palm_follow_v14` CLI 安全红线校验。"""
        from palm_follow_v14 import PinchGripperMapper, apply_cli_overrides, parse_args

        mapper = PinchGripperMapper({
            "pinch": {"closed_px": 20, "open_px": 200, "deadband_deg": 2.0},
            "gripper": {"enabled": True, "hold_while_moving": True,
                        "move_speed_threshold": 0.05, "closed_deg": 5.0, "open_deg": 285.0},
        })
        mapper.seed(100.0)
        # 手臂运动中必须冻结
        self.assertIsNone(mapper.update(180.0, 1.0, arm_joint_speed=0.30))
        self.assertTrue(mapper.frozen_by_motion)
        # 手臂停稳后立即解除冻结并输出合法角度
        deg = mapper.update(180.0, 1.1, arm_joint_speed=0.01)
        self.assertIsNotNone(deg)
        self.assertFalse(mapper.frozen_by_motion)
        self.assertGreater(deg, 200.0)

        # CLI 超过安全红线必须抛出 ValueError
        cfg = load_v14_config(os.path.join(ROOT, "config", "v14.yaml"))
        bad_args = parse_args(["--max-vel", "0.95"])
        with self.assertRaises(ValueError):
            apply_cli_overrides(cfg, bad_args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
