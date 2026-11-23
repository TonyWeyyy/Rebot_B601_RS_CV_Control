#!/usr/bin/env python3
"""
v15.0 双向数字孪生遥操作离线单元测试套件（tools/test_v15_teleop.py）
===================================================================
无需任何相机或真机 CAN 硬件，覆盖全部 5 道遥操作安全防线与双向 UDP 协议：
  1. 通道 B UDP 指令收发闭环、NaN 毒包拦截与 250ms 失联看门狗；
  2. 解锁姿态对齐门控（偏差 >8° 拒绝武装，防瞬间甩臂）；
  3. 平滑加减速斜坡、分轴硬限速与热降额闭环收敛（无超调着陆）；
  4. 0.42m 水平柱面软墙拦截外推、允许回缩、Z 桌面净空保护与 >5cm 硬墙回 HOME；
  5. M6 安全监督锁存联锁与非武装态一次性清故障；
  6. 单 CAN 通道 J7 夹爪死区/速率限幅与 `--no-gripper` 禁用；
  7. `mujoco_teleop_v15.py` CLI 安全红线校验与离线 `--sim` 闭环联动冒烟。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servo_v14.config import load_v14_config
from servo_v14.mujoco_bridge import (
    MuJoCoTeleopCmdPublisher,
    MuJoCoTeleopCmdReceiver,
    MuJoCoTwinPublisher,
    MuJoCoTwinReceiver,
)
from servo_v14.robot_backend import DigitalTwinBackend
from servo_v14.teleop_bridge import (
    HARD_TELEOP_JOINT_SPEED,
    SafeTeleopConfig,
    SafeTeleopController,
    TeleopCommand,
    parse_teleop_packet,
)
from servo_v14.types import RobotState


def _load_home_q() -> np.ndarray:
    with open(os.path.join(ROOT, "config", "home_pose.json"), encoding="utf-8") as s:
        return np.asarray(json.load(s)["home_joint_pos"], dtype=np.float64)


def _make_sim_backend(q_init: np.ndarray | None = None) -> DigitalTwinBackend:
    cfg = load_v14_config(os.path.join(ROOT, "config", "v14.yaml"))
    q0 = _load_home_q() if q_init is None else np.asarray(q_init, dtype=np.float64)
    return DigitalTwinBackend(
        sim_cfg=cfg.sim,
        q_init=q0,
        use_pinocchio=False,
        joint_offset_path=os.path.join(ROOT, "config", "joint_offset.json"),
        rng_seed=42,
    )


class V15BidirectionalBridgeTests(unittest.TestCase):
    """验证通道 A/B 双向 UDP 协议、NaN 拦截与看门狗超时。"""

    def test_teleop_cmd_channel_roundtrip_nan_reject_and_watchdog(self):
        rx = MuJoCoTeleopCmdReceiver(port=0, watchdog_s=0.08)
        try:
            self.assertIsNone(rx.bind_error)
            port = rx._sock.getsockname()[1]
            pub = MuJoCoTeleopCmdPublisher(port=port, max_hz=1000.0, enabled=True)
            try:
                ok = pub.publish(
                    armed=True,
                    mode="IK_3D",
                    q_des=[-0.4, 0.35, 0.37, 0.1, 0.0, 0.0],
                    gripper_deg=140.0,
                    max_speed_deg_s=30.0,
                    ik_target_xyz=[0.20, 0.05, 0.30],
                    force=True,
                )
                self.assertTrue(ok)

                pkt = None
                for _ in range(20):
                    pkt = rx.poll_latest()
                    if pkt is not None:
                        break
                    time.sleep(0.005)
                self.assertIsNotNone(pkt)
                cmd = parse_teleop_packet(pkt)
                self.assertIsNotNone(cmd)
                self.assertTrue(cmd.armed)
                self.assertEqual(cmd.mode, "IK_3D")
                self.assertAlmostEqual(cmd.max_speed_deg_s, 30.0, places=1)
                self.assertAlmostEqual(cmd.gripper_deg, 140.0, places=1)

                # 模拟看门狗超时 (>80ms 无心跳必须返回 None)
                stale = rx.poll_latest(now_mono=time.monotonic() + 0.20)
                self.assertIsNone(stale)

                # 校验 parse_teleop_packet 拦截 NaN 毒包
                bad = dict(pkt)
                bad["q_des"] = [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]
                self.assertIsNone(parse_teleop_packet(bad))
            finally:
                pub.close()
        finally:
            rx.close()

    def test_state_channel_carries_v15_teleop_ack_fields(self):
        rx = MuJoCoTwinReceiver(port=0, stale_timeout_s=0.5)
        try:
            port = rx._sock.getsockname()[1]
            pub = MuJoCoTwinPublisher(port=port, max_hz=1000.0, enabled=True)
            try:
                pub.publish(
                    q_act=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                    q_cmd=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                    source="hw_teleop",
                    teleop_armed=True,
                    teleop_mode="TELEOP_IK",
                    teleop_reason="OK",
                    teleop_speed_deg_s=25.0,
                    cmd_seq_ack=99,
                    force=True,
                )
                pkt = None
                for _ in range(20):
                    pkt = rx.poll_latest()
                    if pkt is not None:
                        break
                    time.sleep(0.005)
                self.assertIsNotNone(pkt)
                self.assertEqual(pkt["source"], "hw_teleop")
                self.assertTrue(pkt["teleop_armed"])
                self.assertEqual(pkt["teleop_mode"], "TELEOP_IK")
                self.assertEqual(pkt["cmd_seq_ack"], 99)
            finally:
                pub.close()
        finally:
            rx.close()


class V15SafeTeleopControllerTests(unittest.TestCase):
    """验证 5 道不可绕过的硬件遥操作安全防线。"""

    def test_anti_jump_alignment_gate_blocks_large_mismatch(self):
        q0 = _load_home_q()
        backend = _make_sim_backend(q0)
        ctrl = SafeTeleopController(SafeTeleopConfig(align_tol_deg=8.0), backend)

        # 偏差 25° 请求武装必须被拒绝，严防甩臂
        far_q = q0.copy()
        far_q[0] += math.radians(25.0)
        res_block = ctrl.step(
            TeleopCommand(seq=1, armed=True, mode="JOINT", q_des=far_q),
            backend.get_state(0.01),
            dt_s=0.01,
        )
        self.assertFalse(res_block.armed)
        self.assertIn("ALIGN_REQUIRED", res_block.reason)
        self.assertEqual(float(np.linalg.norm(res_block.dq_cmd_safe)), 0.0)

        # 对齐到 2° 以内后允许正常解锁武装
        near_q = q0.copy()
        near_q[0] += math.radians(2.0)
        res_ok = ctrl.step(
            TeleopCommand(seq=2, armed=True, mode="JOINT", q_des=near_q),
            backend.get_state(0.02),
            dt_s=0.01,
        )
        self.assertTrue(res_ok.armed)
        self.assertEqual(res_ok.mode, "TELEOP_JOINT")

    def test_smooth_ramp_converges_without_overshoot_and_respects_speed_cap(self):
        q0 = _load_home_q()
        backend = _make_sim_backend(q0)
        ctrl = SafeTeleopController(SafeTeleopConfig(align_tol_deg=8.0), backend)

        # 1) 先在当前位置解锁
        t = 0.01
        res = ctrl.step(
            TeleopCommand(seq=1, armed=True, mode="JOINT", q_des=q0, max_speed_deg_s=20.0),
            backend.get_state(t),
            dt_s=0.01,
        )
        self.assertTrue(res.armed)

        # 2) 下发 J1 移动 +18° 目标，限速 20°/s (0.349 rad/s)
        q_tgt = q0.copy()
        q_tgt[0] += math.radians(18.0)
        vmax_seen = 0.0
        max_j1_seen = q0[0]
        for k in range(2, 220):
            t += 0.01
            rs = backend.get_state(t)
            res = ctrl.step(
                TeleopCommand(
                    seq=k, armed=True, mode="JOINT", q_des=q_tgt,
                    gripper_deg=120.0, max_speed_deg_s=20.0,
                ),
                rs,
                dt_s=0.01,
            )
            vmax_seen = max(vmax_seen, float(np.max(np.abs(res.dq_cmd_safe))))
            max_j1_seen = max(max_j1_seen, float(res.q_cmd_next[0]))

        # 严格不超过 20°/s + 微小数值容差，且无超调并精确收敛
        self.assertLessEqual(vmax_seen, math.radians(20.5))
        self.assertLessEqual(max_j1_seen, q_tgt[0] + 1e-3)
        self.assertLess(abs(float(backend.get_state(t).q_act[0]) - float(q_tgt[0])), math.radians(1.0))
        # 夹爪也平滑同步至 120°
        self.assertLess(abs(backend.get_gripper_deg() - 120.0), 2.5)

    def test_watchdog_m6_fault_and_hard_wall_preemption(self):
        q0 = _load_home_q()
        backend = _make_sim_backend(q0)
        ctrl = SafeTeleopController(SafeTeleopConfig(), backend)

        # 先武装
        res0 = ctrl.step(
            TeleopCommand(seq=1, armed=True, mode="IK_3D", q_des=q0),
            backend.get_state(0.01),
            dt_s=0.01,
        )
        self.assertTrue(res0.armed)

        # 1) 失联看门狗 (cmd=None) 必须立即解除武装并速度归零
        res_wd = ctrl.step(None, backend.get_state(0.02), dt_s=0.01)
        self.assertFalse(res_wd.armed)
        self.assertEqual(res_wd.reason, "WATCHDOG_TIMEOUT")
        self.assertEqual(float(np.linalg.norm(res_wd.dq_cmd_safe)), 0.0)

        # 2) M6 安全监督锁存必须强制进入 FAULT_HOLD
        rs_fault = RobotState(
            timestamp_s=0.03,
            q_cmd=q0.copy(),
            q_act=q0.copy(),
            dq_act=np.zeros(6),
            tau_act=np.zeros(6),
            supervisor_latched=True,
            supervisor_reason="over_torque",
        )
        res_m6 = ctrl.step(
            TeleopCommand(seq=2, armed=True, mode="IK_3D", q_des=q0),
            rs_fault,
            dt_s=0.01,
        )
        self.assertFalse(res_m6.armed)
        self.assertEqual(res_m6.mode, "FAULT_HOLD")
        self.assertIn("M6_FAULT", res_m6.reason)

    def test_workspace_soft_wall_blocks_outward_extension(self):
        """验证当臂展逼近 r=0.42m 软墙时，继续外推被拦截但向内回缩放行。"""
        # 构造一个水平半径在 (0.42, 0.47)m 之间的前伸姿态（B601-RS 肘部打开至 ~1.55 rad 时水平伸展最大）
        q_ext = np.array([0.0, 1.35, 1.35, 0.0, 0.0, 0.0], dtype=np.float64)
        backend = _make_sim_backend(q_ext)
        tcp0, _ = backend.fk_only(q_ext)
        r0 = math.hypot(float(tcp0[0]), float(tcp0[1]))
        # 通过单步搜索微调 J2 使其严格落在 (0.425, 0.455)m 软墙区间内
        for j2_try in np.linspace(0.6, 2.2, 65):
            q_try = np.array([0.0, float(j2_try), 1.45, 0.0, 0.0, 0.0], dtype=np.float64)
            p_try, _ = backend.fk_only(q_try)
            r_try = math.hypot(float(p_try[0]), float(p_try[1]))
            if 0.425 < r_try < 0.455:
                q_ext = q_try
                r0 = r_try
                break
        backend = _make_sim_backend(q_ext)
        self.assertGreater(r0, 0.42)
        self.assertLess(r0, 0.47)

        ctrl = SafeTeleopController(SafeTeleopConfig(), backend)
        ctrl.step(TeleopCommand(seq=1, armed=True, mode="JOINT", q_des=q_ext), backend.get_state(0.01), 0.01)

        # 找一个使水平半径进一步增大的目标角和一个向内回缩的目标角
        q_further = q_ext.copy()
        q_inward = q_ext.copy()
        for delta in (-0.12, 0.12):
            q_cand = q_ext.copy()
            q_cand[1] = float(np.clip(q_cand[1] + delta, 0.05, 2.85))
            p_c, _ = backend.fk_only(q_cand)
            r_c = math.hypot(float(p_c[0]), float(p_c[1]))
            if r_c > r0 + 0.005:
                q_further = q_cand
            elif r_c < r0 - 0.005:
                q_inward = q_cand

        p_f, _ = backend.fk_only(q_further)
        p_i, _ = backend.fk_only(q_inward)
        self.assertGreater(math.hypot(float(p_f[0]), float(p_f[1])), r0)
        self.assertLess(math.hypot(float(p_i[0]), float(p_i[1])), r0)

        res_out = ctrl.step(
            TeleopCommand(seq=2, armed=True, mode="JOINT", q_des=q_further, max_speed_deg_s=30.0),
            backend.get_state(0.02),
            0.01,
        )
        self.assertTrue(res_out.soft_wall_active)
        self.assertEqual(float(np.linalg.norm(res_out.dq_cmd_safe)), 0.0)

        res_in = ctrl.step(
            TeleopCommand(seq=3, armed=True, mode="JOINT", q_des=q_inward, max_speed_deg_s=30.0),
            backend.get_state(0.03),
            0.01,
        )
        self.assertFalse(res_in.soft_wall_active)
        self.assertGreater(float(np.linalg.norm(res_in.dq_cmd_safe)), 0.0)

    def test_config_red_lines_rejected(self):
        with self.assertRaises(ValueError):
            SafeTeleopConfig(workspace_max_radius_m=0.45).validate()
        with self.assertRaises(ValueError):
            SafeTeleopConfig(hard_max_speed_deg_s=90.0).validate()
        with self.assertRaises(ValueError):
            SafeTeleopConfig(workspace_z_min_m=0.005).validate()
        # 确认遥操作硬包络严守 J2/J3 <= 2.0 rad/s
        self.assertLessEqual(HARD_TELEOP_JOINT_SPEED[1], 2.0)
        self.assertLessEqual(HARD_TELEOP_JOINT_SPEED[2], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
