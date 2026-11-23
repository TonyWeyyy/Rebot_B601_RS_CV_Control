#!/usr/bin/env python3
"""升级逻辑的离线单元测试；不导入相机、MediaPipe 或机械臂 SDK。"""

import csv
import importlib.util
import os
import sys
import types
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servo_timing import (AdaptiveLeadGate, FastStartGate, SlowFollowGate,
                          CameraTimestampCalibrator, AxisProcessNoiseGate,
                          StaticVelocityHold, DelayedTwistCompensator,
                          PostBrakeSettleGuard, OscillationDamper,
                          CameraPoseHistory, ActualVelocityFilter,
                          compute_reach_schedule,
                          compute_orientation_cruise_weight,
                          compute_weighted_damped_pseudoinverse,
                          consensus_velocity, apply_joint_limit_cushion,
                          compute_roi_box, remap_roi_landmarks,
                          adaptive_feedforward_scale,
                          apply_axis_reversal_damping, scale_joint_velocities,
                          camera_twist_from_tcp, compensate_palm_velocity,
                          compensate_palm_velocity_delayed,
                          camera_point_to_world, world_point_to_camera,
                          scaled_alpha, scaled_decay, horizontal_radial_soft_wall,
                          measurement_age, dynamic_lead_seconds)
from kalman_target import KalmanTarget3D
from servo_diagnostics import (DiagnosticRecorder, ActualFeedbackSampler,
                               feedback_age_ms, DIAG_FIELDS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_servo_log import summarize


class TimingTests(unittest.TestCase):
    def test_nominal_alpha_unchanged(self):
        self.assertAlmostEqual(scaled_alpha(0.48, 0.01), 0.48, places=12)

    def test_double_dt_alpha_is_two_ema_steps(self):
        a = scaled_alpha(0.48, 0.02)
        expected = 1.0 - (1.0 - 0.48) ** 2
        self.assertAlmostEqual(a, expected, places=12)

    def test_decay_uses_actual_dt(self):
        self.assertAlmostEqual(scaled_decay(0.9, 0.02, 0.01), 0.81)

    def test_bad_timestamp_does_not_extrapolate(self):
        self.assertEqual(measurement_age(1.0, 2.0), 0.0)
        self.assertEqual(measurement_age(1.0, float("nan")), 0.0)

    def test_dynamic_lead_is_bounded_and_not_cumulative(self):
        lead = dynamic_lead_seconds(8.0, 0.004, 20.0)
        self.assertAlmostEqual(lead, 0.012)
        # 同一帧在后续重复控制周期仍应得到相同 lead，而不是随帧年龄增长。
        self.assertAlmostEqual(dynamic_lead_seconds(8.0, 0.004, 20.0), lead)
        self.assertAlmostEqual(dynamic_lead_seconds(8.0, 0.100, 20.0), 0.020)


class EyeInHandCompensationTests(unittest.TestCase):
    def test_static_point_cancels_pure_camera_translation(self):
        twist = np.array([0.30, -0.10, 0.05, 0.0, 0.0, 0.0])
        palm = np.array([0.12, -0.07, 0.35])
        observed = -(twist[:3] + np.cross(twist[3:], palm))
        compensated = compensate_palm_velocity(observed, twist, palm)
        np.testing.assert_allclose(compensated, np.zeros(3), atol=1e-12)

    def test_static_point_cancels_pure_camera_rotation(self):
        twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.50])
        palm = np.array([0.20, -0.10, 0.35])
        observed = -(twist[:3] + np.cross(twist[3:], palm))
        compensated = compensate_palm_velocity(observed, twist, palm)
        np.testing.assert_allclose(compensated, np.zeros(3), atol=1e-12)

    def test_tcp_angular_motion_includes_camera_mount_lever_arm(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.05, 0.0, 0.0]
        tcp_twist = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.20])
        camera_twist = camera_twist_from_tcp(tcp_twist, np.eye(3), transform)
        np.testing.assert_allclose(camera_twist,
                                   [0.10, 0.01, 0.0, 0.0, 0.0, 0.20],
                                   atol=1e-12)
        palm = np.zeros(3)
        observed = -(camera_twist[:3] + np.cross(camera_twist[3:], palm))
        np.testing.assert_allclose(
            compensate_palm_velocity(observed, camera_twist, palm),
            np.zeros(3), atol=1e-12)

    def test_true_hand_velocity_survives_compensation(self):
        twist = np.array([0.15, 0.02, -0.03, 0.01, -0.02, 0.03])
        palm = np.array([0.10, 0.05, 0.35])
        hand_velocity = np.array([0.40, -0.20, 0.10])
        observed = hand_velocity - (
            twist[:3] + np.cross(twist[3:], palm))
        compensated = compensate_palm_velocity(observed, twist, palm)
        np.testing.assert_allclose(compensated, hand_velocity, atol=1e-12)

    def test_delayed_camera_twist_history_cancels_lagged_stop_motion(self):
        # 机械臂 40ms 前以 0.42 m/s 冲刺，当前周期刚进死区指令归零 (0.0 m/s)；
        # 滞后 40ms 的视觉 KF 仍看到 -0.40 m/s 相对速度。
        history = [
            np.array([0.42, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.00, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        palm = np.array([0.0, 0.0, 0.40])
        observed = np.array([-0.40, 0.0, 0.0])
        compensated = compensate_palm_velocity_delayed(observed, history, palm)
        self.assertLess(float(np.linalg.norm(compensated)), 0.03)

    def test_camera_world_point_round_trip(self):
        def rot_z(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        transform = np.eye(4)
        transform[:3, :3] = rot_z(0.7)
        transform[:3, 3] = [0.05, -0.02, 0.095]
        tcp_position = np.array([0.20, -0.10, 0.30])
        tcp_rotation = rot_z(-0.4)
        point_world = np.array([0.35, 0.12, 0.28])
        point_camera = world_point_to_camera(
            point_world, tcp_position, tcp_rotation, transform)
        np.testing.assert_allclose(
            camera_point_to_world(point_camera, tcp_position,
                                  tcp_rotation, transform),
            point_world, atol=1e-12)

    def test_invalid_inputs_fall_back_safely(self):
        observed = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_equal(
            compensate_palm_velocity(observed, np.zeros(3), np.zeros(3)),
            observed)
        np.testing.assert_array_equal(
            compensate_palm_velocity(np.array([np.nan, 0.0, 0.0]),
                                     np.zeros(6), np.zeros(3)),
            np.zeros(3))
        np.testing.assert_array_equal(
            camera_twist_from_tcp(np.zeros(5), np.eye(3), np.eye(4)),
            np.zeros(6))
        np.testing.assert_array_equal(
            camera_twist_from_tcp(np.zeros(6), np.eye(3), np.eye(3)),
            np.zeros(6))


class StaticVelocityHoldTests(unittest.TestCase):
    def setUp(self):
        self.gate = StaticVelocityHold(speed_limit=0.025, confirm_frames=3)
        self.low = np.array([0.010, -0.005, 0.0, 0.0, 0.0, 0.0])

    def test_low_speed_tail_zeroes_only_after_confirmation(self):
        for expected in (False, False, True, True):
            result, active = self.gate.update(self.low, True, True)
            self.assertEqual(active, expected)
            if expected:
                np.testing.assert_array_equal(result, np.zeros(6))
            else:
                np.testing.assert_allclose(result, self.low)

    def test_high_speed_docking_keeps_original_decay(self):
        fast = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(5):
            result, active = self.gate.update(fast, True, True)
            self.assertFalse(active)
            np.testing.assert_allclose(result, fast)

    def test_motion_and_deadband_exit_reset_confirmation(self):
        self.gate.update(self.low, True, True)
        self.gate.update(self.low, True, True)
        result, active = self.gate.update(self.low, False, True)
        self.assertFalse(active)
        np.testing.assert_allclose(result, self.low)
        self.assertFalse(self.gate.update(self.low, True, True)[1])
        self.gate.update(self.low, True, True)
        self.assertFalse(self.gate.update(self.low, True, False)[1])
        self.assertFalse(self.gate.update(self.low, True, True)[1])

    def test_disable_preserves_v9_velocity_state(self):
        for _ in range(5):
            result, active = self.gate.update(self.low, True, True, enabled=False)
            self.assertFalse(active)
            np.testing.assert_allclose(result, self.low)

    def test_nonfinite_input_fails_to_zero(self):
        bad = self.low.copy()
        bad[1] = np.nan
        result, active = self.gate.update(bad, True, True)
        self.assertFalse(active)
        np.testing.assert_array_equal(result, np.zeros(6))

    def test_input_array_is_not_modified(self):
        original = self.low.copy()
        self.gate.update(self.low, True, True)
        np.testing.assert_allclose(self.low, original)


class FastStartTests(unittest.TestCase):
    def test_static_noise_does_not_unlock(self):
        gate = FastStartGate(True, 2, 0.030, 0.8)
        gate.reset(np.array([0.0, 0.0, 0.35]))
        for p in ([0.002, 0.0, 0.35], [-0.002, 0.001, 0.351],
                  [0.001, -0.001, 0.349], [0.0, 0.0, 0.35]):
            self.assertFalse(gate.update(np.array(p)))

    def test_coherent_motion_unlocks_on_second_new_frame(self):
        gate = FastStartGate(True, 2, 0.030, 0.8)
        gate.reset(np.array([0.0, 0.0, 0.35]))
        self.assertFalse(gate.update(np.array([0.032, 0.0, 0.35])))
        self.assertTrue(gate.update(np.array([0.038, 0.0, 0.35])))

    def test_reversal_cannot_inherit_confirmation(self):
        gate = FastStartGate(True, 2, 0.030, 0.8)
        gate.reset(np.array([0.0, 0.0, 0.35]))
        self.assertFalse(gate.update(np.array([0.040, 0.0, 0.35])))
        self.assertFalse(gate.update(np.array([-0.040, 0.0, 0.35])))
        self.assertFalse(gate.update(np.array([-0.048, 0.0, 0.35])))


class SlowFollowTests(unittest.TestCase):
    def test_static_noise_never_activates(self):
        gate = SlowFollowGate()
        gate.reset(np.zeros(3))
        for point in ([0.002, 0.0, 0.0], [-0.002, 0.001, 0.0],
                      [0.001, -0.001, 0.002], [0.0, 0.0, 0.0]):
            self.assertFalse(gate.update(np.array(point)))

    def test_slow_coherent_drag_activates_without_fast_start(self):
        gate = SlowFollowGate(distance=0.014, confirm_frames=3)
        gate.reset(np.zeros(3))
        states = [gate.update(np.array([x, 0.0, 0.0]))
                  for x in (0.006, 0.012, 0.018, 0.024, 0.030)]
        self.assertTrue(states[-1])
        self.assertFalse(any(states[:2]))

    def test_reversal_restarts_confirmation(self):
        gate = SlowFollowGate(distance=0.014, confirm_frames=2)
        gate.reset(np.zeros(3))
        gate.update(np.array([0.02, 0.0, 0.0]))
        self.assertTrue(gate.update(np.array([0.03, 0.0, 0.0])))
        self.assertFalse(gate.update(np.array([-0.02, 0.0, 0.0])))
        self.assertFalse(gate.active)

    def test_disable_and_nonfinite_fail_safe(self):
        gate = SlowFollowGate()
        self.assertFalse(gate.update(np.array([0.1, 0.0, 0.0]), enabled=False))
        self.assertFalse(gate.update(np.array([np.nan, 0.0, 0.0])))
        self.assertFalse(gate.active)


class KalmanAxisNoiseTests(unittest.TestCase):
    def test_scalar_behavior_is_backward_compatible(self):
        scalar = KalmanTarget3D(q=1.5, r=0.004)
        vector = KalmanTarget3D(q=1.5, r=0.004, r_xy=None, r_z=None)
        for dt, meas in ((0.0, [0.0, 0.0, 0.35]),
                         (0.016, [0.01, -0.005, 0.36]),
                         (0.020, [0.02, -0.008, 0.35])):
            scalar.update(np.array(meas), dt)
            vector.update(np.array(meas), dt)
            np.testing.assert_allclose(scalar.x, vector.x, atol=1e-12)

    def test_xy_noise_is_independent_from_depth(self):
        kf = KalmanTarget3D(q=1.5, r=0.004, r_xy=0.0025, r_z=0.004)
        np.testing.assert_allclose(kf.r_vector, [0.0025, 0.0025, 0.004])
        kf.update(np.array([0.0, 0.0, 0.35]), 0.0)
        kf.update(np.array([0.02, 0.0, 0.37]), 0.016)
        self.assertGreater(abs(kf.x[0]), abs(kf.x[2] - 0.35))
        with self.assertRaises(ValueError):
            KalmanTarget3D(r_xy=0.0)


    def test_scalar_and_vector_process_noise_are_equivalent(self):
        scalar = KalmanTarget3D(q=1.5)
        vector = KalmanTarget3D(q=1.5)
        scalar.q_scale = 1.4
        vector.q_scale = np.array([1.4, 1.4, 1.4])
        f_scalar, q_scalar = scalar._FQ(0.016)
        f_vector, q_vector = vector._FQ(0.016)
        np.testing.assert_allclose(f_scalar, f_vector, atol=1e-12)
        np.testing.assert_allclose(q_scalar, q_vector, atol=1e-12)
        vector.q_scale = np.array([1.6, 1.2, 1.0])
        _, q_vector = vector._FQ(0.016)
        self.assertGreater(q_vector[0, 0], q_vector[1, 1])
        self.assertGreater(q_vector[1, 1], q_vector[2, 2])


class AxisProcessNoiseGateTests(unittest.TestCase):
    def test_static_noise_never_amplifies_q(self):
        gate = AxisProcessNoiseGate()
        for _ in range(20):
            scales = gate.update(np.array([0.02, -0.02, 0.03]), 0.016)
        np.testing.assert_array_equal(scales, np.ones(3))

    def test_coherent_xy_innovation_amplifies_after_confirmation(self):
        gate = AxisProcessNoiseGate(confirm_frames=2)
        first = gate.update(np.array([0.08, 0.07, 0.20]), 0.016)
        second = gate.update(np.array([0.12, 0.09, 0.20]), 0.016)
        np.testing.assert_array_equal(first, np.ones(3))
        self.assertGreater(second[0], 1.0)
        self.assertGreater(second[1], 1.0)
        self.assertEqual(second[2], 1.0)  # 深度噪声默认绝不放大

    def test_reversal_must_confirm_again(self):
        gate = AxisProcessNoiseGate(confirm_frames=2)
        gate.update(np.array([0.10, 0.0, 0.0]), 0.016)
        before = gate.update(np.array([0.12, 0.0, 0.0]), 0.016)
        reversed_once = gate.update(np.array([-0.12, 0.0, 0.0]), 0.016)
        self.assertEqual(gate.counts[0], 1)
        self.assertEqual(gate.signs[0], -1)
        self.assertLess(reversed_once[0], before[0])
        gate.update(np.array([-0.12, 0.0, 0.0]), 0.016)
        self.assertEqual(gate.counts[0], 2)

    def test_invalid_input_and_reset_restore_baseline_immediately(self):
        gate = AxisProcessNoiseGate(confirm_frames=1)
        gate.update(np.array([0.12, 0.12, 0.0]), 0.10)
        self.assertTrue(np.any(gate.scales > 1.0))
        scales = gate.update(np.array([np.nan, 0.0, 0.0]), 0.016)
        np.testing.assert_array_equal(scales, np.ones(3))
        np.testing.assert_array_equal(gate.counts, np.zeros(3))
        gate.update(np.array([0.12, 0.12, 0.0]), 0.10)
        gate.reset()
        np.testing.assert_array_equal(gate.scales, np.ones(3))


class AdaptiveDockingTests(unittest.TestCase):
    def setUp(self):
        self.error = np.array([0.06, 0.0, 0.0])
        self.old_scale = self.error[0] / 0.07

    def test_slow_approach_and_disable_match_v8_exactly(self):
        slow, taper, approach = adaptive_feedforward_scale(
            self.error, np.array([-0.05, 0.0, 0.0]), 0.07)
        disabled, _, _ = adaptive_feedforward_scale(
            self.error, np.array([-0.60, 0.0, 0.0]), 0.07, enabled=False)
        self.assertAlmostEqual(slow, self.old_scale)
        self.assertAlmostEqual(disabled, self.old_scale)
        self.assertEqual(taper, 0.07)
        self.assertAlmostEqual(approach, 0.05)

    def test_fast_approach_only_reduces_feedforward(self):
        scale, taper, approach = adaptive_feedforward_scale(
            self.error, np.array([-0.60, 0.0, 0.0]), 0.07)
        self.assertAlmostEqual(approach, 0.60)
        self.assertEqual(taper, 0.12)
        self.assertLess(scale, self.old_scale)
        self.assertAlmostEqual(scale, 0.50)

    def test_moving_away_has_no_extra_braking(self):
        scale, taper, approach = adaptive_feedforward_scale(
            self.error, np.array([0.60, 0.0, 0.0]), 0.07)
        self.assertAlmostEqual(scale, self.old_scale)
        self.assertEqual(taper, 0.07)
        self.assertEqual(approach, 0.0)

    def test_zero_error_and_nonfinite_input_fail_safe(self):
        scale, _, approach = adaptive_feedforward_scale(
            np.zeros(3), np.array([-0.6, 0.0, 0.0]), 0.07)
        self.assertEqual((scale, approach), (0.0, 0.0))
        scale, _, _ = adaptive_feedforward_scale(
            np.array([np.nan, 0.0, 0.0]), np.zeros(3), 0.07)
        self.assertEqual(scale, 0.0)


class ActualFeedbackTests(unittest.TestCase):
    class Robot:
        def __init__(self):
            self.actual = np.linspace(0.1, 0.6, 6)
            self.command = np.linspace(0.2, 0.7, 6)
            self.fail = False

        def get_current_pos(self):
            if self.fail:
                raise RuntimeError("假 CAN 故障")
            return self.actual.copy()

        def get_command_pos(self):
            return self.command.copy()

    def test_sample_pairs_actual_with_command_at_same_point(self):
        robot = self.Robot()
        sampler = ActualFeedbackSampler(robot, hz=1.0)
        self.assertTrue(sampler.sample_once())
        snapshot = sampler.snapshot()
        np.testing.assert_array_equal(snapshot["actual"], robot.actual)
        np.testing.assert_array_equal(snapshot["command"], robot.command)
        self.assertGreaterEqual(snapshot["read_ms"], 0.0)
        self.assertTrue(np.isfinite(snapshot["time"]))

    def test_nonfinite_and_failed_reads_invalidate_snapshot(self):
        robot = self.Robot()
        sampler = ActualFeedbackSampler(robot)
        self.assertTrue(sampler.sample_once())
        robot.actual[2] = np.nan
        self.assertFalse(sampler.sample_once())
        self.assertIsNone(sampler.snapshot())
        robot.actual[2] = 0.3
        robot.fail = True
        self.assertFalse(sampler.sample_once())
        self.assertFalse(sampler.sample_once())
        self.assertTrue(sampler._stop.is_set())  # 连续三次故障永久停用诊断

    def test_frequency_and_freshness_guards(self):
        with self.assertRaises(ValueError):
            ActualFeedbackSampler(self.Robot(), hz=0.0)
        with self.assertRaises(ValueError):
            ActualFeedbackSampler(self.Robot(), hz=2.1)
        snapshot = {"time": 10.0}
        self.assertAlmostEqual(feedback_age_ms(snapshot, 10.2, 1.0), 200.0)
        self.assertIsNone(feedback_age_ms(snapshot, 9.9, 1.0))   # 未来时间异常
        self.assertIsNone(feedback_age_ms(snapshot, 13.1, 1.0))  # 过期
        self.assertIsNone(feedback_age_ms(snapshot, 10.2, 0.0))
        self.assertIsNone(feedback_age_ms(None, 10.2, 1.0))


class CameraTimestampInterfaceTests(unittest.TestCase):
    def test_sdk_capability_and_unit_adapters(self):
        sdk = types.ModuleType("pyorbbecsdk")
        for name in ("Pipeline", "Config", "OBSensorType", "OBAlignMode", "OBFormat"):
            setattr(sdk, name, object)
        sdk.OBError = RuntimeError
        previous = sys.modules.get("pyorbbecsdk")
        sys.modules["pyorbbecsdk"] = sdk
        try:
            path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "gemini2_camera.py")
            spec = importlib.util.spec_from_file_location("camera_timestamp_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            class UsFrame:
                def get_timestamp_us(self):
                    return 123456
            class MsFrame:
                def get_time_stamp(self):
                    return 1234
            value, _ = module.Gemini2Camera._read_timestamp_seconds(UsFrame())
            self.assertAlmostEqual(value, 0.123456)
            value, _ = module.Gemini2Camera._read_timestamp_seconds(MsFrame())
            self.assertAlmostEqual(value, 1.234)
            self.assertEqual(module.Gemini2Camera._read_timestamp_seconds(object()),
                             (None, ""))
        finally:
            if previous is None:
                sys.modules.pop("pyorbbecsdk", None)
            else:
                sys.modules["pyorbbecsdk"] = previous


class CameraTimestampTests(unittest.TestCase):
    def test_mapping_and_clock_drift(self):
        calibrator = CameraTimestampCalibrator()
        mapped, used = calibrator.update(100.0, 10.0)
        self.assertTrue(used)
        self.assertAlmostEqual(mapped, 10.0)
        mapped, used = calibrator.update(100.016, 10.020)
        self.assertTrue(used)
        self.assertLessEqual(mapped, 10.020)
        self.assertGreaterEqual(mapped, 10.015)

    def test_backwards_and_jump_fall_back_to_host(self):
        calibrator = CameraTimestampCalibrator()
        calibrator.update(100.0, 10.0)
        mapped, used = calibrator.update(99.0, 10.016)
        self.assertFalse(used)
        self.assertEqual(mapped, 10.016)
        _, used = calibrator.update(100.016, 10.032)
        self.assertTrue(used)  # 重置后的首个有效样本重新锚定
        mapped, used = calibrator.update(110.0, 10.048)
        self.assertFalse(used)
        self.assertEqual(mapped, 10.048)
        _, used = calibrator.update(110.016, 10.064)
        self.assertTrue(used)

    def test_bad_api_values_and_transport_anomaly(self):
        calibrator = CameraTimestampCalibrator()
        for raw in (None, float("nan"), -1.0, "bad"):
            mapped, used = calibrator.update(raw, 20.0)
            self.assertFalse(used)
            self.assertEqual(mapped, 20.0)
        calibrator.update(200.0, 30.0)
        mapped, used = calibrator.update(200.016, 30.30)
        self.assertFalse(used)
        self.assertEqual(mapped, 30.30)


class DiagnosticTests(unittest.TestCase):
    def test_bounded_queue_flush_and_offline_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "servo.csv")
            recorder = DiagnosticRecorder(path, queue_size=128).start()
            for index in range(5000):
                recorder.log({
                    "host_time": index * 0.01, "following": 1, "visible": 1,
                    "ff_moving": int(index < 100), "hold_latch": int(index >= 100),
                    "joint_speed": 0.2 if index < 100 else 0.0,
                    "error_norm": 0.05, "obs_age_ms": 10,
                    "processing_ms": 4, "ctrl_work_ms": 1,
                })
            recorder.close()
            self.assertEqual(recorder.written + recorder.dropped, 5000)
            self.assertGreater(recorder.dropped, 0)
            with open(path, newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, DIAG_FIELDS)
            result = summarize(path)
            self.assertEqual(result["samples"], recorder.written)
            self.assertIn("static_nonzero_ratio", result)


    def test_feedback_summary_deduplicates_repeated_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "feedback.csv")
            rows = []
            for index, (stamp, error) in enumerate(((10.0, 0.01), (10.0, 0.01),
                                                    (11.0, 0.03), (11.0, 0.03))):
                rows.append({"host_time": str(index * 0.01), "fb_valid": 1,
                             "fb_time": stamp, "fb_read_ms": 5.0,
                             "tracking_err_max": error})
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = summarize(path)
            self.assertEqual(result["feedback_valid_rows"], 4)
            self.assertEqual(result["feedback_samples"], 2)
            self.assertAlmostEqual(result["feedback_tracking_median_rad"], 0.01)
            self.assertAlmostEqual(result["feedback_tracking_max_rad"], 0.03)


class WorkspaceTests(unittest.TestCase):
    def test_outward_radial_velocity_is_removed(self):
        velocity, touched = horizontal_radial_soft_wall(
            [0.42, 0.0, 0.0], [0.20, 0.05, 0.0], 0.42)
        self.assertTrue(touched)
        np.testing.assert_allclose(velocity, [0.0, 0.05, 0.0], atol=1e-12)

    def test_inward_and_tangential_motion_are_not_blocked(self):
        inward, touched = horizontal_radial_soft_wall(
            [0.42, 0.0, 0.0], [-0.20, 0.0, 0.0], 0.42)
        self.assertFalse(touched)
        np.testing.assert_allclose(inward, [-0.20, 0.0, 0.0])
        tangent, touched = horizontal_radial_soft_wall(
            [0.42, 0.0, 0.0], [0.0, 0.20, 0.0], 0.42)
        self.assertFalse(touched)
        np.testing.assert_allclose(tangent, [0.0, 0.20, 0.0])

    def test_height_does_not_consume_horizontal_workspace(self):
        velocity, touched = horizontal_radial_soft_wall(
            [0.0, 0.0, 0.60], [0.20, -0.10, 0.15], 0.42)
        self.assertFalse(touched)
        np.testing.assert_allclose(velocity, [0.20, -0.10, 0.15])

    def test_vertical_motion_survives_at_horizontal_boundary(self):
        velocity, touched = horizontal_radial_soft_wall(
            [0.42, 0.0, 0.60], [0.20, 0.05, 0.15], 0.42)
        self.assertTrue(touched)
        np.testing.assert_allclose(velocity, [0.0, 0.05, 0.15], atol=1e-12)

    def test_nonfinite_pose_fails_to_zero(self):
        velocity, touched = horizontal_radial_soft_wall(
            [np.nan, 0.0, 0.0], [0.2, 0.0, 0.0], 0.42)
        self.assertTrue(touched)
        np.testing.assert_array_equal(velocity, np.zeros(3))


class AdaptiveLeadTests(unittest.TestCase):
    def setUp(self):
        self.gate = AdaptiveLeadGate(
            speed_off=0.06, speed_on=0.13, near_dist=0.022,
            full_dist=0.045, direction_full_cos=0.50,
            attack_tau=0.025, release_tau=0.015)

    def settle(self, velocity, error, moving=True, valid=True, cycles=40):
        weight = target = 0.0
        for _ in range(cycles):
            weight, target = self.gate.update(
                np.asarray(velocity, dtype=float),
                np.asarray(error, dtype=float), 0.01, moving, valid)
        return weight, target

    def test_coherent_cruise_reaches_full_lead(self):
        weight, target = self.settle([0.30, 0.0, 0.0], [0.10, 0.0, 0.0])
        self.assertAlmostEqual(target, 1.0, places=12)
        self.assertGreater(weight, 0.98)

    def test_static_residual_speed_has_zero_target_weight(self):
        _, target = self.settle([0.059, 0.0, 0.0], [0.10, 0.0, 0.0])
        self.assertEqual(target, 0.0)
        weight, _ = self.settle([0.059, 0.0, 0.0], [0.10, 0.0, 0.0])
        self.assertEqual(weight, 0.0)

    def test_near_target_withdraws_lead(self):
        self.settle([0.30, 0.0, 0.0], [0.10, 0.0, 0.0])
        weight, target = self.settle([0.30, 0.0, 0.0], [0.020, 0.0, 0.0])
        self.assertEqual(target, 0.0)
        self.assertEqual(weight, 0.0)

    def test_reversal_cannot_push_past_target(self):
        self.settle([0.30, 0.0, 0.0], [0.10, 0.0, 0.0])
        weight, target = self.gate.update(
            np.array([0.30, 0.0, 0.0]), np.array([-0.10, 0.0, 0.0]),
            0.01, True)
        self.assertEqual(target, 0.0)
        self.assertLess(weight, 0.50)

    def test_stationary_latch_and_invalid_measurement_reset_immediately(self):
        self.settle([0.30, 0.0, 0.0], [0.10, 0.0, 0.0])
        weight, target = self.gate.update(
            np.array([0.30, 0.0, 0.0]), np.array([0.10, 0.0, 0.0]),
            0.01, False)
        self.assertEqual((weight, target), (0.0, 0.0))
        self.settle([0.30, 0.0, 0.0], [0.10, 0.0, 0.0])
        weight, target = self.gate.update(
            np.array([0.30, 0.0, 0.0]), np.array([0.10, 0.0, 0.0]),
            0.01, True, valid=False)
        self.assertEqual((weight, target), (0.0, 0.0))

    def test_release_is_faster_than_attack(self):
        attack_gate = AdaptiveLeadGate()
        attack_w, _ = attack_gate.update(
            np.array([0.30, 0.0, 0.0]), np.array([0.10, 0.0, 0.0]),
            0.01, True)
        self.settle([0.30, 0.0, 0.0], [0.10, 0.0, 0.0])
        before = self.gate.weight
        release_w, _ = self.gate.update(
            np.array([0.30, 0.0, 0.0]), np.array([0.020, 0.0, 0.0]),
            0.01, True)
        self.assertGreater(attack_w, 0.0)
        self.assertLess(before - release_w, before)
        self.assertLess(release_w, 1.0 - attack_w)

    def test_nonfinite_input_fails_safe(self):
        weight, target = self.gate.update(
            np.array([np.nan, 0.0, 0.0]), np.array([0.10, 0.0, 0.0]),
            0.01, True)
        self.assertEqual((weight, target), (0.0, 0.0))
        self.assertTrue(np.isfinite(self.gate.weight))

    def test_direction_threshold_is_continuous(self):
        gate = AdaptiveLeadGate(direction_full_cos=0.50)
        # 两者距离权重都已为 1，只让方向余弦跨过 0.5。
        below, _ = gate.update(
            np.array([0.30, 0.0, 0.0]), np.array([0.030, 0.055, 0.0]),
            10.0, True)
        above, _ = gate.update(
            np.array([0.30, 0.0, 0.0]), np.array([0.040, 0.055, 0.0]),
            10.0, True)
        self.assertGreater(above, below)
        self.assertLessEqual(above, 1.0)


class JointScalingAndReversalTests(unittest.TestCase):
    def test_joint_velocity_proportional_scaling_preserves_direction(self):
        raw = np.array([3.0, -1.5, 0.75, 0.0, -0.3, 0.6])
        scaled = scale_joint_velocities(raw, 1.5, grouped=False)
        self.assertAlmostEqual(float(np.max(np.abs(scaled))), 1.5, places=12)
        # 比例方向必须严格保持不变
        np.testing.assert_allclose(scaled / scaled[0], raw / raw[0], atol=1e-12)
        # 分组解耦模式：水平偏航组 [J1, J5] 饱和时绝不拖慢竖直俯仰组 [J2, J3, J4, J6]
        grouped_in = np.array([3.0, 1.0, -0.8, -0.2, -3.0, 0.1])
        grouped_out = scale_joint_velocities(grouped_in, 1.5, grouped=True)
        np.testing.assert_allclose(grouped_out[[0, 4]], [1.5, -1.5], atol=1e-12)
        np.testing.assert_allclose(grouped_out[[1, 2, 3, 5]],
                                   grouped_in[[1, 2, 3, 5]], atol=1e-12)
        within = np.array([0.8, -1.2, 0.5, 0.0, 0.1, -0.3])
        np.testing.assert_allclose(scale_joint_velocities(within, 1.5), within)
        np.testing.assert_array_equal(
            scale_joint_velocities(np.array([np.nan, 1.0, 0.0]), 1.5),
            np.zeros(3))

    def test_per_axis_reversal_damps_reversing_axis_without_hurting_orthogonal_axis(self):
        v_target = np.array([-0.40, 0.50, 0.0])
        actual = np.array([0.40, 0.45, 0.0])
        prev = np.array([0.40, 0.45, 0.0])
        new_act, new_prev, reversing = apply_axis_reversal_damping(
            v_target, actual, prev, residual_scale=0.20, enabled=True)
        self.assertTrue(reversing)
        # X 轴反向：actual 与 prev 均按比例收缩，避免 prev 把 actual 重新拉回满速
        self.assertAlmostEqual(new_act[0], 0.08, places=12)
        self.assertAlmostEqual(new_prev[0], 0.24, places=12)
        # Y 轴同向巡航：丝毫不受 X 轴换向牵连
        self.assertAlmostEqual(new_act[1], 0.45, places=12)
        self.assertAlmostEqual(new_prev[1], 0.45, places=12)


class DelayedTwistCompensatorTests(unittest.TestCase):
    def test_cancels_delayed_1_67hz_oscillation_without_phase_inversion(self):
        # 模拟 1.67Hz 振荡下机械臂相机速度经 100ms 延迟 + 80ms 一阶低通进入 KF 速度
        comp = DelayedTwistCompensator(delay_s=0.10, lpf_tau_s=0.08,
                                       local_window_s=0.04)
        dt = 0.01
        omega = 2.0 * np.pi * 1.67
        palm = np.array([0.0, 0.0, 0.40])
        lpf_true = 0.0
        alpha = 1.0 - np.exp(-dt / 0.08)
        raw_hist = []
        matched_res = []
        naive_res = []
        for k in range(300):
            t = k * dt
            vx_cmd = 0.45 * np.sin(omega * t)
            raw_hist.append(vx_cmd)
            vx_del = raw_hist[k - 10] if k >= 10 else 0.0
            lpf_true += (vx_del - lpf_true) * alpha
            # 静止手掌在相机系看到的滞后速度 = -lpf_true
            obs_kv = np.array([-lpf_true, 0.0, 0.0])
            twist_now = np.array([vx_cmd, 0.0, 0.0, 0.0, 0.0, 0.0])
            comp.push(twist_now, t, dt)
            if k >= 80:
                kv_matched = comp.compensate(obs_kv, palm, now=t, near_stop=False)
                kv_naive = compensate_palm_velocity(obs_kv, twist_now, palm)
                matched_res.append(abs(kv_matched[0]))
                naive_res.append(abs(kv_naive[0]))
        # 零延迟补偿因相位反转反而把静止自运动残影放大（0.37 > 原始均值 0.29）；
        # 延迟+低通匹配补偿应精确抵消至接近 0
        self.assertGreater(float(np.mean(naive_res)), 0.35)
        self.assertLess(float(np.mean(matched_res)), 1e-6)

    def test_constant_moving_hand_survives_cruise_compensation(self):
        comp = DelayedTwistCompensator(delay_s=0.10, lpf_tau_s=0.08,
                                       local_window_s=0.04)
        palm = np.array([0.0, 0.0, 0.40])
        twist = np.array([0.40, 0.0, 0.0, 0.0, 0.0, 0.0])
        for k in range(80):
            comp.push(twist, k * 0.01, 0.01)
        # 手掌以 0.40 m/s 匀速移动且机械臂以 0.40 m/s 同步追随 → 相机系相对速度=0
        kv = comp.compensate(np.zeros(3), palm, now=0.79, near_stop=False)
        self.assertGreater(kv[0], 0.38)

    def test_near_stop_local_window_absorbs_extended_lag_drift(self):
        # 伸展态实际滞后比标称多 30ms（130ms vs 100ms），near_stop 局部窗应能抵消
        comp = DelayedTwistCompensator(delay_s=0.10, lpf_tau_s=0.08,
                                       local_window_s=0.04)
        palm = np.array([0.0, 0.0, 0.40])
        dt = 0.01
        raw_hist = []
        lpf_ext = 0.0
        alpha = 1.0 - np.exp(-dt / 0.08)
        for k in range(60):
            t = k * dt
            vx = 0.50 if k < 35 else 0.0
            raw_hist.append(vx)
            vx_del = raw_hist[k - 13] if k >= 13 else 0.0
            lpf_ext += (vx_del - lpf_ext) * alpha
            comp.push(np.array([vx, 0.0, 0.0, 0.0, 0.0, 0.0]), t, dt)
        obs = np.array([-lpf_ext, 0.0, 0.0])
        kv = comp.compensate(obs, palm, now=0.59, near_stop=True)
        self.assertLess(float(np.linalg.norm(kv)), 0.03)

    def test_invalid_inputs_and_reset_fail_safe(self):
        comp = DelayedTwistCompensator()
        comp.push(np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.01, 0.01)
        kv = comp.compensate(np.array([0.1, 0.0, 0.0]), np.zeros(3), now=0.01)
        np.testing.assert_allclose(kv, [0.1, 0.0, 0.0])
        comp.reset()
        self.assertEqual(len(comp._buf), 0)
        with self.assertRaises(ValueError):
            DelayedTwistCompensator(delay_s=-0.01)


class PostBrakeSettleGuardTests(unittest.TestCase):
    def test_fresh_start_has_no_cooldown_and_brake_triggers_window(self):
        guard = PostBrakeSettleGuard(base_settle_s=0.22, ext_extra_s=0.08)
        self.assertFalse(guard.in_cooldown(0.0))
        self.assertFalse(guard.in_cooldown(1.0))
        guard.update(0.01, 0.80)
        self.assertTrue(guard.in_cooldown(0.0))
        # 停稳 0.15s 后仍在冷静窗内，但应触发一次 KF 重锚
        reanchored = 0
        for _ in range(15):
            guard.update(0.01, 0.0)
            if guard.consume_kf_reanchor(0.0):
                reanchored += 1
        self.assertEqual(reanchored, 1)
        self.assertTrue(guard.in_cooldown(0.0))
        # 冷静窗内唤醒速度门槛被抬高，防止 70ms 后回弹假速度炸开锁存
        self.assertGreater(guard.effective_unlock_threshold(0.13, 0.0), 0.25)
        # 再过 0.10s（总计 0.25s）：近区退出冷静窗，但满伸展（0.30s）仍在冷静窗
        for _ in range(10):
            guard.update(0.01, 0.0)
        self.assertFalse(guard.in_cooldown(0.0))
        self.assertTrue(guard.in_cooldown(1.0))
        self.assertAlmostEqual(guard.effective_unlock_threshold(0.13, 0.0), 0.13)


class ReachScheduleTests(unittest.TestCase):
    def test_near_workspace_preserves_full_performance(self):
        sched = compute_reach_schedule(reach_r=0.20, pos_err_norm=0.05)
        self.assertEqual(sched["ext_ratio"], 0.0)
        self.assertEqual(sched["vel_att_target"], 1.0)
        self.assertEqual(sched["ff_scale"], 1.0)
        self.assertEqual(sched["lead_scale"], 1.0)
        self.assertEqual(sched["accel_scale"], 1.0)
        self.assertEqual(sched["deadband_on_scale"], 1.0)
        self.assertEqual(sched["fast_start_dist_scale"], 1.0)

    def test_full_extension_softens_docking_and_preserves_far_cruise(self):
        dock = compute_reach_schedule(reach_r=0.38, pos_err_norm=0.05)
        far = compute_reach_schedule(reach_r=0.38, pos_err_norm=0.30)
        self.assertAlmostEqual(dock["ext_ratio"], 1.0)
        self.assertLess(dock["vel_att_target"], 0.65)
        self.assertAlmostEqual(far["vel_att_target"], 1.0)  # 大误差远距追赶不限速
        self.assertAlmostEqual(dock["ff_scale"], 0.50)
        self.assertAlmostEqual(dock["lead_scale"], 0.30)
        self.assertAlmostEqual(dock["accel_scale"], 0.65)
        self.assertAlmostEqual(dock["deadband_on_scale"], 1.35)
        self.assertAlmostEqual(dock["fast_start_dist_scale"], 1.40)

    def test_nonfinite_inputs_fall_back_to_neutral(self):
        sched = compute_reach_schedule(reach_r=float("nan"), pos_err_norm=0.1)
        self.assertEqual(sched["vel_att_target"], 1.0)
        self.assertEqual(sched["ext_ratio"], 0.0)


class OscillationDamperTests(unittest.TestCase):
    def test_single_move_single_reversal_and_slow_wave_never_trigger(self):
        damper = OscillationDamper(err_thresh=0.018, min_half_s=0.14,
                                   max_half_s=0.42, decay_tau_s=0.55)
        # 单程运动 + 单次折返：最多 1 次过零，绝不触发
        for k in range(100):
            t = k * 0.01
            err_x = 0.10 if t < 0.30 else -0.10
            levels = damper.update(np.array([err_x, 0.0, 0.0]), t, 0.01)
            np.testing.assert_array_equal(levels, np.zeros(3))
        # 正常慢速来回挥手（0.7Hz，半周期 0.71s > 0.42s）：绝不触发
        damper.reset()
        for k in range(300):
            t = k * 0.01
            err_x = 0.12 * np.sin(2.0 * np.pi * 0.7 * t)
            levels = damper.update(np.array([err_x, 0.0, 0.0]), t, 0.01)
            np.testing.assert_array_equal(levels, np.zeros(3))

        # 单程运动 + 快速两连折返（仅 2 次过零，< confirm_reversals=3）：绝不误触发
        damper.reset()
        for k in range(120):
            t = k * 0.01
            if t < 0.25:
                err_x = 0.08
            elif t < 0.50:
                err_x = -0.08
            elif t < 0.75:
                err_x = 0.08
            else:
                err_x = 0.005
            levels = damper.update(np.array([err_x, 0.0, 0.0]), t, 0.01)
            np.testing.assert_array_equal(levels, np.zeros(3))

    def test_large_error_chase_releases_oscillation_damper(self):
        damper = OscillationDamper(err_thresh=0.018, min_half_s=0.14,
                                   max_half_s=0.42, decay_tau_s=0.55,
                                   confirm_reversals=3,
                                   large_err_release=0.15)
        # 先用 1.7Hz 中幅极限环拉起阻尼
        for k in range(110):
            t = k * 0.01
            err_x = 0.08 * np.sin(2.0 * np.pi * 1.7 * t)
            damper.update(np.array([err_x, 0.0, 0.0]), t, 0.01)
        self.assertGreaterEqual(damper.levels[0], 0.5)
        # 随后人手大幅主动拉远（|err| >= 0.15m），连击清零且 0.25s 内迅速释放阻尼
        for k in range(110, 145):
            damper.update(np.array([0.20, 0.0, 0.0]), k * 0.01, 0.01)
        self.assertEqual(damper.streak[0], 0)
        self.assertLess(damper.levels[0], 0.12)

    def test_1_7hz_limit_cycle_triggers_rapidly_only_on_oscillating_axis(self):
        damper = OscillationDamper(err_thresh=0.018, min_half_s=0.14,
                                   max_half_s=0.42, decay_tau_s=0.55)
        max_x = 0.0
        max_y = 0.0
        for k in range(120):
            t = k * 0.01
            err_x = 0.08 * np.sin(2.0 * np.pi * 1.7 * t)
            levels = damper.update(np.array([err_x, 0.10, 0.0]), t, 0.01)
            max_x = max(max_x, float(levels[0]))
            max_y = max(max_y, float(levels[1]))
        self.assertGreaterEqual(max_x, 0.65)
        self.assertEqual(max_y, 0.0)  # 正交巡航轴丝毫不受影响

    def test_decays_after_settling_and_handles_invalid_inputs(self):
        damper = OscillationDamper(decay_tau_s=0.30)
        for k in range(100):
            t = k * 0.01
            damper.update(np.array([0.08 * np.sin(2.0 * np.pi * 1.7 * t), 0.0, 0.0]), t, 0.01)
        self.assertGreater(damper.levels[0], 0.5)
        # 振荡停止后应平滑衰减回 0
        for k in range(100, 260):
            damper.update(np.array([0.005, 0.0, 0.0]), k * 0.01, 0.01)
        self.assertLess(damper.levels[0], 0.05)
        # 异常输入安全清零
        out = damper.update(np.array([np.nan, 0.0, 0.0]), 3.0, 0.01)
        np.testing.assert_array_equal(out, np.zeros(3))
        with self.assertRaises(ValueError):
            OscillationDamper(min_half_s=0.5, max_half_s=0.3)


class CameraPoseHistoryTests(unittest.TestCase):
    def test_static_world_point_invariant_under_translating_rotating_camera(self):
        hist = CameraPoseHistory(lpf_tau_s=0.0, max_history_s=0.60)
        T_tc = np.eye(4)
        T_tc[:3, 3] = np.array([0.02, -0.01, 0.05])
        world_hand = np.array([0.35, 0.12, 0.55])
        delay_s = 0.14

        def pose_at(tau):
            p = np.array([0.10 + 0.30 * tau, 0.05 * np.sin(3.0 * tau), 0.30])
            ang = 0.25 * tau
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            return p, R

        for k in range(50):
            t = k * 0.01
            p_cmd, R_cmd = pose_at(t)
            self.assertTrue(hist.push(t, p_cmd, R_cmd, T_tc, dt=0.01))
            if t >= delay_s + 0.02:
                # 物理相机实际位姿落后命令 delay_s
                p_phys, R_phys = pose_at(t - delay_s)
                p_cam_phys = p_phys + R_phys @ T_tc[:3, 3]
                R_cam_phys = R_phys @ T_tc[:3, :3]
                obs_cam = R_cam_phys.T @ (world_hand - p_cam_phys)
                p_lu, R_lu = hist.lookup(t - delay_s, use_lpf=True)
                # 旋转矩阵必须保持正交且 det=+1
                np.testing.assert_allclose(R_lu.T @ R_lu, np.eye(3), atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.det(R_lu)), 1.0, places=6)
                reconstructed_world = p_lu + R_lu @ obs_cam
                np.testing.assert_allclose(
                    reconstructed_world, world_hand, atol=2e-3)

    def test_blended_pose_interpolates_and_handles_invalid_inputs(self):
        hist = CameraPoseHistory(lpf_tau_s=0.0, max_history_s=0.50)
        T_tc = np.eye(4)
        for k in range(25):
            t = k * 0.01
            hist.push(t, np.array([t, 0.0, 0.0]), np.eye(3), T_tc, dt=0.01)
        p0, _ = hist.blended_pose(0.24, 0.14, 0.0)
        p1, _ = hist.blended_pose(0.24, 0.14, 1.0)
        pm, _ = hist.blended_pose(0.24, 0.14, 0.5)
        self.assertAlmostEqual(p0[0], 0.10, places=4)
        self.assertAlmostEqual(p1[0], 0.24, places=4)
        self.assertAlmostEqual(pm[0], 0.17, places=4)
        self.assertFalse(hist.push(0.25, np.array([np.nan, 0.0, 0.0]),
                                   np.eye(3), T_tc, dt=0.01))


class OrientationPriorityAndJointLimitsTests(unittest.TestCase):
    def test_orientation_weight_relaxes_in_far_cruise_and_guards_tilt_limit(self):
        # 停靠区或未激活：保持 1.0 刚性姿态
        self.assertEqual(compute_orientation_cruise_weight(0.05, 0.01), 1.0)
        self.assertEqual(compute_orientation_cruise_weight(
            0.20, 0.01, active=False), 1.0)
        # 大误差且小倾角：平滑下调至 min_weight=0.40
        w_far = compute_orientation_cruise_weight(
            0.20, np.deg2rad(2.0), min_weight=0.40,
            ori_soft_max_rad=np.deg2rad(10.0))
        self.assertAlmostEqual(w_far, 0.40, places=6)
        # 倾角接近 10° 软上限：自动拉回 1.0 防止相机丢手
        w_guard = compute_orientation_cruise_weight(
            0.20, np.deg2rad(10.0), min_weight=0.40,
            ori_soft_max_rad=np.deg2rad(10.0))
        self.assertEqual(w_guard, 1.0)

    def test_weighted_dls_matches_standard_at_unit_weight_and_frees_wrist(self):
        # 构造类似 B601 的竖直+俯仰耦合雅可比：z 靠 J2/J3，俯仰由 J2+J3+J4 决定
        J = np.eye(6, dtype=np.float64)
        J[2, 1] = 0.25
        J[2, 2] = 0.25
        J[2, 2] = 0.20
        J[4, 1] = 1.0
        J[4, 2] = 1.0
        J[4, 3] = 1.0
        J[4, 4] = 0.0
        J[3, 4] = 1.0
        v = np.array([0.0, 0.0, 0.50, 0.0, 0.0, 0.0])
        dq_rigid = compute_weighted_damped_pseudoinverse(
            J, np.ones(6), damping=0.05) @ v
        # 校验单位权重与标准伪逆恒等
        JJT = J @ J.T
        dq_std = J.T @ np.linalg.inv(JJT + (0.05 ** 2) * np.eye(6)) @ v
        np.testing.assert_allclose(dq_rigid, dq_std, atol=1e-10)
        # 降低姿态行权重后，平移速度仍高精度满足，且手腕 J4 负担下降
        dq_soft = compute_weighted_damped_pseudoinverse(
            J, np.array([1.0, 1.0, 1.0, 0.40, 0.40, 0.40]), damping=0.05) @ v
        self.assertLess(abs(dq_soft[3]), abs(dq_rigid[3]))
        self.assertAlmostEqual(float((J @ dq_soft)[2]), float((J @ dq_rigid)[2]), places=2)

    def test_per_joint_vector_limits_free_wrist_without_exceeding_j2_j3(self):
        limits = np.array([2.4, 2.0, 2.0, 2.8, 2.8, 2.2])
        # J4=2.6, J2=1.5, J3=1.4：按统一 2.0 会把 J2/J3 砍到 1.15/1.08；按 L2 向量不限速
        dq = np.array([1.8, 1.5, 1.4, 2.6, 1.8, 0.2])
        scaled_l2 = scale_joint_velocities(dq, limits, grouped=False)
        np.testing.assert_allclose(scaled_l2, dq, atol=1e-12)
        # 一旦承重 J2 超 2.0，必须严格压回 <=2.0
        dq_over = np.array([1.0, 2.5, 1.5, 2.6, 1.0, 0.0])
        scaled_safe = scale_joint_velocities(dq_over, limits, grouped=False)
        self.assertAlmostEqual(scaled_safe[1], 2.0, places=6)
        self.assertTrue(np.all(np.abs(scaled_safe) <= limits + 1e-9))

    def test_consensus_velocity_suppresses_single_estimator_leak(self):
        # 真实运动：两估计器同向一致，取保守较小幅值
        v_ok = consensus_velocity([0.45, -0.30, 0.10], [0.40, -0.35, 0.12])
        np.testing.assert_allclose(v_ok, [0.40, -0.30, 0.10], atol=1e-9)
        # 急停回弹/共振失配：一侧漏出 0.45m/s、一侧已压至 0.04m/s 或异向 -> 抑制
        v_leak = consensus_velocity([0.45, -0.40, 0.20], [0.04, 0.15, 0.0])
        np.testing.assert_allclose(v_leak, [0.04, 0.0, 0.0], atol=1e-9)
        np.testing.assert_array_equal(
            consensus_velocity([np.nan, 0.1, 0.0], [0.1, 0.1, 0.0]), np.zeros(3))

    def test_joint_limit_cushion_softens_outward_and_preserves_inward(self):
        q_min = np.array([-2.0, 0.0, 0.17, -1.40, -2.0, -2.0])
        q_max = np.array([ 2.0, 2.0, 2.50,  1.40,  2.0,  2.0])
        # J4 距上限 0.04rad（margin=0.08 -> 50%），J3 距下限 0.02rad（25%）
        q_now = np.array([0.0, 1.0, 0.19, 1.36, 0.0, 0.0])
        dq_out = np.array([1.0, 1.0, -2.0, 2.0, 0.0, 0.0])
        out, flag = apply_joint_limit_cushion(q_now, dq_out, q_min, q_max, 0.08)
        self.assertTrue(flag)
        self.assertAlmostEqual(out[3], 1.0, places=6)   # 2.0 * (0.04/0.08)
        self.assertAlmostEqual(out[2], -0.5, places=6)  # -2.0 * (0.02/0.08)
        self.assertAlmostEqual(out[0], 1.0, places=6)
        # 离开限位方向绝不衰减
        dq_in = np.array([1.0, 1.0, 2.0, -2.0, 0.0, 0.0])
        out_in, flag_in = apply_joint_limit_cushion(q_now, dq_in, q_min, q_max, 0.08)
        self.assertFalse(flag_in)
        np.testing.assert_allclose(out_in, dq_in, atol=1e-9)

    def test_oscillation_damper_retains_memory_across_brief_stop_dip(self):
        damper = OscillationDamper(err_thresh=0.018, min_half_s=0.14,
                                   max_half_s=0.42, decay_tau_s=0.55,
                                   confirm_reversals=3, large_err_release=0.15)
        # 先产生 2 次 1.7Hz 过零反转（streak=2）
        damper.update([0.06, 0.0, 0.0], 0.00, 0.01, active=True)
        damper.update([-0.06, 0.0, 0.0], 0.25, 0.01, active=True)
        damper.update([0.06, 0.0, 0.0], 0.50, 0.01, active=True)
        self.assertEqual(damper.streak[0], 2)
        # 短暂 100ms 停手（active=False, hard_reset=False）不得清零连击记忆
        for k in range(10):
            damper.update([0.01, 0.0, 0.0], 0.51 + k * 0.01, 0.01,
                          active=False, hard_reset=False)
        self.assertEqual(damper.streak[0], 2)
        # 紧接着第 3 次反冲（t=0.75s，距上次翻转 0.25s）应立刻触发阻尼
        lv = damper.update([-0.06, 0.0, 0.0], 0.75, 0.01, active=True)
        self.assertGreaterEqual(lv[0], 0.65)
        # hard_reset=True 必须彻底清零
        lv_rst = damper.update([0.0, 0.0, 0.0], 0.76, 0.01,
                               active=False, hard_reset=True)
        self.assertEqual(damper.streak[0], 0)
        np.testing.assert_array_equal(lv_rst, np.zeros(3))

    def test_kf_partial_xy_update_tracks_xy_without_touching_frozen_z(self):
        kf = KalmanTarget3D(q=2.0, r_xy=0.0023, r_z=0.0038)
        # 未初始化时不得用部分观测强行初始化
        self.assertFalse(kf.update_partial([0.1, 0.2], [[1, 0, 0], [0, 1, 0]], 0.02))
        kf.update([0.0, 0.0, 0.40], 0.0)
        kf.x[5] = 0.0
        # 模拟 10 帧深度冻结期间，手掌沿 X/Y 高速移动，仅更新 XY
        H_xy = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for k in range(1, 11):
            t = k * 0.02
            ok = kf.update_partial([0.40 * t, -0.30 * t], H_xy, 0.02)
            self.assertTrue(ok)
        # X/Y 位置与速度已紧跟移动，Z 位置和 Z 速度丝毫未被污染
        self.assertAlmostEqual(kf.position()[2], 0.40, places=9)
        self.assertAlmostEqual(kf.velocity()[2], 0.0, places=9)
        self.assertGreater(kf.position()[0], 0.06)
        self.assertGreater(kf.velocity()[0], 0.25)
        self.assertLess(kf.velocity()[1], -0.18)
        # 旋转子空间（世界系相机 XY 平面）验证：沿光轴方向的深度分量保持不变
        kf_w = KalmanTarget3D(q=2.0, r_xy=0.0023, r_z=0.0038)
        kf_w.update([0.0, 0.0, 0.50], 0.0)
        ang = np.deg2rad(30.0)
        R = np.array([[np.cos(ang), 0.0, np.sin(ang)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(ang), 0.0, np.cos(ang)]])
        H_w = R[:, :2].T
        opt_axis = R[:, 2]
        z_before = float(np.dot(opt_axis, kf_w.position()))
        self.assertTrue(kf_w.update_partial([0.10, 0.05], H_w, 0.02))
        z_after = float(np.dot(opt_axis, kf_w.position()))
        self.assertAlmostEqual(z_after, z_before, places=9)

    def test_roi_box_hysteresis_and_exact_landmark_remapping(self):
        # 640x480 画面下 scale=0.68 -> 约 326x326 ROI
        box1 = compute_roi_box(640, 480, (320, 240), prev_box=None, scale=0.68)
        self.assertIsNotNone(box1)
        x0, y0, x1, y1 = box1
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertLess(x1 - x0, 400)
        # 掌心在安全死区内小幅移动：ROI 框必须完全不动，避免干扰 VIDEO 内部追踪
        box2 = compute_roi_box(640, 480, (345, 255), prev_box=box1, scale=0.68)
        self.assertEqual(box2, box1)
        # 掌心逼近边缘：自动重新居中且严格钳位在画面内
        box3 = compute_roi_box(640, 480, (610, 460), prev_box=box1, scale=0.68)
        self.assertNotEqual(box3, box1)
        self.assertEqual(box3[2], 640)
        self.assertEqual(box3[3], 480)
        self.assertIsNone(compute_roi_box(640, 480, None))
        # ROI 关键点映射回全图归一化坐标后，像素位置与捏合距离必须与全图 100% 一致
        roi_w = box1[2] - box1[0]
        roi_h = box1[3] - box1[1]
        # 假设全图某两点像素为 (300, 220) 与 (360, 250)
        lm_a = types.SimpleNamespace(
            x=(300.0 - box1[0]) / roi_w, y=(220.0 - box1[1]) / roi_h, z=-0.05)
        lm_b = types.SimpleNamespace(
            x=(360.0 - box1[0]) / roi_w, y=(250.0 - box1[1]) / roi_h, z=-0.05)
        remapped = remap_roi_landmarks([lm_a, lm_b], box1, 640, 480)
        self.assertAlmostEqual(remapped[0].x * 640.0, 300.0, places=6)
        self.assertAlmostEqual(remapped[0].y * 480.0, 220.0, places=6)
        self.assertAlmostEqual(remapped[1].x * 640.0, 360.0, places=6)
        self.assertAlmostEqual(remapped[1].y * 480.0, 250.0, places=6)
        pinch_px = float(np.hypot(
            (remapped[1].x - remapped[0].x) * 640.0,
            (remapped[1].y - remapped[0].y) * 480.0))
        self.assertAlmostEqual(pinch_px, float(np.hypot(60.0, 30.0)), places=6)


class MitExtensionP0Tests(unittest.TestCase):
    """v13 P0：MIT 扩展配置、CAN 瞬时故障容错（S1）、反馈缓存与上游版本校验。"""

    def _make_fake_upstream(self):
        import contextlib
        import threading

        class FakeMotor:
            def __init__(self, idx):
                self.idx = idx
                self.state = types.SimpleNamespace(
                    pos=0.0, vel=0.0, torq=0.5 * idx,
                    t_mos=35.0 + idx, t_rotor=36.0 + idx, status_code=0)
                self.fail_get_state = False
                self.mit_cmds = []

            def get_state(self):
                if self.fail_get_state:
                    raise RuntimeError("get_state err")
                return self.state

            def send_mit(self, pos, vel, kp, kd, tau):
                self.mit_cmds.append((pos, vel, kp, kd, tau))

        class FakeController:
            def __init__(self):
                self.poll_calls = 0

            def poll_feedback_once(self):
                self.poll_calls += 1

        class FakeUpstreamBase:
            def __init__(self):
                self.control_hz = 200.0
                self.motor_count = 7
                _gains = [(72.0, 3.6), (216.0, 12.0), (216.0, 12.0),
                          (50.0, 5.0), (50.0, 4.0), (50.0, 4.0), (50.0, 4.0)]
                self.config = types.SimpleNamespace(
                    motors=[types.SimpleNamespace(kp=kp, kd=kd) for kp, kd in _gains])
                self.target_positions = [0.0] * 7
                self.command_positions = [0.0] * 7
                self.max_speeds_rad_s = [2.0] * 7
                self.target_lock = threading.RLock()
                self.io_lock = threading.RLock()
                self.worker_stop_event = threading.Event()
                self.shutdown_started = False
                self.last_error = None
                self.motors = [FakeMotor(i) for i in range(7)]
                self.controller = FakeController()
                self.send_calls = 0
                self.fail_next_sends = 0
                self.stop_calls = []

            @contextlib.contextmanager
            def _io_lock_guard(self, timeout=None):
                with self.io_lock:
                    yield

            def _control_loop(self):
                pass

            def _send_mit_positions(self, positions_rad, *, lock_timeout=None):
                self.send_calls += 1
                if self.fail_next_sends > 0:
                    self.fail_next_sends -= 1
                    raise RuntimeError("transient CAN TX error")
                for idx, motor in enumerate(self.motors):
                    mc = self.config.motors[idx]
                    motor.send_mit(float(positions_rad[idx]), 0.0,
                                   float(mc.kp), float(mc.kd), 0.0)

            def _read_positions_rad(self):
                return [0.0] * 7

            def request_stop(self, **kwargs):
                self.stop_calls.append(kwargs)

            def set_joint_angles(self, angles_deg):
                with self.target_lock:
                    self.target_positions[:] = [np.deg2rad(float(a)) for a in angles_deg]

            def set_joint_angle(self, joint_id, angle_deg):
                with self.target_lock:
                    self.target_positions[joint_id - 1] = float(np.deg2rad(float(angle_deg)))

            def get_command_angles(self):
                return [0.0] * 7

            def set_max_speeds(self, speeds_deg_s):
                pass

            def start(self, **kwargs):
                pass

            def stop(self, **kwargs):
                pass

        return FakeUpstreamBase

    def test_config_validation_and_yaml_loading(self):
        from mit_controller_ext import load_mit_ext_cfg, validate_mit_ext_cfg

        cfg = load_mit_ext_cfg()
        self.assertTrue(cfg["enable"])
        self.assertTrue(cfg["feedback_trusted"])
        self.assertFalse(cfg["feedback_poll_once"])
        self.assertTrue(cfg["vel_ff_enable"])
        self.assertEqual(len(cfg["vel_ff_scale"]), 6)
        self.assertAlmostEqual(cfg["can_fault_window_s"], 0.03)
        self.assertEqual(cfg["can_fault_action"], "return_to_zero")

        for bad in (
            {"unknown_key": 1},
            {"can_fault_window_s": 0.005},
            {"can_fault_window_s": 0.50},
            {"can_fault_action": "drop"},
            {"feedback_cache": False, "feedback_trusted": True},
            {"enable": "yes"},
            {"vel_ff_scale": [0.5, 0.5, 0.5]},
            {"vel_ff_scale": [1.2, 0.8, 0.8, 0.8, 0.8, 0.8]},
            {"vel_ff_scale": [-0.1, 0.8, 0.8, 0.8, 0.8, 0.8]},
            {"vel_ff_watchdog_s": 0.005},
            {"vel_ff_lpf_tau_s": 0.20},
            {"gravity_ff_scale": [0.5, 0.5, 0.5, 0.5, 0.0, 0.0]},   # J1 必须为 0
            {"gravity_ff_scale": [0.0, 1.1, 0.5, 0.5, 0.0, 0.0]},   # 超过 0.95
            {"gravity_ff_max_nm": [0.0, 20.0, 12.0, 2.5, 2.0, 1.0]},  # 超代码级硬上限 14Nm
            {"gain_sched_r_min_m": 0.35, "gain_sched_r_max_m": 0.30},
            {"gain_sched_kd_far": [6.4, 14.5, 15.0, 6.0, 4.0, 4.0]},  # RS00 kd > 5.0
            {"track_err_limit_deg": 2.0},
            {"tau_limit_nm": [40.0, 28.0, 28.0, 10.0, 10.0, 10.0]},
            {"thermal_derate_start_c": 100.0, "thermal_derate_full_c": 95.0},
        ):
            with self.assertRaises(ValueError):
                validate_mit_ext_cfg(bad)

    def test_can_fault_monitor_transient_vs_sustained(self):
        from mit_controller_ext import CanFaultMonitor

        mon = CanFaultMonitor(window_s=0.03, control_hz=200.0)
        self.assertEqual(mon.threshold, 6)
        # 5 次连续失败不触发，成功 1 次即清零
        for _ in range(5):
            self.assertFalse(mon.record_failure())
        mon.record_success()
        self.assertEqual(mon.consecutive, 0)
        # 连续 6 次才触发故障
        for i in range(5):
            self.assertFalse(mon.record_failure(), f"step {i}")
        self.assertTrue(mon.record_failure())
        self.assertEqual(mon.max_consecutive, 6)

    def test_upstream_compat_and_fallback(self):
        from mit_controller_ext import check_upstream_compat, build_controller

        FakeBase = self._make_fake_upstream()
        ok, reason, _ = check_upstream_compat(FakeBase, require_pinned=False)
        self.assertTrue(ok, reason)
        # 钉死哈希不匹配时必须自动回退原版，且不得抛异常崩溃
        ctrl, info = build_controller(
            FakeBase,
            cfg={"enable": True, "require_pinned_upstream": True})
        self.assertFalse(info["active"])
        self.assertIs(type(ctrl), FakeBase)

        # 关闭开关同样回退原版
        ctrl2, info2 = build_controller(FakeBase, cfg={"enable": False})
        self.assertFalse(info2["active"])
        self.assertIs(type(ctrl2), FakeBase)

    def test_extended_controller_s1_tolerance_feedback_and_shutdown_retry(self):
        import threading
        import time
        from mit_controller_ext import make_extended_controller_class

        FakeBase = self._make_fake_upstream()
        ExtCls = make_extended_controller_class(FakeBase)
        ctrl = ExtCls(ext_cfg={"can_fault_window_s": 0.03,
                               "can_fault_action": "return_to_zero",
                               "feedback_cache": True,
                               "feedback_poll_once": True})
        # 先注入 3 次瞬时发送异常（< 6 拍阈值）：绝不能触发停机
        ctrl.fail_next_sends = 3
        t = threading.Thread(target=ctrl._control_loop, daemon=True)
        t.start()
        time.sleep(0.08)
        self.assertEqual(ctrl.stop_calls, [])
        snap = ctrl.get_feedback_snapshot()
        self.assertIsNotNone(snap)
        self.assertTrue(all(snap["valid"]))
        self.assertGreater(ctrl.controller.poll_calls, 0)

        # 录波功能验证
        ctrl.start_recording()
        time.sleep(0.04)
        rows = ctrl.stop_recording()
        self.assertGreaterEqual(len(rows), 3)

        # 注入 10 次连续发送异常（≥ 6 拍阈值）：必须触发 return_to_zero（非 emergency 失能）
        ctrl.fail_next_sends = 10
        t.join(timeout=1.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(ctrl.stop_calls), 1)
        self.assertTrue(ctrl.stop_calls[0]["return_to_zero"])
        self.assertFalse(ctrl.stop_calls[0]["emergency"])

        # 安全停机阶段瞬时异常必须自动重试并成功，不轻易中止回零
        ctrl.shutdown_started = True
        ctrl.fail_next_sends = 2
        ctrl._send_mit_positions([0.0] * 7)
        self.assertEqual(ctrl.fail_next_sends, 0)

    def test_feedback_error_auto_disable_does_not_trip_can_fault(self):
        from mit_controller_ext import make_extended_controller_class

        FakeBase = self._make_fake_upstream()
        ExtCls = make_extended_controller_class(FakeBase)
        ctrl = ExtCls(ext_cfg={"feedback_cache": True})
        ctrl.motors[0].fail_get_state = True
        for _ in range(ctrl._fb_disable_threshold + 2):
            ctrl._send_and_sample([0.0] * 7)
        self.assertFalse(ctrl._fb_enabled)
        self.assertEqual(ctrl.stop_calls, [])
        self.assertEqual(ctrl._fault_monitor.total_failures, 0)

    def test_p1_foh_zero_ripple_watchdog_and_conflict_guards(self):
        from mit_controller_ext import compute_p1_mit_step

        # 1) 异步 100Hz→200Hz 连续匀速：任意线程相位 phi=3ms 下，每拍步长必须精确等于 v*5ms（零阶梯纹波）
        v = 1.0
        period = 0.005
        cmd = 0.0
        dq_prev = 0.0
        steps = []
        for tick in range(8):
            t_now = 0.003 + tick * period
            update_idx = int(t_now // 0.010)
            target = (update_idx + 1) * 0.010 * v
            elapsed = t_now - update_idx * 0.010
            new_cmd, dq_prev, wd = compute_p1_mit_step(
                target, cmd, v, dq_prev, max_speed=2.0, period=period, elapsed=elapsed,
                vel_ff_enable=True, vel_ff_scale=0.85, foh_enable=True,
                foh_upper_dt_s=0.010, watchdog_s=0.040, watchdog_tau_s=0.030,
                lpf_tau_s=0.008, mit_vel_mode=True)
            steps.append(new_cmd - cmd)
            cmd = new_cmd
            self.assertFalse(wd)
        # 首拍从 0 走到 0.003，之后每一拍必须严格等于 0.005（零 2x/0x 纹波）
        self.assertAlmostEqual(steps[0], 0.003, places=6)
        for st in steps[1:]:
            self.assertAlmostEqual(st, 0.005, places=6)
        self.assertGreater(dq_prev, 0.70)
        self.assertLessEqual(dq_prev, 0.85 + 1e-6)

        # 2) 上层卡死看门狗：cmd 绝不外推越过最后 target，dq_send 指数衰减归零
        frozen_target = 0.040
        cmd = frozen_target - 0.002
        dq_s = 0.85
        for k in range(40):
            elapsed = 0.045 + k * period
            cmd, dq_s, wd = compute_p1_mit_step(
                frozen_target, cmd, 1.0, dq_s, max_speed=2.0, period=period,
                elapsed=elapsed, vel_ff_enable=True, vel_ff_scale=0.85,
                foh_enable=True, foh_upper_dt_s=0.010, watchdog_s=0.040,
                watchdog_tau_s=0.030, lpf_tau_s=0.008, mit_vel_mode=True)
            self.assertTrue(wd)
            self.assertLessEqual(cmd, frozen_target + 1e-9)
        self.assertLess(abs(dq_s), 0.01)

        # 3) 方向冲突保护：dq_target 与 (target - cmd) 反向 >0.2° 时，前馈必须归零
        _, dq_conf, _ = compute_p1_mit_step(
            target=0.0, command=0.05, dq_target=+1.0, dq_prev_send=0.0,
            max_speed=2.0, period=period, elapsed=0.005,
            vel_ff_enable=True, vel_ff_scale=0.90, foh_enable=True,
            foh_upper_dt_s=0.010, watchdog_s=0.040, watchdog_tau_s=0.030,
            lpf_tau_s=0.0, mit_vel_mode=True)
        self.assertEqual(dq_conf, 0.0)

        # 4) 硬限速钳制：即使外部传入超大速度，步长和 dq_send 绝不越过 max_speed
        cmd_c, dq_c, _ = compute_p1_mit_step(
            target=1.0, command=0.0, dq_target=10.0, dq_prev_send=0.0,
            max_speed=2.0, period=period, elapsed=0.005,
            vel_ff_enable=True, vel_ff_scale=1.0, foh_enable=False,
            foh_upper_dt_s=0.010, watchdog_s=0.040, watchdog_tau_s=0.030,
            lpf_tau_s=0.0, mit_vel_mode=True)
        self.assertLessEqual(cmd_c, 2.0 * period + 1e-9)
        self.assertLessEqual(dq_c, 2.0 + 1e-9)

    def test_p1_controller_sends_dq_isolates_j7_and_exits_on_legacy_call(self):
        import threading
        import time
        from mit_controller_ext import make_extended_controller_class

        FakeBase = self._make_fake_upstream()
        ExtCls = make_extended_controller_class(FakeBase)
        ctrl = ExtCls(ext_cfg={"vel_ff_enable": True,
                               "vel_ff_scale": [0.7, 0.85, 0.85, 0.9, 0.9, 0.9],
                               "vel_ff_watchdog_s": 0.04,
                               "vel_ff_watchdog_tau_s": 0.02,
                               "vel_ff_lpf_tau_s": 0.005})
        # 非有限速度必须被立即拒绝
        with self.assertRaises(ValueError):
            ctrl.set_joint_mit_targets([0.0] * 7, [float("nan")] + [0.0] * 6)

        t = threading.Thread(target=ctrl._control_loop, daemon=True)
        t.start()
        try:
            # 连续喂 60ms 匀速轨迹（即便第 7 通道误传 100°/s，J7 也必须恒为 0）
            for k in range(6):
                ang = [0.0, 0.0, 0.0, (k + 1) * 0.573, 0.0, 0.0, 10.0]
                vel = [0.0, 0.0, 0.0, 57.3, 0.0, 0.0, 100.0]
                ctrl.set_joint_mit_targets(ang, vel)
                time.sleep(0.01)
            j4_vels = [c[1] for c in ctrl.motors[3].mit_cmds]
            j7_vels = [c[1] for c in ctrl.motors[6].mit_cmds]
            self.assertGreater(max(j4_vels), 0.5)
            self.assertLessEqual(max(j4_vels), 0.95)
            self.assertTrue(all(v == 0.0 for v in j7_vels), "J7 夹爪严禁透传速度前馈")

            # 调用老接口 set_joint_angles（如 hold/move_to_joints）必须立刻清零速度前馈
            ctrl.set_joint_angles([0.0, 0.0, 0.0, 3.4, 0.0, 0.0, 10.0])
            time.sleep(0.025)
            self.assertEqual(ctrl.motors[3].mit_cmds[-1][1], 0.0)

            # 停更触发看门狗衰减验证
            ctrl.set_joint_mit_targets([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 10.0],
                                       [0.0, 0.0, 0.0, 57.3, 0.0, 0.0, 0.0])
            time.sleep(0.18)
            stats = ctrl.get_loop_stats()
            self.assertGreaterEqual(stats["watchdog_trips"], 1)
            self.assertLess(abs(ctrl.motors[3].mit_cmds[-1][1]), 0.02)
        finally:
            ctrl.worker_stop_event.set()
            t.join(timeout=1.0)

    def test_p3_gravity_tau_cradle_fade_caps_and_gain_scheduling(self):
        from mit_controller_ext import (
            cradle_zero_fade, compute_p3_gravity_tau, compute_p3_gains)
        from tools.p3_gravity_verify import check_sag_monotonic_safety

        # 1) 零位支架淡出：零位为 0.0，HOME (J2≈19.6°, J3≈19.1°) 为 1.0
        self.assertEqual(cradle_zero_fade([0.0] * 6), 0.0)
        self.assertAlmostEqual(cradle_zero_fade([0.0, 0.34, 0.37, 0.09, 0.0, 0.0]), 1.0)

        # 2) 重力矩计算：硬限幅、J1 恒为 0、变化率限制与异常平滑归零
        q_home = [-0.43, 0.34, 0.37, 0.09, 0.0, 0.0]
        raw_fn = lambda q: [9.0, 20.0, 20.0, 5.0, 1.0, 0.5]
        tau = [0.0] * 6
        for _ in range(150):
            tau, err = compute_p3_gravity_tau(
                q_home, tau, raw_fn, period=0.005, enabled=True,
                scales=[0.0, 0.8, 0.8, 0.8, 0.0, 0.0],
                max_nm=[0.0, 12.0, 12.0, 2.5, 2.0, 1.0],
                ramp_factor=1.0, slew_nm_s=30.0)
            self.assertFalse(err)
        self.assertEqual(tau[0], 0.0)          # J1 恒为 0
        self.assertAlmostEqual(tau[1], 12.0)   # 20*0.8=16 被硬钳位到 12.0
        self.assertAlmostEqual(tau[2], 12.0)
        self.assertAlmostEqual(tau[3], 2.5)    # 5*0.8=4 被硬钳位到 2.5
        self.assertEqual(tau[4], 0.0)

        # 回调异常时不能崩，必须平滑降力矩并报 err=True
        tau_err, err = compute_p3_gravity_tau(
            q_home, tau, lambda q: [float("nan")] * 6, period=0.005,
            enabled=True, scales=[0.0, 0.8, 0.8, 0.8, 0.0, 0.0],
            max_nm=[0.0, 12.0, 12.0, 2.5, 2.0, 1.0],
            ramp_factor=1.0, slew_nm_s=30.0)
        self.assertTrue(err)
        self.assertAlmostEqual(tau[2] - tau_err[2], 30.0 * 0.005, places=6)

        # 3) 伸展阻尼调度：平滑升至目标且严格落在 [0.8x, 1.8x] 基线内
        base_kp = [72.0, 216.0, 216.0, 50.0, 50.0, 50.0]
        base_kd = [3.6, 12.0, 12.0, 5.0, 4.0, 4.0]
        kp, kd = list(base_kp), list(base_kd)
        for _ in range(100):
            kp, kd = compute_p3_gains(
                base_kp, base_kd, kp, kd, reach_r=0.38, period=0.005,
                enabled=True, r_min=0.18, r_max=0.36,
                kd_near=[4.6, 13.5, 14.0, 5.0, 4.0, 4.0],
                kd_far=[8.0, 14.5, 15.0, 5.0, 4.0, 4.0],  # 8.0 超 1.8*3.6=6.48 必须被钳位
                kp_scale=[1.0] * 6, slew_per_s=10.0)
        self.assertAlmostEqual(kd[0], 1.8 * 3.6, places=4)
        self.assertAlmostEqual(kd[1], 14.5, places=4)
        self.assertAlmostEqual(kd[2], 15.0, places=4)
        self.assertAlmostEqual(kd[3], 5.0, places=4)

        # 4) 静态核验反向保护逻辑
        ok, _ = check_sag_monotonic_safety([0, 0.2, 2.0, 2.7, 0, 0], [0, 0.1, 1.2, 1.6, 0, 0])
        self.assertTrue(ok)
        ok_bad, _ = check_sag_monotonic_safety([0, 0.2, 2.0, 2.7, 0, 0], [0, 0.2, 2.8, 2.7, 0, 0])
        self.assertFalse(ok_bad)

    def test_p3_controller_sends_tau_ff_scheduled_kd_and_fades_at_zero(self):
        import threading
        import time
        from mit_controller_ext import make_extended_controller_class

        FakeBase = self._make_fake_upstream()
        ExtCls = make_extended_controller_class(FakeBase)
        ctrl = ExtCls(ext_cfg={
            "gravity_ff_enable": True,
            "gravity_ff_scale": [0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
            "gravity_ff_ramp_in_s": 0.05,
            "gravity_ff_slew_nm_s": 60.0,
            "gain_sched_enable": True,
            "gain_sched_slew_per_s": 40.0,
        })
        ctrl.set_gravity_torque_fn(lambda q: [0.0, 2.0, 8.0, 2.4, 0.0, 0.0])
        # 设为 HOME 悬臂位姿 + 深伸展半径
        q_home = [-0.43, 0.34, 0.37, 0.09, 0.0, 0.0, 0.0]
        ctrl.target_positions[:] = q_home
        ctrl.command_positions[:] = q_home
        for i, qv in enumerate(q_home):
            ctrl.motors[i].state.pos = qv
        ctrl.set_reach_radius(0.36)

        t = threading.Thread(target=ctrl._control_loop, daemon=True)
        t.start()
        try:
            time.sleep(0.18)
        finally:
            ctrl.worker_stop_event.set()
            t.join(timeout=1.0)

        # J3 应收到约 8.0*0.5 = 4.0 Nm；J4 约 1.2 Nm；J1 tau_ff=0 且 kd 升至 6.4；J7 恒基线
        j1_last = ctrl.motors[0].mit_cmds[-1]
        j3_last = ctrl.motors[2].mit_cmds[-1]
        j4_last = ctrl.motors[3].mit_cmds[-1]
        j7_last = ctrl.motors[6].mit_cmds[-1]
        self.assertEqual(j1_last[4], 0.0)
        self.assertAlmostEqual(j1_last[3], 6.4, delta=0.15)
        self.assertAlmostEqual(j3_last[4], 4.0, delta=0.15)
        self.assertAlmostEqual(j4_last[4], 1.2, delta=0.10)
        self.assertEqual(j7_last[1], 0.0)
        self.assertEqual(j7_last[3], 4.0)
        self.assertEqual(j7_last[4], 0.0)

        # 回零过程（shutdown_started）：到达零位支架时 tau_ff 必须平滑淡出至 0，kd 回基线
        ctrl.shutdown_started = True
        for _ in range(40):
            ctrl._send_mit_positions([0.0] * 7)
        self.assertAlmostEqual(ctrl.motors[2].mit_cmds[-1][4], 0.0, delta=0.05)
        self.assertAlmostEqual(ctrl.motors[0].mit_cmds[-1][3], 3.6, delta=0.15)

    def test_p4_safety_supervisor_collision_torque_thermal_and_soft_hold(self):
        from mit_controller_ext import (
            SafetySupervisor, validate_mit_ext_cfg, make_extended_controller_class)

        cfg = validate_mit_ext_cfg({
            "supervisor_enable": True,
            "track_err_limit_deg": 12.0,
            "track_err_window_s": 0.05,   # 200Hz -> 10 拍
            "tau_limit_nm": [18.0, 28.0, 28.0, 10.0, 10.0, 10.0],
            "tau_limit_window_s": 0.05,
            "thermal_derate_start_c": 90.0,
            "thermal_derate_full_c": 110.0,
            "thermal_derate_min_scale": 0.40,
        })
        sup = SafetySupervisor(cfg, control_hz=200.0)
        q0 = [0.0] * 6
        tau0 = [0.0] * 6
        ok6 = [True] * 6

        # 1) 温升前置降速：80°C=1.0，100°C=0.70，115°C=0.40
        sup.update(q0, q0, tau0, [80.0] * 7, ok6)
        self.assertAlmostEqual(sup.thermal_scale, 1.0)
        sup.update(q0, q0, tau0, [100.0] * 7, ok6)
        self.assertAlmostEqual(sup.thermal_scale, 0.70, places=3)
        sup.update(q0, q0, tau0, [115.0] * 7, ok6)
        self.assertAlmostEqual(sup.thermal_scale, 0.40, places=3)
        self.assertFalse(sup.fault)

        # 2) 短时毛刺（9 拍 < 10 拍）不误触发，恢复即清零
        q_bad = [0.0, 0.0, np.deg2rad(15.0), 0.0, 0.0, 0.0]
        for _ in range(9):
            self.assertFalse(sup.update(q_bad, q0, tau0, [45.0] * 7, ok6))
        self.assertFalse(sup.update(q0, q0, tau0, [45.0] * 7, ok6))
        self.assertFalse(sup.fault)

        # 3) 持续 10 拍跟踪误差超限必须触发软停机锁存
        for _ in range(9):
            self.assertFalse(sup.update(q_bad, q0, tau0, [45.0] * 7, ok6))
        self.assertTrue(sup.update(q_bad, q0, tau0, [45.0] * 7, ok6))
        self.assertTrue(sup.fault)
        self.assertIn("跟踪误差", sup.fault_reason)

        # 清除后测试力矩堵转保护
        sup.clear_fault()
        self.assertFalse(sup.fault)
        tau_stall = [0.0, 0.0, 30.0, 0.0, 0.0, 0.0]
        for _ in range(9):
            self.assertFalse(sup.update(q0, q0, tau_stall, [45.0] * 7, ok6))
        self.assertTrue(sup.update(q0, q0, tau_stall, [45.0] * 7, ok6))
        self.assertTrue(sup.fault)
        self.assertIn("力矩", sup.fault_reason)

        # 4) 端到端控制器测试：堵转软停机锁存必须锚定当前位姿、清零 dq_des，且绝不调用失能！
        FakeBase = self._make_fake_upstream()
        ExtCls = make_extended_controller_class(FakeBase)
        ctrl = ExtCls(ext_cfg=cfg)
        ctrl.target_positions[2] = 0.5
        ctrl.command_positions[2] = 0.3
        ctrl.motors[2].state.pos = 0.3
        ctrl._mit_vel_mode = True
        ctrl._target_velocities[2] = 1.0
        ctrl.motors[2].state.torq = 31.0  # 注入 J3 堵转力矩
        for _ in range(12):
            ctrl._send_and_sample(ctrl.command_positions.copy())
        st = ctrl.get_safety_status()
        self.assertTrue(st["fault"])
        self.assertIn("力矩", st["reason"])
        self.assertEqual(ctrl.stop_calls, [])  # 绝不紧急失能
        self.assertAlmostEqual(ctrl.target_positions[2], ctrl.command_positions[2])
        self.assertFalse(ctrl._mit_vel_mode)
        # 锁存期间新的速度指令被拒绝；清除后可恢复
        ctrl.set_joint_mit_targets([0.0] * 7, [0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0])
        self.assertFalse(ctrl._mit_vel_mode)
        ctrl.clear_safety_fault()
        ctrl.motors[2].state.torq = 5.0
        ctrl.set_joint_mit_targets([0.0] * 7, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(ctrl._mit_vel_mode)


class JointSysidAndProbeTests(unittest.TestCase):
    """v13 P0：单关节阶跃/扫频辨识数学、只读探针评估与日志分析兼容测试。"""

    def test_step_and_sine_identification_recovers_second_order_params(self):
        from tools.joint_sysid import (
            _simulate_filtered_step, analyze_step_response,
            analyze_sine_segment, safe_sine_amp_rad)

        dt = 0.005
        t = np.arange(0.0, 1.4, dt)
        u = np.clip((t - 0.20) / 0.02, 0.0, 1.0)
        step_rad = np.deg2rad(2.5)
        true_delay = 0.015
        true_wn = 2.0 * np.pi * 1.80
        true_zeta = 0.35
        q_cmd = u * step_rad
        q_act = _simulate_filtered_step(t, u, true_delay, true_wn, true_zeta) * step_rad

        res = analyze_step_response(t, q_cmd, q_act, kp=72.0, pre_s=0.15)
        self.assertTrue(res["ok"])
        self.assertTrue(res["fit"]["ok"])
        self.assertGreater(res["fit"]["r2"], 0.995)
        self.assertAlmostEqual(res["fit"]["delay_ms"], 15.0, delta=2.0)
        self.assertAlmostEqual(res["fit"]["wn_hz"], 1.80, delta=0.06)
        self.assertAlmostEqual(res["fit"]["zeta"], 0.35, delta=0.03)
        self.assertGreater(res["overshoot_pct"], 25.0)
        self.assertIsNotNone(res["fit"]["kd_for_zeta_07"])

        # 扫频锁相：在有直流重力偏置的情况下应精确还原 1.8Hz 共振峰值与相位滞后
        f = 1.8
        ts = np.arange(0.0, 4.0 / f, dt)
        amp = safe_sine_amp_rad(2.0, f)
        self.assertLessEqual(2.0 * np.pi * f * amp, 0.80 + 1e-9)
        w = 2.0 * np.pi * f
        H = (true_wn ** 2 / ((true_wn ** 2 - w ** 2) + 1j * 2.0 * true_zeta * true_wn * w)
             * np.exp(-1j * w * true_delay))
        qc = amp * np.sin(w * ts)
        qa = amp * abs(H) * np.sin(w * ts + np.angle(H)) + 0.035  # 2° 直流下垂
        sr = analyze_sine_segment(ts, qc, qa, f)
        self.assertTrue(sr["ok"])
        self.assertAlmostEqual(sr["mag_ratio"], abs(H), delta=0.02)
        self.assertAlmostEqual(sr["phase_deg"], np.rad2deg(np.angle(H)), delta=1.5)

    def test_probe_evaluation_verdict_and_gravity_sign_check(self):
        from tools.mit_feedback_probe import evaluate_probe_metrics

        mode_a = {"valid_ratio": 1.0, "period_p99_ms": 5.3, "change_ratio": 0.85}
        mode_b = {"valid_ratio": 1.0, "period_p99_ms": 5.8, "change_ratio": 0.90}
        q_ref = np.array([0.0, 0.35, 0.37, 0.10, 0.0, 0.0])
        q_cache = q_ref + np.deg2rad([0.02, -0.04, 0.03, 0.01, 0.0, 0.0])
        q_cmd = q_ref + np.deg2rad([0.0, 1.8, -0.9, 0.4, 0.0, 0.0])
        tau_pin = np.array([0.0, 6.5, -2.8, 0.45, 0.05, 0.0])
        tau_act = np.array([0.02, 7.8, -3.2, 0.55, 0.03, 0.0])
        kp = np.array([72.0, 216.0, 216.0, 50.0, 50.0, 50.0])
        ev = evaluate_probe_metrics(mode_a, mode_b, q_cache, q_ref, q_cmd,
                                    tau_act, tau_pin, kp)
        self.assertTrue(ev["pos_match_ok"])
        self.assertTrue(ev["recommend_feedback_trusted"])
        self.assertFalse(ev["recommend_feedback_poll_once"])
        # J2/J3/J4 符号全部一致
        self.assertTrue(ev["gravity_check"][1]["sign_match"])
        self.assertTrue(ev["gravity_check"][2]["sign_match"])
        self.assertTrue(ev["gravity_check"][3]["sign_match"])

        # 若位置偏差 > 0.3° 必须拒绝启用 trusted
        ev_bad = evaluate_probe_metrics(
            mode_a, mode_b, q_ref + np.deg2rad(1.0), q_ref, q_cmd,
            tau_act, tau_pin, kp)
        self.assertFalse(ev_bad["pos_match_ok"])
        self.assertFalse(ev_bad["recommend_feedback_trusted"])

    def test_analyze_servo_log_summarizes_mit_cached_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mit_p0.csv")
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=DIAG_FIELDS)
                writer.writeheader()
                for i in range(10):
                    row = {k: "" for k in DIAG_FIELDS}
                    row.update({
                        "host_time": f"{i * 0.01:.6f}", "dt_s": "0.010000",
                        "following": 1, "visible": 1, "ff_moving": 1,
                        "error_norm": 0.05, "ctrl_work_ms": 0.5, "overrun": 0,
                        "fbc_valid": 1, "fbc_age_ms": 2.5, "fbc_seq": i + 1,
                        "tmos_max": 42.5 + i * 0.1, "mit_period_p99_ms": 5.3,
                        "can_fail_total": 1,
                    })
                    for j in range(1, 7):
                        row[f"q_cmd_{j}"] = 0.2
                        row[f"qa_{j}"] = 0.2 - np.deg2rad(1.5 if j == 2 else 0.2)
                        row[f"dqa_{j}"] = 0.0
                        row[f"tau_{j}"] = 6.0 if j == 2 else 0.5
                    writer.writerow(row)
            summary = summarize(path)
            self.assertAlmostEqual(summary["mit_fbc_valid_ratio"], 1.0)
            self.assertAlmostEqual(summary["mit_tracking_err_median_deg"], 1.5, delta=0.05)
            self.assertAlmostEqual(summary["mit_tau_abs_max_nm"][1], 6.0)
            self.assertAlmostEqual(summary["mit_tmos_max_c"], 43.4, delta=0.05)
            self.assertEqual(summary["mit_can_fail_total"], 1)

    def test_p2_actual_velocity_filter_and_real_pose_sag_cancellation(self):
        # 1) ActualVelocityFilter：有效抑制 0.06 rad/s 静止白噪、保留真实速度、钳位尖峰并在失效时复位
        filt = ActualVelocityFilter(tau_s=0.020, max_speed=np.array([2.5, 2.0, 2.0, 3.0, 3.0, 2.5]))
        rng = np.random.default_rng(42)
        noise_outs = [filt.update(rng.normal(0.0, 0.06, size=6), dt=0.01) for _ in range(100)]
        self.assertLess(float(np.std(noise_outs)), 0.045)
        # 真实匀速阶跃后快速收敛且被硬钳制在 max_speed 内
        for _ in range(15):
            out = filt.update(np.array([5.0, 1.0, 0.0, 0.0, 0.0, 0.0]), dt=0.01)
        self.assertAlmostEqual(out[0], 2.5, delta=0.02)
        self.assertAlmostEqual(out[1], 1.0, delta=0.02)
        # 异常/失效输入必须返回 None 并复位内部状态
        self.assertIsNone(filt.update(np.array([np.nan] * 6), dt=0.01))
        self.assertIsNone(filt.update(np.zeros(6), dt=0.01, valid=False))
        out_fresh = filt.update(np.zeros(6), dt=0.01, valid=True)
        self.assertTrue(np.allclose(out_fresh, 0.0))

        # 2) 静态重力下垂自洽性：pose_history 存真实 q_act（含 4.7° 俯仰下垂），
        #    前向查表与 smith_blend=1.0 反投影同用 q_act 时，静止掌心反投影误差精确为 0
        hist = CameraPoseHistory(lpf_tau_s=0.012)
        ang = np.deg2rad(4.7)
        R_sag = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ang), -np.sin(ang)],
            [0.0, np.sin(ang), np.cos(ang)],
        ])
        p_sag = np.array([0.14, 0.06, 0.29])
        T_tc = np.eye(4)
        for k in range(10):
            hist.push(k * 0.01, p_sag, R_sag, T_tc, dt=0.01)
        p_d, R_d = hist.lookup(0.09 - 0.025, use_lpf=True)
        p_now, R_now = hist.blended_pose(0.09, delay_s=0.025, smith_blend=1.0)
        palm_cam = np.array([0.0, 0.0, 0.40])
        palm_w = p_d + R_d @ palm_cam
        reproj = R_now.T @ (palm_w - p_now)
        self.assertTrue(np.allclose(reproj, palm_cam, atol=1e-9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
