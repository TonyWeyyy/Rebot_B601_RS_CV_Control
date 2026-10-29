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
                          StaticVelocityHold, adaptive_feedforward_scale,
                          camera_twist_from_tcp, compensate_palm_velocity,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
