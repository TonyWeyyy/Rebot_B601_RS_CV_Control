#!/usr/bin/env python3
"""升级逻辑的离线单元测试；不导入相机、MediaPipe 或机械臂 SDK。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servo_timing import (AdaptiveLeadGate, FastStartGate,
                          scaled_alpha, scaled_decay, horizontal_radial_soft_wall,
                          measurement_age, dynamic_lead_seconds)


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
