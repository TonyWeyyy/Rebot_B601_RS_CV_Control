# AGENTS.md

## Overview
Eye-in-hand visual servoing + pinch grasping for the Seeed reBot Arm B601-RS (6-DOF + J7 gripper on CAN `can0`) and Orbbec Gemini 2 RGB-D camera.
- Main entrypoint: `palm_follow_grasp.py` (3-thread architecture: 60fps vision/main thread, 100Hz servo thread, `pynput` keyboard thread, plus optional low-rate feedback/logging threads).
- Legacy follow-only script: `palm_follow_fixed_orientation.py` (kept for baseline comparison; changes to core servo logic mainly target `palm_follow_grasp.py`, `servo_timing.py`, `kalman_target.py`, and `rebot_adapter.py`).

## Verification & Testing (Run Before Hardware)
Always run the offline test suite after modifying control, timing, filtering, or lifecycle code—neither test requires camera or CAN hardware:
```bash
# 1. Unit tests for timing, KF, adaptive gates, eye-in-hand compensation, soft walls, and log analysis
python3 tools/test_servo_upgrades.py
# Run a single test class or method:
python3 -m unittest tools.test_servo_upgrades.EyeInHandCompensationTests

# 2. Full main() smoke test with mocked SDKs/hardware (tests normal, CAN fault, --no-gripper, eye-in-hand static)
python3 tools/offline_smoke_test.py

# 3. Analyze 100Hz servo CSV logs (excludes first 0.30s settle window after stopping from static metrics)
python3 tools/analyze_servo_log.py logs/servo_<timestamp>.csv --json
```

## External Dependencies & Environment Gotchas
- **External Source Repos**: `rebot_adapter.py` loads two non-pip repos via hardcoded paths at the top of the file:
  - `REBOT_CONTROL_PATH = ~/wey/rebot_developer/MIT_control/rebot_control` (imports `rebot.ReBotRSMITController`)
  - `URDF_PATH = ~/wey/rebot_developer/reBotArm_control_py/urdf/RS/urdf/ReBot_Arm_RS.urdf`
- **ROS Environment Conflict**: `/opt/ros/*` in `PYTHONPATH` or `LD_LIBRARY_PATH` breaks conda's `pinocchio` (Python version mismatch segfault). Strip `/opt/ros` paths before running with hardware/Pinocchio.
- **Orbbec SDK v2**: Package is `pyorbbecsdk2`, imported as `pyorbbecsdk`. Uses Orbbec `Pipeline` with hardware D2C alignment (not `cv2.VideoCapture`).
  - **Camera Distortion**: Never use SDK-reported distortion coefficients with OpenCV (`k1/k2` are invalid for Brown model); always force `dist_coeffs = np.zeros(5)`.
  - **60fps Mode**: Requires switching Gemini 2 depth work mode to `Binned Sparse Default` (handled automatically in `gemini2_camera.py` when `fps > 30`).
- **MediaPipe Tasks API**: Uses `mp.tasks.vision.HandLandmarker` in `VIDEO` mode (`config/hand_landmarker.task`) with monotonic ms timestamps. Do not use deprecated `mp.solutions.hands`.
- **OpenCV 5.0 Compatibility**: `cv2.calibrateHandEye` was removed in OpenCV 5.0; calibration scripts use `tools/hand_eye_solver.py`.

## Hardware Safety Red Lines (Do Not Violate)
1. **Safe Shutdown Sequence**: Never disable motors in a non-zero / bent-elbow pose (causes free-fall crash). Every exit path must execute: stop control thread → close feedback sampler (`feedback_sampler.close()`) → flush diagnostics → `robot.hold()` → `robot.stop()` (`return_to_zero=True, wait=True`).
2. **Single CAN Master on `can0`**: J7 gripper commands MUST go through `rebot_adapter.set_gripper_deg()`, which packs J7 as the 7th channel of `set_joint_angles()` at 100Hz. Never open a second `motorbridge` instance.
3. **Unit Boundary**: `rebot_control` SDK uses **degrees** (7 joints); Pinocchio and control math use **radians** (6 joints). All degree/radian conversions and `config/joint_offset.json` FK corrections must stay encapsulated inside `rebot_adapter.py`.
4. **4-Layer Depth Protection Chain**: Do not remove or bypass the depth guards in `palm_follow_grasp.py`: (1) 7×7→14×14→21×21 median window fallback, (2) jump rejection `|Δz| > z3d_max_jump`, (3) per-frame rate limit `±z3d_max_rate`, (4) 3-frame consistency check (`reacq_buf`) after long loss (≥30 frames) or startup.
5. **Workspace & Speed Limits**:
   - `workspace.max_radius` (≤ 0.42m, 70% reach rule to prevent J2 stall) applies **only to horizontal radial distance** `hypot(X, Y)` via `horizontal_radial_soft_wall()`, preserving inward, tangential, and vertical Z motion. Exceeding hard boundaries by >5cm triggers automatic slow return to HOME.
   - Cartesian `max_vel` (≤ 0.85 m/s) and per-joint limits `max_vel_joints` are doubly enforced in `validate_follow_cfg()` and `rebot_adapter.py` (`HARD_MAX_JOINT_SPEED = [2.5, 2.0, 2.0, 3.0, 3.0, 2.5]`, strictly keeping gravity-bearing `J2/J3 <= 2.0 rad/s`; docking/extension/oscillation tapers back to `<= 2.0 rad/s` on all 6 axes, with `apply_joint_limit_cushion()` near `q_min/q_max`).
6. **Time-Derived Frame Constants**: Never hardcode tick counts for durations. Always derive frame counts from `CTRL_RATE` (e.g., `round(duration_s / CTRL_RATE)`) and scale EMA/decay factors with actual `dt` (`scaled_alpha`, `scaled_decay`).

## Control & Configuration Conventions
- **Parameter Precedence**: CLI flags > `config/follow.yaml` & `config/grasp.yaml` > module-level defaults in `palm_follow_grasp.py`.
  - When adding a config parameter to `follow.yaml`, you must: (1) define the module-level default in `palm_follow_grasp.py`, (2) declare it `global` inside `apply_follow_cfg()`, and (3) validate bounds/NaN in `validate_follow_cfg()`.
- **Fallback Configs**: Keep `config/follow_v12.yaml`, `config/follow_v11.yaml`, `config/follow_v10.yaml`, `config/follow_v9_fallback.yaml`, `config/follow_v8_fallback.yaml`, and `config/follow_v7.1_fallback.yaml` valid against `validate_follow_cfg()` (verified by `offline_smoke_test.py`).
- **Stationary & Eye-in-Hand Behavior (v12.0+)**:
  - `KalmanTarget3D` updates full 3D only when `target_seq` increments on a fresh 3D measurement; during `frozen=1` (fresh 2D landmarks, temporarily invalid depth), `update_partial()` updates ONLY the XY subspace (orthogonal to camera optical axis in world frame) without ever contaminating Z.
  - Lag-aligned World-Frame KF (`CameraPoseHistory` + `kf_world` + Smith-predictor blended projection) runs alongside `DelayedTwistCompensator`, combined via asymmetric dual-estimator gating (`consensus_velocity` for wake-up/near-stop/FF/lead; both-below-off confirmation for mid-air reversals), paired with `PostBrakeSettleGuard`, `compute_reach_schedule` (asymmetric `reach_release_tau`), `OscillationDamper` (decay-on-inactive, `confirm_reversals=3`, `large_err_release=0.15m`), and `compute_weighted_damped_pseudoinverse` (soft orientation cruise weight with tilt guard).
  - `PalmDetector` supports hysteresis ROI cropping (`roi_crop_enable`, `compute_roi_box`, `remap_roi_landmarks`) with immediate same-frame full-frame fallback and strictly monotonic VIDEO timestamps.
  - When `ff_moving=False`, the arm holds its current pose (`kf_slow_follow_enable: false`).
- **Gripper Coupling**:
  - `--no-gripper` actively opens the gripper to `gripper.open_deg` on startup to prevent camera occlusion.
  - Pinch updates require fresh 2D landmarks (`track['lost'] == 0`, regardless of depth validity) and freeze while arm joint speed exceeds `move_speed_threshold` (`hold_while_moving: true`) to prevent J7 vibration from shaking the wrist camera.
- **Diagnostics**: 100Hz servo thread logs via non-blocking `put_nowait` (`DiagnosticRecorder`). Synchronous CAN reads (`ActualFeedbackSampler`, `diag_actual_feedback_hz` ≤ 2.0Hz, default `0.0`) must never run inside the 100Hz control loop.
