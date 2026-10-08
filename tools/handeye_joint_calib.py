#!/usr/bin/env python3
"""
手眼 + 关节零位 联合标定（B601 RS + Gemini 2）
================================================
为什么需要它：
    诊断显示：旋转链一致（SDK 角度↔实际旋转 1:1），但 FK 平移比实物
    系统性地小 ~30% 且比例随关节不同 —— 典型症状是 SDK 关节零位与
    URDF 关节零位存在固定偏差 δ（每关节一个小角度）。
    普通手眼标定假设 FK 完全正确，因此永远收敛不了；
    本脚本把 δ 和 T_tcp_camera 放进同一个优化里联合求解。

数学：
    标记固定于基座系 → 对每个样本 i：
        T_bm_i = FK(q_i + δ) · X · T_m2c_i   必须全部相等
    优化变量 θ = [δ(6), X 的 rotvec(3) + t(3)]，共 12 维
    残差 = 各样本 T_bm_i 相对 T_bm_0 的 se(3) 对数，scipy 最小二乘

安全：
    - 仅在示教位姿附近做小幅组合扰动（每关节 ≤ 幅度表），全部限位内
    - 每 6 个位姿回零休息 40 秒（电机 3 散热）
    - 退出路径与标定脚本相同（保持 → 缓慢回零 → 失能）

用法：
    python handeye_joint_calib.py            # 需已存在 viewpoint.json
产物：
    hand_eye_calibration.json   （T_tcp_camera，跟随脚本直接用）
    joint_offset.json           （6 关节零位修正，需合入 rebot_adapter.py）
"""

import os
import sys
import json
import time

import numpy as np
import cv2

# 项目根目录（tools/ 的上一级）加入模块搜索路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from gemini2_camera import Gemini2Camera
from rebot_adapter import RebotAdapter, URDF_PATH, B601_RS_JOINT_NAMES
from hand_eye_calibration_auto import ArucoTracker, VIEWPOINT_FILE

import pinocchio as pin
from scipy.optimize import least_squares

# ── 采样参数 ────────────────────────────────────────────────
N_POSES = 16                          # 组合扰动位姿数（样本 = N+1 含示教位姿）
PERTURB_DEG = [10, 8, 8, 10, 10, 25]  # 各关节扰动幅度（度），组合扰动
RANDOM_SEED = 7
MOVE_SPEED_DEG = 12.0
SETTLE_S = 0.5
AVG_FRAMES = 12
POSE_TIMEOUT_S = 3.0

REST_EVERY = 6                        # 每 6 个位姿回零休息
REST_S = 40.0

OUT_CAL = os.path.join(_ROOT, "config", "hand_eye_calibration.json")
OUT_OFF = os.path.join(_ROOT, "config", "joint_offset.json")
SAMPLES_NPZ = os.path.join(_ROOT, "config", "handeye_joint_samples.npz")

# 优化约束：δ 物理上应是小角度；J1(竖直基轴)/J6(腕滚轴)可观测性弱，
# 无约束时最小二乘会在这两个方向发散到荒唐值 → 边界 + 弱正则拉回 0
DELTA_BOUND_RAD = np.deg2rad(12.0)
DELTA_REG = 0.05                      # 正则权重：每 1° 零位偏差≈0.05 残差


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


class FKModel:
    """独立 FK（pinocchio），支持零位偏置 δ，供优化器高频调用。"""

    def __init__(self):
        self.model = pin.buildModelFromUrdf(os.path.expanduser(URDF_PATH))
        self.data = self.model.createData()
        self.idx_q = [self.model.joints[self.model.getJointId(n)].idx_q
                      for n in B601_RS_JOINT_NAMES]
        self.jid6 = self.model.getJointId(B601_RS_JOINT_NAMES[-1])

    def tcp(self, q6, delta):
        q_full = np.zeros(self.model.nq)
        for k, iq in enumerate(self.idx_q):
            q_full[iq] = q6[k] + delta[k]
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        oMi = self.data.oMi[self.jid6]
        T = np.eye(4)
        T[:3, :3] = oMi.rotation
        T[:3, 3] = oMi.translation
        return T


def sample_avg(tracker, cam, n=AVG_FRAMES):
    """平均 n 帧标记位姿（位置平均；小动作下旋转用中位帧）。"""
    ts, Rs = [], []
    t_end = time.time() + POSE_TIMEOUT_S
    while len(ts) < n and time.time() < t_end:
        color, _, _ = cam.get_frames()
        if color is None:
            continue
        try:
            T, annotated = tracker.detect(color)
        except Exception:
            continue
        cv2.imshow('Joint Calib', annotated)
        cv2.waitKey(1)
        if T is None:
            continue
        ts.append(T[:3, 3].copy())
        Rs.append(T[:3, :3].copy())
    if not ts:
        return None
    T_out = np.eye(4)
    T_out[:3, :3] = Rs[len(Rs) // 2]
    T_out[:3, 3] = np.mean(ts, axis=0)
    return T_out


def main():
    if not os.path.exists(VIEWPOINT_FILE):
        print(f"✗ 未找到 {VIEWPOINT_FILE}，先跑一次 hand_eye_calibration_auto.py 示教")
        return
    with open(VIEWPOINT_FILE) as f:
        q_view = np.array(json.load(f)['q_view'], dtype=np.float64)

    fk = FKModel()
    reuse = "--reuse" in sys.argv
    qs, Ts = [], []

    if reuse and os.path.exists(SAMPLES_NPZ):
        # ── 复用已存样本，只重跑优化（不动机械臂、不开相机）────────
        d = np.load(SAMPLES_NPZ)
        qs = [q for q in d['qs']]
        Ts = [T for T in d['Ts']]
        print(f"[复用] 从 {SAMPLES_NPZ} 载入 {len(qs)} 组样本，跳过采样")
        robot = None
        cam = None
        tracker = None
    else:
        cam = Gemini2Camera()
        tracker = ArucoTracker(cam.camera_matrix, cam.dist_coeffs)
        robot = RebotAdapter()
        time.sleep(1.0)

    try:
        print("=" * 60)
        print("手眼 + 关节零位 联合标定")
        print("=" * 60)

        if robot is not None:
            # ── 生成组合扰动位姿（限位内 + 无需 IK）──────────
            rng = np.random.default_rng(RANDOM_SEED)
            amp = np.deg2rad(np.array(PERTURB_DEG))
            poses = []
            tries = 0
            while len(poses) < N_POSES and tries < 500:
                tries += 1
                q = q_view + rng.uniform(-amp, amp)
                q = np.clip(q, robot.q_min, robot.q_max)
                if np.all(np.abs(q - q_view) > np.deg2rad(2.0)):
                    poses.append(q)
            print(f"[准备] {len(poses)} 个扰动位姿已生成")

            # ── 示教位姿采样（样本 0，残差基准）───────────────
            print("[采样] 移动到示教位姿（基准样本）...")
            robot.move_to_joints(q_view, speed_deg_s=MOVE_SPEED_DEG,
                                 wait=True, timeout=60.0)
            time.sleep(SETTLE_S)
            T0 = sample_avg(tracker, cam)
            if T0 is None:
                print("✗ 示教位姿看不到标记，请重新示教")
                return
            qs.append(robot.get_current_pos())
            Ts.append(T0)
            print(f"  ✓ 基准样本，标记距离 {np.linalg.norm(T0[:3,3])*1000:.0f}mm")

            # ── 扰动位姿采样 ────────────────────────────────
            for idx, q in enumerate(poses):
                print(f"[采样 {idx+1}/{len(poses)}]")
                robot.move_to_joints(q, speed_deg_s=MOVE_SPEED_DEG,
                                     wait=True, timeout=60.0)
                time.sleep(SETTLE_S)
                T = sample_avg(tracker, cam)
                if T is None:
                    print("  看不到标记，跳过")
                else:
                    qs.append(robot.get_current_pos())
                    Ts.append(T)
                    print(f"  ✓ 样本 {len(qs)}，标记距离 "
                          f"{np.linalg.norm(T[:3,3])*1000:.0f}mm")

                if idx + 1 < len(poses) and (idx + 1) % REST_EVERY == 0:
                    print(f"[散热] 回零休息 {REST_S:.0f}s...")
                    robot.move_to_joints(np.zeros(6), speed_deg_s=MOVE_SPEED_DEG,
                                         wait=True, timeout=60.0)
                    t_end = time.time() + REST_S
                    while time.time() < t_end:
                        if cv2.waitKey(200) & 0xFF in (ord('q'), 27):
                            break
                    print("[散热] 继续")

            print(f"\n采样完成：{len(qs)} 组")
            np.savez(SAMPLES_NPZ, qs=np.array(qs), Ts=np.array(Ts))
            print(f"样本已存 {SAMPLES_NPZ}（下次 --reuse 免采样）")

        n = len(qs)
        if n < 10:
            print(f"✗ 样本不足 10 组（{n}），无法联合优化")
            return

        # ── 联合优化：δ (6) + X (6)，带边界 + 弱正则 ─────────
        print("[优化] 联合求解关节零位偏差 + T_tcp_camera ...")
        qs = np.array(qs)

        def unpack(theta):
            delta = theta[:6]
            X = np.eye(4)
            X[:3, :3], _ = cv2.Rodrigues(theta[6:9])
            X[:3, 3] = theta[9:12]
            return delta, X

        def residuals(theta):
            delta, X = unpack(theta)
            T_bm = [fk.tcp(qs[i], delta) @ X @ Ts[i] for i in range(n)]
            res = []
            for i in range(1, n):
                E = np.linalg.inv(T_bm[0]) @ T_bm[i]
                rvec, _ = cv2.Rodrigues(E[:3, :3])
                res.append(rvec.reshape(3) * 3.0)      # 旋转加权（弧度）
                res.append(E[:3, 3] * 20.0)            # 平移（米→cm 量级加权）
            # 弱正则：把可观测性弱的方向（J1/J6）拉回 0，
            # 不压制有数据支撑的真实偏差
            res.append(np.sqrt(DELTA_REG) * delta)
            return np.concatenate(res)

        theta0 = np.zeros(12)
        theta0[9:12] = [0.05, 0.0, -0.10]              # X 平移初值（相机在腕部附近）
        lb = np.full(12, -np.inf)
        ub = np.full(12, np.inf)
        lb[:6], ub[:6] = -DELTA_BOUND_RAD, DELTA_BOUND_RAD
        sol = least_squares(residuals, theta0, method='trf',
                            bounds=(lb, ub), max_nfev=2000)
        delta, X = unpack(sol.x)

        # ── 效果对比：修正前后标记基座系坐标分散度 ────────────
        def spread_cm(d, X_):
            pts = np.array([(fk.tcp(qs[i], d) @ X_ @
                             np.append(Ts[i][:3, 3], 1.0))[:3] for i in range(n)])
            return np.linalg.norm(pts.std(axis=0)) * 100, pts

        X0 = np.eye(4); X0[:3, 3] = theta0[9:12]
        s_before, _ = spread_cm(np.zeros(6), X0)
        s_after, pts = spread_cm(delta, X)

        print("\n" + "=" * 60)
        print("[结果]")
        print(f"  关节零位偏差 δ (度): {np.rad2deg(delta).round(2).tolist()}")
        print(f"  T_tcp_camera 平移 (m): {X[:3,3].round(4).tolist()}")
        print(f"  分散度: 修正前 {s_before:.2f} cm → 修正后 {s_after:.2f} cm")
        marker_base = pts.mean(axis=0)
        print(f"  标记在基座系位置: {marker_base.round(3).tolist()}")
        print("=" * 60)

        verdict_ok = s_after < 2.0 and np.all(np.abs(np.rad2deg(delta)) < 12)
        if verdict_ok:
            print("✓✓ 联合标定成功")
        elif s_after < 2.0:
            print("⚠ 分散度达标但 δ 偏大（>15°），需人工核对合理性")
        else:
            print("✗ 修正后分散度仍 >2cm：病灶不是零位偏差，")
            print("  可能是 URDF 连杆几何与实物不符 → 考虑切官方抓取栈路线")

        # 无论成败都保存（附带质量字段，供我判读）
        with open(OUT_CAL, 'w') as f:
            json.dump({'T_tcp_camera': X.tolist(),
                       'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'num_samples': n,
                       'residual_spread_cm': round(float(s_after), 2),
                       'method': 'joint_offset_handeye_least_squares'}, f, indent=2)
        with open(OUT_OFF, 'w') as f:
            json.dump({'joint_offset_rad': delta.tolist(),
                       'joint_offset_deg': np.rad2deg(delta).round(3).tolist(),
                       'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
        print(f"\n已保存 {OUT_CAL} 和 {OUT_OFF}")
        print("把完整输出贴给我，我来裁决是否需要把 δ 合入 rebot_adapter.py")

    finally:
        cv2.destroyAllWindows()
        if cam is not None:
            cam.stop()
        if robot is not None:
            try:
                robot.hold()
            except Exception:
                pass
            robot.stop()
            print("程序退出（已安全回零）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消（机械臂将由 rebot_control 平滑回零）")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
