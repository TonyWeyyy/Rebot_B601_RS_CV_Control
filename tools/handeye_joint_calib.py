#!/usr/bin/env python3
"""
手眼外参 + 6 关节零位偏置 联合标定工具（reBot Arm B601-RS + Orbbec Gemini 2）。
===============================================================================
背景与设计动机：
    体检诊断显示：旋转链一致（SDK 关节角增减与实际旋转 1:1 吻合），但 FK 平移
    相对实物存在系统性偏差且比例随关节不同 —— 典型成因是 SDK 关节零位与 URDF
    零位存在固定偏置 δ（每关节一个小角度，单位：rad）。
    传统 AX=XB 手眼标定默认 FK 绝对精确，因此无法收敛；本脚本将 6 维零位偏置 δ
    与 6 自由度手眼外参 X = T_tcp_camera 放入同一非线性最小二乘中联合优化。

坐标系与数学模型：
    标定标记固定于基座系不动 → 对所有样本 i = 0, ..., N-1：
        T_base_marker_i = FK(q_i + δ) · T_tcp_camera · T_cam_marker_i   ≡ const
    - 优化变量 θ ∈ R¹²：[δ(6 维, rad), log(R_X)(3 维, rad), t_X(3 维, m)]
    - 残差构造：对 i = 1..N-1 计算相对基准样本 0 的位姿差 E_i = T_bm_0⁻¹ · T_bm_i，
      提取旋转向量（×3.0 加权）与平移向量（×20.0 加权），并附加对 δ 的弱 L2 正则
      √DELTA_REG · δ（防止可观测性弱的 J1/J6 轴向漂移），用 SciPy TRF 有界求解。

安全说明：
    - 仅在已示教的 config/viewpoint.json 附近做软限位内小幅随机扰动（无需 IK）
    - 每采 6 个位姿自动回零休息 40s，防止 J3 负重关节电机过热
    - 退出路径走标准安全停机（hold → stop 平滑回零 → 失能）

用法：
    python tools/handeye_joint_calib.py           # 现场采样 + 联合优化（需存在 viewpoint.json）
    python tools/handeye_joint_calib.py --reuse   # 离线复用 handeye_joint_samples.npz 重跑优化

输出产物：
    - config/hand_eye_calibration.json  （T_tcp_camera，主程序直接读取）
    - config/joint_offset.json          （6 轴零位修正 δ，由 rebot_adapter.py 封装加载）
    - config/handeye_joint_samples.npz  （原始样本缓存，供 --reuse 离线复算）
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

# ── 采样与运动控制参数 ───────────────────────────────────────
N_POSES = 16                          # 组合扰动位姿数（总样本上限 = N_POSES + 1，含基准示教位姿）
PERTURB_DEG = [10, 8, 8, 10, 10, 25]  # 各关节最大扰动半幅（单位：deg）
RANDOM_SEED = 7                       # 固定随机种子，保证采样位姿序列可复现
MOVE_SPEED_DEG = 12.0                 # 关节插补移动速度（单位：deg/s）
SETTLE_S = 0.5                        # 到位稳态等待时间（单位：s）
AVG_FRAMES = 12                       # 每个位姿平均的 ArUco 观测帧数
POSE_TIMEOUT_S = 3.0                  # 单个位姿观测超时（单位：s）

REST_EVERY = 6                        # 每采样 6 个位姿强制回零散热一次
REST_S = 40.0                         # 散热停留时长（单位：s，保护重力承载关节 J3）

OUT_CAL = os.path.join(_ROOT, "config", "hand_eye_calibration.json")
OUT_OFF = os.path.join(_ROOT, "config", "joint_offset.json")
SAMPLES_NPZ = os.path.join(_ROOT, "config", "handeye_joint_samples.npz")

# 优化边界与正则化参数：
# δ 物理上应为小角度；J1（竖直基轴）与 J6（末端滚转轴）在固定视野下可观测性弱，
# 无约束时最小二乘容易沿这两轴发散，因此施加盒约束 ±12° 与弱 L2 正则拉回 0
DELTA_BOUND_RAD = np.deg2rad(12.0)    # 各关节零位偏置优化上下界（单位：rad）
DELTA_REG = 0.05                      # 弱 Tikhonov 正则权重


def rot_angle_deg(R):
    """计算 3×3 旋转矩阵的等效轴角转角（单位：deg）。

    参数:
        R (np.ndarray): 3×3 正交旋转矩阵。

    返回:
        float: 旋转角，范围 [0.0, 180.0]（单位：deg）。
    """
    c = (np.trace(R) - 1.0) / 2.0
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


class FKModel:
    """独立的 Pinocchio 正向运动学模型，支持叠加 6 轴零位偏置 δ，供非线性优化高频调用。

    设计说明：
        不直接调用 RebotAdapter.forward_kinematics()，一是为了在 `--reuse` 离线模式下
        完全脱离机械臂硬件运行，二是避免优化迭代期间干扰控制线程的 Pinocchio data 缓存。
    """

    def __init__(self):
        """从 URDF 构建 Pinocchio 模型并缓存 6 个主动关节的配置向量索引及末端关节 ID。"""
        self.model = pin.buildModelFromUrdf(os.path.expanduser(URDF_PATH))
        self.data = self.model.createData()
        self.idx_q = [self.model.joints[self.model.getJointId(n)].idx_q
                      for n in B601_RS_JOINT_NAMES]
        self.jid6 = self.model.getJointId(B601_RS_JOINT_NAMES[-1])

    def tcp(self, q6, delta):
        """计算施加零位偏置 δ 后的末端法兰（joint6）位姿矩阵 T_base_tcp。

        参数:
            q6 (np.ndarray): 长度 6 的实测关节角向量（单位：rad）。
            delta (np.ndarray): 长度 6 的关节零位偏置向量（单位：rad）。

        返回:
            np.ndarray: 4×4 齐次变换矩阵 T_base_tcp（TCP → base，平移单位：m）。
        """
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
    """在当前静止位姿下连续采集最多 n 帧 ArUco 标记观测并求稳健平均位姿。

    位置采用算术均值以压低深度/角点随机噪声，旋转取中位序号帧避免 SO(3) 非线性平均开销。

    参数:
        tracker (ArucoTracker): ArUco 标记检测器。
        cam (Gemini2Camera): 已打开的 Gemini 2 相机实例。
        n (int): 目标有效帧数（默认 AVG_FRAMES），受 POSE_TIMEOUT_S 超时限制。

    返回:
        np.ndarray | None: 4×4 齐次矩阵 T_cam_marker（平移单位：m）；若超时无检测返回 None。
    """
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
    """主流程：采集（或离线复用）多组关节角与标记位姿，执行 12 维联合优化并落盘结果。"""
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
            # ── 生成组合扰动位姿（全部裁剪至软限位内，无需 IK）────────
            rng = np.random.default_rng(RANDOM_SEED)
            amp = np.deg2rad(np.array(PERTURB_DEG))
            poses = []
            tries = 0
            while len(poses) < N_POSES and tries < 500:
                tries += 1
                q = q_view + rng.uniform(-amp, amp)
                q = np.clip(q, robot.q_min, robot.q_max)
                # 要求 6 个关节相对基准均有 >2° 变化，保证对各轴零位偏差都有激励
                if np.all(np.abs(q - q_view) > np.deg2rad(2.0)):
                    poses.append(q)
            print(f"[准备] {len(poses)} 个扰动位姿已生成")

            # ── 示教位姿采样（样本 0，作为残差比对基准）───────────────
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

            # ── 逐个扰动位姿自动采样（含周期性回零散热）───────────────
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

                # 每 REST_EVERY 个位姿回零卸掉重力力矩，保护 J3 电机散热
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

        # ── 联合优化：δ (6) + X (6)，带盒约束 + 弱正则 ───────────
        print("[优化] 联合求解关节零位偏差 + T_tcp_camera ...")
        qs = np.array(qs)

        def unpack(theta):
            """将 12 维优化向量拆解为关节零位偏置 δ(rad) 与 4×4 外参 X(T_tcp_camera)。"""
            delta = theta[:6]
            X = np.eye(4)
            X[:3, :3], _ = cv2.Rodrigues(theta[6:9])
            X[:3, 3] = theta[9:12]
            return delta, X

        def residuals(theta):
            """构造非线性最小二乘残差向量：各样本相对基准样本 0 的 se(3) 误差 + δ 正则项。"""
            delta, X = unpack(theta)
            # 计算所有样本推算出的固定标记在基座系下的位姿 T_base_marker_i
            T_bm = [fk.tcp(qs[i], delta) @ X @ Ts[i] for i in range(n)]
            res = []
            for i in range(1, n):
                # 理想情况下所有 T_bm[i] 相等，E = T_bm[0]⁻¹ · T_bm[i] 应为单位阵 I₄
                E = np.linalg.inv(T_bm[0]) @ T_bm[i]
                rvec, _ = cv2.Rodrigues(E[:3, :3])
                res.append(rvec.reshape(3) * 3.0)      # 旋转误差加权（弧度）
                res.append(E[:3, 3] * 20.0)            # 平移误差加权（米→约 5cm 对齐 1 rad 量级）
            # 弱正则：把可观测性弱的方向（J1/J6）拉回 0，
            # 不压制有数据支撑的真实偏差
            res.append(np.sqrt(DELTA_REG) * delta)
            return np.concatenate(res)

        theta0 = np.zeros(12)
        theta0[9:12] = [0.05, 0.0, -0.10]              # X 平移初值（相机支架位于法兰附近，单位：m）
        lb = np.full(12, -np.inf)
        ub = np.full(12, np.inf)
        lb[:6], ub[:6] = -DELTA_BOUND_RAD, DELTA_BOUND_RAD
        # Trust Region Reflective（trf）支持变量盒约束
        sol = least_squares(residuals, theta0, method='trf',
                            bounds=(lb, ub), max_nfev=2000)
        delta, X = unpack(sol.x)

        # ── 效果对比：修正前后固定标记在基座系下的坐标分散度 ──────
        def spread_cm(d, X_):
            """计算给定 (δ, X) 下各样本反算的标记基座系 3D 坐标标准差模长（单位：cm）。"""
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

        # 无论是否通过阈值均落盘保存，并附带分散度质量字段供人工复核
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
        # 无论在线还是异常中止，释放 GUI/相机并执行标准安全回零失能
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
