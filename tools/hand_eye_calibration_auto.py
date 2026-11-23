#!/usr/bin/env python3
"""
自动 Eye-in-Hand 手眼标定 v3（reBot Arm B601-RS + Orbbec Gemini 2）——
单点示教 + 关节空间自动扰动 + 在线共轭角一致性守卫。
========================================================================
★ 为什么放弃官方 50 个预设笛卡尔位姿：
    官方位姿的姿态角基于官方 SDK 控制器的 TCP 坐标系约定，与本项目直接使用的
    URDF（joint6 坐标系）约定不一致，且腕部限位 ±90°；经数值验证位置 40/40 可达，
    但全姿态约束下 0/40 可达。手眼标定本质只需要「多视角观测同一静止标记」，
    因此改为：人工示教 1 个看清标记的中心位姿 → 在关节空间自动生成扰动位姿
    （全限位内、无需 IK、天然可达）→ 最近邻排序自动走位、在线守卫采样与求解。

坐标系与单位约定：
    - T_base_tcp    : 4×4 齐次矩阵（TCP/joint6 → base，平移单位：m）
    - T_cam_marker  : 4×4 齐次矩阵（ArUco marker → camera，solvePnP IPPE_SQUARE 输出，单位：m）
    - T_tcp_camera  : 4×4 齐次矩阵（camera → TCP，AX=XB 求解输出，单位：m），
                      自动适配 OpenCV 4.x（Tsai-Lenz）与 OpenCV 5.0（Park-Martin）

流程与交互：
    - 离线复算：若存在 config/handeye_samples.npz，启动时输入 L 可不开相机不动臂直接求解
    - 阶段 1【示教】：键盘微动机械臂使 ArUco 标记居中（25~40cm，w ≥ 120px 绿字），
      按空格保存 config/viewpoint.json（已有示教点时可直接回车复用，或输入 T 重教）
    - 阶段 2【预检】：生成扰动位姿 + Pinocchio FK 内缩安全盒校验 + 贪心最近邻排序
      （支持 `--dry-run` 仅运行到预检，不驱动机械臂）
    - 阶段 3【自动采样】：逐点走位 → 连续帧位置稳定判定 → 在线共轭角一致性守卫
      （拦截标记滑动/PnP 翻转，单点失败自动减半幅度重试一次）→ 增量持久化至 npz
    - 阶段 4【求解自检】：输出 T_tcp_camera、不动点分散度与全对共轭角一致性并保存 JSON

按键：
    示教阶段：1~6 选关节 | +/- 微动(±2°) | 空格 保存示教位姿 | Q/Esc 退出
    自动阶段：Q/Esc 提前结束并用已有样本求解（≥8 组）

安全红线：
    - 所有扰动位姿均裁剪在关节软限位内，且经 FK 校验落在内缩 CAMERA_CLEARANCE 后的安全盒内
    - 走位限速 12°/s，采前打印 J3 温度；检测 SDK shutdown_started/is_stopped 立即中止
    - 正常收尾先回示教位姿再走标准安全停机（hold → stop 平滑回零 → 失能）
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from pynput import keyboard as pynput_kb

# 项目根目录（tools/ 的上一级）加入模块搜索路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from gemini2_camera import Gemini2Camera
from rebot_adapter import RebotAdapter
from hand_eye_solver import calibrate_hand_eye   # OpenCV 5.0 移除了
                                                  # cv2.calibrateHandEye，用自实现兜底

# ── ArUco 标记参数（必须与实物一致！）───────────────────────
ARUCO_DICT = cv2.aruco.DICT_4X4_50    # 实物 ArUco 字典：4X4_50
EXPECTED_MARKER_ID = 0                # 只认 ID=0，防止视野里其它标记/误检污染样本
MARKER_SIZE = 0.10                    # 黑边外边长（单位：m），需用尺子实测确认

# ── ★ 工作空间安全边界（基座系，单位：m）★ 按桌面环境核对 ─────
WS_X_MIN, WS_X_MAX = 0.15, 0.42
WS_Y_MIN, WS_Y_MAX = -0.12, 0.30      # 右侧 25cm 存在障碍物 → Y 下限设为 -0.12m
WS_Z_MIN, WS_Z_MAX = 0.18, 0.45

# FK 预检检查的是法兰（joint6）原点，但相机从法兰向外伸出若干厘米且
# T_tcp_camera 尚未标定 → 将安全盒各边向内收缩 CAMERA_CLEARANCE 作为碰撞裕度
CAMERA_CLEARANCE = 0.06               # 相机支架相对法兰的最大外伸估计（单位：m）

# ── 扰动位姿生成参数 ────────────────────────────────────────
N_AUTO_POSES = 24                     # 自动扰动位姿数（24~28 足够求解收敛，
                                      # 过多会拉长电机带载时间导致 J3 发热）
# 各关节扰动半幅（单位：deg）。相机光轴一般偏离关节轴，腕部关节（J4~J6）
# 小转角也会大幅摆动视野 → 腕部幅度偏保守；单点失败会自动减半幅度重试
PERTURB_AMP_DEG = [15, 10, 10, 12, 12, 30]
RANDOM_SEED = 20260929                # 固定随机种子，便于复现同一组预检位姿
VIEWPOINT_FILE = os.path.join(_ROOT, "config", "viewpoint.json")
SAMPLES_FILE = os.path.join(_ROOT, "config",
                            "handeye_samples.npz")   # 样本增量备份，中途中断不丢失
OUT_CAL_FILE = os.path.join(_ROOT, "config", "hand_eye_calibration.json")

# ── 采样流程与稳定判定参数 ──────────────────────────────────
MOVE_SPEED_DEG = 12.0       # 位姿间走位速度（单位：deg/s）；最近邻排序后单步多 <30°
TEACH_JOG_STEP = np.deg2rad(2.0)  # 示教单步点动角（单位：rad）
SETTLE_S = 0.3              # 到位后机械余振等待时长（单位：s）
STABLE_FRAMES = 3           # 触发采样所需的连续稳定帧数
STABLE_TOL_M = 0.006        # 相邻帧标记位置抖动阈值（单位：m）；solvePnP 在 30~40cm
                            # 处本身有 2~5mm 噪声，过严会导致大量误判超时
POSE_TIMEOUT_S = 2.5        # 单个位姿等待稳定采样的超时时间（单位：s）
MIN_SAMPLES = 8             # 执行手眼求解的最少有效样本数

# ── 在线一致性守卫（防止跑完才发现数据污染）──────────────────
# 手眼标定的共轭约束：任意两组样本的机械臂相对运动 A 与相机相对运动 B
# 满足 R_A = R_X · R_B · R_Xᵀ（相似矩阵），二者等效旋转角必须相等。
# 标记被碰动/滑动、或远距离下 solvePnP 位姿解翻转，都会立刻破坏该等式。
CONSISTENCY_TOL_DEG = 5.0     # 新样本与已有样本共轭角偏差中位数容差（单位：deg）
MIN_PAIR_ROT_DEG = 3.0        # 相对运动转角小于此值缺乏判别力，跳过该配对（单位：deg）
MAX_REJECTED = 3              # 累计被守卫拒绝达到此次数立即中止采样（标记大概率已移位）
MIN_MARKER_PX = 120           # 标记图像平均边长告警阈值（单位：px，低于此值约对应 z>0.43m）


# ═══ ArUco 检测（solvePnP IPPE_SQUARE 实现，兼容 OpenCV 4.7+/5.x）═══

class ArucoTracker:
    """基于 OpenCV ArUco + SOLVEPNP_IPPE_SQUARE 的单标记 6D 位姿跟踪器。

    同时被 calib_diagnose.py 与 handeye_joint_calib.py 复用。
    """

    def __init__(self, camera_matrix, dist_coeffs):
        """初始化 ArUco 检测器与标记局部 3D 角点模板。

        参数:
            camera_matrix (np.ndarray): 3×3 相机内参矩阵（单位：px）。
            dist_coeffs (np.ndarray): 畸变系数（Gemini 2 已强制置零）。
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters())
        self.K = camera_matrix
        self.dist = dist_coeffs
        # 标记中心为原点的 4 个角点（顺序：左上、右上、右下、左下，单位：m）
        s = MARKER_SIZE / 2.0
        self.obj_pts = np.array([[-s,  s, 0], [ s,  s, 0],
                                 [ s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    def detect(self, color):
        """在彩色帧中检测目标 ID 的 ArUco 标记并求解位姿矩阵 T_cam_marker。

        参数:
            color (np.ndarray): 输入 BGR 彩色图像。

        返回:
            tuple[np.ndarray | None, np.ndarray]: (T_cam_marker, annotated)
                - T_cam_marker: 4×4 齐次变换矩阵（marker → camera，平移单位：m），
                  未检测到目标 ID 或 PnP 失败时为 None
                - annotated: 绘制了检测框、坐标轴、距离与像素宽度告警的可视化图像
        """
        annotated = color.copy()
        corners, ids, _ = self.detector.detectMarkers(color)
        if ids is None:
            cv2.putText(annotated, "No marker", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, annotated
        ids_flat = np.asarray(ids).reshape(-1)
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        # 只使用期望 ID 的标记；视野里有多个标记时绝不能盲取 corners[0]
        matches = np.where(ids_flat == EXPECTED_MARKER_ID)[0]
        if len(matches) == 0:
            cv2.putText(annotated, f"Marker ID={EXPECTED_MARKER_ID} not found "
                        f"(saw {ids_flat.tolist()})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return None, annotated
        img_pts = corners[matches[0]].reshape(4, 2).astype(np.float32)
        # IPPE_SQUARE 专为正方形平面标记设计，避免普通 PnP 在正视角下的平面歧义
        ok, rvec, tvec = cv2.solvePnP(self.obj_pts, img_pts, self.K, self.dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, annotated
        try:
            R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
            t = np.asarray(tvec, dtype=np.float64).reshape(3)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            z = float(t[2])
            marker_id = EXPECTED_MARKER_ID
        except Exception:
            return None, annotated
        cv2.drawFrameAxes(annotated, self.K, self.dist, rvec, tvec,
                          MARKER_SIZE * 0.5)
        # 计算 4 条边平均像素长度，过小时提示用户移近标记以防位姿解翻转
        side_px = float(np.mean([
            np.linalg.norm(img_pts[0] - img_pts[1]),
            np.linalg.norm(img_pts[1] - img_pts[2]),
            np.linalg.norm(img_pts[2] - img_pts[3]),
            np.linalg.norm(img_pts[3] - img_pts[0])]))
        txt_color = (0, 255, 0) if side_px >= MIN_MARKER_PX else (0, 0, 255)
        cv2.putText(annotated, f"ID={marker_id}  z={z:.3f}m  w={side_px:.0f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)
        if side_px < MIN_MARKER_PX:
            cv2.putText(annotated, f"MARKER TOO SMALL (<{MIN_MARKER_PX}px), MOVE CLOSER",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return T, annotated


def get_tcp_matrix(robot):
    """读取当前机械臂正向运动学位姿矩阵 T_base_tcp。

    参数:
        robot (RebotAdapter): 机械臂适配层实例。

    返回:
        np.ndarray: 4×4 齐次变换矩阵 T_base_tcp（TCP → base，平移单位：m）。
    """
    fk = robot.forward_kinematics()
    T = np.eye(4); T[:3, :3] = fk['rotation']; T[:3, 3] = fk['position']
    return T


def _rot_angle_deg(R):
    """由 3×3 旋转矩阵的迹计算等效轴角转角（单位：deg）。

    参数:
        R (np.ndarray): 3×3 正交旋转矩阵。

    返回:
        float: 旋转角度，范围 [0.0, 180.0]（单位：deg）。
    """
    c = (float(np.trace(R)) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))


def check_new_sample(robot_poses, camera_poses, T_r_new, T_c_new):
    """在线共轭角一致性守卫：校验新样本与全部已有样本的相似变换旋转角一致性。

    对每个已有样本 (T_r, T_c)，计算机械臂相对运动 A = T_r_new⁻¹ · T_r 与相机相对运动
    B = T_c_new · T_c⁻¹，若 ∠A ≥ MIN_PAIR_ROT_DEG，记录 |∠A - ∠B|。
    健康数据中位数一般 < 1~2°；超过 CONSISTENCY_TOL_DEG 则判定为离群/标记移位并拒绝。

    参数:
        robot_poses (list[np.ndarray]): 已有 T_base_tcp 样本列表。
        camera_poses (list[np.ndarray]): 已有 T_cam_marker 样本列表。
        T_r_new (np.ndarray): 新采样点的 4×4 T_base_tcp 矩阵。
        T_c_new (np.ndarray): 新采样点的 4×4 T_cam_marker 矩阵。

    返回:
        tuple[bool, float, int]: (是否通过, 共轭角偏差中位数 deg, 参与判别的有效配对数)。
    """
    diffs = []
    for T_r, T_c in zip(robot_poses, camera_poses):
        A = np.linalg.inv(T_r_new) @ T_r          # 机械臂相对运动
        B = T_c_new @ np.linalg.inv(T_c)          # 相机相对运动
        ang_a = _rot_angle_deg(A[:3, :3])
        if ang_a < MIN_PAIR_ROT_DEG:
            continue                              # 几乎没转动的对没有判别力
        diffs.append(abs(ang_a - _rot_angle_deg(B[:3, :3])))
    if not diffs:
        return True, 0.0, 0
    med = float(np.median(diffs))
    return med <= CONSISTENCY_TOL_DEG, med, len(diffs)


def wait_arm_settled(robot, tol_deg=0.5, timeout=3.0):
    """等待非阻塞微动（wait=False）命令斜坡收敛并机械停稳后再记录示教位姿。

    参数:
        robot (RebotAdapter): 机械臂适配层实例。
        tol_deg (float): 命令与目标允许最大关节误差（单位：deg）。
        timeout (float): 超时时长（单位：s）。

    返回:
        bool: 停稳返回 True，超时返回 False。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if np.max(np.abs(robot.get_command_pos() - robot._q_target)) < np.deg2rad(tol_deg):
            time.sleep(0.2)  # 等待实际机械跟踪到位
            return True
        time.sleep(0.05)
    print("⚠ 等待停稳超时")
    return False


# ═══ 阶段 1：人工示教中心观测位姿 ═══

def teach_viewpoint(robot, cam, tracker):
    """通过键盘微动示教一个清晰观测 ArUco 标记的中心位姿。

    参数:
        robot (RebotAdapter): 机械臂适配层实例。
        cam (Gemini2Camera): Gemini 2 相机实例。
        tracker (ArucoTracker): ArUco 检测器。

    返回:
        np.ndarray | None: 确认保存时返回长度 6 的关节角数组（单位：rad）；
            按 Q/Esc 退出时返回 None。
    """
    state = {"joint": 0, "save": False, "quit": False}
    pressed = set()

    def on_press(key):
        """pynput 按键按下回调：更新当前微动关节、点动状态与保存/退出标志。"""
        try:
            c = key.char
            if c in '123456':
                state["joint"] = int(c) - 1
                print(f"当前关节: J{int(c)}")
            elif c in ('+', '='):
                pressed.add('+')
            elif c == '-':
                pressed.add('-')
            elif c in ('q', 'Q'):
                state["quit"] = True
        except AttributeError:
            if key == pynput_kb.Key.space:
                state["save"] = True
            elif key == pynput_kb.Key.esc:
                state["quit"] = True

    def on_release(key):
        """pynput 按键释放回调：同时清理 '+' 与 '='，防止 Shift 组合键松开卡键。"""
        try:
            pressed.discard(key.char)
            if key.char in ('+', '='):
                pressed.discard('+'); pressed.discard('=')
        except (AttributeError, KeyError):
            pass

    listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("""
[示教] 键盘微动机械臂，让相机看清 ArUco 标记：
  1~6 选关节 | +/- 微动 | 空格 保存示教位姿 | Q/Esc 退出
  要求：标记在画面中央附近，距离 25~40cm（画面左上角 w≥120px 且为绿字），
        标记角点检测稳定。w 为红字说明太远，必须移近后再保存
""")

    while not state["quit"]:
        if '+' in pressed or '-' in pressed:
            d = (1.0 if '+' in pressed else 0.0) - (1.0 if '-' in pressed else 0.0)
            q = robot.get_command_pos()
            q[state["joint"]] += d * TEACH_JOG_STEP
            q = np.clip(q, robot.q_min, robot.q_max)  # 软限位保护
            robot.move_to_joints(q, speed_deg_s=6.0, wait=False)
            time.sleep(0.15)

        color, _, _ = cam.get_frames()
        if color is None:
            continue
        try:
            T_m2c, annotated = tracker.detect(color)
        except Exception:
            continue
        h, w = annotated.shape[:2]
        # 画面中心十字准星，辅助将标记对中
        cv2.drawMarker(annotated, (w//2, h//2), (0, 255, 0),
                       cv2.MARKER_CROSS, 30, 1)
        cv2.putText(annotated, "TEACH: jog to see marker, SPACE=save",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.imshow('Auto Hand-Eye Calibration', annotated)
        cv2.waitKey(1)

        if state["save"]:
            if T_m2c is None:
                print("✗ 看不到标记，不能保存。继续调整...")
            else:
                wait_arm_settled(robot)   # 等停稳再记录，防止保存到运动中间值
                q_view = robot.get_current_pos()
                fk = robot.forward_kinematics()
                print(f"\n✓ 示教位姿已保存（关节角 rad）: {q_view.round(3).tolist()}")
                print(f"  FK TCP 位置: {fk['position'].round(3)} m")
                print("  ★ 请用尺子核对：TCP 距桌面高度/距基座水平距离是否与上图一致，")
                print("    若明显不符说明 URDF 与实际不一致，先不要进入自动阶段")
                listener.stop()
                return q_view
            state["save"] = False
        time.sleep(0.01)

    listener.stop()
    return None


# ═══ 阶段 2：关节空间扰动位姿生成 + FK 安全盒预检 ═══

def generate_perturbed_poses(robot, q_view):
    """以示教位姿为中心在关节空间随机采样扰动位姿，并用 Pinocchio FK 校验安全盒。

    每个候选位姿均先裁剪至 [q_min, q_max]，再经正向运动学检查法兰位置是否落于
    向内收缩 CAMERA_CLEARANCE 后的安全工作空间盒内，无需逆运动学即可保证 100% 可达。

    参数:
        robot (RebotAdapter): 提供软限位与 Pinocchio 模型/数据的机械臂适配层。
        q_view (np.ndarray): 长度 6 的示教基准关节角（单位：rad）。

    返回:
        np.ndarray: 形状 (M, 6) 的有效关节角数组（单位：rad），首项为 q_view。
    """
    import pinocchio as pin
    rng = np.random.default_rng(RANDOM_SEED)
    amp = np.deg2rad(np.array(PERTURB_AMP_DEG, dtype=np.float64))
    # 相机从法兰外伸且标定前未知，安全盒向内收缩 CAMERA_CLEARANCE
    c = CAMERA_CLEARANCE
    box_min = np.array([WS_X_MIN + c, WS_Y_MIN + c, WS_Z_MIN + c])
    box_max = np.array([WS_X_MAX - c, WS_Y_MAX - c, WS_Z_MAX - c])
    jid = robot.model.getJointId(robot.joint_names[-1])
    poses = [q_view.copy()]       # 示教位姿本身作为第 1 个样本点
    tries = 0
    while len(poses) < N_AUTO_POSES + 1 and tries < N_AUTO_POSES * 30:
        tries += 1
        q = q_view + rng.uniform(-amp, amp)
        q = np.clip(q, robot.q_min, robot.q_max)
        if np.max(np.abs(q - q_view)) < np.deg2rad(3):
            continue              # 扰动太小对标定旋转多样性无贡献
        # FK 校验 TCP 位置在收缩后的安全盒内
        q_full = robot._build_q_full(q)
        pin.forwardKinematics(robot.model, robot.data, q_full)
        pin.updateFramePlacements(robot.model, robot.data)
        p = robot.data.oMi[jid].translation
        if np.all(p >= box_min) and np.all(p <= box_max):
            poses.append(q)
    return np.array(poses)


def order_by_nearest(poses, q_start):
    """在关节空间做贪心最近邻排序，最小化相邻采样点间的大幅度扫动（更安全且省时）。

    参数:
        poses (np.ndarray): 形状 (M, 6) 的待访问关节角数组（单位：rad）。
        q_start (np.ndarray): 长度 6 的起始关节角（单位：rad）。

    返回:
        np.ndarray: 重新排序后的 (M, 6) 关节角数组（单位：rad）。
    """
    remaining = list(range(len(poses)))
    ordered = []
    cur = q_start
    while remaining:
        d = [np.linalg.norm(poses[i] - cur) for i in remaining]
        i = remaining.pop(int(np.argmin(d)))
        ordered.append(i)
        cur = poses[i]
    return poses[ordered]


# ═══ 求解 + 双重自检 + 落盘（独立于硬件，支持离线 npz 复算）═══

def solve_and_report(robot_poses, camera_poses):
    """由采样位姿对求解手眼外参 T_tcp_camera，执行双重精度自检并写入 JSON 文件。

    自检包含两项独立指标：
        1. 固定标记在基座系下的 3D 坐标标准差（分散度 <1cm 优秀、<2cm 合格）；
        2. 全部样本两两配对的共轭旋转角误差中位数与最差值（反映原始数据自洽性）。

    参数:
        robot_poses (list[np.ndarray]): 4×4 T_base_tcp 列表（单位：m）。
        camera_poses (list[np.ndarray]): 4×4 T_cam_marker 列表（单位：m）。

    返回:
        np.ndarray | None: 样本充足时返回 4×4 T_tcp_camera 矩阵；不足 MIN_SAMPLES 返回 None。
    """
    n = len(robot_poses)
    print(f"\n求解，共 {n} 组样本")
    if n < MIN_SAMPLES:
        print(f"✗ 样本不足（{n} < {MIN_SAMPLES}）")
        print("  建议：示教时让标记离相机近一点/更居中，或增大 PERTURB_AMP_DEG")
        return None

    R_g2b = [T[:3, :3] for T in robot_poses]
    t_g2b = [T[:3, 3] for T in robot_poses]
    R_t2c = [T[:3, :3] for T in camera_poses]
    t_t2c = [T[:3, 3] for T in camera_poses]

    T, method = calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c)
    print(f"求解方法: {method}")
    print("\n✓ 标定成功！T_tcp_camera =")
    print(T)
    print(f"平移 (m): {T[:3,3].round(4)}")

    # 自检 1：将各样本观测到的标记原点反算至基座系，计算 3D 分散度（cm）
    base_points = np.array([
        (robot_poses[i] @ T @ np.append(camera_poses[i][:3, 3], 1.0))[:3]
        for i in range(n)])
    spread = np.linalg.norm(base_points.std(axis=0)) * 100
    print(f"\n[自检] 标记原点基座系坐标分散度: {spread:.2f} cm  "
          f"({'✓ 优秀' if spread < 1.0 else '✓ 合格' if spread < 2.0 else '⚠ 偏大，建议重跑'})")

    # 自检 2：全体配对共轭角一致性：数据自洽时应 <1~2°；中位 >5° 说明样本本身坏，
    # 此时任何求解方法都救不回来（分散度再小也不可信）
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            A = np.linalg.inv(robot_poses[j]) @ robot_poses[i]
            B = camera_poses[j] @ np.linalg.inv(camera_poses[i])
            ang_a = _rot_angle_deg(A[:3, :3])
            if ang_a < MIN_PAIR_ROT_DEG:
                continue
            diffs.append(abs(ang_a - _rot_angle_deg(B[:3, :3])))
    if diffs:
        med, worst = float(np.median(diffs)), float(np.max(diffs))
        verdict = '✓ 数据自洽' if med <= 2.0 else \
                  ('✓ 基本自洽' if med <= CONSISTENCY_TOL_DEG else
                   '✗ 数据不自洽，结果不可信！检查标记是否滑动/是否太远')
        print(f"[自检] 共轭角一致性: 中位 {med:.2f}° / 最差 {worst:.1f}°  {verdict}")
    if n < 15:
        print(f"[提示] 样本数 {n} < 15，建议增大样本量重跑")

    result = {
        'T_tcp_camera': T.tolist(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': n,
        'residual_spread_cm': round(float(spread), 2),
        'arm': 'reBot Arm B601-RS',
        'camera': 'Orbbec Gemini 2',
        'method': f'teach_viewpoint_joint_perturb_{method}',
    }
    with open(OUT_CAL_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n已保存 {OUT_CAL_FILE}（跟随脚本直接可用）")
    return T


# ═══ 主流程 ═══

def main(dry_run=False):
    """自动手眼标定主入口：协调示教、预检、自动采样、一致性守卫与求解收尾。

    参数:
        dry_run (bool): 为 True 时仅执行示教加载/位姿生成与安全盒预检，不驱动机械臂走位。
    """
    print("=" * 60)
    print("自动手眼标定 v3 —— 示教一个点 + 关节空间自动扰动")
    print("=" * 60)
    print(f"""
[工作空间安全边界] X:[{WS_X_MIN},{WS_X_MAX}] Y:[{WS_Y_MIN},{WS_Y_MAX}] Z:[{WS_Z_MIN},{WS_Z_MAX}]
开始前请确认：
  1. ArUco 标记（黑边 {MARKER_SIZE*1000:.0f}mm）平贴在基座正前方
     约 25~30cm 桌面、正面朝上，全程不会移动
  2. 桌面已清空；Esc 急停可用（Esc/Ctrl+C 会平滑回零后失能）
""")

    # 有备份样本时可以选择直接离线求解（不连接硬件、不动机械臂）
    if os.path.exists(SAMPLES_FILE) and not dry_run:
        data = np.load(SAMPLES_FILE)
        ans = input(f"检测到 {len(data['robot_poses'])} 组已保存样本："
                    "输入 L 直接求解（不开相机不动臂），回车重新采集: "
                    ).strip().lower()
        if ans == 'l':
            solve_and_report([T for T in data["robot_poses"]],
                             [T for T in data["camera_poses"]])
            return

    cam = Gemini2Camera()
    tracker = ArucoTracker(cam.camera_matrix, cam.dist_coeffs)
    robot = RebotAdapter()
    cv2.namedWindow('Auto Hand-Eye Calibration', cv2.WINDOW_NORMAL)

    try:
        # ── 阶段 1：示教（或加载已有示教位姿）─────────────────
        q_view = None
        if os.path.exists(VIEWPOINT_FILE):
            with open(VIEWPOINT_FILE) as f:
                q_view = np.array(json.load(f)["q_view"], dtype=np.float64)
            ans = input(f"检测到已有示教位姿 {VIEWPOINT_FILE}，"
                        "直接回车使用，输入 T 重新示教: ").strip().lower()
            if ans == 't':
                q_view = None

        if q_view is None:
            q_view = teach_viewpoint(robot, cam, tracker)
            if q_view is None:
                print("已取消")
                return
            with open(VIEWPOINT_FILE, 'w') as f:
                json.dump({"q_view": q_view.tolist(),
                           "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}, f)
            print(f"示教位姿已写入 {VIEWPOINT_FILE}（下次运行可复用）")

        # ── 阶段 2：生成扰动位姿 + 安全盒预检 + 最近邻排序 ─────
        print(f"\n[预检] 以示教位姿为中心生成 {N_AUTO_POSES} 个扰动位姿（无 IK）...")
        poses = generate_perturbed_poses(robot, q_view)
        poses = order_by_nearest(poses, robot.get_command_pos())
        print(f"[预检] 有效位姿: {len(poses)} 个（全部限位内 + TCP 在安全盒内）")
        amps = np.rad2deg(np.abs(poses - q_view).max(axis=0))
        print(f"[预检] 实际最大扰动(度): J1={amps[0]:.0f} J2={amps[1]:.0f} "
              f"J3={amps[2]:.0f} J4={amps[3]:.0f} J5={amps[4]:.0f} J6={amps[5]:.0f}")

        if len(poses) < MIN_SAMPLES:
            print("✗ 有效位姿太少：示教位姿可能太靠近限位/安全盒边缘，请重新示教")
            return
        if dry_run:
            print("\n[dry-run] 预检完成，未执行任何机械臂运动")
            return

        input(f"\n预检通过（{len(poses)} 个位姿）。确认环境安全后按回车开始运动...")

        # ── 阶段 3：自动走位与在线守卫采样 ─────────────────────
        robot_poses, camera_poses = [], []
        quit_flag = [False]
        rejected = [0]

        def attempt_sample(q_target, tag):
            """驱动机械臂到 q_target，等待标记连续稳定并通过共轭角一致性守卫后落盘。

            参数:
                q_target (np.ndarray): 目标 6 轴关节角（单位：rad）。
                tag (str): 终端与画面显示的位姿进度标签（如 "3/25" 或 "3/25 减半重试"）。

            返回:
                bool: 成功采得并通过守卫校验返回 True，超时/中断/被拒返回 False。
            """
            print(f"\n[位姿 {tag}] 扰动={np.rad2deg(np.abs(q_target - q_view)).round(0)}")
            # 打印 J3（主要负重关节）实时温度，提示过温保护阈值
            temps = getattr(robot.io.arm, "last_temperatures", [])
            if len(temps) >= 3 and temps[2] is not None:
                print(f"  J3 电机温度: {temps[2]:.0f}°C  "
                      f"(110°C 报警 / 125°C 自动回零 / 140°C 紧急失能)")
            robot.move_to_joints(q_target, speed_deg_s=MOVE_SPEED_DEG,
                                 wait=True, timeout=40.0)
            time.sleep(SETTLE_S)

            t_deadline = time.time() + POSE_TIMEOUT_S
            stable, last_tvec = 0, None
            while time.time() < t_deadline and not quit_flag[0]:
                color, _, _ = cam.get_frames()
                if color is None:
                    continue
                try:
                    T_m2c, annotated = tracker.detect(color)
                except Exception:
                    continue
                cv2.putText(annotated,
                            f"Pose {tag}  Samples: {len(robot_poses)}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow('Auto Hand-Eye Calibration', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    quit_flag[0] = True

                if T_m2c is None:
                    stable, last_tvec = 0, None
                    continue
                # 连续帧平移抖动判定：相邻两帧位移 < STABLE_TOL_M 计入稳定帧
                tvec = T_m2c[:3, 3]
                if last_tvec is not None and np.linalg.norm(tvec - last_tvec) < STABLE_TOL_M:
                    stable += 1
                else:
                    stable = 1
                last_tvec = tvec

                if stable >= STABLE_FRAMES:
                    T_r_new = get_tcp_matrix(robot)
                    ok, med_diff, n_chk = check_new_sample(
                        robot_poses, camera_poses, T_r_new, T_m2c)
                    if not ok:
                        rejected[0] += 1
                        print(f"  ✗ 一致性守卫拒绝该样本（共轭角偏差中位 "
                              f"{med_diff:.1f}° > {CONSISTENCY_TOL_DEG}°，"
                              f"与 {n_chk} 个已有样本比对）")
                        print("    → 标记很可能被碰动/滑动，或距离太远位姿解翻转。"
                              "该样本已丢弃，不写入")
                        return False
                    robot_poses.append(T_r_new)
                    camera_poses.append(T_m2c)
                    # 每组样本立即增量落盘，中途崩溃/中断也不丢失已采数据
                    np.savez(SAMPLES_FILE,
                             robot_poses=np.array(robot_poses),
                             camera_poses=np.array(camera_poses))
                    print(f"  ✓ 采样成功（第 {len(robot_poses)} 组，"
                          f"标记距离 {np.linalg.norm(tvec)*1000:.0f}mm"
                          f"{'' if n_chk == 0 else f'，一致性 {med_diff:.1f}°'}）")
                    return True
            return False

        for idx, q_target in enumerate(poses):
            if quit_flag[0]:
                print("\n用户中断，用已有样本求解...")
                break
            if rejected[0] >= MAX_REJECTED:
                print(f"\n✗ 已有 {rejected[0]} 个样本被一致性守卫拒绝，中止采样！")
                print("  请检查：① 标记是否贴牢（用手推一下，不能有任何滑动）")
                print("          ② 标记距离是否 ≤0.45m（画面里 w≥120px，绿色字）")
                print("  修复后删除 viewpoint.json 重新示教再跑。")
                break
            if robot.io.arm.shutdown_started or robot.io.arm.is_stopped:
                # Esc/温度保护等已触发 SDK 关闭流程，继续走位没有意义
                print("\n检测到 SDK 已进入安全关闭流程（Esc/温度保护），立即中止采样")
                return
            if attempt_sample(q_target, f"{idx+1}/{len(poses)}"):
                continue
            # 全幅扰动失败（如标记出视野）→ 减半幅度重试一次，尽量保留该方向姿态多样性
            if quit_flag[0]:
                break
            q_half = q_view + 0.5 * (q_target - q_view)
            if not attempt_sample(q_half, f"{idx+1}/{len(poses)} 减半重试"):
                if not quit_flag[0]:
                    print("  减半后仍失败，跳过")

        # ── 阶段 4：求解 + 自检 + 保存 ─────────────────────────
        solve_and_report(robot_poses, camera_poses)

        # 收尾：正常退出路径先回到示教位姿（收拢、已知安全）再进入安全停机，
        # 避免停在某个大伸展位姿导致回零/失能风险
        if not (robot.io.arm.shutdown_started or robot.io.arm.is_stopped):
            print("\n收尾：回到示教位姿...")
            robot.move_to_joints(q_view, speed_deg_s=MOVE_SPEED_DEG,
                                 wait=True, timeout=30.0)

    finally:
        # 标准安全停机路径：销毁窗口 → 停相机 → hold → stop（平滑回零后失能）
        cv2.destroyAllWindows()
        cam.stop()
        try:
            robot.hold()
        except Exception:
            pass
        robot.stop()
        print("程序退出（Esc/Ctrl+C → SDK 平滑回零；正常退出 → 保持当前位后失能）")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    try:
        main(dry_run=dry_run)
    except KeyboardInterrupt:
        print("\n已取消（机械臂将由 rebot_control 平滑回零）")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
