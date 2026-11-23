#!/usr/bin/env python3
"""
手掌识别跟随 - 固定姿态视觉伺服（旧版仅跟随对照基线 · B601 RS + Orbbec Gemini 2）

文件定位：
    保留为旧版「纯跟随（无夹爪控制）」对照基线，用于与主入口 `palm_follow_grasp.py`
    （100Hz 伺服 + 世界系 KF/自运动延迟补偿 + J7 夹爪捏合 + 200Hz MIT 扩展前馈）
    进行回归对比。本脚本采用简洁的 50Hz 视觉伺服主干，不启用 J7 夹爪、卡尔曼预测
    或高阶眼在手补偿。

核心控制流水线（移植自 HighTorque-Robotics/Panthera_HT_SDK_Extensions 的
`16_blue_circle_ibvs_fixed_orientation.py`，控制数学原样保留）：
    RGB-D 掌心 2D 检测与跟踪 → 中值深度 + One Euro 滤波 → 掌心 3D 坐标（相机系）
    → 位置误差 e = P_cam - [0, 0, DESIRED_DEPTH]（含死区迟滞）
    → 相机系线速度 → EMA 平滑 → 6×6 伴随变换到 TCP 系 → 旋转到世界系 + 固定姿态角速度
    → 桌面软墙裁切（单向消去外推分量）+ 硬墙（越界 >5cm 自动回 HOME）
    → 可操作度自适应阻尼伪逆雅可比 → 关节速度限幅 → 适配层积分下发

线程架构与退出安全链：
    - 主线程：Gemini 2 RGB-D 采集 + MediaPipe 掌心检测/深度解算 + OpenCV HUD 渲染；
    - `ctrl_thread`（50Hz 守护线程）：读取共享掌心坐标，执行 FK/雅可比与伺服控制；
    - `kb_listener`（`pynput` 全局键盘线程）：监听 `g` / 空格切换跟随状态；
    - `finally` 安全退出顺序：停止伺服线程 → 停键盘监听 → 释放检测器/窗口/相机
      → `robot.hold()` → `robot.stop()`（平滑回零后失能，严禁高举姿态直接断电坠落）。

差异点（相对 Panthera 原项目）：
    - 检测目标：蓝球 → 手掌（MediaPipe 掌心点）
    - 相机：RealSense D405 → Orbbec Gemini 2（期望距离 0.15m → 0.30m）
    - 机械臂：Panthera → reBot B601 RS（`rebot_adapter.py`）

操作：
    g / 空格 : 启动/停止跟随
    q        : 退出（保持当前位置并平滑回零失能）
    Esc      : `rebot_control` 内置急停（平滑回零）

⚠️ 首次运行前：
    1. 完成手眼标定（`tools/handeye_joint_calib.py`），生成 `config/hand_eye_calibration.json`
    2. 确认 `rebot_adapter.py` 顶部配置（URDF 路径、关节名、限位）
    3. 先用一个固定不动的手掌照片/假目标测试，再上真人手
    4. 速度参数先调低（`LAMBDA_GAIN=1.5`, `MAX_VEL_CART=0.1`），跑顺再提
"""

import os
import sys
import json
import time
import threading
import numpy as np
import cv2
from pynput import keyboard as pynput_kb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini2_camera import Gemini2Camera
from palm_detection import (PalmDetector, PalmTracker, OneEuroFilter,
                            get_median_depth, pixel_to_3d,
                            draw_palm, draw_landmarks)
from rebot_adapter import RebotAdapter

# ─── 配置参数 ────────────────────────────────────────────────
# 初始关节角（rad）。已用 URDF + Pinocchio 网格搜索验证：
# 可操作度 0.0142（远离奇异阈值 0.004，邻域内稳定），TCP 位于
# [0.293, 0, 0.384]，距虚拟墙各边 ≥10cm。⚠️ 不要用全零位——
# 全零位在 J2/J3 限位边界上且可操作度仅 0.0017（近奇异）。
HOME_JOINT_POS = [0.0, 1.2, 1.05, 0.0, 0.0, 0.0]

# 自定义 HOME：若存在 tools/teach_home.py 生成的 config/home_pose.json 则优先使用
_HOME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config", "home_pose.json")
if os.path.exists(_HOME_FILE):
    with open(_HOME_FILE) as _f:
        _home = json.load(_f)
    HOME_JOINT_POS = _home["home_joint_pos"]
    print(f"[HOME] 使用示教 HOME: {np.round(HOME_JOINT_POS, 3).tolist()}"
          f"  (TCP≈{np.round(_home['tcp_position'], 3).tolist()})")

# 视觉伺服控制参数（对应原项目参数，数值已按手掌场景调整）
TOOL_OFFSET = np.array([0.0, 0.0, 0.0])   # TCP 相对末关节的偏移（米）。
                                          # 若 URDF 末关节即法兰中心则保持 0；
                                          # 有夹爪/转接件时按实际填写
LAMBDA_GAIN = 3.2             # 位置控制增益（首次调试建议先降到 1.5）
K_ORIENTATION = 1.6           # 姿态修正增益
DESIRED_DEPTH = 0.30          # 期望相机到掌心距离 (m)。Gemini2 近距极限 ~0.15m，
                              # 安全起见取 0.25~0.35，不要小于 0.20
SERVO_DIRECTION = 1.0         # 若运动方向完全相反，改为 -1.0（标定反了的症状）
POSITION_DEADBAND = 0.015     # 3D 误差死区 (m)
DEADBAND_HYSTERESIS = 1.5     # 死区迟滞：进入死区后，误差要超过 1.5×死区
                              # 才重新启动——防止误差在死区边界反复穿入穿出
                              # （静止时持续小幅振荡的机械根源）
DEPTH_OE_MIN_CUTOFF = 1.0     # 深度通道 One Euro：静止时压低 z3d 抖动
DEPTH_OE_BETA = 0.02          # （手快速靠近/远离时自动放开滤波）
ORIENTATION_DEADBAND = np.deg2rad(2.0)
MAX_VEL_CART = 0.30           # 最大末端线速度 (m/s)，首次调试建议 0.10
MAX_VEL_ANGULAR = 1.20        # 最大角速度 (rad/s)
MAX_VEL_JOINT = 0.90          # 最大关节速度 (rad/s)，适配层还有第二道限幅
CTRL_RATE = 0.02              # 控制周期 20ms (50Hz)

# 阻尼和平滑参数（与原项目一致）
DAMPING_BASE = 0.010
DAMPING_ADAPTIVE = True
MANIPULABILITY_THRESHOLD = 0.004
MIN_MANIPULABILITY_SPEED_SCALE = 0.35
SMOOTH_ALPHA = 0.38

# 工作空间虚拟墙（基座系，米）——已按你的桌面环境收缩：
# 右侧（-Y）只有 25cm 净空 → Y 下限 -0.10（含相机外伸余量）
WORKSPACE_MIN = np.array([-0.40, -0.10, 0.03])
WORKSPACE_MAX = np.array([ 0.40,  0.40, 0.60])

LOST_HOLD_FRAMES = 15         # 连续丢手超过 15 个控制周期 → 保持不动

# ── 命令行参数与配置文件 ────────────────────────────────────
# 优先级：命令行参数 > config/follow.yaml > 上面的代码内置默认值
#   python palm_follow_fixed_orientation.py --distance 0.35 --gain 2.5
import argparse
import yaml as _yaml

_FOLLOW_CFG = os.path.join(os.path.dirname(__file__), "config", "follow.yaml")
_fcfg = {}
if os.path.exists(_FOLLOW_CFG):
    with open(_FOLLOW_CFG, "r", encoding="utf-8") as _f:
        _fcfg = (_yaml.safe_load(_f) or {}).get("follow", {}) or {}

# yaml 提供的默认值先落到常量上（命令行稍后覆盖）
DESIRED_DEPTH = float(_fcfg.get("distance", DESIRED_DEPTH))
LAMBDA_GAIN = float(_fcfg.get("gain", LAMBDA_GAIN))
MAX_VEL_CART = float(_fcfg.get("max_vel", MAX_VEL_CART))
POSITION_DEADBAND = float(_fcfg.get("position_deadband", POSITION_DEADBAND))
DEADBAND_HYSTERESIS = float(_fcfg.get("deadband_hysteresis", DEADBAND_HYSTERESIS))
DEPTH_OE_MIN_CUTOFF = float(_fcfg.get("depth_oe_min_cutoff", DEPTH_OE_MIN_CUTOFF))
DEPTH_OE_BETA = float(_fcfg.get("depth_oe_beta", DEPTH_OE_BETA))
SMOOTH_ALPHA = float(_fcfg.get("smooth_alpha", SMOOTH_ALPHA))
_ws = (_yaml.safe_load(open(_FOLLOW_CFG, encoding="utf-8")).get("workspace")
       if os.path.exists(_FOLLOW_CFG) else None) or {}
if _ws.get("min") and _ws.get("max"):
    WORKSPACE_MIN = np.array(_ws["min"], dtype=np.float64)
    WORKSPACE_MAX = np.array(_ws["max"], dtype=np.float64)

_parser = argparse.ArgumentParser(description="手掌跟随 - 固定姿态视觉伺服")
_parser.add_argument("--distance", type=float, default=None,
                     help="期望相机到掌心距离（米），建议 0.20~0.50；"
                          "不传则用 config/follow.yaml 的 follow.distance")
_parser.add_argument("--gain", type=float, default=None,
                     help="位置控制增益 LAMBDA_GAIN；"
                          "不传则用 follow.yaml 的 follow.gain")
_parser.add_argument("--max-vel", type=float, default=None,
                     help="最大末端线速度 m/s；不传则用 follow.yaml 的 follow.max_vel")
_ARGS, _ = _parser.parse_known_args()
if _ARGS.distance is not None:
    if not 0.18 <= _ARGS.distance <= 0.60:
        raise ValueError("--distance 建议在 0.18~0.60 米之间"
                         "（Gemini2 近距极限 ~0.15m，太远跟踪不稳）")
    DESIRED_DEPTH = _ARGS.distance
if _ARGS.gain is not None:
    LAMBDA_GAIN = _ARGS.gain
if _ARGS.max_vel is not None:
    MAX_VEL_CART = _ARGS.max_vel

# ─── 加载手眼标定 ────────────────────────────────────────────
_CALIB_FILE = os.path.join(os.path.dirname(__file__),
                           "config", "hand_eye_calibration.json")
if not os.path.exists(_CALIB_FILE):
    raise FileNotFoundError(
        f"未找到 {_CALIB_FILE}，请先运行 tools/handeye_joint_calib.py 完成标定")
with open(_CALIB_FILE, 'r') as _f:
    T_tcp_camera = np.array(json.load(_f)["T_tcp_camera"])


# ═══ 以下数学与运动学函数与原项目 16_*.py 完全一致（与具体硬件解耦）═══

def compute_damped_pseudoinverse(J, damping=0.01):
    """
    计算雅可比矩阵的阻尼最小二乘伪逆（Damped Least Squares, DLS）。

    公式：
        J⁺ = Jᵀ (J Jᵀ + λ² I_m)⁻¹
    当雅可比接近奇异导致矩阵病态不可逆时，自动将阻尼因子放大 10 倍兜底，
    防止关节速度指令发散。

    参数:
        J (np.ndarray): 任务空间雅可比矩阵，形状 `(m, n)`（本脚本为 `6×6`）。
        damping (float): 阻尼系数 λ（无量纲，典型值 `0.01`）。

    返回:
        np.ndarray: 阻尼伪逆矩阵，形状 `(n, m)`。
    """
    m, n = J.shape
    JJT = J @ J.T
    damping_matrix = (damping ** 2) * np.eye(m)
    try:
        return J.T @ np.linalg.inv(JJT + damping_matrix)
    except np.linalg.LinAlgError:
        # 极端奇异兜底：放大阻尼 10 倍保证数值可逆
        return J.T @ np.linalg.inv(JJT + ((damping * 10) ** 2) * np.eye(m))


def compute_manipulability(J):
    """
    计算 Yoshikawa 可操作度指标 w = sqrt(max(0, det(J Jᵀ)))。

    用于在线评估当前构型距离运动学奇异点的远近：值越接近 0 表示末端在某方向
    丧失自由度，需增大阻尼并降低末端目标速度。

    参数:
        J (np.ndarray): 任务空间雅可比矩阵，形状 `(6, 6)`。

    返回:
        float: 非负标量可操作度。
    """
    return np.sqrt(max(0.0, np.linalg.det(J @ J.T)))


def rotation_matrix_to_axis_angle(R):
    """
    将 3×3 旋转矩阵 R ∈ SO(3) 转换为等效旋转向量 θ·k̂（轴角表示）。

    数值保护：
        - 当 |θ| < 1e-6 rad 时直接返回零向量，避免除零；
        - 当 |π - θ| < 1e-3 rad（接近 180° 翻转）时，sin(θ) → 0 会导致标准
          反对称公式爆炸，改从对称矩阵 (R + I) / 2 提取单位转轴。

    参数:
        R (np.ndarray): 3×3 正交旋转矩阵。

    返回:
        np.ndarray: 3 维旋转向量 θ·k̂（rad），方向为右手法则转轴，模长为转角。
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if abs(theta) < 1e-6:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-3:
        # θ≈π 时 sinθ→0，1/(2sinθ) 公式数值爆炸；改用对称部分提取转轴
        # （固定姿态伺服的姿态误差一般远小于 π，这只是兜底保护）
        S = (R + np.eye(3)) / 2.0
        axis = np.array([np.sqrt(max(S[0, 0], 0.0)),
                         np.sqrt(max(S[1, 1], 0.0)),
                         np.sqrt(max(S[2, 2], 0.0))])
        # 用非对角元符号确定轴分量符号
        axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
        axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.zeros(3)
        return theta * axis / n
    k = 1 / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * k


def compute_orientation_error(R_current, R_desired):
    """
    计算基座（世界）系下的末端姿态误差向量（rad）。

    公式：
        R_err = R_desired @ R_currentᵀ，满足 R_err @ R_current = R_desired，
        其轴角向量即为在世界系下把当前姿态旋向期望姿态的等效角位移方向。

    参数:
        R_current (np.ndarray): 当前 TCP 姿态旋转矩阵（3×3，世界系）。
        R_desired (np.ndarray): 目标固定姿态旋转矩阵（3×3，世界系）。

    返回:
        np.ndarray: 3 维世界系姿态误差向量（rad）。
    """
    return rotation_matrix_to_axis_angle(R_desired @ R_current.T)


def skew(v):
    """
    构造 3 维向量的反对称矩阵（叉乘矩阵）[v]×，满足 [v]× @ w = v × w。

    参数:
        v (Sequence[float] | np.ndarray): 3 维向量 `[vx, vy, vz]`。

    返回:
        np.ndarray: 3×3 反对称矩阵。
    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def compute_tcp_fk_and_jacobian(robot, joint_positions):
    """
    使用 Pinocchio 计算当前关节角对应的 TCP 位姿与 6×6 `LOCAL_WORLD_ALIGNED` 雅可比。

    工具偏置修正：
        已知末关节原点位置 p_last、姿态 R 及工具偏置 p_tool（`TOOL_OFFSET`，表达在末关节系下），则：
        - TCP 位置：p_tcp = p_last + R @ p_tool；
        - 由刚体速度关系 v_tcp = v_last + ω × (R @ p_tool) = v_last - [R @ p_tool]× @ ω，
          故线速度雅可比块减去 `skew(R @ TOOL_OFFSET) @ J_angular`。

    参数:
        robot (RebotAdapter): 机械臂适配层实例（提供 `model`, `data`, `joint_names`）。
        joint_positions (Sequence[float]): 6 轴关节角（rad）。

    返回:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - `tcp_position`: 3 维 TCP 位置（m，基座世界系）；
            - `tcp_rotation`: 3×3 TCP 姿态旋转矩阵（基座世界系）；
            - `J_tcp`: 6×6 几何雅可比（前 3 行线速度 m/s，后 3 行角速度 rad/s，世界系方向）。
    """
    import pinocchio as pin
    q_full = robot._build_q_full(joint_positions)

    pin.forwardKinematics(robot.model, robot.data, q_full)
    pin.updateFramePlacements(robot.model, robot.data)
    pin.computeJointJacobians(robot.model, robot.data, q_full)

    last_jid = robot.model.getJointId(robot.joint_names[-1])
    oMi = robot.data.oMi[last_jid]
    tcp_position = oMi.translation + oMi.rotation @ TOOL_OFFSET
    tcp_rotation = oMi.rotation.copy()

    # LOCAL_WORLD_ALIGNED：原点在末关节处、坐标轴与基座世界系平行，直接与世界系 twist 对齐
    J_full = pin.getJointJacobian(robot.model, robot.data, last_jid,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_tcp = J_full.copy()
    J_tcp[:3, :] -= skew(oMi.rotation @ TOOL_OFFSET) @ J_full[3:, :]

    # 仅提取 6 个主动臂关节的速度列（过滤固定关节或夹爪自由度）
    cols = [robot.model.joints[robot.model.getJointId(n)].idx_v
            for n in robot.joint_names]
    return tcp_position.copy(), tcp_rotation, J_tcp[:, cols]


def camera_velocity_to_tcp(v_camera, T_tcp_camera):
    """
    通过刚体速度伴随变换矩阵 Ad(T_tcp_cam) 将相机系 6D 速度映射到 TCP 系。

    变换关系（刚体固联）：
        v_tcp = R @ v_cam + [t]× @ R @ ω_cam
        ω_tcp = R @ ω_cam
        其中 R = T_tcp_camera[:3, :3], t = T_tcp_camera[:3, 3] 由手眼标定给出。

    参数:
        v_camera (np.ndarray): 6 维相机系速度 `[vx, vy, vz, wx, wy, wz]`（m/s, rad/s）。
        T_tcp_camera (np.ndarray): 4×4 齐次变换矩阵（相机系到 TCP 系）。

    返回:
        np.ndarray: 6 维 TCP 系速度 `[vx, vy, vz, wx, wy, wz]`（m/s, rad/s）。
    """
    R_tcp_cam = T_tcp_camera[:3, :3]
    t_tcp_cam = T_tcp_camera[:3, 3]
    V = np.zeros((6, 6))
    V[:3, :3] = R_tcp_cam
    V[:3, 3:] = skew(t_tcp_cam) @ R_tcp_cam
    V[3:, 3:] = R_tcp_cam
    return V @ v_camera


# ═══ 主程序 ═══

def main():
    """
    旧版固定姿态手掌跟随主入口。

    运行与同步架构：
        1. 顺序初始化 Gemini 2 相机、MediaPipe 手掌检测/跟踪器与 `RebotAdapter`，
           先缓慢移动至 `HOME_JOINT_POS` 并锁存该位姿的 `desired_rotation` 作为全程
           固定姿态基准；
        2. 共享状态（`current_target_cam`, `palm_visible`, `g_pressed`, `control_status`）
           全部由单把 `lock` 保护，供主线程（视觉）、`ctrl_thread`（50Hz 伺服）与
           `kb_listener`（全局热键）无竞态读写；
        3. `finally` 块严格按硬件安全停机序列收尾：
           停控制线程 → 停键盘监听 → 释放视觉资源 → `robot.hold()` → `robot.stop()`
           （平滑回零后失能，防止高举姿态失能坠臂）。
    """
    print("初始化 Gemini 2 相机...")
    cam = Gemini2Camera()
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    desired_u, desired_v = int(round(cx)), int(round(cy))   # 期望掌心点 = 画面中心

    print("初始化手掌检测...")
    detector = PalmDetector()
    tracker = PalmTracker()

    print("初始化机械臂...")
    robot = RebotAdapter()

    # 上电先以低速（15°/s）到达安全的示教/预设 HOME 位姿，远离奇异点与限位边界
    print(f"移动到初始位置: {HOME_JOINT_POS}")
    robot.move_to_joints(HOME_JOINT_POS, speed_deg_s=15.0)
    time.sleep(0.5)

    # 锁存 HOME 位姿下的 TCP 姿态矩阵作为固定姿态伺服参考基准
    fk_home = robot.forward_kinematics()
    desired_rotation = np.array(fk_home['rotation'])     # 固定姿态基准
    print(f"初始 TCP 位置: {np.array(fk_home['position']).round(3)} m")

    # ── 跨线程共享状态（全部受 lock 保护；用单元素列表方便闭包修改）──
    lock = threading.Lock()
    current_target_cam = [None]       # 掌心在相机系下的 3D 坐标 [x, y, z] (m)
    palm_visible = [False]
    g_pressed = [False]
    control_status = {"joint_speed": 0.0, "error": "", "active": False}

    print("\n" + "=" * 60)
    print("手掌跟随 - 固定姿态视觉伺服 (B601 RS + Gemini 2)")
    print("=" * 60)
    print(f"期望: 掌心在画面中心({desired_u},{desired_v})，距离 {DESIRED_DEPTH}m"
          f"  [gain={LAMBDA_GAIN}, max_vel={MAX_VEL_CART}]")
    print("操作: g/空格 启停跟随 | q 退出 | Esc 急停回零")
    print("=" * 60 + "\n")

    # ── 全局键盘监听（pynput 独立后台线程，带 250ms 防抖）────
    last_toggle = [0.0]

    def toggle_tracking():
        """切换跟随启停状态（0.25s 防抖，兼容终端外全局快捷键与 OpenCV 窗口按键）。"""
        now = time.time()
        with lock:
            if now - last_toggle[0] < 0.25:
                return
            g_pressed[0] = not g_pressed[0]
            enabled = g_pressed[0]
            last_toggle[0] = now
        print("\n[跟随] 启动" if enabled else "\n[跟随] 停止（保持当前位置）")

    def on_press(key):
        """`pynput` 按键回调：按下 `g` 或空格键触发 `toggle_tracking()`。"""
        try:
            if key.char == 'g':
                toggle_tracking()
        except AttributeError:
            if key == pynput_kb.Key.space:
                toggle_tracking()

    kb_listener = pynput_kb.Listener(on_press=on_press)
    kb_listener.start()

    # ── 50Hz 伺服控制线程 ───────────────────────────────────
    stop_ctrl = threading.Event()
    actual_velocity = [np.zeros(6)]     # EMA 平滑后的相机系 6D 速度缓存
    lost_counter = [0]                  # 连续丢手拍数计数（超过 LOST_HOLD_FRAMES 则保持不动）
    in_deadband = [False]               # 死区迟滞状态（True=已收敛到死区内）
    depth_oe = OneEuroFilter(DEPTH_OE_MIN_CUTOFF, DEPTH_OE_BETA)  # 深度通道自适应低通滤波

    def servo_control_loop():
        """
        50Hz（`CTRL_RATE=0.02s`）固定姿态视觉伺服控制循环。

        单拍执行顺序：
            1. 读取视觉共享状态并更新丢手计数；未启停或连续丢手超限时立即 `robot.hold()`；
            2. 基于内部指令关节角 `get_command_pos()` 求解 FK 与雅可比（避免热循环触发同步 CAN 读）；
            3. 硬墙兜底保护：TCP 超出 `[WORKSPACE_MIN - 5cm, WORKSPACE_MAX + 5cm]` 自动低速回 HOME；
            4. 主任务（位置误差 + 迟滞死区）与次任务（固定姿态误差 + 死区）分别限幅与平滑；
            5. 相机系 → TCP 系（伴随变换）→ 世界系速度变换，叠加单向软墙速度裁切防止边界死锁；
            6. 可操作度自适应阻尼 + 降速 → 阻尼伪逆求解关节速度 → 限幅下发 `joint_vel_cmd()`；
            7. 任意控制异常立即捕获并 `robot.hold()`，避免失控指令送往电机。
        """
        while not stop_ctrl.is_set():
            t0 = time.time()

            with lock:
                following = g_pressed[0]
                target_cam = (None if current_target_cam[0] is None
                              else np.array(current_target_cam[0], dtype=np.float64))
                visible = palm_visible[0]

            if not visible:
                lost_counter[0] += 1
            else:
                lost_counter[0] = 0

            if following and target_cam is not None and lost_counter[0] <= LOST_HOLD_FRAMES:
                try:
                    # 热循环用内部下发值做 FK/雅可比（无 CAN I/O，不挤占底层 200Hz MIT 下发）
                    current_joints = robot.get_command_pos()
                    tcp_pos, tcp_rot, J_robot = compute_tcp_fk_and_jacobian(
                        robot, current_joints)

                    with lock:
                        control_status["tcp"] = tcp_pos.round(3).tolist()

                    # ── 硬墙：越界 >5cm → 自动回 HOME 后继续跟随 ──
                    # （软墙正常时到不了这里；这是异常工况的兜底安全保护）
                    if (np.any(tcp_pos < WORKSPACE_MIN - 0.05)
                            or np.any(tcp_pos > WORKSPACE_MAX + 0.05)):
                        robot.hold()
                        print(f"\n[安全] TCP 越过硬墙 {tcp_pos.round(2).tolist()}"
                              f"，自动回 HOME...")
                        robot.move_to_joints(HOME_JOINT_POS, speed_deg_s=12.0,
                                             wait=True, timeout=30.0)
                        actual_velocity[0] = np.zeros(6)
                        with lock:
                            control_status["error"] = "已自动回 HOME"
                        continue

                    # 主任务：掌心收敛到相机系期望点 (0, 0, DESIRED_DEPTH)
                    target_desired = np.array([0.0, 0.0, DESIRED_DEPTH])
                    pos_err_cam = target_cam - target_desired
                    pos_err_norm = float(np.linalg.norm(pos_err_cam))
                    # 死区迟滞：进入死区后，误差要超过 1.5×死区才重新启动，
                    # 防止掌心 3D 噪声让指令在死区边界反复穿入穿出
                    if in_deadband[0]:
                        if pos_err_norm > POSITION_DEADBAND * DEADBAND_HYSTERESIS:
                            in_deadband[0] = False
                    elif pos_err_norm < POSITION_DEADBAND:
                        in_deadband[0] = True
                    v_cam = np.zeros(6)
                    if not in_deadband[0]:
                        v_cam[:3] = SERVO_DIRECTION * LAMBDA_GAIN * pos_err_cam

                    # 次任务：姿态保持在 HOME 锁存基准（线速度按轴向限幅，角速度在世界系限幅）
                    ori_err = compute_orientation_error(tcp_rot, desired_rotation)
                    v_cam[:3] = np.clip(v_cam[:3], -MAX_VEL_CART, MAX_VEL_CART)
                    angular_world = (K_ORIENTATION * ori_err
                                     if np.linalg.norm(ori_err) > ORIENTATION_DEADBAND
                                     else np.zeros(3))
                    angular_world = np.clip(angular_world, -MAX_VEL_ANGULAR, MAX_VEL_ANGULAR)

                    # 相机系速度 EMA 低通平滑，抑制视觉帧间抖动
                    actual_velocity[0] += (v_cam - actual_velocity[0]) * SMOOTH_ALPHA

                    # 坐标变换：相机系速度 → TCP 系（伴随变换）→ 世界系线速度 + 世界系角速度
                    v_tcp = camera_velocity_to_tcp(actual_velocity[0], T_tcp_camera)
                    v_world = np.zeros(6)
                    v_world[0:3] = tcp_rot @ v_tcp[0:3]
                    v_world[3:6] = angular_world

                    # ── 软墙：只削掉「继续往外走」的速度分量 ──
                    # 允许贴墙切向滑动和向工作空间内部返回，避免到达边界时死锁
                    wall_touch = False
                    for ax in range(3):
                        if tcp_pos[ax] <= WORKSPACE_MIN[ax] and v_world[ax] < 0:
                            v_world[ax] = 0.0
                            wall_touch = True
                        elif tcp_pos[ax] >= WORKSPACE_MAX[ax] and v_world[ax] > 0:
                            v_world[ax] = 0.0
                            wall_touch = True

                    # 可操作度自适应阻尼与降速（靠近奇异构型时同步增大阻尼、削减目标速度）
                    manip = compute_manipulability(J_robot)
                    if DAMPING_ADAPTIVE and manip < MANIPULABILITY_THRESHOLD:
                        damping = DAMPING_BASE * (1.0 + (MANIPULABILITY_THRESHOLD - manip) * 100)
                        v_world *= max(MIN_MANIPULABILITY_SPEED_SCALE,
                                       manip / MANIPULABILITY_THRESHOLD)
                    else:
                        damping = DAMPING_BASE

                    # 阻尼伪逆求解 6 轴关节速度并施加第一道限幅（适配层还有第二道硬件安全限幅）
                    joint_vel = compute_damped_pseudoinverse(J_robot, damping) @ v_world
                    joint_vel = np.clip(joint_vel, -MAX_VEL_JOINT, MAX_VEL_JOINT)
                    # 位置与姿态均落入死区时彻底清零，避免 EMA 拖尾引起静止微蠕动
                    if (in_deadband[0] and
                            np.linalg.norm(ori_err) <= ORIENTATION_DEADBAND):
                        joint_vel[:] = 0.0

                    robot.joint_vel_cmd(joint_vel)
                    with lock:
                        control_status["joint_speed"] = float(np.linalg.norm(joint_vel))
                        control_status["error"] = ("贴墙滑动中" if wall_touch else "")
                        control_status["active"] = True

                except Exception as exc:
                    # 任意运动学/数值/控制异常立即锁死当前姿态，绝不发送异常指令
                    robot.hold()
                    with lock:
                        control_status["error"] = str(exc)
                        control_status["active"] = False
            else:
                # 未启动跟随或连续丢手超阈值：清零速度缓存并保持当前位置
                actual_velocity[0] = np.zeros(6)
                in_deadband[0] = False              # 重新启动跟随时重新收敛
                robot.hold()
                with lock:
                    control_status["joint_speed"] = 0.0
                    control_status["active"] = False
                    if following and lost_counter[0] > LOST_HOLD_FRAMES:
                        control_status["error"] = "手掌丢失，保持位置"

            # 按固定周期补齐剩余休眠时间，维持约 50Hz 伺服节拍
            elapsed = time.time() - t0
            if CTRL_RATE - elapsed > 0:
                time.sleep(CTRL_RATE - elapsed)

    ctrl_thread = threading.Thread(target=servo_control_loop, daemon=True)
    ctrl_thread.start()

    # ── 视觉主循环（主线程：RGB-D 采帧 + MediaPipe 掌心检测 + 深度反投影 + HUD）──
    cv2.namedWindow('Palm Follow', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Palm Follow', 640, 480)

    try:
        while True:
            color_image, depth_raw, depth_scale = cam.get_frames()
            if color_image is None:
                continue

            # 1) 2D 掌心检测与跨帧平滑跟踪
            u, v, landmarks = detector.detect(color_image)
            track = tracker.update((u, v) if u is not None else None)

            # 2) 深度窗口中值采样 + One Euro 滤波 + 针孔反投影至相机系 3D (m)
            if track is not None and track['lost'] == 0:
                ui, vi = int(round(track['u'])), int(round(track['v']))
                depth_m = get_median_depth(depth_raw, depth_scale, ui, vi)
                if depth_m > 0:
                    # 深度通道滤波：静止时 z3d 抖动的主要来源。
                    # One Euro 静止时强滤波，手快速靠近/远离时自动放开
                    depth_m = depth_oe(depth_m)
                    track['x3d'], track['y3d'], track['z3d'] = pixel_to_3d(
                        track['u'], track['v'], depth_m, fx, fy, cx, cy)

            # 3) 原子发布目标坐标给 50Hz 控制线程
            with lock:
                if track is not None and track['lost'] == 0 and track['z3d'] > 0:
                    current_target_cam[0] = [track['x3d'], track['y3d'], track['z3d']]
                    palm_visible[0] = True
                else:
                    palm_visible[0] = False
                    # 注意：不清空 current_target_cam，短暂丢帧时控制侧沿用上值直至超过 LOST_HOLD_FRAMES
                    if track is None:
                        current_target_cam[0] = None

            # ── 调试状态打印（每 0.5s 一行，便于离线/现场定位卡顿环节）──
            if time.time() - getattr(main, "_last_dbg", 0) > 0.5:
                main._last_dbg = time.time()
                det = "None"
                if track is not None:
                    det = (f"u={track['u']:.0f} v={track['v']:.0f} "
                           f"lost={track['lost']} z3d={track['z3d']:.3f}")
                with lock:
                    _fol, _vis = g_pressed[0], palm_visible[0]
                    _spd = control_status["joint_speed"]
                    _err = control_status["error"]
                _tcp = control_status.get("tcp", None)
                print(f"[DBG] palm=({det})  visible={_vis} "
                      f"following={_fol}  joint_speed={_spd:.3f}  "
                      f"tcp={_tcp}  err={_err}")

            # ── OpenCV 实时 HUD 渲染与窗口热键响应 ──
            display = color_image.copy()
            if landmarks is not None:
                draw_landmarks(display, landmarks)
            draw_palm(display, track, desired_u, desired_v)

            with lock:
                following = g_pressed[0]
                status = dict(control_status)

            state = "FOLLOWING" if (following and status["active"]) else \
                    ("HOLD (hand lost)" if following else "IDLE (press g/space)")
            color_map = {"FOLLOWING": (0, 255, 0), "IDLE (press g/space)": (0, 200, 255)}
            cv2.putText(display, state, (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color_map.get(state, (0, 165, 255)), 2)
            cv2.putText(display, f"JointSpeed: {status['joint_speed']:.3f} rad/s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if status["error"]:
                cv2.putText(display, f"CTRL: {status['error'][:60]}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('Palm Follow', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord('g'), ord(' ')):
                toggle_tracking()

    except KeyboardInterrupt:
        print("\n程序被中断")
    finally:
        # ── 硬件安全退出链（红线：严禁在非零/抬臂姿态直接失能电机）──
        # 1) 先通知并等待 50Hz 伺服线程退出，杜绝退出期间继续下发速度指令
        stop_ctrl.set()
        ctrl_thread.join(timeout=1.0)
        # 2) 停止键盘监听并释放检测器、GUI 窗口与相机管线
        kb_listener.stop()
        detector.close()
        cv2.destroyAllWindows()
        cam.stop()
        # 3) 锁定当前姿态稳停 0.3s，再由 robot.stop() 执行平滑回零 → 失能全流程
        print("机械臂保持当前位置...")
        robot.hold()
        time.sleep(0.3)
        robot.stop()
        print("程序退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
