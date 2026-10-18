#!/usr/bin/env python3
"""
手掌跟随 + 捏合抓取 一体主程序（B601 RS + Orbbec Gemini 2）
=================================================================
一个脚本同时实现两个功能（共用一台相机、一次 MediaPipe 检测、一条 CAN 总线）：

  【跟随】掌心 3D（相机系）→ 误差 e = P_cam - [0,0,d]
         → 目标速度前馈 + 分方向增益 → tanh 软饱和
         → 相机系速度 → 伴随变换到 TCP → 世界系 twist
         → 阻尼伪逆雅可比（可操作度奇异保护）→ 关节速度 → 适配层积分下发
         （控制数学与 palm_follow_fixed_orientation.py 一致，
           新增前馈/分方向增益/软饱和/反转清惯性四项高速跟随改进）

  【抓取】拇指尖(4)—食指尖(8) 像素距离 → One Euro 滤波 → 线性映射为
         夹爪 J7 目标角度（度）→ 随 50Hz 关节指令经 rebot_control
         第 7 通道下发（单一 CAN 主机，无总线冲突）。

运行模式：
    python palm_follow_grasp.py                  # 跟随 + 夹爪（完整模式）
    python palm_follow_grasp.py --no-gripper     # 仅手掌跟随
    python palm_follow_grasp.py --no-follow      # 仅捏合夹爪（手臂保持不动，
                                                 #   无需手眼标定，可先验证夹爪映射）
    python palm_follow_grasp.py --no-arm         # 纯视觉演练（不碰硬件）
    python palm_follow_grasp.py --headless       # 无窗口模式（SSH 场景，q 键退出）

操作：
    g / 空格 : 启动/停止跟随（--no-follow 时无效）
    q        : 退出（停止并保持 → 缓慢回零 → 失能）
    Esc      : rebot_control 内置急停（平滑回零）

⚠️ 首次运行前：
    1. 夹爪开合标定：tools/gripper_j7_calibrate.py（结果写入 config/grasp.yaml）
    2. 手眼标定：tools/handeye_joint_calib.py（生成 config/hand_eye_calibration.json
       和 config/joint_offset.json，仅跟随功能需要）
    3. 先用固定假目标低速测试跟随（--gain 1.5 --max-vel 0.10），再上真人手
"""

import os
import sys
import json
import time
import threading
import argparse

import numpy as np
import cv2
import yaml
from pynput import keyboard as pynput_kb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini2_camera import Gemini2Camera
from palm_detection import (PalmDetector, PalmTracker, OneEuroFilter,
                            get_median_depth, pixel_to_3d,
                            draw_palm, draw_landmarks, pinch_distance_px)
# 注意：RebotAdapter 在 main() 中惰性导入（--no-arm 模式不依赖 pinocchio/rebot_control）

# ─── 初始位姿 ────────────────────────────────────────────────
# 已用 URDF + Pinocchio 网格搜索验证远离奇异。⚠️ 不要用全零位
# （全零位在 J2/J3 限位边界上且可操作度仅 0.0017，近奇异）。
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

# ─── 视觉伺服控制参数 ───────────────────────────────────────
TOOL_OFFSET = np.array([0.0, 0.0, 0.0])   # TCP 相对末关节的偏移（米）
# 相机系分方向增益：[X=画面水平(左右), Y=竖直(上下), Z=深度]
# 水平方向跟随主要靠底座 J1 偏航（惯量最大、限速最低），增益高了会
# 饱和振荡（先反向、再晃回来）；竖直/深度用 J2/J3 俯仰，可以高些。
LAMBDA_GAIN_XYZ = np.array([1.8, 2.5, 2.5])
K_ORIENTATION = 1.6           # 姿态修正增益
DESIRED_DEPTH = 0.30          # 期望相机到掌心距离 (m)，不要小于 0.20
SERVO_DIRECTION = 1.0         # 若运动方向完全相反，改为 -1.0（标定反了的症状）
POSITION_DEADBAND = 0.015     # 3D 误差死区 (m)，想贴更近可降到 0.010
ORIENTATION_DEADBAND = np.deg2rad(2.0)
MAX_VEL_CART = 0.30           # 最大末端线速度 (m/s)，首次调试建议 0.10
MAX_VEL_ANGULAR = 1.20        # 最大角速度 (rad/s)
MAX_VEL_JOINT = 1.20          # 最大关节速度 (rad/s)，适配层还有第二道限幅
CTRL_RATE = 0.01              # 控制周期 10ms (100Hz)：指令量化延迟减半
                              # 注意：下面三个常量由 CTRL_RATE 推导（时间制），
                              # 改频率时无需手改，安全时长不会随频率变化

# 目标速度前馈：纯误差反馈的跟随必有滞后（滞后 ≈ 目标速度/增益），
# 手移动越快手臂落后越多。估计掌心目标速度并前馈叠加。0 = 关闭。
VEL_FEEDFORWARD = 0.7         # 前馈系数，推荐 0.5~0.8；出现超前摆动则调低
FF_SMOOTH = 0.3               # 目标速度估计的平滑系数（防检测跳变放大）
FF_MAX_VEL = 0.6              # 前馈速度限幅 (m/s)，防检测丢帧跳变
FF_MIN_SPEED = 0.05           # 目标速度低于此值视为静止，前馈置零：
                              # 否则深度噪声被 /CTRL_RATE 放大后持续注入，
                              # 是「手不动臂却小幅振荡」的主因之一
FF_TAPER_DIST = 0.07          # 前馈近端渐弱距离 (m)：误差小于此值时前馈
                              # 线性衰减到 0——手停下瞬间前馈惯性会把机械臂
                              # 推过目标（到位后回环振荡的主因），巡航段
                              # 不受影响，只在最后 ~7cm 停靠段退出
DEADBAND_HYSTERESIS = 1.5     # 死区迟滞：进入死区后，误差要超过 1.5×死区
                              # 才重新启动——防止误差在死区边界反复穿入穿出
DEPTH_OE_MIN_CUTOFF = 1.0     # 深度通道 One Euro 参数：静止时压低 z3d 抖动
DEPTH_OE_BETA = 0.02          # （手快速靠近/远离时自动放开滤波）
Z3D_MAX_RATE = 0.05           # 深度限速 (m/帧)：参照无效时的最后防线
Z3D_MAX_JUMP = 0.08           # 深度跳变剔除阈值 (m)：单帧 |Δz| 超过此值
                              # 视为误检帧丢弃（手 16ms 内不可能移动 8cm）

# 伸展度增益调度：手臂前伸越多，J1 杠杆臂越长、整臂刚度越低，
# 同样增益在伸展位姿会超出稳定裕度（前伸后停靠大振荡的根源）。
# TCP 水平半径 ≤ REACH_SOFT_START 时参数原样，到 REACH_SOFT_END
# 线性衰减到 REACH_MIN_SCALE，超过后保持 REACH_MIN_SCALE。
REACH_SOFT_START = 0.25       # 开始衰减的水平伸展半径 (m)
REACH_SOFT_END = 0.40         # 衰减到底的半径 (m)
REACH_MIN_SCALE = 0.60        # 最深伸展时停靠段指令的保留比例
REACH_DOCK_DIST = 0.20        # 停靠段半径 (m)：误差小于此值才开始衰减，
                              # 巡航段（误差大）不衰减、全速跟手

CAMERA_FPS = 30               # 相机帧率；Gemini2 支持 640x480@60，
                              # 高速跟随建议 --fps 60（感知延迟砍半）

# 阻尼和平滑参数
DAMPING_BASE = 0.010
DAMPING_ADAPTIVE = True
MANIPULABILITY_THRESHOLD = 0.004
MIN_MANIPULABILITY_SPEED_SCALE = 0.35
SMOOTH_ALPHA = 0.38
# 方向反转时清除速度平滑的「惯性」：手突然反向快移时，若不清除，
# 平滑后的指令仍维持原方向 → 手臂先反向冲一下再拉回
REVERSAL_FAST_RESPONSE = True
MAX_ACCEL = 2.0               # 末端速度指令加速度上限 (m/s^2)，启停梯形化
REVERSAL_ACCEL_BOOST = 1.8    # 方向反转时的加速度额度倍数

# 工作空间虚拟墙（基座系，米）——按实际桌面环境设置
WORKSPACE_MIN = np.array([-0.40, -0.10, 0.03])
WORKSPACE_MAX = np.array([ 0.40,  0.40, 0.60])

LOST_HOLD_FRAMES = round(0.30 / CTRL_RATE)   # 连续丢手 0.30s → 保持不动
COAST_FRAMES = round(0.12 / CTRL_RATE)       # 丢失外推窗口 0.12s
COAST_DECAY = 0.90 ** (CTRL_RATE / 0.02)     # 每 20ms 等效衰减 0.90（防爆冲）
COAST_MAX_VEL = 0.35          # 外推速度限幅 (m/s)：防前馈污染带跑


# ─── 命令行与配置 ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="手掌跟随 + 捏合抓取 一体程序")
    p.add_argument("--distance", type=float, default=None,
                   help="期望相机到掌心距离（米），建议 0.20~0.50；"
                        "不传则用 config/follow.yaml 的 follow.distance")
    p.add_argument("--gain", type=float, default=None,
                   help="竖直/深度方向位置增益（Y/Z）；"
                        "不传则用 follow.yaml 的 follow.gain")
    p.add_argument("--gain-x", type=float, default=None,
                   help="水平方向（左右跟随）位置增益，左右晃动就调低它；"
                        "不传则用 follow.yaml 的 follow.gain_x")
    p.add_argument("--max-vel", type=float, default=None,
                   help="最大末端线速度 m/s；不传则用 follow.yaml 的 follow.max_vel")
    p.add_argument("--fps", type=int, default=None,
                   help="相机帧率，默认取 config/follow.yaml 的 follow.fps；"
                        "Gemini2 支持 60，高速跟随建议 60")
    p.add_argument("--follow-cfg",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "config", "follow.yaml"),
                   help="跟随伺服参数配置文件路径（yaml 为默认，命令行优先）")
    p.add_argument("--grasp-cfg",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "config", "grasp.yaml"),
                   help="捏合→夹爪映射配置文件路径")
    p.add_argument("--detector-backend", default="auto",
                   choices=["auto", "gpu", "cpu"],
                   help="手掌检测推理后端：auto=优先GPU失败回退CPU（默认）")
    p.add_argument("--no-follow", action="store_true",
                   help="仅捏合夹爪：手臂保持不动（此模式无需手眼标定）")
    p.add_argument("--no-gripper", action="store_true",
                   help="仅手掌跟随：夹爪保持当前角度")
    p.add_argument("--no-arm", action="store_true",
                   help="纯视觉演练：不连接机械臂，只验证检测与映射")
    p.add_argument("--headless", action="store_true",
                   help="无窗口模式（SSH/远程场景；按 q 键或 Ctrl+C 退出）")
    return p.parse_args()


def load_grasp_cfg(path):
    """读取捏合→夹爪映射配置，缺失字段给安全默认值。"""
    cfg = {"pinch": {"closed_px": 20.0, "open_px": 200.0,
                     "filter": "one_euro",
                     "oe_min_cutoff": 1.2, "oe_beta": 0.05,
                     "smoothing": 0.35, "deadband_deg": 2.0},
           "gripper": {"enabled": True, "closed_deg": -19.7, "open_deg": 293.6,
                       "hold_while_moving": True, "move_speed_threshold": 0.05}}
    if not os.path.exists(path):
        print(f"⚠️ 未找到 {path}，使用内置默认映射")
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    for section, defaults in cfg.items():
        if isinstance(user.get(section), dict):
            defaults.update(user[section])
    # 兼容两套键名：filter_type / min_cutoff_hz / beta / dcutoff_hz
    p = cfg["pinch"]
    if "filter_type" in p:
        p["filter"] = p["filter_type"]
    if "min_cutoff_hz" in p:
        p["oe_min_cutoff"] = p["min_cutoff_hz"]
    if "beta" in p:
        p["oe_beta"] = p["beta"]
    return cfg


def load_follow_cfg(path):
    """读取跟随伺服参数配置，缺失字段给代码内置默认值。

    优先级：命令行参数 > 本 yaml > 代码内置默认值。
    """
    cfg = {"follow": {
               "distance": DESIRED_DEPTH,
               "gain": float(LAMBDA_GAIN_XYZ[1]),
               "gain_x": float(LAMBDA_GAIN_XYZ[0]),
               "max_vel": MAX_VEL_CART,
               "max_vel_joint": MAX_VEL_JOINT,
               "max_vel_angular": MAX_VEL_ANGULAR,
               "position_deadband": POSITION_DEADBAND,
               "deadband_hysteresis": DEADBAND_HYSTERESIS,
               "orientation_deadband_deg": float(np.rad2deg(ORIENTATION_DEADBAND)),
               "vel_feedforward": VEL_FEEDFORWARD,
               "ff_smooth": FF_SMOOTH,
               "ff_max_vel": FF_MAX_VEL,
               "ff_min_speed": FF_MIN_SPEED,
               "ff_taper_dist": FF_TAPER_DIST,
               "smooth_alpha": SMOOTH_ALPHA,
               "reversal_fast_response": REVERSAL_FAST_RESPONSE,
               "depth_oe_min_cutoff": DEPTH_OE_MIN_CUTOFF,
               "depth_oe_beta": DEPTH_OE_BETA,
               "k_orientation": K_ORIENTATION,
               "fps": CAMERA_FPS,
           },
           "workspace": {"min": WORKSPACE_MIN.tolist(),
                         "max": WORKSPACE_MAX.tolist()}}
    if not os.path.exists(path):
        print(f"⚠️ 未找到 {path}，使用内置默认跟随参数")
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    for section, defaults in cfg.items():
        if isinstance(user.get(section), dict):
            defaults.update(user[section])
    return cfg


def apply_follow_cfg(cfg):
    """把 follow.yaml 的值写入伺服全局参数（伺服线程直接读全局量）。"""
    global POSITION_DEADBAND, DEADBAND_HYSTERESIS, ORIENTATION_DEADBAND
    global VEL_FEEDFORWARD, FF_SMOOTH, FF_MAX_VEL, FF_MIN_SPEED, FF_TAPER_DIST
    global SMOOTH_ALPHA, REVERSAL_FAST_RESPONSE
    global MAX_ACCEL, REVERSAL_ACCEL_BOOST
    global DEPTH_OE_MIN_CUTOFF, DEPTH_OE_BETA, Z3D_MAX_JUMP
    global K_ORIENTATION, MAX_VEL_JOINT, MAX_VEL_ANGULAR
    global REACH_SOFT_START, REACH_SOFT_END, REACH_MIN_SCALE, REACH_DOCK_DIST
    global WORKSPACE_MIN, WORKSPACE_MAX
    fcfg = cfg["follow"]
    POSITION_DEADBAND = float(fcfg["position_deadband"])
    DEADBAND_HYSTERESIS = float(fcfg["deadband_hysteresis"])
    ORIENTATION_DEADBAND = np.deg2rad(float(fcfg["orientation_deadband_deg"]))
    VEL_FEEDFORWARD = float(fcfg["vel_feedforward"])
    FF_SMOOTH = float(fcfg["ff_smooth"])
    FF_MAX_VEL = float(fcfg["ff_max_vel"])
    FF_MIN_SPEED = float(fcfg["ff_min_speed"])
    FF_TAPER_DIST = float(fcfg["ff_taper_dist"])
    SMOOTH_ALPHA = float(fcfg["smooth_alpha"])
    REVERSAL_FAST_RESPONSE = bool(fcfg["reversal_fast_response"])
    MAX_ACCEL = float(fcfg.get("max_accel", 2.0))
    REVERSAL_ACCEL_BOOST = float(fcfg.get("reversal_accel_boost", 1.8))
    DEPTH_OE_MIN_CUTOFF = float(fcfg["depth_oe_min_cutoff"])
    DEPTH_OE_BETA = float(fcfg["depth_oe_beta"])
    Z3D_MAX_JUMP = float(fcfg.get("z3d_max_jump", Z3D_MAX_JUMP))
    K_ORIENTATION = float(fcfg["k_orientation"])
    MAX_VEL_JOINT = float(fcfg["max_vel_joint"])
    MAX_VEL_ANGULAR = float(fcfg["max_vel_angular"])
    # 伸展度增益调度（旧 yaml 没有这四个键时用代码默认值）
    REACH_SOFT_START = float(fcfg.get("reach_soft_start", REACH_SOFT_START))
    REACH_SOFT_END = float(fcfg.get("reach_soft_end", REACH_SOFT_END))
    REACH_MIN_SCALE = float(fcfg.get("reach_min_scale", REACH_MIN_SCALE))
    REACH_DOCK_DIST = float(fcfg.get("reach_dock_dist", REACH_DOCK_DIST))
    ws = cfg["workspace"]
    WORKSPACE_MIN = np.array(ws["min"], dtype=np.float64)
    WORKSPACE_MAX = np.array(ws["max"], dtype=np.float64)


# ═══ 以下数学函数与 palm_follow_fixed_orientation.py 完全一致 ═══

def compute_damped_pseudoinverse(J, damping=0.01):
    m, n = J.shape
    JJT = J @ J.T
    damping_matrix = (damping ** 2) * np.eye(m)
    try:
        return J.T @ np.linalg.inv(JJT + damping_matrix)
    except np.linalg.LinAlgError:
        return J.T @ np.linalg.inv(JJT + ((damping * 10) ** 2) * np.eye(m))


def compute_manipulability(J):
    return np.sqrt(max(0.0, np.linalg.det(J @ J.T)))


def rotation_matrix_to_axis_angle(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if abs(theta) < 1e-6:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-3:
        S = (R + np.eye(3)) / 2.0
        axis = np.array([np.sqrt(max(S[0, 0], 0.0)),
                         np.sqrt(max(S[1, 1], 0.0)),
                         np.sqrt(max(S[2, 2], 0.0))])
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
    return rotation_matrix_to_axis_angle(R_desired @ R_current.T)


def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def compute_tcp_fk_and_jacobian(robot, joint_positions):
    """返回 TCP 位姿和 LOCAL_WORLD_ALIGNED TCP 雅可比（6x6）"""
    import pinocchio as pin
    q_full = robot._build_q_full(joint_positions)

    pin.forwardKinematics(robot.model, robot.data, q_full)
    pin.updateFramePlacements(robot.model, robot.data)
    pin.computeJointJacobians(robot.model, robot.data, q_full)

    last_jid = robot.model.getJointId(robot.joint_names[-1])
    oMi = robot.data.oMi[last_jid]
    tcp_position = oMi.translation + oMi.rotation @ TOOL_OFFSET
    tcp_rotation = oMi.rotation.copy()

    J_full = pin.getJointJacobian(robot.model, robot.data, last_jid,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_tcp = J_full.copy()
    J_tcp[:3, :] -= skew(oMi.rotation @ TOOL_OFFSET) @ J_full[3:, :]

    cols = [robot.model.joints[robot.model.getJointId(n)].idx_v
            for n in robot.joint_names]
    return tcp_position.copy(), tcp_rotation, J_tcp[:, cols]


def camera_velocity_to_tcp(v_camera, T_tcp_camera):
    """相机系速度 → TCP 系速度（速度伴随变换）"""
    R_tcp_cam = T_tcp_camera[:3, :3]
    t_tcp_cam = T_tcp_camera[:3, 3]
    V = np.zeros((6, 6))
    V[:3, :3] = R_tcp_cam
    V[:3, 3:] = skew(t_tcp_cam) @ R_tcp_cam
    V[3:, 3:] = R_tcp_cam
    return V @ v_camera


# ═══ 主程序 ═══

def main():
    args = parse_args()

    # 跟随参数：yaml 为默认值，命令行参数优先级更高
    follow_cfg = load_follow_cfg(args.follow_cfg)
    apply_follow_cfg(follow_cfg)
    fcfg = follow_cfg["follow"]

    desired_depth = float(fcfg["distance"])
    lambda_gain_xyz = np.array([float(fcfg["gain_x"]),
                                float(fcfg["gain"]), float(fcfg["gain"])])
    max_vel_cart = float(fcfg["max_vel"])
    camera_fps = int(fcfg["fps"])
    if args.distance is not None:
        if not 0.18 <= args.distance <= 0.60:
            raise ValueError("--distance 建议在 0.18~0.60 米之间"
                             "（Gemini2 近距极限 ~0.15m，太远跟踪不稳）")
        desired_depth = args.distance
    if args.gain is not None:
        lambda_gain_xyz[1:] = args.gain       # Y/Z（竖直、深度）
    if args.gain_x is not None:
        lambda_gain_xyz[0] = args.gain_x      # X（水平，左右跟随）
    if args.max_vel is not None:
        max_vel_cart = args.max_vel
    if args.fps is not None:
        camera_fps = args.fps

    follow_enabled = not args.no_follow
    grasp_enabled = not args.no_gripper

    # 捏合 → 夹爪映射配置
    grasp_cfg = load_grasp_cfg(args.grasp_cfg)
    grasp_cfg["gripper"]["enabled"] = (grasp_cfg["gripper"]["enabled"]
                                       and grasp_enabled)
    grasp_on = grasp_cfg["gripper"]["enabled"] and not args.no_arm
    pinch_closed_px = float(grasp_cfg["pinch"]["closed_px"])
    pinch_open_px = float(grasp_cfg["pinch"]["open_px"])
    grasp_smoothing = float(grasp_cfg["pinch"]["smoothing"])
    grasp_deadband = float(grasp_cfg["pinch"]["deadband_deg"])
    pinch_filter_mode = str(grasp_cfg["pinch"].get("filter", "one_euro")).lower()
    gripper_closed_deg = float(grasp_cfg["gripper"]["closed_deg"])
    gripper_open_deg = float(grasp_cfg["gripper"]["open_deg"])
    # 手臂运动时冻结夹爪（相机装在手腕上，夹爪动作的振动会干扰跟随）
    hold_while_moving = bool(grasp_cfg["gripper"].get("hold_while_moving", True))
    move_speed_thresh = float(grasp_cfg["gripper"].get("move_speed_threshold", 0.05))
    # 配置合法性检查（映射区间退化会导致 np.interp/除零异常）
    if pinch_open_px <= pinch_closed_px:
        raise ValueError(f"grasp.yaml 配置错误: pinch.open_px({pinch_open_px}) "
                         f"必须大于 pinch.closed_px({pinch_closed_px})")
    if abs(gripper_open_deg - gripper_closed_deg) < 5.0:
        raise ValueError(f"grasp.yaml 配置错误: 夹爪开合区间过小 "
                         f"({gripper_closed_deg}° ~ {gripper_open_deg}°)，"
                         f"请运行 gripper_j7_calibrate.py 重新标定")

    print("\n" + "=" * 60)
    print("手掌跟随 + 捏合抓取 (B601 RS + Gemini 2)")
    print("=" * 60)
    print(f"模式: follow={'ON' if follow_enabled else 'OFF'}  "
          f"gripper={'PREVIEW(仅预览)' if (grasp_cfg['gripper']['enabled'] and args.no_arm) else ('ON' if grasp_on else 'OFF')}  "
          f"{'[纯视觉演练]' if args.no_arm else ''}"
          f"{'[无窗口]' if args.headless else ''}")

    # ── 手眼标定（仅跟随模式需要）──
    T_tcp_camera = None
    if follow_enabled and not args.no_arm:
        calib_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "config", "hand_eye_calibration.json")
        if not os.path.exists(calib_file):
            raise FileNotFoundError(
                f"未找到 {calib_file}，请先运行 tools/handeye_joint_calib.py 完成标定；"
                f"或先用 --no-follow 只验证夹爪功能")
        with open(calib_file, 'r') as f:
            T_tcp_camera = np.array(json.load(f)["T_tcp_camera"])

    # ── 相机与检测 ──
    print(f"初始化 Gemini 2 相机（{camera_fps} fps）...")
    cam = Gemini2Camera(fps=camera_fps)
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    # 捏合像素阈值按 640 宽标定（grasp.yaml 中 closed_px/open_px）；
    # 相机实际分辨率不同时（如 60fps 模式下 1280x720）按比例自动换算
    pinch_scale = cam.width / 640.0 if cam.width != 640 else 1.0
    if pinch_scale != 1.0:
        pinch_closed_px *= pinch_scale
        pinch_open_px *= pinch_scale
        print(f"[配置] 相机宽度 {cam.width}px，捏合阈值自动 x{pinch_scale:.2f}: "
              f"{pinch_closed_px:.0f}px / {pinch_open_px:.0f}px")
    desired_u, desired_v = int(round(cx)), int(round(cy))   # 期望掌心点 = 画面中心

    print("初始化手掌检测...")
    detector = PalmDetector(backend=args.detector_backend)
    tracker = PalmTracker()

    # ── 机械臂 ──
    robot = None
    desired_rotation = None
    if not args.no_arm:
        from rebot_adapter import RebotAdapter
        print("初始化机械臂...")
        robot = RebotAdapter()

        print(f"移动到初始位置: {HOME_JOINT_POS}")
        robot.move_to_joints(HOME_JOINT_POS, speed_deg_s=15.0)
        time.sleep(0.5)

        if follow_enabled:
            fk_home = robot.forward_kinematics()
            desired_rotation = np.array(fk_home['rotation'])     # 固定姿态基准
            print(f"初始 TCP 位置: {np.array(fk_home['position']).round(3)} m")

    # 夹爪平滑初值 = 当前实际角度，避免启动瞬间夹爪猛动
    if robot is not None:
        smoothed_gripper_deg = robot.get_gripper_deg()
    else:
        smoothed_gripper_deg = (gripper_closed_deg + gripper_open_deg) / 2.0
    last_sent_gripper_deg = smoothed_gripper_deg
    # 捏合滤波器：One Euro（默认，静止稳/动时快）或 EMA（旧行为）
    pinch_oe = OneEuroFilter(
        float(grasp_cfg["pinch"].get("oe_min_cutoff", 1.2)),
        float(grasp_cfg["pinch"].get("oe_beta", 0.05)))
    if pinch_filter_mode == "one_euro":
        pinch_oe(smoothed_gripper_deg)      # 用当前角度预填充，避免首次检测跳变
    pinch_px_now = [None]           # 供 HUD/调试显示
    gripper_frozen_now = [False]    # 夹爪是否因手臂运动被冻结

    # ── 共享状态 ────────────────────────────────────────────
    lock = threading.Lock()
    current_target_cam = [None]       # 掌心在相机系下的 3D 坐标
    palm_visible = [False]
    g_pressed = [False]
    quit_flag = [False]
    control_status = {"joint_speed": 0.0, "error": "", "active": False}

    if follow_enabled:
        print(f"跟随参数: 距离 {desired_depth}m  "
              f"gain(x/y/z)={lambda_gain_xyz.tolist()}  "
              f"max_vel={max_vel_cart}  前馈={VEL_FEEDFORWARD}")
    if grasp_on or (grasp_cfg["gripper"]["enabled"] and args.no_arm):
        print(f"夹爪映射: 捏合 {pinch_closed_px:.0f}px → {gripper_closed_deg:.1f}° | "
              f"张开 {pinch_open_px:.0f}px → {gripper_open_deg:.1f}°"
              f"  滤波={pinch_filter_mode}"
              + ("  [手臂运动时冻结]" if hold_while_moving else ""))
    print("操作: g/空格 启停跟随 | q/Esc/Ctrl+C 退出并安全回零")
    print("=" * 60 + "\n")

    # ── 键盘监听 ────────────────────────────────────────────
    last_toggle = [0.0]

    def toggle_tracking():
        if not follow_enabled:
            print("\n[跟随] 当前为 --no-follow 模式，跟随不可用")
            return
        now = time.time()
        with lock:
            if now - last_toggle[0] < 0.25:
                return
            g_pressed[0] = not g_pressed[0]
            enabled = g_pressed[0]
            last_toggle[0] = now
        print("\n[跟随] 启动" if enabled else "\n[跟随] 停止（保持当前位置）")

    def on_press(key):
        try:
            if key.char == 'g':
                toggle_tracking()
            elif key.char == 'q':
                quit_flag[0] = True    # headless 模式下的退出通道
        except AttributeError:
            if key == pynput_kb.Key.space:
                toggle_tracking()
            elif key == pynput_kb.Key.esc:
                quit_flag[0] = True    # Esc = 退出并安全回零（统一退出路径）

    kb_listener = pynput_kb.Listener(on_press=on_press)
    kb_listener.start()

    # ── 伺服控制线程（50Hz；--no-arm 时不启动）────────────
    stop_ctrl = threading.Event()
    actual_velocity = [np.zeros(6)]
    prev_cmd_vel = [np.zeros(3)]          # 上周期下发的速度指令（加速度限制参照）
    lost_counter = [0]
    prev_target_cam = [None]            # 上一周期掌心目标（前馈用）
    target_vel_f = [np.zeros(3)]        # 平滑后的目标速度估计（相机系）
    in_deadband = [False]               # 死区迟滞状态（True=已收敛到死区内）
    coast_offset = [np.zeros(3)]        # 丢失外推的累计位移（相机系）
    coast_vel = [np.zeros(3)]           # 丢失瞬间的目标速度（外推用）
    coast_err = [1e9]                   # 丢失瞬间的误差（大误差禁止外推）
    near_wall = [True]                  # 上一周期 TCP 是否贴近虚拟墙
    reach_r_now = [0.0]                 # 上一周期水平伸展半径
    depth_oe = OneEuroFilter(DEPTH_OE_MIN_CUTOFF, DEPTH_OE_BETA)  # 深度滤波
    last_valid_depth = [0.0]    # 上一帧有效深度（跳变剔除的参照）
    z3d_reject_count = [0]      # 连续跳变剔除帧数（防锁死：≥15 帧放行）
    reacq_buf = [[]]            # 参照无效时的恢复校验缓冲（3 帧一致性）
    ctrl_thread = None

    def servo_control_loop():
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

            # ── 丢失外推（滑行）：运动模糊掉检通常只有 2~6 帧。
            # 按丢失瞬间的目标速度继续外推目标 ≤6 帧(120ms)，速度逐帧衰减。
            # 安全约束（log(15) 撞墙事故的教训）：
            #  1) 外推速度限幅 0.35m/s——恢复瞬间的目标跳变会污染前馈
            #     速度估计，不限幅会朝幻影目标冲出去；
            #  2) 接近虚拟墙（<5cm）或手臂伸展（>reach_soft_start）时
            #     禁止外推——这些区域没有容错余量；
            #  3) 外推只在误差较小时启用（跟丢的是"移动中的手"，
            #     误差大说明目标本身可疑）。
            if visible:
                coast_offset[0][:] = 0.0
                cv = target_vel_f[0].copy()
                cvn = float(np.linalg.norm(cv))
                if cvn > COAST_MAX_VEL:
                    cv *= COAST_MAX_VEL / cvn
                coast_vel[0] = cv
                coast_err[0] = float(np.linalg.norm(
                    target_cam - np.array([0.0, 0.0, desired_depth]))) \
                    if target_cam is not None else 1e9
            elif (target_cam is not None and 0 < lost_counter[0] <= COAST_FRAMES
                    and coast_err[0] < 0.15
                    and not near_wall[0] and reach_r_now[0] <= REACH_SOFT_START):
                coast_offset[0] += coast_vel[0] * CTRL_RATE
                coast_vel[0] *= COAST_DECAY
            if target_cam is not None and lost_counter[0] > 0:
                target_cam = target_cam + coast_offset[0]

            if following and target_cam is not None and lost_counter[0] <= LOST_HOLD_FRAMES:
                try:
                    # 热循环用内部下发值做 FK/雅可比（无 CAN I/O，不挤占 200Hz 下发）
                    current_joints = robot.get_command_pos()
                    tcp_pos, tcp_rot, J_robot = compute_tcp_fk_and_jacobian(
                        robot, current_joints)

                    with lock:
                        control_status["tcp"] = tcp_pos.round(3).tolist()

                    # ── 硬墙：越界 >5cm → 自动回 HOME 后继续跟随 ──
                    if (np.any(tcp_pos < WORKSPACE_MIN - 0.05)
                            or np.any(tcp_pos > WORKSPACE_MAX + 0.05)):
                        robot.hold()
                        print(f"\n[安全] TCP 越过硬墙 {tcp_pos.round(2).tolist()}"
                              f"，自动回 HOME...")
                        robot.move_to_joints(HOME_JOINT_POS, speed_deg_s=12.0,
                                             wait=True, timeout=30.0)
                        actual_velocity[0] = np.zeros(6)
                        prev_cmd_vel[0][:] = 0.0
                        with lock:
                            control_status["error"] = "已自动回 HOME"
                        continue

                    # 主任务：掌心收敛到相机系 (0, 0, desired_depth)
                    target_desired = np.array([0.0, 0.0, desired_depth])
                    pos_err_cam = target_cam - target_desired
                    pos_err_norm = float(np.linalg.norm(pos_err_cam))
                    # 死区迟滞：进入死区后，误差要超过 1.5×死区才重新启动，
                    # 防止掌心 3D 噪声让指令在死区边界反复穿入穿出
                    # （静止时持续小幅振荡的机械根源）
                    if in_deadband[0]:
                        if pos_err_norm > POSITION_DEADBAND * DEADBAND_HYSTERESIS:
                            in_deadband[0] = False
                    elif pos_err_norm < POSITION_DEADBAND:
                        in_deadband[0] = True
                    v_cam = np.zeros(6)
                    if not in_deadband[0]:
                        # 分方向增益：X(水平)单独较低，防止 J1 偏航饱和振荡
                        v_cam[:3] = SERVO_DIRECTION * lambda_gain_xyz * pos_err_cam

                    # 目标速度前馈：掌心在动时叠加其速度估计，
                    # 消除纯反馈跟随的高速滞后（手快臂慢的根源之一）
                    if VEL_FEEDFORWARD > 0.0:
                        if prev_target_cam[0] is not None:
                            # 目标跳变守卫：恢复/误检导致单周期目标位移
                            # >6cm 时，本帧的速度估计是纯污染，跳过更新
                            # （否则假速度会被记住，下次掉检时被外推放大）
                            if np.linalg.norm(target_cam - prev_target_cam[0]) < 0.06:
                                tv = (target_cam - prev_target_cam[0]) / CTRL_RATE
                                tv = np.clip(tv, -FF_MAX_VEL, FF_MAX_VEL)
                                target_vel_f[0] = (FF_SMOOTH * tv
                                                   + (1 - FF_SMOOTH) * target_vel_f[0])
                        prev_target_cam[0] = target_cam.copy()
                        # 静止门控：手不动时目标速度估计≈纯噪声
                        # （深度抖动 /CTRL_RATE 会被放大成 0.2+ m/s 假速度），
                        # 低于阈值直接置零，不注入伺服
                        if in_deadband[0] or \
                                np.linalg.norm(target_vel_f[0]) < FF_MIN_SPEED:
                            target_vel_f[0] = np.zeros(3)
                        else:
                            # 近端渐弱：接近目标时前馈按比例退出，
                            # 避免停靠瞬间前馈惯性把机械臂推过目标
                            # （到位后振荡几下才停稳的主因）
                            ff_scale = min(1.0, pos_err_norm / FF_TAPER_DIST)
                            v_cam[:3] += (VEL_FEEDFORWARD * ff_scale
                                          * target_vel_f[0])

                    # 次任务：姿态保持在初始值
                    ori_err = compute_orientation_error(tcp_rot, desired_rotation)
                    # 软饱和：tanh 平滑限幅。硬 clip 在大幅水平移动时会使
                    # J1 指令长期顶在限速上（非线性饱和），与延迟叠加形成
                    # 极限环振荡；tanh 让接近上限时增益平滑衰减
                    v_cam[:3] = max_vel_cart * np.tanh(v_cam[:3] / max_vel_cart)

                    # 伸展度增益调度：手臂前伸越多，J1 杠杆臂越长、整臂刚度
                    # 越低，收缩位姿刚好的增益在伸展位姿会超出稳定裕度
                    # （前伸较多时停靠大振荡的根源）。
                    # 只衰减停靠段（误差 ≤ REACH_DOCK_DIST）：巡航段（误差大、
                    # 追赶手）保持全速，避免伸展区"不跟手"。
                    reach_r = float(np.hypot(tcp_pos[0], tcp_pos[1]))
                    reach_r_now[0] = reach_r
                    reach_scale = 1.0
                    if reach_r > REACH_SOFT_START:
                        dock = float(np.clip(
                            1.0 - pos_err_norm / REACH_DOCK_DIST, 0.0, 1.0))
                        reach_scale = float(np.clip(
                            1.0 - (1.0 - REACH_MIN_SCALE)
                            * (reach_r - REACH_SOFT_START)
                            / max(REACH_SOFT_END - REACH_SOFT_START, 1e-6),
                            REACH_MIN_SCALE, 1.0))
                        v_cam[:3] *= 1.0 - (1.0 - reach_scale) * dock

                    angular_world = (K_ORIENTATION * ori_err
                                     if np.linalg.norm(ori_err) > ORIENTATION_DEADBAND
                                     else np.zeros(3))
                    angular_world = np.clip(angular_world, -MAX_VEL_ANGULAR, MAX_VEL_ANGULAR)

                    # 方向反转时清除平滑惯性，消除「先反向冲一下再拉回」
                    if (REVERSAL_FAST_RESPONSE
                            and np.dot(v_cam[:3], actual_velocity[0][:3]) < 0):
                        actual_velocity[0][:3] = 0.0

                    # 速度平滑
                    actual_velocity[0] += (v_cam - actual_velocity[0]) * SMOOTH_ALPHA

                    # ── 加速度限制（启停梯形化）──
                    # EMA 只平滑数值、不限变化率：启动/刹停瞬间指令仍在
                    # 1-2 个周期内跳到目标值，对关节等效阶跃冲击
                    # （log17 抖动主因：joint_speed 在 0 与 2-3 rad/s
                    # 之间反复弹跳，每次跳变激发一次结构振动）。
                    # 限制每周期速度增量 ≤ MAX_ACCEL×CTRL_RATE，启停变斜坡；
                    # 方向反转时给 BOOST 倍额度，保住纠错响应速度
                    dv_cmd = actual_velocity[0][:3] - prev_cmd_vel[0]
                    accel_cap = MAX_ACCEL
                    if np.dot(actual_velocity[0][:3], prev_cmd_vel[0]) < 0:
                        accel_cap *= REVERSAL_ACCEL_BOOST
                    max_dv = accel_cap * CTRL_RATE
                    dv_norm = float(np.linalg.norm(dv_cmd))
                    if dv_norm > max_dv:
                        actual_velocity[0][:3] = (prev_cmd_vel[0]
                                                  + dv_cmd * (max_dv / dv_norm))
                    prev_cmd_vel[0] = actual_velocity[0][:3].copy()

                    # 相机系 → TCP 系 → 世界系
                    v_tcp = camera_velocity_to_tcp(actual_velocity[0], T_tcp_camera)
                    v_world = np.zeros(6)
                    v_world[0:3] = tcp_rot @ v_tcp[0:3]
                    v_world[3:6] = angular_world

                    # ── 软墙：只削掉「继续往外走」的速度分量 ──
                    wall_touch = False
                    for ax in range(3):
                        if tcp_pos[ax] <= WORKSPACE_MIN[ax] and v_world[ax] < 0:
                            v_world[ax] = 0.0
                            wall_touch = True
                        elif tcp_pos[ax] >= WORKSPACE_MAX[ax] and v_world[ax] > 0:
                            v_world[ax] = 0.0
                            wall_touch = True
                    # 距墙 <5cm 视为贴墙（丢失外推的安全禁区）
                    near_wall[0] = bool(
                        np.any(tcp_pos < WORKSPACE_MIN + 0.05)
                        or np.any(tcp_pos > WORKSPACE_MAX - 0.05))

                    # 可操作度自适应阻尼（奇异保护）
                    manip = compute_manipulability(J_robot)
                    if DAMPING_ADAPTIVE and manip < MANIPULABILITY_THRESHOLD:
                        damping = DAMPING_BASE * (1.0 + (MANIPULABILITY_THRESHOLD - manip) * 100)
                        v_world *= max(MIN_MANIPULABILITY_SPEED_SCALE,
                                       manip / MANIPULABILITY_THRESHOLD)
                    else:
                        damping = DAMPING_BASE

                    joint_vel = compute_damped_pseudoinverse(J_robot, damping) @ v_world
                    joint_vel = np.clip(joint_vel, -MAX_VEL_JOINT, MAX_VEL_JOINT)
                    if (in_deadband[0] and
                            np.linalg.norm(ori_err) <= ORIENTATION_DEADBAND):
                        joint_vel[:] = 0.0

                    robot.joint_vel_cmd(joint_vel)   # 第 7 通道自动携带夹爪目标角
                    with lock:
                        control_status["joint_speed"] = float(np.linalg.norm(joint_vel))
                        _status = "贴墙滑动中" if wall_touch else ""
                        if reach_scale < 0.95:
                            _status += ("" if not _status else " ") + \
                                f"伸展衰减x{reach_scale:.2f}"
                        control_status["error"] = _status
                        control_status["active"] = True

                except Exception as exc:
                    robot.hold()
                    with lock:
                        control_status["error"] = str(exc)
                        control_status["active"] = False
            else:
                # 未启动跟随 / 丢手超阈值 / 仅夹爪模式：手臂保持不动。
                # hold() 每周期重发 7 通道指令，夹爪目标角随之下发，
                # 因此 --no-follow 模式下捏合夹爪依然实时生效。
                actual_velocity[0] = np.zeros(6)
                prev_cmd_vel[0][:] = 0.0
                prev_target_cam[0] = None           # 前馈状态复位
                target_vel_f[0] = np.zeros(3)
                in_deadband[0] = False              # 重新启动跟随时重新收敛
                coast_offset[0][:] = 0.0            # 外推状态一并复位
                coast_vel[0][:] = 0.0
                coast_err[0] = 1e9
                robot.hold()
                with lock:
                    control_status["joint_speed"] = 0.0
                    control_status["active"] = False
                    if following and lost_counter[0] > LOST_HOLD_FRAMES:
                        control_status["error"] = "手掌丢失，保持位置"

            elapsed = time.time() - t0
            if CTRL_RATE - elapsed > 0:
                time.sleep(CTRL_RATE - elapsed)

    if robot is not None:
        ctrl_thread = threading.Thread(target=servo_control_loop, daemon=True)
        ctrl_thread.start()

    # ── 视觉主循环 ──────────────────────────────────────────
    if not args.headless:
        cv2.namedWindow('Palm Follow + Grasp', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Palm Follow + Grasp', 640, 480)

    fps = 0.0
    t_prev_frame = time.time()
    ms_grab = [0.0]   # get_frames 耗时 EMA (ms)
    ms_det = [0.0]    # 手掌识别耗时 EMA (ms)

    try:
        last_frame_time = time.time()
        while not quit_flag[0]:
            _t0 = time.perf_counter()
            color_image, depth_raw, depth_scale = cam.get_frames()
            _t1 = time.perf_counter()
            if color_image is None:
                # 无帧分支（USB 抖动/超时）：必须继续泵 GUI 事件，
                # 否则 waitKey 长期不执行，窗口被系统判"未响应"（卡死）
                if not args.headless:
                    cv2.waitKey(1)
                time.sleep(0.002)   # 防空转拉满 CPU
                # 帧看门狗：连续 2 秒无有效帧 → 判定相机掉线，自动重连
                # （手腕线束随机械臂弯折，USB 瞬断后 SDK 不会自愈）
                if time.time() - last_frame_time > 2.0:
                    print("\n[相机] 超过 2 秒无有效帧，尝试断开重连...")
                    try:
                        cam.stop()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    try:
                        cam = Gemini2Camera(fps=camera_fps)
                        last_valid_depth[0] = 0.0   # 深度参照重建（走重抓校验）
                        reacq_buf[0].clear()
                        print("[相机] 重连成功")
                    except Exception as exc:
                        print(f"[相机] 重连失败（{exc}），2 秒后重试...")
                    last_frame_time = time.time()
                continue
            last_frame_time = time.time()

            # 帧率统计（EMA）
            t_now = time.time()
            if t_now > t_prev_frame:
                fps = 0.2 * (1.0 / (t_now - t_prev_frame)) + 0.8 * fps
            t_prev_frame = t_now

            u, v, landmarks = detector.detect(color_image)
            _t2 = time.perf_counter()
            # 耗时统计（EMA，毫秒）：取帧（相机等待+USB传输）/ 识别
            ms_grab[0] = 0.9 * ms_grab[0] + 0.1 * (_t1 - _t0) * 1000
            ms_det[0] = 0.9 * ms_det[0] + 0.1 * (_t2 - _t1) * 1000
            track = tracker.update((u, v) if u is not None else None)

            if track is not None and track['lost'] == 0:
                ui, vi = int(round(track['u'])), int(round(track['v']))
                depth_m = get_median_depth(depth_raw, depth_scale, ui, vi)
                # 深度跳变剔除：BS 模式 + MediaPipe 误检会让单帧 z3d 跳到
                # 0.8~1.0m（实测），直接打进伺服就是一次猛冲（整体抖动
                # 的主因）。手一帧（16ms）物理上不可能移动 8cm，
                # 超阈值视为无效帧：不更新 3D 目标（等效于本帧手掌丢失），
                # 也不喂给滤波器（避免污染 One Euro 状态）。
                # 防锁死：真尖峰只有 1~2 帧，连续多帧超阈值才说明
                # 参照已过时。阈值定 15 帧（0.25s）：真实快速贴近/远离
                # 每帧只有几毫米变化，永远碰不到 8cm 阈值；定太低会被
                # "掌心空洞读到背景"这类连续假深度（如 0.4→0.94m）骗开
                if depth_m > 0 and last_valid_depth[0] > 0 and \
                        abs(depth_m - last_valid_depth[0]) > Z3D_MAX_JUMP:
                    z3d_reject_count[0] += 1
                    if z3d_reject_count[0] < 15:
                        depth_m = 0.0
                    # 连续 ≥15 帧：认定为参照彻底过时，放行并重置参照
                else:
                    z3d_reject_count[0] = 0
                # 恢复校验（log(16) 撞墙事故的修复）：参照无效时
                # （长丢失重置后/首次启动），限速器和跳变剔除都不设防，
                # 第一帧假深度（如 0.765m 背景）会直接打穿到墙上。
                # 要求连续 3 帧读数互相一致（max-min ≤ 跳变阈值）才接受，
                # 取中位数建参照。只发生在长丢失恢复时，代价 50ms；
                # 短暂模糊掉检参照保留，完全不受影响。
                if depth_m > 0 and last_valid_depth[0] == 0:
                    reacq_buf[0].append(depth_m)
                    if len(reacq_buf[0]) > 3:
                        reacq_buf[0].pop(0)
                    if len(reacq_buf[0]) == 3 and \
                            max(reacq_buf[0]) - min(reacq_buf[0]) <= Z3D_MAX_JUMP:
                        depth_m = float(np.median(reacq_buf[0]))
                    else:
                        depth_m = 0.0   # 校验未通过，本帧视为无效
                if depth_m > 0:
                    # 深度限速（仿真验证：恢复尖峰归零）：真实手掌在 60fps
                    # 下纵向每帧最多移动 ~1cm，单帧 |Δz| 超过 5cm 必然是
                    # 误检。跳变剔除在参照重置后（长丢失恢复）不设防，
                    # 限速是最后防线——0.4→0.88m 的假深度被钳成缓坡
                    if last_valid_depth[0] > 0:
                        depth_m = float(np.clip(
                            depth_m,
                            last_valid_depth[0] - Z3D_MAX_RATE,
                            last_valid_depth[0] + Z3D_MAX_RATE))
                    # 深度通道滤波：(u,v) 有 One Euro，但深度一直是裸值——
                    # 静止时 z3d 抖动主要来自这里。One Euro 静止时强滤波，
                    # 手快速靠近/远离时自动放开，不影响纵向跟随速度
                    depth_m = depth_oe(depth_m)
                    last_valid_depth[0] = depth_m
                    track['x3d'], track['y3d'], track['z3d'] = pixel_to_3d(
                        track['u'], track['v'], depth_m, fx, fy, cx, cy)
                else:
                    track['z3d'] = 0.0
            elif track is None or track['lost'] >= 30:
                # 长时间（≥30帧≈0.5s）真正丢失才重置深度参照；
                # 短暂丢失（运动模糊等 1~29 帧）保留参照，让跳变剔除
                # 在重新捕获瞬间继续生效——否则恢复帧的假深度会
                # 绕过门控直接打进伺服，造成 joint_speed 突刺
                last_valid_depth[0] = 0.0
                z3d_reject_count[0] = 0
                reacq_buf[0].clear()

            with lock:
                if track is not None and track['lost'] == 0 and track['z3d'] > 0:
                    current_target_cam[0] = [track['x3d'], track['y3d'], track['z3d']]
                    palm_visible[0] = True
                else:
                    palm_visible[0] = False
                    # 注意：不清空 current_target_cam，短暂丢失时控制侧沿用上值
                    if track is None:
                        current_target_cam[0] = None

            # ── 捏合 → 夹爪（与跟随共用同一次检测结果）──
            # 新鲜度门控（log19 修复 + log21 修正）：
            # 跟踪器外推帧的捏合距离是纯噪声（实测 24~201px 乱跳），
            # 必须冻结；但捏合是 2D 测量、与深度无关——深度失效帧
            # （z3d=0）的捏合值依然准确，不能一并冻结，否则夹爪在
            # 近距/侧倾等深度易丢的场景表现得"没有反应"。
            # 因此门控只要求关键点是本帧实测（lost==0），不看深度。
            pinch_fresh = (track is not None and track['lost'] == 0)
            pinch_pts = None
            if (grasp_cfg["gripper"]["enabled"] and landmarks is not None
                    and pinch_fresh):
                pinch_px, p1, p2 = pinch_distance_px(landmarks, color_image.shape)
                pinch_px_now[0] = pinch_px
                pinch_pts = (p1, p2)

                # 归一化：0 = 捏合（闭合），1 = 张开；两端截断
                t_norm = float(np.interp(pinch_px,
                                         [pinch_closed_px, pinch_open_px], [0.0, 1.0]))
                target_deg = gripper_closed_deg + t_norm * (
                    gripper_open_deg - gripper_closed_deg)

                # 滤波防抖：One Euro（静止强滤波、运动弱滤波）或 EMA
                if pinch_filter_mode == "one_euro":
                    smoothed_gripper_deg = pinch_oe(target_deg)
                else:
                    smoothed_gripper_deg = (grasp_smoothing * target_deg
                                            + (1.0 - grasp_smoothing) * smoothed_gripper_deg)

                # 手臂运动中冻结夹爪：相机在手腕上，夹爪动作的振动会
                # 抖动画面的掌心 3D 估计，让跟随显得不稳（hold_while_moving）
                with lock:
                    arm_moving = control_status["joint_speed"] > move_speed_thresh
                gripper_frozen = hold_while_moving and arm_moving
                gripper_frozen_now[0] = gripper_frozen

                # 死区：变化太小不更新（指尖是关键点中噪声最大的）
                if (not gripper_frozen and
                        abs(smoothed_gripper_deg - last_sent_gripper_deg) >= grasp_deadband):
                    if robot is not None:
                        robot.set_gripper_deg(smoothed_gripper_deg)
                    last_sent_gripper_deg = smoothed_gripper_deg
            # 手丢失时：夹爪保持最后一次目标角，不漂移

            # ── 调试状态打印（每 0.5s 一行）──
            if time.time() - getattr(main, "_last_dbg", 0) > 0.5:
                main._last_dbg = time.time()
                det = "None"
                if track is not None:
                    det = (f"u={track['u']:.0f} v={track['v']:.0f} "
                           f"lost={track['lost']} z3d={track['z3d']:.3f}")
                pinch_info = (f"pinch={pinch_px_now[0]:.0f}px→"
                              f"{last_sent_gripper_deg:.1f}°"
                              if pinch_px_now[0] is not None else "pinch=None")
                with lock:
                    _fol, _vis = g_pressed[0], palm_visible[0]
                    _spd = control_status["joint_speed"]
                    _err = control_status["error"]
                _tcp = control_status.get("tcp", None)
                print(f"[DBG] fps={fps:4.1f}  palm=({det})  visible={_vis} "
                      f"following={_fol}  joint_speed={_spd:.3f}  "
                      f"{pinch_info}  tcp={_tcp}  err={_err}  "
                      f"延迟ms(取帧/识别)={ms_grab[0]:.0f}/{ms_det[0]:.0f}")

            # ── 可视化 ──
            if not args.headless:
                display = color_image.copy()
                if landmarks is not None:
                    draw_landmarks(display, landmarks)
                draw_palm(display, track, desired_u, desired_v)

                # 捏合连线 + 开合百分比
                if pinch_pts is not None:
                    cv2.line(display, pinch_pts[0], pinch_pts[1], (255, 0, 255), 3)
                    cv2.circle(display, pinch_pts[0], 8, (255, 0, 255), cv2.FILLED)
                    cv2.circle(display, pinch_pts[1], 8, (255, 0, 255), cv2.FILLED)
                if grasp_cfg["gripper"]["enabled"]:
                    span = gripper_open_deg - gripper_closed_deg
                    open_pct = (float(np.clip(
                        (last_sent_gripper_deg - gripper_closed_deg) / span, 0.0, 1.0))
                        * 100.0 if abs(span) > 1e-6 else 0.0)
                    frozen_txt = " [FROZEN]" if gripper_frozen_now[0] else ""
                    cv2.putText(display, f"Gripper: {open_pct:.0f}% "
                                f"({last_sent_gripper_deg:.1f}deg){frozen_txt}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 255), 2)

                with lock:
                    following = g_pressed[0]
                    status = dict(control_status)

                if args.no_arm:
                    state = "VISION ONLY (--no-arm)"
                elif not follow_enabled:
                    state = "GRASP ONLY (arm holding)"
                else:
                    state = "FOLLOWING" if (following and status["active"]) else \
                            ("HOLD (hand lost)" if following else "IDLE (press g/space)")
                color_map = {"FOLLOWING": (0, 255, 0), "IDLE (press g/space)": (0, 200, 255)}
                cv2.putText(display, state, (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color_map.get(state, (0, 165, 255)), 2)
                cv2.putText(display, f"FPS: {fps:.0f}  JointSpeed: "
                            f"{status['joint_speed']:.3f} rad/s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if status["error"]:
                    cv2.putText(display, f"CTRL: {status['error'][:60]}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.imshow('Palm Follow + Grasp', display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    quit_flag[0] = True
                if key in (ord('g'), ord(' ')):
                    toggle_tracking()
            else:
                time.sleep(0.005)   # headless：让出 CPU，等待键盘事件

    except KeyboardInterrupt:
        print("\n程序被中断（Ctrl+C），走统一退出路径...")
    finally:
        stop_ctrl.set()
        if ctrl_thread is not None:
            ctrl_thread.join(timeout=1.0)
        kb_listener.stop()
        detector.close()
        if robot is not None:
            print("机械臂保持当前位置...")
            robot.hold()
            time.sleep(0.3)
            # 安全停机：缓慢回零 → 确认到位 → 才失能（防止自由下落）。
            # 这几秒主线程被占用、视觉窗口不刷新是正常现象，勿重复按键
            print("正在安全回零（约3秒），窗口短暂无响应属正常现象，请勿重复按键...")
            robot.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        try:
            cam.stop()
        except Exception:
            pass    # 相机可能已被上游断开，清理幂等
        print("程序退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
