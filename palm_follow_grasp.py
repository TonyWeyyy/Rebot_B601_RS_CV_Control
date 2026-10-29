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
         夹爪 J7 目标角度（度）→ 随 100Hz 关节指令经 rebot_control
         第 7 通道下发（单一 CAN 主机，无总线冲突）。

运行模式：
    python palm_follow_grasp.py                  # 跟随 + 夹爪（完整模式）
    python palm_follow_grasp.py --no-gripper     # 仅手掌跟随（夹爪保持打开）
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
from datetime import datetime

import numpy as np
import cv2
import yaml
from pynput import keyboard as pynput_kb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini2_camera import Gemini2Camera
from kalman_target import KalmanTarget3D
from servo_timing import (AdaptiveLeadGate, FastStartGate, SlowFollowGate,
                          CameraTimestampCalibrator, AxisProcessNoiseGate,
                          StaticVelocityHold, adaptive_feedforward_scale,
                          camera_twist_from_tcp, compensate_palm_velocity,
                          camera_point_to_world, world_point_to_camera,
                          scaled_alpha, scaled_decay, horizontal_radial_soft_wall,
                          measurement_age, dynamic_lead_seconds)
from servo_diagnostics import (DiagnosticRecorder, ActualFeedbackSampler,
                               feedback_age_ms)
from palm_detection import (PalmDetector, PalmTracker, OneEuroFilter,
                            LKPalmBridge,
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
FF_KF_DEADBAND_OFF = 0.06   # KF 速度死区下沿 (m/s)：速度估计低于此值判定手已停，
                            # 前馈与 lead 外推整体归零——静止时 CV 模型会把 3D 观测
                            # 噪声解释成 ~0.15m/s 的慢速漂移，直接注入前馈就是
                            # 「臂伸展、手停了却来回抖」的主因（log33）；
                            # 伸展态 MIT 杠杆最长、有效阻尼最低，扰动被放大成可见晃动
FF_KF_DEADBAND_ON = 0.11    # 回差上沿：速度超过才恢复运动状态，防阈值边界往复抖动
KF_FF_EXIT_FRAMES = 5       # 跳出静止确认帧数：速度须连续 N 周期越过上沿才
                            # 恢复运动（log38：死区状态在边界逐帧闪烁，每次跳出
                            # 都解锁锁存+lead 打一拍目标=停手振荡；50ms 确认
                            # 窗让噪声单帧毛刺无法解除锁存；进入静止仍即时）
KF_HOLD_TOL = 0.03          # 旧版相机系锁存迁移容差；仅保留配置兼容。
                            # 静止目标已改为世界系 hold_world；旧“偏离容差
                            # 就 0.06m/s 拉回”的兜底已删除，避免噪声漂移。
FF_ADAPTIVE_TAPER_ENABLE = True  # 快速接近时提前减前馈；输出 scale 不大于 v8
FF_TAPER_SPEED_REF = 0.35        # 朝目标接近速度参考 (m/s)
FF_TAPER_MAX_DIST = 0.12         # 自适应停靠距离硬上限 (m)
FF_TAPER_DIST = 0.07          # 前馈近端渐弱距离 (m)：误差小于此值时前馈
                              # 线性衰减到 0——手停下瞬间前馈惯性会把机械臂
                              # 推过目标（到位后回环振荡的主因），巡航段
                              # 不受影响，只在最后 ~7cm 停靠段退出

# 卡尔曼目标滤波（恒速模型）：替代「帧间差分+EMA」的目标速度估计。
# 差分把 2cm 位置噪声 /0.01s 放大成 1.7m/s 假速度，靠 EMA 硬压又引入
# 群延迟；卡尔曼把速度作为内部状态联合估计，仿真（σ=2cm 观测噪声）：
#   速度 RMSE 0.54→0.20 m/s（-63%），静止假速度 0.58→0.15 m/s（-74%）。
# 第一版 lead=0 只取速度喂前馈，位置通道不接入（死区/增益行为不变）。
KF_ENABLE = True              # false = 退回旧差分路径（独立回退开关）
KF_Q = 2.0                    # 过程噪声：手的加速度能力 (m/s^2)^2
                              # 大=跟踪快噪声多；小=平滑好滞后大。1~5 是合理区
KF_R = 4e-3                   # 标量观测噪声方差（旧配置回退）
KF_R_XY = 2.5e-3              # 横向适度信任观测；保守于直接大幅降低 R
KF_R_Z = 4e-3                 # 深度维持 v7.1 噪声强度，不放大 Z 抖动
# 慢拖参数：在世界系掌心点阵上确认，确认后只限速迁移 hold_world。
# 相机自运动不会进入这个通道；静止时前馈/lead 仍保持撤销。
KF_SLOW_FOLLOW_ENABLE = True
KF_SLOW_FOLLOW_DIST = 0.014
KF_SLOW_FOLLOW_RELEASE = 0.008
KF_SLOW_FOLLOW_FRAMES = 3
KF_SLOW_FOLLOW_DIR_COS = 0.80
KF_SLOW_FOLLOW_SPEED = 0.045
CAMERA_HW_TIMESTAMP_ENABLE = False  # 未确认实机 SDK 前保持主机时刻基线
CAMERA_HW_MAX_TRANSPORT_S = 0.20
DIAG_ENABLE = True
DIAG_QUEUE_SIZE = 8192
DIAG_ACTUAL_FEEDBACK_HZ = 0.0  # 同步 CAN 诊断默认关闭；开启时只允许 0<hz<=2
KF_AXIS_ADAPT_ENABLE = True    # 连续同向创新才增强 XY；Z 默认不放大
KF_AXIS_ADAPT_REF_XY = 0.060
KF_AXIS_ADAPT_REF_Z = 0.080
KF_AXIS_ADAPT_MAX_XY = 1.60
KF_AXIS_ADAPT_MAX_Z = 1.00
KF_AXIS_ADAPT_CONFIRM = 2
KF_AXIS_ADAPT_ATTACK_TAU = 0.040
KF_AXIS_ADAPT_RELEASE_TAU = 0.120
KF_AXIS_ADAPT_STATIC_GUARD = True  # 静止观测先用基准 Q，运动确认后再增强
KF_STATIC_Q_SCALE = 1.0            # 静止时额外压低 Q 的倍率（1=关闭）
                                   # follow.yaml 默认 0.75：静止速度估计更平滑，
                                   # 运动确认后立即恢复 1.0 + 分轴增强
STATIC_VEL_HOLD_ENABLE = True      # 只清除静止死区内的低速残余指令状态
STATIC_VEL_HOLD_SPEED = 0.025      # 高于此速度仍走原 EMA + 加速度限制
STATIC_VEL_HOLD_DELAY_S = 0.03     # 连续确认时长；帧数由 CTRL_RATE 推导
STATIC_VEL_HOLD_FRAMES = max(1, round(STATIC_VEL_HOLD_DELAY_S / CTRL_RATE))
KF_RESYNC_FRAMES = 3          # 连续跳变剔除帧数上限：盲期后手真的移了位时，
KF_ADAPT_ENABLE = True        # 自适应过程噪声：创新（观测-预测偏差）大说明
                              # 手在加速，临时放大 Q 让滤波器更信观测、跟手更快；
                              # 平稳段回到基准 Q，保持静止/匀速段的平滑度
KF_ADAPT_REF = 0.05           # 创新参考值 (m)：倍率 = clip(创新/REF, 1, MAX)
KF_ADAPT_MAX = 4.0            # Q 最大放大倍率（≈响应提速一档，再大噪声压不住）
KF_ADAPT_DECAY = 0.90         # 无新观测周期的倍率衰减（毛刺过后 ~0.1s 回到 1×）
                              # 每帧都被 6cm 守卫拒收 → 滤波器永久锁死、前馈
                              # 持续注入过期速度（log26「不跟手」的根因）。
                              # 连续剔除 ≥3 帧认定是真实位移：重同步——
                              # 位置采纳新观测、速度清零（不注入假速度）
KF_LEAD_MS = 0.0              # 卡尔曼外推（第二版）：伺服目标 = 滤波位置 +
                              # 速度×lead，主动补偿「帧等待+识别+滤波」段延迟。
                              # 0 = 关闭（= 第一版行为，代码默认）。
                              # 安全设计：仅手掌可见帧启用（丢失帧仍走 coast）；
                              # 跳变剔除连续帧期间 lead 清零（换向时恒速模型
                              # 不可信）；重同步后速度为 0，外推自然归零。
                              # 推荐 10~15ms 起步；>20ms 换向过冲风险增大
KF_DYNAMIC_LEAD = True        # 基础 lead + 本帧视觉处理耗时；重复帧不累计 lead
KF_LEAD_MAX_MS = 20.0         # 动态 lead 硬上限，换向/剔除/冻结仍单独归零
KF_LEAD_ADAPTIVE = True       # 速度/近目标/方向三重门控；false 仅用于旧逻辑对照
KF_LEAD_NEAR_DIST = 0.022     # 误差小于此值不再额外 lead（与位置死区一致）
KF_LEAD_FULL_DIST = 0.045     # 实机稳定后提前进入全量 lead，减少中速跟随拖后
KF_LEAD_DIR_FULL_COS = 0.50   # 速度与剩余误差方向足够一致才恢复全量 lead
KF_LEAD_ATTACK_TAU = 0.025    # 运动确认后更快恢复；静止/反向撤权规则不变
KF_LEAD_RELEASE_TAU = 0.015   # 撤回提前量时间常数：更快，优先停稳
KF_FAST_START_ENABLE = False  # 实机 A/B 稳定基线；原速度确认通道仍保留
KF_FAST_START_FRAMES = 2      # 只统计新三维观测，不按100Hz重复周期计数
KF_FAST_START_DIST = 0.030    # 累计位移门槛 (m)，须高于静止三维噪声
KF_FAST_START_DIR_COS = 0.80  # 相邻位移方向一致性，换向时重新确认
DEADBAND_HYSTERESIS = 1.5     # 死区迟滞：进入死区后，误差要超过 1.5×死区
                              # 才重新启动——防止误差在死区边界反复穿入穿出
DEPTH_OE_MIN_CUTOFF = 1.0     # 深度通道 One Euro 参数：静止时压低 z3d 抖动
DEPTH_OE_BETA = 0.02          # （手快速靠近/远离时自动放开滤波）
Z3D_MAX_RATE = 0.05           # 深度限速 (m/帧)：参照无效时的最后防线
Z3D_MAX_JUMP = 0.08           # 深度跳变剔除阈值 (m)：单帧 |Δz| 超过此值
                              # 视为误检帧丢弃（手 16ms 内不可能移动 8cm）
Z_FREEZE_MAX = 0.20           # 深度冻结最长时长 (s)：2D 新鲜但深度无效时
                              # （快速挥手的深度空洞，log28 主诉），用最后
                              # 通过四层保护链验证的深度冻结 z 分量、横向
                              # 继续跟踪；不采纳任何新深度（保护链不拆）。
                              # 超时回退原有丢失流程（coast → 保持）。
                              # 0.20s × 手纵向速度 0.5m/s ≈ 最坏 10cm 深度
                              # 保持误差，有界且无尖峰，可安全接受

# 伸展度增益调度：手臂前伸越多，J1 杠杆臂越长、整臂刚度越低，
# 同样增益在伸展位姿会超出稳定裕度（前伸后停靠大振荡的根源）。
# TCP 水平半径 ≤ REACH_SOFT_START 时参数原样，到 REACH_SOFT_END
# 线性衰减到 REACH_MIN_SCALE，超过后保持 REACH_MIN_SCALE。
REACH_SOFT_START = 0.25       # 开始衰减的水平伸展半径 (m)
REACH_SOFT_END = 0.40         # 衰减到底的半径 (m)
REACH_MIN_SCALE = 0.60        # 最深伸展时停靠段指令的保留比例
REACH_DOCK_DIST = 0.20        # 停靠段半径 (m)：误差小于此值才开始衰减，
REACH_ATT_TAU = 0.15          # 伸展衰减乘子 EMA 平滑时间常数 (s)（log36 修复）
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
REVERSAL_RESIDUAL_SCALE = 0.35  # 反转时保留的旧方向速度比例
                                # 0=原逻辑（先完全停住，易顿挫）
                                # 0.35=保留少量惯性并加速反向，减少换向卡顿

# 工作空间虚拟墙（基座系，米）——按实际桌面环境设置
WORKSPACE_MIN = np.array([-0.40, -0.10, 0.03])
WORKSPACE_MAX = np.array([ 0.40,  0.40, 0.60])
WORKSPACE_RADIUS = 0.42       # 70% 臂展：水平径向 hypot(X,Y) 上限；不限制 Z

LOST_HOLD_FRAMES = round(0.30 / CTRL_RATE)   # 连续丢手 0.30s → 保持不动
COAST_FRAMES = round(0.12 / CTRL_RATE)       # 丢失外推窗口 0.12s
COAST_DECAY = 0.90 ** (CTRL_RATE / 0.02)     # 每 20ms 等效衰减 0.90（防爆冲）
COAST_MAX_VEL = 0.35          # 外推速度限幅 (m/s)：防前馈污染带跑
TARGET_STALE_S = 0.12         # 控制侧观测看门狗：视觉线程卡住也不能永远沿用旧目标


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
    p.add_argument("--color-exposure", type=float, default=None, metavar="MS",
                   help="彩色手动曝光（毫秒）：不填=用 follow.yaml 的"
                        " color_exposure_ms（默认 0=自动）。快速挥手时"
                        "MediaPipe 整帧掉检（palm=None）可试 5~8——压运动"
                        "模糊，代价是画面变暗；暗光环境不要用")
    p.add_argument("--no-follow", action="store_true",
                   help="仅捏合夹爪：手臂保持不动（此模式无需手眼标定）")
    p.add_argument("--no-gripper", action="store_true",
                   help="仅手掌跟随：夹爪保持打开（不做捏合控制，减少遮挡）")
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
               "ff_adaptive_taper_enable": FF_ADAPTIVE_TAPER_ENABLE,
               "ff_taper_speed_ref": FF_TAPER_SPEED_REF,
               "ff_taper_max_dist": FF_TAPER_MAX_DIST,
               "kf_enable": KF_ENABLE,
               "kf_q": KF_Q,
               "kf_r": KF_R,
               "kf_r_xy": KF_R_XY,
               "kf_r_z": KF_R_Z,
               "kf_slow_follow_enable": KF_SLOW_FOLLOW_ENABLE,
               "kf_slow_follow_dist": KF_SLOW_FOLLOW_DIST,
               "kf_slow_follow_release": KF_SLOW_FOLLOW_RELEASE,
               "kf_slow_follow_frames": KF_SLOW_FOLLOW_FRAMES,
               "kf_slow_follow_dir_cos": KF_SLOW_FOLLOW_DIR_COS,
               "kf_slow_follow_speed": KF_SLOW_FOLLOW_SPEED,
               "camera_hw_timestamp_enable": CAMERA_HW_TIMESTAMP_ENABLE,
               "camera_hw_max_transport_s": CAMERA_HW_MAX_TRANSPORT_S,
               "diag_enable": DIAG_ENABLE,
               "diag_queue_size": DIAG_QUEUE_SIZE,
               "diag_actual_feedback_hz": DIAG_ACTUAL_FEEDBACK_HZ,
               "kf_axis_adapt_enable": KF_AXIS_ADAPT_ENABLE,
               "kf_axis_adapt_ref_xy": KF_AXIS_ADAPT_REF_XY,
               "kf_axis_adapt_ref_z": KF_AXIS_ADAPT_REF_Z,
               "kf_axis_adapt_max_xy": KF_AXIS_ADAPT_MAX_XY,
               "kf_axis_adapt_max_z": KF_AXIS_ADAPT_MAX_Z,
               "kf_axis_adapt_confirm": KF_AXIS_ADAPT_CONFIRM,
               "kf_axis_adapt_attack_tau": KF_AXIS_ADAPT_ATTACK_TAU,
               "kf_axis_adapt_release_tau": KF_AXIS_ADAPT_RELEASE_TAU,
               "kf_axis_adapt_static_guard": KF_AXIS_ADAPT_STATIC_GUARD,
               "kf_static_q_scale": 1.0,
               "static_velocity_hold_enable": STATIC_VEL_HOLD_ENABLE,
               "static_velocity_hold_speed": STATIC_VEL_HOLD_SPEED,
               "static_velocity_hold_delay_s": STATIC_VEL_HOLD_DELAY_S,
               "kf_lead_ms": KF_LEAD_MS,
               "kf_dynamic_lead": KF_DYNAMIC_LEAD,
               "kf_lead_max_ms": KF_LEAD_MAX_MS,
               "kf_lead_adaptive": KF_LEAD_ADAPTIVE,
               "kf_lead_near_dist": KF_LEAD_NEAR_DIST,
               "kf_lead_full_dist": KF_LEAD_FULL_DIST,
               "kf_lead_dir_full_cos": KF_LEAD_DIR_FULL_COS,
               "kf_lead_attack_tau": KF_LEAD_ATTACK_TAU,
               "kf_lead_release_tau": KF_LEAD_RELEASE_TAU,
               "kf_fast_start_enable": KF_FAST_START_ENABLE,
               "kf_fast_start_frames": KF_FAST_START_FRAMES,
               "kf_fast_start_dist": KF_FAST_START_DIST,
               "kf_fast_start_dir_cos": KF_FAST_START_DIR_COS,
               "kf_ff_deadband_off": FF_KF_DEADBAND_OFF,
               "kf_ff_deadband_on": FF_KF_DEADBAND_ON,
               "kf_hold_tol": KF_HOLD_TOL,
               "kf_ff_exit_frames": KF_FF_EXIT_FRAMES,
               "smooth_alpha": SMOOTH_ALPHA,
               "reversal_fast_response": REVERSAL_FAST_RESPONSE,
               "reversal_residual_scale": REVERSAL_RESIDUAL_SCALE,
               "depth_oe_min_cutoff": DEPTH_OE_MIN_CUTOFF,
               "depth_oe_beta": DEPTH_OE_BETA,
               "k_orientation": K_ORIENTATION,
               "fps": CAMERA_FPS,
               # ---- 以下为进阶参数（原硬编码，全部开放 yaml 可调）----
               "servo_direction": SERVO_DIRECTION,
               "kf_resync_frames": KF_RESYNC_FRAMES,
               "kf_adapt_enable": KF_ADAPT_ENABLE,
               "kf_adapt_ref": KF_ADAPT_REF,
               "kf_adapt_max": KF_ADAPT_MAX,
               "kf_adapt_decay": KF_ADAPT_DECAY,
               "z3d_max_rate": Z3D_MAX_RATE,
               "z_freeze_max": Z_FREEZE_MAX,
               "lost_hold_s": LOST_HOLD_FRAMES * CTRL_RATE,
               "coast_window_s": COAST_FRAMES * CTRL_RATE,
               "coast_decay_per20ms": 0.90,
               "coast_max_vel": COAST_MAX_VEL,
               "target_stale_s": TARGET_STALE_S,
               "reach_att_tau": REACH_ATT_TAU,
               "damping_base": DAMPING_BASE,
               "damping_adaptive": DAMPING_ADAPTIVE,
               "manipulability_threshold": MANIPULABILITY_THRESHOLD,
               "min_manip_speed_scale": MIN_MANIPULABILITY_SPEED_SCALE,
           },
           "workspace": {"min": WORKSPACE_MIN.tolist(),
                         "max": WORKSPACE_MAX.tolist(),
                         "max_radius": WORKSPACE_RADIUS}}
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
    global FF_ADAPTIVE_TAPER_ENABLE, FF_TAPER_SPEED_REF, FF_TAPER_MAX_DIST
    global KF_ENABLE, KF_Q, KF_R, KF_R_XY, KF_R_Z, KF_LEAD_MS
    global KF_SLOW_FOLLOW_ENABLE, KF_SLOW_FOLLOW_DIST, KF_SLOW_FOLLOW_RELEASE
    global KF_SLOW_FOLLOW_FRAMES, KF_SLOW_FOLLOW_DIR_COS, KF_SLOW_FOLLOW_SPEED
    global CAMERA_HW_TIMESTAMP_ENABLE, CAMERA_HW_MAX_TRANSPORT_S
    global DIAG_ENABLE, DIAG_QUEUE_SIZE, DIAG_ACTUAL_FEEDBACK_HZ
    global KF_AXIS_ADAPT_ENABLE, KF_AXIS_ADAPT_REF_XY, KF_AXIS_ADAPT_REF_Z
    global KF_AXIS_ADAPT_MAX_XY, KF_AXIS_ADAPT_MAX_Z, KF_AXIS_ADAPT_CONFIRM
    global KF_AXIS_ADAPT_ATTACK_TAU, KF_AXIS_ADAPT_RELEASE_TAU
    global KF_AXIS_ADAPT_STATIC_GUARD, KF_STATIC_Q_SCALE
    global STATIC_VEL_HOLD_ENABLE, STATIC_VEL_HOLD_SPEED
    global STATIC_VEL_HOLD_DELAY_S, STATIC_VEL_HOLD_FRAMES
    global KF_DYNAMIC_LEAD, KF_LEAD_MAX_MS, KF_LEAD_ADAPTIVE
    global KF_LEAD_NEAR_DIST, KF_LEAD_FULL_DIST, KF_LEAD_DIR_FULL_COS
    global KF_LEAD_ATTACK_TAU, KF_LEAD_RELEASE_TAU
    global KF_FAST_START_ENABLE, KF_FAST_START_FRAMES
    global KF_FAST_START_DIST, KF_FAST_START_DIR_COS
    global FF_KF_DEADBAND_OFF, FF_KF_DEADBAND_ON, KF_HOLD_TOL, KF_FF_EXIT_FRAMES
    global SMOOTH_ALPHA, REVERSAL_FAST_RESPONSE
    global MAX_ACCEL, REVERSAL_ACCEL_BOOST, REVERSAL_RESIDUAL_SCALE
    global DEPTH_OE_MIN_CUTOFF, DEPTH_OE_BETA, Z3D_MAX_JUMP
    global K_ORIENTATION, MAX_VEL_JOINT, MAX_VEL_ANGULAR
    global REACH_SOFT_START, REACH_SOFT_END, REACH_MIN_SCALE, REACH_DOCK_DIST
    global WORKSPACE_MIN, WORKSPACE_MAX, WORKSPACE_RADIUS
    global SERVO_DIRECTION, KF_RESYNC_FRAMES, Z3D_MAX_RATE, Z_FREEZE_MAX
    global KF_ADAPT_ENABLE, KF_ADAPT_REF, KF_ADAPT_MAX, KF_ADAPT_DECAY
    global LOST_HOLD_FRAMES, COAST_FRAMES, COAST_DECAY, COAST_MAX_VEL
    global TARGET_STALE_S
    global REACH_ATT_TAU, DAMPING_BASE, DAMPING_ADAPTIVE
    global MANIPULABILITY_THRESHOLD, MIN_MANIPULABILITY_SPEED_SCALE
    fcfg = cfg["follow"]
    POSITION_DEADBAND = float(fcfg["position_deadband"])
    DEADBAND_HYSTERESIS = float(fcfg["deadband_hysteresis"])
    ORIENTATION_DEADBAND = np.deg2rad(float(fcfg["orientation_deadband_deg"]))
    VEL_FEEDFORWARD = float(fcfg["vel_feedforward"])
    FF_SMOOTH = float(fcfg["ff_smooth"])
    FF_MAX_VEL = float(fcfg["ff_max_vel"])
    FF_MIN_SPEED = float(fcfg["ff_min_speed"])
    FF_TAPER_DIST = float(fcfg["ff_taper_dist"])
    FF_ADAPTIVE_TAPER_ENABLE = bool(fcfg.get(
        "ff_adaptive_taper_enable", FF_ADAPTIVE_TAPER_ENABLE))
    FF_TAPER_SPEED_REF = float(fcfg.get("ff_taper_speed_ref", FF_TAPER_SPEED_REF))
    FF_TAPER_MAX_DIST = float(fcfg.get("ff_taper_max_dist", FF_TAPER_MAX_DIST))
    KF_ENABLE = bool(fcfg.get("kf_enable", KF_ENABLE))
    KF_Q = float(fcfg.get("kf_q", KF_Q))
    KF_R = float(fcfg.get("kf_r", KF_R))
    KF_R_XY = float(fcfg.get("kf_r_xy", KF_R))
    KF_R_Z = float(fcfg.get("kf_r_z", KF_R))
    KF_SLOW_FOLLOW_ENABLE = bool(fcfg.get(
        "kf_slow_follow_enable", KF_SLOW_FOLLOW_ENABLE))
    KF_SLOW_FOLLOW_DIST = float(fcfg.get(
        "kf_slow_follow_dist", KF_SLOW_FOLLOW_DIST))
    KF_SLOW_FOLLOW_RELEASE = float(fcfg.get(
        "kf_slow_follow_release", KF_SLOW_FOLLOW_RELEASE))
    KF_SLOW_FOLLOW_FRAMES = int(fcfg.get(
        "kf_slow_follow_frames", KF_SLOW_FOLLOW_FRAMES))
    KF_SLOW_FOLLOW_DIR_COS = float(fcfg.get(
        "kf_slow_follow_dir_cos", KF_SLOW_FOLLOW_DIR_COS))
    KF_SLOW_FOLLOW_SPEED = float(fcfg.get(
        "kf_slow_follow_speed", KF_SLOW_FOLLOW_SPEED))
    CAMERA_HW_TIMESTAMP_ENABLE = bool(fcfg.get(
        "camera_hw_timestamp_enable", CAMERA_HW_TIMESTAMP_ENABLE))
    CAMERA_HW_MAX_TRANSPORT_S = float(fcfg.get(
        "camera_hw_max_transport_s", CAMERA_HW_MAX_TRANSPORT_S))
    DIAG_ENABLE = bool(fcfg.get("diag_enable", DIAG_ENABLE))
    DIAG_QUEUE_SIZE = int(fcfg.get("diag_queue_size", DIAG_QUEUE_SIZE))
    DIAG_ACTUAL_FEEDBACK_HZ = float(fcfg.get(
        "diag_actual_feedback_hz", DIAG_ACTUAL_FEEDBACK_HZ))
    KF_AXIS_ADAPT_ENABLE = bool(fcfg.get(
        "kf_axis_adapt_enable", KF_AXIS_ADAPT_ENABLE))
    KF_AXIS_ADAPT_REF_XY = float(fcfg.get(
        "kf_axis_adapt_ref_xy", KF_AXIS_ADAPT_REF_XY))
    KF_AXIS_ADAPT_REF_Z = float(fcfg.get(
        "kf_axis_adapt_ref_z", KF_AXIS_ADAPT_REF_Z))
    KF_AXIS_ADAPT_MAX_XY = float(fcfg.get(
        "kf_axis_adapt_max_xy", KF_AXIS_ADAPT_MAX_XY))
    KF_AXIS_ADAPT_MAX_Z = float(fcfg.get(
        "kf_axis_adapt_max_z", KF_AXIS_ADAPT_MAX_Z))
    KF_AXIS_ADAPT_CONFIRM = int(fcfg.get(
        "kf_axis_adapt_confirm", KF_AXIS_ADAPT_CONFIRM))
    KF_AXIS_ADAPT_ATTACK_TAU = float(fcfg.get(
        "kf_axis_adapt_attack_tau", KF_AXIS_ADAPT_ATTACK_TAU))
    KF_AXIS_ADAPT_RELEASE_TAU = float(fcfg.get(
        "kf_axis_adapt_release_tau", KF_AXIS_ADAPT_RELEASE_TAU))
    KF_AXIS_ADAPT_STATIC_GUARD = bool(fcfg.get(
        "kf_axis_adapt_static_guard", KF_AXIS_ADAPT_STATIC_GUARD))
    KF_STATIC_Q_SCALE = float(fcfg.get(
        "kf_static_q_scale", KF_STATIC_Q_SCALE))
    STATIC_VEL_HOLD_ENABLE = bool(fcfg.get(
        "static_velocity_hold_enable", STATIC_VEL_HOLD_ENABLE))
    STATIC_VEL_HOLD_SPEED = float(fcfg.get(
        "static_velocity_hold_speed", STATIC_VEL_HOLD_SPEED))
    STATIC_VEL_HOLD_DELAY_S = float(fcfg.get(
        "static_velocity_hold_delay_s", STATIC_VEL_HOLD_DELAY_S))
    STATIC_VEL_HOLD_FRAMES = max(1, round(
        STATIC_VEL_HOLD_DELAY_S / CTRL_RATE))
    KF_LEAD_MS = float(fcfg.get("kf_lead_ms", KF_LEAD_MS))
    KF_DYNAMIC_LEAD = bool(fcfg.get("kf_dynamic_lead", KF_DYNAMIC_LEAD))
    KF_LEAD_MAX_MS = float(fcfg.get("kf_lead_max_ms", KF_LEAD_MAX_MS))
    KF_LEAD_ADAPTIVE = bool(fcfg.get("kf_lead_adaptive", KF_LEAD_ADAPTIVE))
    KF_LEAD_NEAR_DIST = float(fcfg.get("kf_lead_near_dist", KF_LEAD_NEAR_DIST))
    KF_LEAD_FULL_DIST = float(fcfg.get("kf_lead_full_dist", KF_LEAD_FULL_DIST))
    KF_LEAD_DIR_FULL_COS = float(fcfg.get(
        "kf_lead_dir_full_cos", KF_LEAD_DIR_FULL_COS))
    KF_LEAD_ATTACK_TAU = float(fcfg.get("kf_lead_attack_tau", KF_LEAD_ATTACK_TAU))
    KF_LEAD_RELEASE_TAU = float(fcfg.get("kf_lead_release_tau", KF_LEAD_RELEASE_TAU))
    KF_FAST_START_ENABLE = bool(fcfg.get("kf_fast_start_enable", KF_FAST_START_ENABLE))
    KF_FAST_START_FRAMES = int(fcfg.get("kf_fast_start_frames", KF_FAST_START_FRAMES))
    KF_FAST_START_DIST = float(fcfg.get("kf_fast_start_dist", KF_FAST_START_DIST))
    KF_FAST_START_DIR_COS = float(fcfg.get("kf_fast_start_dir_cos", KF_FAST_START_DIR_COS))
    FF_KF_DEADBAND_OFF = float(fcfg.get("kf_ff_deadband_off", FF_KF_DEADBAND_OFF))
    FF_KF_DEADBAND_ON = float(fcfg.get("kf_ff_deadband_on", FF_KF_DEADBAND_ON))
    KF_HOLD_TOL = float(fcfg.get("kf_hold_tol", KF_HOLD_TOL))
    KF_FF_EXIT_FRAMES = int(fcfg.get("kf_ff_exit_frames", KF_FF_EXIT_FRAMES))
    SMOOTH_ALPHA = float(fcfg["smooth_alpha"])
    REVERSAL_FAST_RESPONSE = bool(fcfg["reversal_fast_response"])
    MAX_ACCEL = float(fcfg.get("max_accel", 2.0))
    REVERSAL_ACCEL_BOOST = float(fcfg.get("reversal_accel_boost", 1.8))
    REVERSAL_RESIDUAL_SCALE = float(fcfg.get(
        "reversal_residual_scale", REVERSAL_RESIDUAL_SCALE))
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
    WORKSPACE_RADIUS = float(ws.get("max_radius", WORKSPACE_RADIUS))
    # ---- 进阶参数（缺失时用代码内置默认值，帧计数常量由 CTRL_RATE 推导）----
    SERVO_DIRECTION = float(fcfg.get("servo_direction", SERVO_DIRECTION))
    KF_RESYNC_FRAMES = int(fcfg.get("kf_resync_frames", KF_RESYNC_FRAMES))
    KF_ADAPT_ENABLE = bool(fcfg.get("kf_adapt_enable", KF_ADAPT_ENABLE))
    KF_ADAPT_REF = float(fcfg.get("kf_adapt_ref", KF_ADAPT_REF))
    KF_ADAPT_MAX = float(fcfg.get("kf_adapt_max", KF_ADAPT_MAX))
    KF_ADAPT_DECAY = float(fcfg.get("kf_adapt_decay", KF_ADAPT_DECAY))
    Z3D_MAX_RATE = float(fcfg.get("z3d_max_rate", Z3D_MAX_RATE))
    Z_FREEZE_MAX = float(fcfg.get("z_freeze_max", Z_FREEZE_MAX))
    LOST_HOLD_FRAMES = round(float(fcfg.get(
        "lost_hold_s", LOST_HOLD_FRAMES * CTRL_RATE)) / CTRL_RATE)
    COAST_FRAMES = round(float(fcfg.get(
        "coast_window_s", COAST_FRAMES * CTRL_RATE)) / CTRL_RATE)
    COAST_DECAY = float(fcfg.get("coast_decay_per20ms", 0.90)) ** (CTRL_RATE / 0.02)
    COAST_MAX_VEL = float(fcfg.get("coast_max_vel", COAST_MAX_VEL))
    TARGET_STALE_S = float(fcfg.get("target_stale_s", TARGET_STALE_S))
    REACH_ATT_TAU = float(fcfg.get("reach_att_tau", REACH_ATT_TAU))
    DAMPING_BASE = float(fcfg.get("damping_base", DAMPING_BASE))
    DAMPING_ADAPTIVE = bool(fcfg.get("damping_adaptive", DAMPING_ADAPTIVE))
    MANIPULABILITY_THRESHOLD = float(fcfg.get(
        "manipulability_threshold", MANIPULABILITY_THRESHOLD))
    MIN_MANIPULABILITY_SPEED_SCALE = float(fcfg.get(
        "min_manip_speed_scale", MIN_MANIPULABILITY_SPEED_SCALE))


def validate_follow_cfg(cfg):
    """在相机/机械臂初始化前拒绝非有限值和危险控制配置。"""
    def walk(value, path="follow"):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isfinite(value):
                raise ValueError(f"配置错误: {path} 必须是有限数值")

    walk(cfg)
    f = cfg["follow"]
    checks = [
        (0.18 <= f["distance"] <= 0.60, "distance 须在 0.18~0.60m"),
        (0.0 < f["gain"] <= 3.5, "gain 须在 (0, 3.5]"),
        (0.0 < f["gain_x"] <= 2.5, "gain_x 须在 (0, 2.5]"),
        (0.0 < f["max_vel"] <= 0.70, "max_vel 须在 (0, 0.70]m/s"),
        (0.0 < f["max_vel_joint"] <= 2.0, "max_vel_joint 须在 (0, 2.0]rad/s"),
        (f["position_deadband"] > 0.0, "position_deadband 必须为正"),
        (f["deadband_hysteresis"] >= 1.0, "deadband_hysteresis 不得小于 1"),
        (f["kf_ff_deadband_on"] > f["kf_ff_deadband_off"] >= 0.0,
         "kf_ff_deadband_on 必须大于 kf_ff_deadband_off"),
        (f["kf_ff_exit_frames"] >= 1, "kf_ff_exit_frames 至少为 1"),
        (f["kf_r"] > 0.0 and f["kf_r_xy"] > 0.0 and f["kf_r_z"] > 0.0,
         "KF 观测噪声方差必须为正"),
        (f["kf_slow_follow_dist"] > f["kf_slow_follow_release"] >= 0.0,
         "慢速跟随确认距离必须大于释放距离"),
        (f["kf_slow_follow_frames"] >= 1 and
         0.0 <= f["kf_slow_follow_dir_cos"] <= 1.0 and
         0.0 < f["kf_slow_follow_speed"] <= 0.06,
         "慢速跟随帧数/方向/速度参数不合法"),
        (0.05 <= f["camera_hw_max_transport_s"] <= 0.20,
         "硬件时间最大传输延迟须在 0.05~0.20s"),
        (f["diag_queue_size"] >= 128, "diag_queue_size 至少为 128"),
        (0.0 <= f["diag_actual_feedback_hz"] <= 2.0,
         "实际关节诊断频率须为 0（关闭）或不超过 2Hz"),
        (f["ff_taper_speed_ref"] > 0.0 and
         f["ff_taper_max_dist"] >= f["ff_taper_dist"] and
         f["ff_taper_max_dist"] <= 0.15,
         "自适应停靠距离/速度参考不合法"),
        (f["kf_axis_adapt_ref_xy"] > 0.0 and f["kf_axis_adapt_ref_z"] > 0.0 and
         1.0 <= f["kf_axis_adapt_max_xy"] <= 2.0 and
         1.0 <= f["kf_axis_adapt_max_z"] <= 1.5 and
         f["kf_axis_adapt_confirm"] >= 1 and
         f["kf_axis_adapt_attack_tau"] > 0.0 and
         f["kf_axis_adapt_release_tau"] > 0.0 and
         0.1 <= f["kf_static_q_scale"] <= 1.0,
         "分轴自适应 Q / 静止 Q 倍率参数不合法"),
        (0.0 < f["static_velocity_hold_speed"] <= 0.08 and
         0.0 <= f["static_velocity_hold_delay_s"] <= 0.10,
         "静止残余速度门槛/确认时长不合法"),
        (f["kf_lead_ms"] >= 0.0 and f["kf_lead_max_ms"] >= 0.0,
         "lead 时间不得为负"),
        (not f["kf_dynamic_lead"] or f["kf_lead_max_ms"] >= f["kf_lead_ms"],
         "动态 lead 上限不得小于基础 lead"),
        (f["kf_lead_full_dist"] > f["kf_lead_near_dist"] >= 0.0,
         "kf_lead_full_dist 必须大于 kf_lead_near_dist"),
        (0.01 <= f["kf_lead_dir_full_cos"] <= 0.99,
         "kf_lead_dir_full_cos 须在 0.01~0.99"),
        (f["kf_lead_attack_tau"] > 0.0 and f["kf_lead_release_tau"] > 0.0,
         "lead 渐变时间常数必须为正"),
        (0.0 < f["target_stale_s"] <= f["lost_hold_s"],
         "target_stale_s 须为正且不得大于 lost_hold_s"),
        (f["servo_direction"] in (-1.0, 1.0), "servo_direction 只能为 ±1"),
        (0.0 <= f["reversal_residual_scale"] <= 1.0,
         "reversal_residual_scale 须在 0~1"),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(f"follow.yaml 配置错误: {message}")
    ws = cfg["workspace"]
    lower = np.asarray(ws["min"], dtype=np.float64)
    upper = np.asarray(ws["max"], dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("follow.yaml 配置错误: workspace min/max 必须为递增的三维边界")
    if not 0.10 <= ws["max_radius"] <= 0.42:
        raise ValueError("follow.yaml 配置错误: workspace.max_radius 须在 0.10~0.42m，"
                         "水平径向工作范围不得超过 70% 臂展安全规则")


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
    validate_follow_cfg(follow_cfg)
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
    if (not np.all(np.isfinite(lambda_gain_xyz))
            or not 0.0 < lambda_gain_xyz[0] <= 2.5
            or not np.all((lambda_gain_xyz[1:] > 0.0)
                          & (lambda_gain_xyz[1:] <= 3.5))):
        raise ValueError("命令行增益超出安全范围")
    if not np.isfinite(max_vel_cart) or not 0.0 < max_vel_cart <= 0.70:
        raise ValueError("max_vel 须在 (0, 0.70]m/s")
    if camera_fps not in (30, 60):
        raise ValueError("Gemini 2 跟随模式仅允许 --fps 30 或 60")
    # 曝光：命令行 > yaml（color_exposure_ms），0 = 自动曝光
    color_exposure_ms = (float(args.color_exposure)
                         if args.color_exposure is not None
                         else float(fcfg.get("color_exposure_ms", 0.0)))

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
    if args.no_gripper and not args.no_arm:
        gripper_mode_text = "OPEN (--no-gripper)"
    elif grasp_cfg["gripper"]["enabled"] and args.no_arm:
        gripper_mode_text = "PREVIEW(仅预览)"
    else:
        gripper_mode_text = "ON" if grasp_on else "OFF"
    print(f"模式: follow={'ON' if follow_enabled else 'OFF'}  "
          f"gripper={gripper_mode_text}  "
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
    cam = Gemini2Camera(fps=camera_fps,
                        color_exposure_ms=color_exposure_ms)
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
    # 检测+跟踪混合：掉检帧光流桥接（hybrid 系列参数见 follow.yaml）
    lk_bridge = LKPalmBridge(
        enable=bool(fcfg.get("hybrid_track_enable", True)),
        max_frames=int(fcfg.get("hybrid_max_frames", 15)),
        box_half=int(fcfg.get("hybrid_box_half", 45)),
        max_step_px=float(fcfg.get("hybrid_max_step_px", 90.0)))
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

    # --no-gripper 是“不做捏合控制”，不是“夹爪保持当前角度”：
    # 夹爪打开可减少对腕部相机的视野遮挡。仍然只走 J7 适配层，
    # 与 6 关节共用单一 CAN 主机，不另开 motorbridge。
    if args.no_gripper and robot is not None:
        robot.set_gripper_deg(gripper_open_deg)
        print(f"[夹爪] --no-gripper：保持打开 {gripper_open_deg:.1f}°"
              "（不做捏合控制）")

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
    target_frozen = [False]           # 当前目标是深度冻结帧（z=desired 的占位值）
    target_seq = [0]                  # 每个有效视觉目标递增，位置相同也能识别新帧
    target_obs_time = [0.0]           # 成像时刻（硬件时间启用时）或主机取帧时刻
    target_arrival_time = [0.0]       # 帧到达主机时刻；处理耗时与传输延迟分离
    target_pub_time = [0.0]           # 检测/深度处理完成、目标发布时刻
    target_processing_ms = [0.0]      # 主机取帧→发布处理耗时（不含传输）
    target_hw_time = [False]          # 本帧是否通过硬件时间校准
    palm_visible = [False]
    g_pressed = [False]
    quit_flag = [False]
    control_status = {"joint_speed": 0.0, "error": "", "active": False}

    if follow_enabled:
        print(f"跟随参数: 距离 {desired_depth}m  "
              f"gain(x/y/z)={lambda_gain_xyz.tolist()}  "
              f"max_vel={max_vel_cart}  前馈={VEL_FEEDFORWARD}")
        if KF_ENABLE:
            print(f"卡尔曼: Q={KF_Q} R={KF_R} Rxy={KF_R_XY} Rz={KF_R_Z} "
                  f"lead={'动态' if KF_DYNAMIC_LEAD else '固定'} "
                  f"base={KF_LEAD_MS}ms max={KF_LEAD_MAX_MS}ms  "
                  f"速度死区={FF_KF_DEADBAND_OFF}/{FF_KF_DEADBAND_ON}m/s")
            print(f"自适应lead: {'ON' if KF_LEAD_ADAPTIVE else 'OFF'} "
                  f"(近端={KF_LEAD_NEAR_DIST:.3f}m 全量={KF_LEAD_FULL_DIST:.3f}m "
                  f"攻/释={KF_LEAD_ATTACK_TAU:.3f}/{KF_LEAD_RELEASE_TAU:.3f}s)")
            print(f"静止快速退出: {'ON' if KF_FAST_START_ENABLE else 'OFF'} "
                  f"({KF_FAST_START_FRAMES}帧 / {KF_FAST_START_DIST:.3f}m)  "
                  f"eye-in-hand 自运动补偿: ON")
            print(f"静止稳定: 基准Q保护={'ON' if KF_AXIS_ADAPT_STATIC_GUARD else 'OFF'} "
                  f"静止Q倍率={KF_STATIC_Q_SCALE:.2f} "
                  f"残余速度保持={'ON' if STATIC_VEL_HOLD_ENABLE else 'OFF'} "
                  f"(≤{STATIC_VEL_HOLD_SPEED:.3f}m/s, "
                  f"{STATIC_VEL_HOLD_DELAY_S:.3f}s)")
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

    # ── 伺服控制线程（100Hz；--no-arm 时不启动）───────────
    stop_ctrl = threading.Event()
    actual_velocity = [np.zeros(6)]
    prev_cmd_vel = [np.zeros(3)]          # 上周期下发的速度指令（加速度限制参照）
    # 上周期实际下发的 TCP twist 对应的相机系 twist（含安装杠杆臂）。
    # KF 观测是相机系掌心位置，机械臂自身运动必须从手掌速度中扣除，
    # 否则静止判定会被相机自运动反复触发。
    prev_camera_twist = [np.zeros(6)]
    prev_command_joints = [None]      # 上周期适配层内部下发角（含 200Hz 斜坡）
    lost_counter = [0]
    prev_target_cam = [None]            # 上一周期掌心目标（前馈用）
    target_vel_f = [np.zeros(3)]        # 平滑后的目标速度估计（相机系）
    in_deadband = [False]               # 死区迟滞状态（True=已收敛到死区内）
    coast_offset = [np.zeros(3)]        # 丢失外推的累计位移（相机系）
    coast_vel = [np.zeros(3)]           # 丢失瞬间的目标速度（外推用）
    coast_err = [1e9]                   # 丢失瞬间的误差（大误差禁止外推）
    near_wall = [True]                  # 上一周期 TCP 是否贴近虚拟墙
    reach_r_now = [0.0]                 # 上一周期水平伸展半径
    reach_att = [1.0]                   # 伸展衰减乘子（EMA 平滑后实际生效值）
    depth_oe = OneEuroFilter(DEPTH_OE_MIN_CUTOFF, DEPTH_OE_BETA)  # 深度滤波
    kf = KalmanTarget3D(q=KF_Q, r=KF_R, r_xy=KF_R_XY, r_z=KF_R_Z)
    axis_q_gate = AxisProcessNoiseGate(
        KF_AXIS_ADAPT_REF_XY, KF_AXIS_ADAPT_REF_Z,
        KF_AXIS_ADAPT_MAX_XY, KF_AXIS_ADAPT_MAX_Z, KF_AXIS_ADAPT_CONFIRM,
        KF_AXIS_ADAPT_ATTACK_TAU, KF_AXIS_ADAPT_RELEASE_TAU)
    static_vel_hold = StaticVelocityHold(
        STATIC_VEL_HOLD_SPEED, STATIC_VEL_HOLD_FRAMES)
    static_hold_now = [False]
    # 状态保存在最近观测时刻，不随重复控制周期污染。
    kf_time = [0.0]                       # 最近 KF 观测的 monotonic 时刻
    kf_last_seq = [-1]                    # 用帧序号识别新观测，不再比较位置差
    kf_reject_count = [0]                 # 连续跳变剔除计数（≥KF_RESYNC_FRAMES 重同步）
    kf_rejected_now = [False]             # 本周期观测被跳变守卫拒绝（lead 清零信号）
    ff_moving = [False]                   # KF 速度死区状态：True=判定手在运动
    ff_exit_count = [0]                   # 跳出静止的连续确认计数
    last_joint_speed = [0.0]               # 上一控制周期下发的关节速度模长
    settled_frames = [0]                   # 机械臂在死区内连续静止的周期数
    hold_latch = [None]                   # 静止状态标记/诊断快照
    hold_world = [None]                   # 静止时锁定的世界系掌心目标（消噪）
    lead_gate = AdaptiveLeadGate(
        FF_KF_DEADBAND_OFF, FF_KF_DEADBAND_ON,
        KF_LEAD_NEAR_DIST, KF_LEAD_FULL_DIST, KF_LEAD_DIR_FULL_COS,
        KF_LEAD_ATTACK_TAU, KF_LEAD_RELEASE_TAU)
    lead_weight_now = [0.0]
    lead_target_weight = [0.0]
    ff_taper_now = [FF_TAPER_DIST]
    ff_approach_speed = [0.0]
    ff_scale_now = [0.0]
    fast_start = FastStartGate(
        KF_FAST_START_ENABLE, KF_FAST_START_FRAMES,
        KF_FAST_START_DIST, KF_FAST_START_DIR_COS)
    # 慢拖在世界系世界目标上判断，避免把相机自运动当成慢拖。
    slow_follow = SlowFollowGate(
        KF_SLOW_FOLLOW_DIST, KF_SLOW_FOLLOW_RELEASE,
        KF_SLOW_FOLLOW_FRAMES, KF_SLOW_FOLLOW_DIR_COS)
    slow_active = [False]
    last_valid_depth = [0.0]    # 上一帧有效深度（跳变剔除的参照）
    last_valid_depth_time = [0.0]   # 最后一次深度通过验证的墙钟时间（深度冻结用）
    z3d_reject_count = [0]      # 连续跳变剔除帧数（防锁死：≥15 帧放行）
    reacq_buf = [[]]            # 参照无效时的恢复校验缓冲（3 帧一致性）
    ctrl_thread = None
    timestamp_calibrator = CameraTimestampCalibrator(
        max_transport_s=CAMERA_HW_MAX_TRANSPORT_S)
    diag_recorder = None
    if DIAG_ENABLE and robot is not None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, "servo_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
        try:
            diag_recorder = DiagnosticRecorder(log_path, DIAG_QUEUE_SIZE).start()
            print(f"[诊断] 高频日志: {log_path}")
        except OSError as exc:
            print(f"[诊断] 日志初始化失败，不影响控制: {exc}")
            diag_recorder = None
    feedback_sampler = None
    if diag_recorder is not None and DIAG_ACTUAL_FEEDBACK_HZ > 0.0:
        try:
            feedback_sampler = ActualFeedbackSampler(
                robot, hz=DIAG_ACTUAL_FEEDBACK_HZ).start()
            print(f"[诊断] 实际关节采样 {DIAG_ACTUAL_FEEDBACK_HZ:.1f}Hz "
                  "（同步 CAN，仅诊断，不参与控制）")
        except Exception as exc:
            print(f"[诊断] 实际关节采样启动失败，已禁用: {exc}")
            feedback_sampler = None

    def servo_control_loop():
        next_deadline = time.monotonic()
        last_loop_time = None
        _zero3 = np.zeros(3)
        while not stop_ctrl.is_set():
            next_deadline += CTRL_RATE
            t0 = time.monotonic()
            dt = (CTRL_RATE if last_loop_time is None else
                  float(np.clip(t0 - last_loop_time, 1e-3, 0.06)))
            last_loop_time = t0
            q_cmd_log = np.zeros(6)
            static_hold_now[0] = False

            with lock:
                following = g_pressed[0]
                target_cam = (None if current_target_cam[0] is None
                              else np.array(current_target_cam[0], dtype=np.float64))
                frozen_meas = target_frozen[0]
                visible = palm_visible[0]
                obs_seq = target_seq[0]
                obs_time = target_obs_time[0]
                arrival_time = target_arrival_time[0]
                pub_time = target_pub_time[0]
                hw_time_used = target_hw_time[0]

            # 控制侧独立看门狗：即使视觉线程卡在检测/后处理，也不能让
            # visible 永久停留在 True。保留最后目标供原有有界 coast 使用。
            if (visible and obs_time > 0.0
                    and t0 - obs_time > TARGET_STALE_S):
                visible = False

            if not visible:
                lost_counter[0] += 1
            else:
                lost_counter[0] = 0

            # ── 时间戳卡尔曼：状态保存在最近观测时刻 ─────────────────
            # 新观测按真实观测间隔 update；重复控制周期不再反复修改后验状态。
            # 使用时再按 now-kf_time 外推到现在，避免异步观测的时间错位。
            new_kf_meas = bool(visible and target_cam is not None
                               and not frozen_meas and obs_seq != kf_last_seq[0])
            lead_s = 0.0
            if KF_ENABLE:
                kf_rejected_now[0] = False
                if not visible or frozen_meas:
                    slow_active[0] = False
                if KF_AXIS_ADAPT_ENABLE:
                    if not visible or frozen_meas:
                        axis_q_gate.reset()
                    else:
                        axis_q_gate.relax(dt)
                    kf.q_scale = axis_q_gate.scales.copy()
                    if not ff_moving[0] and KF_STATIC_Q_SCALE < 1.0:
                        kf.q_scale = np.full(3, KF_STATIC_Q_SCALE)
                else:
                    kf.q_scale = max(1.0, kf.q_scale * scaled_decay(
                        KF_ADAPT_DECAY, dt, CTRL_RATE))
                if new_kf_meas:
                    meas_time = obs_time if obs_time > 0.0 else t0
                    if kf.initialized:
                        # 异常/乱序时间戳不允许滤波器时间倒退。
                        meas_time = max(meas_time, kf_time[0] + 1e-3)
                    if not kf.initialized:
                        kf.update(target_cam, 0.0)
                        kf_time[0] = meas_time
                        kf_last_seq[0] = obs_seq
                        kf_reject_count[0] = 0
                    else:
                        dt_meas = float(np.clip(meas_time - kf_time[0], 1e-3, 0.50))
                        predicted = kf.position() + kf.velocity() * dt_meas
                        innov = float(np.linalg.norm(target_cam - predicted))
                        gate = (0.06 + 2.0 * float(np.linalg.norm(kf.velocity()))
                                * dt_meas)
                        if innov > gate:
                            if KF_AXIS_ADAPT_ENABLE:
                                axis_q_gate.reset()
                                kf.q_scale = axis_q_gate.scales.copy()
                            kf_reject_count[0] += 1
                            kf_rejected_now[0] = True
                            if kf_reject_count[0] >= KF_RESYNC_FRAMES:
                                kf.reset()
                                kf.update(target_cam, 0.0)
                                kf_time[0] = meas_time
                                kf_reject_count[0] = 0
                                hold_latch[0] = None
                                hold_world[0] = None
                                fast_start.reset()
                                slow_follow.reset()
                                slow_active[0] = False
                                with lock:
                                    control_status["error"] = "KF重同步"
                            else:
                                kf.predict(dt_meas)
                                kf_time[0] = meas_time
                        else:
                            if KF_AXIS_ADAPT_ENABLE:
                                # v9 会先放大 Q、随后才复位，静止噪声可能因此被
                                # 放大并误解锁。静止观测直接用基准 Q；运动确认后
                                # 原分轴加速增强不变，false 可独立复现 v9 顺序。
                                q_valid = (ff_moving[0]
                                           or not KF_AXIS_ADAPT_STATIC_GUARD)
                                kf.q_scale = axis_q_gate.update(
                                    target_cam - predicted, dt, valid=q_valid)
                            elif KF_ADAPT_ENABLE:
                                kf.q_scale = float(np.clip(
                                    innov / KF_ADAPT_REF, 1.0, KF_ADAPT_MAX))
                            kf.update(target_cam, dt_meas)
                            kf_time[0] = meas_time
                            kf_reject_count[0] = 0
                        kf_last_seq[0] = obs_seq

                _kv_raw = kf.velocity() if kf.initialized else np.zeros(3)
                if frozen_meas:
                    # 深度无效时横向仍可跟随，但旧纵向速度不得继续注入。
                    _kv_raw[2] = 0.0
                # 相机装在手腕上：KF 看到的是“手掌相对相机”的速度，
                # 机械臂自身的运动也会让静止手掌在画面里移动。静止判定
                # 必须使用扣除相机自运动后的手掌速度，否则停手后会形成
                # “锁存→机械臂动→KF 看到假速度→解锁”的极限环。
                # 运动段 FF/lead 保持原相机系速度（实机验证过的稳定手感）；
                # 补偿速度只用于静止判定，避免自运动反馈进前馈/lead。
                _kv = compensate_palm_velocity(
                    _kv_raw, prev_camera_twist[0], kf.position()) \
                    if kf.initialized else _kv_raw.copy()
                if frozen_meas:
                    _kv[2] = 0.0
                _kvn = float(np.linalg.norm(_kv))
                _was_moving = ff_moving[0]
                if ff_moving[0]:
                    if _kvn < FF_KF_DEADBAND_OFF:
                        ff_moving[0] = False
                        ff_exit_count[0] = 0
                elif _kvn > FF_KF_DEADBAND_ON:
                    ff_exit_count[0] += 1
                    if ff_exit_count[0] >= KF_FF_EXIT_FRAMES:
                        ff_moving[0] = True
                        ff_exit_count[0] = 0
                else:
                    ff_exit_count[0] = 0

                # 机械臂已在死区内连续多周期零速，但 KF 残余速度仍把
                # ff_moving 拉在 True 时，强制进入静止。否则残余目标漂移
                # 会每隔几周期推一次关节，表现为上下“点动”。
                if (ff_moving[0] and in_deadband[0]
                        and last_joint_speed[0] < 0.03):
                    settled_frames[0] += 1
                    if settled_frames[0] >= 8:
                        ff_moving[0] = False
                        ff_exit_count[0] = 0
                        settled_frames[0] = 0
                else:
                    settled_frames[0] = 0

                # 静止确认后立即回到 v8 基准 Q，不让加速段倍率残留到停靠。
                if KF_AXIS_ADAPT_ENABLE and not ff_moving[0]:
                    axis_q_gate.reset()
                    kf.q_scale = (np.full(3, KF_STATIC_Q_SCALE)
                                  if KF_STATIC_Q_SCALE < 1.0
                                  else axis_q_gate.scales.copy())

                # 额外快速通道：连续新三维观测同向位移，解除静止锁存。
                # 原速度确认通道保持不变，静止进入条件也不变。
                # 必须同时存在补偿后的真实手掌速度，避免纯相机自运动
                # 通过原始像素位移误触发快速通道。
                if (not ff_moving[0] and hold_latch[0] is not None
                        and new_kf_meas and not kf_rejected_now[0]
                        and _kvn > FF_KF_DEADBAND_OFF
                        and fast_start.update(target_cam)):
                    ff_moving[0] = True
                    hold_latch[0] = None
                    ff_exit_count[0] = 0
                    fast_start.reset()
                    with lock:
                        control_status["fast_start"] = True

                if ff_moving[0] != _was_moving:
                    with lock:
                        control_status["ffm"] = ff_moving[0]

                if ff_moving[0]:
                    hold_latch[0] = None
                    hold_world[0] = None
                    fast_start.reset()
                    slow_follow.reset()
                    slow_active[0] = False
                elif kf.initialized:
                    if hold_latch[0] is None:
                        # hold_latch 是静止状态标记；世界系目标 hold_world
                        # 在控制线程拿到 FK 位姿后建立（见控制段）。
                        hold_latch[0] = kf.position().copy()
                        hold_world[0] = None
                        fast_start.reset(target_cam if target_cam is not None
                                         else kf.position())
                        slow_follow.reset()
                        slow_active[0] = False
                    elif kf_rejected_now[0]:
                        slow_follow.reset()
                        slow_active[0] = False
                # 静止判定用补偿速度；正常运动段的前馈仍沿用原相机系
                # KF 速度，保持已经实机验证过的巡航/高速手感不变。
                # 静止时 ff_moving=False 会把前馈整体归零，不会把自运动
                # 注入停手瞬间。
                target_vel_f[0] = (np.clip(_kv_raw, -FF_MAX_VEL, FF_MAX_VEL)
                                   if ff_moving[0] else np.zeros(3))

            # ── 丢失外推：全部按实际 dt 积分/衰减 ─────────────────────
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
                coast_offset[0] += coast_vel[0] * dt
                coast_vel[0] *= scaled_decay(COAST_DECAY, dt, CTRL_RATE)
            if target_cam is not None and lost_counter[0] > 0:
                target_cam = target_cam + coast_offset[0]

            # KF 位置先按观测年龄外推到现在；额外 lead 单独加权。
            # 年龄补偿始终保留；静止时只撤销额外 lead 和前馈。
            lead_raw_s = 0.0
            lead_error_now = 0.0
            kf_speed_now = (float(np.linalg.norm(_kv_raw))
                            if KF_ENABLE and kf.initialized else 0.0)
            lead_weight_now[0] = 0.0
            lead_target_weight[0] = 0.0
            if (not KF_ENABLE or not visible or frozen_meas
                    or kf_rejected_now[0]):
                lead_gate.reset()
            if (KF_ENABLE and visible and kf.initialized
                    and not kf_rejected_now[0]):
                if not frozen_meas:
                    # 硬件成像时刻包含传输段；lead 的处理项只计算主机处理，
                    # 传输延迟已经包含在 age_s 中，不能重复补偿。
                    processing_s = max(0.0, pub_time - arrival_time)
                    age_s = measurement_age(t0, kf_time[0], max_age=0.25)
                    # 年龄外推必须用原始相机系速度：静止手掌在机械臂运动时
                    # 的视位置变化，正是“现在”应有的观测位置。
                    now_target = kf.position() + _kv_raw * age_s
                    lead_error_now = float(np.linalg.norm(
                        now_target - np.array([0.0, 0.0, desired_depth])))
                    if hold_latch[0] is not None:
                        # 静止时只撤销前馈和额外 lead。这里先给实时滤波
                        # 目标作为退路；控制段会用世界系 hold_world 投影
                        # 覆盖它，避免位置环追相机系噪声。
                        lead_gate.reset()
                        lead_s = 0.0
                        target_cam = now_target
                    else:
                        lead_raw_s = (dynamic_lead_seconds(
                            KF_LEAD_MS, processing_s, KF_LEAD_MAX_MS)
                            if KF_DYNAMIC_LEAD and KF_LEAD_MS > 0.0
                            else KF_LEAD_MS / 1000.0)
                        if KF_LEAD_ADAPTIVE and lead_raw_s > 0.0:
                            lead_weight_now[0], lead_target_weight[0] = lead_gate.update(
                                _kv_raw, now_target - np.array(
                                    [0.0, 0.0, desired_depth]),
                                dt, ff_moving[0], valid=True)
                        else:
                            lead_gate.reset()
                            # false 只用于复现旧控制逻辑；正常实机保持 true。
                            lead_weight_now[0] = (1.0 if lead_raw_s > 0.0 else 0.0)
                            lead_target_weight[0] = lead_weight_now[0]
                        lead_s = lead_raw_s * lead_weight_now[0]
                        target_cam = now_target + _kv_raw * lead_s

            if following and target_cam is not None and lost_counter[0] <= LOST_HOLD_FRAMES:
                try:
                    current_joints = robot.get_command_pos()
                    q_cmd_log = current_joints.copy()
                    tcp_pos, tcp_rot, J_robot = compute_tcp_fk_and_jacobian(
                        robot, current_joints)
                    with lock:
                        control_status["tcp"] = tcp_pos.round(3).tolist()

                    tcp_radius = float(np.hypot(tcp_pos[0], tcp_pos[1]))
                    if (np.any(tcp_pos < WORKSPACE_MIN - 0.05)
                            or np.any(tcp_pos > WORKSPACE_MAX + 0.05)
                            or tcp_radius > WORKSPACE_RADIUS + 0.05):
                        robot.hold()
                        print(f"\n[安全] TCP 越过硬墙 {tcp_pos.round(2).tolist()}"
                              f"，自动回 HOME...")
                        robot.move_to_joints(HOME_JOINT_POS, speed_deg_s=12.0,
                                             wait=True, timeout=30.0)
                        actual_velocity[0] = np.zeros(6)
                        prev_cmd_vel[0][:] = 0.0
                        kf.reset()
                        kf_last_seq[0] = -1
                        kf_time[0] = 0.0
                        kf_reject_count[0] = 0
                        ff_moving[0] = False
                        ff_exit_count[0] = 0
                        hold_latch[0] = None
                        hold_world[0] = None
                        lead_gate.reset()
                        fast_start.reset()
                        slow_active[0] = False
                        axis_q_gate.reset()
                        static_vel_hold.reset()
                        static_hold_now[0] = False
                        last_joint_speed[0] = 0.0
                        settled_frames[0] = 0
                        reach_att[0] = 1.0
                        target_vel_f[0] = np.zeros(3)
                        prev_camera_twist[0][:] = 0.0
                        prev_command_joints[0] = None
                        with lock:
                            control_status["error"] = "已自动回 HOME"
                        next_deadline = time.monotonic()
                        continue

                    # ── 静止世界系目标 ─────────────────────────────
                    # 静止时把掌心锁在世界系：位置环不再追相机系噪声或
                    # 随机械臂自运动漂移的画面点。慢拖在 world_obs 上确认，
                    # 只会以受限速度迁移 hold_world，跟手性不靠相机系噪声。
                    if (T_tcp_camera is not None and hold_latch[0] is not None):
                        if hold_world[0] is None:
                            world_hold = camera_point_to_world(
                                target_cam, tcp_pos, tcp_rot, T_tcp_camera)
                            if np.linalg.norm(world_hold) > 1e-6:
                                hold_world[0] = world_hold
                                slow_follow.reset(hold_world[0])
                                slow_active[0] = False
                        if hold_world[0] is not None:
                            if (new_kf_meas and not frozen_meas
                                    and not kf_rejected_now[0]):
                                world_obs = camera_point_to_world(
                                    target_cam, tcp_pos, tcp_rot,
                                    T_tcp_camera)
                                if np.linalg.norm(world_obs) > 1e-6:
                                    slow_active[0] = slow_follow.update(
                                        world_obs,
                                        enabled=(KF_SLOW_FOLLOW_ENABLE
                                                 and not in_deadband[0]))
                                    delta = world_obs - hold_world[0]
                                    distance = float(np.linalg.norm(delta))
                                    if (slow_active[0]
                                            and distance > KF_SLOW_FOLLOW_RELEASE):
                                        step = min(
                                            distance - KF_SLOW_FOLLOW_RELEASE,
                                            KF_SLOW_FOLLOW_SPEED * dt)
                                        hold_world[0] = (hold_world[0]
                                                         + delta / distance * step)
                                    # 不再使用“偏离 hold_tol 就 0.06m/s 拉回”
                                    # 的旧兜底：它会放大量测噪声并造成静止
                                    # 上下漂移。慢拖只由方向/帧确认的
                                    # SlowFollowGate 迁移。
                            held_cam = world_point_to_camera(
                                hold_world[0], tcp_pos, tcp_rot, T_tcp_camera)
                            if np.linalg.norm(held_cam) > 1e-6:
                                target_cam = held_cam

                    target_desired = np.array([0.0, 0.0, desired_depth])
                    pos_err_cam = target_cam - target_desired
                    pos_err_norm = float(np.linalg.norm(pos_err_cam))
                    if in_deadband[0]:
                        if pos_err_norm > POSITION_DEADBAND * DEADBAND_HYSTERESIS:
                            in_deadband[0] = False
                    elif pos_err_norm < POSITION_DEADBAND:
                        in_deadband[0] = True
                    v_cam = np.zeros(6)
                    if not in_deadband[0]:
                        v_cam[:3] = SERVO_DIRECTION * lambda_gain_xyz * pos_err_cam

                    ff_taper_now[0] = FF_TAPER_DIST
                    ff_approach_speed[0] = 0.0
                    ff_scale_now[0] = 0.0
                    if VEL_FEEDFORWARD > 0.0:
                        if not KF_ENABLE and prev_target_cam[0] is not None:
                            if np.linalg.norm(target_cam - prev_target_cam[0]) < 0.06:
                                tv = (target_cam - prev_target_cam[0]) / dt
                                tv = np.clip(tv, -FF_MAX_VEL, FF_MAX_VEL)
                                target_vel_f[0] = (FF_SMOOTH * tv
                                                   + (1 - FF_SMOOTH) * target_vel_f[0])
                        prev_target_cam[0] = target_cam.copy()
                        if in_deadband[0] or \
                                np.linalg.norm(target_vel_f[0]) < FF_MIN_SPEED:
                            target_vel_f[0] = np.zeros(3)
                        else:
                            ff_scale, ff_taper_now[0], ff_approach_speed[0] = \
                                adaptive_feedforward_scale(
                                    pos_err_cam, target_vel_f[0], FF_TAPER_DIST,
                                    FF_ADAPTIVE_TAPER_ENABLE, FF_TAPER_SPEED_REF,
                                    FF_TAPER_MAX_DIST)
                            ff_scale_now[0] = ff_scale
                            v_cam[:3] += (VEL_FEEDFORWARD * ff_scale
                                          * target_vel_f[0])

                    # 静止时保持当前机械臂位姿：不再追世界系掌心点做
                    # “对中/补距离”运动。手掌停住后目标立即冻结，避免
                    # 残余目标漂移或慢慢对中造成肉眼可见的小幅晃动。
                    if hold_latch[0] is not None:
                        v_cam[:3] = 0.0

                    ori_err = compute_orientation_error(tcp_rot, desired_rotation)
                    v_cam[:3] = max_vel_cart * np.tanh(v_cam[:3] / max_vel_cart)

                    reach_r = float(np.hypot(tcp_pos[0], tcp_pos[1]))
                    reach_r_now[0] = reach_r
                    reach_scale = 1.0
                    att_target = 1.0
                    if reach_r > REACH_SOFT_START:
                        dock = float(np.clip(
                            1.0 - pos_err_norm / REACH_DOCK_DIST, 0.0, 1.0))
                        reach_scale = float(np.clip(
                            1.0 - (1.0 - REACH_MIN_SCALE)
                            * (reach_r - REACH_SOFT_START)
                            / max(REACH_SOFT_END - REACH_SOFT_START, 1e-6),
                            REACH_MIN_SCALE, 1.0))
                        att_target = 1.0 - (1.0 - reach_scale) * dock
                    reach_att[0] += ((att_target - reach_att[0])
                                     * min(1.0, dt / REACH_ATT_TAU))
                    v_cam[:3] *= reach_att[0]

                    angular_world = (K_ORIENTATION * ori_err
                                     if np.linalg.norm(ori_err) > ORIENTATION_DEADBAND
                                     else np.zeros(3))
                    angular_world = np.clip(angular_world, -MAX_VEL_ANGULAR, MAX_VEL_ANGULAR)

                    target_reversing = (np.dot(v_cam[:3], prev_cmd_vel[0]) < 0)
                    if (REVERSAL_FAST_RESPONSE
                            and np.dot(v_cam[:3], actual_velocity[0][:3]) < 0):
                        # 不再把旧速度直接清零（会先完全停住再反向，产生
                        # 换向顿挫）；保留少量旧方向惯性，让 EMA 平滑穿零。
                        actual_velocity[0][:3] *= REVERSAL_RESIDUAL_SCALE
                    actual_velocity[0] += ((v_cam - actual_velocity[0])
                                           * scaled_alpha(SMOOTH_ALPHA, dt, CTRL_RATE))

                    dv_cmd = actual_velocity[0][:3] - prev_cmd_vel[0]
                    accel_cap = MAX_ACCEL
                    if target_reversing or np.dot(
                            actual_velocity[0][:3], prev_cmd_vel[0]) < 0:
                        accel_cap *= REVERSAL_ACCEL_BOOST
                    max_dv = accel_cap * dt
                    dv_norm = float(np.linalg.norm(dv_cmd))
                    if dv_norm > max_dv:
                        actual_velocity[0][:3] = (prev_cmd_vel[0]
                                                  + dv_cmd * (max_dv / dv_norm))
                    # 死区内关节命令原本就会归零；这里进一步清除 EMA/加速度
                    # 限制器中的低速残态，避免下一次死区边界波动带出旧速度。
                    # 高速停靠不清零，仍由原有加速度限制器平顺衰减。
                    actual_velocity[0], static_hold_now[0] = static_vel_hold.update(
                        actual_velocity[0],
                        stationary=(not ff_moving[0]
                                    and hold_latch[0] is not None),
                        in_deadband=in_deadband[0],
                        enabled=STATIC_VEL_HOLD_ENABLE)
                    if static_hold_now[0]:
                        prev_cmd_vel[0][:] = 0.0
                    else:
                        prev_cmd_vel[0] = actual_velocity[0][:3].copy()

                    v_tcp = camera_velocity_to_tcp(actual_velocity[0], T_tcp_camera)
                    v_world = np.zeros(6)
                    v_world[0:3] = tcp_rot @ v_tcp[0:3]
                    v_world[3:6] = angular_world

                    wall_touch = False
                    for ax in range(3):
                        if tcp_pos[ax] <= WORKSPACE_MIN[ax] and v_world[ax] < 0:
                            v_world[ax] = 0.0
                            wall_touch = True
                        elif tcp_pos[ax] >= WORKSPACE_MAX[ax] and v_world[ax] > 0:
                            v_world[ax] = 0.0
                            wall_touch = True
                    # 70% 臂展水平径向软墙：只削 X/Y 继续向外的分量，保留
                    # 切向、向内及 Z 运动；避免三维球墙在高姿态过度压缩空间。
                    v_world[:3], radial_touch = horizontal_radial_soft_wall(
                        tcp_pos, v_world[:3], WORKSPACE_RADIUS)
                    wall_touch = wall_touch or radial_touch
                    near_wall[0] = bool(
                        np.any(tcp_pos < WORKSPACE_MIN + 0.05)
                        or np.any(tcp_pos > WORKSPACE_MAX - 0.05)
                        or tcp_radius > WORKSPACE_RADIUS - 0.05)

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
                    last_joint_speed[0] = float(np.linalg.norm(joint_vel))

                    # 用适配层内部下发角（含 200Hz 斜坡）的差分估计本周
                    # 实际相机 twist，下一周期从 KF 观测速度中扣除。
                    # 这比直接用期望 joint_vel 更接近真实 deceleration：
                    # 指令归零后斜坡仍在走，不能把相机当成已静止。
                    if (T_tcp_camera is not None
                            and prev_command_joints[0] is not None):
                        qdot_cmd = ((current_joints - prev_command_joints[0])
                                    / max(float(dt), 1e-3))
                        # 控制周期抖动时防止差分被放大；正常斜坡本就在
                        # MAX_VEL_JOINT 内，钳位只影响异常采样。
                        qdot_cmd = np.clip(qdot_cmd,
                                           -MAX_VEL_JOINT, MAX_VEL_JOINT)
                        if np.all(np.isfinite(qdot_cmd)):
                            prev_camera_twist[0] = camera_twist_from_tcp(
                                J_robot @ qdot_cmd, tcp_rot, T_tcp_camera)
                        else:
                            prev_camera_twist[0][:] = 0.0
                    else:
                        prev_camera_twist[0][:] = 0.0
                    prev_command_joints[0] = current_joints.copy()

                    robot.joint_vel_cmd(joint_vel)
                    with lock:
                        control_status["joint_speed"] = float(np.linalg.norm(joint_vel))
                        _status = "贴墙滑动中" if wall_touch else ""
                        if reach_att[0] < 0.95:
                            _status += ("" if not _status else " ") + \
                                f"伸展衰减x{reach_att[0]:.2f}"
                        control_status["error"] = _status
                        control_status["active"] = True

                except Exception as exc:
                    robot.hold()
                    prev_camera_twist[0][:] = 0.0
                    prev_command_joints[0] = None
                    hold_world[0] = None
                    last_joint_speed[0] = 0.0
                    settled_frames[0] = 0
                    with lock:
                        control_status["error"] = str(exc)
                        control_status["active"] = False
            else:
                actual_velocity[0] = np.zeros(6)
                prev_cmd_vel[0][:] = 0.0
                prev_target_cam[0] = None
                prev_camera_twist[0][:] = 0.0
                prev_command_joints[0] = None
                kf.reset()
                kf_last_seq[0] = -1
                kf_time[0] = 0.0
                kf_reject_count[0] = 0
                ff_moving[0] = False
                ff_exit_count[0] = 0
                hold_latch[0] = None
                hold_world[0] = None
                lead_gate.reset()
                fast_start.reset()
                slow_active[0] = False
                axis_q_gate.reset()
                static_vel_hold.reset()
                static_hold_now[0] = False
                last_joint_speed[0] = 0.0
                settled_frames[0] = 0
                reach_att[0] = 1.0
                target_vel_f[0] = np.zeros(3)
                in_deadband[0] = False
                coast_offset[0][:] = 0.0
                coast_vel[0][:] = 0.0
                coast_err[0] = 1e9
                robot.hold()
                with lock:
                    control_status["joint_speed"] = 0.0
                    control_status["active"] = False
                    if following and lost_counter[0] > LOST_HOLD_FRAMES:
                        control_status["error"] = "手掌丢失，保持位置"

            elapsed = time.monotonic() - t0
            with lock:
                control_status["ctrl_dt_ms"] = dt * 1000.0
                control_status["ctrl_work_ms"] = elapsed * 1000.0
                control_status["ctrl_overruns"] = (
                    control_status.get("ctrl_overruns", 0)
                    + (1 if elapsed > CTRL_RATE else 0))
                control_status["target_age_ms"] = (
                    measurement_age(t0, obs_time) * 1000.0 if obs_time > 0 else 0.0)
                control_status["lead_ms"] = lead_s * 1000.0
                control_status["lead_raw_ms"] = lead_raw_s * 1000.0
                control_status["lead_weight"] = lead_weight_now[0]
                control_status["lead_target_weight"] = lead_target_weight[0]
                control_status["kf_speed"] = kf_speed_now
                control_status["lead_error"] = lead_error_now
                control_status["ffm"] = ff_moving[0]
                control_status["hold_latched"] = hold_latch[0] is not None
                control_status["target_seq"] = obs_seq
                control_status["target_frozen"] = frozen_meas
                control_status["slow_active"] = slow_active[0]
            if diag_recorder is not None:
                target_values = ([""] * 3 if target_cam is None
                                 else np.asarray(target_cam).tolist())
                kv_log = _kv_raw if KF_ENABLE else _zero3
                with lock:
                    status_snapshot = dict(control_status)
                    processing_ms = target_processing_ms[0]
                feedback = (feedback_sampler.snapshot()
                            if feedback_sampler is not None else None)
                feedback_age = feedback_age_ms(
                    feedback, time.monotonic(), DIAG_ACTUAL_FEEDBACK_HZ)
                feedback_fresh = feedback_age is not None
                q_scale_log = np.asarray(kf.q_scale, dtype=np.float64)
                if q_scale_log.ndim == 0:
                    q_scale_log = np.full(3, float(q_scale_log))
                row = {
                    "host_time": f"{t0:.6f}", "dt_s": f"{dt:.6f}",
                    "seq": obs_seq, "new_obs": int(new_kf_meas),
                    "following": int(following), "visible": int(visible),
                    "frozen": int(frozen_meas), "kf_valid": int(kf.initialized),
                    "ff_moving": int(ff_moving[0]),
                    "hold_latch": int(hold_latch[0] is not None),
                    "slow_active": int(slow_active[0]),
                    "static_hold": int(static_hold_now[0]),
                    "lead_weight": lead_weight_now[0],
                    "lead_raw_s": lead_raw_s, "lead_s": lead_s,
                    "obs_age_ms": status_snapshot.get("target_age_ms", 0.0),
                    "processing_ms": processing_ms,
                    "ctrl_work_ms": elapsed * 1000.0,
                    "overrun": int(elapsed > CTRL_RATE),
                    "kf_q_scale_x": q_scale_log[0],
                    "kf_q_scale_y": q_scale_log[1],
                    "kf_q_scale_z": q_scale_log[2],
                    "ff_taper_now": ff_taper_now[0],
                    "ff_approach_speed": ff_approach_speed[0],
                    "ff_scale": ff_scale_now[0],
                    "fb_valid": int(feedback_fresh),
                    "fb_time": (f"{feedback['time']:.6f}"
                                if feedback_fresh else ""),
                    "fb_age_ms": feedback_age if feedback_fresh else "",
                    "fb_read_ms": feedback["read_ms"] if feedback_fresh else "",
                    "tracking_err_max": (float(np.max(np.abs(
                        feedback["actual"] - feedback["command"])))
                        if feedback_fresh else ""),
                    "error_norm": (float(np.linalg.norm(
                        target_cam - np.array([0.0, 0.0, desired_depth])))
                        if target_cam is not None else ""),
                    "kv_x": kv_log[0], "kv_y": kv_log[1], "kv_z": kv_log[2],
                    "joint_speed": status_snapshot.get("joint_speed", 0.0),
                    "hw_time_used": int(hw_time_used),
                    "error": status_snapshot.get("error", ""),
                }
                for axis in range(3):
                    row[f"target_{'xyz'[axis]}"] = target_values[axis]
                for joint in range(6):
                    row[f"q_cmd_{joint + 1}"] = q_cmd_log[joint]
                    row[f"q_act_{joint + 1}"] = (
                        feedback["actual"][joint] if feedback_fresh else "")
                    row[f"q_fb_cmd_{joint + 1}"] = (
                        feedback["command"][joint] if feedback_fresh else "")
                diag_recorder.log(row)
            delay = next_deadline - time.monotonic()
            if delay > 0:
                stop_ctrl.wait(delay)
            else:
                # 超时后从当前时刻重新排期，不为补周期而连续空转。
                next_deadline = time.monotonic()

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
    camera_timestamp_warned = [False]

    try:
        last_frame_time = time.time()
        while not quit_flag[0]:
            _t0 = time.perf_counter()
            color_image, depth_raw, depth_scale = cam.get_frames()
            frame_arrival_time = time.monotonic()
            raw_timestamp_s = None
            if CAMERA_HW_TIMESTAMP_ENABLE and color_image is not None:
                try:
                    raw_timestamp_s, _ = cam.get_last_frame_timing()
                except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                    if not camera_timestamp_warned[0]:
                        print(f"[相机时间] 接口不兼容，回退主机时刻: {exc}")
                        camera_timestamp_warned[0] = True
            frame_obs_time, frame_hw_time = timestamp_calibrator.update(
                raw_timestamp_s, frame_arrival_time)
            _t1 = time.perf_counter()
            if color_image is None:
                # 取流失败也属于观测丢失：让伺服进入有界 coast→保持流程，
                # 不能把最后一次 visible=True 永久留在共享状态中。
                with lock:
                    palm_visible[0] = False
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
                        cam = Gemini2Camera(
                            fps=camera_fps,
                            color_exposure_ms=color_exposure_ms)
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
            # 光流桥接（检测+跟踪混合）：MediaPipe 掉检帧用 LK 点阵
            # 顶帧继续喂 PalmTracker，短暂模糊掉检不再触发丢失保持；
            # 桥接帧无关键点 → landmarks=None，捏合门控自然冻结
            bu, bv, bsrc = lk_bridge.update(
                color_image, (u, v) if u is not None else None)
            if bsrc == 'lk':
                u, v = bu, bv
                landmarks = None
            track = tracker.update((u, v) if u is not None else None)
            if track is not None:
                track['bridged'] = (bsrc == 'lk')
            # 耗时统计（EMA，毫秒）：取帧（相机等待+USB传输）/ 识别
            ms_grab[0] = 0.9 * ms_grab[0] + 0.1 * (_t1 - _t0) * 1000
            ms_det[0] = 0.9 * ms_det[0] + 0.1 * (_t2 - _t1) * 1000

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
                # 要求连续 5 帧读数互相一致（max-min ≤ 跳变阈值）才接受，
                # 取中位数建参照。log37 教训：手后方背景读数天然一致，
                # 3 帧会被稳定错误的背景深度骗过 → 5 帧 + 跟随距离上限
                # 双重设防。只发生在长丢失恢复时，代价 ~80ms；
                # 短暂模糊掉检参照保留，完全不受影响。
                if depth_m > 0 and last_valid_depth[0] == 0:
                    if depth_m > desired_depth + 0.25:
                        depth_m = 0.0       # 明显超出跟随场景的深度=背景
                        reacq_buf[0].clear()
                    else:
                        reacq_buf[0].append(depth_m)
                        if len(reacq_buf[0]) > 5:
                            reacq_buf[0].pop(0)
                        if len(reacq_buf[0]) == 5 and \
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
                    last_valid_depth_time[0] = time.time()
                    track['x3d'], track['y3d'], track['z3d'] = pixel_to_3d(
                        track['u'], track['v'], depth_m, fx, fy, cx, cy)
                    track['z3d_frozen'] = False
                else:
                    # 深度冻结（log28）：2D 新鲜但深度无效时（快速挥手的
                    # 深度空洞/尖峰剔除帧），用最后通过四层验证的深度
                    # 冻结 z：x,y 由 2D 像素+冻结深度反投影（横向照常
                    # 跟踪），z 填 desired_depth（伺服侧 z 误差=0——
                    # 手臂保持当前深度，不纵向追）。不采纳任何新深度值，
                    # 四层保护链一条不拆；超过 Z_FREEZE_MAX 秒回退原
                    # 丢失流程（z3d=0 → coast → 保持）。
                    if (last_valid_depth[0] > 0 and
                            time.time() - last_valid_depth_time[0] <= Z_FREEZE_MAX):
                        track['x3d'], track['y3d'], _ = pixel_to_3d(
                            track['u'], track['v'],
                            last_valid_depth[0], fx, fy, cx, cy)
                        track['z3d'] = desired_depth
                        track['z3d_frozen'] = True
                    else:
                        track['z3d'] = 0.0
                        track['z3d_frozen'] = False
            elif track is None or track['lost'] >= 30:
                # 长时间（≥30帧≈0.5s）真正丢失才重置深度参照；
                # 短暂丢失（运动模糊等 1~29 帧）保留参照，让跳变剔除
                # 在重新捕获瞬间继续生效——否则恢复帧的假深度会
                # 绕过门控直接打进伺服，造成 joint_speed 突刺
                last_valid_depth[0] = 0.0
                z3d_reject_count[0] = 0
                reacq_buf[0].clear()

            target_publish_time = time.monotonic()
            with lock:
                if track is not None and track['lost'] == 0 and track['z3d'] > 0:
                    current_target_cam[0] = [track['x3d'], track['y3d'], track['z3d']]
                    target_frozen[0] = bool(track.get('z3d_frozen', False))
                    target_seq[0] += 1
                    target_obs_time[0] = frame_obs_time
                    target_arrival_time[0] = frame_arrival_time
                    target_pub_time[0] = target_publish_time
                    target_hw_time[0] = frame_hw_time
                    target_processing_ms[0] = ((target_publish_time - frame_arrival_time)
                                               * 1000.0)
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
                           f"lost={track['lost']} z3d={track['z3d']:.3f}"
                           f"{'❄冻结' if (track.get('z3d_frozen') and track['lost'] == 0) else ''}"
                           f"{'◦桥接' if track.get('bridged') else ''}")
                pinch_info = (f"pinch={pinch_px_now[0]:.0f}px→"
                              f"{last_sent_gripper_deg:.1f}°"
                              if pinch_px_now[0] is not None else "pinch=None")
                with lock:
                    _fol, _vis = g_pressed[0], palm_visible[0]
                    _spd = control_status["joint_speed"]
                    _err = control_status["error"]
                    _ffm = control_status.get("ffm", True)
                    _age = control_status.get("target_age_ms", 0.0)
                    _lead = control_status.get("lead_ms", 0.0)
                    _lead_raw = control_status.get("lead_raw_ms", 0.0)
                    _lead_w = control_status.get("lead_weight", 0.0)
                    _kf_speed = control_status.get("kf_speed", 0.0)
                    _lead_err = control_status.get("lead_error", 0.0)
                    _latched = control_status.get("hold_latched", False)
                    _slow = control_status.get("slow_active", False)
                    _cdt = control_status.get("ctrl_dt_ms", 0.0)
                    _cwork = control_status.get("ctrl_work_ms", 0.0)
                    _over = control_status.get("ctrl_overruns", 0)
                    _seq = control_status.get("target_seq", 0)
                if _fol and not _ffm:
                    _err = (f"{_err}·静止" if _err else "·静止")
                _tcp = control_status.get("tcp", None)
                print(f"[DBG] fps={fps:4.1f}  palm=({det})  visible={_vis} "
                      f"following={_fol}  joint_speed={_spd:.3f}  "
                      f"{pinch_info}  tcp={_tcp}  err={_err}  "
                      f"延迟ms(取帧/识别)={ms_grab[0]:.0f}/{ms_det[0]:.0f} "
                      f"观测={_age:.0f}ms lead={_lead:.1f}/{_lead_raw:.1f}ms "
                      f"w={_lead_w:.2f} kv={_kf_speed:.3f}m/s "
                      f"e={_lead_err:.3f}m latch={int(_latched)} slow={int(_slow)} "
                      f"控制={_cdt:.1f}/{_cwork:.1f}ms 超时={_over} seq={_seq} "
                      f"处理={target_processing_ms[0]:.1f}ms "
                      f"hw={int(target_hw_time[0])} "
                      f"丢帧={getattr(cam, 'dropped_frames', 0)}")

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
        # 同步 CAN 实际反馈只用于诊断；必须先于安全回零停止，避免争用总线。
        if feedback_sampler is not None:
            feedback_sampler.close()
        if diag_recorder is not None:
            diag_recorder.close()
            print(f"[诊断] 写入 {diag_recorder.written} 行，"
                  f"队列丢弃 {diag_recorder.dropped} 行"
                  + (f"，错误={diag_recorder.error}" if diag_recorder.error else ""))
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
