#!/usr/bin/env python3
"""
Eye-in-Hand 棋盘格手动微动标定工具（reBot Arm B601-RS + Orbbec Gemini 2）。
========================================================================
用途：
    移植自原项目 9_hand_eye_calibration.py。因 B601-RS 底层未提供手动拖动示教，
    本脚本改为【键盘逐关节微动（JOG）采样】，操作同样直观且受软限位与速度限幅保护。
    支持采满样本后现场求解手眼外参 T_tcp_camera，并内置【不动点验证（V 键）】
    实时检验标定精度。

坐标系与单位约定：
    - robot_poses  : 4×4 齐次矩阵列表 T_base_tcp（TCP → base，平移单位：m）
    - camera_poses : 4×4 齐次矩阵列表 T_camera_board（board → camera，平移单位：m）
    - 输出结果      : 4×4 齐次矩阵 T_tcp_camera（camera → TCP，平移单位：m），
                     通过 hand_eye_solver.calibrate_hand_eye 求解（兼容 OpenCV 4.x/5.0）

按键说明：
    数字键 1~6 : 选择微动关节（J1~J6）
    +/-        : 当前关节 ±2° 点动（按住连续运动，限速 8°/s）
    S / 空格   : 等停稳并采集一组最新位姿数据
    C          : 执行 AX=XB 标定并保存至 config/hand_eye_calibration.json
    V          : 标定后进行不动点验证（换不同位姿按 V，检查棋盘原点基座坐标分散度）
    R          : 清空已采样本、验证点与角点原点记忆
    Esc / q    : 退出并安全回零

标定板要求：
    8×8 棋盘格（7×7 内角点），方格边长 20mm，固定于桌面不可移动；
    建议采集 15~20 组，姿态覆盖充分的旋转多样性，相机距标定板 20~50cm。
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

OUT_CAL_FILE = os.path.join(_ROOT, "config", "hand_eye_calibration.json")

# ── 标定板与点动参数 ─────────────────────────────────────────
CHESSBOARD_SIZE = (7, 7)     # 内角点列数×行数（对应 8×8 棋盘格，注意对称棋盘易 180° 翻转）
SQUARE_SIZE = 0.020          # 单个黑白方格边长（单位：m）
CHECK_CORNER_CONSISTENCY = True  # 开启首角点位置跳变检查，防止 7×7 对称棋盘格原点 180° 翻转

MIN_SAMPLES = 10             # 触发求解的最少样本数（建议采集 15~20 组）
JOG_STEP = np.deg2rad(2.0)   # 单次点动关节角增量（单位：rad，对应 2°）

# 全局采样状态缓存
robot_poses = []             # T_base_tcp 列表（单位：m）
camera_poses = []            # T_camera_board 列表（单位：m）
first_corner_position = None # 首次检测到的 0 号内角点像素位置，用于原点一致性守卫


def detect_chessboard(image, camera_matrix, dist_coeffs):
    """检测 7×7 棋盘格内角点并执行 solvePnP，附带对称棋盘原点一致性守卫。

    由于 7×7 内角点呈中心对称，视角翻转时 OpenCV 返回的 corners[0] 可能在棋盘
    对角间 180° 跳变，导致 T_camera_board 坐标系绕 Z 轴翻转 180° 并污染标定。
    本函数通过记录首次 0 号角点像素坐标并限制跳变阈值（50 px）拦截此类异常。

    参数:
        image (np.ndarray): 输入 BGR 彩色图像。
        camera_matrix (np.ndarray): 3×3 相机内参矩阵（单位：px）。
        dist_coeffs (np.ndarray): 畸变系数（Gemini 2 已由适配层强制置零）。

    返回:
        tuple: (ok, rvec, tvec, annotated)
            - ok (bool): 检测、原点一致性及 solvePnP 均成功时为 True
            - rvec (np.ndarray | None): (3, 1) 旋转向量（board → camera，单位：rad）
            - tvec (np.ndarray | None): (3, 1) 平移向量（board → camera，单位：m）
            - annotated (np.ndarray): 叠加角点、坐标轴与距离/告警文字的显示图像
    """
    global first_corner_position
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotated = image.copy()
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
    if not ret:
        cv2.putText(annotated, "No chessboard", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return False, None, None, annotated

    # 亚像素角点细化（11×11 搜索窗，提升 solvePnP 旋转和平移精度）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 原点一致性守卫：防止对称棋盘格角点排序 180° 翻转导致标定板坐标轴反向
    if CHECK_CORNER_CONSISTENCY:
        cur = corners[0].ravel()
        if first_corner_position is None:
            first_corner_position = cur.copy()
        elif np.linalg.norm(cur - first_corner_position) > 50:
            cv2.drawChessboardCorners(annotated, CHESSBOARD_SIZE, corners, ret)
            cv2.putText(annotated, "INCONSISTENT ORIGIN! Press R to reset",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False, None, None, annotated

    # 构造棋盘格局部系 3D 内角点（Z=0 平面，单位：m）
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not ok:
        return False, None, None, annotated

    cv2.drawChessboardCorners(annotated, CHESSBOARD_SIZE, corners, ret)
    cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec,
                      SQUARE_SIZE * 3)
    cv2.putText(annotated, f"Dist: {np.linalg.norm(tvec)*1000:.0f}mm",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return True, rvec, tvec, annotated


def get_robot_tcp_pose(robot):
    """通过正向运动学读取当前机械臂末端位姿矩阵 T_base_tcp。

    参数:
        robot (RebotAdapter): 机械臂适配层实例。

    返回:
        np.ndarray: 4×4 齐次变换矩阵 T_base_tcp（TCP → base，平移单位：m）。
    """
    fk = robot.forward_kinematics()
    T = np.eye(4)
    T[:3, :3] = fk['rotation']
    T[:3, 3] = fk['position']
    return T


def rvec_tvec_to_matrix(rvec, tvec):
    """将 OpenCV solvePnP 输出的旋转向量和平移向量组装为 4×4 齐次变换矩阵。

    参数:
        rvec (np.ndarray): 罗德里格斯旋转向量（单位：rad）。
        tvec (np.ndarray): 平移向量（单位：m）。

    返回:
        np.ndarray: 4×4 齐次变换矩阵 T_camera_board（平移单位：m）。
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def wait_arm_settled(robot, tol_deg=0.5, timeout=3.0):
    """等待非阻塞点动（wait=False）斜坡走完并机械停稳，避免运动模糊或时间错拍。

    参数:
        robot (RebotAdapter): 机械臂适配层实例。
        tol_deg (float): 命令位置与目标位置的最大允许关节误差（单位：deg）。
        timeout (float): 最大等待超时时间（单位：s）。

    返回:
        bool: 在超时内收敛停稳返回 True，否则返回 False。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        err = np.max(np.abs(robot.get_command_pos() - robot._q_target))
        if err < np.deg2rad(tol_deg):
            time.sleep(0.2)     # 命令到位后再额外等待 0.2s 让物理机构消除余振
            return True
        time.sleep(0.05)
    print("⚠ 等待停稳超时，仍继续采样（结果可能不准）")
    return False


def grab_fresh_color(cam, tries=10):
    """尝试从相机管道读取一帧最新的有效彩色图像，避免使用点动前的旧画面。

    参数:
        cam (Gemini2Camera): Gemini 2 相机实例。
        tries (int): 最大重试拉帧次数。

    返回:
        np.ndarray | None: 最新 BGR 彩色图；若连续失败返回 None。
    """
    for _ in range(tries):
        color, _, _ = cam.get_frames()
        if color is not None:
            return color
    return None


def perform_calibration():
    """利用全局采样列表 robot_poses / camera_poses 求解 Eye-in-Hand 外参 T_tcp_camera。

    通过 hand_eye_solver.calibrate_hand_eye 自动选择 OpenCV 原生 Tsai-Lenz（4.x）
    或纯 NumPy Park-Martin（OpenCV 5.0 兼容）。

    返回:
        tuple[bool, np.ndarray | None]: (成功标志, 4×4 T_tcp_camera 矩阵或 None)。
    """
    if len(robot_poses) < MIN_SAMPLES:
        print(f"数据不足：{len(robot_poses)}/{MIN_SAMPLES}")
        return False, None

    R_g2b = [T[:3, :3] for T in robot_poses]
    t_g2b = [T[:3, 3] for T in robot_poses]
    R_t2c = [T[:3, :3] for T in camera_poses]
    t_t2c = [T[:3, 3] for T in camera_poses]

    from hand_eye_solver import calibrate_hand_eye
    T, method = calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c)
    print(f"求解方法: {method}")

    print("\n✓ 标定成功！T_tcp_camera =")
    print(T)
    print(f"平移 (m): {T[:3,3].round(4)}")
    return True, T


verify_points = []           # 不动点验证：棋盘原点在基座系下的坐标列表（单位：m）
last_calib_T = [None]        # 最近一次标定结果 T_tcp_camera（列表包装供闭包更新）


def verify_fixed_point(robot, T_tcp_camera, T_camera_board):
    """不动点验证：将静止棋盘格原点经手眼外参变换至基座系并统计多次观测的分散度。

    变换公式：
        p_base = T_base_tcp · T_tcp_camera · p_cam
    当外参标定准确时，不同机械臂姿态下算出的 p_base 应高度重合（标准差模长 < 2.0 cm）。

    参数:
        robot (RebotAdapter): 机械臂适配层实例。
        T_tcp_camera (np.ndarray): 待验证的 4×4 手眼外参矩阵（camera → TCP）。
        T_camera_board (np.ndarray): 当前帧棋盘格在相机系下的 4×4 位姿矩阵。

    返回:
        np.ndarray: 本次计算得到的棋盘格原点在基座系下的 3D 坐标（单位：m）。
    """
    p_cam = np.append(T_camera_board[:3, 3], 1.0)   # 棋盘原点在相机系下的齐次坐标
    p_base = (get_robot_tcp_pose(robot) @ T_tcp_camera @ p_cam)[:3]
    verify_points.append(p_base)
    pts = np.array(verify_points)
    print(f"[验证] 第 {len(pts)} 次: 棋盘原点基座系坐标 = {p_base.round(4)} m")
    if len(pts) >= 2:
        spread = np.linalg.norm(pts.std(axis=0)) * 100
        print(f"[验证] 当前分散度(标准差): {spread:.2f} cm  "
              f"({'✓ 合格' if spread < 2.0 else '✗ 偏大，建议重标定'})")
    return p_base


def main():
    """主流程：初始化相机与机械臂，运行交互式点动、棋盘格采集、手眼求解与不动点验证循环。"""
    global first_corner_position, robot_poses, camera_poses

    print("=" * 60)
    print("Eye-in-Hand 手眼标定 (B601 RS + Gemini 2)")
    print("=" * 60)

    cam = Gemini2Camera()
    camera_matrix, dist_coeffs = cam.camera_matrix, cam.dist_coeffs

    robot = RebotAdapter()
    # 先运动至屈肘安全准备位，避开全零奇异位并使腕部相机朝向桌面
    print("移动到初始位姿（屈肘安全位，非全零奇异位）...")
    robot.move_to_joints([0.0, 1.2, 1.05, 0.0, 0.0, 0.0], speed_deg_s=15.0)

    state = {"joint": 0, "jog": 0.0, "collect": False,
             "calibrate": False, "verify": False, "quit": False}
    pressed = set()

    def on_press(key):
        """pynput 按下回调：响应关节选择、点动、采样、标定、验证、重置与退出快捷键。"""
        try:
            c = key.char
            if c in '123456':
                state["joint"] = int(c) - 1
                print(f"当前关节: J{int(c)}")
            elif c in ('+', '='):
                pressed.add('+')
            elif c == '-':
                pressed.add('-')
            elif c in ('s', 'S'):
                state["collect"] = True
            elif c in ('c', 'C'):
                state["calibrate"] = True
            elif c in ('v', 'V'):
                state["verify"] = True
            elif c in ('r', 'R'):
                robot_poses.clear(); camera_poses.clear()
                verify_points.clear()
                first_corner_position = None
                print("数据已重置")
            elif c in ('q', 'Q'):
                state["quit"] = True
        except AttributeError:
            if key == pynput_kb.Key.space:
                state["collect"] = True
            elif key == pynput_kb.Key.esc:
                state["quit"] = True

    def on_release(key):
        """pynput 释放回调：清除点动按键状态，防止 Shift+'=' 字符不一致引发按键卡死。"""
        # '+' 通常由 Shift+'=' 输入，press/release 拿到的 key.char 可能不一致，
        # 两种字符都要清掉，否则会出现"按键卡死、关节一直微动"的危险情况
        try:
            pressed.discard(key.char)
            if key.char in ('+', '='):
                pressed.discard('+')
                pressed.discard('=')
        except (AttributeError, KeyError):
            pass

    listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("""
操作说明：
  1~6   选择关节      +/-   微动当前关节（±2°/次，按住连续动）
  S/空格 采集数据      C     执行标定      V 不动点验证（标定后）
  R     重置数据       Esc/q 退出
标定板固定桌面不动，采集 15~20 组，姿态要多样（多转角度）！
""")

    cv2.namedWindow('Hand-Eye Calibration', cv2.WINDOW_NORMAL)

    try:
        while not state["quit"]:
            # 关节微动（从内部命令位置出发，避免在交互热循环里做同步 CAN 读）
            if '+' in pressed or '-' in pressed:
                direction = (1.0 if '+' in pressed else 0.0) - (1.0 if '-' in pressed else 0.0)
                q = robot.get_command_pos()
                q[state["joint"]] += direction * JOG_STEP
                q = np.clip(q, robot.q_min, robot.q_max)  # 软限位安全裁剪
                robot.move_to_joints(q, speed_deg_s=8.0, wait=False)
                time.sleep(0.15)

            color, _, _ = cam.get_frames()
            if color is None:
                continue

            _, _, _, annotated = detect_chessboard(color, camera_matrix, dist_coeffs)
            cv2.putText(annotated, f"Samples: {len(robot_poses)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(annotated, f"Jog joint: J{state['joint']+1}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Hand-Eye Calibration', annotated)
            cv2.waitKey(1)

            # ── 采集分支：等机械臂停稳并取新帧，同步配对 (T_base_tcp, T_camera_board) ──
            if state["collect"]:
                state["collect"] = False
                print(f"\n采集第 {len(robot_poses)+1} 组...")
                wait_arm_settled(robot)
                fresh = grab_fresh_color(cam)   # 停稳后取新帧，别用微动期间的旧帧
                if fresh is None:
                    print("✗ 取帧失败，未保存")
                else:
                    ok, rvec, tvec, _ = detect_chessboard(
                        fresh, camera_matrix, dist_coeffs)
                    if ok:
                        robot_poses.append(get_robot_tcp_pose(robot))
                        camera_poses.append(rvec_tvec_to_matrix(rvec, tvec))
                        print(f"✓ 已保存（标记距离 {np.linalg.norm(tvec)*1000:.0f}mm）")
                    else:
                        print("✗ 检测失败或原点不一致，未保存")

            # ── 标定分支：求解 T_tcp_camera 并落盘 config/hand_eye_calibration.json ──
            if state["calibrate"]:
                state["calibrate"] = False
                ok, T = perform_calibration()
                if ok:
                    last_calib_T[0] = T
                    verify_points.clear()
                    result = {
                        'T_tcp_camera': T.tolist(),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'num_samples': len(robot_poses),
                        'arm': 'reBot Arm B601-RS',
                        'camera': 'Orbbec Gemini 2',
                    }
                    with open(OUT_CAL_FILE, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"已保存 {OUT_CAL_FILE}")
                    print("请按 V 做不动点验证：标定板不动，换 5 个位姿按 V，"
                          "分散度 <2cm 合格")

            # ── 不动点验证分支：检查标定外参在不同姿态下的世界坐标一致性 ──
            if state["verify"]:
                state["verify"] = False
                if last_calib_T[0] is None:
                    print("请先按 C 完成标定，再按 V 验证")
                else:
                    wait_arm_settled(robot)
                    fresh = grab_fresh_color(cam)
                    if fresh is None:
                        print("✗ 取帧失败，无法验证")
                    else:
                        ok, rvec, tvec, _ = detect_chessboard(
                            fresh, camera_matrix, dist_coeffs)
                        if ok:
                            verify_fixed_point(robot, last_calib_T[0],
                                               rvec_tvec_to_matrix(rvec, tvec))
                        else:
                            print("✗ 未检测到棋盘或原点不一致，无法验证")

            time.sleep(0.01)

    finally:
        # 标准安全停机路径：停键盘监听 → 关窗口 → 停相机 → hold → stop（平滑回零后失能）
        listener.stop()
        cv2.destroyAllWindows()
        cam.stop()
        robot.hold()
        robot.stop()
        print("程序退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
