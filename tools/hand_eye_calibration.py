#!/usr/bin/env python3
"""
Eye-in-Hand 手眼标定（B601 RS + Orbbec Gemini 2）

移植自原项目 9_hand_eye_calibration.py。
原项目用「重力补偿 + 手动拖动」采样；B601 RS 的 rebot_control 没有现成
重力补偿接口，本脚本改为【键盘微动（JOG）采样】——效果一样且更安全：

    数字键 1~6 : 选择关节
    +/-        : 当前关节 ±2°（按住连续动）
    w/s        : TCP 前后微调不可用，请用关节微动组合调整视角
    S / 空格   : 采集当前位姿数据
    C          : 执行标定（cv2.calibrateHandEye, TSAI）
    R          : 重置数据
    Esc / q    : 退出

标定板：8x8 棋盘格（内角点 7x7），方格 20mm，固定在桌面不要动。
采样 15~20 组，姿态要有多样性（多旋转、少纯平移）。
相机距标定板建议 20~50cm（Gemini 2 有效区间）。

成功后生成 config/hand_eye_calibration.json，跟随脚本直接读取。
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

# ── 标定板参数 ─────────────────────────────────────────────
CHESSBOARD_SIZE = (7, 7)     # 内角点数（8x8 棋盘格）
SQUARE_SIZE = 0.020          # 方格边长（米）
CHECK_CORNER_CONSISTENCY = True

MIN_SAMPLES = 10             # 至少 10 组，建议 15~20
JOG_STEP = np.deg2rad(2.0)   # 每次微动步长（弧度）

robot_poses = []             # T_base_tcp 列表
camera_poses = []            # T_camera_board 列表
first_corner_position = None


def detect_chessboard(image, camera_matrix, dist_coeffs):
    """检测棋盘格并 solvePnP，含原点一致性检查（与原项目一致）"""
    global first_corner_position
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotated = image.copy()
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
    if not ret:
        cv2.putText(annotated, "No chessboard", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return False, None, None, annotated

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if CHECK_CORNER_CONSISTENCY:
        cur = corners[0].ravel()
        if first_corner_position is None:
            first_corner_position = cur.copy()
        elif np.linalg.norm(cur - first_corner_position) > 50:
            cv2.drawChessboardCorners(annotated, CHESSBOARD_SIZE, corners, ret)
            cv2.putText(annotated, "INCONSISTENT ORIGIN! Press R to reset",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False, None, None, annotated

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
    fk = robot.forward_kinematics()
    T = np.eye(4)
    T[:3, :3] = fk['rotation']
    T[:3, 3] = fk['position']
    return T


def rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def wait_arm_settled(robot, tol_deg=0.5, timeout=3.0):
    """JOG 是 wait=False 下发的，采样前必须等斜坡走完、机械臂停稳"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        err = np.max(np.abs(robot.get_command_pos() - robot._q_target))
        if err < np.deg2rad(tol_deg):
            time.sleep(0.2)     # 再等实际机械响应跟上
            return True
        time.sleep(0.05)
    print("⚠ 等待停稳超时，仍继续采样（结果可能不准）")
    return False


def grab_fresh_color(cam, tries=10):
    """丢弃缓存帧，取一帧最新彩色图"""
    for _ in range(tries):
        color, _, _ = cam.get_frames()
        if color is not None:
            return color
    return None


def perform_calibration():
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


verify_points = []           # 不动点验证：棋盘原点在基座系下的坐标列表
last_calib_T = [None]        # 最近一次标定结果 T_tcp_camera


def verify_fixed_point(robot, T_tcp_camera, T_camera_board):
    """
    不动点验证：标定板固定不动，在不同机械臂位姿下计算棋盘原点在
    基座系下的坐标。标定准确时，各次结果应高度一致（分散度 < 2cm）。
        p_base = T_base_tcp @ T_tcp_camera @ p_cam
    """
    p_cam = np.append(T_camera_board[:3, 3], 1.0)   # 棋盘原点在相机系坐标
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
    global first_corner_position, robot_poses, camera_poses

    print("=" * 60)
    print("Eye-in-Hand 手眼标定 (B601 RS + Gemini 2)")
    print("=" * 60)

    cam = Gemini2Camera()
    camera_matrix, dist_coeffs = cam.camera_matrix, cam.dist_coeffs

    robot = RebotAdapter()
    print("移动到初始位姿（屈肘安全位，非全零奇异位）...")
    robot.move_to_joints([0.0, 1.2, 1.05, 0.0, 0.0, 0.0], speed_deg_s=15.0)

    state = {"joint": 0, "jog": 0.0, "collect": False,
             "calibrate": False, "verify": False, "quit": False}
    pressed = set()

    def on_press(key):
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
            # 关节微动（从内部下发值出发，避免热循环里做同步 CAN 读）
            if '+' in pressed or '-' in pressed:
                direction = (1.0 if '+' in pressed else 0.0) - (1.0 if '-' in pressed else 0.0)
                q = robot.get_command_pos()
                q[state["joint"]] += direction * JOG_STEP
                q = np.clip(q, robot.q_min, robot.q_max)
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
