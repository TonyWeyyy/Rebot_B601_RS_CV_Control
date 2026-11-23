#!/usr/bin/env python3
"""
HOME 位姿示教工具（reBot Arm B601-RS）。
=========================================
用途：
    通过键盘逐关节微动（JOG）机械臂至合适的初始工作位姿，按空格键将其
    保存至 config/home_pose.json。主控脚本（palm_follow_grasp.py /
    palm_follow_fixed_orientation.py）启动时会优先加载该位姿作为 HOME
    （上电初始位姿 + 越出硬边界后的自动安全回退位姿）。

用法：
    python tools/teach_home.py

输出产物（config/home_pose.json）：
    - home_joint_pos (list[float]): 6 轴关节角（单位：rad）
    - tcp_position (list[float]):   基座系下 FK TCP 位置 [X, Y, Z]（单位：m）
    - timestamp (str):              保存时间戳

按键：
    1~6   选择当前微动关节（J1~J6）
    +/-   按住微动当前关节（单步 2°，6°/s）
    空格  保存当前实测关节角为 HOME 并退出
    Q/Esc 不保存直接退出

安全红线：
    - 微动速度限幅 6°/s，且全部裁剪在 RebotAdapter 关节软限位 [q_min, q_max] 内
    - 保存时打印基座系 FK TCP 位置，请确认其位于工作空间中心附近
    - 无论正常退出、取消或异常，均在 finally 中执行标准安全停机（hold → stop 平滑回零 → 失能），
      严禁在非零/屈肘悬空位直接失能电机
"""

import os
import sys
import json
import time

import numpy as np
import cv2

from pynput import keyboard as pynput_kb

# 项目根目录（tools/ 的上一级）加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebot_adapter import RebotAdapter

# HOME 位姿保存路径与安全微动参数
HOME_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "config", "home_pose.json")
JOG_STEP = np.deg2rad(2.0)   # 单次点动增量（单位：rad，对应 2°）
JOG_SPEED = 6.0              # 点动斜坡速率上限（单位：deg/s）


def main():
    """主流程：启动机械臂与全局键盘监听，循环响应关节微动并在确认后保存 HOME 位姿。"""
    robot = RebotAdapter()
    time.sleep(1.0)  # 等待控制器与 CAN 通信稳定

    state = {"joint": 0, "save": False, "quit": False}
    pressed = set()

    def on_press(key):
        """pynput 按键按下回调：更新选中关节、持续微动集合或保存/退出标志。"""
        try:
            c = key.char
            if c in '123456':
                state["joint"] = int(c) - 1
                print(f"当前关节: J{c}")
            elif c in ('+', '='):
                pressed.add('+')
            elif c == '-':
                pressed.add('-')
            elif c in ('q', 'Q'):
                state["quit"] = True
        except AttributeError:
            # 特殊功能键（Space / Esc）无 char 属性，通过 Key 枚举匹配
            if key == pynput_kb.Key.space:
                state["save"] = True
            elif key == pynput_kb.Key.esc:
                state["quit"] = True

    def on_release(key):
        """pynput 按键释放回调：清除按住状态，兼容 Shift+'='→'+' 防止松键卡死。"""
        try:
            pressed.discard(key.char)
            if key.char in ('+', '='):
                pressed.discard('+'); pressed.discard('=')
        except (AttributeError, KeyError):
            pass

    listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("""
[HOME 示教] 键盘微动到顺手位置：
  1~6 选关节 | +/- 微动 | 空格 保存并退出 | Q/Esc 不保存退出
  建议：TCP 在基座正前方 25~35cm、高度 30~40cm、相机视野开阔的位置
""")

    saved = False
    try:
        while not state["quit"]:
            if '+' in pressed or '-' in pressed:
                # 仅在按住 +/- 时下发增量目标；从命令位置 get_command_pos() 累加，
                # 避免在热循环里触发阻塞式 CAN 同步读取
                d = (1.0 if '+' in pressed else 0.0) - (1.0 if '-' in pressed else 0.0)
                q = robot.get_command_pos()
                q[state["joint"]] += d * JOG_STEP
                q = np.clip(q, robot.q_min, robot.q_max)  # 强制裁剪到关节软限位
                robot.move_to_joints(q, speed_deg_s=JOG_SPEED, wait=False)
                time.sleep(0.15)  # 限流连发节拍，避免步进过快

            if state["save"]:
                # 保存前读取当前实际关节角（单位：rad）与基座系 FK 位姿（单位：m）
                q_home = robot.get_current_pos()
                fk = robot.forward_kinematics()
                tcp = fk['position']
                print(f"\n当前关节角 (rad): {q_home.round(4).tolist()}")
                print(f"FK TCP 位置: {tcp.round(3)} m")
                with open(HOME_FILE, 'w') as f:
                    json.dump({'home_joint_pos': q_home.tolist(),
                               'tcp_position': tcp.tolist(),
                               'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
                              f, indent=2)
                print(f"✓ 已保存到 {HOME_FILE}")
                saved = True
                break
            time.sleep(0.02)
    finally:
        # 标准安全停机路径：停止监听 → 保持当前位 → 平滑回零并失能
        listener.stop()
        try:
            robot.hold()
        except Exception:
            pass
        robot.stop()
        print("程序退出（已安全回零）"
              + ("，HOME 已生效" if saved else "，未保存"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消（机械臂将由 rebot_control 平滑回零）")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
