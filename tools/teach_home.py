#!/usr/bin/env python3
"""
HOME 位姿示教工具（B601 RS）
==============================
键盘微动机械臂到一个顺手的位置，按空格保存为 config/home_pose.json。
palm_follow_fixed_orientation.py 启动时会优先加载它作为 HOME
（初始位姿 + 硬墙自动回退位姿）。

按键：
  1~6   选择关节
  +/-   按住微动当前关节（每步 2°）
  空格  保存当前位置为 HOME 并退出
  Q/Esc 不保存退出

安全：
  - 微动速度 6°/s，全部限制在软限位内
  - 保存时会打印 FK TCP 位置，请确认它在你想要的工作区中心附近
  - 退出走标准安全流程（保持 → 缓慢回零 → 失能）
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

HOME_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "config", "home_pose.json")
JOG_STEP = np.deg2rad(2.0)
JOG_SPEED = 6.0


def main():
    robot = RebotAdapter()
    time.sleep(1.0)

    state = {"joint": 0, "save": False, "quit": False}
    pressed = set()

    def on_press(key):
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
            if key == pynput_kb.Key.space:
                state["save"] = True
            elif key == pynput_kb.Key.esc:
                state["quit"] = True

    def on_release(key):
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
                d = (1.0 if '+' in pressed else 0.0) - (1.0 if '-' in pressed else 0.0)
                q = robot.get_command_pos()
                q[state["joint"]] += d * JOG_STEP
                q = np.clip(q, robot.q_min, robot.q_max)
                robot.move_to_joints(q, speed_deg_s=JOG_SPEED, wait=False)
                time.sleep(0.15)

            if state["save"]:
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
