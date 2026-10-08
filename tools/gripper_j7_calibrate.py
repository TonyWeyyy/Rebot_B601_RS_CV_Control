#!/usr/bin/env python3
"""
夹爪（J7）开合标定工具 —— palm_follow_grasp 合并项目专用
=================================================================
通过 RebotAdapter 的 J7 通道点动夹爪，实测「完全张开 / 完全闭合」
两个角度（度），按 s 写回 config/grasp.yaml。

为什么不用旧 gripper 项目的 gripper_calibrate.py：
    旧工具自己开一个 motorbridge 连接直连电机 7（MIT 模式），
    与机械臂控制器并存时两个 CAN 主机会争抢电机 7。
    本工具复用 rebot_control 的第 7 通道，单一 CAN 主机，无冲突；
    且角度单位（度）与 palm_follow_grasp.py 运行时完全一致，
    不存在弧度/度换算误差。

按键：
    + / = / ]   夹爪目标角 +5°（按住连发）
    - / _ / [   夹爪目标角 -5°
    o           记录当前角度为「完全张开」
    c           记录当前角度为「完全闭合」
    r           读取电机实际 J7 角度（同步 CAN 读，用于核对）
    s           保存到 config/grasp.yaml 并退出
    q / Esc     不保存退出

安全：
    - 6 个手臂关节全程保持不动，只有 J7 动
    - 目标角始终被限制在适配层硬限幅内（-100° ~ +400°）
    - 标定从当前实际角度起步，不会上电猛动
    - 退出走标准安全流程（保持 → 缓慢回零 → 失能）
    - 点动方向不确定时先小步试，方向反了不影响标定（只看实测角度）
"""

import os
import sys
import time

import numpy as np
import yaml
from pynput import keyboard as pynput_kb

# 项目根目录（tools/ 的上一级）加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebot_adapter import RebotAdapter

GRASP_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "config", "grasp.yaml")
JOG_STEP_DEG = 5.0


def main():
    print("=" * 56)
    print("  夹爪（J7）开合标定")
    print("=" * 56)
    print("初始化机械臂（手臂关节将保持不动，只有夹爪会动）...")
    robot = RebotAdapter()
    time.sleep(1.0)

    target = robot.get_gripper_deg()     # 从当前实际角度起步
    recorded = {"open_deg": None, "closed_deg": None}
    quit_flag = [False]

    print(f"\n当前夹爪角度: {target:.1f}°")
    print("按键: +/- 点动 ±5° | o 记张开 | c 记闭合 | r 读实际角度 | "
          "s 保存退出 | q 退出\n")

    def jog(delta):
        nonlocal target
        target += delta
        robot.set_gripper_deg(target)
        robot.hold()                     # 立即下发（含新夹爪目标角）
        target = robot.get_gripper_deg()  # 回读限幅后的实际目标值
        print(f"\r目标角: {target:+8.1f}°   ", end="", flush=True)

    def on_press(key):
        try:
            ch = key.char
        except AttributeError:
            if key == pynput_kb.Key.esc:
                quit_flag[0] = True
            return
        if ch in ('+', '=', ']'):
            jog(+JOG_STEP_DEG)
        elif ch in ('-', '_', '['):
            jog(-JOG_STEP_DEG)
        elif ch == 'o':
            recorded["open_deg"] = round(target, 1)
            print(f"\n[记录] 完全张开 = {recorded['open_deg']}°")
        elif ch == 'c':
            recorded["closed_deg"] = round(target, 1)
            print(f"\n[记录] 完全闭合 = {recorded['closed_deg']}°")
        elif ch == 'r':
            try:
                actual = float(robot.io.arm.read_joint_angles()[6])
                print(f"\n[读取] 电机实际 J7 = {actual:.1f}°（目标 {target:.1f}°）")
            except Exception as e:
                print(f"\n[读取] 失败: {e}")
        elif ch == 's':
            quit_flag[0] = "save"
        elif ch == 'q':
            quit_flag[0] = True

    listener = pynput_kb.Listener(on_press=on_press)
    listener.start()
    try:
        while not quit_flag[0]:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

    if quit_flag[0] == "save":
        if recorded["open_deg"] is None or recorded["closed_deg"] is None:
            print("\n⚠️ 张开/闭合尚未都记录（按 o 和 c），未保存。")
        elif recorded["open_deg"] == recorded["closed_deg"]:
            print("\n⚠️ 张开与闭合角度相同，明显有误，未保存。")
        else:
            cfg = {}
            if os.path.exists(GRASP_CFG):
                with open(GRASP_CFG, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            cfg.setdefault("gripper", {})
            cfg["gripper"]["open_deg"] = recorded["open_deg"]
            cfg["gripper"]["closed_deg"] = recorded["closed_deg"]
            with open(GRASP_CFG, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            print(f"\n✅ 已保存到 {GRASP_CFG}:")
            print(f"   open_deg   = {recorded['open_deg']}°")
            print(f"   closed_deg = {recorded['closed_deg']}°")
            print("   （注意：yaml 回写不保留原注释，字段值已更新）")

    print("\n安全停机：保持 → 缓慢回零 → 失能 ...")
    robot.hold()
    time.sleep(0.3)
    robot.stop()
    print("退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
