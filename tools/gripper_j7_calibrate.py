#!/usr/bin/env python3
"""
夹爪（J7）开合行程标定工具 —— palm_follow_grasp 合并项目专用。
=================================================================
用途：
    通过 RebotAdapter 的第 7 通道点动夹爪电机（J7），实测并记录
    「完全张开（open_deg）」与「完全闭合（closed_deg）」两个极限角度
    （单位：deg），按 s 键写回 config/grasp.yaml。

为什么严禁复用旧 gripper 项目的 gripper_calibrate.py：
    旧工具会独立开启第二个 motorbridge CAN 连接直连电机 7，与机械臂控制器
    并存时两个 CAN 主机会在 can0 上争抢电机 7（违反单一 CAN 主机安全红线）。
    本工具完全经由 RebotAdapter 复用 rebot_control 的第 7 通道：
    - 单一 CAN 主机，无总线冲突；
    - 角度单位（度，deg）与 palm_follow_grasp.py 运行时完全一致，无 rad/deg 换算误差。

用法：
    python tools/gripper_j7_calibrate.py

按键：
    + / = / ]   夹爪目标角 +5°（按住可连发）
    - / _ / [   夹爪目标角 -5°
    o           记录当前目标角为「完全张开（open_deg）」
    c           记录当前目标角为「完全闭合（closed_deg）」
    r           读取电机实际 J7 角度（按需触发的同步 CAN 读，用于核对丢步/机械限位）
    s           保存至 config/grasp.yaml 并退出
    q / Esc     不保存退出

安全红线：
    - J1~J6 手臂关节全程锁位不动，仅通过 J7 通道驱动夹爪
    - 目标角始终受 RebotAdapter 内部硬限幅保护（-100° ~ +400°）
    - 标定从当前实际 J7 角度起步，避免上电阶跃猛动
    - 退出走标准安全停机（hold → 稳态等待 → stop 平滑回零 → 失能）
    - 点动方向不确定时先小步试探；方向相反不影响标定（仅记录两个实测端点）
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

# 夹爪标定配置文件路径与单次点动步长（单位：deg）
GRASP_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "config", "grasp.yaml")
JOG_STEP_DEG = 5.0


def main():
    """主流程：初始化 RebotAdapter，响应键盘点动 J7，校验并写回 grasp.yaml 后安全停机。"""
    print("=" * 56)
    print("  夹爪（J7）开合标定")
    print("=" * 56)
    print("初始化机械臂（手臂关节将保持不动，只有夹爪会动）...")
    robot = RebotAdapter()
    time.sleep(1.0)  # 等待控制器就绪

    target = robot.get_gripper_deg()     # 从当前实际角度起步（单位：deg），防止上电跳变
    recorded = {"open_deg": None, "closed_deg": None}
    quit_flag = [False]

    print(f"\n当前夹爪角度: {target:.1f}°")
    print("按键: +/- 点动 ±5° | o 记张开 | c 记闭合 | r 读实际角度 | "
          "s 保存退出 | q 退出\n")

    def jog(delta):
        """按增量 delta（单位：deg）更新 J7 目标角并立即经统一 7 通道下发。

        参数:
            delta (float): 目标角增量（单位：deg，正负表示开合方向）。
        """
        nonlocal target
        target += delta
        robot.set_gripper_deg(target)
        robot.hold()                     # 保持 J1~J6 不动并立即打包下发新 J7 目标角
        target = robot.get_gripper_deg()  # 回读适配层硬限幅后的实际目标值，防止显示超界
        print(f"\r目标角: {target:+8.1f}°   ", end="", flush=True)

    def on_press(key):
        """pynput 按键回调：处理 J7 点动、端点记录、实测角度核对与保存/退出指令。"""
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
            # 仅在用户手动按键时执行一次同步 CAN 读取，用于核对电机实际跟到位情况
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

    # 校验两个极限角度完整性与有效性后再写入 config/grasp.yaml
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

    # 标准安全停机序列：先保持当前位，等待稳态后再缓慢回零并失能
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
