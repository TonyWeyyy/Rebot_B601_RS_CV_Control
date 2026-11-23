#!/usr/bin/env python3
"""
手掌检测推理后端基准测试工具（MediaPipe Tasks HandLandmarker: CPU vs GPU）。

用途：
    使用 Orbbec Gemini 2 60fps 真实彩色画面，依次测评 PalmDetector 在
    CPU 与 GPU 委托后端下的单帧推理耗时（平均 / 中位数 / P95，单位 ms），
    以实测数据决定主程序常驻的 `--detector-backend` 配置。

用法：
    python tools/benchmark_detector.py            # CPU/GPU 各跑 300 帧
    python tools/benchmark_detector.py -n 600     # 自定义采样帧数
    python tools/benchmark_detector.py --headless # 预留无界面参数（不弹窗）

注意：
    本脚本会独占 Gemini 2 相机，运行前请先退出 palm_follow_grasp.py；
    测试时建议将手掌置于镜头视野中，以测得含手掌关键点回归的真实耗时。
"""

import os
import sys
import time
import argparse
import numpy as np

# 将项目根目录（tools/ 的上一级）加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from palm_detection import PalmDetector
from gemini2_camera import Gemini2Camera


def bench_backend(backend, cam, n_frames, headless):
    """对指定的检测后端执行连续帧推理耗时统计。

    会先丢弃前 30 帧预热样本（消除首个推理图编译、缓存冷启动抖动），
    再统计剩余有效帧的平均、中位与 P95 耗时。

    参数:
        backend (str): 检测后端名称（"cpu" 或 "gpu"），透传给 PalmDetector。
        cam (Gemini2Camera): 已打开并处于采流状态的 Gemini 2 相机实例。
        n_frames (int): 目标采样总帧数（含前 30 帧预热；建议远大于 30）。
        headless (bool): 预留的无界面开关（当前测试不创建 OpenCV 窗口）。

    返回:
        dict | None: 初始化失败返回 None；成功返回统计字典：
            - mean (float): 平均单帧推理耗时（ms）
            - median (float): 中位单帧推理耗时（ms）
            - p95 (float): 第 95 百分位单帧推理耗时（ms）
            - n (int): 剔除预热后的有效统计帧数
    """
    try:
        det = PalmDetector(backend=backend)
    except Exception as exc:
        print(f"  [{backend.upper()}] 初始化失败：{exc}")
        return None
    times = []
    # 单调时钟用于整体超时兜底，避免相机降帧或断流导致无限阻塞
    ts_start = time.monotonic()
    while len(times) < n_frames:
        color, depth_raw, depth_scale = cam.get_frames()
        if color is None:
            continue
        # 使用高精度 perf_counter 仅统计 detect() 推理耗时，不含取帧等待
        t0 = time.perf_counter()
        det.detect(color)
        times.append((time.perf_counter() - t0) * 1000)
        # 采满一半以上帧且总耗时超出 30fps 基准时长时提前终止
        if len(times) > n_frames // 2 and \
                time.monotonic() - ts_start > n_frames / 30.0:
            break   # 相机帧率兜底，防止无帧时死等
    det.close()
    times = np.array(times[30:])    # 去掉前 30 帧预热
    return dict(mean=float(times.mean()), median=float(np.median(times)),
                p95=float(np.percentile(times, 95)), n=len(times))


def main():
    """主入口：解析命令行参数，依次评测 CPU 与 GPU 推理性能并输出对比结论。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300, help="每种后端采样帧数")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    # 按主程序一致的 60fps 档位打开相机（gemini2_camera 会自动切换 Binned Sparse 深度模式）
    cam = Gemini2Camera(fps=60)
    print("相机已就绪，请把手放到镜头前（检测有手时的耗时更真实）\n")

    results = {}
    for backend in ("cpu", "gpu"):
        print(f"── 测试 {backend.upper()} 后端（{args.n} 帧）──")
        r = bench_backend(backend, cam, args.n, args.headless)
        if r:
            results[backend] = r
            print(f"  平均 {r['mean']:.1f} ms   中位 {r['median']:.1f} ms   "
                  f"P95 {r['p95']:.1f} ms   (有效帧 {r['n']})\n")

    # 释放相机管线与 USB 带宽资源
    cam.stop()

    # 若 CPU 与 GPU 均测试成功，计算单帧节省时间并给出部署建议
    if len(results) == 2:
        c, g = results["cpu"], results["gpu"]
        saved = c["mean"] - g["mean"]
        print("=" * 50)
        print(f"结论：GPU 比 CPU 每帧快 {saved:.1f} ms"
              f"（{saved / c['mean'] * 100:.0f}%）")
        if saved > 2.0:
            print("建议：保持默认 --detector-backend auto（走 GPU）")
        else:
            print("差距不大，用哪个都行；GPU 省下的 CPU 可留给控制线程")


if __name__ == "__main__":
    main()
