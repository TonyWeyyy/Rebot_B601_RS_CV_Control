#!/usr/bin/env python3
"""
检测后端基准对比：分别用 CPU / GPU 跑 N 帧真实相机画面，
输出各自的平均/中位/P95 耗时（ms），用数据决定常驻后端。

用法：
    python tools/benchmark_detector.py            # 各跑 300 帧
    python tools/benchmark_detector.py -n 600     # 自定义帧数
    python tools/benchmark_detector.py --headless # 不显示画面

注意：与主程序一样占用相机，运行前请退出 palm_follow_grasp.py。
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from palm_detection import PalmDetector
from gemini2_camera import Gemini2Camera


def bench_backend(backend, cam, n_frames, headless):
    try:
        det = PalmDetector(backend=backend)
    except Exception as exc:
        print(f"  [{backend.upper()}] 初始化失败：{exc}")
        return None
    times = []
    ts_start = time.monotonic()
    while len(times) < n_frames:
        color, depth_raw, depth_scale = cam.get_frames()
        if color is None:
            continue
        t0 = time.perf_counter()
        det.detect(color)
        times.append((time.perf_counter() - t0) * 1000)
        if len(times) > n_frames // 2 and \
                time.monotonic() - ts_start > n_frames / 30.0:
            break   # 相机帧率兜底，防止无帧时死等
    det.close()
    times = np.array(times[30:])    # 去掉前 30 帧预热
    return dict(mean=float(times.mean()), median=float(np.median(times)),
                p95=float(np.percentile(times, 95)), n=len(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300, help="每种后端采样帧数")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

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

    cam.stop()

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
