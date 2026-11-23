#!/usr/bin/env python3
"""枚举 Orbbec Gemini 2 彩色/深度流支持的全部「分辨率 × 帧率 × 像素格式」组合。

用途：
    确认相机在降低分辨率后是否开放 >60fps 档位（如 90fps）——
    高帧率可同步降低感知延迟与运动模糊，是硬件不变前提下提升视觉伺服
    前端采样率的主要通道。同时也可用于排查 SDK v2（pyorbbecsdk2）下
    彩色/深度流档位与工作模式（如 60fps 所需的 Binned Sparse Default）匹配情况。

用法（需在与主程序相同的 Python 环境下运行，且主程序未占用相机）：
    python tools/enumerate_camera_modes.py

输出：
    分别打印彩色流与深度流的全部可用档位，并用 `★ >60fps!` 标出高帧率模式。
    若彩色流存在高帧率档位，可据此调整 gemini2_camera.py 中的流配置请求。
"""

import sys

try:
    # pyorbbecsdk2 包的导入名仍为 pyorbbecsdk
    from pyorbbecsdk import Pipeline, OBSensorType
except ImportError:
    print("未找到 pyorbbecsdk——请在 rebot-cv 环境下运行本脚本")
    sys.exit(1)


def dump(name, sensor):
    """查询并格式化打印指定传感器支持的全部视频流档位。

    参数:
        name (str): 传感器中文显示名称（如 "彩色"、"深度"），仅用于终端日志。
        sensor (OBSensorType): Orbbec SDK 传感器类型枚举（如 COLOR_SENSOR / DEPTH_SENSOR）。

    返回:
        None: 结果直接打印至标准输出，并统计是否存在 >60fps 的高帧率档位。
    """
    pipe = Pipeline()
    try:
        profiles = pipe.get_stream_profile_list(sensor)
    except Exception as exc:
        print(f"[{name}] 读取失败: {exc}")
        return
    rows = []
    # 兼容 pyorbbecsdk 不同版本的列表长度接口（v1/v2 可能分别为 get_count() 或 __len__）
    try:
        n = profiles.get_count()
    except AttributeError:
        n = len(profiles)
    for i in range(n):
        try:
            p = profiles.get_stream_profile_by_index(i)
        except Exception:
            continue
        try:
            # 将通用 StreamProfile 转换为 VideoStreamProfile 以读取宽高、帧率与像素格式
            vp = p.as_video_stream_profile() if hasattr(p, 'as_video_stream_profile') else p
            w, h, f = vp.get_width(), vp.get_height(), vp.get_fps()
            fmt = str(vp.get_format()).split('.')[-1]
        except Exception:
            # 非视频 profile（如 IMU），跳过
            continue
        rows.append((w, h, f, fmt))
    # 去重后按「高 → 宽 → 帧率」升序排列，便于按分辨率分组阅读
    rows = sorted(set(rows), key=lambda r: (r[1], r[0], r[2]))
    print(f"\n=== {name} 流支持档位（{len(rows)} 种）===")
    cur = None
    for w, h, f, fmt in rows:
        tag = f"{w}x{h}"
        if tag != cur:
            print(f"  {tag}:")
            cur = tag
        mark = "  ★ >60fps!" if f > 60 else ""
        print(f"      {f:>3} fps  {fmt}{mark}")
    # 汇总 >60fps 的高帧率档位，找出最高帧率配置
    hi = [r for r in rows if r[2] > 60]
    if hi:
        best = max(hi, key=lambda r: r[2])
        print(f"\n  → 存在高帧率档位：最高 {best[2]}fps @ {best[0]}x{best[1]}")
    else:
        print(f"\n  → 无 >60fps 档位，60fps 已是该流上限")


def main():
    """主入口：依次枚举 Gemini 2 的彩色流与深度流可用档位并输出配置建议。"""
    print("枚举 Gemini 2 流档位（无需启动主程序）...")
    dump("彩色", OBSensorType.COLOR_SENSOR)
    dump("深度", OBSensorType.DEPTH_SENSOR)
    print("\n说明：彩色流是识别延迟与运动模糊的瓶颈；若彩色流有 >60fps")
    print("档位（如 720p@90），把 gemini2_camera.py 里彩色 profile 请求")
    print("改成该分辨率+fps 即可享用（注意 USB 带宽与检测耗时也要跟上）。")


if __name__ == "__main__":
    main()
