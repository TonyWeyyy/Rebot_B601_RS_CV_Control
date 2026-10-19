#!/usr/bin/env python3
"""枚举 Gemini 2 彩色/深度流支持的全部「分辨率 × 帧率 × 格式」组合。

用途：确认相机在降低分辨率后是否开放 >60fps 档位（如 90fps）——
高帧率 = 感知延迟与运动模糊同步下降，是硬件不换前提下唯一的
帧率提升通道。

用法（与主程序同环境运行）：
    python tools/enumerate_camera_modes.py

输出每张表后会标出 >60fps 的档位。若彩色流存在高帧率档位，
可进一步修改 gemini2_camera.py 的流配置请求它（找作者协助）。
"""

import sys

try:
    from pyorbbecsdk import Pipeline, OBSensorType
except ImportError:
    print("未找到 pyorbbecsdk——请在 rebot-cv 环境下运行本脚本")
    sys.exit(1)


def dump(name, sensor):
    pipe = Pipeline()
    try:
        profiles = pipe.get_stream_profile_list(sensor)
    except Exception as exc:
        print(f"[{name}] 读取失败: {exc}")
        return
    rows = []
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
            vp = p.as_video_stream_profile() if hasattr(p, 'as_video_stream_profile') else p
            w, h, f = vp.get_width(), vp.get_height(), vp.get_fps()
            fmt = str(vp.get_format()).split('.')[-1]
        except Exception:
            # 非视频 profile（如 IMU），跳过
            continue
        rows.append((w, h, f, fmt))
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
    hi = [r for r in rows if r[2] > 60]
    if hi:
        best = max(hi, key=lambda r: r[2])
        print(f"\n  → 存在高帧率档位：最高 {best[2]}fps @ {best[0]}x{best[1]}")
    else:
        print(f"\n  → 无 >60fps 档位，60fps 已是该流上限")


def main():
    print("枚举 Gemini 2 流档位（无需启动主程序）...")
    dump("彩色", OBSensorType.COLOR_SENSOR)
    dump("深度", OBSensorType.DEPTH_SENSOR)
    print("\n说明：彩色流是识别延迟与运动模糊的瓶颈；若彩色流有 >60fps")
    print("档位（如 720p@90），把 gemini2_camera.py 里彩色 profile 请求")
    print("改成该分辨率+fps 即可享用（注意 USB 带宽与检测耗时也要跟上）。")


if __name__ == "__main__":
    main()
