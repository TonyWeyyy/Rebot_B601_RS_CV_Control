#!/usr/bin/env python3
"""
Seeed reBot Arm B601-RS MuJoCo Studio 3D 实时数字孪生独立启动器
===============================================================
完全基于官方 `mujoco>=4.0.1`（`mujoco.studio`），不依赖 Antigravity 或外部守护进程，
支持两种独立运行模式：

  1. 本地原生桌面窗口模式（推荐本地使用，无网页/无端口，本地 OpenGL/Vulkan 渲染）：
       python3 mujoco_twin/run_twin.py --native
  2. 独立浏览器 Web 模式（远程/SSH/跨屏查看）：
       python3 mujoco_twin/run_twin.py --web --http-port 8080

启动后运行 `python3 palm_follow_v14.py`（真机）或 `python3 palm_follow_v14.py --sim`（仿真），
即可经 `udp://127.0.0.1:19870` 自动进入实时双机同步。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

TWIN_DIR = pathlib.Path(os.path.abspath(__file__)).parent
if str(TWIN_DIR) not in sys.path:
    sys.path.insert(0, str(TWIN_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="reBot Arm B601-RS MuJoCo Studio 3D 实时数字孪生独立启动器"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--native",
        action="store_true",
        help="以本地原生桌面 3D 窗口启动（OpenGL/Vulkan，无需浏览器与网页端口）",
    )
    mode.add_argument(
        "--web",
        action="store_true",
        help="以浏览器 WebViewer 模式启动（默认 http://localhost:8080/）",
    )
    p.add_argument(
        "--udp-port",
        type=int,
        default=19870,
        help="监听真机/SIM 实时遥测的本机 UDP 端口（默认 19870）",
    )
    p.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="WebViewer HTTP 端口（仅 --web 生效，默认 8080）",
    )
    p.add_argument(
        "--open-browser",
        action="store_true",
        help="启动 --web 后自动打开系统默认浏览器",
    )
    p.add_argument(
        "--gfx",
        default="",
        help="指定原生图形后端（留空自动选择，可选 opengl / vulkan）",
    )
    p.add_argument("--width", type=int, default=1440, help="原生窗口初始宽度（默认 1440）")
    p.add_argument("--height", type=int, default=900, help="原生窗口初始高度（默认 900）")
    p.add_argument(
        "--no-hot-reload",
        action="store_true",
        help="禁用插件热重载监听",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    use_web = bool(args.web)

    from mujoco.studio import app as studio_app
    from mujoco.studio import launch_native
    from mujoco.studio import launch_web
    from mujoco.studio import plugin
    from mujoco.studio import viewer_protocol

    user_plugins = plugin.load_plugins(
        [TWIN_DIR / "sim_plugin.py", TWIN_DIR / "viewer_plugin.py"]
    )
    for plg in user_plugins:
        if hasattr(plg, "udp_port"):
            plg.udp_port = int(args.udp_port)

    config = viewer_protocol.ViewerConfig(
        title="reBot Arm B601-RS Digital Twin",
        width=int(args.width),
        height=int(args.height),
        gfx=str(args.gfx),
        http_port=int(args.http_port) if use_web else 0,
        open_browser=bool(args.open_browser) if use_web else False,
        hot_reload="" if args.no_hot_reload else "all",
    )
    all_plugins = [studio_app.App(), *studio_app.SIM_PLUGINS, *user_plugins]

    if use_web:
        print(
            f"[MuJoCo Twin] WebViewer 模式启动 -> http://localhost:{args.http_port}/ "
            f"| UDP 监听 127.0.0.1:{args.udp_port}",
            flush=True,
        )
        launch_web.run(config, plugins=all_plugins)
    else:
        print(
            f"[MuJoCo Twin] 本地原生桌面窗口启动 ({args.width}x{args.height}) "
            f"| UDP 监听 127.0.0.1:{args.udp_port}",
            flush=True,
        )
        launch_native.run(config, plugins=all_plugins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
