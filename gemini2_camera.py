#!/usr/bin/env python3
"""
Orbbec Gemini 2 相机封装
替代原项目的 pyrealsense2 (RealSense D405) 相机层。

对外接口与原项目 init_realsense() 返回的东西一一对应：
    cam.intrinsic   -> fx, fy, cx, cy（RGB 相机内参，深度已对齐到 RGB）
    cam.depth_scale -> 深度原始单位到米的换算系数
    cam.get_frames() -> (color_bgr, depth_raw)  numpy 数组

依赖：
    pip install pyorbbecsdk opencv-contrib-python numpy

注意：
    - Gemini 2 深度有效范围约 0.15m ~ 10m（近距不如 D405，跟随距离别设太近）
    - pip 包名是 pyorbbecsdk2（v2.x），`pip install pyorbbecsdk` 装的是旧 v1 分支
    - 不同版本 pyorbbecsdk 的 API 名字可能略有差异，
      若报错请对照你安装的 SDK 示例（example/python 目录）微调本文件
"""

import numpy as np
import cv2

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBAlignMode, OBFormat, OBError
)


class Gemini2Camera:
    def __init__(self, width=640, height=480, fps=30, align_to_color=True):
        self.width = width
        self.height = height

        self.pipeline = Pipeline()
        config = Config()

        # ---- 彩色流（请求 MJPG，取流后用 OpenCV 解码，省 USB 带宽）----
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(
                width, height, OBFormat.MJPG, fps)
        except OBError:
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"[Gemini2] 未找到 {width}x{height}@{fps} MJPG，使用默认彩色配置: "
                  f"{color_profile.get_width()}x{color_profile.get_height()}")
        config.enable_stream(color_profile)

        # ---- 深度流（HW D2C 时优先用官方推荐的 D2C profile 列表选择）----
        depth_profile = None
        if align_to_color:
            try:
                d2c_profiles = self.pipeline.get_d2c_depth_profile_list(
                    color_profile, OBAlignMode.HW_MODE)
                for i in range(d2c_profiles.get_count()):
                    p = d2c_profiles.get_stream_profile_by_index(i)
                    if (p.get_width() == width and p.get_height() == height
                            and p.get_fps() == fps):
                        depth_profile = p
                        break
                if depth_profile is None and d2c_profiles.get_count() > 0:
                    depth_profile = d2c_profiles.get_stream_profile_by_index(0)
            except OBError:
                depth_profile = None

        if depth_profile is None:
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                depth_profile = depth_profiles.get_video_stream_profile(
                    width, height, OBFormat.Y16, fps)
            except OBError:
                depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        print(f"[Gemini2] 深度流: {depth_profile.get_width()}x"
              f"{depth_profile.get_height()}@{depth_profile.get_fps()}")

        # ---- 深度对齐到彩色（D2C，硬件对齐）+ 帧同步 ----
        if align_to_color:
            config.set_align_mode(OBAlignMode.HW_MODE)
            self.pipeline.enable_frame_sync()

        self.pipeline.start(config)

        # ---- 读取 RGB 相机内参（对齐后统一用 RGB 内参做反投影和标定）----
        camera_param = self.pipeline.get_camera_param()
        intr = camera_param.rgb_intrinsic   # 对齐模式下用 rgb_intrinsic
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.cx, intr.cy
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]], dtype=np.float32)
        # OBCameraDistortion 字段是 k1..k6/p1/p2（没有 .coeffs），
        # OpenCV 顺序为 (k1, k2, p1, p2, k3)
        # SDK 返回的畸变系数语义与 OpenCV Brown 模型不符（实测 k1=42.6，
        # 导致 solvePnP 重投影误差 ~90px、距离错误）。该相机流实测零畸变
        # 时 PnP 与深度图完全吻合（0.27px / 0% 误差），故强制置零。
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

        # 深度 scale 在首帧读取（见 get_frames）
        self.depth_scale = None

        # 预热，丢弃前几帧
        print("[Gemini2] 预热相机...")
        for _ in range(30):
            self.pipeline.wait_for_frames(100)

        print(f"[Gemini2] 相机就绪  内参: fx={self.fx:.2f} fy={self.fy:.2f} "
              f"cx={self.cx:.2f} cy={self.cy:.2f}")

    def get_frames(self):
        """
        返回 (color_bgr, depth_raw, depth_scale)：
            color_bgr  : HxWx3 uint8 BGR 图
            depth_raw  : HxW uint16 深度原始值（已对齐到 color）
            depth_scale: float，米 = depth_raw * depth_scale
        失败返回 (None, None, None)
        """
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None, None

        # 彩色：MJPG 解码为 BGR
        if color_frame.get_format() == OBFormat.MJPG:
            color_image = cv2.imdecode(
                np.asanyarray(color_frame.get_data()), cv2.IMREAD_COLOR)
        else:  # RGB 格式直接 reshape
            color_image = np.asanyarray(color_frame.get_data()).reshape(
                color_frame.get_height(), color_frame.get_width(), 3)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # 深度：uint16 原始值
        h, w = depth_frame.get_height(), depth_frame.get_width()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)

        if self.depth_scale is None:
            # get_depth_scale() 返回的是"原始值 → 毫米"的换算系数（Gemini 2 ≈ 1.0，
            # 即原始单位本来就是 1mm）。本模块对外统一用米，所以除以 1000。
            scale_mm = depth_frame.get_depth_scale()
            self.depth_scale = scale_mm / 1000.0
            print(f"[Gemini2] depth_scale = {scale_mm} mm/unit "
                  f"(Gemini 2 应约等于 1.0；若相差很大请检查 SDK 版本)")

        return color_image, depth_raw, self.depth_scale

    def stop(self):
        self.pipeline.stop()


if __name__ == "__main__":
    # 单独测试：python gemini2_camera.py
    cam = Gemini2Camera()
    print("按 q 退出")
    try:
        while True:
            color, depth, scale = cam.get_frames()
            if color is None:
                continue
            depth_m = depth.astype(np.float32) * scale
            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("color", color)
            cv2.imshow("depth", depth_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
