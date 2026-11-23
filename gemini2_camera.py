#!/usr/bin/env python3
"""
Orbbec Gemini 2 RGB-D 相机封装模块（基于 Orbbec SDK v2 / pyorbbecsdk2）
========================================================================
用于替代原项目的 pyrealsense2（RealSense D405）相机层，向上层视觉伺服提供统一接口：
    cam.fx, cam.fy, cam.cx, cam.cy  -> 对齐后的 RGB 相机内参（像素单位）
    cam.camera_matrix               -> 3x3 相机内参矩阵（float32）
    cam.dist_coeffs                 -> 5 维畸变系数（强制置零，见下文说明）
    cam.depth_scale                 -> 深度原始值到米的换算系数（m/unit）
    cam.get_frames(latest_only=True)-> (color_bgr, depth_raw, depth_scale)
    cam.get_last_frame_timing()     -> (timestamp_s, source_str) 硬件帧时间戳

关键硬件与 SDK 特性：
    1. Orbbec SDK v2 Pipeline + 硬件 D2C 对齐（OBAlignMode.HW_MODE）+ 帧同步：
       深度图在相机硬件端直接对齐到 RGB 坐标系，因此反投影与手眼标定统一使用 RGB 内参。
    2. 60fps 深度工作模式切换：
       Gemini 2 默认 Unbinned Dense 模式 RGB 上限为 30fps；当请求 fps > 30 时，
       必须先将设备深度工作模式切换至 ``Binned Sparse Default`` 才能开放 60fps。
    3. 强制零畸变（dist_coeffs = np.zeros(5)）：
       SDK 上报的畸变系数与 OpenCV Brown 模型不兼容（实测 k1≈42.6 会导致 PnP 严重畸变），
       而该相机出厂流经内部校正后按零畸变计算可与深度图达到亚像素级吻合。
    4. 低延迟取流（latest_only=True）：
       排空 SDK 接收队列只保留最新一帧组，跳过旧帧的 MJPG 解码，防止视觉延迟随处理抖动累积。

依赖：
    pip install pyorbbecsdk2 opencv-contrib-python numpy

注意：
    - Gemini 2 深度有效范围约 0.15m ~ 10m（近距不如 D405，跟随距离别设太近）
    - pip 包名是 pyorbbecsdk2（v2.x），``pip install pyorbbecsdk`` 装的是旧 v1 分支
    - 不同版本 pyorbbecsdk2 的时间戳 API 命名略有差异，已通过 ``_read_timestamp_seconds`` 兼容
"""

import numpy as np
import cv2

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBAlignMode, OBFormat, OBError
)


class Gemini2Camera:
    """Orbbec Gemini 2 RGB-D 相机采集封装（Orbbec SDK v2 Pipeline）。

    核心属性：
        width, height (int): 实际生效的彩色图分辨率（像素），高帧率回退时可能不同于请求值。
        fx, fy, cx, cy (float): 对齐后 RGB 相机内参（像素）。
        camera_matrix (np.ndarray): 3x3 float32 内参矩阵。
        dist_coeffs (np.ndarray): 5 维 float32 畸变向量，恒为全零。
        depth_scale (float | None): 深度单位换算系数（米/原始单位），首帧到达时初始化。
        dropped_frames (int): ``latest_only=True`` 排空队列累计跳过的旧帧组数。
    """

    def __init__(self, width=640, height=480, fps=30, align_to_color=True,
                 color_exposure_ms=0.0):
        """初始化并启动 Gemini 2 相机流、硬件 D2C 对齐、内参读取与可选手动曝光。

        Args:
            width (int): 期望的彩色/深度图像宽度（像素），默认 640。
            height (int): 期望的彩色/深度图像高度（像素），默认 480。
            fps (int): 期望帧率（Hz），默认 30；当 ``fps > 30`` 时会自动尝试切换至
                ``Binned Sparse Default`` 深度工作模式以开放 60fps。
            align_to_color (bool): 是否开启硬件级深度到彩色对齐（HW D2C）及帧同步，默认 True。
            color_exposure_ms (float): 可选的彩色手动曝光时长（毫秒）；``<= 0`` 表示保持自动曝光。
                开启后会以「试写 + 实测画面亮度」闭环自校准设备寄存器单位，过暗时自动退回自动曝光。
        """
        self.width = width
        self.height = height

        # 初始化 Orbbec SDK v2 采集管线与流配置对象
        self.pipeline = Pipeline()
        config = Config()

        # ---- 高帧率（>30fps）需要先切换深度工作模式 ----
        # Gemini 2 规格：默认 Unbinned Dense 模式下 RGB 封顶 30fps；
        # 只有 Binned Sparse Default 模式才开放 深度640x400@60 + RGB 1280x720@60。
        # 代价：BS 模式深度是降采样的，精度/细节略低于默认模式。
        if fps > 30:
            try:
                device = self.pipeline.get_device()
                mode_list = device.get_depth_work_mode_list()
                bs_mode = None
                for i in range(mode_list.get_count()):
                    m = mode_list.get_depth_work_mode_by_index(i)
                    name = m.get_name() if hasattr(m, 'get_name') else str(m)
                    print(f"[Gemini2] 可用深度工作模式: {name}")
                    if "Binned" in name and "Sparse" in name:
                        bs_mode = m
                if bs_mode is not None:
                    device.set_depth_work_mode(bs_mode)
                    print("[Gemini2] 已切换深度工作模式 -> Binned Sparse Default"
                          "（开放 60fps；深度精度略降）")
                else:
                    print("[Gemini2] 未找到 Binned Sparse 模式，60fps 不可用，"
                          "将回退到 30fps")
            except Exception as e:
                print(f"[Gemini2] 切换深度工作模式失败({e})，将回退到 30fps")

        # ---- 彩色流（请求 MJPG，取流后用 OpenCV 解码，省 USB 带宽）----
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(
                width, height, OBFormat.MJPG, fps)
        except OBError:
            # 诊断：打印相机当前实际枚举到的全部彩色 profile
            # （60fps MJPG 只在 USB 3.x 下出现；看不到就说明链路掉到 USB 2.0）
            print(f"[Gemini2] 未找到 {width}x{height}@{fps} MJPG，"
                  f"相机实际支持的彩色流:")
            for i in range(color_profiles.get_count()):
                p = color_profiles.get_stream_profile_by_index(i)
                print(f"    {p.get_width()}x{p.get_height()}@"
                      f"{p.get_fps()} {p.get_format()}")
            # 退而求其次：任意分辨率但帧率达标的 MJPG（保帧率优先于保分辨率），
            # 在满足目标 fps 的候选中优先选取像素面积最小的一档以降低解码和检测开销
            color_profile = None
            best_area = None
            for i in range(color_profiles.get_count()):
                p = color_profiles.get_stream_profile_by_index(i)
                if p.get_format() == OBFormat.MJPG and p.get_fps() == fps:
                    area = p.get_width() * p.get_height()
                    if best_area is None or area < best_area:
                        color_profile, best_area = p, area
            if color_profile is None:
                # 仍未匹配目标帧率（常见于 USB 2.0 降速链路），退回 SDK 默认配置
                color_profile = color_profiles.get_default_video_stream_profile()
            print(f"[Gemini2] 回退使用: {color_profile.get_width()}x"
                  f"{color_profile.get_height()}@{color_profile.get_fps()}")
        config.enable_stream(color_profile)
        # 记录实际生效的彩色分辨率（可能与请求的 640x480 不同，如 60fps 模式下 1280x720）
        self.width = color_profile.get_width()
        self.height = color_profile.get_height()

        # ---- 深度流（HW D2C 时优先用官方推荐的 D2C profile 列表选择）----
        depth_profile = None
        if align_to_color:
            try:
                # 查询与当前彩色 profile 兼容的硬件 D2C 深度流配置列表
                d2c_profiles = self.pipeline.get_d2c_depth_profile_list(
                    color_profile, OBAlignMode.HW_MODE)
                for i in range(d2c_profiles.get_count()):
                    p = d2c_profiles.get_stream_profile_by_index(i)
                    if (p.get_width() == width and p.get_height() == height
                            and p.get_fps() == fps):
                        depth_profile = p
                        break
                if depth_profile is None and d2c_profiles.get_count() > 0:
                    # 兜底取 D2C 列表里帧率最高的（取第一个帧率随缘：实测出现过 10/30fps）
                    depth_profile = max(
                        (d2c_profiles.get_stream_profile_by_index(i)
                         for i in range(d2c_profiles.get_count())),
                        key=lambda p: p.get_fps())
            except OBError:
                depth_profile = None

        if depth_profile is None:
            # 未开启 D2C 或 D2C 列表查询失败时，直接从深度传感器枚举 Y16（16-bit 原始深度）流
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
            # 启用相机固件硬件对齐并将彩色/深度帧按时间戳同步成对输出
            config.set_align_mode(OBAlignMode.HW_MODE)
            self.pipeline.enable_frame_sync()

        # 启动采集管线（之后才能读取有效内参并开始拉流）
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

        # 深度 scale 在首帧读取（见 get_frames），此处先置空等待首帧填充
        self.depth_scale = None
        self.dropped_frames = 0          # latest_only 排空丢弃的帧组数（DBG 诊断用）
        self._last_frame_timestamp_s = None  # 最近一帧硬件时间戳（秒），无可用接口时为 None
        self._timestamp_source = ""          # 命中时间戳方法名与缩放因子，用于日志定位

        # 预热：丢弃前 30 帧，等待自动曝光/白平衡与硬件 D2C 同步收敛稳定
        print("[Gemini2] 预热相机...")
        for _ in range(30):
            self.pipeline.wait_for_frames(100)

        # ---- 彩色手动曝光（可选）：压运动模糊 ----
        # 60fps 下自动曝光在室内常顶到 ~16ms 满帧曝光，快速挥手时
        # 手部拖影严重 → MediaPipe 整帧掉检（palm=None，log29 剩余
        # 丢失事件的主因之一）。手动压到 5~8ms 模糊减半，检测找回率
        # 明显提升。代价：画面变暗（曝光短进光少），对检测无影响，
        # 但暗光环境下可能反而掉检——所以默认关闭，按需开启。
        #
        # 单位自校准（log30 全黑的修复）：Gemini 2 彩色曝光寄存器的
        # 单位语义不明确——自动曝光模式下读到的原始值与手动写入值
        # 并非同一尺度，按读数反推单位不可靠（实测 6ms 写成 60 个
        # 单位 → 画面全黑）。改为「试写 + 实测亮度」闭环：
        # 按 100μs/10μs/1μs 三种假设依次写入目标毫秒数，
        # 画面亮度 ≥ 自动曝光基准的 25% 即接受；三档都太暗则
        # 恢复自动曝光并放弃。
        if color_exposure_ms and color_exposure_ms > 0:
            try:
                from pyorbbecsdk import OBPropertyID
                device = self.pipeline.get_device()

                def _grab_brightness(discard=6, take=2):
                    """丢 discard 帧等曝光生效，再取 take 帧的最大亮度 (0~255)。
                    取最大而非平均：预热刚结束时可能夹黑帧（log31 基准
                    测成 0），平均会被拉低；亮度评估要的是"有效帧的水平"。"""
                    vals = []
                    for i in range(discard + take):
                        c, _, _ = self.get_frames()
                        if c is not None and i >= discard:
                            vals.append(float(c.mean()))
                    return float(max(vals)) if vals else 0.0

                auto_brightness = _grab_brightness()   # 自动曝光基准
                if auto_brightness < 10.0:
                    # 基准帧全黑（相机刚启动同步抖动）：用固定参考值，
                    # 保证 25% 阈值仍然有意义
                    auto_brightness = 80.0
                    print("[Gemini2] 自动曝光基准帧过暗，按参考亮度 80 校准")
                device.set_bool_property(
                    OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
                ok = False
                for us_per_unit in (100, 10, 1):
                    units = max(1, int(round(
                        color_exposure_ms * 1000 / us_per_unit)))
                    device.set_int_property(
                        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, units)
                    bri = _grab_brightness()
                    print(f"[Gemini2] 曝光试写 {units}（按 {us_per_unit}μs/单位）"
                          f"→ 亮度 {bri:.0f}（自动基准 {auto_brightness:.0f}）")
                    # 接受阈值要严（log32 的误判教训）：目标曝光与自动
                    # 曝光接近（10ms vs 12ms）时，正确单位的亮度应在基准
                    # 的 ~80%；单位猜错一档差 10 倍曝光，亮度只剩零头。
                    # 25% 阈值曾把 1ms 误判成 10ms（画面全暗）。
                    if bri >= max(25.0, 0.40 * auto_brightness):
                        print(f"[Gemini2] 手动彩色曝光生效: "
                              f"{color_exposure_ms}ms = {units} 单位；"
                              f"画面变暗属正常现象")
                        ok = True
                        break
                    print(f"[Gemini2]   → 亮度不足基准 40%，单位假设错误，换下一档")
                if not ok:
                    device.set_bool_property(
                        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
                    print("[Gemini2] 三档单位都太暗，已恢复自动曝光。"
                          "环境光可能不足，不建议用手动曝光")
            except Exception as e:
                print(f"[Gemini2] 手动曝光设置失败({e})，保持自动曝光")

        print(f"[Gemini2] 相机就绪  内参: fx={self.fx:.2f} fy={self.fy:.2f} "
              f"cx={self.cx:.2f} cy={self.cy:.2f}")

    @staticmethod
    def _read_timestamp_seconds(frame):
        """从 Orbbec 帧对象读取硬件时间戳并统一换算为秒。

        不同版本的 ``pyorbbecsdk2`` 对时间戳方法命名不统一（如 ``get_timestamp_us`` /
        ``get_time_stamp_us`` / ``get_timestamp_ns`` / ``get_timestamp``）。
        按微秒 -> 纳秒 -> 毫秒（通用接口 SDK 默认约定）顺序依次探测。

        Args:
            frame: Orbbec SDK 的 ColorFrame 或 DepthFrame 实例。

        Returns:
            tuple[float | None, str]: ``(timestamp_s, source)``；其中 ``timestamp_s`` 为
            非负有限秒数，``source`` 为命中的接口名与倍率描述；所有候选均不可用时返回
            ``(None, "")``。
        """
        candidates = (
            ("get_timestamp_us", 1e-6), ("get_time_stamp_us", 1e-6),
            ("get_timestamp_ns", 1e-9), ("get_time_stamp_ns", 1e-9),
            ("get_timestamp", 1e-3), ("get_time_stamp", 1e-3),
        )
        for name, scale in candidates:
            method = getattr(frame, name, None)
            if not callable(method):
                continue
            try:
                value = float(method())
                if np.isfinite(value) and value >= 0.0:
                    return value * scale, f"{name}×{scale:g}"
            except Exception:
                # 某些 SDK 的 OBError 并非 RuntimeError；时间能力失败不得中断取流。
                continue
        return None, ""

    def get_last_frame_timing(self):
        """返回最近一次成功取帧对应的硬件时间戳与来源标识。

        独立提供该方法以在不破坏 ``get_frames()`` 三元组返回签名的前提下，向时序模块
        （如 ``FrameTimestampMapper``）暴露硬件帧时间。

        Returns:
            tuple[float | None, str]: ``(timestamp_s, source_str)``，单位为秒；
            若最近取帧失败或设备不支持硬件时间戳，则 ``timestamp_s`` 为 ``None``。
        """
        return self._last_frame_timestamp_s, self._timestamp_source

    def get_frames(self, latest_only=True):
        """读取一帧同步对齐的彩色图与深度图。

        Args:
            latest_only (bool): 默认 True。阻塞等到一帧后立即以 0ms 超时非阻塞排空
                SDK 接收队列，仅保留最新帧组（被跳过的帧组不做 MJPG 解码，几乎零成本）。
                当上层处理速度慢于相机帧率时（如检测耗时上涨，log27 识别 3ms→12ms），
                可避免帧在队列里积压成数百 ms 滞后，确保单帧视觉延迟有界不增长。

        Returns:
            tuple: ``(color_bgr, depth_raw, depth_scale)``：
                - ``color_bgr`` (np.ndarray | None): ``(H, W, 3)`` uint8 BGR 彩色图像。
                - ``depth_raw`` (np.ndarray | None): ``(H, W)`` uint16 原始深度图（已对齐到彩色）。
                - ``depth_scale`` (float | None): 换算系数（m/unit），满足
                  ``depth_m = depth_raw * depth_scale``。
            超时或任一帧缺失时返回 ``(None, None, None)``。
        """
        # 阻塞等待至多 100ms 获取一组帧
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None, None
        if latest_only:
            # 非阻塞轮询把队列里更晚到的帧组全部取出，只留最后一组
            # （被跳过的帧组不做 MJPG 解码，几乎零成本）
            while True:
                newer = self.pipeline.wait_for_frames(0)
                if newer is None:
                    break
                frames = newer
                self.dropped_frames += 1

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            self._last_frame_timestamp_s = None
            return None, None, None

        # 优先读取彩色帧硬件时间戳，缺失时回退读取深度帧时间戳
        timestamp_s, source = self._read_timestamp_seconds(color_frame)
        if timestamp_s is None:
            timestamp_s, source = self._read_timestamp_seconds(depth_frame)
        self._last_frame_timestamp_s = timestamp_s
        if source and source != self._timestamp_source:
            self._timestamp_source = source
            print(f"[Gemini2] 帧硬件时间接口: {source}")

        # 彩色帧解码：MJPG 用 OpenCV 解码为 BGR，未压缩 RGB 则直接 reshape 并转通道顺序
        if color_frame.get_format() == OBFormat.MJPG:
            color_image = cv2.imdecode(
                np.asanyarray(color_frame.get_data()), cv2.IMREAD_COLOR)
        else:  # RGB 格式直接 reshape
            color_image = np.asanyarray(color_frame.get_data()).reshape(
                color_frame.get_height(), color_frame.get_width(), 3)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # 深度帧解析：Y16 小端 uint16 缓冲区直接视图化为 (H, W)
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
        """停止 Orbbec Pipeline 采集并释放 USB 设备流资源。"""
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
