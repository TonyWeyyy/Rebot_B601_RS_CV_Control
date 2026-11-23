#!/usr/bin/env python3
"""
手掌检测、跟踪与 3D 掌心/捏合提取模块（替代原项目 blue_circle_detection.py）
==========================================================================
基于 MediaPipe Tasks API（``HandLandmarker``, ``VIDEO`` 运行模式）从 Gemini 2
RGB-D 图像中实时提取掌心 2D 像素坐标、3D 相机系坐标及拇指-食指捏合距离。

核心流水线与设计要点：
    1. MediaPipe Tasks API + VIDEO 模式单调时间戳（``PalmDetector``）：
       - 严格保证每次 ``detect_for_video`` 传入单调递增的毫秒整数时间戳；
       - 支持迟滞 ROI 裁剪加速（``compute_roi_box`` / ``remap_roi_landmarks``），
         ROI 漏检时在同一帧内立即回退全图推理并保证时间戳仍 ``+1ms`` 递增，零额外掉帧。
    2. 光流桥接（``LKPalmBridge``）：
       - 快速挥手导致运动模糊使 MediaPipe 掉检时，用 Lucas-Kanade 金字塔光流跟踪
         掌心 3x3 点阵短时桥接，具备「实检即重置、最长连续帧上限、点数与步长门限」三道保险。
    3. 2D 跨帧追踪与自适应平滑（``PalmTracker`` + ``OneEuroFilter``）：
       - One Euro Filter 静止低截止频率抑抖、高速随速度升高截止频率降滞后；
       - 保留 ``MAX_LOST_FRAMES`` 短暂丢手保持，兼容旧版固定系数 EMA。
    4. 深度鲁棒提取与反投影（``get_median_depth`` / ``pixel_to_3d``）：
       - 三级扩展窗口（``patch → 2x → 3x``）应对 60fps BS 模式深度空洞；
       - 结合上一帧参考深度 ``ref_depth`` 优先对前景像素取中值，过滤背景墙边缘跳变。
    5. 捏合距离提取（``pinch_distance_px``）：
       - 复用同一次 MediaPipe 21 点输出计算拇指尖(4)—食指尖(8)像素距离，驱动 J7 夹爪。

依赖：
    pip install mediapipe opencv-contrib-python numpy
    并下载模型文件 ``hand_landmarker.task`` 放置于 ``config/`` 目录：
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

掌心代表点：9 号关键点（中指根部 MCP），比腕点更稳、比 21 点中心更抗遮挡。
"""

import os
import time
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from servo_timing import compute_roi_box, remap_roi_landmarks

# ── 配置 ──────────────────────────────────────────────────
PALM_LANDMARK_ID = 9        # MediaPipe 21 点中的 9 号点（中指根部 MCP），作为掌心代表点
PINCH_TIP_IDS = (4, 8)      # 捏合手势：拇指尖(4)—食指尖(8)，用于夹爪开合控制
MIN_DEPTH_M = 0.15          # 深度有效下限（米）：Gemini 2 近距极限，小于此值的深度丢弃
MAX_DEPTH_M = 1.50          # 深度有效上限（米）：跟随工作场景用不到更远距离，兼防背景误入
DEPTH_PATCH = 3             # 深度采样初始半窗口（像素），对应 (2*3+1)=7x7 窗口

MAX_LOST_FRAMES = 8         # 短时丢失保持上限（帧）：与原项目一致，超限后清空轨迹
SMOOTH_ALPHA = 0.4          # EMA 回退模式下的指数平滑系数（无量纲 0~1，越小越平滑）
NUM_HANDS = 1               # 最大检测手数：只跟一只手，避免双手同框引发左右手跳变

MIN_DETECTION_CONFIDENCE = 0.5  # 手部初检置信度下限（0~1）
MIN_TRACKING_CONFIDENCE = 0.4   # 跨帧跟踪置信度下限（0~1），适当调低以容忍运动模糊帧

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config", "hand_landmarker.task")


class PalmDetector:
    """MediaPipe Tasks API 手掌关键点检测器（``VIDEO`` 模式）。

    支持 GPU/CPU 推理后端自动选择，以及基于上一帧掌心位置的迟滞 ROI 裁剪加速。
    ``backend`` 取值：``"auto"``（优先 GPU，失败回退 CPU）/ ``"gpu"`` / ``"cpu"``。
    GPU delegate 依赖 OpenGL ES/EGL（桌面独显 Linux 一般可用），可将单帧推理从
    CPU XNNPACK ~10ms 降至 GPU ~3~5ms。
    """

    def __init__(self, backend="auto", roi_enable=False, roi_scale=0.68):
        """初始化 MediaPipe HandLandmarker 推理器。

        Args:
            backend (str): 推理后端，支持 ``"auto"``、``"gpu"``、``"cpu"``（大小写不敏感）。
                选择 ``"gpu"`` 且初始化失败时直接抛出异常；``"auto"`` 则自动回退至 CPU。
            roi_enable (bool): 是否启用迟滞 ROI 裁剪加速，默认 False。
            roi_scale (float): ROI 窗口边长占整图宽高的比例（无量纲），默认 0.68。

        Raises:
            FileNotFoundError: ``config/hand_landmarker.task`` 模型文件不存在时抛出。
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"未找到模型文件 {MODEL_PATH}\n"
                "请下载 hand_landmarker.task 放到 config/ 目录：\n"
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")
        backend = str(backend).lower()
        try_gpu = backend in ("auto", "gpu")
        self.backend = "cpu"
        if try_gpu:
            try:
                self.detector = self._create(
                    mp_python.BaseOptions.Delegate.GPU)
                self.backend = "gpu"
            except Exception as exc:
                if backend == "gpu":
                    raise
                print(f"[检测器] GPU 初始化失败（{exc}），回退 CPU")
        if self.backend == "cpu":
            self.detector = self._create(
                mp_python.BaseOptions.Delegate.CPU)
        self.roi_enable = bool(roi_enable)
        self.roi_scale = float(roi_scale)
        self._roi_box = None   # 当前迟滞 ROI 框 (x0, y0, x1, y1) 像素坐标，None 表示下次跑全图
        print(f"[检测器] 推理后端: {self.backend.upper()}"
              + (f"  ROI加速: ON (scale={self.roi_scale:.2f})"
                 if self.roi_enable else ""))
        self._timestamp_ms = 0

    def _next_timestamp_ms(self):
        """生成严格单调递增的毫秒整数时间戳，满足 MediaPipe VIDEO 模式硬约束。

        当同一物理帧内先跑 ROI 推理漏检、随后立刻回退全图推理时，系统毫秒时钟可能
        尚未前进，使用 ``max(self._timestamp_ms + 1, now_ms)`` 确保至少 ``+1ms``，
        防止 MediaPipe 抛出非单调时间戳异常。

        Returns:
            int: 严格大于上一次调用值的单调时间戳（毫秒）。
        """
        # VIDEO 模式严格要求时间戳单调递增（同帧 ROI 漏检回退全图时也能保证 +1ms）
        now_ms = int(time.monotonic() * 1000)
        self._timestamp_ms = max(self._timestamp_ms + 1, now_ms)
        return self._timestamp_ms

    def _create(self, delegate):
        """按指定硬件 Delegate 构造 MediaPipe ``HandLandmarker`` 实例。

        Args:
            delegate: ``mp_python.BaseOptions.Delegate.GPU`` 或 ``CPU``。

        Returns:
            mp_vision.HandLandmarker: 配置为 ``VIDEO`` 模式的检测器实例。
        """
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=MODEL_PATH, delegate=delegate),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
        return mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, color_bgr):
        """对单帧 BGR 彩色图执行手掌关键点检测。

        ``roi_enable=True`` 且存在有效历史 ROI 框时，优先对掌心周边的迟滞 ROI 裁剪图
        推理（约省 65~74% 像素），并将关键点归一化坐标映射回全图坐标系；若 ROI 漏检
        （如超快甩手冲出裁剪窗口），同一帧立即清空 ROI 并回退全图推理，确保零额外掉帧。

        Args:
            color_bgr (np.ndarray): ``(H, W, 3)`` uint8 OpenCV BGR 格式图像。

        Returns:
            tuple: ``(u, v, landmarks)``：
                - ``u, v`` (int | None): 掌心代表点（9 号关键点）在全图中的像素坐标（px）。
                - ``landmarks`` (list | None): 21 个全图归一化关键点（``x, y ∈ [0, 1]``）。
            未检测到手部时返回 ``(None, None, None)``。
        """
        h, w = color_bgr.shape[:2]
        if self.roi_enable and self._roi_box is not None:
            x0, y0, x1, y1 = self._roi_box
            if 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h:
                # 切片裁剪后数组可能非连续，MediaPipe Image 要求 C-contiguous 内存布局
                roi_rgb = np.ascontiguousarray(
                    cv2.cvtColor(color_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
                mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
                roi_res = self.detector.detect_for_video(
                    mp_roi, self._next_timestamp_ms())
                if roi_res.hand_landmarks:
                    # 将 ROI 局部归一化坐标还原为全图归一化坐标，再更新下一帧迟滞 ROI 框
                    landmarks = remap_roi_landmarks(
                        roi_res.hand_landmarks[0], self._roi_box, w, h)
                    lm = landmarks[PALM_LANDMARK_ID]
                    u, v = int(lm.x * w), int(lm.y * h)
                    self._roi_box = compute_roi_box(
                        w, h, (u, v), prev_box=self._roi_box,
                        scale=self.roi_scale)
                    return u, v, landmarks
            # ROI 漏检：立刻同帧回退全图推理，绝不白丢这一帧
            self._roi_box = None

        # 全图推理路径（首次检测、未开启 ROI、或 ROI 漏检同帧回退）
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(
            mp_image, self._next_timestamp_ms())

        if not result.hand_landmarks:
            self._roi_box = None
            return None, None, None

        landmarks = result.hand_landmarks[0]
        lm = landmarks[PALM_LANDMARK_ID]
        u, v = int(lm.x * w), int(lm.y * h)
        if self.roi_enable:
            # 全图重新找回手掌后，以新掌心为中心初始化迟滞 ROI 框
            self._roi_box = compute_roi_box(
                w, h, (u, v), prev_box=None, scale=self.roi_scale)
        return u, v, landmarks

    def close(self):
        """释放 MediaPipe HandLandmarker 底层推理资源与 GPU/CPU 上下文。"""
        self.detector.close()


class OneEuroFilter:
    """One Euro Filter（Casiez et al. 2012）：静止时强滤波、运动时自动减弱滤波。

    相比固定系数 EMA，解决了「平滑则滞后、灵敏则抖动」的两难：
    手静止时截止频率低（输出稳定），手快速移动时截止频率随估计速度线性升高
    （显著降低动态相位滞后），是体感跟随与视觉手势交互的标准滤波算法。
    """

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        """初始化标量 One Euro 滤波器。

        Args:
            min_cutoff (float): 最小截止频率（Hz），控制静止时的平滑强度，越小越稳。
            beta (float): 速度灵敏度系数（Hz / (信号单位/s)），越大高速滞后越小，
                但过大会将高频抖动误判为运动。
            d_cutoff (float): 用于平滑原始一阶差分速度的低通截止频率（Hz），默认 1.0。
        """
        self.min_cutoff = float(min_cutoff)   # Hz，越小静止时越稳
        self.beta = float(beta)               # 速度系数，越大高速滞后越小
        self.d_cutoff = float(d_cutoff)       # 速度估计的滤波截止频率（Hz）
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        """清除滤波器历史状态（长丢手重置或相机重连后避免旧值拖拽）。"""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        """由截止频率 ``cutoff``（Hz）和采样间隔 ``dt``（s）计算一阶低通平滑因子 α。

        离散化公式：``tau = 1 / (2π * cutoff)``，``alpha = 1 / (1 + tau / dt)``。

        Args:
            cutoff (float): 截止频率（Hz，正数）。
            dt (float): 采样间隔（秒，正数）。

        Returns:
            float: 平滑权重 ``alpha ∈ (0, 1)``。
        """
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        """输入新采样并输出自适应滤波后的结果。

        Args:
            x (float): 当前时刻原始标量观测值（本模块中为像素坐标 u 或 v，单位 px）。
            t (float | None): 单调递增时间戳（秒）；为 ``None`` 时自动读取 ``time.monotonic()``。

        Returns:
            float: 滤波后的标量输出值（与输入 ``x`` 同单位）。
        """
        if t is None:
            t = time.monotonic()
        if self.t_prev is None:
            # 首帧直接直通初始化，不引入起跳瞬态
            self.t_prev = t
            self.x_prev = float(x)
            return self.x_prev
        # 防止重复时间戳或极端高频调用导致除零
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t
        # 1) 估计并低通滤波信号变化率 dx/dt
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev += a_d * (dx - self.dx_prev)
        # 2) 按平滑后的速度幅值动态抬高主滤波器截止频率
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, dt)
        self.x_prev += a * (x - self.x_prev)
        return self.x_prev


# 掌心滤波器选择：one_euro（推荐，高速跟随滞后小）或 ema（旧行为）
PALM_FILTER = "one_euro"
OE_MIN_CUTOFF = 1.0         # One Euro 最小截止频率（Hz）：越小静止越稳
OE_BETA = 0.02              # One Euro 速度系数：越大高速越快（太大会把噪声当运动）


class LKPalmBridge:
    """检测失败帧的光流桥接器（检测 + 跟踪混合架构）。

    MediaPipe 因运动模糊整帧掉检时，用 Lucas-Kanade 金字塔光流追踪
    掌心周围 3×3 点阵，产出估计掌心位置继续喂 ``PalmTracker``——
    短暂掉检不再触发丢失保持（快速挥手丢手的主要兜底）。
    只桥接、不替代检测，内置三道保险：
      1) 检测成功帧立刻用真实掌心重置点阵，桥接永不带偏目标；
      2) 连续桥接超 ``max_frames``（默认 15 帧 ≈ 0.25s@60fps）自动放弃，
         回退原有丢失流程（防漂移累积）；
      3) 单帧中值位移 > ``max_step_px`` 或有效跟踪点 < 5 → 放弃本帧。
    纯 OpenCV 主库实现（``cv2.calcOpticalFlowPyrLK``），不依赖 ``opencv-contrib``。
    桥接帧无 MediaPipe 关键点 → 捏合门控自然冻结夹爪，无误夹风险。
    """

    def __init__(self, enable=True, max_frames=15, box_half=45,
                 max_step_px=90.0):
        """初始化 LK 光流桥接器。

        Args:
            enable (bool): 是否启用光流桥接，默认 True；False 时直接透传检测结果。
            max_frames (int): 允许连续光流桥接的最大帧数，超限后强制放弃等待实检。
            box_half (int): 3x3 种子点阵半宽参考值（像素），实际点阵偏移为 ``±box_half // 2``。
            max_step_px (float): 单帧允许的最大中值像素位移（px），超过视为光流发散/跳变。
        """
        self.enable = bool(enable)
        self.max_frames = int(max_frames)
        self.box_half = int(box_half)
        self.max_step_px = float(max_step_px)
        self._prev_gray = None    # 上一帧 uint8 灰度图
        self._pts = None          # 上一帧 3×3 点阵 (N, 1, 2) float32
        self._tracked = 0         # 已连续桥接帧数

    def reset(self):
        """清空历史灰度图、跟踪点阵与连续桥接计数（丢手重置或模式切换时调用）。"""
        self._prev_gray = None
        self._pts = None
        self._tracked = 0

    def _seed(self, gray, uv):
        """以真实检测掌心 ``uv`` 为中心生成 3×3 光流特征种子点阵。

        自动剔除距离图像四边 ``<= 5 px`` 的越界或贴边点；若剩余有效点数少于 5 个，
        则放弃播种（置 ``self._pts = None``）以防光流退化。

        Args:
            gray (np.ndarray): 当前帧 ``(H, W)`` uint8 灰度图。
            uv (tuple[float, float]): 掌心像素坐标 ``(u, v)``（px）。
        """
        u, v = float(uv[0]), float(uv[1])
        h, w = gray.shape[:2]
        b = self.box_half
        pts = [[float(x), float(y)]
               for y in (v - b // 2, v, v + b // 2)
               for x in (u - b // 2, u, u + b // 2)
               if 5 < x < w - 5 and 5 < y < h - 5]
        self._pts = (np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                     if len(pts) >= 5 else None)

    def update(self, color_bgr, det_uv):
        """结合当前帧图像与 MediaPipe 检测结果输出掌心位置（实检优先，漏检桥接）。

        Args:
            color_bgr (np.ndarray): 当前帧 ``(H, W, 3)`` uint8 BGR 图像。
            det_uv (tuple[float, float] | None): 本帧 MediaPipe 检测到的掌心像素 ``(u, v)``，
                漏检时传入 ``None``。

        Returns:
            tuple: ``(u, v, source)``：
                - ``u, v`` (float | None): 输出掌心像素坐标（px）。
                - ``source`` (str | None): ``'det'`` 表示来自实检，``'lk'`` 表示来自光流桥接
                  （无 21 点关键点），``None`` 表示本帧实检与桥接均不可用。
        """
        if not self.enable:
            return (*det_uv, 'det') if det_uv is not None else (None, None, None)
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        out_u, out_v, src = None, None, None
        if det_uv is not None:
            # 保险 1：实检成功立刻重新播种并清零桥接计数，杜绝长期漂移
            self._seed(gray, det_uv)      # 真实掌心重置点阵
            out_u, out_v, src = float(det_uv[0]), float(det_uv[1]), 'det'
            self._tracked = 0
        elif self._pts is not None and self._tracked < self.max_frames \
                and self._prev_gray is not None:
            # 保险 2：仅在连续桥接未达上限且存在历史帧与点阵时运行 4 层金字塔 LK 光流
            pts2, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._pts, None,
                winSize=(31, 31), maxLevel=3)
            if pts2 is not None:
                st = st.reshape(-1).astype(bool)
                good_new = pts2.reshape(-1, 2)[st]
                good_old = self._pts.reshape(-1, 2)[st]
                # 保险 3：至少保留 5 个内点，且点阵中值位移未超过 max_step_px 门限
                if good_new.shape[0] >= 5:
                    step = float(np.median(
                        np.linalg.norm(good_new - good_old, axis=1)))
                    if step <= self.max_step_px:
                        # 取各有效点坐标中值作为掌心估计，抗局部异常背景点干扰
                        out_u = float(np.median(good_new[:, 0]))
                        out_v = float(np.median(good_new[:, 1]))
                        src = 'lk'
                        self._tracked += 1
                        self._pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        if src != 'lk':
            self._tracked = 0
        if src is None:
            self._pts = None              # 桥接失败：等下次检测重新播种
        self._prev_gray = gray
        return out_u, out_v, src


class PalmTracker:
    """单目标掌心 2D 跨帧追踪器（对应原项目 ``CircleTracker`` 单目标版）。

    维护掌心像素位置平滑状态与短时丢失计数；3D 坐标字段（``x3d, y3d, z3d``）
    由上层深度反投影后写入同一字典以便统一传递与可视化。
    """

    def __init__(self):
        """初始化空轨迹与 u/v 双轴 One Euro 滤波器。"""
        self.track = None    # 轨迹字典：{'u','v','lost','x3d','y3d','z3d'}（像素/帧数/米）
        self._fu = OneEuroFilter(OE_MIN_CUTOFF, OE_BETA)
        self._fv = OneEuroFilter(OE_MIN_CUTOFF, OE_BETA)

    def update(self, detection):
        """用本帧 2D 掌心观测更新追踪状态。

        Args:
            detection (tuple[float, float] | None): 本帧检测或光流桥接得到的掌心像素
                ``(u, v)``（px）；完全丢失时传入 ``None``。

        Returns:
            dict | None: 当前轨迹字典（含 ``u, v, lost, x3d, y3d, z3d``）；当处于
            ``1 <= lost <= MAX_LOST_FRAMES`` 时返回保持状态的旧轨迹，超过上限返回 ``None``。
        """
        if detection is not None:
            u, v = detection
            if self.track is None:
                # 首次建立轨迹：直接以观测初始化 2D 坐标，3D 坐标待深度层填充
                self.track = {'u': float(u), 'v': float(v), 'lost': 0,
                              'x3d': 0.0, 'y3d': 0.0, 'z3d': 0.0}
            else:
                if PALM_FILTER == "one_euro":
                    t_now = time.monotonic()
                    self.track['u'] = self._fu(u, t_now)
                    self.track['v'] = self._fv(v, t_now)
                else:
                    # 保持浮点累积，避免每帧 int() 截断造成的系统性漂移
                    self.track['u'] = SMOOTH_ALPHA * u + (1 - SMOOTH_ALPHA) * self.track['u']
                    self.track['v'] = SMOOTH_ALPHA * v + (1 - SMOOTH_ALPHA) * self.track['v']
                self.track['lost'] = 0
        else:
            if self.track is not None:
                # 短时丢手递增计数，超过 MAX_LOST_FRAMES 后彻底清除轨迹
                self.track['lost'] += 1
                if self.track['lost'] > MAX_LOST_FRAMES:
                    self.track = None

        return self.track


def get_median_depth(depth_raw, depth_scale, u, v, patch=DEPTH_PATCH,
                     ref_depth=0.0, max_jump=0.08):
    """在像素 ``(u, v)`` 邻域提取抗空洞、抗背景跳变的中值深度（单位：米）。

    针对 60fps Binned Sparse 模式下掌心小窗口易出现深度空洞的问题，采样半窗口按
    ``patch → 2*patch → 3*patch``（对应边长 ``(2p+1)×(2p+1)``，默认 7×7 → 13×13 → 19×19）
    逐级扩展，仅当最大窗口内有效像素仍不足 5 个时才放弃本帧。

    当提供有效的上一帧参考深度 ``ref_depth > 0`` 时，若窗口跨过手掌边缘混入远端
    背景墙像素，但窗口内仍存在 ``>= 5`` 个与 ``ref_depth`` 差值在 ``max_jump`` 内的
    前景手掌像素，则优先仅对前景像素取中值，显著降低高速挥手时的边缘假跳变率。

    Args:
        depth_raw (np.ndarray): ``(H, W)`` uint16 原始深度图（已对齐到彩色图）。
        depth_scale (float): 原始深度单位到米的换算系数（m/unit）。
        u (int): 中心像素列坐标（px）。
        v (int): 中心像素行坐标（px）。
        patch (int): 初始半窗口大小（px），默认 ``DEPTH_PATCH``（3）。
        ref_depth (float): 上一帧有效参考深度（m），``<= 0`` 表示不启用前景连续过滤。
        max_jump (float): 前景连续性允许的最大深度差（m），默认 0.08m。

    Returns:
        float: 估计的掌心深度（米）；若全窗口无效或超量程返回 ``0.0``。
    """
    h, w = depth_raw.shape
    for p in (patch, patch * 2, patch * 3):
        x0, x1 = max(0, u - p), min(w, u + p + 1)
        y0, y1 = max(0, v - p), min(h, v + p + 1)
        window = depth_raw[y0:y1, x0:x1].astype(np.float32) * depth_scale
        valid = window[(window >= MIN_DEPTH_M) & (window <= MAX_DEPTH_M)]
        if valid.size >= 5:     # 有效点太少说明在边缘/遮挡处，不可信
            if ref_depth > 0.0 and max_jump > 0.0:
                # 前景优先过滤：只保留与上一帧深度连续的手掌区域像素
                fg = valid[np.abs(valid - ref_depth) <= max_jump]
                if fg.size >= 5:
                    return float(np.median(fg))
            return float(np.median(valid))
    return 0.0


def pixel_to_3d(u, v, depth_m, fx, fy, cx, cy):
    """基于针孔相机模型将像素坐标与深度反投影至相机 3D 坐标系（单位：米）。

    相机坐标系采用标准光学右手系：X 朝右、Y 朝下、Z 沿光轴朝前。
    与原项目 ``pixel_to_3d`` 数学等价，不依赖 RealSense SDK。

    Args:
        u (float): 图像像素横坐标（px）。
        v (float): 图像像素纵坐标（px）。
        depth_m (float): 该像素处的 Z 轴深度（m）。
        fx (float): X 方向焦距（px）。
        fy (float): Y 方向焦距（px）。
        cx (float): 主点 X 坐标（px）。
        cy (float): 主点 Y 坐标（px）。

    Returns:
        tuple[float, float, float]: 相机坐标系下的三维点 ``(X, Y, Z)``，单位米。
    """
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return X, Y, Z


def pinch_distance_px(landmarks, img_shape):
    """计算拇指尖(4)—食指尖(8)的欧氏像素距离（用于捏合手势 → 夹爪开合映射）。

    与掌心跟随复用同一次 MediaPipe 21 点检测结果，无需额外推理开销。

    Args:
        landmarks (Sequence): MediaPipe 返回的 21 个全图归一化关键点列表。
        img_shape (tuple): 彩色图像尺寸 ``(H, W, ...)``（像素）。

    Returns:
        tuple: ``(dist_px, p1, p2)``：
            - ``dist_px`` (float): 两指尖欧氏像素距离（px）。
            - ``p1`` (tuple[int, int]): 拇指尖整数像素坐标 ``(x1, y1)``，用于可视化。
            - ``p2`` (tuple[int, int]): 食指尖整数像素坐标 ``(x2, y2)``，用于可视化。
    """
    h, w = img_shape[:2]
    x1 = landmarks[PINCH_TIP_IDS[0]].x * w
    y1 = landmarks[PINCH_TIP_IDS[0]].y * h
    x2 = landmarks[PINCH_TIP_IDS[1]].x * w
    y2 = landmarks[PINCH_TIP_IDS[1]].y * h
    return float(np.hypot(x2 - x1, y2 - y1)), (int(x1), int(y1)), (int(x2), int(y2))


def draw_palm(display, track, desired_u, desired_v):
    """在预览图像上原地绘制期望中心十字标、掌心点及 3D 坐标文字。

    颜色约定：绿色表示本帧实检命中（``lost == 0``），橙色表示短时丢失保持状态，
    红色文字 ``Palm: LOST`` 表示轨迹完全丢失。

    Args:
        display (np.ndarray): ``(H, W, 3)`` uint8 BGR 图像（原地绘制）。
        track (dict | None): ``PalmTracker`` 维护的轨迹字典或 ``None``。
        desired_u (int): 视觉伺服期望目标中心横坐标（px）。
        desired_v (int): 视觉伺服期望目标中心纵坐标（px）。
    """
    cv2.circle(display, (desired_u, desired_v), 6, (0, 255, 0), 2)
    cv2.drawMarker(display, (desired_u, desired_v), (0, 255, 0),
                   cv2.MARKER_CROSS, 20, 1)

    if track is None:
        cv2.putText(display, "Palm: LOST", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    u, v = int(round(track['u'])), int(round(track['v']))
    color = (0, 255, 0) if track['lost'] == 0 else (0, 165, 255)
    cv2.circle(display, (u, v), 8, color, -1)
    cv2.circle(display, (u, v), 24, color, 2)

    if track['z3d'] > 0:
        label = f"X:{track['x3d']:+.3f} Y:{track['y3d']:+.3f} Z:{track['z3d']:.3f}m"
        cv2.putText(display, label, (u + 30, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# MediaPipe 21 点手部骨架拓扑连线（用于调试可视化）
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

def draw_landmarks(display, landmarks):
    """在预览图像上原地绘制 MediaPipe 21 点手部骨架连线与关节点。

    Args:
        display (np.ndarray): ``(H, W, 3)`` uint8 BGR 图像（原地绘制）。
        landmarks (Sequence): 21 个全图归一化关键点序列。
    """
    h, w = display.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(display, pts[a], pts[b], (255, 200, 0), 1)
    for p in pts:
        cv2.circle(display, p, 2, (255, 255, 255), -1)
