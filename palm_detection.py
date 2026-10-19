#!/usr/bin/env python3
"""
手掌检测模块（替代原项目的 blue_circle_detection.py）
基于 MediaPipe HandLandmarker，输出掌心点的 2D 像素 + 3D 相机系坐标。

框架完全沿用原项目的设计：
    - 跨帧追踪（短暂丢失保持，最多 MAX_LOST_FRAMES 帧）
    - 指数平滑消除抖动
    - 深度取邻域中值，避免单点噪声

依赖：
    pip install mediapipe opencv-contrib-python numpy
    并下载模型文件 hand_landmarker.task：
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

# ── 配置 ──────────────────────────────────────────────────
PALM_LANDMARK_ID = 9        # 中指根部，作为掌心代表点
PINCH_TIP_IDS = (4, 8)      # 捏合手势：拇指尖(4)—食指尖(8)，用于夹爪开合控制
MIN_DEPTH_M = 0.15          # Gemini 2 近距极限，小于此值的深度丢弃
MAX_DEPTH_M = 1.50          # 跟随场景用不到太远
DEPTH_PATCH = 3             # 深度采样半窗口（实际 7x7）

MAX_LOST_FRAMES = 8         # 与原项目一致：丢失 8 帧内保持上一帧
SMOOTH_ALPHA = 0.4          # 指数平滑系数（越小越平滑）
NUM_HANDS = 1               # 只跟一只手，避免左右手跳变

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.4   # 调低：运动模糊帧也能勉强跟上，减少快挥掉检

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config", "hand_landmarker.task")


class PalmDetector:
    """MediaPipe 手掌检测器（VIDEO 模式，利用跨帧跟踪更稳）

    backend: "auto"（优先 GPU，失败回退 CPU）/ "gpu" / "cpu"
    GPU delegate 依赖 OpenGL ES/EGL，桌面独显 Linux 一般可用；
    推理从 CPU XNNPACK ~10ms 降到 GPU ~3-5ms。
    """

    def __init__(self, backend="auto"):
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
        print(f"[检测器] 推理后端: {self.backend.upper()}")
        self._timestamp_ms = 0

    def _create(self, delegate):
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
        """
        返回掌心像素坐标 (u, v) 和全部关键点；未检测到返回 (None, None, None)
        """
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO 模式要求时间戳单调递增；用真实时钟比固定 +33ms 更利于内部追踪
        self._timestamp_ms = int(time.monotonic() * 1000)
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return None, None, None

        landmarks = result.hand_landmarks[0]
        h, w = color_bgr.shape[:2]
        lm = landmarks[PALM_LANDMARK_ID]
        u, v = int(lm.x * w), int(lm.y * h)
        return u, v, landmarks

    def close(self):
        self.detector.close()


class OneEuroFilter:
    """One Euro Filter（Casiez et al. 2012）：静止时强滤波、运动时自动减弱滤波。

    相比固定系数 EMA，解决了「平滑则滞后、灵敏则抖动」的两难：
    手静止时 cutoff 低（稳），手快速移动时 cutoff 随速度升高（跟得上）。
    是体感跟随/手势交互的标准做法。
    """

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)   # Hz，越小静止时越稳
        self.beta = float(beta)               # 速度系数，越大高速滞后越小
        self.d_cutoff = float(d_cutoff)       # 速度估计的滤波截止
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        if t is None:
            t = time.monotonic()
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = float(x)
            return self.x_prev
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev += a_d * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, dt)
        self.x_prev += a * (x - self.x_prev)
        return self.x_prev


# 掌心滤波器选择：one_euro（推荐，高速跟随滞后小）或 ema（旧行为）
PALM_FILTER = "one_euro"
OE_MIN_CUTOFF = 1.0         # 静止稳定性：越小越稳
OE_BETA = 0.02              # 高速响应：越大越快（太大会把噪声当运动）


class LKPalmBridge:
    """检测失败帧的光流桥接（检测+跟踪混合架构）。

    MediaPipe 因运动模糊整帧掉检时，用 Lucas-Kanade 金字塔光流追踪
    掌心周围 3×3 点阵，产出估计掌心位置继续喂 PalmTracker——
    短暂掉检不再触发丢失保持（快速挥手丢手的主要兜底）。
    只桥接、不替代检测，三道保险：
      1) 检测成功帧立刻用真实掌心重置点阵，桥接永不带偏目标；
      2) 连续桥接超 max_frames（默认 15 帧≈0.25s）自动放弃，
         回退原有丢失流程（防漂移累积）；
      3) 单帧位移 > max_step_px 或有效跟踪点 <5 → 放弃本帧。
    纯 cv2 主库实现（calcOpticalFlowPyrLK），不依赖 opencv-contrib。
    桥接帧无 MediaPipe 关键点 → 捏合门控自然冻结夹爪，无风险。
    """

    def __init__(self, enable=True, max_frames=15, box_half=45,
                 max_step_px=90.0):
        self.enable = bool(enable)
        self.max_frames = int(max_frames)
        self.box_half = int(box_half)
        self.max_step_px = float(max_step_px)
        self._prev_gray = None
        self._pts = None          # 上一帧 3×3 点阵 (N,1,2) float32
        self._tracked = 0         # 已连续桥接帧数

    def reset(self):
        self._prev_gray = None
        self._pts = None
        self._tracked = 0

    def _seed(self, gray, uv):
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
        """det_uv: 本帧 MediaPipe 检测 (u,v) 或 None。

        返回 (u, v, source)，source ∈ {'det', 'lk', None}；
        'lk' = 本帧来自光流桥接（无关键点）。
        """
        if not self.enable:
            return (*det_uv, 'det') if det_uv is not None else (None, None, None)
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        out_u, out_v, src = None, None, None
        if det_uv is not None:
            self._seed(gray, det_uv)      # 真实掌心重置点阵
            out_u, out_v, src = float(det_uv[0]), float(det_uv[1]), 'det'
            self._tracked = 0
        elif self._pts is not None and self._tracked < self.max_frames \
                and self._prev_gray is not None:
            pts2, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._pts, None,
                winSize=(25, 25), maxLevel=2)
            if pts2 is not None:
                st = st.reshape(-1).astype(bool)
                good_new = pts2.reshape(-1, 2)[st]
                good_old = self._pts.reshape(-1, 2)[st]
                if good_new.shape[0] >= 5:
                    step = float(np.median(
                        np.linalg.norm(good_new - good_old, axis=1)))
                    if step <= self.max_step_px:
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
    """掌心点跨帧追踪器（对应原项目 CircleTracker，单目标版）"""

    def __init__(self):
        self.track = None    # {'u','v','lost','x3d','y3d','z3d'}
        self._fu = OneEuroFilter(OE_MIN_CUTOFF, OE_BETA)
        self._fv = OneEuroFilter(OE_MIN_CUTOFF, OE_BETA)

    def update(self, detection):
        """
        detection: (u, v) 或 None
        返回当前轨迹（可能处于 lost 保持状态），完全丢失返回 None
        """
        if detection is not None:
            u, v = detection
            if self.track is None:
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
                self.track['lost'] += 1
                if self.track['lost'] > MAX_LOST_FRAMES:
                    self.track = None

        return self.track


def get_median_depth(depth_raw, depth_scale, u, v, patch=DEPTH_PATCH):
    """
    在 (u,v) 邻域取中值深度（米）。
    depth_raw: uint16 深度图（已对齐到彩色）
    无效/超量程返回 0.0

    BS 高帧率模式下掌心小窗口常出现深度空洞（实测表现为 palm 可见
    但 z3d=0，手臂顿挫）：采样窗口逐级扩大 patch → 2x → 3x，
    只有最大窗口仍无效才放弃本帧。
    """
    h, w = depth_raw.shape
    for p in (patch, patch * 2, patch * 3):
        x0, x1 = max(0, u - p), min(w, u + p + 1)
        y0, y1 = max(0, v - p), min(h, v + p + 1)
        window = depth_raw[y0:y1, x0:x1].astype(np.float32) * depth_scale
        valid = window[(window >= MIN_DEPTH_M) & (window <= MAX_DEPTH_M)]
        if valid.size >= 5:     # 有效点太少说明在边缘/遮挡处，不可信
            return float(np.median(valid))
    return 0.0


def pixel_to_3d(u, v, depth_m, fx, fy, cx, cy):
    """相机内参反投影（与原项目 pixel_to_3d 等价，不依赖 realsense 接口）"""
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return X, Y, Z


def pinch_distance_px(landmarks, img_shape):
    """拇指尖(4)—食指尖(8) 的像素距离（捏合手势 → 夹爪开合映射用）。

    与掌心跟随共用同一次 MediaPipe 检测结果，无需再跑第二个检测器。

    Returns:
        dist_px: 两指尖像素距离
        p1, p2:  拇指尖 / 食指尖的像素坐标 (x, y)，用于可视化
    """
    h, w = img_shape[:2]
    x1 = landmarks[PINCH_TIP_IDS[0]].x * w
    y1 = landmarks[PINCH_TIP_IDS[0]].y * h
    x2 = landmarks[PINCH_TIP_IDS[1]].x * w
    y2 = landmarks[PINCH_TIP_IDS[1]].y * h
    return float(np.hypot(x2 - x1, y2 - y1)), (int(x1), int(y1)), (int(x2), int(y2))


def draw_palm(display, track, desired_u, desired_v):
    """可视化：掌心点、期望中心点、3D 坐标"""
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


# 手部关键点连线（用于可视化，可选）
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

def draw_landmarks(display, landmarks):
    h, w = display.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(display, pts[a], pts[b], (255, 200, 0), 1)
    for p in pts:
        cv2.circle(display, p, 2, (255, 255, 255), -1)
