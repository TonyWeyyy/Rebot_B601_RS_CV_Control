#!/usr/bin/env python3
"""
低阻塞伺服 CSV 诊断记录与独立低频实际关节反馈采样模块
======================================================
面向 100Hz 实时视觉伺服热循环设计，严格遵循「诊断绝不阻塞控制、诊断故障绝不波及机械臂」原则：

1. ``DiagnosticRecorder``（生产者-消费者非阻塞写盘）：
   - 100Hz 伺服控制线程仅调用 ``log(row)`` 内的 ``queue.put_nowait(row)``，队列满时计数丢弃，
     绝不执行任何磁盘 I/O 或阻塞等待；
   - 后台守护线程 ``diag-writer`` 按批量（128 行 / 0.25s）攒批落盘并 ``flush``，异常全部捕获在
     ``self.error`` 内，防止写盘故障中断机械臂控制。
2. ``DIAG_FIELDS``（全量 CSV 列定义）：
   - 覆盖视觉观测时序、伺服状态机、卡尔曼速度/外推、6 关节指令角、低频同步反馈，以及
     v13 P0/P3 零开销 200Hz MIT 应答内存快照（``qa/dqa/tau/tmos/tau_ff/kd``）。
3. ``ActualFeedbackSampler``（独立低频同步 CAN 采样，仅限诊断）：
   - 硬限制采样率在 ``(0, 2.0] Hz``，避免串行同步 CAN 读挤占 200Hz MIT 下发线程；
   - 线程安全快照读取、异常时立即清空失效快照防脏读、连续故障达 ``max_failures`` 后自动熔断停用。
"""

import csv
import math
import queue
import threading
import time


# ── 100Hz 伺服诊断 CSV 固定表头（供 tools/analyze_servo_log.py 离线解析）──
# 字段分组与单位约定：
#   [时序与状态] host_time(s), dt_s(s), seq(帧序号), new_obs/following/visible/frozen(0/1),
#               kf_valid/ff_moving/hold_latch/slow_active/static_hold(0/1)
#   [延迟与耗时] lead_weight(0~1), lead_raw_s/lead_s(s), obs_age_ms/processing_ms/ctrl_work_ms(ms), overrun(0/1)
#   [误差与估计] error_norm(m), target_x/y/z(m), kv_x/y/z(m/s), joint_speed(rad/s),
#               q_cmd_1..6(rad), kf_q_scale_x/y/z(倍率), ff_taper_now(0~1), ff_approach_speed(m/s), ff_scale(0~1)
#   [低频同步反馈] fb_valid(0/1), fb_time(s), fb_age_ms/fb_read_ms(ms), tracking_err_max(rad),
#               q_act_1..6(rad), q_fb_cmd_1..6(rad), hw_time_used(0/1), error(文本)
#   [v13 P0 200Hz MIT 内存缓存] fbc_valid(0/1), fbc_age_ms(ms), fbc_seq(拍号),
#               qa_1..6(rad), dqa_1..6(rad/s), tau_1..6(Nm), tmos_max(°C),
#               mit_period_p99_ms(ms), can_fail_total(次), mit_dq_ff_max(rad/s)
#   [v13 P3 重力前馈与阻尼] tau_ff_2..4(Nm), kd_j1(Nm·s/rad)
DIAG_FIELDS = [
    "host_time", "dt_s", "seq", "new_obs", "following", "visible", "frozen",
    "kf_valid", "ff_moving", "hold_latch", "slow_active", "static_hold",
    "lead_weight",
    "lead_raw_s", "lead_s", "obs_age_ms", "processing_ms", "ctrl_work_ms",
    "overrun", "error_norm", "target_x", "target_y", "target_z",
    "kv_x", "kv_y", "kv_z", "joint_speed", "q_cmd_1", "q_cmd_2", "q_cmd_3",
    "q_cmd_4", "q_cmd_5", "q_cmd_6", "kf_q_scale_x", "kf_q_scale_y",
    "kf_q_scale_z", "ff_taper_now", "ff_approach_speed", "ff_scale",
    "fb_valid", "fb_time", "fb_age_ms", "fb_read_ms", "tracking_err_max",
    "q_act_1", "q_act_2", "q_act_3", "q_act_4", "q_act_5", "q_act_6",
    "q_fb_cmd_1", "q_fb_cmd_2", "q_fb_cmd_3", "q_fb_cmd_4", "q_fb_cmd_5",
    "q_fb_cmd_6", "hw_time_used", "error",
    # v13 P0：MIT 扩展 200Hz 应答缓存（内存读取，无 CAN I/O）
    "fbc_valid", "fbc_age_ms", "fbc_seq",
    "qa_1", "qa_2", "qa_3", "qa_4", "qa_5", "qa_6",
    "dqa_1", "dqa_2", "dqa_3", "dqa_4", "dqa_5", "dqa_6",
    "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6",
    "tmos_max", "mit_period_p99_ms", "can_fail_total", "mit_dq_ff_max",
    # v13 P3：静态重力前馈力矩与伸展调度阻尼诊断
    "tau_ff_2", "tau_ff_3", "tau_ff_4", "kd_j1",
]


class DiagnosticRecorder:
    """非阻塞 CSV 诊断记录器：控制线程 ``put_nowait`` 入队，后台线程攒批写盘。

    核心属性：
        path (str): 输出 CSV 文件路径。
        queue (queue.Queue): 有界内存缓冲区，满队列时丢弃新行而不阻塞控制线程。
        dropped (int): 因队列满而被丢弃的记录行数。
        written (int): 已成功写入 CSV 的总行数。
        error (str): 后台写盘线程捕获的异常文本（空字符串表示正常）。
    """

    def __init__(self, path, queue_size=8192):
        """初始化记录器（尚未打开文件或启动写盘线程，需调用 ``start()``）。

        Args:
            path (str | os.PathLike): CSV 输出文件路径。
            queue_size (int): 有界队列最大容量（行），至少钳位为 128；
                默认 8192 行在 100Hz 伺服下可提供约 81.9 秒的磁盘抖动缓冲。
        """
        self.path = str(path)
        self.queue = queue.Queue(maxsize=max(128, int(queue_size)))
        self.dropped = 0
        self.written = 0
        self.error = ""
        self._stop = threading.Event()
        self._thread = None
        self._file = None
        self._writer = None

    def start(self):
        """打开 CSV 文件、写入表头并启动后台写盘守护线程 ``diag-writer``（幂等）。

        Returns:
            DiagnosticRecorder: 返回 ``self`` 便于链式调用。
        """
        if self._thread is not None:
            return self
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        # extrasaction="ignore" 允许调用方按版本携带额外调试键而不破坏 CSV 列对齐
        self._writer = csv.DictWriter(self._file, fieldnames=DIAG_FIELDS,
                                     extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        self._thread = threading.Thread(target=self._run, name="diag-writer",
                                        daemon=True)
        self._thread.start()
        return self

    def log(self, row):
        """在控制线程热路径中非阻塞投递一行诊断字典。

        若记录器尚未启动或已请求关闭，直接静默忽略；若内存队列已满，立即丢弃本行并
        递增 ``self.dropped``，确保绝不阻塞 100Hz 实时伺服周期。

        Args:
            row (dict): 键名对应 ``DIAG_FIELDS`` 的单周期诊断数据字典。
        """
        if self._thread is None or self._stop.is_set():
            return
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        """后台写盘主循环：攒批写盘并隔离所有 I/O 异常，退出前排空剩余队列。"""
        pending = []
        last_flush = time.monotonic()
        try:
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    pending.append(self.queue.get(timeout=0.10))
                except queue.Empty:
                    pass
                now = time.monotonic()
                # 满足「累计 128 行 / 距上次刷盘 ≥0.25s / 收到退出信号」任一条件即批量落盘
                if pending and (len(pending) >= 128 or now - last_flush >= 0.25
                                or self._stop.is_set()):
                    self._writer.writerows(pending)
                    self.written += len(pending)
                    pending.clear()
                    self._file.flush()
                    last_flush = now
        except Exception as exc:  # 日志故障不得进入机械臂控制线程
            self.error = str(exc)
        finally:
            try:
                if pending:
                    self._writer.writerows(pending)
                    self.written += len(pending)
                self._file.flush()
                self._file.close()
            except Exception as exc:
                self.error = self.error or str(exc)

    def close(self):
        """通知写盘线程排空队列并等待退出（最多等待 5.0 秒）。

        应在安全停机链中、停止控制线程之后调用，确保所有已入队诊断行落盘完整。
        """
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.error = self.error or "诊断写盘线程退出超时"


def feedback_age_ms(snapshot, now, hz):
    """计算低频反馈快照的年龄（毫秒）并校验其时效与合法性。

    校验规则：
      - 采样率 ``hz`` 必须处于 ``(0, 2.0]`` Hz；
      - ``age_s = now - snapshot["time"]`` 必须有限、非负，且不超过过期阈值
        ``max(1.0, 3.0 / hz)`` 秒（允许最多连续缺失约 3 个采样周期）。

    Args:
        snapshot (dict | None): ``ActualFeedbackSampler.snapshot()`` 返回的快照字典，
            至少包含 ``"time"``（单调时钟秒）。
        now (float): 当前单调时钟时间戳（秒，``time.monotonic()``）。
        hz (float): 采样器配置的采样频率（Hz）。

    Returns:
        float | None: 合法快照的年龄（毫秒）；缺失、时钟倒流、超时过期或字段异常统一返回 ``None``。
    """
    try:
        rate = float(hz)
        finished = float(snapshot["time"])
        age_s = float(now) - finished
        if (not 0.0 < rate <= 2.0 or not math.isfinite(age_s)
                or age_s < 0.0 or age_s > max(1.0, 3.0 / rate)):
            return None
        return age_s * 1000.0
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


class ActualFeedbackSampler:
    """独立低频实际关节采样线程（旧版同步 CAN 诊断通道，绝不进入控制律）。

    通过独立线程以不超过 2.0Hz 的频率调用 ``robot.get_current_pos()`` 与
    ``robot.get_command_pos()`` 获取同点实际/指令角对，用于对比跟踪误差；
    所有快照读写均受 ``self._lock`` 保护，且在连续读取失败达到 ``max_failures`` 次后
    自动熔断自身线程，确保诊断异常绝不拖垮伺服控制与 CAN 总线。
    """

    def __init__(self, robot, hz=1.0, max_failures=3):
        """初始化低频实际关节采样器。

        Args:
            robot: 机械臂适配器实例（需提供 ``get_current_pos()`` 与 ``get_command_pos()``）。
            hz (float): 同步 CAN 采样频率（Hz），硬性限制在 ``(0.0, 2.0]`` 区间，默认 1.0Hz。
            max_failures (int): 连续读取失败触发永久熔断的阈值（次），至少为 1，默认 3。

        Raises:
            ValueError: ``hz`` 不在 ``(0, 2]`` Hz 范围内时抛出。
        """
        self.robot = robot
        self.hz = float(hz)
        if not 0.0 < self.hz <= 2.0:
            raise ValueError("实际关节诊断采样频率须在 (0, 2] Hz")
        self.period_s = 1.0 / self.hz
        self.max_failures = max(1, int(max_failures))
        self.failures = 0          # 当前连续失败次数（一次成功即清零）
        self.last_error = ""       # 最近一次采样异常信息
        self._lock = threading.Lock()
        self._snapshot = None      # 最新有效快照字典，失败后立即置 None 防旧样本误用
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        """启动后台低频采样守护线程 ``actual-feedback``（幂等）。

        Returns:
            ActualFeedbackSampler: 返回 ``self`` 便于链式调用。
        """
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="actual-feedback",
                                            daemon=True)
            self._thread.start()
        return self

    def sample_once(self):
        """执行单次同步采样并线程安全地更新快照（异常时清空快照并累计熔断计数）。

        先执行阻塞式 CAN 读取 ``get_current_pos()``，完成后立即读取纯内存指令角
        ``get_command_pos()``，以尽量缩小两者之间的时间对齐偏差，便于计算同点跟踪误差。

        Returns:
            bool: 本次采样有效且更新成功返回 ``True``；发生异常返回 ``False``。
        """
        started = time.monotonic()
        try:
            actual = self.robot.get_current_pos()
            command = self.robot.get_command_pos()
            finished = time.monotonic()
            actual = self._validated(actual)
            command = self._validated(command)
            snapshot = {
                "time": finished, "read_ms": (finished - started) * 1000.0,
                "actual": actual, "command": command,
            }
            with self._lock:
                self._snapshot = snapshot
                self.failures = 0
                self.last_error = ""
            return True
        except Exception as exc:
            with self._lock:
                # 读取异常后不再把旧样本标成有效；诊断空缺绝不影响控制。
                self._snapshot = None
                self.failures += 1
                self.last_error = str(exc)
                failures = self.failures
            if failures >= self.max_failures:
                self._stop.set()  # 连续故障后永久停用诊断，不影响伺服线程
            return False

    @staticmethod
    def _validated(values):
        """校验关节角反馈为有限 6 维 float64 向量并返回独立副本（单位：rad）。

        Args:
            values (array-like): 待校验的 6 关节角度序列。

        Returns:
            np.ndarray: 形状为 ``(6,)`` 的有限 float64 数组副本。

        Raises:
            ValueError: 形状不是 ``(6,)`` 或包含 NaN/Inf 时抛出。
        """
        import numpy as np
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError("实际关节反馈不是有限六维向量")
        return array.copy()

    def snapshot(self):
        """线程安全地获取最新一次有效采样快照的浅拷贝。

        Returns:
            dict | None: 包含 ``time``（s）、``read_ms``（ms）、``actual``（6 维 rad）、
            ``command``（6 维 rad）的字典副本；无有效样本或刚发生异常时返回 ``None``。
        """
        with self._lock:
            if self._snapshot is None:
                return None
            return dict(self._snapshot)

    def _run(self):
        """后台定频采样循环：按固定周期触发 ``sample_once``，通过 ``Event.wait`` 响应关闭。"""
        next_sample = time.monotonic()
        while not self._stop.is_set():
            next_sample += self.period_s
            self.sample_once()
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def close(self):
        """停止采样线程并等待退出（最多 1.0 秒）。

        安全停机红线：必须在调用 ``robot.stop()`` 之前执行，避免回零/失能期间并发 CAN 读。
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
