#!/usr/bin/env python3
"""低阻塞 CSV 诊断记录：控制线程只入队，后台线程批量写盘。"""

import csv
import math
import queue
import threading
import time


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
]


class DiagnosticRecorder:
    def __init__(self, path, queue_size=8192):
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
        if self._thread is not None:
            return self
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=DIAG_FIELDS,
                                     extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        self._thread = threading.Thread(target=self._run, name="diag-writer",
                                        daemon=True)
        self._thread.start()
        return self

    def log(self, row):
        if self._thread is None or self._stop.is_set():
            return
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        pending = []
        last_flush = time.monotonic()
        try:
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    pending.append(self.queue.get(timeout=0.10))
                except queue.Empty:
                    pass
                now = time.monotonic()
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
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.error = self.error or "诊断写盘线程退出超时"


def feedback_age_ms(snapshot, now, hz):
    """返回有效反馈年龄（ms）；缺失、未来、过期或异常样本统一返回 None。"""
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
    """独立低频实际关节采样；只读现有适配器，绝不进入控制律。"""

    def __init__(self, robot, hz=1.0, max_failures=3):
        self.robot = robot
        self.hz = float(hz)
        if not 0.0 < self.hz <= 2.0:
            raise ValueError("实际关节诊断采样频率须在 (0, 2] Hz")
        self.period_s = 1.0 / self.hz
        self.max_failures = max(1, int(max_failures))
        self.failures = 0
        self.last_error = ""
        self._lock = threading.Lock()
        self._snapshot = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="actual-feedback",
                                            daemon=True)
            self._thread.start()
        return self

    def sample_once(self):
        """读取实际角后立即记录内存指令角，供同点跟踪误差分析。"""
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
        import numpy as np
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError("实际关节反馈不是有限六维向量")
        return array.copy()

    def snapshot(self):
        with self._lock:
            if self._snapshot is None:
                return None
            return dict(self._snapshot)

    def _run(self):
        next_sample = time.monotonic()
        while not self._stop.is_set():
            next_sample += self.period_s
            self.sample_once()
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
