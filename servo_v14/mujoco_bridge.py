"""
MuJoCo 数字孪生双向零阻塞实时遥测与遥操作桥接模块（UDP Localhost Bridge, v15.0）
================================================================================
提供双通道、零阻塞、零外部依赖（仅用标准库 `socket/json/time/math`）的本机 UDP 协议：

  - 通道 A（Port 19870，真机/SIM → MuJoCo，100Hz）：
      `MuJoCoTwinPublisher` / `MuJoCoTwinReceiver`
      实时镜像真机 200Hz 实测关节角 `q_act`、指令角 `q_cmd`、速度 `dq_act`、力矩 `tau_act`、
      J7 夹爪、3D 手掌、MOS 温度、M6 故障与 v15 遥操作解锁回执。
  - 通道 B（Port 19871，MuJoCo → 真机/SIM，50-100Hz，v15.0 新增）：
      `MuJoCoTeleopCmdPublisher` / `MuJoCoTeleopCmdReceiver`
      将 MuJoCo 面板关节滑块或 3D 拖拽目标球 IK 解算出的期望关节角 `q_des`、夹爪角、
      显式解锁状态 `armed` 与限速 `max_speed_deg_s` 安全下发给真机遥操作桥接进程。

硬实时安全设计：
  - 发送端非阻塞 `SOCK_DGRAM`，任何 `OSError` 静默丢弃，绝不阻塞 100Hz/200Hz 控制环；
  - 接收端每次 `poll_latest()` 一次性排空套接字缓冲，仅保留最新帧，零队列积压；
  - 通道 B 接收端内置 NaN/Inf 过滤与 `watchdog_s = 0.25s` 失联看门狗，断连即返回 `None` 触发原地保持。
"""

from __future__ import annotations

import json
import math
import socket
import time
from typing import Any, Optional, Sequence

DEFAULT_MUJOCO_HOST = "127.0.0.1"
DEFAULT_MUJOCO_PORT = 19870          # 通道 A：真机/SIM -> MuJoCo 状态流
DEFAULT_TELEOP_CMD_PORT = 19871      # 通道 B：MuJoCo -> 真机/SIM 遥操作指令流
PROTOCOL_VERSION = 2


def _to_float_list(vec: Optional[Sequence[float]], length: int = 6) -> list[float]:
    """将任意序列/ndarray 安全转换为固定长度的有限 Python float 列表。"""
    if vec is None:
        return [0.0] * length
    out: list[float] = []
    for x in vec:
        v = float(x)
        out.append(v if math.isfinite(v) else 0.0)
        if len(out) >= length:
            break
    if len(out) < length:
        out.extend([0.0] * (length - len(out)))
    return out


def _all_finite(vec: Sequence[Any], expected_len: int) -> bool:
    """校验序列长度与每个元素的有限性，拦截 NaN/Inf 毒包。"""
    if not isinstance(vec, (list, tuple)) or len(vec) != expected_len:
        return False
    for x in vec:
        try:
            if not math.isfinite(float(x)):
                return False
        except (TypeError, ValueError):
            return False
    return True


class MuJoCoTwinPublisher:
    """通道 A 发送端（真机 / SIM -> MuJoCo，非阻塞，单帧 < 20µs）。"""

    def __init__(
        self,
        host: str = DEFAULT_MUJOCO_HOST,
        port: int = DEFAULT_MUJOCO_PORT,
        max_hz: float = 100.0,
        enabled: bool = True,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.min_interval_s = 1.0 / max(1.0, float(max_hz))
        self.enabled = bool(enabled)
        self._sock: Optional[socket.socket] = None
        self._last_send_mono: float = 0.0
        self._seq: int = 0

        if self.enabled:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setblocking(False)
                self._sock = sock
            except OSError:
                self._sock = None
                self.enabled = False

    def publish(
        self,
        *,
        q_act: Sequence[float],
        q_cmd: Sequence[float],
        dq_act: Optional[Sequence[float]] = None,
        tau_act: Optional[Sequence[float]] = None,
        gripper_deg: float = 285.2,
        p_world: Optional[Sequence[float]] = None,
        p_cam: Optional[Sequence[float]] = None,
        visible: bool = False,
        following: bool = False,
        mode: str = "IDLE",
        err_norm_m: float = 0.0,
        track_err_deg: float = 0.0,
        tmos_max: float = 25.0,
        m6_fault: bool = False,
        source: str = "hw",
        teleop_armed: bool = False,
        teleop_mode: str = "MONITOR",
        teleop_reason: str = "",
        teleop_speed_deg_s: float = 25.0,
        cmd_seq_ack: int = 0,
        force: bool = False,
    ) -> bool:
        """非阻塞发送一帧实时遥测包至 MuJoCo Studio 数字孪生；失败静默返回 False。"""
        if not self.enabled or self._sock is None:
            return False

        now_mono = time.monotonic()
        if not force and (now_mono - self._last_send_mono) < self.min_interval_s:
            return False
        self._last_send_mono = now_mono
        self._seq += 1

        payload: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "seq": self._seq,
            "ts": time.time(),
            "source": str(source),
            "q_act": [round(x, 5) for x in _to_float_list(q_act, 6)],
            "q_cmd": [round(x, 5) for x in _to_float_list(q_cmd, 6)],
            "dq_act": [round(x, 4) for x in _to_float_list(dq_act, 6)],
            "tau_act": [round(x, 3) for x in _to_float_list(tau_act, 6)],
            "gripper_deg": round(float(gripper_deg), 2),
            "p_world": (
                [round(float(x), 4) for x in p_world[:3]]
                if p_world is not None and len(p_world) >= 3
                else None
            ),
            "p_cam": (
                [round(float(x), 4) for x in p_cam[:3]]
                if p_cam is not None and len(p_cam) >= 3
                else None
            ),
            "visible": bool(visible),
            "following": bool(following),
            "mode": str(mode),
            "err_norm_m": round(float(err_norm_m), 4),
            "track_err_deg": round(float(track_err_deg), 2),
            "tmos_max": round(float(tmos_max), 1),
            "m6_fault": bool(m6_fault),
            "teleop_armed": bool(teleop_armed),
            "teleop_mode": str(teleop_mode),
            "teleop_reason": str(teleop_reason),
            "teleop_speed_deg_s": round(float(teleop_speed_deg_s), 1),
            "cmd_seq_ack": int(cmd_seq_ack),
        }

        try:
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._sock.sendto(raw, (self.host, self.port))
            return True
        except OSError:
            return False

    def close(self) -> None:
        """释放 UDP 套接字资源。"""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class MuJoCoTwinReceiver:
    """通道 A 接收端（MuJoCo Studio 侧监听真机/SIM 状态，每次排空仅留最新帧）。"""

    def __init__(
        self,
        host: str = DEFAULT_MUJOCO_HOST,
        port: int = DEFAULT_MUJOCO_PORT,
        stale_timeout_s: float = 0.6,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.stale_timeout_s = float(stale_timeout_s)
        self._sock: Optional[socket.socket] = None
        self._latest: Optional[dict[str, Any]] = None
        self._last_rx_mono: float = 0.0
        self._rx_count_window: int = 0
        self._rate_window_start: float = time.monotonic()
        self.rx_hz: float = 0.0
        self.bind_error: Optional[str] = None
        self._bind()

    def _bind(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.setblocking(False)
            self._sock = sock
            self.bind_error = None
        except OSError as exc:
            self._sock = None
            self.bind_error = str(exc)

    def poll_latest(self) -> Optional[dict[str, Any]]:
        """非阻塞排空全部待读数据报，返回最新遥测字典（超时自动置为非在线）。"""
        now = time.monotonic()
        if self._sock is not None:
            got = 0
            while True:
                try:
                    data, _ = self._sock.recvfrom(8192)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    msg = json.loads(data.decode("utf-8"))
                    if isinstance(msg, dict) and _all_finite(msg.get("q_act", []), 6):
                        self._latest = msg
                        self._last_rx_mono = now
                        got += 1
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            self._rx_count_window += got

        elapsed = now - self._rate_window_start
        if elapsed >= 0.5:
            self.rx_hz = self._rx_count_window / elapsed
            self._rx_count_window = 0
            self._rate_window_start = now

        if self._latest is not None and (now - self._last_rx_mono) <= self.stale_timeout_s:
            return self._latest
        return None

    @property
    def is_live(self) -> bool:
        """当前是否在实时接收真机或 SIM 控制流。"""
        if self._latest is None:
            return False
        return (time.monotonic() - self._last_rx_mono) <= self.stale_timeout_s

    def close(self) -> None:
        """关闭 UDP 监听套接字。"""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class MuJoCoTeleopCmdPublisher:
    """通道 B 发送端（v15.0 MuJoCo Studio -> 真机/SIM 遥操作指令发布器）。"""

    def __init__(
        self,
        host: str = DEFAULT_MUJOCO_HOST,
        port: int = DEFAULT_TELEOP_CMD_PORT,
        max_hz: float = 60.0,
        enabled: bool = True,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.min_interval_s = 1.0 / max(1.0, float(max_hz))
        self.enabled = bool(enabled)
        self._sock: Optional[socket.socket] = None
        self._last_send_mono: float = 0.0
        self._seq: int = 0

        if self.enabled:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setblocking(False)
                self._sock = sock
            except OSError:
                self._sock = None
                self.enabled = False

    def publish(
        self,
        *,
        armed: bool,
        mode: str,
        q_des: Sequence[float],
        gripper_deg: float = 285.2,
        max_speed_deg_s: float = 25.0,
        ik_target_xyz: Optional[Sequence[float]] = None,
        clear_fault: bool = False,
        force: bool = False,
    ) -> bool:
        """非阻塞下发一帧 MuJoCo 遥操作目标指令；绝不阻塞仿真渲染线程。"""
        if not self.enabled or self._sock is None:
            return False

        now_mono = time.monotonic()
        if not force and (now_mono - self._last_send_mono) < self.min_interval_s:
            return False
        self._last_send_mono = now_mono
        self._seq += 1

        q_clean = _to_float_list(q_des, 6)
        spd = float(max_speed_deg_s) if math.isfinite(float(max_speed_deg_s)) else 20.0
        spd = max(5.0, min(60.0, spd))
        g_deg = float(gripper_deg) if math.isfinite(float(gripper_deg)) else 285.2
        g_deg = max(5.2, min(285.2, g_deg))

        payload: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "seq": self._seq,
            "ts": time.time(),
            "armed": bool(armed),
            "mode": str(mode),
            "q_des": [round(x, 5) for x in q_clean],
            "gripper_deg": round(g_deg, 2),
            "max_speed_deg_s": round(spd, 1),
            "ik_target_xyz": (
                [round(float(x), 4) for x in ik_target_xyz[:3]]
                if ik_target_xyz is not None and len(ik_target_xyz) >= 3
                else None
            ),
            "clear_fault": bool(clear_fault),
        }

        try:
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._sock.sendto(raw, (self.host, self.port))
            return True
        except OSError:
            return False

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class MuJoCoTeleopCmdReceiver:
    """通道 B 接收端（v15.0 真机/SIM 侧监听 MuJoCo 遥操作指令，带 250ms 死人开关看门狗）。"""

    def __init__(
        self,
        host: str = DEFAULT_MUJOCO_HOST,
        port: int = DEFAULT_TELEOP_CMD_PORT,
        watchdog_s: float = 0.25,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.watchdog_s = max(0.05, float(watchdog_s))
        self._sock: Optional[socket.socket] = None
        self._latest: Optional[dict[str, Any]] = None
        self._last_rx_mono: float = 0.0
        self._rx_count_window: int = 0
        self._rate_window_start: float = time.monotonic()
        self.rx_hz: float = 0.0
        self.bind_error: Optional[str] = None
        self._bind()

    def _bind(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.setblocking(False)
            self._sock = sock
            self.bind_error = None
        except OSError as exc:
            self._sock = None
            self.bind_error = str(exc)

    def poll_latest(self, now_mono: Optional[float] = None) -> Optional[dict[str, Any]]:
        """非阻塞排空并校验最新遥操作包；若超过 watchdog_s 无心跳则返回 None。"""
        now = time.monotonic() if now_mono is None else float(now_mono)
        if self._sock is not None:
            got = 0
            while True:
                try:
                    data, _ = self._sock.recvfrom(8192)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    msg = json.loads(data.decode("utf-8"))
                    if (
                        isinstance(msg, dict)
                        and _all_finite(msg.get("q_des", []), 6)
                        and math.isfinite(float(msg.get("gripper_deg", 285.2)))
                        and math.isfinite(float(msg.get("max_speed_deg_s", 25.0)))
                    ):
                        self._latest = msg
                        self._last_rx_mono = now
                        got += 1
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                    continue
            self._rx_count_window += got

        elapsed = now - self._rate_window_start
        if elapsed >= 0.5:
            self.rx_hz = self._rx_count_window / elapsed
            self._rx_count_window = 0
            self._rate_window_start = now

        if self._latest is not None and (now - self._last_rx_mono) <= self.watchdog_s:
            return self._latest
        return None

    @property
    def is_alive(self) -> bool:
        """MuJoCo 遥操作心跳是否活跃（未超时）。"""
        if self._latest is None:
            return False
        return (time.monotonic() - self._last_rx_mono) <= self.watchdog_s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
