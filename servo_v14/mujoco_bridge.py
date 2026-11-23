"""
MuJoCo 数字孪生零阻塞实时遥测桥接模块（UDP Localhost Bridge）
================================================================
为实现“真实 reBot Arm B601-RS 机械臂与 MuJoCo 虚拟数字孪生同步实时运动”，
本模块提供完全解耦、零阻塞、零外部依赖（仅用标准库 `socket/json/time`）的
本机 UDP 遥测发送端与接收端：

  - 进程隔离：真机控制运行在 `rebot-cv`（含 Pinocchio/Orbbec/CAN SDK），
    MuJoCo Studio 3D 渲染运行在独立 venv（`mujoco>=4.0.1`），二者互不干扰。
  - 硬实时安全：发送端采用非阻塞 `SOCK_DGRAM` + 频率节流（默认 100Hz），
    任何 `OSError` / 缓冲满均静默丢弃，绝不阻塞 100Hz 伺服或 200Hz MIT 控制线程。
  - 零积压低延迟：接收端每次 `poll_latest()` 一次性排空套接字缓冲区，仅保留最新一帧，
    彻底杜绝队列堆积导致的视觉拖尾与延迟漂移。
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Optional, Sequence

DEFAULT_MUJOCO_HOST = "127.0.0.1"
DEFAULT_MUJOCO_PORT = 19870
PROTOCOL_VERSION = 1


def _to_float_list(vec: Optional[Sequence[float]], length: int = 6) -> list[float]:
    """将任意序列/ndarray 安全转换为固定长度的 Python float 列表。"""
    if vec is None:
        return [0.0] * length
    out = [float(x) for x in vec]
    if len(out) < length:
        out.extend([0.0] * (length - len(out)))
    return out[:length]


class MuJoCoTwinPublisher:
    """真机 / SIM 控制端非阻塞 UDP 遥测发布器（单帧 < 20µs，绝不阻塞伺服环）。"""

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
    """MuJoCo Studio 侧非阻塞 UDP 接收器：每次排空缓冲，仅保留最新帧。"""

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
                    if isinstance(msg, dict) and "q_act" in msg:
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
