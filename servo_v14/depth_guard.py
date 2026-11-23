"""
4 层深度保护链独立模块（servo_v14/depth_guard.py）
=================================================
将 AGENTS.md 中的硬件安全红线——4 层深度保护链——从视觉主循环中解耦为独立、
可单元测试的状态机类 `DepthProtectionChain`：

    保护层 1：7×7 → 14×14 → 21×21 前景优先中值窗口回退（由调用方或 `extract_depth` 完成）
    保护层 2：帧间跳变拒收 |Δz| > max_jump_m（连续拒收超 anti_lock_frames=15 帧自动解锁重锚）
    保护层 3：长丢失 / 启动后无参照的 5 帧一致恢复校验（reacq_buf）+ 背景深度过滤
    保护层 4：单帧变化率硬限幅 ±max_rate_m + 速度自适应 One Euro 滤波

深度暂时无效时进入 `frozen=True`（最长持续 `z_freeze_max_s`），沿用最后合法深度 Z，
确保下游估计层仅更新 XY 正交子空间而不污染深度 Z。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from palm_detection import OneEuroFilter, get_median_depth
from .config import DepthGuardConfig


@dataclass(frozen=True)
class DepthGuardResult:
    """4 层深度保护链单帧输出。"""

    valid: bool                   # 是否输出了可用的深度值（含短时冻结复用）
    frozen: bool                  # True 表示本帧原始深度无效/被拒，正在复用上帧安全深度
    z_m: float                    # 保护后的掌心深度 (m)，valid=False 时为 0.0
    raw_z_m: float                # 保护层 1 提取的原始中值深度 (m)
    reject_reason: str = ""       # 触发拒收或冻结的原因标签（供诊断）


class DepthProtectionChain:
    """严格实现 4 层深度保护链的有状态滤波器。"""

    def __init__(self, cfg: DepthGuardConfig, target_distance_m: float = 0.40) -> None:
        """
        初始化 4 层深度保护链。

        Args:
            cfg: 深度保护参数（`DepthGuardConfig`）。
            target_distance_m: 期望跟随距离 (m)，用于第 3 层背景深度过滤上限。
        """
        self.cfg = cfg
        self.target_distance_m = float(target_distance_m)
        self._oe = OneEuroFilter(min_cutoff=cfg.oe_min_cutoff, beta=cfg.oe_beta)
        self._z_ref: float = 0.0
        self._z_filt: float = 0.0
        self._jump_reject_count: int = 0
        self._reacq_buf: List[float] = []
        self._freeze_start_s: Optional[float] = None

    def reset(self) -> None:
        """长时丢手、停止跟随或显式复位时清空全部历史深度状态。"""
        self._oe.reset()
        self._z_ref = 0.0
        self._z_filt = 0.0
        self._jump_reject_count = 0
        self._reacq_buf.clear()
        self._freeze_start_s = None

    def extract_raw_depth(
        self,
        depth_img: Optional[np.ndarray],
        u: int,
        v: int,
        depth_scale: float,
    ) -> float:
        """
        保护层 1：三级扩展窗口（7×7 → 13×13 → 19×19）+ 前景连续优先中值深度提取。

        Args:
            depth_img: 硬件对齐后的深度图（uint16 / float）。
            u: 掌心像素横坐标。
            v: 掌心像素纵坐标。
            depth_scale: 深度值到米 (m) 的换算系数。

        Returns:
            有效深度 (m)，若三级窗口均无有效像素返回 0.0。
        """
        if depth_img is None:
            return 0.0
        z = float(get_median_depth(
            depth_img,
            float(depth_scale),
            int(u),
            int(v),
            ref_depth=self._z_ref,
            max_jump=self.cfg.max_jump_m,
        ))
        return z if (self.cfg.min_valid_m <= z <= self.cfg.max_valid_m) else 0.0

    def update(self, raw_z_m: float, timestamp_s: float) -> DepthGuardResult:
        """
        对保护层 1 输出的原始深度依次执行第 2、3、4 层保护。

        Args:
            raw_z_m: 保护层 1 测得的原始深度 (m)，无效时传 0.0。
            timestamp_s: 当前帧单调时间戳 (s)。

        Returns:
            `DepthGuardResult`，指示深度是否有效、是否处于冻结态以及滤波后深度。
        """
        z_raw = float(raw_z_m) if math.isfinite(raw_z_m) else 0.0
        in_range = self.cfg.min_valid_m <= z_raw <= self.cfg.max_valid_m

        # ── 原始深度有效分支 ──────────────────────────────────────
        if in_range:
            # 情况 A：已有历史参考深度，走保护层 2（跳变拒收 + 15 帧防锁死）
            if self._z_ref > 0.0:
                delta = abs(z_raw - self._z_ref)
                if delta > self.cfg.max_jump_m:
                    self._jump_reject_count += 1
                    if self._jump_reject_count < self.cfg.anti_lock_frames:
                        return self._handle_freeze(z_raw, timestamp_s, reason="jump_reject")
                    # 连续 15 帧均处于新深度，说明手掌真实大幅移动，防锁死解锁重锚
                    self._jump_reject_count = 0
                    self._oe.reset()
                    self._z_ref = z_raw
                    self._z_filt = self._oe(z_raw, timestamp_s)
                    self._freeze_start_s = None
                    return DepthGuardResult(
                        valid=True,
                        frozen=False,
                        z_m=self._z_filt,
                        raw_z_m=z_raw,
                        reject_reason="anti_lock_reanchor",
                    )

                # 正常小步长深度更新 → 保护层 4（单帧变化率限幅 + One Euro）
                self._jump_reject_count = 0
                self._freeze_start_s = None
                z_clamped = float(np.clip(
                    z_raw,
                    self._z_ref - self.cfg.max_rate_m,
                    self._z_ref + self.cfg.max_rate_m,
                ))
                self._z_ref = z_clamped
                self._z_filt = self._oe(z_clamped, timestamp_s)
                return DepthGuardResult(
                    valid=True,
                    frozen=False,
                    z_m=self._z_filt,
                    raw_z_m=z_raw,
                )

            # 情况 B：无历史参考（首次启动或长丢失后恢复），走保护层 3（背景过滤 + 5 帧一致）
            bg_cap = self.target_distance_m + self.cfg.reacq_bg_margin_m
            if z_raw > bg_cap:
                self._reacq_buf.clear()
                return DepthGuardResult(
                    valid=False,
                    frozen=False,
                    z_m=0.0,
                    raw_z_m=z_raw,
                    reject_reason="bg_depth_reject",
                )

            self._reacq_buf.append(z_raw)
            if len(self._reacq_buf) > self.cfg.reacq_frames:
                self._reacq_buf.pop(0)
            if len(self._reacq_buf) < self.cfg.reacq_frames:
                return DepthGuardResult(
                    valid=False,
                    frozen=False,
                    z_m=0.0,
                    raw_z_m=z_raw,
                    reject_reason="reacq_collecting",
                )
            if (max(self._reacq_buf) - min(self._reacq_buf)) > self.cfg.reacq_tol_m:
                # 窗口内极差过大，滑出最早一帧等待稳定
                self._reacq_buf.pop(0)
                return DepthGuardResult(
                    valid=False,
                    frozen=False,
                    z_m=0.0,
                    raw_z_m=z_raw,
                    reject_reason="reacq_inconsistent",
                )

            # 连续 N 帧一致通过！以中值作为初始安全参考
            z_init = float(np.median(self._reacq_buf))
            self._reacq_buf.clear()
            self._jump_reject_count = 0
            self._freeze_start_s = None
            self._oe.reset()
            self._z_ref = z_init
            self._z_filt = self._oe(z_init, timestamp_s)
            return DepthGuardResult(
                valid=True,
                frozen=False,
                z_m=self._z_filt,
                raw_z_m=z_raw,
            )

        # ── 原始深度无效分支：尝试短时冻结复用，超时则清零 ────────────
        return self._handle_freeze(z_raw, timestamp_s, reason="invalid_depth")

    def _handle_freeze(
        self,
        z_raw: float,
        timestamp_s: float,
        reason: str,
    ) -> DepthGuardResult:
        """在深度空洞或跳变拒收期间维持最多 `z_freeze_max_s` 的安全冻结态。"""
        if self._z_filt <= 0.0:
            return DepthGuardResult(
                valid=False,
                frozen=False,
                z_m=0.0,
                raw_z_m=z_raw,
                reject_reason=reason,
            )
        if self._freeze_start_s is None:
            self._freeze_start_s = timestamp_s
        if (timestamp_s - self._freeze_start_s) <= self.cfg.z_freeze_max_s:
            return DepthGuardResult(
                valid=True,
                frozen=True,
                z_m=self._z_filt,
                raw_z_m=z_raw,
                reject_reason=reason,
            )
        # 冻结超时，清空参考防止沿用过期深度，重新走保护层 3 恢复
        self.reset()
        return DepthGuardResult(
            valid=False,
            frozen=False,
            z_m=0.0,
            raw_z_m=z_raw,
            reject_reason="freeze_timeout",
        )
