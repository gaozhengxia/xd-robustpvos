"""Optical flow utilities and mask warping.

We use Farneback optical flow throughout: it is deterministic, dependency-free
(OpenCV only), and runs on CPU, which keeps the "training-free / weight-free"
claim of DAG-RS intact. Flow is computed between consecutive *degraded* frames
(the same observation the segmenter sees), and chained to propagate a mask from
the last accepted anchor to an arbitrary later frame.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


def to_gray(img: np.ndarray) -> np.ndarray:
    """8-bit grayscale, accepting either float [0,1] or uint8 [0,255] input.

    The dtype check is not cosmetic. This function used to scale blindly with
    `np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)`, which is correct for
    the float32 frames the pipeline uses but turns a **uint8** frame into a
    constant 255 image. Farneback then sees a flat field and returns an
    identically zero flow -- silently, with no error. That is exactly what
    happened to the first run of `scripts/_flow_probe2.py`, which fed
    `Dataset.frame()` (uint8) straight in and reported "optical flow is
    useless". Accepting both dtypes removes the trap.
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        u8 = a
    else:
        u8 = np.clip(a.astype(np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if u8.ndim == 3:
        return cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    return u8


def compute_flow(prev: np.ndarray, cur: np.ndarray,
                 levels: int = 4, winsize: int = 21,
                 iterations: int = 3, poly_n: int = 5,
                 poly_sigma: float = 1.2) -> np.ndarray:
    """Dense flow from `prev` to `cur`, shape (H, W, 2) in pixels."""
    p = to_gray(prev)
    c = to_gray(cur)
    f = cv2.calcOpticalFlowFarneback(p, c, None,
                                     pyr_scale=0.5, levels=int(levels),
                                     winsize=int(winsize), iterations=int(iterations),
                                     poly_n=int(poly_n), poly_sigma=float(poly_sigma),
                                     flags=0)
    return f.astype(np.float32)


def warp_with_flow(src: np.ndarray, flow: np.ndarray,
                   interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Warp `src` from its own time step to the next one along `flow`.

    Convention (verified against real data, see the note below): `flow` from
    `compute_flow(prev, cur)` satisfies `prev(x) ~= cur(x + flow(x))`, i.e. it
    is the forward displacement of each pixel of `prev`. `cv2.remap` gathers, so
    the forward warp is `remap(prev, xs - flow)`.

    This previously used `xs + flow`, which is the recipe for the *opposite*
    convention. On 48 frame pairs drawn from 16 DAVIS val sequences the wrong
    sign scored a mean IoU of 0.5166 against the next ground truth versus 0.6282
    for not warping at all and 0.8054 for the correct sign -- i.e. propagation
    was actively pushing the anchor off the object, and with it the whole
    baseline. The sign was wrong in all 48 pairs with no ties; re-check before
    changing it. `scripts/00_smoke_test.py::test_flow` now pins it.
    """
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = xs - flow[..., 0]
    map_y = ys - flow[..., 1]
    if src.ndim == 2:
        return cv2.remap(src.astype(np.float32), map_x, map_y,
                         interpolation, borderMode=cv2.BORDER_REPLICATE) > 0.5
    return cv2.remap(src.astype(np.float32), map_x, map_y,
                     interpolation, borderMode=cv2.BORDER_REPLICATE)


def warp_soft(src: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a continuous map (e.g. a mask probability in [0, 1]) bilinearly.

    `warp_with_flow` thresholds its 2-D input at 0.5, so it cannot carry a soft
    mask. Soft carries matter because the re-prompt quality depends on them --
    see `evidence.soft_mask_to_logits`.
    """
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(np.asarray(src, np.float32),
                     xs - flow[..., 0], ys - flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FlowCache:
    """Lazily computes and caches consecutive-frame flows for one sequence."""

    def __init__(self, frames: List[np.ndarray]):
        self.frames = frames
        self.n = len(frames)
        self._flow: Dict[int, np.ndarray] = {}     # key t -> flow from t-1 to t

    def flow_between(self, t_from: int, t_to: Optional[int] = None) -> np.ndarray:
        """Flow from frame `t_from` to frame `t_from + 1` (a single step).

        `t_to` is accepted only for call-site clarity and must be `t_from + 1`
        when given; single-step flows are the only ones we cache.
        """
        if t_to is not None and int(t_to) != int(t_from) + 1:
            raise ValueError(
                f"flow_between is single-step: expected t_to={int(t_from) + 1}, "
                f"got {int(t_to)}. Use warp_mask(m, t_from, t_to) for multi-step.")
        if t_from < 0 or t_from >= self.n - 1:
            raise IndexError(f"flow step {t_from} out of range for {self.n} frames")
        if t_from not in self._flow:
            self._flow[t_from] = compute_flow(self.frames[t_from], self.frames[t_from + 1])
        return self._flow[t_from]

    def warp_mask(self, mask: np.ndarray, t_from: int, t_to: int) -> np.ndarray:
        """Propagate a binary mask from t_from to t_to by chaining flows."""
        if t_to == t_from:
            return mask.astype(bool)
        if t_to < t_from:
            raise ValueError("only forward propagation is supported")
        cur = mask.astype(np.uint8)
        for t in range(int(t_from), int(t_to)):
            cur = warp_with_flow(cur, self.flow_between(t)).astype(np.uint8)
        return cur > 0

    def warp_soft_mask(self, soft: np.ndarray, t_from: int, t_to: int) -> np.ndarray:
        """Propagate a continuous mask in [0, 1] from t_from to t_to by chaining."""
        if t_to == t_from:
            return np.asarray(soft, np.float32)
        if t_to < t_from:
            raise ValueError("only forward propagation is supported")
        cur = np.asarray(soft, np.float32)
        for t in range(int(t_from), int(t_to)):
            cur = warp_soft(cur, self.flow_between(t))
        return cur

    def warp_confidence(self, mask: np.ndarray, t_from: int, t_to: int) -> float:
        """Fraction of the warped mask that stays inside the frame (validity proxy)."""
        cur = mask.astype(np.uint8)
        for t in range(int(t_from), int(t_to)):
            f = self.flow_between(t)
            h, w = f.shape[:2]
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            # forward warp: gather at (x - flow(x)), matching warp_with_flow
            mx = (xs - f[..., 0]).astype(np.float32)
            my = (ys - f[..., 1]).astype(np.float32)
            valid = ((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1))
            src = np.where(valid, cur, 0).astype(np.uint8)
            cur = cv2.remap(src, mx, my, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.uint8)
        n0 = max(int(mask.sum()), 1)
        return float(cur.sum()) / float(n0)


def forward_backward_flow_error(prev: np.ndarray, cur: np.ndarray,
                                levels: int = 4, winsize: int = 21) -> float:
    """Mean forward-backward flow cycle-consistency error (a no-reference proxy)."""
    f = compute_flow(prev, cur, levels=levels, winsize=winsize)
    b = compute_flow(cur, prev, levels=levels, winsize=winsize)
    h, w = f.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    mx = np.clip(xs + f[..., 0], 0, w - 1)
    my = np.clip(ys + f[..., 1], 0, h - 1)
    bx = cv2.remap(b[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(b[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.sqrt((f[..., 0] + bx) ** 2 + (f[..., 1] + by) ** 2)
    return float(err.mean())
