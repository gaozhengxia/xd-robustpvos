"""VOS evaluation metrics: region similarity J, boundary F-measure F.

Follows the standard DAVIS protocol so numbers are directly comparable with the
RobustPVOS benchmark and the VOS literature.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2


def j_region(pred: np.ndarray, gt: np.ndarray) -> float:
    """Region similarity (IoU) for one object, one frame."""
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum()) / float(union)


def _boundary(mask: np.ndarray, dilation: int = 2) -> np.ndarray:
    m = np.asarray(mask).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, k, iterations=1)
    b = (m - er) > 0
    if dilation > 0:
        dk = np.ones((2 * int(dilation) + 1, 2 * int(dilation) + 1), np.uint8)
        b = cv2.dilate(b.astype(np.uint8), dk, iterations=1).astype(bool)
    return b


def f_boundary(pred: np.ndarray, gt: np.ndarray, dilation: int = 2) -> float:
    """Boundary F-measure: precision/recall of boundary pixels with tolerance.

    `dilation` is the tolerance in pixels (DAVIS default = 2 at 480p).
    """
    bp = _boundary(pred, dilation)
    bg = _boundary(gt, dilation)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    prec = float((bp & bg).sum()) / float(bp.sum())
    rec = float((bp & bg).sum()) / float(bg.sum())
    if prec + rec <= 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def jf(pred: np.ndarray, gt: np.ndarray, dilation: int = 2) -> Tuple[float, float, float]:
    j = j_region(pred, gt)
    f = f_boundary(pred, gt, dilation)
    return j, f, 0.5 * (j + f)


def seq_metrics(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                dilation: int = 2) -> Dict[str, float]:
    """Mean J / F / J&F over a sequence (means of per-frame values)."""
    js, fs = [], []
    for p, g in zip(preds, gts):
        if np.asarray(g).sum() == 0 and np.asarray(p).sum() == 0:
            continue
        j, f, _ = jf(p, g, dilation)
        js.append(j)
        fs.append(f)
    if not js:
        return {"J": 0.0, "F": 0.0, "J&F": 0.0, "n_frames": 0}
    return {"J": float(np.mean(js)), "F": float(np.mean(fs)),
            "J&F": float(0.5 * (np.mean(js) + np.mean(fs))),
            "n_frames": len(js)}


def retention(jf_deg: float, jf_clean: float) -> float:
    """J&F retention rate relative to the clean-input run."""
    if jf_clean <= 0:
        return float("nan")
    return float(jf_deg) / float(jf_clean)


def per_attribute_breakdown(records: Sequence[Dict], attr_key: str = "degradation"
                            ) -> Dict[str, Dict[str, float]]:
    """Group a list of result records by an attribute and average the metrics."""
    out: Dict[str, List[float]] = {}
    for r in records:
        out.setdefault(str(r[attr_key]), []).append(float(r["J&F"]))
    return {k: {"J&F_mean": float(np.mean(v)), "n": float(len(v))}
            for k, v in sorted(out.items())}
