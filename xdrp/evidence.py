"""Downstream-agreement evidence, Dirichlet fusion, and the degradation profile.

This module implements the decisional core of DAG-RS (paper Sec. 4.3-4.5).

IMPORTANT SCOPE NOTE
--------------------
`vacuity` produced here is an *unsupervised heuristic evidence mass*, not a
calibrated probability. The paper reports only its rank correlation with the
true per-frame error and explicitly claims no coverage guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2

EPS = 1e-8

#: Fixed (non-learned) agreement weights w1, w2, w3 -- see paper Sec. 4.3.
DEFAULT_AGREEMENT_WEIGHTS = (0.45, 0.35, 0.20)


# --------------------------------------------------------------------------- #
# basic mask geometry
# --------------------------------------------------------------------------- #

def as_bool(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.dtype == bool:
        return m
    return m > 0.5


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = as_bool(a)
    b = as_bool(b)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def area(mask: np.ndarray) -> int:
    return int(as_bool(mask).sum())


def perimeter(mask: np.ndarray) -> int:
    """Boundary pixel count via a 3x3 erosion difference."""
    m = as_bool(mask).astype(np.uint8)
    if m.sum() == 0:
        return 0
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, k, iterations=1)
    return int((m - er).sum())


def bbox_of(mask: np.ndarray, pad_ratio: float = 0.0, shape=None) -> np.ndarray:
    """Return [x0, y0, x1, y1] (inclusive-ish) for a binary mask."""
    m = as_bool(mask)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        if shape is None:
            h, w = m.shape[:2]
        else:
            h, w = shape[:2]
        return np.array([0, 0, w - 1, h - 1], np.float32)
    h, w = m.shape[:2]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if pad_ratio > 0:
        bw, bh = (x1 - x0), (y1 - y0)
        x0 -= pad_ratio * bw
        x1 += pad_ratio * bw
        y0 -= pad_ratio * bh
        y1 += pad_ratio * bh
    return np.array([max(0.0, x0), max(0.0, y0),
                     min(w - 1.0, x1), min(h - 1.0, y1)], np.float32)


def mask_to_logits(mask: np.ndarray, size: int = 256, scale: float = 8.0) -> np.ndarray:
    """Convert a binary mask into the low-resolution logit prompt SAM expects.

    Shape is (1, size, size) as required by SAM/SAM2 image predictors.

    NOTE this is the *hard* prompt: binarising asserts every pixel inside the
    mask as foreground with the same large confidence. See `soft_mask_to_logits`
    for the alternative and why it matters.
    """
    m = as_bool(mask).astype(np.float32) * 2.0 - 1.0
    m = cv2.resize(m, (int(size), int(size)), interpolation=cv2.INTER_NEAREST)
    return (m * float(scale))[None, :, :]


def soft_mask_to_logits(soft: np.ndarray, size: int = 256,
                        lo: float = -6.0, hi: float = 8.0) -> np.ndarray:
    """Convert a *soft* mask in [0, 1] into the low-resolution logit prompt.

    Why this exists
    ---------------
    `mask_to_logits` first binarises, which turns the re-prompt into a hard wall:
    every pixel in the warped mask is asserted as foreground with identical,
    maximal confidence, leaving SAM 2 no room to disagree about the boundary.
    Combined with a box that is derived from that same mask, this closes a
    positive-feedback loop -- the mask grows, the box grows, the mask grows again
    -- which is one of the drivers of the closed-loop runaway measured in
    docs/BASELINE_DIAGNOSIS.md (`bmx-trees` ends at 110x the ground-truth area).

    Keeping the probability and mapping it through a logit lets the segmenter
    re-decide the boundary from the *current* image, which is what the
    `mask_input` argument of the SAM 2 image predictor is for.
    """
    p = np.clip(np.asarray(soft, np.float32), 1e-4, 1.0 - 1e-4)
    p = cv2.resize(p, (int(size), int(size)), interpolation=cv2.INTER_LINEAR)
    logits = np.log(p / (1.0 - p))
    return np.clip(logits, float(lo), float(hi))[None, :, :]


def prompt_logits(mask: np.ndarray, size: int = 256,
                  scale: float = 8.0) -> np.ndarray:
    """Dispatch on dtype: bool/integer -> hard wall, float -> soft logits."""
    m = np.asarray(mask)
    if m.dtype == bool or np.issubdtype(m.dtype, np.integer):
        return mask_to_logits(m, size=size, scale=scale)
    return soft_mask_to_logits(m, size=size)


# --------------------------------------------------------------------------- #
# agreement scoring
# --------------------------------------------------------------------------- #

def agreement_components(candidates: Sequence[np.ndarray],
                         anchor_warped: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, c, u): temporal consistency, consensus, boundary stability."""
    M = len(candidates)
    if M == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    a = np.array([iou_binary(c, anchor_warped) for c in candidates], np.float64)

    c = np.zeros(M, np.float64)
    for i in range(M):
        if M > 1:
            c[i] = float(np.mean([iou_binary(candidates[i], candidates[j])
                                  for j in range(M) if j != i]))
        else:
            c[i] = a[i]

    per_anchor = max(perimeter(anchor_warped), 1)
    per = np.array([perimeter(cc) for cc in candidates], np.float64)
    u = 1.0 - np.minimum(1.0, np.abs(per / float(per_anchor) - 1.0))

    return a, c, u


def agreement_scores(candidates: Sequence[np.ndarray], anchor_warped: np.ndarray,
                     weights: Tuple[float, float, float] = DEFAULT_AGREEMENT_WEIGHTS
                     ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """A_m = w1*a + w2*c + w3*u, all terms in [0, 1]."""
    a, c, u = agreement_components(candidates, anchor_warped)
    w = np.asarray(weights, np.float64)
    w = w / (w.sum() + EPS)
    A = w[0] * a + w[1] * c + w[2] * u
    return np.clip(A, 0.0, 1.0), {"a": a, "c": c, "u": u}


# --------------------------------------------------------------------------- #
# Dirichlet (evidential) fusion
# --------------------------------------------------------------------------- #

@dataclass
class DirichletEvidence:
    e: np.ndarray            # evidence mass per hypothesis
    alpha: np.ndarray        # Dirichlet parameters = e + 1
    S: float                 # Dirichlet strength
    belief: np.ndarray       # b_m = e_m / S
    vacuity: float           # nu = M / S

    def top(self, k: int) -> np.ndarray:
        k = min(int(k), self.belief.size)
        return np.argsort(-self.belief)[:k]

    def as_dict(self) -> Dict[str, float]:
        d = {"S": float(self.S), "vacuity": float(self.vacuity)}
        for i, b in enumerate(self.belief):
            d[f"belief_{i}"] = float(b)
        return d


def dirichlet_from_agreement(A: np.ndarray, kappa: float = 8.0,
                             gamma: float = 2.0) -> DirichletEvidence:
    """Map agreement scores to a Dirichlet opinion over the hypothesis set.

    e_m = kappa * A_m^gamma ; alpha_m = e_m + 1 ; S = sum alpha ; nu = M / S.
    With A == 1 everywhere, nu -> 1/(kappa+1); with A == 0, nu -> 1.
    """
    A = np.clip(np.asarray(A, np.float64), 0.0, 1.0)
    e = float(kappa) * np.power(A, float(gamma)) + 1e-9
    alpha = e + 1.0
    S = float(alpha.sum())
    belief = e / (S + EPS)
    vacuity = float(len(A)) / (S + EPS)
    return DirichletEvidence(e=e, alpha=alpha, S=S, belief=belief, vacuity=vacuity)


def fuse_candidates(candidates: Sequence[np.ndarray], belief: np.ndarray,
                    top_k: int = 2, thresh: float = 0.5
                    ) -> Tuple[np.ndarray, List[int]]:
    """Belief-weighted soft fusion of the top-k candidates, then binarise."""
    M = len(candidates)
    if M == 0:
        raise ValueError("no candidates to fuse")
    order = list(np.argsort(-np.asarray(belief))[:min(int(top_k), M)])
    w = np.asarray([belief[i] for i in order], np.float64)
    w = w / (w.sum() + EPS)
    acc = np.zeros(as_bool(candidates[0]).shape, np.float32)
    for wi, i in zip(w, order):
        acc += float(wi) * as_bool(candidates[i]).astype(np.float32)
    return acc >= float(thresh), order


# --------------------------------------------------------------------------- #
# per-object degradation profile
# --------------------------------------------------------------------------- #

@dataclass
class DegradationProfile:
    """A distribution over restoration operators, accumulated as evidence over time.

    NOT a memory bank: we never select, store or drop video frames. The profile
    only records *which restoration hypothesis has been agreeing with the
    trajectory*, and is used to (i) shrink the evaluation set for efficiency and
    (ii) trigger re-exploration after sustained disagreement.
    """
    n_operators: int
    lam: float = 0.45
    min_j: int = 3
    w: Optional[np.ndarray] = None
    n_updates: int = 0

    def __post_init__(self):
        if self.w is None:
            self.w = np.ones(int(self.n_operators), np.float64) / float(self.n_operators)

    def update(self, evidence: np.ndarray, idxs: Optional[Sequence[int]] = None) -> None:
        ev = np.asarray(evidence, np.float64).ravel()
        if idxs is None:
            idxs = np.arange(len(ev))
        idxs = list(idxs)
        p = np.zeros(int(self.n_operators), np.float64)
        s = ev.sum()
        if s <= 0:
            return
        for local, gi in enumerate(idxs):
            p[int(gi)] = ev[local] / s
        self.w = (1.0 - float(self.lam)) * self.w + float(self.lam) * p
        tot = self.w.sum()
        if tot > 0:
            self.w = self.w / tot
        self.n_updates += 1

    def select(self, j: int) -> List[int]:
        """Top-j operators by current profile mass."""
        j = max(int(self.min_j), min(int(j), int(self.n_operators)))
        return sorted(np.argsort(-self.w)[:j].tolist())

    def all_operators(self) -> List[int]:
        return list(range(int(self.n_operators)))

    def argmax(self) -> int:
        return int(np.argmax(self.w))

    @property
    def ready(self) -> bool:
        return self.n_updates > 0

    def entropy(self) -> float:
        p = np.clip(self.w, EPS, 1.0)
        return float(-(p * np.log(p)).sum() / np.log(len(p) + EPS))


# --------------------------------------------------------------------------- #
# appearance descriptor (for global re-anchoring only)
# --------------------------------------------------------------------------- #

def color_descriptor(img: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """Cheap, deterministic appearance descriptor: masked colour histogram.

    Used ONLY for global re-anchoring candidate scoring. Deliberately not a
    learned embedding so the method stays weight-free.
    """
    m = as_bool(mask)
    if m.sum() == 0:
        return np.zeros(3 * bins, np.float32)
    px = img[m].reshape(-1, 3)
    feats = []
    for c in range(3):
        h, _ = np.histogram(px[:, c], bins=bins, range=(0.0, 1.0), density=False)
        h = h.astype(np.float32)
        h /= (h.sum() + EPS)
        feats.append(h)
    v = np.concatenate(feats)
    return v / (np.linalg.norm(v) + EPS)


def descriptor_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + EPS))
