"""Training-free Restoration Operator Bank (ROB).

Every operator is:
  * classical (no learned parameters),
  * deterministic,
  * dependency-light (numpy + OpenCV only),
  * and documented with its target degradation family.

The bank is the ONLY source of "hypotheses" in DAG-RS. Because it contains no
learned weights, the whole method is reproducible bit-for-bit on any machine.

Operator families (`family`) are used only by the ablation that fixes a single
domain-prior operator (A4) and by the failure analysis; the main method never
consults them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import cv2

from .degradations import to_float, to_u8, EPS


# --------------------------------------------------------------------------- #
# primitive implementations
# --------------------------------------------------------------------------- #

def op_identity(img: np.ndarray) -> np.ndarray:
    return np.clip(to_float(img), 0.0, 1.0)


def op_gamma(x: np.ndarray, g: float) -> np.ndarray:
    return np.clip(np.power(np.clip(x, 0.0, 1.0), g), 0.0, 1.0)


def op_clahe(x: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    lab[..., 0] = clahe.apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return to_float(out)


def op_hist_stretch(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[2]):
        ch = x[..., c]
        a, b = np.percentile(ch, lo), np.percentile(ch, hi)
        out[..., c] = np.clip((ch - a) / (b - a + EPS), 0.0, 1.0)
    return out


def op_dark_channel_dehaze(x: np.ndarray, omega: float = 0.85,
                           t0: float = 0.15, radius: int = 7) -> np.ndarray:
    """Simplified Dark Channel Prior dehazing (He et al., TPAMI 2011)."""
    h, w = x.shape[:2]
    dark = np.min(x, axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    dark = cv2.erode(dark, k)
    flat = dark.ravel()
    n_take = max(1, int(0.001 * flat.size))
    idx = np.argsort(flat)[-n_take:]
    A = np.array([float(x[..., c].ravel()[idx].mean()) for c in range(3)], np.float32)
    A = np.clip(A, 0.40, 1.0)
    t = 1.0 - omega * dark
    t = cv2.blur(t.astype(np.float32), (max(3, radius), max(3, radius)))
    t = np.clip(t, t0, 1.0)[..., None]
    out = (x - A[None, None, :]) / t + A[None, None, :]
    return np.clip(out, 0.0, 1.0)


def op_msr(x: np.ndarray, sigmas=(15.0, 80.0, 250.0)) -> np.ndarray:
    """Multi-scale Retinex with per-channel percentile normalisation."""
    log_x = np.log(np.clip(x, EPS, 1.0))
    acc = np.zeros_like(x, dtype=np.float32)
    for s in sigmas:
        bl = cv2.GaussianBlur(x, (0, 0), float(s))
        acc += log_x - np.log(np.clip(bl, EPS, 1.0))
    acc /= float(len(sigmas))
    out = np.empty_like(acc)
    for c in range(3):
        ch = acc[..., c]
        a, b = np.percentile(ch, 1.0), np.percentile(ch, 99.0)
        out[..., c] = np.clip((ch - a) / (b - a + EPS), 0.0, 1.0)
    return out


def op_denoise_nlm(x: np.ndarray, h: float = 6.0) -> np.ndarray:
    """Non-local means on uint8 (slow; used only if explicitly enabled)."""
    u8 = to_u8(x)
    out = cv2.fastNlMeansDenoisingColored(u8, None, float(h), float(h), 7, 21)
    return to_float(out)


def op_denoise_bilateral(x: np.ndarray, d: int = 7,
                         sc: float = 40.0, ss: float = 40.0) -> np.ndarray:
    u8 = to_u8(x)
    out = cv2.bilateralFilter(u8, int(d), float(sc), float(ss))
    return to_float(out)


def op_unsharp(x: np.ndarray, strength: float = 0.8, sigma: float = 1.2) -> np.ndarray:
    bl = cv2.GaussianBlur(x, (0, 0), float(sigma))
    out = x + float(strength) * (x - bl)
    return np.clip(out, 0.0, 1.0)


def op_gray_world(x: np.ndarray) -> np.ndarray:
    means = x.reshape(-1, 3).mean(axis=0) + EPS
    out = x / means[None, None, :]
    m = float(out.max())
    if m > 0:
        out = out / m
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# operator registry
# --------------------------------------------------------------------------- #

@dataclass
class Operator:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    family: str
    target: str
    cost: str = "low"          # low / mid / high  (relative wall-clock)
    note: str = ""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.clip(self.fn(to_float(img)), 0.0, 1.0)


def build_default_bank(include_slow: bool = False) -> List[Operator]:
    """The default 13-operator bank (12 + identity when include_slow=False)."""
    bank: List[Operator] = [
        Operator("id", op_identity, "identity", "none", "low",
                 "no restoration; the degenerate hypothesis"),

        Operator("gamma_lo", lambda x: op_gamma(x, 0.60), "lowlight",
                 "low-light", "low", "brightens shadows"),
        Operator("gamma_hi", lambda x: op_gamma(x, 1.60), "fog",
                 "haze / veil", "low", "restores contrast lost to a veil"),
        Operator("gamma_xhi", lambda x: op_gamma(x, 2.20), "fog",
                 "heavy veil", "low", "aggressive contrast restoration"),

        Operator("clahe", lambda x: op_clahe(x, 2.0, 8), "lowlight",
                 "low local contrast", "mid", "local contrast equalisation"),
        Operator("hist", op_hist_stretch, "generic",
                 "global contrast loss", "low", "percentile stretch"),

        Operator("dcp", lambda x: op_dark_channel_dehaze(x, 0.85, 0.15, 7),
                 "fog", "haze", "mid", "dark channel prior, standard strength"),
        Operator("dcp_strong", lambda x: op_dark_channel_dehaze(x, 0.95, 0.08, 11),
                 "fog", "dense haze", "mid", "dark channel prior, aggressive"),

        Operator("msr", op_msr, "lowlight",
                 "low light / colour cast", "mid", "multi-scale Retinex"),
        Operator("grayworld", op_gray_world, "generic",
                 "colour cast (underwater, dust)", "low", "white balance"),

        Operator("unsharp", lambda x: op_unsharp(x, 0.8, 1.2), "blur",
                 "mild blur", "low", "USM sharpening"),
        Operator("unsharp_hi", lambda x: op_unsharp(x, 1.6, 2.0), "blur",
                 "motion blur", "low", "aggressive USM"),
        Operator("bilateral", lambda x: op_denoise_bilateral(x, 7, 40.0, 40.0),
                 "noise", "sensor noise", "mid", "edge-preserving denoise"),
    ]
    if include_slow:
        bank.append(Operator("nlm", lambda x: op_denoise_nlm(x, 6.0), "noise",
                             "heavy sensor noise", "high",
                             "non-local means (slow)"))
    return bank


class OperatorBank:
    """Holds the bank and provides name/index access used by the pipeline."""

    def __init__(self, operators: Optional[List[Operator]] = None,
                 include_slow: bool = False):
        self.operators = operators if operators is not None else build_default_bank(include_slow)

    def __len__(self) -> int:
        return len(self.operators)

    def __getitem__(self, i: int) -> Operator:
        return self.operators[i]

    @property
    def names(self) -> List[str]:
        return [o.name for o in self.operators]

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def apply(self, i: int, img: np.ndarray) -> np.ndarray:
        return self.operators[i](img)

    def apply_name(self, name: str, img: np.ndarray) -> np.ndarray:
        return self.operators[self.index_of(name)](img)

    def subset(self, idxs) -> "OperatorBank":
        return OperatorBank([self.operators[i] for i in idxs])

    def table(self) -> List[Dict[str, str]]:
        return [{"idx": str(i), "name": o.name, "family": o.family,
                 "target": o.target, "cost": o.cost, "note": o.note}
                for i, o in enumerate(self.operators)]

    def family_map(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for o in self.operators:
            out.setdefault(o.family, []).append(o.name)
        return out
