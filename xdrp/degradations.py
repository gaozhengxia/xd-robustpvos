"""Reproducible compound-degradation engine for XD-RobustPVOS.

Design rules (these are load-bearing for the paper's reproducibility claim):
  1. Every operator is a pure function of (image, severity, rng).
  2. Severity is CONTINUOUS in [1, 5]; internally normalised to s in [0, 1].
  3. All randomness comes from a numpy Generator seeded from a single integer,
     composed as  seed = base_seed * 1_000_003 + stream_id.
  4. Operator order inside a compound degradation is FIXED and documented, so the
     same (image, config, seed) triple yields bit-identical output on any machine.

Images are float32 RGB in [0, 1] throughout. Use to_float / to_u8 at the borders.
"""
from __future__ import annotations

import hashlib

import numpy as np
import cv2

EPS = 1e-8


def _stable_hash(text: str) -> int:
    """Deterministic string hash (Python's built-in hash is salted per process)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

# --------------------------------------------------------------------------- #
# conversions and helpers
# --------------------------------------------------------------------------- #

def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0)
    return np.asarray(img, dtype=np.float32)


def to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def severity_to_s(level: float) -> float:
    """Map a continuous severity level in [1, 5] to s in [0, 1]."""
    return float((np.clip(float(level), 1.0, 5.0) - 1.0) / 4.0)


def value_noise(shape, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth low-frequency noise field in [0, 1] with a deterministic seed.

    Used as a pseudo-depth / transmission field. Deliberately dependency-free
    (no depth estimator) so the benchmark is reproducible without extra weights.
    """
    h, w = int(shape[0]), int(shape[1])
    gh = max(2, int(np.ceil(h / max(scale, 1.0))) + 1)
    gw = max(2, int(np.ceil(w / max(scale, 1.0))) + 1)
    grid = rng.random((gh, gw)).astype(np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    field = np.clip(field, 0.0, 1.0)
    lo, hi = float(field.min()), float(field.max())
    return (field - lo) / (hi - lo + EPS)


# --------------------------------------------------------------------------- #
# individual degradations
# --------------------------------------------------------------------------- #

def fog(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Koschmieder atmospheric scattering: I = J*t + A*(1-t), t = exp(-beta*d)."""
    s = severity_to_s(level)
    h, w = img.shape[:2]
    scale = max(16.0, min(h, w) / 6.0)
    d = value_noise((h, w), scale, rng)                    # pseudo-depth in [0,1]
    beta = 0.6 + 3.4 * s                                   # extinction coefficient
    t = np.exp(-beta * (0.15 + 0.85 * d))[..., None]
    A = 0.75 + 0.20 * s                                    # atmospheric light
    return np.clip(img * t + A * (1.0 - t), 0.0, 1.0)


def lowlight(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Power-law response + gain reduction + Poisson-Gaussian sensor noise."""
    s = severity_to_s(level)
    gamma = 1.0 + 2.4 * s
    out = np.power(np.clip(img, 0.0, 1.0), gamma)
    out = out * (1.0 - 0.35 * s)
    sigma = 0.02 + 0.06 * s
    out = out + rng.normal(0.0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def rain(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Rain streak layer plus an atmospheric veil (rain always hazes a little)."""
    s = severity_to_s(level)
    h, w = img.shape[:2]
    base = fog(img, 1.0 + 2.0 * s, rng) if s > 0.2 else img.copy()
    layer = np.zeros((h, w), np.float32)
    n = int(20 + 320 * s)
    L = int(8 + 26 * s)
    th = max(1, int(1 + 2 * s))
    for _ in range(n):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        ang = float((-25.0 + 50.0 * rng.random()) * np.pi / 180.0)
        dy = int(L * np.cos(ang))
        dx = int(L * np.sin(ang))
        val = float(0.5 + 0.5 * rng.random())
        cv2.line(layer, (x, y), (x + dx, y + dy), val, th)
    layer = cv2.blur(layer, (3, 3))
    layer = np.clip(layer, 0.0, 1.0)[..., None]
    return np.clip(1.0 - (1.0 - base) * (1.0 - 0.7 * layer), 0.0, 1.0)


def snow(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """White particle layer plus a veil."""
    s = severity_to_s(level)
    h, w = img.shape[:2]
    base = fog(img, 1.0 + 1.5 * s, rng)
    layer = np.zeros((h, w), np.float32)
    n = int(60 + 600 * s)
    rmax = max(1, int(1 + 3 * s))
    for _ in range(n):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, rmax + 1))
        cv2.circle(layer, (x, y), r, float(0.6 + 0.4 * rng.random()), -1)
    layer = cv2.blur(layer, (3, 3))
    layer = np.clip(layer, 0.0, 1.0)[..., None]
    return np.clip(base * (1.0 - 0.35 * layer) + layer, 0.0, 1.0)


def motion_blur(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Linear motion blur with a seeded angle."""
    s = severity_to_s(level)
    L = int(3 + 24 * s)
    ang = float(rng.random() * 180.0)
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2.0 - 0.5, L / 2.0 - 0.5), ang, 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    k /= (k.sum() + EPS)
    out = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)
    return np.clip(out, 0.0, 1.0)


def sensor_noise(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian plus chroma noise (simulates high-ISO acquisition)."""
    s = severity_to_s(level)
    sigma = 0.01 + 0.09 * s
    out = img + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    for c in range(3):
        out[..., c] += rng.normal(0.0, 0.3 * sigma, img.shape[:2]).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def underwater(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Wavelength-dependent attenuation + backscatter (turbidity domain shift proxy)."""
    s = severity_to_s(level)
    h, w = img.shape[:2]
    att = np.array([0.35 + 1.60 * s, 0.18 + 0.90 * s, 0.08 + 0.35 * s], np.float32)
    out = img * np.exp(-att)[None, None, :]
    bs = np.array([0.10, 0.28, 0.34], np.float32) * (0.25 + 0.85 * s)
    scale = max(20.0, min(h, w) / 5.0)
    t = np.exp(-(0.4 + 2.2 * s) * value_noise((h, w), scale, rng))[..., None]
    out = out * t + bs[None, None, :] * (1.0 - t)
    return np.clip(out, 0.0, 1.0)


def dust(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Sand/dust storm: veil + yellow tint + desaturation."""
    s = severity_to_s(level)
    out = fog(img, 1.0 + 1.8 * s, rng)
    tint = np.array([1.0 + 0.18 * s, 1.0 + 0.06 * s, 1.0 - 0.28 * s], np.float32)
    out = out * tint[None, None, :]
    g = out.mean(axis=2, keepdims=True)
    out = out * (1.0 - 0.30 * s) + g * (0.30 * s)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

DEGRADATIONS = {
    "fog": fog,
    "lowlight": lowlight,
    "rain": rain,
    "snow": snow,
    "motion_blur": motion_blur,
    "sensor_noise": sensor_noise,
    "underwater": underwater,
    "dust": dust,
}

SINGLE_DEGRADATIONS = list(DEGRADATIONS.keys())

#: Compound degradations. Order is FIXED and is part of the benchmark definition.
COMPOUND_DEGRADATIONS = {
    "C1_fog_noise":      ["fog", "sensor_noise"],
    "C2_lowlight_blur":  ["lowlight", "motion_blur"],
    "C3_rain_snow":      ["rain", "snow"],
    "C4_fog_blur_noise": ["fog", "motion_blur", "sensor_noise"],
}

ALL_DEGRADATION_NAMES = SINGLE_DEGRADATIONS + list(COMPOUND_DEGRADATIONS.keys())


def apply_single(img: np.ndarray, name: str, level: float,
                 rng: np.random.Generator) -> np.ndarray:
    if name not in DEGRADATIONS:
        raise KeyError(f"unknown degradation '{name}'; known: {list(DEGRADATIONS)}")
    return DEGRADATIONS[name](to_float(img), level, rng)


def apply_degradation(img: np.ndarray, name: str, level: float,
                      seed: int = 0, stream_id: int = 0) -> np.ndarray:
    """Apply a single OR compound degradation deterministically.

    Parameters
    ----------
    name : one of SINGLE_DEGRADATIONS or COMPOUND_DEGRADATIONS keys.
    level : continuous severity in [1, 5].
    seed, stream_id : compose into the RNG seed, so results are reproducible
        given the same (sequence, frame, degradation, level) coordinates.
    """
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(stream_id))
    x = to_float(img)
    if name in COMPOUND_DEGRADATIONS:
        for sub in COMPOUND_DEGRADATIONS[name]:
            # Each sub-degradation gets its own deterministic sub-stream so that
            # adding a new member to a compound does not perturb the others.
            sub_rng = np.random.default_rng(
                int(seed) * 1_000_003 + int(stream_id) * 97 + _stable_hash(sub) % 9973)
            x = DEGRADATIONS[sub](x, level, sub_rng)
        return np.clip(x, 0.0, 1.0)
    return np.clip(DEGRADATIONS[name](x, level, rng), 0.0, 1.0)


def is_compound(name: str) -> bool:
    return name in COMPOUND_DEGRADATIONS


def members(name: str) -> list:
    return COMPOUND_DEGRADATIONS.get(name, [name])


# --------------------------------------------------------------------------- #
# temporal severity scheduling
# --------------------------------------------------------------------------- #

def severity_schedule(n_frames: int, base: float = 3.0, amp: float = 0.30,
                      period: int = 0, seed: int = 0) -> np.ndarray:
    """Smoothly time-varying severity in [1, 5].

    Constant severity lets a model exploit a static prior; real degradation
    (fog thickening as a drone flies, a cloud passing) varies over time. This is
    the protocol used by the RobustPVOS line and we keep it for comparability.
    """
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    period = int(period) if period else max(8, int(n_frames // 2))
    phase = float(rng.random() * 2.0 * np.pi)
    t = np.arange(int(n_frames), dtype=np.float64)
    s = float(base) * (1.0 + float(amp) * np.sin(2.0 * np.pi * t / period + phase))
    return np.clip(s, 1.0, 5.0).astype(np.float32)
