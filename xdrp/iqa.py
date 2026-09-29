"""Image-quality statistics used for the paper's core correlation study (CL3).

We compute BOTH full-reference metrics (available because our degradations are
synthetic, so a clean reference exists) and no-reference statistics. The point of
the study is that NONE of them predicts downstream segmentation quality.

Naming is deliberate: we do not call the no-reference statistics "BRISQUE" or
"NIQE" because they are not those metrics. They are named for what they are.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import cv2

EPS = 1e-8


def _u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(np.asarray(img, np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def psnr(ref: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    a = _u8(ref).astype(np.float64)
    b = _u8(test).astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(ref: np.ndarray, test: np.ndarray, sigma: float = 1.5,
         data_range: float = 255.0) -> float:
    """Mean SSIM with an 11x11 Gaussian window (Wang et al., 2004).

    The window is separable (`win = k @ k.T`), so it is applied with
    `sepFilter2D` and the 1D kernel: 22 multiply-adds per pixel instead of 121.
    Measured on a 480p frame: 243 ms -> 163 ms for a difference of 1e-16, i.e.
    the same value to machine precision. This metric is 68% of the IQA panel,
    and the panel is ~half of the per-frame cost of a degraded run, so the
    saving matters for the multi-hour sweeps.
    """
    a = _u8(ref).astype(np.float64)
    b = _u8(test).astype(np.float64)
    if a.ndim == 2:
        a = a[..., None]
        b = b[..., None]
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    k = cv2.getGaussianKernel(11, float(sigma)).ravel()
    vals = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mu_x = cv2.sepFilter2D(x, -1, k, k, borderType=cv2.BORDER_REFLECT)
        mu_y = cv2.sepFilter2D(y, -1, k, k, borderType=cv2.BORDER_REFLECT)
        x2 = cv2.sepFilter2D(x * x, -1, k, k, borderType=cv2.BORDER_REFLECT)
        y2 = cv2.sepFilter2D(y * y, -1, k, k, borderType=cv2.BORDER_REFLECT)
        xy = cv2.sepFilter2D(x * y, -1, k, k, borderType=cv2.BORDER_REFLECT)
        sxx = x2 - mu_x * mu_x
        syy = y2 - mu_y * mu_y
        sxy = xy - mu_x * mu_y
        s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
            ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2) + EPS)
        vals.append(float(s.mean()))
    return float(np.mean(vals))


def laplacian_variance(img: np.ndarray) -> float:
    """No-reference sharpness statistic."""
    g = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def dark_channel_mean(img: np.ndarray, radius: int = 7) -> float:
    """No-reference haze statistic (mean of the dark channel)."""
    x = np.asarray(img, np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    dark = np.min(x, axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return float(cv2.erode(dark, k).mean())


def global_contrast(img: np.ndarray) -> float:
    """RMS contrast of luminance."""
    g = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(g.std())


def entropy(img: np.ndarray, bins: int = 256) -> float:
    g = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY)
    h, _ = np.histogram(g, bins=bins, range=(0, 256))
    p = h.astype(np.float64) / (h.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sharpness_gradient(img: np.ndarray) -> float:
    """Tenengrad: mean squared gradient magnitude."""
    g = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx ** 2 + gy ** 2).mean())


def saturation_mean(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1].astype(np.float32).mean() / 255.0)


def compute_iqa(degraded: np.ndarray, clean: Optional[np.ndarray] = None,
                restored: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute the full IQA battery.

    `degraded` is the input actually seen by the segmenter (degraded, or restored
    if a restoration was applied). `clean` is the reference, if available.
    """
    out: Dict[str, float] = {
        "iqa_laplacian_var": laplacian_variance(degraded),
        "iqa_dark_channel": dark_channel_mean(degraded),
        "iqa_contrast": global_contrast(degraded),
        "iqa_entropy": entropy(degraded),
        "iqa_tenengrad": sharpness_gradient(degraded),
        "iqa_saturation": saturation_mean(degraded),
    }
    if clean is not None:
        out["iqa_psnr"] = psnr(clean, degraded)
        out["iqa_ssim"] = ssim(clean, degraded)
    if restored is not None and clean is not None:
        out["iqa_psnr_restored"] = psnr(clean, restored)
        out["iqa_ssim_restored"] = ssim(clean, restored)
    return out


#: Metric names that the correlation study (CL3) regresses against J&F.
FULL_REFERENCE_METRICS = ["iqa_psnr", "iqa_ssim"]
NO_REFERENCE_METRICS = ["iqa_laplacian_var", "iqa_dark_channel", "iqa_contrast",
                        "iqa_entropy", "iqa_tenengrad", "iqa_saturation"]
ALL_IQA_METRICS = FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS
