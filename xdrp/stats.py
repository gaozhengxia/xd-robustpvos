"""Statistics for a reviewer-proof evaluation section.

All tests treat the VIDEO SEQUENCE as the paired unit. Frames within a sequence
are strongly correlated, so frame-level tests would inflate significance -- a
mistake that is common in this literature and that we explicitly avoid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # scipy is in requirements, but degrade gracefully
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0, statistic=np.mean) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of a statistic. Returns (point, lo, hi)."""
    v = np.asarray(list(values), np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(statistic(v)), float(v[0]), float(v[0])
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    boots = statistic(v[idx], axis=1)
    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return float(statistic(v)), lo, hi


# --------------------------------------------------------------------------- #
# paired tests
# --------------------------------------------------------------------------- #

def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on paired sequence-level scores."""
    x = np.asarray(list(a), np.float64)
    y = np.asarray(list(b), np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    d = y - x
    d = d[np.abs(d) > 0]
    n = int(d.size)
    if n == 0:
        return {"n": 0, "stat": float("nan"), "p": 1.0,
                "mean_diff": 0.0, "median_diff": 0.0}
    if _HAVE_SCIPY:
        try:
            res = _sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided",
                                correction=False)
            stat, p = float(res.statistic), float(res.pvalue)
        except Exception:
            stat, p = _sign_permutation(d)
    else:
        stat, p = _sign_permutation(d)
    return {"n": n, "stat": float(stat), "p": float(p),
            "mean_diff": float(np.mean(y - x)), "median_diff": float(np.median(y - x))}


def _sign_permutation(diff: np.ndarray, n_perm: int = 20000,
                      seed: int = 0) -> Tuple[float, float]:
    """Fallback exact-ish sign-flip permutation test."""
    rng = np.random.default_rng(int(seed))
    obs = float(np.abs(diff).sum())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), diff.size))
    null = np.abs(signs * diff[None, :]).sum(axis=1)
    p = float((null >= obs - 1e-12).mean())
    return obs, min(1.0, p)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1].

    SIGN CONVENTION: returns P(value from `b` > value from `a`) minus the
    reverse. So a POSITIVE value means group `b` is larger. In the paper's
    statistics table we call ``cliffs_delta(baseline, method)``, hence a positive
    number always means the method is better.
    """
    x = np.asarray(list(a), np.float64)
    y = np.asarray(list(b), np.float64)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = sum((yi > x).sum() for yi in y)
    lt = sum((yi < x).sum() for yi in y)
    return float(gt - lt) / float(x.size * y.size)


def holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (order preserved)."""
    p = np.asarray(list(pvalues), np.float64)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n, np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_report(baseline: Sequence[float], method: Sequence[float],
                  baseline_name: str = "baseline", method_name: str = "method",
                  seed: int = 0) -> Dict[str, Optional[float]]:
    """One row of the paper's statistics table (TABLE-14)."""
    x = np.asarray(list(baseline), np.float64)
    y = np.asarray(list(method), np.float64)
    m, lo, hi = bootstrap_ci((y - x), seed=seed)
    w = wilcoxon_paired(x, y)
    return {
        "comparison": f"{method_name} vs {baseline_name}",
        "n_sequences": float(w["n"]),
        "mean_delta": w["mean_diff"],
        "median_delta": w["median_diff"],
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "wilcoxon_p": w["p"],
        "cliffs_delta": cliffs_delta(x, y),
    }


# --------------------------------------------------------------------------- #
# correlation (the CL3 study)
# --------------------------------------------------------------------------- #

def spearman_with_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 10000,
                     seed: int = 0) -> Dict[str, float]:
    """Spearman rho with percentile bootstrap CI and a two-sided p-value."""
    a = np.asarray(list(x), np.float64)
    b = np.asarray(list(y), np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = a.size
    if n < 4:
        return {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "n": float(n)}
    rho = _spearman(a, b)
    if _HAVE_SCIPY:
        try:
            res = _sps.spearmanr(a, b)
            p = float(res.pvalue)
        except Exception:
            p = _spearman_p(rho, n)
    else:
        p = _spearman_p(rho, n)
    rng = np.random.default_rng(int(seed))
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        rr = _spearman(a[idx], b[idx])
        if not np.isnan(rr):
            boots.append(rr)
    if boots:
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
    else:
        lo = hi = float("nan")
    return {"rho": float(rho), "lo": lo, "hi": hi, "p": float(p), "n": float(n)}


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    # average ties
    sv = v[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _spearman_p(rho: float, n: int) -> float:
    if n < 3 or np.isnan(rho):
        return float("nan")
    t = rho * np.sqrt((n - 2) / max(1e-12, 1.0 - rho ** 2))
    if _HAVE_SCIPY:
        return float(2.0 * _sps.t.sf(abs(t), df=n - 2))
    # normal approximation fallback
    import math
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))
