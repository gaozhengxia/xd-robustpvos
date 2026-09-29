"""Figure generation helpers (matplotlib). Every function saves a PDF+PNG pair.

Figures produced here map to the paper's figure list:
  F1 overview      -> schematic (drawn separately, not here)
  F2 iqa_scatter   -> CL3: quality metric vs downstream J&F
  F3 severity_curve-> J&F retention vs severity
  F4 qualitative   -> 4-row panel incl. a failure case
  F5 trajectory    -> degradation-profile heatmap + vacuity vs true error
  F6 efficiency    -> accuracy vs encoder calls
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

PALETTE = {
    "greedy": "#6b7280", "cascade": "#b45309", "equal_tta": "#0e7490",
    "random_op": "#a16207", "dagrs": "#b91c1c", "oracle_clean": "#15803d",
    "dagrs_no_agree": "#7c3aed", "dagrs_no_profile": "#2563eb",
    "dagrs_no_gating": "#db2777", "dagrs_no_reanchor": "#059669",
}


def _save(fig, out_dir: str, name: str) -> List[str]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("pdf", "png"):
        p = d / f"{name}.{ext}"
        fig.savefig(str(p), bbox_inches="tight")
        outs.append(str(p))
    plt.close(fig)
    return outs


# --------------------------------------------------------------------------- #
# F2: the core correlation study
# --------------------------------------------------------------------------- #

def fig_iqa_scatter(rows: Sequence[Dict], metrics: Sequence[str],
                    out_dir: str = "results/figs", name: str = "F2_iqa_scatter",
                    correlations: Optional[Dict[str, Dict[str, float]]] = None
                    ) -> List[str]:
    rows = [r for r in rows if r.get("clean_reference") in (0, "0", False)]
    if not rows:
        rows = list(rows)
    n = len(metrics)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), squeeze=False)

    y = np.array([float(r["JF_frame"]) for r in rows])
    for i, mk in enumerate(metrics):
        ax = axes[i // ncol][i % ncol]
        x = np.array([float(r[mk]) for r in rows if mk in r and r[mk] != ""])
        yy = np.array([float(r["JF_frame"]) for r in rows if mk in r and r[mk] != ""])
        if x.size == 0:
            ax.axis("off")
            continue
        ax.scatter(x, yy, s=4, alpha=0.35, color="#374151", linewidths=0)
        if correlations and mk in correlations:
            c = correlations[mk]
            ax.set_title(f"{mk}\nrho={c['rho']:.3f} [{c['lo']:.2f},{c['hi']:.2f}]",
                         fontsize=8)
        else:
            ax.set_title(mk, fontsize=8)
        ax.set_xlabel(mk, fontsize=8)
        ax.set_ylabel("frame J&F", fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Image-quality metrics vs downstream segmentation quality", fontsize=10)
    return _save(fig, out_dir, name)


# --------------------------------------------------------------------------- #
# F3: severity curves
# --------------------------------------------------------------------------- #

def fig_severity_curve(rows: Sequence[Dict], out_dir: str = "results/figs",
                       name: str = "F3_severity_curve") -> List[str]:
    by = {}
    for r in rows:
        by.setdefault(r["mode"], {}).setdefault(float(r["level"]), []).append(float(r["J&F"]))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for mode, d in sorted(by.items()):
        lv = sorted(d.keys())
        mu = [float(np.mean(d[k])) for k in lv]
        sd = [float(np.std(d[k])) for k in lv]
        ax.errorbar(lv, mu, yerr=sd, marker="o", ms=4, capsize=3,
                    color=PALETTE.get(mode, None), label=mode)
    ax.set_xlabel("degradation severity level")
    ax.set_ylabel("J&F")
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, out_dir, name)


# --------------------------------------------------------------------------- #
# F5: trajectory diagnostics
# --------------------------------------------------------------------------- #

def fig_trajectory(profile_w: np.ndarray, vacuity: np.ndarray,
                   true_error: Optional[np.ndarray] = None,
                   op_names: Optional[Sequence[str]] = None,
                   out_dir: str = "results/figs",
                   name: str = "F5_trajectory") -> List[str]:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 3.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1.0]})
    im = ax1.imshow(profile_w.T, aspect="auto", cmap="magma",
                    interpolation="nearest")
    ax1.set_ylabel("operator")
    if op_names:
        ax1.set_yticks(np.arange(len(op_names)))
        ax1.set_yticklabels(op_names, fontsize=6)
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, label="profile mass", pad=0.01)

    ax2.plot(vacuity, color="#b91c1c", lw=1.4, label="vacuity (nu)")
    if true_error is not None:
        ax2.plot(true_error, color="#1d4ed8", lw=1.4, label="true error (1 - J)")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.legend(fontsize=7, frameon=False)
    return _save(fig, out_dir, name)


# --------------------------------------------------------------------------- #
# F6: efficiency
# --------------------------------------------------------------------------- #

def fig_efficiency(rows: Sequence[Dict], out_dir: str = "results/figs",
                   name: str = "F6_efficiency") -> List[str]:
    by = {}
    for r in rows:
        by.setdefault(r["mode"], {"calls": [], "jf": []})
        by[r["mode"]]["calls"].append(float(r["n_encoder_calls"]))
        by[r["mode"]]["jf"].append(float(r["J&F"]))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for mode, d in sorted(by.items()):
        ax.scatter(np.mean(d["calls"]), np.mean(d["jf"]), s=45,
                   color=PALETTE.get(mode, None), label=mode, zorder=3)
        ax.annotate(mode, (np.mean(d["calls"]), np.mean(d["jf"])),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("encoder calls per sequence (cost)")
    ax.set_ylabel("J&F")
    return _save(fig, out_dir, name)


# --------------------------------------------------------------------------- #
# F4: qualitative panels
# --------------------------------------------------------------------------- #

def fig_qualitative(panels: Sequence[Sequence[Dict]], out_dir: str = "results/figs",
                    name: str = "F4_qualitative") -> List[str]:
    """panels: rows x cols of dicts with keys {'img','mask','title'}.

    `img` float RGB, `mask` binary (optional), `title` str.
    """
    nr, nc = len(panels), max(len(r) for r in panels)
    fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 2.2 * nr), squeeze=False)
    for i, row in enumerate(panels):
        for j in range(nc):
            ax = axes[i][j]
            ax.axis("off")
            if j >= len(row):
                continue
            item = row[j]
            img = np.clip(np.asarray(item["img"], np.float32), 0, 1)
            ax.imshow(img)
            if item.get("mask") is not None:
                m = np.asarray(item["mask"]).astype(bool)
                over = np.zeros_like(img)
                over[..., 0] = 1.0
                ax.imshow(np.dstack([over, m.astype(np.float32) * 0.45]))
            ax.set_title(str(item.get("title", "")), fontsize=7)
    return _save(fig, out_dir, name)


def fig_bar_modes(rows: Sequence[Dict], metric: str = "J&F",
                  group_key: str = "degradation", out_dir: str = "results/figs",
                  name: str = "F_bar_modes") -> List[str]:
    groups = sorted({str(r[group_key]) for r in rows})
    modes = sorted({r["mode"] for r in rows})
    x = np.arange(len(groups))
    w = 0.8 / max(1, len(modes))
    fig, ax = plt.subplots(figsize=(1.3 * len(groups) + 2.0, 3.0))
    for i, m in enumerate(modes):
        vals = []
        for g in groups:
            v = [float(r[metric]) for r in rows
                 if str(r[group_key]) == g and r["mode"] == m]
            vals.append(float(np.mean(v)) if v else 0.0)
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w,
               color=PALETTE.get(m, None), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, out_dir, name)
