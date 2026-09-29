"""Figure builder for the manuscript's data-driven figures: FIG-2, FIG-3, FIG-5.

Why this script exists (the defect it closes)
---------------------------------------------
`results/figs/` held two PNGs and **no script owned either of them**, so both had
silently drifted away from the table they illustrate:

* `F2_iqa_scatter.png` was written by `05_analysis.study_iqa` from **frame-level**
  rows (`JF_frame`), while the claim it illustrates (Sec. 5.4 / Table 14-15) was
  finalised in the **segment-level `across`** view, n = 360. The two conventions
  report different statistics (the frame-level probe for `greedy`/PSNR is not the
  `across` value), so the figure contradicted its own table.
* `F3_severity.png` (mtime 08:49:10) was written in the *same run* as
  `results/severity_curve.{json,csv,md}` (08:49:09), i.e. from the **P0** numbers,
  while Sec. 5.6 / Table 17 report the **P1** re-run (13:36). 36 of the 45 cells
  differ, by up to 3.6e-03.

Neither mismatch was reachable by the existing guards (118-123), because every one
of them consumes *tables* and none consumes a *figure*.  A figure is a
claim-bearing artifact: it needs the same "value must be traceable to the table
that owns it" discipline.

Convention contract
-------------------
A figure is drawn **only** from the published artifact that its table is built
from, and `--check` asserts the plotted series equal that artifact's values.  A
figure that cannot be tied to its table is not written.  Specifically:

| figure | source artifact (the table's own input)        | table      |
|--------|------------------------------------------------|------------|
| FIG-1  | `xdrp.operators.OperatorBank` (the live code)   | Sec. 4.1   |
| FIG-2  | `results/p1/iqa_config_level_p1_cells.csv` (x) + | Table 14,  |
|        | the J&F side of `--jf-inputs` (y)                | Table 15   |
| FIG-3  | `results/p1/severity_curve_p1.csv`               | Table 17   |
| FIG-5  | `results/dagrs_main_raw.csv` per-frame columns   | Sec. 5.8   |

FIG-2 is drawn on the **P1** convention, and its table must be too: the top-level
`results/iqa_config_level.{json,cells.csv}` are the P0 versions.  Until
2026-09-24 Table 14 was still built from the P0 file, so `--emit` refused to draw
FIG-2 rather than publish a figure that contradicted its own table (the tables are
now owned by `scripts/125_cl3_table_rows.py`, which regenerates them from the P1
artifact).

`FIGURES.json` records, per figure, the source file, the convention string, the
number of points and a SHA-1 digest of the exact numbers plotted.  `--check`
re-derives them from the sources; a figure whose digest no longer matches its
table is reported as drifted.

Usage
-----
    python scripts/124_fig_build.py                       # write figures + manifest
    python scripts/124_fig_build.py --check               # re-derive and compare only
    python scripts/124_fig_build.py --self-test           # guards on planted damage
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:  # a check that crashes on a legal non-ASCII title is worse than no check
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - only on exotic streams
    pass

#: IQA columns, in Table 14's presentation order (full-reference first).
METRIC_ORDER = ["iqa_psnr", "iqa_ssim", "iqa_laplacian_var", "iqa_tenengrad",
                "iqa_dark_channel", "iqa_entropy", "iqa_contrast", "iqa_saturation"]
METRIC_LABEL = {
    "iqa_psnr": "PSNR vs clean",
    "iqa_ssim": "SSIM vs clean",
    "iqa_laplacian_var": "Laplacian variance",
    "iqa_tenengrad": "Tenengrad energy",
    "iqa_dark_channel": "Dark-channel mean",
    "iqa_entropy": "Image entropy",
    "iqa_contrast": "Contrast",
    "iqa_saturation": "Saturation",
}
METRIC_KIND = {m: ("full-ref" if m in ("iqa_psnr", "iqa_ssim") else "no-ref")
               for m in METRIC_ORDER}

#: FIG-3 panels, in Table 17's order.  C1 is exactly fog + sensor_noise, which is
#: why the compound axis sits beside its two members.
DEG_ORDER = ["fog", "sensor_noise", "C1_fog_noise"]
ARM_ORDER = ["sam2video", "greedy", "dagrs"]     # memory bank, memory-free, tested
LEVELS = (1.0, 2.0, 3.0, 4.0, 5.0)

#: the arm Table 14's headline is computed on (protocol constraint 1: one arm).
MAIN_ARM = "greedy"
IQA_USABLE_RHO = 0.6

#: CL3 artefacts on the of-record P1 convention (section 5.1b).  The top-level
#: `results/iqa_config_level.{json,cells.csv}` are the P0 versions and must not be
#: used as a figure source: plotting them would re-create exactly the mismatch
#: this builder exists to refuse.
CL3_JSON = "results/p1/iqa_config_level_p1.json"
CELLS_CSV = "results/p1/iqa_config_level_p1_cells.csv"
CL3_P0_JSON = "results/iqa_config_level.json"     # kept only to prove the difference
SEV_P1_CSV = "results/p1/severity_curve_p1.csv"
#: The JSON sibling is read for one thing only: the declared subtrahend.  The
#: per-cell numbers come from the CSV, but the MEANING of the printed gap is
#: owned by `111_severity_table.py` and must not be re-assumed here.
SEV_P1_JSON = "results/p1/severity_curve_p1.json"
SEV_P0_CSV = "results/severity_curve.csv"        # kept only to prove the difference
RAW_FRAME_CSV = "results/dagrs_main_raw.csv"
TAU_CFG = "configs/_tau013.yaml"

DEFAULT_JF_INPUTS = [
    "results/p1/results/cl4_full_v2.csv",
    "results/p1/results/singles_greedy.csv",
    "results/p1/results/singles_dagrs.csv",
    "results/p1/results/singles_sam2video.csv",
    "results/p1/results/c1c4_a_dagrs.csv",
    "results/p1/results/c1c4_b_dagrs.csv",
    "results/p1/results/c1c4_a_base.csv",
    "results/p1/results/c1c4_b_base.csv",
]

FIG_DIR = "results/figs"
MANIFEST = "results/figs/FIGURES.json"
#: FIG-4's own record, written by `128_fig4_qualitative.py`.  FIG-4 is the one
#: plate this script cannot redraw: it needs a GPU pass that keeps the predicted
#: masks, which `03_run_dagrs.py` does not persist.  So `128` owns the plate and
#: the selection, and `124` owns the *manifest entry* -- with the plate's SHA-1 --
#: so the manifest cannot point at a file that no longer matches its run.
FIG4_JSON = "results/figs/FIG4.json"

#: file stem per figure -- one constant each, used by BOTH the writer and the
#: manifest, so the name a reader looks for and the name that gets written cannot
#: drift apart.  F4's writer is `128_fig4_qualitative.py`.
FIG_NAME = {"F1": "F1_overview", "F2": "F2_iqa_scatter", "F3": "F3_severity",
            "F4": "F4_qualitative", "F5": "F5_trajectory"}
#: every figure this script owns -- ONE constant, so a new figure cannot be added to
#: the builder but forgotten by the checker (which is how F1 first broke the suite).
ALL_FIGS = ["1", "2", "3", "4", "5"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(rel: str) -> List[Dict[str, str]]:
    p = ROOT / rel
    if not p.exists():
        raise SystemExit(f"missing source artifact: {rel}")
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _digest(values: Sequence[float]) -> str:
    """SHA-1 over the exact numbers, so drift is detectable and the file stays small."""
    h = hashlib.sha1()
    for v in values:
        h.update(("%.9g" % float(v)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _tau_vacuity() -> float:
    """Read tau_vacuity from the RUNNING config instead of hard-coding it.

    Hard-coding 0.13 here would be the same class of mistake red line 14 describes:
    the value is a property of the launcher, not of this script.
    """
    p = ROOT / TAU_CFG
    if not p.exists():
        raise SystemExit(f"missing config: {TAU_CFG}")
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("tau_vacuity:"):
            return float(s.split(":", 1)[1].strip())
    raise SystemExit(f"tau_vacuity not found in {TAU_CFG}")


def _save(fig, name: str, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {}
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(str(p), bbox_inches="tight")
        outs[ext] = str(p.relative_to(ROOT)).replace("\\", "/")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return outs


# --------------------------------------------------------------------------- #
# FIG-2: image quality vs downstream J&F, segment-level `across`
# --------------------------------------------------------------------------- #

def _cross_check_msg(mk: str, rho: float, n: int,
                     declared: Dict[str, Any]) -> Optional[str]:
    """The disagreement, as one line -- or None when the figure matches its table.

    This is the single place the comparison lives.  `_cross_check` turns a message
    into a raise (fail-closed emission); `_fig2_disagreements` turns it into a list
    (the withheld record, and the checker's re-test of that record).  Two copies of
    the comparison would let the raising path and the reporting path drift apart.
    """
    if n != int(declared["n"]) or abs(rho - float(declared["rho"])) > 5e-7:
        return (f"{mk}: figure rho={rho:.9f} n={n} "
                f"vs table rho={float(declared['rho']):.9f} n={declared['n']}")
    return None


def _cross_check(mk: str, rho: float, n: int, declared: Dict[str, Any]) -> None:
    """The figure may not disagree with the table it illustrates.

    Kept as a pure function so `--self-test` can exercise it on synthetic values:
    a guard whose own test can only run when the data is already consistent cannot
    tell "the guard is broken" from "the data is broken", and this check is
    fail-closed precisely so that the second case is loud.
    """
    msg = _cross_check_msg(mk, rho, n, declared)
    if msg:
        raise SystemExit(f"FIG-2/Table 14 disagreement on {msg}. "
                         f"The figure and the table are built from different data.")


def _fig2_disagreements(fig: Dict[str, Any]) -> List[str]:
    """Figure-vs-table disagreements in a derived FIG-2 dict, in the same wording.

    Wording matters: the checker compares this list against the one recorded in the
    manifest, so the two must be produced by the same formatter from the same
    numbers or every run would report a spurious "the reason changed".
    """
    out: List[str] = []
    for mk, s in (fig.get("series") or {}).items():
        msg = _cross_check_msg(mk, s["rho"], s["n"],
                               {"rho": s["table_rho"], "n": s["table_n"]})
        if msg:
            out.append(msg)
    return out


def fig2_points(arm: str, jf_inputs: Sequence[str]):
    """The 360 scatter points per metric, in the convention Table 14-15 reports.

    Reuses scripts/109's `discover_cells` for the y axis -- the production path --
    so the figure and the correlation cannot disagree about what y is.
    """
    m109 = _load("m109", "scripts/109_iqa_config_level.py")
    inputs = [Path(p) if Path(p).is_absolute() else ROOT / p for p in jf_inputs]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"missing J&F inputs: {missing}")
    cells, jf = m109.discover_cells(inputs)

    iqa = {}
    for r in _rows(CELLS_CSV):
        iqa[(r["seq"], r["degradation"], float(r["level"]), int(r["frame_stride"]))] = r

    pts: List[Dict[str, Any]] = []
    for k, c in sorted(cells.items()):
        if c["degradation"] == "clean":       # protocol constraint 2: degraded only
            continue
        y = jf.get((c["seq"], c["degradation"], float(c["level"]), arm))
        row = iqa.get(k)
        if y is None or row is None:
            continue
        pts.append({"seq": c["seq"], "degradation": c["degradation"],
                    "level": float(c["level"]), "y": float(y),
                    "x": {m: float(row[f"{m}_mean"]) for m in METRIC_ORDER}})
    return pts


def build_fig2(manifest: Dict[str, Any], jf_inputs: Sequence[str],
               cl3_json: Path, out_dir: Path, emit: bool,
               cross_check: bool = True) -> Dict[str, Any]:
    """Derive FIG-2's numbers, and only then (if `emit`) draw it.

    The two phases are kept apart because `--check` must be able to run without
    matplotlib at all, and because a checker that silently redraws 20 figures ends
    up testing the drawing code rather than the numbers.
    """
    import numpy as np

    pts = fig2_points(MAIN_ARM, jf_inputs)
    n_seq = len({p["seq"] for p in pts})
    n_deg = len({p["degradation"] for p in pts})
    if (len(pts), n_seq, n_deg) != (360, 30, 12):
        raise SystemExit(f"FIG-2 cell count wrong: {len(pts)} points, "
                         f"{n_seq} sequences, {n_deg} degradations (want 360/30/12)")

    ref = json.loads(cl3_json.read_text(encoding="utf-8"))
    ref_arm = ref["results"].get(MAIN_ARM)
    if ref_arm is None:
        raise SystemExit(f"{cl3_json} has no arm {MAIN_ARM!r}")

    from xdrp.stats import spearman_with_ci

    # ---- phase 1: the numbers (no matplotlib) ------------------------------ #
    series: Dict[str, Any] = {}
    xy: Dict[str, Tuple[Any, Any]] = {}
    disagreements: List[str] = []
    for mk in METRIC_ORDER:
        x = np.array([p["x"][mk] for p in pts], float)
        y = np.array([p["y"] for p in pts], float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        rho = float(spearman_with_ci(list(x), list(y), n_boot=0)["rho"])
        declared = ref_arm["metrics"][mk]["across"]
        msg = _cross_check_msg(mk, rho, len(x), declared)
        if msg:
            disagreements.append(msg)
        series[mk] = {"n": int(len(x)), "rho": rho, "r2": rho * rho,
                      "usable": int(abs(rho) >= IQA_USABLE_RHO),
                      "ci_lo_cluster": float(ref_arm["metrics"][mk]["across_ci_lo_cluster"]),
                      "ci_hi_cluster": float(ref_arm["metrics"][mk]["across_ci_hi_cluster"]),
                      # what the TABLE declares, so a checker can compare the two
                      # without the cross-check having already raised
                      "table_rho": float(declared["rho"]), "table_n": int(declared["n"]),
                      "x_digest": _digest(x), "y_digest": _digest(y)}
        xy[mk] = (x, y)

    # Fail-closed emission, made *explicit* instead of silent.  A plate that
    # contradicts the table it illustrates is never written; what is recorded
    # instead is WHY it is absent, and `check()` re-tests that record on every run.
    # A bare "skip me" flag would silence exactly the defect it was added for, so a
    # withholding that has gone stale is itself a failure.
    withheld = None
    if disagreements and cross_check:
        withheld = {
            "reason": "the plotted points disagree with the table they illustrate, so "
                      "the figure and the table are built from different data "
                      "(a P0/P1 mix does this)",
            "disagreements": disagreements,
            "not_written": True,
        }
        print(f"[124] FIG-2 WITHHELD -- no plate written; {len(disagreements)} of "
              f"{len(METRIC_ORDER)} metrics disagree with the declared table:")
        for line in disagreements:
            print(f"[124]   {line}")

    fig = None
    files: Dict[str, str] = {}
    if emit and withheld is None:
        # ---- phase 2: draw ------------------------------------------------- #
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.4), squeeze=False)
        for i, mk in enumerate(METRIC_ORDER):
            ax = axes[i // 4][i % 4]
            x, y = xy[mk]
            s = series[mk]
            ax.scatter(x, y, s=7, alpha=0.30, color="#374151", linewidths=0, zorder=2)
            # rank-binned median: the honest visual summary of "no usable trend"
            o = np.argsort(x)
            bx, by = [], []
            for b in range(10):
                sl = o[b * len(x) // 10:(b + 1) * len(x) // 10]
                if sl.size:
                    bx.append(float(np.median(x[sl])))
                    by.append(float(np.median(y[sl])))
            ax.plot(bx, by, color="#b91c1c", lw=1.6, marker="o", ms=3, zorder=3)
            ax.axhline(float(np.median(y)), color="#9ca3af", lw=0.8, ls="--", zorder=1)
            ax.set_title(METRIC_LABEL[mk], fontsize=8.5)
            ax.set_xlabel(f"{METRIC_LABEL[mk]} ({METRIC_KIND[mk]})", fontsize=7.5)
            if i % 4 == 0:
                ax.set_ylabel(r"downstream $\mathcal{J}$&$\mathcal{F}$", fontsize=8)
            ax.text(0.03, 0.96,
                    r"$\rho$=%.3f  $R^2$=%.3f" % (s["rho"], s["r2"]) + "\n"
                    + "cluster CI [%.2f, %.2f]" % (s["ci_lo_cluster"], s["ci_hi_cluster"]) + "\n"
                    + "usable = %d  (|$\\rho$| $\\geq$ %.1f)" % (s["usable"], IQA_USABLE_RHO),
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.6,
                    bbox=dict(fc="white", ec="#d1d5db", lw=0.6, alpha=0.9, pad=2.2))
            ax.tick_params(labelsize=7)
        fig.suptitle("Image quality does not predict downstream segmentation quality\n"
                     "segment-level $\\it{across}$ view, arm `%s`, %d (sequence $\\times$ degradation) "
                     "cells, degraded only" % (MAIN_ARM, len(pts)), fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        files = _save(fig, FIG_NAME["F2"], out_dir)

    out = {"id": "FIG-2", "title": "Image quality vs downstream J&F",
           "source": [CELLS_CSV] + list(jf_inputs),
           "table": "Table 14, Table 15",
           "convention": "segment-level `across`; unit = (sequence x degradation) cell; "
                         "one arm (%s); degraded cells only; n = %d" % (MAIN_ARM, len(pts)),
           "n_points": len(pts), "n_sequences": n_seq, "n_degradations": n_deg,
           "arms": [MAIN_ARM], "figure_name": FIG_NAME["F2"], "files": files,
           "series": series}
    if withheld:
        out["withheld"] = withheld
    return out


# --------------------------------------------------------------------------- #
# FIG-1: method overview (schematic, but with its content CHECKED against the code)
# --------------------------------------------------------------------------- #

def bank_names() -> List[str]:
    """The operator names and order the code actually builds.

    FIG-1 lists the bank, so the list in the figure is a *claim about the code*.
    Reading it here is what turns an unowned illustration into a checked one: add
    an operator and the figure is reported stale rather than quietly out of date.
    """
    from xdrp.operators import OperatorBank
    return OperatorBank().names


def build_fig1(out_dir: Path, emit: bool) -> Dict[str, Any]:
    names = bank_names()
    if len(names) != 13:
        raise SystemExit(f"FIG-1 assumes the 13-operator default bank; code has {len(names)}")

    files: Dict[str, str] = {}
    if emit:
        _draw_fig1(names, out_dir)
        for ext in ("png", "pdf"):
            files[ext] = str((Path(FIG_DIR) / f"{FIG_NAME['F1']}.{ext}"))

    return {"id": "FIG-1", "title": "DAG-RS overview",
            "source": ["xdrp/operators.py", "docs/MANUSCRIPT.md"],
            "table": "Sec. 4.1, Sec. 4.2 (text; no table)",
            "figure_name": FIG_NAME["F1"],
            "convention": "vector schematic; the operator list drawn is read from "
                          "xdrp.operators.OperatorBank, not transcribed by hand",
            "n_operators": len(names), "operators": names,
            "operator_digest": _digest([float(len(names))]) + hashlib.sha1(
                "|".join(names).encode("ascii")).hexdigest()[:16],
            "files": files}


def _draw_fig1(names: Sequence[str], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    # A "snake": the forward pass runs left -> right along the top, drops down, and
    # the feedback returns right -> left along the bottom.  No caption is drawn into
    # the image -- journals typeset the caption themselves, and a caption baked into
    # the canvas collides with the diagram.
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 40)
    ax.axis("off")

    INK, BLUE, RED, GREY = "#111827", "#1d4ed8", "#b91c1c", "#6b7280"

    def box(x, y, w, h, text, ec=INK, fc="white", fs=8.0, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.4",
                                    ec=ec, fc=fc, lw=1.2, linestyle=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=ec, zorder=3, linespacing=1.45,
                fontweight=("bold" if bold else "normal"))

    def arrow(x0, y0, x1, y1, color=GREY, ls="-", rad=0.0, lw=1.3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=color, lw=lw, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))

    YT, HT = 24.0, 8.5          # top row (forward pass, left -> right)
    YB, HB = 6.0, 8.5           # bottom row (gate + feedback, right -> left)
    R = YT + HT / 2             # centre line of the top row

    box(1.0, YT, 12.5, HT, "degraded\nframe $x_t$", fs=8.8, bold=True)
    box(17.0, 16.5, 28.0, 20.0, "", ec=BLUE, ls="--")
    ax.text(31.0, 35.2, "restoration operator bank $\\mathcal{R}$",
            ha="center", va="center", fontsize=8.2, color=BLUE, zorder=3)
    ax.text(31.0, 33.3, "$M$ = %d, training-free, deterministic" % len(names),
            ha="center", va="center", fontsize=7.0, color=BLUE, zorder=3)
    half = (len(names) + 1) // 2
    for i, nm in enumerate(names):
        col, row = divmod(i, half)
        ax.text(18.4 + col * 13.8, 31.4 - row * 1.85, "%d %s" % (i, nm),
                ha="left", va="center", fontsize=6.0, color=INK, zorder=3)
    box(47.0, YT, 13.5, HT, "segmentation\nbackend", fs=8.2)
    box(62.5, YT, 14.0, HT, "candidate\nmasks $C_m$", fs=8.2)
    box(78.5, YT, 15.5, HT, "downstream\nagreement\n$a_m,\\ c_m,\\ u_m$", fs=7.8)
    box(96.0, YT, 16.0, HT, "Dirichlet\nevidence $\\alpha$", fs=8.2)

    box(90.0, YB, 22.0, HB, "fusion $\\rightarrow$ $\\hat M_t$\n+ vacuity $\\nu$",
        ec=RED, fs=8.4, bold=True)
    box(72.5, YB, 15.0, HB, "gate on $\\nu$\n(decision frames)", ec=RED, fs=8.0, bold=True)
    box(54.5, YB, 15.5, HB, "accept / reject\nre-anchor", fs=7.8)
    box(30.0, YB, 22.0, HB, "degradation profile\n$\\pi_t$ update", ec=BLUE, fs=8.4)

    for x0, x1 in ((13.5, 17.0), (45.0, 47.0), (60.5, 62.5), (76.5, 78.5),
                   (94.0, 96.0)):
        arrow(x0, R, x1, R)
    arrow(104.0, YT, 104.0, YB + HB + 0.5, color=RED)               # down into fusion
    arrow(90.0, YB + HB / 2, 87.5, YB + HB / 2, color=RED)          # fusion -> gate
    arrow(72.5, YB + HB / 2, 70.0, YB + HB / 2)                     # gate -> outcomes
    arrow(54.5, YB + HB / 2, 52.0, YB + HB / 2)                     # outcomes -> profile
    # feedback: the profile is consulted at the next keyframe
    arrow(30.0, YB + 4.6, 20.0, 16.2, color=BLUE, ls="--", rad=0.22)
    ax.text(17.4, 13.4, "consulted at the\nnext keyframe", ha="left", va="bottom",
            fontsize=6.6, color=BLUE, zorder=3)
    ax.text(1.0, 1.6, "forward pass (top) and the gate / feedback loop (bottom); "
            "no learned parameter anywhere in the loop",
            ha="left", va="center", fontsize=6.8, color=GREY)
    fig.tight_layout()
    _save(fig, FIG_NAME["F1"], out_dir)


# --------------------------------------------------------------------------- #
# FIG-3: severity curves (P1)
# --------------------------------------------------------------------------- #

def sev_cells(rel: str) -> Dict[Tuple[str, float, str], Dict[str, float]]:
    out = {}
    for r in _rows(rel):
        out[(r["degradation"], float(r["level"]), r["mode"])] = {
            "mean": float(r["J&F_mean"]), "lo": float(r["ci_lo"]),
            "hi": float(r["ci_hi"]), "n": int(float(r["n"]))}
    return out


def build_fig3(manifest: Dict[str, Any], out_dir: Path, emit: bool) -> Dict[str, Any]:
    cells = sev_cells(SEV_P1_CSV)
    want = {(d, lv, a) for d in DEG_ORDER for lv in LEVELS for a in ARM_ORDER}
    missing = sorted(want - set(cells))
    if missing:
        raise SystemExit(f"FIG-3 source incomplete, missing {len(missing)} cells: {missing[:4]}")
    ns = {cells[k]["n"] for k in want}
    if ns != {30}:
        raise SystemExit(f"FIG-3 cells do not all rest on 30 sequences: {sorted(ns)}")

    p0 = sev_cells(SEV_P0_CSV) if (ROOT / SEV_P0_CSV).exists() else {}
    drift = sorted(k for k in want if k in p0
                   and abs(p0[k]["mean"] - cells[k]["mean"]) > 1e-9)

    # ---- the subtrahend is READ from the owning script, never assumed here ----
    # The panel titles print a gap, and an UNNAMED gap is exactly how Sec. 5.6
    # came to disagree with Sec. 5.2's definition without anything failing: the
    # title said "family gap", `111` subtracted `dagrs`, and the definition said
    # `greedy`.  So this builder reads the division of labour out of the artifact
    # `111` writes and refuses to draw when it is absent.
    sev_json = json.loads((ROOT / SEV_P1_JSON).read_text(encoding="utf-8"))
    subtrahends = sev_json.get("family_gap_subtrahends") or {}
    if not subtrahends.get("memory_arm") or not subtrahends.get("primary"):
        raise SystemExit(
            f"FIG-3 refused: {SEV_P1_JSON} does not declare `family_gap_subtrahends`, "
            f"so the panel titles would print an unnamed gap. Regenerate it with "
            f"scripts/111_severity_table.py (the subtrahend IS the estimand).")
    mem, sub = subtrahends["memory_arm"], subtrahends["primary"]

    # ---- phase 1: the numbers, including the values printed in the panel titles
    gaps: Dict[str, Dict[str, float]] = {}
    for deg in DEG_ORDER:
        g1 = cells[(deg, 1.0, mem)]["mean"] - cells[(deg, 1.0, sub)]["mean"]
        g5 = cells[(deg, 5.0, mem)]["mean"] - cells[(deg, 5.0, sub)]["mean"]
        gaps[deg] = {"gap_1": g1, "gap_5": g5, "shift": g5 - g1}

    # Cross-owner reconciliation.  These gaps are recomputed here from the per-cell
    # CSV, so they must equal what the OWNING script publishes for the same
    # quantity -- two independent paths over the same cells.  If they disagree one
    # of the two is wrong, and a figure that prints its own arithmetic is exactly
    # how a figure drifts away from its table.
    own = {(g["degradation"], g["level"]): g for g in sev_json.get("family_gap", [])}
    cross: List[str] = []
    for deg in DEG_ORDER:
        e = (own.get((deg, LEVELS[-1])) or {}).get("endpoint")
        if not e:
            cross.append(f"{deg}: the owner's JSON carries no endpoint block")
            continue
        for mine, theirs, what in ((gaps[deg]["gap_1"], e["gap_l1"], "gap@1"),
                                   (gaps[deg]["gap_5"], e["gap_l5"], "gap@5")):
            if abs(mine - theirs) > 1e-12:
                cross.append(f"{deg} {what}: figure {mine:.10f} != owner {theirs:.10f}")
    if cross:
        raise SystemExit("FIG-3 refused: the panel-title gaps disagree with their "
                         "owning script:\n  " + "\n  ".join(cross))

    # The other subtrahend, carried in the manifest so the record shows the
    # verdict is subtrahend-dependent rather than leaving one number to stand in
    # for two different estimands.
    sens = subtrahends.get("sensitivity")
    sens_shift = {g["degradation"]: g["endpoint"]["shift"]
                  for g in sev_json.get("family_gap_sensitivity", [])
                  if "endpoint" in g} if sens else {}

    files: Dict[str, str] = {}
    if emit:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), sharey=True)
        for ax, deg in zip(axes, DEG_ORDER):
            for arm in ARM_ORDER:
                cs = [cells[(deg, lv, arm)] for lv in LEVELS]
                ax.plot(LEVELS, [c["mean"] for c in cs], marker="o", ms=4, lw=1.5,
                        color=PALETTE_ARM[arm], label=arm)
                ax.fill_between(LEVELS, [c["lo"] for c in cs], [c["hi"] for c in cs],
                                color=PALETTE_ARM[arm], alpha=0.15, lw=0)
            g = gaps[deg]
            # The subtrahend is PRINTED.  An unnamed "family gap" in a figure is a
            # number whose meaning has to be looked up somewhere else, and in this
            # project it was looked up wrongly -- the two sections used different
            # arms for weeks while both labels said "family gap".
            ax.set_title("%s\nfamily gap (%s \u2212 %s) @1 = %+.4f, @5 = %+.4f "
                         "(shift %+.4f)" % (deg, mem, sub, g["gap_1"], g["gap_5"],
                                            g["shift"]), fontsize=8.5)
            ax.set_xlabel("severity level", fontsize=8)
            ax.set_xticks(list(LEVELS))
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(r"$\mathcal{J}$&$\mathcal{F}$   (P1)", fontsize=8.5)
        axes[-1].legend(fontsize=7.5, frameon=False)
        fig.suptitle("Severity axis 1-5 for fog, sensor_noise and their compound "
                     "C1 = fog + sensor_noise\n"
                     "P1 convention (denominator = target-present frames), n = 30 sequences per "
                     "cell, band = sequence-level bootstrap 95%", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        files = _save(fig, FIG_NAME["F3"], out_dir)

    return {"id": "FIG-3", "title": "Severity curves",
            "gaps": gaps,
            # Which arm was subtracted is part of the figure's meaning, so it is
            # recorded here next to the numbers it produced rather than left to
            # whichever section a reader happens to open first.
            "subtrahend": sub, "memory_arm": mem,
            "subtrahend_source": f"{SEV_P1_JSON}:family_gap_subtrahends",
            "subtrahend_sensitivity": sens,
            "sensitivity_shift": sens_shift,
            "source": [SEV_P1_CSV, SEV_P1_JSON], "table": "Table 17",
            "figure_name": FIG_NAME["F3"],
            "convention": "P1 (denominator = target-present frames); per-cell point = "
                          "sequence-equal mean over 30 sequences; band = 95% sequence "
                          f"bootstrap. Panel-title gaps are {mem} \u2212 {sub} (the arm "
                          f"under test); the other defensible subtrahend"
                          + (f" ({sens}) " if sens else " ")
                          + "is reported in the manuscript as a sensitivity, because "
                            "the two disagree in sign on fog and C1_fog_noise",
            "n_cells": len(want), "n_sequences_per_cell": 30,
            "cells_differing_from_P0": len(drift),
            "max_abs_P0_minus_P1": max((abs(p0[k]["mean"] - cells[k]["mean"])
                                        for k in want if k in p0), default=0.0),
            "series": {f"{d}|{lv:g}|{a}": {"value": cells[(d, lv, a)]["mean"],
                                           "lo": cells[(d, lv, a)]["lo"],
                                           "hi": cells[(d, lv, a)]["hi"]}
                       for d in DEG_ORDER for lv in LEVELS for a in ARM_ORDER},
            "value_digest": _digest([cells[(d, lv, a)]["mean"]
                                     for d in DEG_ORDER for lv in LEVELS for a in ARM_ORDER]),
            "files": files}


PALETTE_ARM = {"sam2video": "#1d4ed8", "greedy": "#6b7280", "dagrs": "#b91c1c"}


# --------------------------------------------------------------------------- #
# FIG-5: per-frame diagnostics on one instance
# --------------------------------------------------------------------------- #

def pick_trajectory(rows: Sequence[Dict[str, str]]) -> Tuple[str, str, str]:
    """Deterministic choice of the instance FIG-5 shows.

    Rule (stated so the figure is reproducible rather than cherry-picked): among
    `dagrs` instances in the compound C1 phase, take the one with the largest
    number of **decision** frames; ties break on `(seq, obj_id)`.  A reduced
    frame count would give the profile less to migrate over.
    """
    n_dec: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        k = (r["seq"], r["obj_id"], r["degradation"])
        n_dec[k] = n_dec.get(k, 0) + (1 if r["is_keyframe"] == "1" else 0)
    c1 = [(v, k) for k, v in n_dec.items() if k[2] == "C1_fog_noise" and v > 0]
    if not c1:
        raise SystemExit("no decision frames in the C1 phase")
    _, best = max(c1, key=lambda t: (t[0], -1 * 0, t[1]))
    return best


def build_fig5(out_dir: Path, emit: bool) -> Dict[str, Any]:
    import numpy as np

    rows = _rows(RAW_FRAME_CSV)
    if not rows or rows[0].get("vacuity") is None:
        raise SystemExit(f"{RAW_FRAME_CSV} carries no per-frame diagnostics")
    seq, obj, deg = pick_trajectory(rows)
    inst = [r for r in rows if (r["seq"], r["obj_id"], r["degradation"]) == (seq, obj, deg)]
    inst.sort(key=lambda r: int(r["frame"]))
    tau = _tau_vacuity()

    fr = np.array([int(r["frame"]) for r in inst])
    nu = np.array([float(r["vacuity"]) for r in inst], float)
    err = np.array([1.0 - float(r["J_frame"]) for r in inst], float)
    op = np.array([int(r["selected_op"]) for r in inst], int)
    nev = np.array([int(r["n_evaluated"]) for r in inst], int)
    dec = np.array([int(r["is_keyframe"]) for r in inst], int)

    n_dec = int(dec.sum())
    n_nu = int(np.isfinite(nu).sum())
    if n_dec == 0 or n_nu == 0:
        raise SystemExit("FIG-5 has nothing to plot: no decision frames / no nu values")
    over = int((np.isfinite(nu) & (nu >= tau) & (dec == 1)).sum())

    files: Dict[str, str] = {}
    if emit:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9.4, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})
        ax0, ax1, ax2 = axes
        # A step plot, not a colour strip: with 13 operators a discrete ramp needs a
        # legend to be readable, and a legend is exactly what a 1-row image cannot carry.
        ax0.step(fr, op, where="post", color="#b91c1c", lw=1.2,
                 label=r"profile's leading operator $\arg\max_m \pi_{t,m}$")
        ax0.scatter(fr[dec == 1], op[dec == 1], s=13, facecolor="none",
                    edgecolor="#374151", linewidths=0.8, zorder=3,
                    label="decision frame (%d)" % n_dec)
        ax0.set_ylabel("operator\nindex", fontsize=8)
        ax0.set_ylim(-0.6, max(4.5, float(op.max()) + 0.8))
        ax0.set_title("which restoration operator the degradation profile leads with, "
                      "and how many it evaluates ($J$=%d when warm, bank of %d when cold)"
                      % (int(np.median(nev[dec == 1])), int(nev.max())), fontsize=8.5)
        ax0.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)

        ax1.plot(fr, nu, color="#b91c1c", lw=1.3, label=r"vacuity $\nu$")
        ax1.axhline(tau, color="#374151", lw=1.0, ls="--",
                    label=r"$\tau_\nu$ = %g (configs/_tau013.yaml)" % tau)
        d = dec == 1
        ax1.scatter(fr[d], nu[d], s=11, facecolor="none", edgecolor="#b91c1c",
                    linewidths=0.7, zorder=3, label="gate evaluated (%d frames)" % n_dec)
        # zero-based axis: a window of 0.115-0.135 would magnify a 0.02 ripple into a
        # dramatic-looking climb.  The panel's job is to show that nu sits in a band
        # around the threshold and tracks the error, not to maximise apparent motion.
        ax1.set_ylim(0.0, float(np.nanmax(nu)) * 1.25)
        ax1.set_ylabel("vacuity", fontsize=8)
        ax1.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)

        ax2.plot(fr, err, color="#1d4ed8", lw=1.3, label=r"true error ($1-J$)")
        ax2.set_ylabel("error", fontsize=8)
        ax2.set_xlabel("frame", fontsize=8)
        ax2.legend(fontsize=7, frameon=False, loc="upper left")

        fig.suptitle("DAG-RS per-frame diagnostics, instance %s obj %s, %s\n"
                     "%d frames, %d decision frames, %d of them with $\\nu \\geq \\tau_\\nu$"
                     % (seq, obj, deg, len(fr), n_dec, over), fontsize=9.5)
        for ax in axes:
            ax.tick_params(labelsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        files = _save(fig, FIG_NAME["F5"], out_dir)

    return {"id": "FIG-5", "title": "Per-frame mechanism diagnostics",
            "source": [RAW_FRAME_CSV], "table": "Sec. 5.8 / Sec. 5.10",
            "figure_name": FIG_NAME["F5"],
            "convention": "one instance, frame-by-frame; the profile's selected operator "
                          "(argmax of pi) -- the full 13-dim pi mass is NOT recorded by the "
                          "pipeline and would need an opt-in trace",
            "instance": {"seq": seq, "obj_id": obj, "degradation": deg},
            "n_frames": int(len(fr)), "n_decision_frames": n_dec,
            "n_over_tau": over, "tau_vacuity": tau,
            "selection_rule": "max decision frames in the C1 phase, tie-break (seq, obj_id)",
            "series": {"frame": fr.tolist(), "vacuity": nu.tolist(), "error": err.tolist(),
                       "selected_op": op.tolist(), "n_evaluated": nev.tolist()},
            "nu_digest": _digest([v for v in nu]), "err_digest": _digest([v for v in err]),
            "files": files}


# --------------------------------------------------------------------------- #
# build / check / self-test
# --------------------------------------------------------------------------- #

def build_fig4(out_dir: Path, emit: bool, fig4_json: Path | None = None) -> Dict[str, Any]:
    """FIG-4's manifest entry: the plate is owned by `128_fig4_qualitative.py`.

    Nothing here redraws anything, in either mode.  What is transcribed is the
    selection `128` made -- three instances chosen by one stated rule, with the
    frame rule and the ranking subtrahend -- plus the plate's SHA-1, so a manifest
    entry that points at a figure which no longer matches its run is *detectable*
    rather than merely unlikely.
    """
    p = Path(fig4_json) if fig4_json is not None else ROOT / FIG4_JSON
    entry: Dict[str, Any] = {
        "id": "FIG-4",
        "title": "Qualitative panel (clean / degraded / memory bank / arm)",
        "table": "Sec. 5.8, Sec. 6.3",
        "figure_name": FIG_NAME["F4"],
        "source": [], "files": {},
    }
    if not p.exists():
        entry["convention"] = ("not built: scripts/128_fig4_qualitative.py needs one GPU "
                               "pass that keeps the predicted masks")
        entry["status"] = "absent"
        return entry
    d = json.loads(p.read_text(encoding="utf-8"))
    art = d.get("art") or {}
    sel = d.get("selection") or {}
    png_rel = art.get("png")
    fp = (ROOT / png_rel) if png_rel and not Path(png_rel).is_absolute() else (
        Path(png_rel) if png_rel else None)
    entry.update({
        "source": sorted((sel.get("source_sha1") or {}).keys()),
        "convention": d.get("convention", ""),
        "selection_rule": sel.get("selection_rule", ""),
        "frame_rule": d.get("frame_rule", ""),
        "subtrahend": sel.get("ranking_subtrahend"),
        "n_instances_ranked": sel.get("n_instances_ranked"),
        "columns": [{k: c.get(k) for k in ("role", "seq", "obj_id", "delta_vs_greedy")}
                    for c in (sel.get("columns") or [])],
        "value_digest": art.get("value_digest"),
        "png_sha1": art.get("png_sha1"),
        "files": {ext: art[ext] for ext in ("png", "pdf") if art.get(ext)},
        "status": "present",
    })
    if fp is None or not fp.exists():
        entry["status"] = "missing-plate"
    elif _file_sha1(fp) != art.get("png_sha1"):
        entry["status"] = "stale-plate"
    return entry


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(jf_inputs: Sequence[str], cl3_json: Path, out_dir: Path, emit: bool,
          only: Sequence[str], cross_check: bool = True) -> Dict[str, Any]:
    figs: Dict[str, Any] = {}
    if "1" in only:
        figs["F1"] = build_fig1(out_dir, emit)
    if "2" in only:
        figs["F2"] = build_fig2({}, jf_inputs, cl3_json, out_dir, emit,
                                cross_check=cross_check)
    if "3" in only:
        figs["F3"] = build_fig3({}, out_dir, emit)
    if "4" in only:
        figs["F4"] = build_fig4(out_dir, emit)
    if "5" in only:
        figs["F5"] = build_fig5(out_dir, emit)
    return {"generated_by": "scripts/124_fig_build.py",
            "note": "Each entry names the artifact its table is built from and a SHA-1 "
                    "digest of the exact numbers drawn. A figure whose digest no longer "
                    "matches its table has drifted.",
            "figures": figs}


def check(manifest_path: Path, jf_inputs: Sequence[str], cl3_json: Path,
          out_dir: Path, cross_check: bool = False) -> List[str]:
    """Return the list of problems. Never raises: a checker that aborts gives no verdict.

    `cross_check` is False by default because this function *reports* a
    figure/table disagreement itself (as a failure line); letting the derivation
    raise instead would replace a verdict with a traceback.  `--emit` keeps the
    raising behaviour, so a figure that contradicts its table is never written.
    """
    fails: List[str] = []
    if not manifest_path.exists():
        return [f"[124] no {manifest_path.name}; run without --check first"]
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A manifest without a `figures` map is a malformed input, not a crash: the
    # checker must return a verdict for every input it is handed.
    if not isinstance(doc, dict) or not isinstance(doc.get("figures"), dict) \
            or not doc["figures"]:
        return [f"[124] {manifest_path.name} has no `figures` map -- regenerate it"]
    old = doc["figures"]
    new = build(jf_inputs, cl3_json, out_dir, emit=False,
                only=ALL_FIGS, cross_check=cross_check)["figures"]

    for key, nd in new.items():
        od = old.get(key)
        if od is None:
            fails.append(f"[124] {key}: absent from the manifest")
            continue
        if key == "F4" and nd.get("status") != "present":
            fails.append(f"[124] F4: {nd.get('status')} -- Sec. 5.8 promises this plate; "
                         f"run scripts/128_fig4_qualitative.py")
            continue
        # A withheld figure is a CLAIM -- "no plate, because it would contradict its
        # table" -- and the claim is re-tested here rather than trusted.  Otherwise
        # `withheld` is a permanent silencer: on the day the disagreement is fixed the
        # figure would stay missing while the suite reported green.
        if od.get("withheld"):
            rec = list(od["withheld"].get("disagreements") or [])
            dis = _fig2_disagreements(nd)
            if not rec:
                fails.append(f"[124] {key}: withheld with no recorded disagreement -- a "
                             f"withholding must name what it is waiting on")
            elif not dis:
                fails.append(f"[124] {key}: withheld, but it now agrees with its table "
                             f"-- emit it and drop the `withheld` block")
            elif rec != dis:
                fails.append(f"[124] {key}.withheld: the reason changed -- manifest "
                             f"{rec} vs data {dis}")
            continue
        # `files` is deliberately NOT compared for equality: a non-emitting
        # derivation writes nothing, so the two sides would always differ and the
        # check would cry wolf on every run.  The on-disk truth is handled by
        # `_stale()` below, which reads the manifest's own `files`.
        for field in ("source", "table", "convention", "figure_name"):
            if od.get(field) != nd.get(field):
                fails.append(f"[124] {key}.{field}: manifest {od.get(field)!r} "
                             f"vs data {nd.get(field)!r}")
        for field in ("nu_digest", "err_digest", "value_digest", "png_sha1"):
            if field in nd and od.get(field) != nd.get(field):
                fails.append(f"[124] {key}.{field}: the plotted numbers changed "
                             f"(manifest {od.get(field)} vs data {nd.get(field)}) -- "
                             f"either the figure is stale or its source moved")
        if "series" in nd and key == "F2":
            for mk, s in nd["series"].items():
                # (a) the figure vs the TABLE it illustrates
                if s["n"] != s["table_n"] or abs(s["rho"] - s["table_rho"]) > 5e-7:
                    fails.append(
                        f"[124] F2.{mk}: the figure and Table 14 disagree -- points give "
                        f"rho={s['rho']:.9f} n={s['n']} while the table declares "
                        f"rho={s['table_rho']:.9f} n={s['table_n']}; the two are built from "
                        f"different data (a P0/P1 mix does this)")
                # (b) the manifest vs the data now on disk
                o = od.get("series", {}).get(mk)
                if o is None:
                    fails.append(f"[124] F2.{mk}: absent from the manifest")
                    continue
                for f in ("n", "x_digest", "y_digest"):
                    if o.get(f) != s.get(f):
                        fails.append(f"[124] F2.{mk}.{f}: manifest {o.get(f)} vs data {s.get(f)}")
                if abs(float(o.get("rho", 9)) - s["rho"]) > 5e-7:
                    fails.append(f"[124] F2.{mk}.rho: manifest {o.get('rho')} vs data {s['rho']}")
        for f in _stale(nd, ROOT):
            fails.append(f"[124] {key}: {f}")
    return fails


def _stale(fig: Dict[str, Any], root: Path) -> List[str]:
    """Figures that exist but are older than the data they were drawn from.

    A figure being *present* says nothing about it being *current*: the two PNGs
    this script replaces were both present and both wrong.  Extracted so the
    self-test can drive it on synthetic paths instead of having to mutate the
    mtimes of the production figures.
    """
    out: List[str] = []
    src_m = [ (root / s).stat().st_mtime for s in fig.get("source", [])
              if (root / s).exists() ]
    if not src_m:
        return out
    oldest = min(src_m)
    for ext, rel in (fig.get("files") or {}).items():
        fp = root / rel
        if fp.exists() and fp.stat().st_mtime < oldest - 1.0:
            out.append(f"{ext}: figure is older than its source ({rel}) -- regenerate")
    return out


def self_test() -> int:
    """Controls: each planted damage must be caught by `check`."""
    import tempfile
    checks: List[Tuple[str, bool]] = []

    def verdict(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    jf = DEFAULT_JF_INPUTS
    real_cl3 = ROOT / CL3_JSON
    d = ROOT / FIG_DIR

    with tempfile.TemporaryDirectory(prefix="124_selftest_") as td:
        mp = Path(td) / "FIGURES.json"

        # The guards under test must be exercisable whether or not the production
        # data happens to be consistent -- otherwise "the guard is broken" and "the
        # data is broken" produce the same non-result.  So the baseline uses a
        # SYNTHETIC table whose declared rho is derived from the same points; the
        # real (inconsistent) pair gets its own control at the end.
        import numpy as np
        from xdrp.stats import spearman_with_ci
        pts = fig2_points(MAIN_ARM, jf)
        syn = json.loads(real_cl3.read_text(encoding="utf-8"))
        for mk in METRIC_ORDER:
            x = [p["x"][mk] for p in pts]
            y = [p["y"] for p in pts]
            rho = float(spearman_with_ci(x, y, n_boot=0)["rho"])
            a = syn["results"][MAIN_ARM]["metrics"][mk]["across"]
            a["rho"], a["n"] = rho, len(pts)
        cl3 = Path(td) / "cl3_consistent.json"
        cl3.write_text(json.dumps(syn), encoding="utf-8")
        # ALL figures, so the baseline manifest covers every figure `check` derives
        # -- when F1 was added, a baseline built from ["2","3","5"] reported "F1
        # absent" and that single omission read exactly like a broken guard.
        man = build(jf, cl3, d, emit=False, only=ALL_FIGS, cross_check=False)

        def rewrite(fig_key: str, field: str, value: Any) -> None:
            m = json.loads(json.dumps(man))
            m["figures"][fig_key][field] = value
            mp.write_text(json.dumps(m), encoding="utf-8")

        # --- positive: the manifest checker accepts a clean manifest ---------- #
        mp.write_text(json.dumps(man), encoding="utf-8")
        ok_clean = check(mp, jf, cl3, d, cross_check=False) == []
        verdict("a manifest that matches its sources passes (baseline)", ok_clean)

        def fires(mutate, needle: str) -> bool:
            m = json.loads(json.dumps(man))
            mutate(m)
            mp.write_text(json.dumps(m), encoding="utf-8")
            return any(needle in f for f in check(mp, jf, cl3, d, cross_check=False))

        # --- negative controls ----------------------------------------------- #
        verdict("F2 fires: convention changed to the frame-level probe",
                fires(lambda m: m["figures"]["F2"].update(convention="frame-level probe"),
                      "F2.convention"))

        verdict("F2 fires: a plotted x series drifted",
                fires(lambda m: m["figures"]["F2"]["series"]["iqa_psnr"].update(
                    x_digest="0" * 40), "F2.iqa_psnr.x_digest"))

        verdict("F2 fires: point count changed (361 != 360)",
                fires(lambda m: m["figures"]["F2"]["series"]["iqa_ssim"].update(n=361),
                      "F2.iqa_ssim.n"))

        verdict("F2 fires: annotated rho no longer matches the table",
                fires(lambda m: m["figures"]["F2"]["series"]["iqa_psnr"].update(rho=0.9999),
                      "F2.iqa_psnr.rho"))

        verdict("F3 fires: the severity numbers changed",
                fires(lambda m: m["figures"]["F3"].update(value_digest="0" * 40),
                      "F3.value_digest"))

        verdict("F3 fires: source swapped to the P0 (retracted) file",
                fires(lambda m: m["figures"]["F3"].update(source=["results/severity_curve.csv"]),
                      "F3.source"))

        verdict("F5 fires: the plotted nu series changed",
                fires(lambda m: m["figures"]["F5"].update(nu_digest="0" * 40),
                      "F5.nu_digest"))

        mp.write_text(json.dumps({}), encoding="utf-8")
        verdict("fires: empty manifest", check(mp, jf, cl3, d, cross_check=False) != [])

        # --- FIG-3 must refuse a source whose subtrahend is undeclared -------- #
        # Sec. 5.6's real defect was an UNNAMED gap: the panel title said "family
        # gap" while the subtraction used a different arm from the one the paper
        # defines, and nothing failed for weeks.  A builder that prints a
        # subtrahend it assumed for itself is that bug again, so it READS the
        # division of labour from the owning artifact and must refuse when absent.
        keep_sev = SEV_P1_JSON
        sev_dir = d / "_sevtmp"
        sev_dir.mkdir(parents=True, exist_ok=True)
        sev0 = json.loads((ROOT / keep_sev).read_text(encoding="utf-8"))

        def point_at(mut) -> None:
            global SEV_P1_JSON
            p = sev_dir / "s.json"
            p.write_text(json.dumps(mut), encoding="utf-8")
            SEV_P1_JSON = str(p)

        def f3_raises(mut, needle: str) -> bool:
            point_at(mut)
            try:
                build_fig3({}, sev_dir / "out", emit=False)
            except SystemExit as ex:
                return needle in str(ex)
            finally:
                globals()["SEV_P1_JSON"] = keep_sev
            return False

        verdict("FIG-3 draws quietly on the real, unmutated source (baseline)",
                f3_raises(json.loads(json.dumps(sev0)), "\x00") is False)

        verdict("FIG-3 refuses a source that declares no subtrahend",
                f3_raises({k: v for k, v in sev0.items()
                           if k != "family_gap_subtrahends"}, "unnamed gap"))

        def _bump_gap_l1(m):
            for g in m["family_gap"]:
                if g["degradation"] == "fog" and "endpoint" in g:
                    g["endpoint"]["gap_l1"] += 0.01
            return m

        verdict("FIG-3 refuses when its own arithmetic disagrees with the owner",
                f3_raises(_bump_gap_l1(json.loads(json.dumps(sev0))),
                          "disagree with their owning script"))

        # --- the cross-check: figure points vs the table's declared rho ------- #
        mp.write_text(json.dumps(man), encoding="utf-8")
        # a table whose rho is off by 1e-3 must be reported, not shrugged off
        off = json.loads(json.dumps(syn))
        off["results"][MAIN_ARM]["metrics"]["iqa_ssim"]["across"]["rho"] += 1e-3
        offp = Path(td) / "cl3_off.json"
        offp.write_text(json.dumps(off), encoding="utf-8")
        verdict("check reports a table rho that no longer matches the plotted points",
                any("the figure and Table 14 disagree" in f
                    for f in check(mp, jf, offp, d, cross_check=False)))

        # --- a withheld figure: the record is re-tested, never trusted --------- #
        # `offp` perturbs exactly one metric, so the withholding is genuine here.
        wm = build(jf, offp, d, emit=False, only=ALL_FIGS, cross_check=True)
        wrec = wm["figures"]["F2"].get("withheld", {})
        wp = Path(td) / "FIGURES_withheld.json"
        wp.write_text(json.dumps(wm), encoding="utf-8")
        verdict("a withheld figure with a genuine disagreement passes",
                check(wp, jf, offp, d, cross_check=False) == [])
        verdict("the withheld record names exactly the metric that disagrees",
                len(wrec.get("disagreements") or []) == 1
                and wrec["disagreements"][0].startswith("iqa_ssim: "))
        verdict("a withheld figure writes no plate",
                wm["figures"]["F2"]["files"] == {})
        verdict("withholding fires: the disagreement is gone, so it is stale",
                any("withheld, but it now agrees" in f
                    for f in check(wp, jf, cl3, d, cross_check=False)))

        def fires_w(mutate, needle: str) -> bool:
            m = json.loads(json.dumps(wm))
            mutate(m)
            wp.write_text(json.dumps(m), encoding="utf-8")
            return any(needle in f for f in check(wp, jf, offp, d, cross_check=False))

        verdict("withholding fires: the recorded disagreement was edited",
                fires_w(lambda m: m["figures"]["F2"]["withheld"].update(
                    disagreements=["iqa_entropy: figure rho=0.0 n=360 vs table rho=0.5 n=360"]),
                    "withheld: the reason changed"))
        verdict("withholding fires: a withholding that names nothing",
                fires_w(lambda m: m["figures"]["F2"]["withheld"].update(disagreements=[]),
                        "with no recorded disagreement"))

        # --- staleness, on synthetic paths (never touches the real figures) --- #
        import os as _os
        sd = Path(td) / "stale"
        sd.mkdir()
        (sd / "src.csv").write_text("x\n", encoding="utf-8")
        (sd / "fig.png").write_text("png\n", encoding="utf-8")
        (sd / "fig.png").touch()
        _os.utime(sd / "src.csv", (2_000_000_000, 2_000_000_000))
        verdict("staleness fires: figure older than its source",
                _stale({"source": ["stale/src.csv"], "files": {"png": "stale/fig.png"}}, Path(td))
                != [])
        _os.utime(sd / "fig.png", (2_000_000_100, 2_000_000_100))
        verdict("staleness quiet: figure newer than its source",
                _stale({"source": ["stale/src.csv"], "files": {"png": "stale/fig.png"}}, Path(td))
                == [])
        verdict("staleness quiet: a figure that does not exist yet is not 'stale'",
                _stale({"source": ["stale/src.csv"], "files": {"png": "stale/none.png"}}, Path(td))
                == [])

        # --- FIG-4's manifest entry: transcribed from 128, never redrawn -------- #
        real4 = build_fig4(Path(td), emit=False)
        verdict("F4's manifest entry transcribes 128's plate and reports it present",
                real4["status"] == "present" and bool(real4["png_sha1"])
                and len(real4["columns"]) == 3
                and bool(real4["subtrahend"]) and real4["n_instances_ranked"] > 0)
        fake4 = Path(td) / "FIG4_fake.json"
        fake4.write_text(json.dumps({
            "art": {"png": "results/figs/F4_qualitative.png", "png_sha1": "0" * 40,
                    "value_digest": "deadbeef"},
            "selection": {"ranking_subtrahend": "greedy", "n_instances_ranked": 61,
                          "source_sha1": {"results/p1/results/cl4_full_v2.csv": "x"},
                          "columns": [{"role": "best (rescued)", "seq": "s", "obj_id": 1,
                                       "delta_vs_greedy": 0.1}]},
            "convention": "c", "frame_rule": "f"}), encoding="utf-8")
        verdict("F4 fires when the plate's SHA-1 is not the one 128 recorded",
                build_fig4(Path(td), emit=False, fig4_json=fake4)["status"]
                == "stale-plate")
        verdict("F4 reports absence rather than a crash when 128 has not run",
                build_fig4(Path(td), emit=False,
                           fig4_json=Path(td) / "nope.json")["status"] == "absent")

    # --- the cross-check, on synthetic values (independent of the data) ------- #
    good = {"rho": 0.25, "n": 360}
    verdict("cross-check accepts a figure that matches its table",
            _cross_check("iqa_psnr", 0.25, 360, good) is None)
    raised = False
    try:
        _cross_check("iqa_psnr", 0.250001, 360, good)
    except SystemExit:
        raised = True
    verdict("cross-check RAISES when the figure rho differs by 1e-6", raised)
    raised = False
    try:
        _cross_check("iqa_psnr", 0.25, 359, good)
    except SystemExit:
        raised = True
    verdict("cross-check RAISES when the figure has 359 points instead of 360", raised)

    # --- shape of the real FIG-2 point cloud ---------------------------------- #
    pts = fig2_points(MAIN_ARM, jf)
    verdict("F2 has 360 points over 30 sequences and 12 degradations",
            len(pts) == 360 and len({p["seq"] for p in pts}) == 30
            and len({p["degradation"] for p in pts}) == 12)
    n_seq_in_iqa = len({r["seq"] for r in _rows(CELLS_CSV)})
    verdict("the IQA cell table covers the same 30 sequences",
            n_seq_in_iqa == 30)

    seq, obj, deg = pick_trajectory(_rows(RAW_FRAME_CSV))
    verdict("F5 picks a deterministic instance and it exists in the CSV",
            deg == "C1_fog_noise"
            and any((r["seq"], r["obj_id"]) == (seq, obj) for r in _rows(RAW_FRAME_CSV)))
    verdict("F5 selection rule is order-independent",
            pick_trajectory(list(reversed(_rows(RAW_FRAME_CSV)))) == (seq, obj, deg))
    verdict("tau_vacuity is read from the config, not hard-coded",
            abs(_tau_vacuity() - 0.13) < 1e-12)

    print("[124] self-test")
    ok = True
    for name, passed in checks:
        print("   %-70s %s" % (name, "PASS" if passed else "FAIL"))
        ok = ok and passed
    print("[124] self-test: %s (%d controls)" % ("PASS" if ok else "FAIL", len(checks)))
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jf-inputs", nargs="+", default=DEFAULT_JF_INPUTS,
                    help="the J&F side of FIG-2; must be the tree Table 14 is built from")
    ap.add_argument("--cl3-json", default=CL3_JSON)
    ap.add_argument("--out-dir", default=FIG_DIR)
    ap.add_argument("--only", nargs="+", default=ALL_FIGS, choices=ALL_FIGS)
    ap.add_argument("--check", action="store_true",
                    help="re-derive from the sources and compare with the manifest only")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    cl3 = Path(args.cl3_json)
    if not cl3.is_absolute():
        cl3 = ROOT / cl3
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    if args.check:
        fails = check(ROOT / MANIFEST, args.jf_inputs, cl3, out_dir, cross_check=False)
        for f in fails:
            print(f)
        print("[124] VERDICT: %s (%d problem(s))" % ("PASS" if not fails else "FAIL", len(fails)))
        return 0 if not fails else 1

    man = build(args.jf_inputs, cl3, out_dir, emit=True, only=args.only)
    mp = ROOT / MANIFEST
    # `--only` selects what to REDRAW, not what the manifest describes: an entry
    # that was not rebuilt must survive, or a partial run silently drops the
    # figures it did not touch and `--check` reports them as "absent".
    if mp.exists():
        try:
            prev = json.loads(mp.read_text(encoding="utf-8")).get("figures", {})
            for k, v in prev.items():
                man["figures"].setdefault(k, v)
        except (json.JSONDecodeError, AttributeError):
            pass
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, f in man["figures"].items():
        print("[124] %s %-42s %s" % (key, f["title"], f["convention"][:70]))
        for ext, rel in (f.get("files") or {}).items():
            print("        %-4s %s" % (ext, rel))
    print("[124] manifest -> %s" % MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
