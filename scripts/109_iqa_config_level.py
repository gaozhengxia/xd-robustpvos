"""CL3 PRIMARY table at ZERO GPU cost: config-level IQA vs already-landed J&F.

Why this exists
---------------
The paper's CL3 claim is "image quality does not predict downstream J&F under
degradation".  Producing it at the frame level needs a GPU sweep (~3 h); this
computes the same comparison at the **(sequence, degradation) cell** level from
data already in ``results/``, on CPU, in ~50 min -- and since 2026-09-21 that is
the paper's PRIMARY caliber (see "Caliber decision" below).

Degradation in this repository is a *deterministic pure function* of
``(clean frame, degradation, severity, seed=seq_seed, stream_id=original frame
index)`` with ``seq_seed = protocol.seed + _stable_hash(seq) % 100000`` -- see
``xdrp/degradations.apply_degradation``.  Two independent checks confirmed the
replay is bit-exact: the IQA multiset of ``baselines_raw.csv`` reproduces to
``0.000e+00`` (stride 4), and 450/450 cells of the stride-1 runs reproduce their
``severity_mean``/``severity_max`` fingerprint to < 1e-5.

So the IQA of every ALREADY-RUN cell can be recomputed on CPU by calling
``load_degraded_sequence`` -- *the same function the runner called*, not a
re-implementation -- and paired with the J&F already in ``results/``.

What this is NOT
----------------
The observational unit here is the (sequence, degradation) cell, not the frame.
The frame-level table has ~3e4 observations; this has at most 30 x 13 = 390.

**Caliber decision (2026-09-21, user):** this file produces the PAPER'S PRIMARY
CL3 table.  The cell is the right unit because the claim is "can image quality
pick an operator *across conditions*" -- one point per (sequence, config), which
is also what `[[FIG-2]]` plots.  The pooled `across` rho is the headline number;
`within` is the confound guard.

The frame-level run (`scripts/05_analysis.py --study iqa`) is retained as a
*robustness probe*, NOT a second independent confirmation: it pools frames, so
it answers the same question at a finer granularity and its CI is optimistic
(frames of one sequence are not independent).  The two rhos are DIFFERENT
statistics and must never be presented as "the same number computed twice"
(e.g. greedy `iqa_ssim`: +0.018 frame-level vs +0.270 cell-level here).

Two views are reported, because they answer different questions:

* ``across``  -- Spearman over all degraded cells pooled (n = 30 x 12).  The
  headline view.
* ``within``  -- Spearman computed *inside* each degradation (n = 30 sequences),
  then averaged over the 12 degradations.  Pooling across degradations lets a
  "harder corruption -> lower IQA and lower J&F" trend manufacture a positive
  correlation that is not a selection signal; the within- degradation view
  removes that confound.  If both views are small, the claim is robust.

Bootstrap intervals
-------------------
Both views carry a **sequence-clustered** interval.  The same 30 sequences appear
in all 12 degradations, so neither the 360 cells nor the 12 per-degradation rhos
are independent draws, and resampling cells (or degradations) understates the
interval.  ``across`` therefore reports a plain cell-level bootstrap *and* a
cluster version (resample sequences, taking all of a drawn sequence's cells,
which preserves the within-sequence pairing); ``within`` clusters the same way
before averaging.  Only the intervals move -- point estimates are untouched --
and the verdict is still decided on point estimates, with any interval that
reaches the threshold named explicitly rather than rounded to "not usable".

Fidelity guard
--------------
Every cell's replayed ``severity_mean``/``severity_max`` is compared against the
value the runner wrote into the CSV.  A mismatch means the replay drifted and
the script FAILS rather than emit numbers -- a negative CL3 result produced by a
broken replay is worthless.

Usage
-----
    python scripts/109_iqa_config_level.py \
        --inputs results/cl4_full_v2.csv results/singles_greedy.csv \
                 results/singles_dagrs.csv results/singles_sam2video.csv \
                 results/c1c4_a_dagrs.csv results/c1c4_b_dagrs.csv \
                 results/c1c4_a_base.csv results/c1c4_b_base.csv \
        --out-dir results
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from xdrp.benchmark import Protocol, load_degraded_sequence      # noqa: E402
from xdrp.datasets import open_dataset                           # noqa: E402
from xdrp.iqa import ALL_IQA_METRICS, compute_iqa                # noqa: E402
from xdrp.stats import spearman_with_ci                          # noqa: E402
try:  # cheap pure-numpy core, used for the thousands of bootstrap resamples
    from xdrp.stats import _spearman as _rho_fast                # noqa: E402
except ImportError:                                              # pragma: no cover
    def _rho_fast(a, b):                                         # type: ignore
        return spearman_with_ci(list(a), list(b), n_boot=0)["rho"]

#: rho below this is "too weak to select an operator" (see 05_analysis.py).
IQA_USABLE_RHO = 0.6
#: severity fingerprint tolerance between replay and the landed CSV
SEV_TOL = 1e-5
#: sequences whose sam2video rows are known to have a short frame denominator
#: (defect 13); reported as a sensitivity, never silently dropped.
DEFECT13_SEQS = ("india", "kite-surf")

DEFAULT_INPUTS = [
    "results/cl4_full_v2.csv",
    "results/singles_greedy.csv",
    "results/singles_dagrs.csv",
    "results/singles_sam2video.csv",
    "results/c1c4_a_dagrs.csv",
    "results/c1c4_b_dagrs.csv",
    "results/c1c4_a_base.csv",
    "results/c1c4_b_base.csv",
]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a05 = _load("a05", "scripts/05_analysis.py")


# --------------------------------------------------------------------------- #
# worker (one cell = one (sequence, degradation, level, stride))
# --------------------------------------------------------------------------- #

_SRC = None
_DATA_ROOT = ""
_SPLIT = "val"


def _init_worker(data_root: str, split: str) -> None:
    """Open the dataset once per process (the frame source is not picklable)."""
    global _SRC, _DATA_ROOT, _SPLIT
    _DATA_ROOT, _SPLIT = data_root, split
    _SRC = open_dataset("davis", data_root, split=split)


def _cell_worker(cell: Tuple[str, str, float, int]) -> Dict[str, Any]:
    seq, deg, level, stride = cell
    proto = Protocol(degradation=deg, level=float(level),
                     frame_stride=int(stride), max_frames=0,
                     temporal_variation=True, severity_amp=0.30, seed=0)
    # The production loader, not a re-implementation: identical severity
    # schedule, identical RNG stream ids, identical degradation calls.
    dseq = load_degraded_sequence(_SRC, seq, proto)
    T = len(dseq.degraded)
    per_metric: Dict[str, List[float]] = defaultdict(list)
    for t in range(T):
        iq = compute_iqa(dseq.degraded[t], clean=dseq.clean[t])
        for k, v in iq.items():
            if np.isfinite(v):
                per_metric[k].append(float(v))
    row: Dict[str, Any] = {
        "seq": seq, "degradation": deg, "level": float(level),
        "frame_stride": int(stride), "n_frames": T,
        "severity_mean": float(np.mean(dseq.severity)),
        "severity_max": float(np.max(dseq.severity)),
    }
    for k in ALL_IQA_METRICS:
        v = np.asarray(per_metric.get(k, []), np.float64)
        row[f"{k}_mean"] = float(v.mean()) if v.size else float("nan")
        row[f"{k}_std"] = float(v.std()) if v.size else float("nan")
        row[f"{k}_p10"] = float(np.percentile(v, 10)) if v.size else float("nan")
        row[f"{k}_p90"] = float(np.percentile(v, 90)) if v.size else float("nan")
    return row


# --------------------------------------------------------------------------- #
# cell discovery
# --------------------------------------------------------------------------- #

def discover_cells(inputs: Sequence[Path]):
    """Cells to replay + the landed per-arm J&F + the fingerprint to check."""
    raw: List[Dict[str, Any]] = []
    for p in inputs:
        with p.open(encoding="utf-8") as fh:
            raw.extend(csv.DictReader(fh))
    if not raw:
        raise SystemExit("no rows in any input")

    cells: Dict[Tuple[str, str, float, int], Dict[str, Any]] = {}
    for r in raw:
        seq = r["seq"]
        deg = r["degradation"]
        lvl = float(r["level"])
        stride = int(r["frame_stride"])
        k = (seq, deg, lvl, stride)
        c = cells.setdefault(k, {"seq": seq, "degradation": deg, "level": lvl,
                                 "frame_stride": stride,
                                 "sev_mean": float(r["severity_mean"]),
                                 "sev_max": float(r["severity_max"]),
                                 "n_frames_min": 10 ** 9, "n_frames_max": 0,
                                 "arms": set()})
        c["n_frames_min"] = min(c["n_frames_min"], int(r["n_frames"]))
        c["n_frames_max"] = max(c["n_frames_max"], int(r["n_frames"]))
        c["arms"].add(r["mode"])

    landed = a05.collapse_objects(a05.sequence_level(raw))
    jf: Dict[Tuple[str, str, float, str], float] = {}
    for r in landed:
        jf[(r["seq"], r["degradation"], float(r["level"]), r["mode"])] = float(r["J&F"])
    return cells, jf


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def _corr(x: Sequence[float], y: Sequence[float], n_boot: int) -> Dict[str, float]:
    st = spearman_with_ci(list(x), list(y), n_boot=n_boot, seed=0)
    rho = float(st["rho"])
    return {"rho": rho, "ci_lo": float(st["lo"]), "ci_hi": float(st["hi"]),
            "p": float(st["p"]), "n": int(st["n"]), "r2": rho * rho,
            "usable": int(abs(rho) >= IQA_USABLE_RHO)}


def _within_mean_ci(by_deg: Dict[str, List[Tuple[str, float, float]]],
                    n_boot: int, seed: int = 0) -> Tuple[float, float]:
    """Cluster-bootstrap CI for the MEAN of the per-degradation rho.

    Resampling the wrong unit here reproduces the pseudoreplication trap that
    `study_stats` fell into: the same 30 DAVIS sequences appear in all 13
    degradations, so the 13 per-degradation rho values are PAIRED, not
    independent.  Treating the 390 cells -- or the 13 degradations -- as
    independent draws understates the interval by orders of magnitude.  So
    resample SEQUENCES with replacement (the independent sampling unit), hold
    the degradation stratification fixed, and recompute one rho per degradation
    on every draw; the interval is the percentile spread of the resulting means.
    """
    degs = [d for d in sorted(by_deg) if len(by_deg[d]) >= 8]
    if not degs or n_boot <= 0:
        return float("nan"), float("nan")
    seqs = sorted({s for d in degs for s, _x, _y in by_deg[d]})
    if len(seqs) < 8:
        return float("nan"), float("nan")
    lookup = {d: {s: (x, y) for s, x, y in by_deg[d]} for d in degs}
    rng = np.random.default_rng(int(seed))
    means: List[float] = []
    for _ in range(int(n_boot)):
        chosen = [seqs[i] for i in rng.integers(0, len(seqs), size=len(seqs))]
        rhos: List[float] = []
        for d in degs:
            m = lookup[d]
            pairs = [m[s] for s in chosen if s in m]
            if len(pairs) < 8:
                continue
            rr = _rho_fast(np.asarray([p[0] for p in pairs], np.float64),
                           np.asarray([p[1] for p in pairs], np.float64))
            if not np.isnan(rr):
                rhos.append(float(rr))
        if rhos:
            means.append(float(np.mean(rhos)))
    if not means:
        return float("nan"), float("nan")
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _across_cluster_ci(by_seq: Dict[str, List[Tuple[float, float]]],
                       n_boot: int, seed: int = 0) -> Tuple[float, float]:
    """Cluster-bootstrap CI for the POOLED (``across``) rho.

    `across` pools one observation per (sequence, degradation) cell, but the same
    30 sequences appear in all 12 degradations -- so the 360 cells are not
    independent draws and a plain cell-level bootstrap understates the interval.
    The independent sampling unit is the SEQUENCE, so resample sequences with
    replacement, take *all* of a drawn sequence's cells (which keeps the
    within-sequence pairing intact, the same reason a paired test can be tighter
    than an unpaired one), and recompute rho on the pooled draw.

    Point estimates are unaffected: this only widens/shifts the interval.
    """
    keys = [s for s in sorted(by_seq) if by_seq[s]]
    if len(keys) < 8 or n_boot <= 0:
        return float("nan"), float("nan")
    pools = [np.asarray(by_seq[s], np.float64).reshape(-1, 2) for s in keys]
    k = len(keys)
    rng = np.random.default_rng(int(seed))
    out: List[float] = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, k, size=k)
        stacked = np.concatenate([pools[j] for j in pick], axis=0)
        rr = _rho_fast(stacked[:, 0], stacked[:, 1])
        if not np.isnan(rr):
            out.append(float(rr))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def evaluate(cells_rows: List[Dict[str, Any]], jf, arms, n_boot: int,
             include_clean: bool) -> Dict[str, Any]:
    """Per-arm, per-metric correlations in the two views."""
    out: Dict[str, Any] = {}
    for arm in arms:
        rows = []
        for c in cells_rows:
            if not include_clean and c["degradation"] == "clean":
                continue
            y = jf.get((c["seq"], c["degradation"], float(c["level"]), arm))
            if y is None:
                continue
            rr = dict(c)
            rr["JF"] = y
            rows.append(rr)
        if not rows:
            continue
        per_metric: Dict[str, Any] = {}
        for mk in ALL_IQA_METRICS:
            col = f"{mk}_mean"
            xs = [r[col] for r in rows]
            ys = [r["JF"] for r in rows]
            across = _corr(xs, ys, n_boot)
            # The pooled cells are not independent either (the same 30 sequences
            # appear in every degradation), so the headline number carries a
            # sequence-clustered interval too -- computed HERE, in the production
            # path, rather than in a scratch probe.
            by_seq_pool: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
            for r in rows:
                by_seq_pool[r["seq"]].append((r[col], r["JF"]))
            ac_lo, ac_hi = _across_cluster_ci(by_seq_pool, n_boot)
            # within-degradation: one rho per degradation, then average.  The
            # sequence identity is kept so the CI below can resample the paired
            # unit instead of pretending the cells are independent.
            by_deg: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
            for r in rows:
                by_deg[r["degradation"]].append((r["seq"], r[col], r["JF"]))
            inner = []
            for d, triples in sorted(by_deg.items()):
                if len(triples) < 8:
                    continue
                inner.append(_corr([t[1] for t in triples], [t[2] for t in triples],
                                   0)["rho"])
            w_lo, w_hi = _within_mean_ci(by_deg, n_boot)
            per_metric[mk] = {
                "kind": ("full-reference" if mk in ("iqa_psnr", "iqa_ssim")
                         else "no-reference"),
                "across": across,
                "across_ci_lo_cluster": ac_lo,
                "across_ci_hi_cluster": ac_hi,
                "within_mean_rho": float(np.mean(inner)) if inner else float("nan"),
                "within_min_rho": float(np.min(inner)) if inner else float("nan"),
                "within_max_rho": float(np.max(inner)) if inner else float("nan"),
                "within_ci_lo": w_lo,
                "within_ci_hi": w_hi,
                "within_n_degradations": len(inner),
            }
        out[arm] = {"n_cells": len(rows),
                    "n_sequences": len({r["seq"] for r in rows}),
                    "n_degradations": len({r["degradation"] for r in rows}),
                    "metrics": per_metric}
    return out


def verdict_block(res: Dict[str, Any]) -> Dict[str, Any]:
    """The claim is supported when every metric, in BOTH views, is weak."""
    worst = 0.0
    worst_where = ""
    for arm, a in res.items():
        for mk, m in a["metrics"].items():
            for view, v in (("across", abs(m["across"]["rho"])),
                            ("within", abs(m["within_mean_rho"])
                             if np.isfinite(m["within_mean_rho"]) else 0.0)):
                if v > worst:
                    worst, worst_where = v, f"{arm}/{mk}/{view}"
    return {"max_abs_rho_any_view": worst, "max_abs_rho_where": worst_where,
            "usable_threshold": IQA_USABLE_RHO,
            "any_metric_usable": int(worst >= IQA_USABLE_RHO),
            "supported": int(worst < IQA_USABLE_RHO),
            # The verdict is decided on point estimates, consistently with
            # `across`. But a within-mean whose bootstrap interval reaches the
            # threshold cannot be described as "definitely too weak" either, so
            # name those cases instead of letting them round to "not usable".
            "within_ci_reaches_threshold": [f"{arm}/{mk}"
                                            for arm, a in res.items()
                                            for mk, m in a["metrics"].items()
                                            if np.isfinite(m["within_ci_hi"])
                                            and abs(m["within_ci_hi"]) >= IQA_USABLE_RHO],
            # Same logic for the pooled interval.  With `across` as the headline
            # view this is the interval a reviewer will re-derive, so any case
            # that reaches the threshold is named rather than left implied by a
            # point estimate -- and the worst upper bound is reported so the
            # "0.6 is excluded" claim is a number, not an impression.
            "across_cluster_ci_reaches_threshold": [
                f"{arm}/{mk}" for arm, a in res.items()
                for mk, m in a["metrics"].items()
                if np.isfinite(m.get("across_ci_hi_cluster", float("nan")))
                and abs(m["across_ci_hi_cluster"]) >= IQA_USABLE_RHO],
            "max_abs_across_cluster_ci_hi": max(
                [abs(m["across_ci_hi_cluster"]) for a in res.values()
                 for m in a["metrics"].values()
                 if np.isfinite(m.get("across_ci_hi_cluster", float("nan")))],
                default=0.0)}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

LAB_ARM = {"greedy": "`greedy`", "dagrs": "`dagrs`", "sam2video": "`sam2video`"}


def write_md(path: Path, res, verdict, cells_rows, meta) -> None:
    L: List[str] = []
    L.append("# CL3 side evidence: config-level IQA vs J&F (zero-GPU replay)")
    L.append("")
    L.append(f"- observational unit: **(sequence x degradation) cell**, "
             f"n = {meta['n_cells_degraded']} degraded cells "
             f"({meta['n_sequences']} sequences x {meta['n_degradations']} "
             f"degradations)")
    L.append(f"- IQA recomputed by replaying `load_degraded_sequence` "
             f"(CPU only); fidelity guard vs the landed severity fingerprint: "
             f"worst |d| = **{meta['fidelity_worst_abs_severity_diff']:.3e}** over "
             f"{meta['n_cells_all']} cells")
    L.append(f"- usable threshold |rho| >= {IQA_USABLE_RHO}")
    L.append("- `across` CI: a plain cell-level bootstrap **and** a cluster bootstrap over "
             "sequences (resample sequences, taking all of a drawn sequence's cells).  The "
             "360 cells are not independent draws either, so the clustered interval is the "
             "one to quote; the point estimate is identical under both")
    L.append("- within 95% CI: **cluster bootstrap over sequences** -- the sequences are the "
             "paired unit (the same ones appear in every degradation), so resampling cells or "
             "degradations would understate the interval; one rho per degradation is recomputed "
             "on every draw and then averaged")
    L.append("- the verdict is decided on **point estimates**; `CI 触线 = yes` means the "
             "interval reaches the threshold, so that metric must NOT be described as "
             "*definitely* too weak")
    L.append("")
    for arm, a in sorted(res.items()):
        L.append(f"## arm {LAB_ARM.get(arm, arm)}  "
                 f"({a['n_cells']} cells, {a['n_sequences']} seqs)")
        L.append("")
        L.append("| 指标 | 类型 | across rho | across CI 朴素格级 "
                 "| across CI 按序列聚类 | R² | within 均值 | within CI 聚类 "
                 "| within 逐档 min…max | usable | CI 触线 |")
        L.append("|---" * 11 + "|")
        for mk in ALL_IQA_METRICS:
            m = a["metrics"][mk]
            ac = m["across"]
            wci = (f"{m['within_ci_lo']:+.3f} … {m['within_ci_hi']:+.3f}"
                   if np.isfinite(m["within_ci_lo"]) else "n/a")
            acci = (f"{m['across_ci_lo_cluster']:+.3f} … "
                    f"{m['across_ci_hi_cluster']:+.3f}"
                    if np.isfinite(m.get("across_ci_hi_cluster", float("nan")))
                    else "n/a")
            crosses = ((np.isfinite(m["within_ci_hi"])
                        and abs(m["within_ci_hi"]) >= IQA_USABLE_RHO)
                       or (np.isfinite(m.get("across_ci_hi_cluster", float("nan")))
                           and abs(m["across_ci_hi_cluster"]) >= IQA_USABLE_RHO))
            L.append(f"| `{mk}` | {m['kind']} | {ac['rho']:+.3f} "
                     f"| [{ac['ci_lo']:+.3f},{ac['ci_hi']:+.3f}] | {acci} "
                     f"| {ac['r2']:.3f} | "
                     f"{m['within_mean_rho']:+.3f} | {wci} | "
                     f"{m['within_min_rho']:+.3f} … {m['within_max_rho']:+.3f} | "
                     f"{'**YES**' if ac['usable'] else 'no'} | "
                     f"{'**yes**' if crosses else 'no'} |")
        L.append("")
    L.append("## verdict")
    L.append("")
    L.append(f"- max |rho| over every arm x metric x view = "
             f"**{verdict['max_abs_rho_any_view']:.3f}** "
             f"(`{verdict['max_abs_rho_where']}`)")
    L.append(f"- every metric below the usable threshold: "
             f"**{'YES' if verdict['supported'] else 'NO'}**")
    reaches = verdict.get("within_ci_reaches_threshold") or []
    L.append(f"- within-CI reaching the threshold: "
             f"**{len(reaches)}**"
             + (f" ({', '.join('`' + x + '`' for x in reaches)})" if reaches else "")
             + ("  ⇒ those metrics may only be described as *point estimate below "
                "threshold*, never as *definitely too weak*" if reaches else ""))
    L.append(f"- max |rho| upper bound of the `across` **cluster** CI over every "
             f"arm x metric: **{verdict['max_abs_across_cluster_ci_hi']:.3f}** "
             f"(threshold {IQA_USABLE_RHO})")
    areach = verdict.get("across_cluster_ci_reaches_threshold") or []
    L.append(f"- across cluster-CI reaching the threshold: **{len(areach)}**"
             + (f" ({', '.join('`' + x + '`' for x in areach)})" if areach else
                "  ⇒ the headline view excludes the threshold even under sequence "
                "clustering"))
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="iqa_config_level")
    ap.add_argument("--data-root", default="data/DAVIS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--include-clean", action="store_true",
                    help="include the clean reference cells in the correlation "
                         "(off by default: a clean cell has no restoration "
                         "decision to make, see 05_analysis.study_iqa)")
    ap.add_argument("--limit-cells", type=int, default=0,
                    help="debug: replay only the first N cells")
    args = ap.parse_args()

    inputs = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.inputs]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, jf = discover_cells(inputs)
    arms = sorted({k[3] for k in jf})
    print(f"=== CL3 config-level replay ===")
    print(f"inputs      : {len(inputs)} files")
    print(f"cells       : {len(cells)} (seq x degradation x level x stride)")
    print(f"arms        : {arms}")
    degs = sorted({c['degradation'] for c in cells.values()})
    print(f"degradations: {len(degs)} {degs}")

    todo = sorted(cells.keys())
    if args.limit_cells:
        todo = todo[:args.limit_cells]
    print(f"replaying   : {len(todo)} cells with {args.workers} workers ...")

    rows: List[Dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(str(ROOT / args.data_root), args.split)) as ex:
            for i, r in enumerate(ex.map(_cell_worker, todo, chunksize=1), 1):
                rows.append(r)
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} cells")
    else:
        _init_worker(str(ROOT / args.data_root), args.split)
        for i, c in enumerate(todo, 1):
            rows.append(_cell_worker(c))
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} cells")

    # ---- fidelity guard: replay must reproduce the landed fingerprint --------
    worst = 0.0
    worst_where = ""
    for r in rows:
        c = cells[(r["seq"], r["degradation"], r["level"], r["frame_stride"])]
        d = max(abs(r["severity_mean"] - c["sev_mean"]),
                abs(r["severity_max"] - c["sev_max"]))
        if d > worst:
            worst, worst_where = d, f"{r['seq']}/{r['degradation']}"
    print(f"fidelity guard: worst |d severity| = {worst:.3e} "
          f"(at {worst_where}), tol {SEV_TOL:.0e}")
    if worst > SEV_TOL:
        raise SystemExit("ABORT: replay does not reproduce the landed severity "
                         "fingerprint -> the IQA below is not the IQA the runner "
                         "measured. Refusing to emit numbers.")

    # expose defect 13 rather than hide it
    short = [r for r in rows
             if cells[(r["seq"], r["degradation"], r["level"], r["frame_stride"])]
             ["n_frames_min"] < r["n_frames"]]
    if short:
        print(f"[WARN] {len(short)} cells have a sam2video row with n_frames < "
              f"the protocol T (defect 13): "
              f"{sorted({(r['seq'], r['degradation']) for r in short})}")
    for r in rows:
        c = cells[(r["seq"], r["degradation"], r["level"], r["frame_stride"])]
        r["n_frames_min_landed"] = c["n_frames_min"]
        r["arms_present"] = ",".join(sorted(c["arms"]))

    # ---- cell table ---------------------------------------------------------
    cols = (["seq", "degradation", "level", "frame_stride", "n_frames",
             "n_frames_min_landed", "severity_mean", "severity_max", "arms_present"]
            + [f"{mk}_{s}" for mk in ALL_IQA_METRICS for s in ("mean", "std", "p10", "p90")])
    cells_csv = out_dir / f"{args.prefix}_cells.csv"
    with cells_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"cells       -> {cells_csv}")

    # ---- correlations -------------------------------------------------------
    res = evaluate(rows, jf, arms, args.n_boot, args.include_clean)
    verdict = verdict_block(res)

    # sensitivity: drop the two sequences affected by defect 13
    keep = [r for r in rows if r["seq"] not in DEFECT13_SEQS]
    res_sens = evaluate(keep, jf, arms, 2000, args.include_clean)
    sens = {}
    for arm in res:
        for mk in res[arm]["metrics"]:
            a = res[arm]["metrics"][mk]["across"]["rho"]
            b = res_sens.get(arm, {}).get("metrics", {}).get(mk, {}).get(
                "across", {}).get("rho", float("nan"))
            sens[f"{arm}/{mk}"] = {"all": a, "excl_defect13": b,
                                   "shift": (b - a) if np.isfinite(b) else float("nan")}

    payload = {
        "note": ("CL3 PRIMARY evidence (caliber set 2026-09-21 by the PI): the "
                 "observational unit is the (sequence x degradation) cell, NOT the "
                 "frame, because the claim is about selecting an operator across "
                 "conditions. IQA recomputed on CPU by replaying "
                 "load_degraded_sequence; fidelity guard enforces bit-level "
                 "agreement with the landed severity fingerprint. The frame-level "
                 "run in 05_analysis is a robustness PROBE, not a second "
                 "independent confirmation -- it pools frames and therefore "
                 "reports a different statistic."),
        "bootstrap": {"n_boot": int(args.n_boot),
                      "across": "plain cell-level + sequence-clustered",
                      "within": "sequence-clustered (paired unit = sequence)"},
        "inputs": [str(p.relative_to(ROOT)) for p in inputs],
        "protocol": {"frame_stride": sorted({r["frame_stride"] for r in rows}),
                     "max_frames": 0, "severity_amp": 0.30, "seed": 0,
                     "temporal_variation": True},
        "n_cells_all": len(rows),
        "n_cells_degraded": sum(1 for r in rows if r["degradation"] != "clean"),
        "n_sequences": len({r["seq"] for r in rows}),
        "n_degradations": len({r["degradation"] for r in rows}),
        "include_clean": bool(args.include_clean),
        "fidelity_worst_abs_severity_diff": worst,
        "fidelity_tolerance": SEV_TOL,
        "usable_threshold": IQA_USABLE_RHO,
        "results": res,
        "verdict": verdict,
        "sensitivity_excl_defect13": sens,
    }
    jpath = out_dir / f"{args.prefix}.json"
    jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    mpath = out_dir / f"{args.prefix}.md"
    write_md(mpath, res, verdict, rows, payload)

    print()
    print("=== per-arm across-degradation rho (n = %d degraded cells) ==="
          % payload["n_cells_degraded"])
    for arm, a in sorted(res.items()):
        print(f"  {arm}:")
        for mk in ALL_IQA_METRICS:
            m = a["metrics"][mk]
            print(f"    {mk:20s} across {m['across']['rho']:+.3f} "
                  f"CI[{m['across']['ci_lo']:+.3f},{m['across']['ci_hi']:+.3f}] "
                  f"cluster[{m['across_ci_lo_cluster']:+.3f},"
                  f"{m['across_ci_hi_cluster']:+.3f}] "
                  f"(R2 {m['across']['r2']:.3f}, usable={m['across']['usable']})  "
                  f"within {m['within_mean_rho']:+.3f} "
                  f"CI[{m['within_ci_lo']:+.3f},{m['within_ci_hi']:+.3f}] "
                  f"perdeg[{m['within_min_rho']:+.3f},{m['within_max_rho']:+.3f}]")
    print()
    print(f"VERDICT: max |rho| over all arms/metrics/views = "
          f"{verdict['max_abs_rho_any_view']:.3f} "
          f"({verdict['max_abs_rho_where']}); "
          f"claim {'SUPPORTED' if verdict['supported'] else 'NOT SUPPORTED'}")
    print(f"         across cluster CI upper bound (max over arm x metric) = "
          f"{verdict['max_abs_across_cluster_ci_hi']:.3f} "
          f"(threshold {IQA_USABLE_RHO}); reaching it: "
          f"{len(verdict.get('across_cluster_ci_reaches_threshold') or [])}")
    print(f"json -> {jpath}")
    print(f"md   -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
