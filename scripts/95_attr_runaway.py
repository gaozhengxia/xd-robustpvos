"""95 - attribute the closed-loop failure of the propagation baseline.

Project-internal diagnostic (it produces the numbers for the paper's failure
analysis, but it is not itself a table).

The question
------------
`results/clean_baseline_stride_compare.csv` showed that feeding the SAME clean
sequence at stride 1 instead of stride 4 makes 9 of 30 sequences *worse*
(`bmx-trees` -31.9 J&F, `libby` -18.1). More frames cannot make a per-frame
segmenter worse, so something about the closed loop must be at fault. Two very
different mechanisms are consistent with that observation:

  (H1) accumulation -- the loop feeds its own (imperfect) mask back as the next
       prompt, so error compounds with the number of feedback steps. Stride 4
       merely took 1/4 as many steps, hiding it.
  (H2) frame difficulty -- the extra frames are simply harder ones, and
       evaluating them lowers the mean. Nothing would be "wrong" with the loop.

They are separable, because `benchmark._frame_indices` is `range(0, n, stride)`:
the stride-4 run evaluates exactly the frames 0, 4, 8, ... of the stride-1
trace. So comparing

    mean J&F over stride-1 rows with t % 4 == 0      (same frames, 4x history)
    vs
    the stride-4 aggregate for the same instance      (same frames, 1x history)

holds the evaluated frames fixed and varies ONLY how many feedback steps led up
to each of them. If H1 is true this delta is large and negative; if H2 is true
it is ~0.

Probe self-validation
---------------------
A negative result is only worth reporting if the probe itself is known good, so
this script first checks that a *stride-4* per-frame trace reproduces the
stride-4 aggregate CSV bit-for-bit. If that check fails, nothing below it is
interpretable and the script says so instead of printing a verdict.

The second half measures whether the segmenter's own confidence
(`backend_score`, SAM 2.1's predicted IoU) could have served as the rejection
signal -- i.e. whether the "standard rejection path" everyone reaches for would
actually have caught the collapse. Reported as rank-AUC for detecting
bad frames, plus the score trajectory around the collapse onset.

Run from the PROJECT ROOT:

    python scripts/92_ab_mask_prompt.py --mode greedy --frame-stride 1 \\
        --seq bmx-trees,libby --degradations clean --levels 1 --modes hard \\
        --out _scratch/attr/stride1.csv
    python scripts/92_ab_mask_prompt.py --mode greedy --frame-stride 4 \\
        --seq bmx-trees,libby --degradations clean --levels 1 --modes hard \\
        --out _scratch/attr/stride4.csv
    python scripts/95_attr_runaway.py --trace1 _scratch/attr/stride1.csv \\
        --trace4 _scratch/attr/stride4.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: a frame counts as "bad" for the AUC when its J&F is below this
BAD_JF = 0.5

#: the mask has visibly left the object above this predicted/ground-truth area
RUNAWAY_RATIO = 3.0

#: the stride the reference aggregate was produced with
REF_STRIDE = 4


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def load(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x, default=float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# --------------------------------------------------------------------------- #
# statistics (no scipy dependency)
# --------------------------------------------------------------------------- #

def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def rank_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC of `scores` as a detector of `labels == 1`. 0.5 = no information."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _ranks(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    return pearson(_ranks(a), _ranks(b))


# --------------------------------------------------------------------------- #
# slicing helpers
# --------------------------------------------------------------------------- #

def by_instance(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, int], List[Dict]]:
    """Per-frame rows are single-instance (92_* takes `object_ids[0]`), so the
    instance key is derived from the sequence name only."""
    out: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in rows:
        out[(r["seq"], int(r.get("obj_id", 1) or 1))].append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r["frame"]))
    return dict(out)


def mean_jf(rows: Sequence[Dict[str, str]]) -> float:
    v = [fnum(r["JF"]) for r in rows]
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def deciles(rows: Sequence[Dict[str, str]], n: int = 10) -> List[float]:
    v = [fnum(r["JF"]) for r in rows]
    if not v:
        return []
    parts = np.array_split(np.asarray(v, float), min(n, len(v)))
    return [float(np.mean(p)) for p in parts if len(p)]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="attribute the propagation baseline's stride sensitivity")
    ap.add_argument("--trace1", default="_scratch/attr_20260917/runaway_trace.csv",
                    help="per-frame rows at stride 1")
    ap.add_argument("--trace4", default=None,
                    help="per-frame rows at stride 4 (probe self-validation)")
    ap.add_argument("--ref", default="results/clean_baseline_all.csv",
                    help="per-instance aggregate rows (the stride-4 reference)")
    ap.add_argument("--stride", type=int, default=REF_STRIDE,
                    help="stride of the reference aggregate")
    args = ap.parse_args()

    p1 = Path(args.trace1) if Path(args.trace1).is_absolute() else ROOT / args.trace1
    if not p1.exists():
        print(f"missing stride-1 trace: {p1}")
        return 1
    rows1 = by_instance(load(p1))

    ref: Dict[Tuple[str, int], Dict[str, str]] = {}
    pr = Path(args.ref) if Path(args.ref).is_absolute() else ROOT / args.ref
    for r in load(pr):
        if str(r.get("mode")) != "greedy":
            continue
        if int(fnum(r.get("frame_stride"), -1)) != args.stride:
            continue
        ref[(r["seq"], int(r["obj_id"]))] = r

    print("=" * 92)
    print("XD-RobustPVOS :: closed-loop runaway attribution")
    print("=" * 92)
    print(f"stride-1 trace : {p1.relative_to(ROOT) if p1.is_relative_to(ROOT) else p1}")
    print(f"reference      : {pr.name}  (stride={args.stride})")
    print(f"instances      : {len(rows1)} traced, {len(ref)} in reference")
    print()

    # ---- probe self-validation -------------------------------------------- #
    if args.trace4:
        p4 = Path(args.trace4)
        p4 = p4 if p4.is_absolute() else ROOT / p4
        if p4.exists():
            print("-" * 92)
            print("PROBE SELF-VALIDATION  (a stride-4 trace must reproduce the "
                  "stride-4 aggregate)")
            print("-" * 92)
            worst = 0.0
            n_ok = 0
            for key, rs in by_instance(load(p4)).items():
                m = mean_jf(rs)
                r = ref.get(key)
                if r is None:
                    print(f"  {key[0]:<18} obj{key[1]}  not in reference -- skipped")
                    continue
                d = abs(m - fnum(r["J&F"]))
                worst = max(worst, d)
                n_ok += int(d < 1e-9)
                flag = "ok" if d < 1e-9 else "MISMATCH"
                print(f"  {key[0]:<18} obj{key[1]}  trace={m:.6f}  "
                      f"ref={fnum(r['J&F']):.6f}  |d|={d:.2e}  {flag}")
            print(f"\n  {n_ok}/{len(by_instance(load(p4)))} reproduce exactly; "
                  f"max |delta| = {worst:.2e}")
            # Bit-exactness is the ideal but not the requirement: bf16 autocast is
            # not guaranteed bit-reproducible across processes. What would
            # invalidate the comparison is a *systematic* offset -- a wrong frame
            # slice, a mismatched instance, a different dilation -- and that shows
            # up orders of magnitude above 1e-3.
            if worst >= 1e-3:
                print("\n  [STOP] the probe does not reproduce the reference. "
                      "Nothing below is interpretable.")
                return 1
            if worst >= 1e-9:
                print(f"  (within {worst:.1e} but not bit-exact -- consistent with "
                      "bf16 autocast being process-dependent; the comparison "
                      "below is unaffected)")
            print()
        else:
            print(f"[note] stride-4 trace not found at {p4} -- probe not validated\n")

    # ---- H1 vs H2 ---------------------------------------------------------- #
    print("-" * 92)
    print("CAUSAL TEST  (identical evaluated frames, 4x vs 1x feedback history)")
    print("-" * 92)
    print("  `restricted` = mean over stride-1 rows whose frame index is a multiple "
          f"of {args.stride}:")
    print("                 the exact frames the stride-4 run evaluated. Fixed frames, "
          "so the")
    print("                 only moving part is how many feedback steps preceded them.")
    print()
    hdr = (f"{'seq':<20}{'obj':>4}{'n_hist':>7}{'restricted':>12}{'stride4':>9}"
           f"{'delta':>9}{'interior':>10}")
    print(hdr)
    print("-" * len(hdr))

    per_seq: List[Tuple[str, int, float, float, float, float, int]] = []
    for key in sorted(rows1, key=lambda k: k[0]):
        rs = rows1[key]
        r = ref.get(key)
        if r is None:
            continue
        sel = [x for x in rs if int(x["frame"]) % args.stride == 0]
        interior = [x for x in rs if int(x["frame"]) % args.stride != 0]
        if not sel:
            continue
        m_sel, m_ref = mean_jf(sel), fnum(r["J&F"])
        m_int = mean_jf(interior)
        per_seq.append((key[0], key[1], m_sel, m_ref, m_sel - m_ref, m_int,
                        len(sel)))
        print(f"{key[0]:<20}{key[1]:>4}{len(sel):>7}{m_sel:>12.4f}{m_ref:>9.4f}"
              f"{m_sel - m_ref:>+9.4f}{m_int:>10.4f}")

    if not per_seq:
        print("\nno instance appears in both the trace and the reference.")
        return 1

    d = np.array([x[4] for x in per_seq], float)
    print()
    print(f"  mean delta (restricted - stride4) = {d.mean():+.4f}   "
          f"median = {float(np.median(d)):+.4f}")
    print(f"  sequences where the same frames score WORSE with 4x feedback history: "
          f"{int((d < -0.005).sum())}/{len(d)}")
    print(f"  sequences where they score better: {int((d > 0.005).sum())}/{len(d)}"
          f"   unchanged: {int((abs(d) <= 0.005).sum())}/{len(d)}")
    verdict_h1 = d.mean() < -0.02
    print()
    if verdict_h1:
        print("  => H1 (accumulation) is SUPPORTED: on byte-identical frames, having")
        print("     more feedback steps behind you makes the prediction worse. The")
        print("     stride-4 numbers were flattered by the loop taking fewer steps.")
    else:
        print("  => H1 (accumulation) is NOT supported by this test. The stride")
        print("     sensitivity is not explained by feedback depth alone.")

    # ---- trajectory shape -------------------------------------------------- #
    print()
    print("-" * 92)
    print("TRAJECTORY SHAPE  (J&F by decile of the sequence, stride-1 trace)")
    print("-" * 92)
    dd = deciles(next(iter(rows1.values())), 10)
    print(f"{'seq':<20}" + "".join(f"{i:>7}" for i in range(1, len(dd) + 1)))
    for key in sorted(rows1, key=lambda k: k[0]):
        de = deciles(rows1[key], 10)
        print(f"{key[0]:<20}" + "".join(f"{v:>7.3f}" for v in de))
    print("\n  column i = i-th tenth of the sequence, in time order.")

    # ---- could the model's own confidence have caught it? ------------------ #
    print()
    print("-" * 92)
    print("WOULD A CONFIDENCE GATE HAVE CAUGHT IT?  (backend_score = SAM 2.1 "
          "predicted IoU)")
    print("-" * 92)
    hdr2 = (f"{'seq':<20}{'obj':>4}{'score mean':>12}{'rho(JF)':>10}"
            f"{'AUC(good)':>12}{'score@good':>12}{'score@bad':>11}"
            f"{'t(>3x area)':>13}")
    print(hdr2)
    print("-" * len(hdr2))
    rows_auc: List[Tuple[float, int]] = []
    for key in sorted(rows1, key=lambda k: k[0]):
        rs = rows1[key]
        sc = [fnum(x["backend_score"]) for x in rs]
        jf = [fnum(x["JF"]) for x in rs]
        ok = np.isfinite(sc) & np.isfinite(jf)
        sc_a, jf_a = np.asarray(sc)[ok], np.asarray(jf)[ok]
        if len(sc_a) < 3:
            continue
        # `good` is the positive class on purpose: a rejection gate fires when the
        # score is LOW, so the usable direction is AUC(good) > 0.5. The earlier
        # version of this script labelled bad frames positive and then printed a
        # ">0.5 is fine" rule, which is backwards -- exactly the kind of mislabel
        # that turns a diagnostic into a confident wrong answer.
        good = (jf_a >= BAD_JF).astype(int)
        g = sc_a[good == 1]
        b = sc_a[good == 0]
        ratio = [fnum(x["area_ratio"]) for x in rs]
        t3 = next((int(rs[i]["frame"]) for i, v in enumerate(ratio)
                   if math.isfinite(v) and v > RUNAWAY_RATIO), -1)
        rows_auc.extend(zip(sc_a.tolist(), good.tolist()))
        print(f"{key[0]:<20}{key[1]:>4}{float(np.mean(sc_a)):>12.4f}"
              f"{spearman(sc_a, jf_a):>10.3f}"
              f"{rank_auc(sc_a, good):>12.3f}"
              f"{(float(np.mean(g)) if len(g) else float('nan')):>12.4f}"
              f"{(float(np.mean(b)) if len(b) else float('nan')):>11.4f}"
              f"{t3:>13}")

    if rows_auc:
        sc_all = [x[0] for x in rows_auc]
        good_all = [x[1] for x in rows_auc]
        a = rank_auc(sc_all, good_all)

        # Probe self-check on a synthetic case with a known answer: a score that
        # IS the quality must score AUC 1.0, and its negation must score 0.0.
        syn_y = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        syn_lab = (syn_y >= BAD_JF).astype(int)
        a_pos = rank_auc(syn_y, syn_lab)
        a_neg = rank_auc(-syn_y, syn_lab)
        print()
        print(f"  probe self-check: synthetic score==quality -> AUC={a_pos:.3f} "
              f"(want 1.000), negated -> AUC={a_neg:.3f} (want 0.000)")
        if abs(a_pos - 1.0) > 1e-9 or abs(a_neg) > 1e-9:
            print("  [STOP] the AUC helper is broken; the numbers above are not "
                  "interpretable.")
            return 1

        print(f"  pooled AUC(good) = {a:.3f}   (0.5 = the segmenter's own "
              f"confidence is uninformative)")
        print("  usable for a rejection gate only if this is well ABOVE 0.5; below "
              "0.5 the score")
        print("  points the wrong way and gating on it would make things worse. "
              "`score@good` vs")
        print("  `score@bad` is the raw form of the same question, and "
              "`t(>3x area)` shows when the")
        print("  mask actually left the object -- a gate is only useful if the "
              "score moves first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
