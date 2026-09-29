"""05 - Analysis: aggregation, error bars, significance tests, correlation study.

This script is where the paper's numbers come from. It writes machine-readable
CSV tables AND LaTeX fragments you can paste into the manuscript.

    python scripts/05_analysis.py --study iqa      # CL3: THE central claim
    python scripts/05_analysis.py --study stats    # TABLE-14
    python scripts/05_analysis.py --study vacuity  # TABLE-16 (renumbered 2026-09-23; TABLE-15 = cross-source)
    python scripts/05_analysis.py --study tables   # TABLE-9/10/11/13
    python scripts/05_analysis.py --study figs     # F2/F3/F6
    python scripts/05_analysis.py --study all
    python scripts/05_analysis.py --self-test      # known-truth check, no data

CRITICAL: run `--study iqa` as soon as the baseline CSV exists. It either
confirms or refutes the paper's central premise, and you must know before you
write a single paragraph of the manuscript.

`--study iqa` computes the correlation on ONE segmentation arm (`--iqa-mode`,
default `greedy`) and refuses to pool arms; `oracle_clean` is always excluded.
Both guards are regression-tested by `--self-test`, because either one silently
invalidates the claim if it is ever dropped. See ``study_iqa`` for the rationale.

Ablation arms reuse the mode string of the method they ablate (A0/A1/A2/A8..A10 all
report ``mode="dagrs"``). `arm` is therefore part of the aggregation key, and every
main table consumes `main_subset(...)`; only TABLE-13 reads the arm rows. Dropping
either guard silently turns TABLE-9's DAG-RS entry into a mean over the ablations,
including the deliberately weakened ones. Also regression-tested by `--self-test`.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xdrp.iqa import ALL_IQA_METRICS, FULL_REFERENCE_METRICS, NO_REFERENCE_METRICS
from xdrp.stats import (bootstrap_ci, cliffs_delta, holm_bonferroni, paired_report,
                        spearman_with_ci, wilcoxon_paired)


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def read_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            continue
        with open(pp, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def fnum(v, default=np.nan) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return str(p)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return str(p)


def write_latex(rows: Sequence[Dict[str, Any]], path: str, caption: str,
                label: str, float_fmt: str = "%.3f") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("% no data\n", encoding="utf-8")
        return str(p)
    cols = list(rows[0].keys())
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(float_fmt % v if np.isfinite(v) else "--")
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def sequence_level(rows: Sequence[Dict[str, Any]],
                   keys: Sequence[str] = ("seq", "obj_id", "mode", "degradation",
                                          "level", "arm")
                   ) -> List[Dict[str, Any]]:
    """Collapse per-frame records to one J&F per sequence cell.

    THIS IS THE PAIRED UNIT for every significance test in the paper.

    `arm` is PART OF THE GROUPING KEY, and this is load-bearing. Several ablation
    arms deliberately reuse the mode string of the method they ablate -- A0/A1/A2/
    A8/A9/A10 all report `mode="dagrs"`, exactly like the real DAG-RS rows in
    `dagrs_main_raw.csv`. Without `arm` in the key those rows collapse into one
    bucket and the headline DAG-RS number in TABLE-9 becomes a mean over ~18
    heterogeneous configurations, including the deliberately weakened ones
    (equal-weight fusion, no re-anchoring, ...). The failure is silent.

    NOTE: the old A11 ("softmax instead of Dirichlet") was removed on 2026-09-16 --
    its mechanism was never implemented and it silently reproduced A0. See
    docs/BASELINE_DIAGNOSIS.md section 5.

    Main tables must therefore consume `main_subset(...)`; only TABLE-13 uses the
    arm rows.
    """
    buckets: Dict[Tuple, List[float]] = defaultdict(list)
    meta: Dict[Tuple, Dict[str, Any]] = {}
    for r in rows:
        k = tuple(str(r.get(x, "")) for x in keys)
        v = fnum(r.get("JF_frame", r.get("J&F", "")))
        if np.isfinite(v):
            buckets[k].append(v)
            meta.setdefault(k, r)
    out = []
    for k, vals in buckets.items():
        if not vals:
            continue
        m = meta[k]
        out.append({
            "seq": k[0], "obj_id": k[1], "mode": k[2],
            "degradation": k[3], "level": k[4],
            "arm": (k[5] if len(k) > 5 else ""),
            "arm_note": str(m.get("arm_note", "")),
            "is_compound": int(fnum(m.get("is_compound"), 0)),
            "J&F": float(np.mean(vals)),
            "J&F_std": float(np.std(vals)),
            "n_frames": len(vals),
            "n_decisions": fnum(m.get("n_decisions")),
            "n_reanchors": fnum(m.get("n_reanchors")),
            "n_encoder_calls": fnum(m.get("n_encoder_calls")),
            "wall_time_s": fnum(m.get("wall_time_s")),
        })
    return out


def is_arm_row(r: Mapping[str, Any]) -> bool:
    """True for rows produced by `04_ablation.py` (they carry a non-empty `arm`)."""
    return bool(str(r.get("arm", "") or "").strip())


def main_subset(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only non-ablation rows.

    Every main table (TABLE-9/10/11, TABLE-14, TABLE-16) must go through this.
    Ablation arms are reported separately in TABLE-13.
    """
    return [r for r in rows if not is_arm_row(r)]


#: Key that identifies one evaluated cell at sequence granularity. `arm` is part
#: of it for the same reason it is part of `sequence_level`'s key.
SEQ_KEY = ("seq", "mode", "degradation", "level", "arm")


def collapse_objects(seq_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Average the instances of a sequence -> the official DAVIS per-sequence value.

    DAVIS defines a sequence's score as the mean J&F over its annotated objects;
    the benchmark score is then the mean over sequences. Since 2026-09-17 the
    protocol scores every instance (`protocol.max_objects: 0`), so `sequence_level`
    returns one row *per (sequence, object)* and this collapse must happen before
    any table or test:

    * without it, `study_tables` weights sequences by their instance count, and
    * `study_stats.collect` is worse than unweighted -- it keys a dict on
      (seq, degradation, level), so with several objects per sequence the LAST
      object silently overwrites the others and the paired test compares the
      wrong numbers with no error.

    Sequences are given equal weight, which is what "the paired unit is the
    sequence" in TABLE-14 means.
    """
    buckets: Dict[Tuple, List[float]] = defaultdict(list)
    meta: Dict[Tuple, Dict[str, Any]] = {}
    for r in seq_rows:
        k = tuple(str(r.get(x, "")) for x in SEQ_KEY)
        v = fnum(r.get("J&F", r.get("JF_frame", "")))
        if np.isfinite(v):
            buckets[k].append(v)
            meta.setdefault(k, r)
    out: List[Dict[str, Any]] = []
    for k, vals in buckets.items():
        if not vals:
            continue
        m = meta[k]
        row = {kk: vv for kk, vv in zip(SEQ_KEY, k)}
        row["J&F"] = float(np.mean(vals))
        row["J&F_std"] = float(np.std(vals))
        row["n_objects"] = len(vals)
        for f in ("is_compound", "n_decisions", "n_reanchors", "n_encoder_calls",
                  "wall_time_s", "arm_note"):
            row[f] = m.get(f, "")
        out.append(row)
    return out


def group_mean(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    b: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows:
        k = tuple(str(r.get(x, "")) for x in keys)
        b[k].append(float(r["J&F"]))
    out = []
    for k, v in sorted(b.items()):
        row = {kk: kk_v for kk, kk_v in zip(keys, k)}
        row["J&F_mean"] = float(np.mean(v))
        row["J&F_std"] = float(np.std(v))
        row["n"] = len(v)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# study: iqa  (CL3 -- the paper's central claim)
# --------------------------------------------------------------------------- #

#: Arms whose input is NOT the degraded observation. Their IQA is therefore not
#: the IQA of anything the segmenter had to cope with, and including them would
#: bias the CL3 correlation. `oracle_clean` is fed the CLEAN frames, so it always
#: contributes a "good IQA + high J&F" point regardless of the degradation cell.
IQA_EXCLUDED_MODES = {"oracle_clean"}

#: |rho| above which an image-quality metric could plausibly be used to *select* a
#: restoration operator, i.e. the effect is large enough to change decisions.
#:
#: The threshold exists because the verdict must be read off the EFFECT SIZE, not
#: off a p-value. CL3 pools every frame of every degraded cell, so n is in the
#: thousands; at n=2500 a rho of 0.05 already has p < 0.01. A "significant but
#: tiny" correlation is not evidence against the paper's claim -- it is the
#: expected outcome of a large sample -- so `p` is reported but never decides.
#: rho^2 (the explained share of J&F variance) is printed alongside for the same
#: reason: rho=0.6 means ~36% of the variance, which is when a selection rule
#: could actually work.
IQA_USABLE_RHO = 0.6


def collapse_instances_per_frame(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One observation per (cell, frame), with J&F averaged over the instances.

    The frame-level twin of `collapse_objects`, and the same defect family. CL3
    pairs a per-frame image-quality value with the per-frame J&F, but IQA
    describes the *image*, so it is bit-identical across the instances of a
    sequence -- `00_smoke_test.py` asserts exactly that ("IQA is identical across
    the objects of one sequence"). Under `max_objects: 0` a multi-instance
    sequence therefore contributes each frame once **per instance**: the same x
    repeated, paired with different y. That is pseudo-replication. rho moves
    little, but the bootstrap CI is computed as if the evidence were several
    times larger than it is, which is precisely the weakness a reviewer would
    attack on a paper whose central claim is a NULL correlation.

    Averaging the instances at each frame restores the per-frame mean that the
    official protocol uses at sequence level, and makes the reported `n` the
    number of distinct (sequence, frame) observations it really is.
    """
    buckets: Dict[tuple, list] = {}
    first: Dict[tuple, Mapping[str, Any]] = {}
    order: List[tuple] = []
    for r in rows:
        k = (str(r.get("seq", "")), str(r.get("mode", "")),
             str(r.get("degradation", "")), str(r.get("level", "")),
             str(r.get("arm", "") or ""), str(r.get("frame", "")))
        if k not in first:
            first[k] = r
            order.append(k)
            buckets[k] = []
        v = fnum(r.get("JF_frame"))
        if np.isfinite(v):
            buckets[k].append(v)
    out = []
    for k in order:
        vals = buckets[k]
        if not vals:
            continue
        m = dict(first[k])
        m["JF_frame"] = float(np.mean(vals))
        m["n_instances"] = len(vals)
        out.append(m)
    return out


def _iqa_subset(rows: Sequence[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Rows usable for CL3: one fixed segmentation policy, degraded cells only.

    Ablation-arm rows are dropped for the same reason as in the main tables: with
    ``--iqa-mode dagrs`` the arms A0/A1/A2/A8..A10 carry ``mode="dagrs"`` and would
    otherwise be pooled into the correlation. The default mode is `greedy`, which no
    arm uses, so this guard is inert by default -- but it must not be relied on.
    """
    return [r for r in rows
            if str(r.get("mode", "")) == mode
            and str(r.get("degradation", "")) != "clean"
            and str(r.get("clean_reference", "0")) in ("0", "")
            and not is_arm_row(r)]


def study_iqa(rows: Sequence[Dict[str, Any]], out_dir: Path, fig_dir: Path,
              mode: str = "greedy", n_boot: int = 10000) -> Dict:
    """CL3: does image quality predict downstream J&F under degradation?

    Three filters are mandatory, and each one is a reviewer-visible design
    decision documented in the paper's protocol section:

    1. ONE segmentation policy (`mode`). Pooling arms mixes a mode-level effect
       into the correlation: at identical IQA, `cascade` and `greedy` sit at
       different J&F levels, which manufactures a spurious trend.
    2. Degraded cells only (`degradation != "clean"`): a clean cell has no
       restoration decision to make, so it carries no information about whether
       IQA can *select* an operator.
    3. `oracle_clean` removed (see ``IQA_EXCLUDED_MODES``). It is an upper-bound
       reference fed the clean frames; its IQA describes the clean observation,
       not the degraded one, so pairing it with its own high J&F injects a
       systematic positive bias into exactly the correlation under test.
    """
    print("\n=== STUDY: image quality vs downstream quality (CL3) ===")
    used_mode = mode
    sub = _iqa_subset(rows, used_mode)
    if not sub:
        avail = sorted({str(r.get("mode", "")) for r in rows}
                       - IQA_EXCLUDED_MODES - {""})
        print(f"  no rows for mode={used_mode!r}. available: {avail}")
        if avail:
            used_mode = avail[0]
            print(f"  falling back to mode={used_mode!r}")
            sub = _iqa_subset(rows, used_mode)
    if not sub:
        print("  no data. Run scripts/02_run_baselines.py first.")
        return {}
    print(f"  mode: {used_mode!r} (single policy; {sorted(IQA_EXCLUDED_MODES)} excluded)")
    n_inst_frames = len(sub)
    sub = collapse_instances_per_frame(sub)
    print(f"  per-frame observations: {len(sub)}"
          + (f"  (from {n_inst_frames} instance-frames; instances averaged per frame)"
             if len(sub) != n_inst_frames else ""))

    y = [fnum(r.get("JF_frame")) for r in sub]
    ok = [np.isfinite(v) for v in y]
    sub = [r for r, o in zip(sub, ok) if o]
    y = [fnum(r.get("JF_frame")) for r in sub]

    results = []
    corr_map = {}
    for mk in ALL_IQA_METRICS:
        x = [fnum(r.get(mk)) for r in sub]
        mask = [np.isfinite(a) and np.isfinite(b) for a, b in zip(x, y)]
        xs = [a for a, m in zip(x, mask) if m]
        ys = [b for b, m in zip(y, mask) if m]
        if len(xs) < 10:
            continue
        st = spearman_with_ci(xs, ys, n_boot=int(n_boot), seed=0)
        kind = "full-reference" if mk in FULL_REFERENCE_METRICS else "no-reference"
        rho = float(st["rho"])
        results.append({"metric": mk, "kind": kind, "rho": rho,
                        "ci_lo": st["lo"], "ci_hi": st["hi"], "p": st["p"],
                        "n": st["n"], "r2": rho ** 2,
                        # the verdict, computed from the effect size so that it
                        # cannot drift back onto a p-value threshold
                        "usable": int(abs(rho) >= IQA_USABLE_RHO)})
        corr_map[mk] = st
        sig = "USABLE as a selection criterion" if results[-1]["usable"] \
            else "cannot select an operator"
        print(f"  {mk:<22} rho={rho:+.4f} "
              f"[{st['lo']:+.3f},{st['hi']:+.3f}] p={st['p']:.2e} "
              f"n={int(st['n'])} R2={rho ** 2:.3f} | {kind} {sig}")

    p = write_csv(results, str(out_dir / "iqa_correlation.csv"))
    print(f"  -> {p}")
    write_latex(results, str(out_dir / "table11_iqa_correlation.tex"),
                "Spearman rank correlation between image-quality metrics and "
                "downstream frame-level J\\&F. A near-zero or negative correlation "
                "means image quality cannot be used to select a restoration operator.",
                "tab:iqa_corr")

    # ---- robustness of the verdict to the choice of segmentation policy ----- #
    # A reviewer will ask whether the (null) correlation is an artefact of the
    # single arm we picked. Compute the same rho for every usable arm. The claim
    # only survives if the picture is the same under all of them.
    sens: List[Dict[str, Any]] = []
    other_modes = sorted({str(r.get("mode", "")) for r in rows}
                         - IQA_EXCLUDED_MODES - {""})
    for m in other_modes:
        # same instance-averaging as the main CL3 number, otherwise the two
        # panels of TABLE-11b would not be computed on the same footing
        rows_m = collapse_instances_per_frame(_iqa_subset(rows, m))
        if not rows_m:
            continue
        y_m = [fnum(r.get("JF_frame")) for r in rows_m]
        for mk in ALL_IQA_METRICS:
            xs = [fnum(r.get(mk)) for r in rows_m]
            mask = [np.isfinite(a) and np.isfinite(b) for a, b in zip(xs, y_m)]
            xa = [a for a, k in zip(xs, mask) if k]
            ya = [b for b, k in zip(y_m, mask) if k]
            if len(xa) < 10:
                continue
            st = spearman_with_ci(xa, ya, n_boot=min(200, int(n_boot)), seed=0)
            sens.append({"mode": m, "metric": mk, "rho": st["rho"],
                         "p": st["p"], "n": st["n"]})
    if sens:
        write_csv(sens, str(out_dir / "iqa_correlation_by_mode.csv"))
        print("\n  policy sensitivity (rho per arm; claim must hold for ALL arms,"
              " oracle_clean excluded by construction)")
        hdr = f"  {'metric':<22}" + "".join(f"{m:>13}" for m in other_modes)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for mk in ALL_IQA_METRICS:
            cells = []
            any_v = False
            for m in other_modes:
                hit = next((s for s in sens if s["mode"] == m and s["metric"] == mk), None)
                if hit:
                    any_v = True
                    cells.append(f"{hit['rho']:>+13.3f}")
                else:
                    cells.append(f"{'-':>13}")
            if any_v:
                print(f"  {mk:<22}" + "".join(cells))
        print(f"  -> {out_dir / 'iqa_correlation_by_mode.csv'}")

    print("\n  INTERPRETATION GUIDE")
    print("   decide on |rho| / R^2, NEVER on p: n is in the thousands, so a")
    print("   rho of 0.05 is already 'significant'. See IQA_USABLE_RHO.")
    print(f"   * every metric |rho| < {IQA_USABLE_RHO} (R^2 < {IQA_USABLE_RHO ** 2:.2f})")
    print("     -> no image-quality metric can drive an operator choice:")
    print("        the central claim is SUPPORTED. Phrase it as an effect size")
    print("        ('explains <X% of the J&F variance'), not as 'p > 0.05'")
    print(f"   * any metric with |rho| >= {IQA_USABLE_RHO} and p < 0.01")
    print("     -> that metric could be used as the selection criterion instead;")
    print("        you must then show DAG-RS still wins, or reframe the contribution")
    print("   * a small rho whose CI excludes 0 is NOT a contradiction: it means")
    print("     'too weak to select an operator', which is not the same as")
    print("     'no correlation'. Report rho^2 and the CI, and say which you mean")
    print("   * if rho flips sign across arms above, the claim is NOT supported as")
    print("     stated: restrict the scope to the arm(s) where it holds and say so")

    # figure
    try:
        from xdrp.viz import fig_iqa_scatter
        fig_iqa_scatter(sub, [m for m in ALL_IQA_METRICS if m in corr_map],
                        out_dir=str(fig_dir), correlations=corr_map)
        print(f"  -> {fig_dir / 'F2_iqa_scatter.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] figure failed: {e}")
    return {"correlations": results, "n_obs": len(sub), "mode": used_mode,
            "by_mode": sens}


# --------------------------------------------------------------------------- #
# self-test for the CL3 machinery (known ground truth)
# --------------------------------------------------------------------------- #

def _synth_iqa_rows(n: int, relation: str, seed: int = 0,
                    mode: str = "greedy") -> List[Dict[str, Any]]:
    """Build per-frame rows whose IQA -> J&F relation is known by construction.

    relation="independent"  -> J&F is drawn independently of every IQA metric
    relation="monotone"     -> J&F is a strictly increasing function of iqa_entropy
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        ent = float(rng.uniform(0.0, 1.0))
        rec: Dict[str, Any] = {
            "seq": f"s{i % 25:03d}", "obj_id": 1, "mode": mode,
            "degradation": "fog", "level": float(1 + i % 5),
            "clean_reference": "0", "frame": int(i),
        }
        for mk in ALL_IQA_METRICS:
            rec[mk] = float(rng.uniform(0.0, 1.0))     # all independent of J&F
        rec["iqa_entropy"] = ent
        if relation == "independent":
            rec["JF_frame"] = float(rng.uniform(0.0, 1.0))
        else:
            rec["JF_frame"] = 0.05 + 0.90 * ent
        rows.append(rec)
    return rows


def selftest_cl3(n_boot: int = 200) -> int:
    """Verify that study_iqa recovers known truth, and that its guards hold.

    This is the regression test for the two failure modes that would silently
    invalidate the paper's central claim:

    * pooling several segmentation arms into one correlation
    * letting `oracle_clean` (fed clean frames, so a good IQA by construction)
      contribute points that are structurally IQA-good and J\\&F-good
    """
    import tempfile
    print("\n" + "=" * 68)
    print("SELF-TEST: CL3 correlation machinery (known ground truth)")
    print("=" * 68)
    fails: List[str] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="cl3_selftest_"))
    fig = tmp / "figs"
    fig.mkdir(parents=True, exist_ok=True)

    # --- case A: independent -> no metric may show a real correlation -------- #
    rows_a = _synth_iqa_rows(2000, "independent", seed=1)
    r_a = study_iqa(rows_a, tmp, fig, mode="greedy", n_boot=int(n_boot))
    corr_a = {c["metric"]: c for c in r_a["correlations"]}
    worst = max(corr_a.values(), key=lambda c: abs(c["rho"]))
    report("independent ground truth -> all |rho| small",
           abs(worst["rho"]) < 0.15,
           f"largest |rho|={abs(worst['rho']):.3f} on {worst['metric']}")
    sig_a = [c["metric"] for c in r_a["correlations"] if c["p"] < 0.01]
    report("independent ground truth -> nothing is significant at p<0.01",
           not sig_a, f"significant: {sig_a or 'none'}")

    # --- case B: monotone -> the mechanism must find the signal -------------- #
    rows_b = _synth_iqa_rows(2000, "monotone", seed=2)
    r_b = study_iqa(rows_b, tmp, fig, mode="greedy", n_boot=int(n_boot))
    ent_b = next(c for c in r_b["correlations"] if c["metric"] == "iqa_entropy")
    report("monotone ground truth -> iqa_entropy rho is detected",
           abs(ent_b["rho"]) > 0.9 and ent_b["p"] < 1e-6,
           f"rho={ent_b['rho']:+.3f} p={ent_b['p']:.2e}")
    others = [c for c in r_b["correlations"]
              if c["metric"] != "iqa_entropy" and abs(c["rho"]) < 0.15]
    report("monotone ground truth -> the OTHER metrics stay ~0",
           len(others) == len(r_b["correlations"]) - 1,
           f"{len(others)}/{len(r_b['correlations']) - 1} others below 0.15")

    # --- case C: oracle_clean must be excluded (regression guard) ----------- #
    rows_c = list(rows_a)
    rng = np.random.default_rng(9)
    for i in range(400):
        rec = {"seq": f"o{i:03d}", "obj_id": 1, "mode": "oracle_clean",
               "degradation": "fog", "level": 3.0, "clean_reference": "0",
               "frame": 0, "JF_frame": 1.0}
        for mk in ALL_IQA_METRICS:
            rec[mk] = 1.0            # perfect image quality AND perfect J&F
        rows_c.append(rec)
    r_c = study_iqa(rows_c, tmp, fig, mode="greedy", n_boot=int(n_boot))
    corr_c = {c["metric"]: c for c in r_c["correlations"]}
    wc = max(corr_c.values(), key=lambda c: abs(c["rho"]))
    report("oracle_clean rows do not enter the CL3 correlation",
           abs(wc["rho"]) < 0.15 and r_c["n_obs"] == r_a["n_obs"],
           f"n_obs {r_c['n_obs']} (vs {r_a['n_obs']}), largest |rho|={abs(wc['rho']):.3f}")

    # --- case D: a shifted second arm must not be pooled in ---------------- #
    rows_d = list(rows_a)
    for i in range(400):
        rec = {"seq": f"z{i:03d}", "obj_id": 1, "mode": "cascade",
               "degradation": "fog", "level": 3.0, "clean_reference": "0",
               "frame": 0, "JF_frame": 0.001}
        for mk in ALL_IQA_METRICS:
            rec[mk] = float(rng.uniform(0.9, 1.0))   # good IQA, terrible J&F
        rows_d.append(rec)
    r_d = study_iqa(rows_d, tmp, fig, mode="greedy", n_boot=int(n_boot))
    corr_d = {c["metric"]: c for c in r_d["correlations"]}
    wd = max(corr_d.values(), key=lambda c: abs(c["rho"]))
    report("a different arm is not pooled into the greedy correlation",
           abs(wd["rho"]) < 0.15 and r_d["n_obs"] == r_a["n_obs"],
           f"n_obs {r_d['n_obs']}, largest |rho|={abs(wd['rho']):.3f}")

    # --- case E: an explicit mode selection is honoured -------------------- #
    r_e = study_iqa(rows_d, tmp, fig, mode="cascade", n_boot=int(n_boot))
    report("--iqa-mode selects the requested arm",
           r_e["mode"] == "cascade" and r_e["n_obs"] == 400,
           f"mode={r_e['mode']} n={r_e['n_obs']}")

    # --- case F: instances of one frame must be averaged, not repeated ------ #
    # Every frame of a multi-instance sequence reaches 05_analysis once per
    # instance, carrying the SAME IQA (it describes the image) but its own J&F.
    # Uncollapsed, the reported n counts instance-frames, so the bootstrap CI is
    # too narrow. The two synthetic instances below straddle the original value
    # by +-d, so a correct collapse reproduces case A's rho EXACTLY while
    # reporting case A's n.
    rows_f: List[Dict[str, Any]] = []
    dhalf = 0.05
    for r in rows_a:
        v = fnum(r.get("JF_frame"))
        r1 = dict(r)
        r1["obj_id"] = 1
        r1["JF_frame"] = v + dhalf
        r2 = dict(r)
        r2["obj_id"] = 2
        r2["JF_frame"] = v - dhalf
        rows_f.extend([r1, r2])
    r_f = study_iqa(rows_f, tmp, fig, mode="greedy", n_boot=int(n_boot))
    corr_f = {c["metric"]: c for c in r_f["correlations"]}
    drho = max(abs(corr_f[m]["rho"] - corr_a[m]["rho"]) for m in corr_f)
    report("instances of one frame are averaged, not pseudo-replicated",
           r_f["n_obs"] == r_a["n_obs"] and drho < 1e-9,
           f"n_obs {r_f['n_obs']} (want {r_a['n_obs']}), max|drho|={drho:.2e}")

    # --- case G: significance must never be read as usability -------------- #
    # A weak-but-real relation (rho ~ 0.16) at n=2000 is significant at p<0.01,
    # yet explains ~3% of the variance and could not drive an operator choice.
    # If anyone reintroduces a p-value threshold into the verdict, this fails.
    rng_g = np.random.default_rng(3)
    rows_g = _synth_iqa_rows(2000, "independent", seed=3)
    for r in rows_g:
        r["JF_frame"] = 0.1 * fnum(r["iqa_entropy"]) + 0.6 * float(rng_g.uniform())
    r_g = study_iqa(rows_g, tmp, fig, mode="greedy", n_boot=int(n_boot))
    ent_g = next(c for c in r_g["correlations"] if c["metric"] == "iqa_entropy")
    report("a significant-but-tiny rho is flagged NOT usable",
           ent_g["usable"] == 0 and ent_g["p"] < 0.01
           and abs(ent_g["rho"]) < IQA_USABLE_RHO,
           f"rho={ent_g['rho']:+.3f} R2={ent_g['r2']:.3f} "
           f"p={ent_g['p']:.2e} usable={ent_g['usable']}")

    print("\n" + "-" * 68)
    if fails:
        print(f"CL3 SELF-TEST FAILED ({len(fails)}): " + "; ".join(fails))
        return 1
    print("CL3 SELF-TEST PASSED -- the regression guards hold.")
    print("NOTE this validates the ANALYSIS CODE, not the science: it says nothing")
    print("about whether image quality predicts J&F on real data. Only the real")
    print("run can answer that.")
    return 0


def selftest_pooling() -> int:
    """Regression test: ablation arms must not be pooled into the main tables.

    Several ablation arms deliberately reuse the mode string of the method they
    ablate -- A0/A1/A2/A8/A9/A10 all report `mode="dagrs"`, exactly like the
    real DAG-RS rows. If `arm` is dropped from the grouping key, TABLE-9's DAG-RS
    entry silently becomes a mean over ~18 heterogeneous configurations,
    including the deliberately weakened ones. No error is raised, which is what
    makes it dangerous.

    The fixture uses round numbers so the two behaviours differ by a known,
    checkable amount.
    """
    import tempfile
    print("\n" + "=" * 68)
    print("SELF-TEST: ablation-arm pooling (known ground truth)")
    print("=" * 68)
    fails: List[str] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    SEQ = ("seqA", "seqB", "seqC")
    MAIN_DAGRS, MAIN_GREEDY = 0.50, 0.30
    ARM_A0, ARM_A1 = 0.50, 0.10

    def rec(mode, jf, arm="", note=""):
        return {"seq": s, "obj_id": "1", "mode": mode, "degradation": "fog",
                "level": "3.0", "frame": t, "JF_frame": jf,
                "arm": arm, "arm_note": note}

    rows: List[Dict[str, Any]] = []
    for s in SEQ:
        for t in range(4):
            rows.append(rec("dagrs", MAIN_DAGRS))
            rows.append(rec("greedy", MAIN_GREEDY))
        for t in range(4):
            rows.append(rec("dagrs", ARM_A0, "A0", "full DAG-RS"))
            rows.append(rec("dagrs", ARM_A1, "A1", "equal-weight fusion"))

    # --- 1: the grouping key must separate arm from non-arm ------------------ #
    seq_all = sequence_level(rows)
    has_arm = all("arm" in r for r in seq_all)
    dagrs_keys = {r["arm"] for r in seq_all if r["mode"] == "dagrs"}
    report("sequence_level carries `arm` and separates arm from main rows",
           has_arm and dagrs_keys == {"", "A0", "A1"},
           f"arm values on mode=dagrs: {sorted(dagrs_keys)}")

    # --- 2: main_subset drops exactly the arm rows --------------------------- #
    seq_main = main_subset(seq_all)
    report("main_subset keeps exactly the non-arm buckets",
           len(seq_main) == len([r for r in seq_all if not r["arm"]]),
           f"{len(seq_main)} main buckets of {len(seq_all)} total")

    # --- 3: TABLE-9 must report the main-only value -------------------------- #
    tmp = Path(tempfile.mkdtemp(prefix="pool_selftest_"))
    study_tables(rows, tmp)
    t9 = list(csv.DictReader(open(tmp / "table9_main.csv", newline="", encoding="utf-8")))
    got = next((float(r["J&F_mean"]) for r in t9
                if r["mode"] == "dagrs" and r["degradation"] == "fog"), float("nan"))
    report("TABLE-9 dagrs = main-only value (not pooled with the arms)",
           abs(got - MAIN_DAGRS) < 1e-9, f"got {got:.6f}, expected {MAIN_DAGRS:.6f}")

    # --- 4: non-vacuity -- prove the guard actually changes the answer ------- #
    pooled = (MAIN_DAGRS * 4 + ARM_A0 * 4 + ARM_A1 * 4) / 12.0
    report("the guard is load-bearing: pooling would give a different number",
           abs(pooled - MAIN_DAGRS) > 0.1,
           f"pooled-without-arm would be {pooled:.6f} vs {MAIN_DAGRS:.6f} "
           f"(delta {pooled - MAIN_DAGRS:+.6f})")

    # --- 5: TABLE-13 must exist, be non-empty, and carry the arms ----------- #
    t13_path = tmp / "table13_ablation.csv"
    t13 = (list(csv.DictReader(open(t13_path, newline="", encoding="utf-8")))
           if t13_path.exists() and t13_path.stat().st_size else [])
    by_arm = {str(r["arm"]): r for r in t13}
    report("TABLE-13 is written and contains both arms",
           set(by_arm) == {"A0", "A1"},
           f"arms={sorted(by_arm) or 'EMPTY FILE'}")

    # --- 6: TABLE-13 values and the delta_vs_A0 column ---------------------- #
    ok6 = (by_arm and abs(float(by_arm["A0"]["J&F_mean"]) - ARM_A0) < 1e-9
           and abs(float(by_arm["A1"]["J&F_mean"]) - ARM_A1) < 1e-9
           and abs(float(by_arm["A0"]["delta_vs_A0"]) - 0.0) < 1e-9
           and abs(float(by_arm["A1"]["delta_vs_A0"]) - (ARM_A1 - ARM_A0)) < 1e-9)
    report("TABLE-13 values and delta_vs_A0 are correct", bool(ok6),
           f"A0={by_arm.get('A0', {}).get('J&F_mean', 'n/a')} "
           f"A1={by_arm.get('A1', {}).get('J&F_mean', 'n/a')} "
           f"d(A1)={by_arm.get('A1', {}).get('delta_vs_A0', 'n/a')}")

    # --- 7: CL3 must not absorb arms when --iqa-mode points at the method --- #
    sub_dagrs = _iqa_subset(rows, "dagrs")
    report("CL3 subset excludes arm rows even for --iqa-mode dagrs",
           len(sub_dagrs) == 4 * len(SEQ) and all(not is_arm_row(r) for r in sub_dagrs),
           f"{len(sub_dagrs)} rows kept of {4 * len(SEQ) * 3} on mode='dagrs' "
           f"(expected {4 * len(SEQ)})")

    # --- 8: multi-instance sequences must be averaged, not overwritten ------- #
    # Protocol default since 2026-09-17 is "score every instance". Two failure
    # modes then appear silently: instance counts become weights (tables) and
    # dict-key collisions drop instances (stats). Both are checked here with
    # round numbers whose correct answer is known by construction.
    O1_D, O1_G, O2_D, O2_G = 0.60, 0.40, 0.20, 0.30
    want_d, want_g = (O1_D + O2_D) / 2, (O1_G + O2_G) / 2      # 0.40, 0.35
    rows_mo: List[Dict[str, Any]] = []
    for s in SEQ:
        for t in range(2):
            for obj, jd, jg in ((1, O1_D, O1_G), (2, O2_D, O2_G)):
                for mode, jf in (("dagrs", jd), ("greedy", jg)):
                    rows_mo.append({"seq": s, "obj_id": obj, "mode": mode,
                                    "degradation": "fog", "level": "3.0",
                                    "frame": t, "JF_frame": jf,
                                    "arm": "", "arm_note": ""})
    coll = collapse_objects(sequence_level(rows_mo))
    got_d = float(np.mean([float(r["J&F"]) for r in coll if r["mode"] == "dagrs"]))
    got_g = float(np.mean([float(r["J&F"]) for r in coll if r["mode"] == "greedy"]))
    report("collapse_objects averages instances into a per-sequence value",
           len(coll) == 2 * len(SEQ)
           and all(int(r["n_objects"]) == 2 for r in coll)
           and abs(got_d - want_d) < 1e-9 and abs(got_g - want_g) < 1e-9,
           f"n_rows={len(coll)} dagrs={got_d:.4f} greedy={got_g:.4f} "
           f"(expected {want_d:.2f}/{want_g:.2f})")

    tmp_mo = Path(tempfile.mkdtemp(prefix="pool_selftest_mo_"))
    study_stats(rows_mo, tmp_mo)
    st_path = tmp_mo / "stats_main.csv"
    st_rows = (list(csv.DictReader(open(st_path, newline="", encoding="utf-8")))
               if st_path.exists() and st_path.stat().st_size else [])
    delta = next((float(r["mean_delta"]) for r in st_rows
                  if "dagrs" in str(r.get("comparison", ""))), float("nan"))
    report("paired test uses the object-mean, not the last instance",
           np.isfinite(delta) and abs(delta - (want_d - want_g)) < 1e-9,
           f"mean_delta={delta:+.4f}, expected {want_d - want_g:+.4f}; "
           f"'last instance wins' would have given {O2_D - O2_G:+.4f}")

    print("\n" + "-" * 68)
    if fails:
        print(f"POOLING SELF-TEST FAILED ({len(fails)}): " + "; ".join(fails))
        return 1
    print("POOLING SELF-TEST PASSED -- main tables cannot absorb ablation arms.")
    return 0


# --------------------------------------------------------------------------- #
# study: stats  (TABLE-14)
# --------------------------------------------------------------------------- #

def study_stats(rows: Sequence[Dict[str, Any]], out_dir: Path,
                baseline: str = "greedy", method: str = "dagrs") -> Dict:
    print(f"\n=== STUDY: paired significance tests ({method} vs {baseline}) ===")
    # collapse_objects is mandatory here, not cosmetic: `collect` below keys a
    # dict on (seq, degradation, level), so with more than one instance per
    # sequence the last object would overwrite the others and the test would run
    # on the wrong numbers -- silently.
    seq = collapse_objects(main_subset(sequence_level(rows)))
    if not seq:
        print("  no data.")
        return {}

    def collect(mode: str) -> Dict[Tuple[str, str, str], float]:
        d = {}
        for r in seq:
            if r["mode"] == mode:
                d[(r["seq"], r["degradation"], r["level"])] = float(r["J&F"])
        return d

    a, b = collect(baseline), collect(method)
    common = sorted(set(a) & set(b))
    if not common:
        print(f"  no overlapping cells between '{baseline}' and '{method}'.")
        return {}

    comparisons = []
    for other in sorted({r["mode"] for r in seq}):
        if other == baseline:           # skip the trivially-zero self comparison
            continue
        o = collect(other)
        cc = sorted(set(o) & set(a))
        if not cc:
            continue
        comparisons.append(paired_report([a[k] for k in cc], [o[k] for k in cc],
                                         baseline_name=baseline, method_name=other,
                                         seed=0))
    pvals = [c["wilcoxon_p"] for c in comparisons]
    adj = holm_bonferroni(pvals)
    for c, pa in zip(comparisons, adj):
        c["holm_adjusted_p"] = pa
        c["significant_at_0.05"] = int(pa < 0.05)

    print(f"  paired unit: video sequence ({len(common)} paired cells per comparison)")
    cw, dw, pwid = 28, 9, 11
    print(f"  {'comparison':<{cw}}{'meanD':>{dw}}{'95% CI':>20}{'p':>{pwid}}"
          f"{'holm p':>{pwid}}{'delta':>8}")
    for c in comparisons:
        ci = f"[{c['delta_ci_lo']:+.3f},{c['delta_ci_hi']:+.3f}]"
        print(f"  {c['comparison']:<{cw}}{c['mean_delta']:>+{dw}.4f}{ci:>20}"
              f"{c['wilcoxon_p']:>{pwid}.2e}{c['holm_adjusted_p']:>{pwid}.2e}"
              f"{c['cliffs_delta']:>+8.3f}")

    p = write_csv(comparisons, str(out_dir / "stats_main.csv"))
    print(f"  -> {p}")
    write_latex(comparisons, str(out_dir / "table14_stats.tex"),
                "Paired comparisons across video sequences. The paired unit is the "
                "sequence; $p$-values are two-sided Wilcoxon signed-rank, corrected "
                "with Holm--Bonferroni; CIs are percentile bootstrap (10{,}000 "
                "resamples); effect size is Cliff's $\\delta$.",
                "tab:stats")
    return {"comparisons": comparisons}


# --------------------------------------------------------------------------- #
# study: vacuity  (CL10 / TABLE-16 -- was mislabelled TABLE-15 until 2026-09-23,
#                  which collided with the cross-source table in PAPER_FRAMEWORK 5.5b)
# --------------------------------------------------------------------------- #

def study_vacuity(rows: Sequence[Dict[str, Any]], out_dir: Path) -> Dict:
    print("\n=== STUDY: is vacuity informative? (CL10) ===")
    # Ablation arms share mode strings with the method they ablate (A5/A6/A7 use
    # `dagrs_no_*`, A0/A1/A2/A8..A10 use `dagrs`), so they must be dropped here or
    # CL10 would be measured on a mixture of mechanisms.
    pool = main_subset(rows)
    n_dropped = len(rows) - len(pool)
    if n_dropped:
        print(f"  [excluded {n_dropped} ablation-arm frame rows]")
    sub = [r for r in pool
           if str(r.get("mode", "")).startswith("dagrs")
           and str(r.get("is_keyframe", "0")) in ("1", "1.0")]
    if not sub:
        sub = [r for r in pool if str(r.get("mode", "")).startswith("dagrs")]
    if not sub:
        print("  no DAG-RS data yet.")
        return {}
    fam: Dict[str, int] = defaultdict(int)
    for r in sub:
        fam[str(r.get("mode", ""))] += 1
    print(f"  contributing modes: {dict(sorted(fam.items()))}")

    nu = [fnum(r.get("vacuity")) for r in sub]
    err = [1.0 - fnum(r.get("JF_frame", r.get("J&F"))) for r in sub]
    ok = [np.isfinite(a) and np.isfinite(b) for a, b in zip(nu, err)]
    nus = [a for a, m in zip(nu, ok) if m]
    errs = [b for b, m in zip(err, ok) if m]
    if len(nus) < 20:
        print(f"  only {len(nus)} usable decision-frame observations; need >= 20.")
        return {}

    st = spearman_with_ci(nus, errs, n_boot=10000, seed=0)
    print(f"  Spearman(vacuity, true error) = {st['rho']:+.4f} "
          f"[{st['lo']:+.3f},{st['hi']:+.3f}]  p={st['p']:.2e}  n={int(st['n'])}")
    print("  NOTE for the manuscript: this supports vacuity as a RANKING signal only.")
    print("  Do NOT claim calibration or coverage guarantees.")

    rows_out = [{"statistic": "spearman_vacuity_vs_true_error", "rho": st["rho"],
                 "ci_lo": st["lo"], "ci_hi": st["hi"], "p": st["p"], "n": st["n"]}]
    p = write_csv(rows_out, str(out_dir / "vacuity_rank.csv"))
    print(f"  -> {p}")

    # threshold sweep: would a different tau_vacuity be better?
    best = []
    for tau in np.arange(0.30, 0.86, 0.05):
        hi = [e for n_, e in zip(nus, errs) if n_ >= tau]
        lo = [e for n_, e in zip(nus, errs) if n_ < tau]
        if len(hi) >= 5 and len(lo) >= 5:
            best.append({"tau": float(tau), "mean_err_high_nu": float(np.mean(hi)),
                         "mean_err_low_nu": float(np.mean(lo)),
                         "n_high": len(hi), "n_low": len(lo),
                         "gap": float(np.mean(hi) - np.mean(lo))})
    if best:
        print(f"  {'tau':>6}{'err(nu>=tau)':>14}{'err(nu<tau)':>13}{'gap':>9}")
        for r in best:
            print(f"  {r['tau']:>6.2f}{r['mean_err_high_nu']:>14.4f}"
                  f"{r['mean_err_low_nu']:>13.4f}{r['gap']:>+9.4f}")
        write_csv(best, str(out_dir / "vacuity_threshold_sweep.csv"))
    return {"spearman": st, "threshold_sweep": best}


# --------------------------------------------------------------------------- #
# study: tables
# --------------------------------------------------------------------------- #

def study_tables(rows: Sequence[Dict[str, Any]], out_dir: Path) -> Dict:
    print("\n=== STUDY: main tables ===")
    seq_all = sequence_level(rows)
    # Collapse the instances of a sequence into that sequence's value BEFORE any
    # averaging. Official DAVIS scores a sequence as the mean over its objects;
    # averaging (seq, obj) rows directly would weight each sequence by how many
    # instances it happens to have.
    seq = collapse_objects(main_subset(seq_all))
    if not seq:
        print("  no data.")
        return {}
    n_inst = sum(int(r.get("n_objects", 1)) for r in seq)
    print(f"  {len(seq)} sequence cells; {n_inst} sequence-instance pairs "
          f"collapsed to sequence level (DAVIS protocol)")

    # TABLE 9/10: degradation x mode
    t910 = group_mean(seq, ["degradation", "level", "mode"])
    for r in t910:
        r["ci_lo"], r["ci_hi"] = _ci_of(seq, r)
    write_csv(t910, str(out_dir / "table9_main.csv"))
    print(f"  TABLE-9/10 -> {out_dir / 'table9_main.csv'}  ({len(t910)} rows)")

    # pivot for readability
    piv: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in t910:
        k = (r["degradation"], str(r["level"]))
        piv.setdefault(k, {"degradation": k[0], "level": k[1]})[r["mode"]] = r["J&F_mean"]
    modes = sorted({r["mode"] for r in t910})
    prow = []
    for k in sorted(piv):
        row = {"degradation": piv[k]["degradation"], "level": piv[k]["level"]}
        for m in modes:
            row[m] = piv[k].get(m, np.nan)
        if "greedy" in row and isinstance(row.get("greedy"), float):
            for m in modes:
                v = row.get(m)
                if isinstance(v, float) and np.isfinite(v):
                    row[f"d_{m}"] = v - row["greedy"]
        prow.append(row)
    write_csv(prow, str(out_dir / "table9_pivot.csv"))
    # widen the mode column so long arm names (dagrs_no_reanchor) do not collide
    w = max(20, max(len(m) for m in modes) + 2)
    header = f"{'degradation':<20}{'L':>4}" + "".join(f"{m:>{w}}" for m in modes)
    print("\n" + header)
    print("-" * len(header))
    for r in prow:
        line = f"{r['degradation']:<20}{r['level']:>4}"
        for m in modes:
            v = r.get(m, np.nan)
            line += f"{v:>{w}.4f}" if isinstance(v, float) and np.isfinite(v) \
                else f"{'-':>{w}}"
        print(line)
    write_latex(prow, str(out_dir / "table9_main.tex"),
                "Main results. Mean $\\mathcal{J}\\&\\mathcal{F}$ per degradation type "
                "and severity level. $d_{\\cdot}$ columns are differences from the "
                "greedy baseline.", "tab:main")

    # TABLE 9c (CL1): retention relative to clean.  Deliberately NOT "TABLE-11"
    # -- that label belongs to the IQA correlation table
    # (`table11_iqa_correlation.tex`, CL3).  The old filename collided with it.
    clean = {}
    for r in seq:
        if r["degradation"] == "clean":
            clean.setdefault(r["mode"], []).append(float(r["J&F"]))
    ret_rows = []
    for r in t910:
        c = clean.get(r["mode"], [])
        if c and np.mean(c) > 0:
            ret_rows.append({"degradation": r["degradation"], "level": r["level"],
                             "mode": r["mode"],
                             "J&F": r["J&F_mean"],
                             "retention": r["J&F_mean"] / float(np.mean(c))})
    if ret_rows:
        write_csv(ret_rows, str(out_dir / "table9c_retention.csv"))
        print(f"\n  TABLE-9c retention -> {out_dir / 'table9c_retention.csv'}")
    else:
        print("\n  [note] no 'clean' row found; run the sweep with 'clean' included "
              "to get the retention table.")

    # TABLE 13: ablation. Arms carry a non-empty `arm`; they must never reach
    # TABLE-9/9c/10/11 (see sequence_level's docstring), so this reads `seq_all`.
    arm_rows = collapse_objects([r for r in seq_all if is_arm_row(r)])
    if arm_rows:
        arms = sorted(group_mean(arm_rows, ["arm", "arm_note"]),
                      key=lambda r: _arm_sort_key(str(r["arm"])))
        base = next((r["J&F_mean"] for r in arms if str(r["arm"]) == "A0"), None)
        if base is not None:
            for r in arms:
                r["delta_vs_A0"] = r["J&F_mean"] - float(base)
        write_csv(arms, str(out_dir / "table13_ablation.csv"))
        print(f"  TABLE-13 -> {out_dir / 'table13_ablation.csv'}  ({len(arms)} arms)")
        for r in arms:
            d = r.get("delta_vs_A0")
            dtxt = f"{d:+.4f}" if isinstance(d, float) else "   n/a"
            print(f"    {str(r['arm']):<10} J&F={r['J&F_mean']:.4f}  d(A0)={dtxt}"
                  f"  n={r['n']}  {str(r['arm_note'])[:52]}")
        if base is None:
            print("    [note] arm A0 (full DAG-RS) is not among the inputs, so the "
                  "d(A0) column is absent. Re-run 04_ablation.py with A0 included.")
    else:
        print("  [note] no ablation arms in the inputs -> TABLE-13 not written. "
              "Run scripts/04_ablation.py first.")
    return {"main": t910}


def _arm_sort_key(a: str) -> Tuple[int, str]:
    """Natural order: A0, A1, ..., A9, A10, then the swept variants."""
    m = re.match(r"^A(\d+)", a or "")
    return (int(m.group(1)) if m else 999, a or "")


def _ci_of(seq_rows, target: Dict[str, Any]) -> Tuple[float, float]:
    v = [float(r["J&F"]) for r in seq_rows
         if r["degradation"] == target["degradation"]
         and str(r["level"]) == str(target["level"])
         and r["mode"] == target["mode"]]
    if not v:
        return float("nan"), float("nan")
    _, lo, hi = bootstrap_ci(v, n_boot=5000, seed=0)
    return lo, hi


# --------------------------------------------------------------------------- #
# study: figs
# --------------------------------------------------------------------------- #

def study_figs(rows: Sequence[Dict[str, Any]], fig_dir: Path) -> None:
    print("\n=== STUDY: figures ===")
    seq = collapse_objects(main_subset(sequence_level(rows)))
    try:
        from xdrp.viz import fig_bar_modes, fig_efficiency, fig_severity_curve
        if seq:
            fig_severity_curve(seq, out_dir=str(fig_dir))
            print(f"  -> {fig_dir / 'F3_severity_curve.png'}")
            fig_bar_modes(seq, metric="J&F", group_key="degradation",
                          out_dir=str(fig_dir))
            print(f"  -> {fig_dir / 'F_bar_modes.png'}")
            fig_efficiency(seq, out_dir=str(fig_dir))
            print(f"  -> {fig_dir / 'F6_efficiency.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] figure generation failed: {e}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

DEFAULT_INPUTS = [
    "results/baselines_raw.csv",
    "results/dagrs_main_raw.csv",
    "results/dagrs_ood_raw.csv",
    "results/ablation_raw.csv",
]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="all",
                    choices=["iqa", "stats", "vacuity", "tables", "figs", "all"])
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="CSV files (default: the standard results/*.csv)")
    ap.add_argument("--baseline", default="greedy")
    ap.add_argument("--method", default="dagrs")
    ap.add_argument("--iqa-mode", default="greedy",
                    help="segmentation policy CL3 is computed on (default: greedy, "
                         "i.e. the main baseline); pooling arms is not allowed")
    ap.add_argument("--iqa-boot", type=int, default=10000,
                    help="bootstrap resamples for the CL3 CI")
    ap.add_argument("--out-dir", default=None,
                    help="where tables/figures are written (default: results/). "
                         "Point it at a scratch directory to preview a study on a "
                         "PARTIAL sweep without leaving half-finished tables in "
                         "results/ where they could be read as the final verdict")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the analysis machinery against known ground "
                         "truth (no data needed) and exit: the CL3 correlation "
                         "guards and the ablation-pooling guards")
    args = ap.parse_args()

    if args.self_test:
        rc_cl3 = selftest_cl3()
        rc_pool = selftest_pooling()
        return 1 if (rc_cl3 or rc_pool) else 0

    inp = args.inputs if args.inputs else [str(ROOT / p) for p in DEFAULT_INPUTS]
    rows = read_rows(inp)
    print("XD-RobustPVOS :: analysis")
    print("=" * 68)
    print(f"  inputs : {[str(Path(p).name) for p in inp if Path(p).exists()]}")
    print(f"  records: {len(rows)}")
    if not rows:
        print("\nNo data found. Run these first:")
        print("  python scripts/02_run_baselines.py --quick")
        print("  python scripts/03_run_dagrs.py --quick")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results"
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}
    if args.study in ("iqa", "all"):
        summary["iqa"] = study_iqa(rows, out_dir, fig_dir,
                                   mode=args.iqa_mode, n_boot=args.iqa_boot)
    if args.study in ("stats", "all"):
        summary["stats"] = study_stats(rows, out_dir, args.baseline, args.method)
    if args.study in ("vacuity", "all"):
        summary["vacuity"] = study_vacuity(rows, out_dir)
    if args.study in ("tables", "all"):
        summary["tables"] = study_tables(rows, out_dir)
    if args.study in ("figs", "all"):
        study_figs(rows, fig_dir)

    sp = out_dir / "analysis_summary.json"
    sp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nsummary -> {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
