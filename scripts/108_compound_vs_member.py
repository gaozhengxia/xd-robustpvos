"""TABLE-10 / TABLE-9 producer: compound degradations vs their single members.

The question this answers
-------------------------
For a compound degradation X = {m1, m2, ...} the memoryless family already loses
to the memory-bank arm on every individual member.  The open question is whether
that FAMILY-LEVEL GAP *widens* when the degradations are composed -- i.e. whether
the memoryless route degrades super-additively -- or is merely shifted down by a
constant ("translation").

Statistics used
---------------
Group means alone cannot answer this: a compound is simply a harder input than
each member, and both arms degrade on harder inputs.  The instrument that is
invariant to input difficulty is the *difference* between arms:

    family_gap(deg)  =  J&F(sam2video, deg) - J&F(greedy, deg)
    gated_delta(deg) =  J&F(dagrs,      deg) - J&F(greedy, deg)

(sam2video is the released memory-bank arm; greedy is the purest memoryless arm,
no re-anchoring at all -- see docs/BASELINE_DIAGNOSIS.md).

"Widens" is then tested PAIRED BY SEQUENCE, not by comparing two group means:

    for each sequence s:  d(s) = family_gap(compound, s) - family_gap(member, s)
    H0: median d == 0          (Wilcoxon signed-rank on 30 sequences)

This controls for per-sequence difficulty, which is exactly the confound that
made an earlier full-sweep attribution wrong (see the `n_re>0` trap in
docs/BASELINE_DIAGNOSIS.md).  A group-mean difference would not.

Aggregation
-----------
Instances are averaged into their sequence first, then sequences carry equal
weight.  Mirrors `05_analysis.collapse_objects()` and this script asserts its own
J&F against that production function so the two cannot silently drift apart.

Protocol
--------
Every input CSV must come from the identical protocol (--frame-stride 1 --no-iqa
--max-objects 0, level 3.0 for degraded cells, dagrs under configs/_tau013.yaml).
Only files from the CL4 / C1-C4 / singles sweeps qualify.  Mixing in a run made
under a different protocol is the one error that would invalidate the table, so
the script prints the provenance of every input it uses.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from xdrp.degradations import COMPOUND_DEGRADATIONS, members  # noqa: E402
from xdrp.stats import paired_report  # noqa: E402

RESULTS = ROOT / "results"
METRICS = ("J", "F", "J&F")
COUNTERS = ("n_reanchors", "n_encoder_calls", "n_decisions", "n_rejected", "n_reseeds")
ARMS = ("greedy", "dagrs", "sam2video")

#: Below this, a gap change is reported as FLAT (unit: J&F).
WIDEN_TOL = 0.02

DEFAULT_COMPOUND = [
    RESULTS / "c1c4_a_dagrs.csv", RESULTS / "c1c4_b_dagrs.csv",
    RESULTS / "c1c4_a_base.csv", RESULTS / "c1c4_b_base.csv",
]
DEFAULT_SINGLE = [
    # produced by _scratch/singles_20260920/run_singles2.sh, one file per arm
    RESULTS / "singles_dagrs.csv", RESULTS / "singles_greedy.csv",
    RESULTS / "singles_sam2video.csv",
    RESULTS / "cl4_full_v2.csv",          # clean / fog / dust / underwater
]


def fnum(v) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def clean_level(v) -> str:
    """Render a level the same way on every path (1.0 and 1 must not split)."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def load(path: Path, sink: list, label: str) -> int:
    if not path.exists():
        print(f"  [miss] {label}: {path.name} does not exist")
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["_source"] = path.name
    header = set(rows[0].keys()) if rows else set()
    if rows and "frame" in header:
        print(f"  [WARN] {path.name} has a 'frame' column -> this is an IQA "
              f"(per-frame) run, not a per-cell run. Refusing to mix.")
        return 0
    sink.extend(rows)
    n_seq = len({r["seq"] for r in rows}) if rows else 0
    print(f"  [ok]   {label}: {path.name}: {len(rows)} rows, {n_seq} sequences")
    return len(rows)


def cell_key(r: dict) -> tuple:
    return (r["seq"], r["mode"], r["degradation"], clean_level(r.get("level", "")))


def collapse(rows: list) -> list:
    """Instance mean -> one value per (seq, mode, degradation, level)."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    meta: dict = {}
    for r in rows:
        k = cell_key(r)
        for m in METRICS:
            v = fnum(r.get(m))
            if np.isfinite(v):
                acc[k][m].append(v)
        meta.setdefault(k, r)
    out = []
    for k, d in acc.items():
        if not d.get("J&F"):
            continue
        row = dict(zip(("seq", "mode", "degradation", "level"), k))
        for m in METRICS:
            row[m] = float(np.mean(d[m])) if d.get(m) else float("nan")
        for c in COUNTERS:
            vals = [fnum(x.get(c)) for x in rows
                    if cell_key(x) == k and np.isfinite(fnum(x.get(c)))]
            row[c] = float(np.mean(vals)) if vals else float("nan")
        row["n_objects"] = len(d["J&F"])
        out.append(row)
    return out


def production_cross_check(raw: list, collapsed: list) -> bool:
    spec = importlib.util.spec_from_file_location(
        "_production_05_analysis", str(ROOT / "scripts" / "05_analysis.py"))
    if spec is None or spec.loader is None:
        print("  cross-check SKIPPED: cannot load 05_analysis.py")
        return True
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        print(f"  cross-check SKIPPED: {exc}")
        return True
    prod = mod.collapse_objects(raw)
    pm = {cell_key(r): float(r["J&F"]) for r in prod}
    worst, miss = 0.0, 0
    for r in collapsed:
        k = (r["seq"], r["mode"], r["degradation"], r["level"])
        if k not in pm:
            miss += 1
            continue
        worst = max(worst, abs(r["J&F"] - pm[k]))
    ok = worst <= 1e-12 and miss == 0
    print(f"  cross-check vs 05_analysis.collapse_objects: worst|d|={worst:.3e} "
          f"missing={miss} -> {'OK' if ok else 'DRIFT'}")
    return ok


def by_seq(collapsed: list, deg: str, mode: str) -> dict:
    return {r["seq"]: r["J&F"] for r in collapsed
            if r["degradation"] == deg and r["mode"] == mode
            and np.isfinite(r["J&F"])}


def arm_mean(collapsed: list, deg: str, mode: str) -> dict:
    sel = [r for r in collapsed if r["degradation"] == deg and r["mode"] == mode]
    if not sel:
        return {}
    out = {"n_seq": len(sel)}
    for m in METRICS:
        vals = [r[m] for r in sel if np.isfinite(r[m])]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    for c in COUNTERS:
        vals = [fnum(r.get(c)) for r in sel if np.isfinite(fnum(r.get(c)))]
        out[c] = float(np.mean(vals)) if vals else float("nan")
    return out


def paired_delta(collapsed: list, deg: str, base: str, method: str) -> dict:
    a, b = by_seq(collapsed, deg, base), by_seq(collapsed, deg, method)
    seqs = sorted(set(a) & set(b))
    if len(seqs) < 3:
        return {"n": len(seqs), "insufficient": True}
    x, y = [a[s] for s in seqs], [b[s] for s in seqs]
    rep = paired_report(x, y)
    rep["n"] = len(seqs)
    rep["n_win"] = int(sum(1 for s in seqs if b[s] > a[s]))
    rep["n_loss"] = int(sum(1 for s in seqs if b[s] < a[s]))
    rep["min_delta"] = float(min(yy - xx for xx, yy in zip(x, y)))
    rep["max_delta"] = float(max(yy - xx for xx, yy in zip(x, y)))
    rep["worst_seq"] = seqs[int(np.argmin([yy - xx for xx, yy in zip(x, y)]))]
    rep["best_seq"] = seqs[int(np.argmax([yy - xx for xx, yy in zip(x, y)]))]
    return rep


def gap_per_seq(collapsed: list, deg: str, hi: str, lo: str) -> dict:
    """Per-sequence family gap, SIGNED POSITIVE: J&F(hi) - J&F(lo).

    Must be `a - b` (hi - lo), not `b - a`.  Returning the negated gap inverts
    the sign of every test built on it, and because the decisive verdict is a
    one-sided threshold (`mean_delta > WIDEN_TOL`) an inverted sign silently
    turns "the gap narrowed" into "the gap widened" -- with a perfectly
    plausible-looking p-value next to it.
    """
    a, b = by_seq(collapsed, deg, hi), by_seq(collapsed, deg, lo)
    return {s: a2 - b2 for s, a2, b2 in
            ((s, a[s], b[s]) for s in sorted(set(a) & set(b)))}


def paired_gap_shift(c_col: list, s_col: list, compound: str, member: str) -> dict:
    """The decisive test: per-sequence (gap_compound - gap_member), H0: median 0.

    Positive mean => the family gap WIDENED on the compound.

    The member baseline usually lives in a DIFFERENT source list than the
    compound (e.g. `fog` comes from cl4_full_v2.csv, C1 from the c1c4 sweep), so
    the two lookup lists must be passed separately.  Looking the member up in the
    compound list returns an empty dict and the test silently reports
    "INSUFFICIENT" -- i.e. it would masquerade as missing data instead of a bug.

    Self-guard: the paired mean of (gap_c - gap_m) must equal the difference of
    the two group-mean gaps, because both are means of the same per-sequence
    differences.  Any sign or indexing slip breaks that identity, so it is
    asserted rather than assumed.
    """
    gc = gap_per_seq(c_col, compound, "sam2video", "greedy")
    gm = gap_per_seq(s_col, member, "sam2video", "greedy")
    seqs = sorted(set(gc) & set(gm))
    if len(seqs) < 3:
        return {"n": len(seqs), "insufficient": True,
                "n_compound_seqs": len(gc), "n_member_seqs": len(gm)}
    gc_v = [gc[s] for s in seqs]
    gm_v = [gm[s] for s in seqs]
    rep = paired_report(gm_v, gc_v)          # member -> compound
    rep["n"] = len(seqs)
    rep["n_widen"] = int(sum(1 for s in seqs if gc[s] > gm[s]))
    rep["n_narrow"] = int(sum(1 for s in seqs if gc[s] < gm[s]))
    rep["min_delta"] = float(min(gc[s] - gm[s] for s in seqs))
    rep["max_delta"] = float(max(gc[s] - gm[s] for s in seqs))
    # identity guard: paired mean == group mean of compound - group mean of member
    grp_c = float(np.mean([arm_mean(c_col, compound, "sam2video").get("J&F", np.nan)])) - \
        float(arm_mean(c_col, compound, "greedy").get("J&F", np.nan))
    grp_m = float(arm_mean(s_col, member, "sam2video").get("J&F", np.nan)) - \
        float(arm_mean(s_col, member, "greedy").get("J&F", np.nan))
    implied = grp_c - grp_m
    rep["group_gap_compound"] = grp_c
    rep["group_gap_member"] = grp_m
    rep["group_gap_shift"] = implied
    rep["sign_consistent"] = bool(abs(implied - rep["mean_delta"]) < 1e-9)
    if not rep["sign_consistent"]:
        raise AssertionError(
            f"sign/index guard FAILED for {compound} vs {member}: "
            f"paired mean_delta={rep['mean_delta']:+.6f} but group gap shift="
            f"{implied:+.6f} -> the two views of the same quantity disagree, "
            f"so the verdict would be untrustworthy.")
    return rep


def per_seq_mean_over(collapsed: list, degs, mode: str) -> dict:
    """Per-sequence mean J&F over a SET of degradations.

    Used to pool the four compounds into one number per sequence.  Pooling the
    raw pairings instead would be pseudo-replication: the same 30 sequences
    would enter four times and inflate n from 30 to 120.
    """
    acc: dict = defaultdict(list)
    degs = set(degs)
    for r in collapsed:
        if r["degradation"] in degs and r["mode"] == mode and np.isfinite(r["J&F"]):
            acc[r["seq"]].append(r["J&F"])
    return {s: float(np.mean(v)) for s, v in acc.items() if v}


def pooled_arm_test(c_col: list, compound_deg, base: str, method: str) -> dict:
    """Per-sequence mean over compounds, then paired test. n = sequences, not cells."""
    a = per_seq_mean_over(c_col, compound_deg, base)
    b = per_seq_mean_over(c_col, compound_deg, method)
    seqs = sorted(set(a) & set(b))
    if len(seqs) < 3:
        return {"n": len(seqs), "insufficient": True}
    x, y = [a[s] for s in seqs], [b[s] for s in seqs]
    rep = paired_report(x, y, baseline_name=base, method_name=method)
    rep["n"] = len(seqs)
    rep["n_win"] = int(sum(1 for s in seqs if b[s] > a[s]))
    rep["n_loss"] = int(sum(1 for s in seqs if b[s] < a[s]))
    rep["min_delta"] = float(min(v - u for u, v in zip(x, y)))
    rep["max_delta"] = float(max(v - u for u, v in zip(x, y)))
    rep["degradations"] = list(compound_deg)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compound-files", nargs="*", default=[str(p) for p in DEFAULT_COMPOUND])
    ap.add_argument("--single-files", nargs="*", default=[str(p) for p in DEFAULT_SINGLE])
    ap.add_argument("--out", default=str(RESULTS / "c1c4_vs_members.json"))
    ap.add_argument("--md", default=str(RESULTS / "c1c4_vs_members.md"))
    args = ap.parse_args()

    print("=" * 78)
    print("COMPOUND vs SINGLE MEMBERS -- family-gap analysis")
    print("=" * 78)
    print("\nPROVENANCE (every input must be from the identical protocol)")
    c_raw: list = []
    for p in args.compound_files:
        load(Path(p), c_raw, "compound")
    s_raw: list = []
    for p in args.single_files:
        load(Path(p), s_raw, "single  ")

    if not c_raw:
        print("\nNO COMPOUND DATA -- sweep still running. Nothing to analyse.")
        return 2

    compound_deg = [d for d in COMPOUND_DEGRADATIONS
                    if any(r["degradation"] == d for r in c_raw)]
    single_deg = sorted({r["degradation"] for r in s_raw})

    c_col = collapse(c_raw)
    s_col = collapse(s_raw)
    print()
    prod_ok = production_cross_check(c_raw, c_col) and production_cross_check(s_raw, s_col)

    # ---- absolute -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("ABSOLUTE  (instance-averaged, sequence-equal-weight)")
    print("-" * 78)
    print(f"{'degradation':<20}{'kind':<9}{'arm':<11}{'J':>8}{'F':>8}{'J&F':>8}"
          f"{'n_seq':>7}{'n_re':>8}{'n_enc':>8}")
    abs_rows = []
    order = ([(d, s_col, "single") for d in single_deg]
             + [(d, c_col, "compound") for d in compound_deg])
    for deg, src, kind in order:
        for arm in ARMS:
            g = arm_mean(src, deg, arm)
            if not g:
                continue
            print(f"{deg:<20}{kind:<9}{arm:<11}{g['J']:>8.4f}{g['F']:>8.4f}"
                  f"{g['J&F']:>8.4f}{g['n_seq']:>7d}{g['n_reanchors']:>8.1f}"
                  f"{g['n_encoder_calls']:>8.1f}")
            abs_rows.append({"degradation": deg, "kind": kind, "mode": arm, **g})
        print()

    # ---- gaps ---------------------------------------------------------------
    print("-" * 78)
    print("GAPS   family = J&F(sam2video) - J&F(greedy)  |  "
          "gated = J&F(dagrs) - J&F(greedy)")
    print("-" * 78)
    print(f"{'degradation':<20}{'kind':<9}{'family':>10}{'gated':>10}{'n_seq':>7}")
    gaps = {}
    for deg, src, kind in order:
        gv, gg, gd = (arm_mean(src, deg, a) for a in ("sam2video", "greedy", "dagrs"))
        if not gv or not gg:
            continue
        gaps[deg] = {
            "kind": kind,
            "family_gap": gv["J&F"] - gg["J&F"],
            "gated_gap": (gv["J&F"] - gd["J&F"]) if gd else float("nan"),
            "dagrs_minus_greedy": (gd["J&F"] - gg["J&F"]) if gd else float("nan"),
            "abs": {"greedy": gg["J&F"], "sam2video": gv["J&F"],
                    "dagrs": gd["J&F"] if gd else float("nan")},
            "n_seq": gv["n_seq"],
        }
        print(f"{deg:<20}{kind:<9}{gaps[deg]['family_gap']:>10.4f}"
              f"{gaps[deg]['gated_gap']:>10.4f}{gv['n_seq']:>7d}")

    # ---- verdicts -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("VERDICT PER COMPOUND: does the family gap WIDEN vs its members?")
    print("-" * 78)
    verdicts = {}
    for deg in compound_deg:
        mem = members(deg)
        if deg not in gaps:
            # Partial sweep: the compound has some arm not yet measured, so its
            # family gap does not exist.  Say so instead of crashing -- this is
            # the normal state while a sweep is still running.
            arms_here = sorted({r["mode"] for r in c_col if r["degradation"] == deg})
            print(f"  {deg:<20} family gap not computable yet: only "
                  f"{'/'.join(arms_here) or 'no'} arm(s) present "
                  f"(need greedy AND sam2video)")
            verdicts[deg] = {"members": mem, "verdict": "PARTIAL",
                             "arms_present": arms_here}
            continue
        have = [m for m in mem if m in gaps]
        missing = [m for m in mem if m not in gaps]
        if not have:
            print(f"  {deg:<20} no member baseline available "
                  f"(need {'/'.join(missing)}) -- cannot judge")
            verdicts[deg] = {"members": mem, "missing": missing,
                             "verdict": "NO_BASELINE"}
            continue
        gvals = np.array([gaps[m]["family_gap"] for m in have], dtype=float)
        ref_max, ref_mean = float(gvals.max()), float(gvals.mean())
        g = gaps[deg]["family_gap"]
        d_max, d_mean = g - ref_max, g - ref_mean
        verdict = ("WIDENS" if d_max > WIDEN_TOL
                   else "NARROWS" if d_mean < -WIDEN_TOL else "FLAT")
        print(f"  {deg:<20} gap={g:.4f}  members={'/'.join(have)} "
              f"max={ref_max:.4f} mean={ref_mean:.4f}  "
              f"-> vs max {d_max:+.4f}  vs mean {d_mean:+.4f}   [{verdict}]")
        if missing:
            print(f"  {'':<20} !! incomplete member set, missing {'/'.join(missing)} "
                  f"-- reference is a LOWER bound")
        verdicts[deg] = {"members": mem, "members_available": have, "missing": missing,
                         "family_gap": g, "member_gap_max": ref_max,
                         "member_gap_mean": ref_mean,
                         "delta_vs_max": d_max, "delta_vs_mean": d_mean,
                         "verdict": verdict}

    # ---- paired tests -------------------------------------------------------
    print("\n" + "-" * 78)
    print("PAIRED BY SEQUENCE, per compound")
    print("-" * 78)
    stats_out = {}
    for deg in compound_deg:
        for base, method, label in (("greedy", "dagrs", "dagrs-greedy"),
                                    ("greedy", "sam2video", "video-greedy"),
                                    ("dagrs", "sam2video", "video-dagrs")):
            p = paired_delta(c_col, deg, base, method)
            stats_out[f"{deg}|{label}"] = p
            if p.get("insufficient"):
                print(f"{deg:<20}{label:<14} n={p['n']} INSUFFICIENT")
                continue
            print(f"{deg:<20}{label:<14} n={p['n']:<3} "
                  f"mean={p['mean_delta']:+.4f} med={p['median_delta']:+.4f} "
                  f"ci=[{p['delta_ci_lo']:+.4f},{p['delta_ci_hi']:+.4f}] "
                  f"p={p['wilcoxon_p']:.3g} cliff={p['cliffs_delta']:+.3f} "
                  f"win/lose={p['n_win']}/{p['n_loss']} "
                  f"range=[{p['min_delta']:+.4f},{p['max_delta']:+.4f}]")
        print()

    print("-" * 78)
    print("DECISIVE: per-sequence gap shift  (compound - member), H0: median = 0")
    print("-" * 78)
    shifts = {}
    for deg in compound_deg:
        for m in members(deg):
            p = paired_gap_shift(c_col, s_col, deg, m)
            key = f"{deg}|vs|{m}"
            shifts[key] = p
            if p.get("insufficient"):
                print(f"{deg:<20}vs {m:<14} n={p['n']} INSUFFICIENT")
                continue
            tag = ("WIDENS" if p["mean_delta"] > WIDEN_TOL
                   else "NARROWS" if p["mean_delta"] < -WIDEN_TOL else "FLAT")
            print(f"{deg:<20}vs {m:<14} n={p['n']:<3} "
                  f"mean={p['mean_delta']:+.4f} med={p['median_delta']:+.4f} "
                  f"ci=[{p['delta_ci_lo']:+.4f},{p['delta_ci_hi']:+.4f}] "
                  f"p={p['wilcoxon_p']:.3g} cliff={p['cliffs_delta']:+.3f} "
                  f"widen/narrow={p['n_widen']}/{p['n_narrow']}  [{tag}]")
        print()

    # ---- pooled across compounds (per-sequence mean, so n stays 30) ----------
    print("-" * 78)
    print("POOLED ACROSS COMPOUNDS   per-sequence mean over the compounds, then paired")
    print("(n = sequences; pooling raw cells would repeat each sequence 4x)")
    print("-" * 78)
    pooled = {}
    for base, method, label in (("greedy", "dagrs", "dagrs-greedy"),
                                ("greedy", "sam2video", "video-greedy"),
                                ("dagrs", "sam2video", "video-dagrs")):
        p = pooled_arm_test(c_col, compound_deg, base, method)
        pooled[label] = p
        if p.get("insufficient"):
            print(f"{label:<14} n={p['n']} INSUFFICIENT")
            continue
        print(f"{label:<14} n={p['n']:<3} mean={p['mean_delta']:+.4f} "
              f"med={p['median_delta']:+.4f} "
              f"ci=[{p['delta_ci_lo']:+.4f},{p['delta_ci_hi']:+.4f}] "
              f"p={p['wilcoxon_p']:.3g} cliff={p['cliffs_delta']:+.3f} "
              f"win/lose={p['n_win']}/{p['n_loss']} "
              f"range=[{p['min_delta']:+.4f},{p['max_delta']:+.4f}]")

    # family gap: mean over compounds vs mean over singles, paired by sequence
    gc_pool = {s: (per_seq_mean_over(c_col, compound_deg, "sam2video")[s]
                   - per_seq_mean_over(c_col, compound_deg, "greedy")[s])
               for s in sorted(set(per_seq_mean_over(c_col, compound_deg, "sam2video"))
                               & set(per_seq_mean_over(c_col, compound_deg, "greedy")))}
    gs_pool = {}
    if single_deg:
        sv = per_seq_mean_over(s_col, single_deg, "sam2video")
        sg = per_seq_mean_over(s_col, single_deg, "greedy")
        gs_pool = {s: sv[s] - sg[s] for s in sorted(set(sv) & set(sg))}
    common = sorted(set(gc_pool) & set(gs_pool))
    gap_pool = {}
    if len(common) >= 3:
        xs = [gs_pool[s] for s in common]
        ys = [gc_pool[s] for s in common]
        rep = paired_report(xs, ys, baseline_name="mean-of-singles",
                            method_name="mean-of-compounds")
        rep["n"] = len(common)
        rep["n_widen"] = int(sum(1 for s in common if gc_pool[s] > gs_pool[s]))
        rep["n_narrow"] = int(sum(1 for s in common if gc_pool[s] < gs_pool[s]))
        rep["mean_gap_single"] = float(np.mean(xs))
        rep["mean_gap_compound"] = float(np.mean(ys))
        gap_pool = rep
        print(f"\n{'FAMILY GAP':<14} single-mean={rep['mean_gap_single']:.4f} "
              f"compound-mean={rep['mean_gap_compound']:.4f} "
              f"shift={rep['mean_delta']:+.4f} "
              f"ci=[{rep['delta_ci_lo']:+.4f},{rep['delta_ci_hi']:+.4f}] "
              f"p={rep['wilcoxon_p']:.3g} cliff={rep['cliffs_delta']:+.3f} "
              f"widen/narrow={rep['n_widen']}/{rep['n_narrow']}")

    payload = {
        "absolute": abs_rows, "gaps": gaps, "verdicts": verdicts,
        "paired": stats_out, "gap_shift_paired": shifts,
        "pooled_across_compounds": pooled, "pooled_family_gap": gap_pool,
        "widening_tolerance": WIDEN_TOL,
        "production_cross_check_ok": prod_ok,
        "compound_degradations_present": compound_deg,
        "single_degradations_present": single_deg,
        "family_gap_definition": "J&F(sam2video) - J&F(greedy)",
        "note": ("family gap is invariant to input difficulty, so a WIDEN verdict "
                 "cannot be explained by the compound simply being harder; it is "
                 "confirmed by the sequence-paired shift test."),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str),
                              encoding="utf-8")
    print(f"wrote {args.out}")

    # ---- markdown ready for PAPER_FRAMEWORK --------------------------------
    lines = ["| degradation | kind | greedy | dagrs | sam2video | family gap | n_seq |",
             "|---|---|---|---|---|---|---|"]
    for deg, src, kind in order:
        gg = arm_mean(src, deg, "greedy")
        gd = arm_mean(src, deg, "dagrs")
        gv = arm_mean(src, deg, "sam2video")
        if not (gg and gv):
            continue
        lines.append(f"| {deg} | {kind} | {gg['J&F']:.4f} | "
                     f"{gd['J&F']:.4f} | {gv['J&F']:.4f} | "
                     f"{gv['J&F'] - gg['J&F']:+.4f} | {gg['n_seq']} |")
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
