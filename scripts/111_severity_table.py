"""Severity-curve table + figure builder: CL2 / FIG-3 / PAPER_FRAMEWORK 5.6.

`results/table9_pivot.csv` holds exactly ONE severity for every degradation
(level 3.0, plus `clean` at 1.0), so until now the paper advertised a "5-level
continuous severity axis" in Sec. 3.3 while no experiment in it used that axis.
This script turns `run_severity.sh`'s three sweeps into the missing table.

Degradations are `fog`, `sensor_noise` and `C1_fog_noise`, deliberately: C1 is
exactly the compound of the other two, so the figure can put the compound
severity axis next to its two members' axes.

Everything below is the project's standard convention, not a new one:
`05_analysis.sequence_level` -> `05_analysis.collapse_objects` (instances are
averaged into their sequence first, then the 30 sequences are weighted equally).

Guard rails (all raise, none warn-and-continue)
-----------------------------------------------
1. COMPLETENESS. Each (arm, degradation, level) cell must hold exactly 61
   instance rows over the same 30 sequences, and the full 3x5 grid must be
   present. A partially written sweep must never be tabulated.
2. LEVEL-3 RECONCILIATION. The level-3 cells are re-measured here although
   `results/table9_main.csv` already covers them, and they must reproduce the
   published per-instance rows BIT FOR BIT. This is what makes "one curve = one
   protocol" a checked claim instead of an assertion -- and it is the only
   end-to-end evidence that the curve sits on the same protocol as TABLE-9/10.

Usage
-----
    python scripts/111_severity_table.py --dir _scratch/severity_20260921 \\
        --out-dir results --prefix severity_curve --fig
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from xdrp.stats import (bootstrap_ci, cliffs_delta, holm_bonferroni,  # noqa: E402
                        paired_report)

EXPECT_INSTANCES = 61
EXPECT_SEQUENCES = 30
JTOL = 1e-9

#: arm -> file tag.  The sweep writes one CSV per arm (no concurrent append).
ARMS: Tuple[Tuple[str, str], ...] = (
    ("dagrs", "severity_dagrs.csv"),
    ("greedy", "severity_greedy.csv"),
    ("sam2video", "severity_sam2video.csv"),
)
#: the arm the family gap is measured against, and the gated-family member.
MEMORY_ARM = "sam2video"
FAMILY_ARM = "dagrs"

#: The subtrahend is not a detail, it is half of the estimand, and the two
#: defensible choices DISAGREE here.  Sec. 5.2 DEFINES the family gap as
#: `sam2video - greedy` ("the arm that does no re-anchoring at all"); Sec. 5.6's
#: own question -- does severity modulate the memory bank's advantage -- is about
#: the ARM UNDER TEST, i.e. `sam2video - dagrs`.  Those are two different
#: quantities, and on this data they disagree in sign for two of the three
#: degradations (fog +0.0271 vs -0.0079; C1 -0.0895 with a clustered-free
#: interval that EXCLUDES zero, vs -0.0402 whose interval contains it).  So
#: reporting one and calling the other "the family gap" would let the paper's
#: "0 WIDENS" verdict depend silently on which arm was subtracted.
#:
#: Both are therefore computed from the same per-sequence arrays and emitted:
#: `family_gap` (primary, key kept for every existing consumer) and
#: `family_gap_sensitivity`.  `SUBTRAHENDS` below is the single place that
#: decides which arm plays which role -- do not hard-code a subtrahend at a call
#: site, which is exactly how the two sections drifted apart in the first place.
SENSITIVITY_ARM = "greedy"
SUBTRAHENDS: Tuple[Tuple[str, str], ...] = ((FAMILY_ARM, "primary"),
                                           (SENSITIVITY_ARM, "sensitivity"))

DEGRADATIONS: Tuple[str, ...] = ("fog", "sensor_noise", "C1_fog_noise")
LEVELS: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

#: Published per-instance rows already in the repo, keyed by (arm).  The level-3
#: cells of the sweep must match these exactly.  c1c4 was run in two chunks
#: (a: 18 sequences, b: 12) which together hold the full 61 instances.
LEVEL3_SOURCES: Dict[str, Dict[str, List[str]]] = {
    "fog": {"dagrs": ["results/cl4_full_v2.csv"],
            "greedy": ["results/cl4_full_v2.csv"],
            "sam2video": ["results/cl4_full_v2.csv"]},
    "sensor_noise": {"dagrs": ["results/singles_dagrs.csv"],
                     "greedy": ["results/singles_greedy.csv"],
                     "sam2video": ["results/singles_sam2video.csv"]},
    "C1_fog_noise": {"dagrs": ["results/c1c4_a_dagrs.csv", "results/c1c4_b_dagrs.csv"],
                     "greedy": ["results/c1c4_a_base.csv", "results/c1c4_b_base.csv"],
                     "sam2video": ["results/c1c4_a_base.csv", "results/c1c4_b_base.csv"]},
}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a05 = _load("a05", "scripts/05_analysis.py")


def read_csv(path: Path, what: str = "") -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"ABORT: missing {what or ''} {path}")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"ABORT: {path} is empty")
    return rows


def per_sequence(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    """Standard collapse -> {seq: J&F} (instances averaged in, sequences equal)."""
    return {str(r["seq"]): float(r["J&F"])
            for r in a05.collapse_objects(a05.sequence_level(rows))}


def load_grid(src: Path) -> Dict[str, Dict[Tuple[str, float], List[Dict[str, str]]]]:
    """-> {arm: {(degradation, level): rows}}, with the completeness guard."""
    grid: Dict[str, Dict[Tuple[str, float], List[Dict[str, str]]]] = {}
    problems: List[str] = []
    for arm, fname in ARMS:
        rows = read_csv(src / fname, f"arm {arm}")
        cells: Dict[Tuple[str, float], List[Dict[str, str]]] = {}
        for r in rows:
            try:
                key = (r["degradation"], float(r["level"]))
            except (KeyError, ValueError):
                problems.append(f"{fname}: unparsable row {r!r:.120}")
                continue
            cells.setdefault(key, []).append(r)
        grid[arm] = cells
        print(f"read {arm:<10} {len(rows):>5} rows, {len(cells)} cells")
        for deg in DEGRADATIONS:
            for lv in LEVELS:
                got = cells.get((deg, lv))
                if not got:
                    problems.append(f"{arm}: missing cell {deg}@{lv:g}")
                    continue
                if len(got) != EXPECT_INSTANCES:
                    problems.append(
                        f"{arm}: {deg}@{lv:g} has {len(got)} instance rows, "
                        f"expected {EXPECT_INSTANCES} (incomplete unit)")
                seqs = {str(r["seq"]) for r in got}
                if len(seqs) != EXPECT_SEQUENCES:
                    problems.append(f"{arm}: {deg}@{lv:g} covers {len(seqs)} "
                                    f"sequences, expected {EXPECT_SEQUENCES}")
    if problems:
        raise SystemExit("ABORT: the sweep is not a complete 3x5 grid per arm:\n  "
                         + "\n  ".join(problems[:20])
                         + (f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else ""))
    return grid


def reconcile_level3(grid: Dict[str, Dict[Tuple[str, float], List[Dict[str, str]]]],
                     published_root: Optional[Path] = None) -> List[str]:
    """The sweep's level-3 cells must reproduce the PUBLISHED rows bit for bit.

    Without this, the 5-point curve would be stitched together from files
    produced weeks apart under an assumed-identical protocol -- and "assumed
    identical" is exactly the failure this project has paid for twice.  With it,
    the curve carries end-to-end evidence that it sits on TABLE-9/10's protocol.

    `published_root` exists so the same guard can run under the P1 denominator
    protocol (scripts/115_frame_denominator.py): the P1 tree MIRRORS the repo
    layout, so pointing the root at it keeps the comparison apples-to-apples
    instead of checking P1 sweep rows against P0 published rows, which would
    fail for the wrong reason.
    """
    root = Path(published_root) if published_root else ROOT
    lines: List[str] = []
    bad: List[str] = []
    for deg in DEGRADATIONS:
        for arm, _ in ARMS:
            pub: Dict[Tuple[str, str], Dict[str, str]] = {}
            for rel in LEVEL3_SOURCES[deg][arm]:
                for r in read_csv(root / rel, "published rows"):
                    # MODE MUST BE FILTERED.  One published file holds several
                    # modes with the same (seq, obj_id), so keying on those two
                    # alone would let the last mode read (sam2video) overwrite
                    # the rest and every arm would be checked against the wrong
                    # comparator -- a silent pass.  Caught by this file's own
                    # self-test on planted data.
                    if r.get("mode") != arm:
                        continue
                    if r.get("degradation") == deg and float(r.get("level", "nan")) == 3.0:
                        pub[(str(r["seq"]), str(r["obj_id"]))] = r
            got = grid[arm].get((deg, 3.0), [])
            if not pub:
                bad.append(f"{deg}/{arm}: no published level-3 rows found")
                continue
            # An EMPTY sweep cell would otherwise pass silently: with n=0 the
            # loop below leaves worst=0.0 and missing=0, so `worst > JTOL or
            # missing` is False and the cell is reported as reconciled.  Today
            # `load_grid` refuses to run on an incomplete grid, so this is
            # unreachable in production -- but a guard whose failure mode is
            # "pass" must state that requirement itself instead of inheriting
            # it from a caller two functions away.
            if not got:
                bad.append(f"{deg}/{arm}: no sweep rows at level 3 to reconcile "
                           f"(an empty cell must not count as a pass)")
                continue
            worst, worst_key, n = 0.0, None, 0
            missing = 0
            for r in got:
                k = (str(r["seq"]), str(r["obj_id"]))
                p = pub.get(k)
                if p is None:
                    missing += 1
                    continue
                d = abs(float(r["J&F"]) - float(p["J&F"]))
                n += 1
                if d > worst:
                    worst, worst_key = d, k
                if str(r.get("n_frames")) != str(p.get("n_frames")) or \
                        str(r.get("frame_stride")) != str(p.get("frame_stride")):
                    bad.append(f"{deg}/{arm}/{k}: n_frames or frame_stride differs "
                               f"({r.get('n_frames')}@{r.get('frame_stride')} vs "
                               f"{p.get('n_frames')}@{p.get('frame_stride')})")
            lines.append(f"level-3 reconciliation {deg:<14} {arm:<10} "
                         f"n={n:<3} worst|dJ&F|={worst:.3e} unmatched={missing}")
            if worst > JTOL or missing:
                bad.append(f"{deg}/{arm}: worst|dJ&F|={worst:.3e} (> {JTOL:.0e}), "
                           f"unmatched={missing} -- the severity sweep is NOT on "
                           f"the protocol of the published table")
    for ln in lines:
        print("  " + ln)
    if bad:
        raise SystemExit("ABORT: level-3 reconciliation FAILED:\n  " + "\n  ".join(bad))
    return lines


def per_sequence_rank_trend(rows_by_level: Dict[float, Dict[str, float]]) -> Dict[str, Any]:
    """CL2, done the paired way: one Spearman rho per SEQUENCE over its 5 levels.

    Aggregating to 5 points first would give n = 5 and no power.  Computing the
    rank correlation WITHIN each sequence and then testing the 30 rho values
    against 0 keeps the sequence as the independent unit -- the same discipline
    every other test in this paper follows.
    """
    from xdrp.stats import _spearman  # noqa: PLC0415

    lv = sorted(rows_by_level)
    seqs = sorted(rows_by_level[lv[0]])
    rhos: List[float] = []
    for s in seqs:
        y = [rows_by_level[l][s] for l in lv]
        if any(not np.isfinite(v) for v in y):
            continue
        r = _spearman(np.asarray(lv, float), np.asarray(y, float))
        if np.isfinite(r):
            rhos.append(float(r))
    n_neg = sum(1 for r in rhos if r < 0)
    rep = paired_report([0.0] * len(rhos), rhos, baseline_name="no trend",
                        method_name="monotone", seed=0)
    drop = {s: rows_by_level[lv[-1]][s] - rows_by_level[lv[0]][s] for s in seqs}
    dr = paired_report([0.0] * len(drop), list(drop.values()),
                       baseline_name="flat", method_name="l1->l5", seed=0)
    return {"n_sequences": len(rhos), "n_negative": n_neg,
            "rho_mean": float(np.mean(rhos)), "rho_min": float(np.min(rhos)),
            "rho_max": float(np.max(rhos)),
            "rho_p": rep["wilcoxon_p"], "rho_cliff": rep["cliffs_delta"],
            "drop_mean": dr["mean_delta"], "drop_ci_lo": dr["delta_ci_lo"],
            "drop_ci_hi": dr["delta_ci_hi"], "drop_p": dr["wilcoxon_p"],
            "drop_cliff": dr["cliffs_delta"],
            "drop_wins": sum(1 for v in drop.values() if v < 0),
            "drop_losses": sum(1 for v in drop.values() if v > 0)}


def family_gap_block(seq_jf: Dict[str, Dict[str, Dict[float, Dict[str, float]]]],
                     subtrahend: str, n_boot: int) -> List[Dict[str, Any]]:
    """The gap `memory arm - subtrahend`, per cell and per level, plus the endpoint test.

    One function, called once per subtrahend, so the ONLY thing that can differ
    between the primary block and the sensitivity block is the arm subtracted.
    Written this way because the alternative -- two hand-maintained copies of this
    loop -- is how Sec. 5.6's subtrahend drifted away from Sec. 5.2's definition
    without anything failing.

    Per SEQUENCE and per level: gap = MEMORY_ARM - subtrahend.  The question of
    Sec. 5.6 is whether this is a constant translation (=> degradation lowers
    everything equally and gating does not change the response) or widens with
    severity.  Endpoint test = level 5 minus level 1, paired by sequence.
    """
    gaps: List[Dict[str, Any]] = []
    for deg in DEGRADATIONS:
        per_lv: Dict[float, Dict[str, float]] = {}
        for lv in LEVELS:
            s_mem = seq_jf[MEMORY_ARM][deg][lv]
            s_sub = seq_jf[subtrahend][deg][lv]
            per_lv[lv] = {s: s_mem[s] - s_sub[s] for s in sorted(s_mem) if s in s_sub}
        for lv in LEVELS:
            vals = per_lv[lv]
            seqs = sorted(vals)
            # n must be asserted, not inferred: a subtrahend whose sequence set
            # did not overlap would give n = 0 and a silently empty cell.
            if len(seqs) != EXPECT_SEQUENCES:
                raise SystemExit(
                    f"ABORT: gap cell {deg}@{lv:g}/{MEMORY_ARM}-{subtrahend} rests on "
                    f"{len(seqs)} sequences, expected {EXPECT_SEQUENCES} -- the two "
                    f"arms must cover the same sequences for a paired gap")
            point, lo, hi = bootstrap_ci([vals[s] for s in seqs],
                                         n_boot=n_boot, seed=0)
            gaps.append({"degradation": deg, "level": lv, "gap_mean": point,
                         "ci_lo": lo, "ci_hi": hi, "n": len(seqs)})
        d1, d5 = per_lv[LEVELS[0]], per_lv[LEVELS[-1]]
        common = [s for s in sorted(d1) if s in d5]
        shift = {s: d5[s] - d1[s] for s in common}
        rep = paired_report([0.0] * len(common), list(shift.values()),
                            baseline_name="level 1", method_name="level 5", seed=0)
        lo1, hi1 = bootstrap_ci(list(d1.values()), n_boot=n_boot, seed=0)[1:]
        lo5, hi5 = bootstrap_ci(list(d5.values()), n_boot=n_boot, seed=0)[1:]
        gaps[-1]["endpoint"] = {
            "gap_l1": float(np.mean(list(d1.values()))),
            "gap_l1_ci": [lo1, hi1],
            "gap_l5": float(np.mean(list(d5.values()))),
            "gap_l5_ci": [lo5, hi5],
            "shift": rep["mean_delta"], "shift_ci_lo": rep["delta_ci_lo"],
            "shift_ci_hi": rep["delta_ci_hi"], "p": rep["wilcoxon_p"],
            "cliff": rep["cliffs_delta"],
            "wider": sum(1 for v in shift.values() if v > 0),
            "narrower": sum(1 for v in shift.values() if v < 0),
        }
    return gaps


def _synth_grid(dst: Path, widen: str = "C1_fog_noise",
                perturb_level3: bool = False, diverge: bool = False) -> None:
    """Build a complete 3x5 grid from the PUBLISHED level-3 rows.

    Known-positive synthetic input, per the project's rule that a probe must be
    shown to fire on a planted signal before its negative findings are believed.
    Level 3 is copied through untouched (so the reconciliation guard must PASS);
    the other levels get a planted per-sequence slope, and `widen`'s memory arm
    gets an extra severity-dependent bonus (so the gap test must call THAT one
    widening).  `perturb_level3` corrupts one level-3 J&F to prove the guard can
    fail.

    `diverge` plants the bonus on `SENSITIVITY_ARM` ONLY.  That makes the two
    subtrahends disagree by construction -- the sensitivity gap must widen while
    the primary gap stays flat -- which is the only way to show that the two
    blocks are driven by the arm subtracted rather than by a shared code path.
    The shared per-sequence slope is subtracted from every arm, so `memory -
    dagrs` is left exactly constant when only `greedy` is penalised.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for arm, fname in ARMS:
        out: List[Dict[str, str]] = []
        for deg in DEGRADATIONS:
            pub: Dict[Tuple[str, str], Dict[str, str]] = {}
            for rel in LEVEL3_SOURCES[deg][arm]:
                for r in read_csv(ROOT / rel, "published rows"):
                    if r.get("mode") != arm:
                        continue
                    if r.get("degradation") == deg and float(r.get("level", "nan")) == 3.0:
                        pub[(str(r["seq"]), str(r["obj_id"]))] = r
            if len(pub) != EXPECT_INSTANCES:
                raise SystemExit(f"self-test: {deg}/{arm} published rows = "
                                 f"{len(pub)}, expected {EXPECT_INSTANCES}")
            for lv in LEVELS:
                for i, (k, r) in enumerate(sorted(pub.items())):
                    seq = k[0]
                    # deterministic per-sequence slope (NOT hash(): PYTHONHASHSEED
                    # would make the self-test flap between runs).  Applied to
                    # EVERY arm, so it cancels out of any memory - X gap.
                    slope = 0.02 + 0.002 * (sum(ord(c) for c in seq) % 7)
                    jf = float(r["J&F"]) - slope * (lv - 3.0)
                    # Planted WIDENING: the gated family degrades faster with
                    # severity than the memory arm, so memory - family grows.
                    if diverge:
                        hit = arm == SENSITIVITY_ARM
                    else:
                        hit = arm in (FAMILY_ARM, SENSITIVITY_ARM)
                    if deg == widen and hit:
                        jf -= 0.015 * (lv - 3.0)
                    if perturb_level3 and lv == 3.0 and i == 0 and deg == "fog":
                        jf += 0.37
                    row = dict(r)
                    row["level"] = str(lv)
                    row["J&F"] = f"{jf:.10f}"
                    out.append(row)
        with (dst / fname).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


def self_test() -> int:
    """Fire the guards on planted signals, in both directions. n_boot small."""
    import tempfile

    fails: List[str] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    print("111 severity self-test")
    with tempfile.TemporaryDirectory(prefix="sev_self_") as tmp:
        d = Path(tmp)
        _synth_grid(d / "good")
        rc = main(["--dir", str(d / "good"), "--out-dir", str(d / "out"),
                   "--prefix", "t", "--n-boot", "400"])
        ok("planted grid tabulates (reconciliation passes on untouched level 3)",
           rc == 0)
        p = json.loads((d / "out" / "t.json").read_text(encoding="utf-8"))
        tr = p["monotonicity"]["fog"]["dagrs"]
        ok("monotonicity recovers the planted decline (all 30 sequences negative)",
           tr["n_negative"] == tr["n_sequences"] and tr["drop_mean"] < 0,
           f"{tr['n_negative']}/{tr['n_sequences']} negative, "
           f"drop={tr['drop_mean']:+.4f}")
        e_w = [g for g in p["family_gap"]
               if g["degradation"] == "C1_fog_noise" and g["level"] == 5.0][0]["endpoint"]
        e_c = [g for g in p["family_gap"]
               if g["degradation"] == "fog" and g["level"] == 5.0][0]["endpoint"]
        ok("endpoint test calls the planted-widening degradation WIDENING",
           e_w["shift"] > 0 and e_w["wider"] > e_w["narrower"],
           f"shift={e_w['shift']:+.4f} {e_w['wider']}/{e_w['narrower']}")
        ok("endpoint test calls an unplanted degradation FLAT (specificity)",
           abs(e_c["shift"]) < 0.004,
           f"shift={e_c['shift']:+.4f} {e_c['wider']}/{e_c['narrower']}")

        _synth_grid(d / "bad", perturb_level3=True)
        try:
            main(["--dir", str(d / "bad"), "--out-dir", str(d / "out2"),
                  "--prefix", "t", "--n-boot", "200"])
            ok("negative control: a corrupted level-3 J&F makes the guard ABORT",
               False, "it tabulated anyway")
        except SystemExit as ex:
            msg = str(ex)
            ok("negative control: a corrupted level-3 J&F makes the guard ABORT",
               "reconciliation FAILED" in msg, msg.strip().splitlines()[-1][:90])

        # ---- dual subtrahend: both blocks must be real and independent -------
        def _endpoint_shift(block, deg):
            return [g for g in block if g["degradation"] == deg
                    and g["level"] == LEVELS[-1]][0]["endpoint"]["shift"]

        _synth_grid(d / "diverge", diverge=True)
        rc = main(["--dir", str(d / "diverge"), "--out-dir", str(d / "out3"),
                   "--prefix", "t", "--n-boot", "400"])
        ok("a grid with a subtrahend-dependent effect still tabulates", rc == 0)
        p3 = json.loads((d / "out3" / "t.json").read_text(encoding="utf-8"))
        sub = p3.get("family_gap_subtrahends", {})
        ok("the payload names both subtrahends and they are distinct arms",
           sub.get("primary") == FAMILY_ARM
           and sub.get("sensitivity") == SENSITIVITY_ARM
           and sub.get("memory_arm") == MEMORY_ARM
           and sub.get("primary") != sub.get("sensitivity"),
           f"memory={sub.get('memory_arm')!r} primary={sub.get('primary')!r} "
           f"sensitivity={sub.get('sensitivity')!r}")
        sens_block = p3.get("family_gap_sensitivity", [])
        ok("the sensitivity block is a full 3-degradation x 5-level grid",
           len(sens_block) == len(DEGRADATIONS) * len(LEVELS)
           and {g["n"] for g in sens_block} == {EXPECT_SEQUENCES},
           f"{len(sens_block)} cells, n={sorted({g['n'] for g in sens_block})}")
        s_w = _endpoint_shift(sens_block, "C1_fog_noise")
        p_w = _endpoint_shift(p3["family_gap"], "C1_fog_noise")
        ok("planting the effect on the SENSITIVITY arm alone widens the "
           "sensitivity block while the primary block stays flat",
           s_w > 0.02 and abs(p_w) < 0.004,
           f"sensitivity={s_w:+.4f} primary={p_w:+.4f}")

        # ---- negative control: a collapsed declaration must ABORT ------------
        global SUBTRAHENDS
        keep = SUBTRAHENDS
        SUBTRAHENDS = ((FAMILY_ARM, "primary"), (FAMILY_ARM, "sensitivity"))
        try:
            main(["--dir", str(d / "diverge"), "--out-dir", str(d / "out4"),
                  "--prefix", "t", "--n-boot", "200"])
            ok("negative control: declaring the SAME arm for both roles ABORTs",
               False, "it tabulated a decorative sensitivity block anyway")
        except SystemExit as ex:
            ok("negative control: declaring the SAME arm for both roles ABORTs",
               "same arm twice" in str(ex),
               str(ex).strip().splitlines()[-1][:90])
        finally:
            SUBTRAHENDS = keep

        # ---- positive control against the PUBLISHED Table 17 -----------------
        # Calibration lives in the owned script now: on the REAL P1 inputs the
        # primary block must reproduce the shifts printed in the manuscript bit
        # for bit.  `shift` is a mean of per-sequence differences, so it does not
        # depend on --n-boot and this control is stable across regenerations.
        # (This is the calibration `_scratch/family_arm_audit.py` performed by
        # hand when the disagreement was first found.)
        real = ROOT / "results" / "p1" / "severity_curve_p1.json"
        if not real.exists():
            ok("primary block calibrated against the published Table 17", False,
               f"{real.relative_to(ROOT)} is absent -- no calibration possible")
        else:
            pr = json.loads(real.read_text(encoding="utf-8"))
            pub = {"fog": -0.007877881181837298,
                   "sensor_noise": 0.010871200562214245,
                   "C1_fog_noise": -0.04018915919585134}
            got = {g["degradation"]: g["endpoint"]["shift"]
                   for g in pr["family_gap"] if "endpoint" in g}
            worst = max(abs(got[k] - v) for k, v in pub.items())
            ok("primary block reproduces the published Table 17 shifts bit for bit",
               worst < 1e-12, f"worst|d|={worst:.3e}")
            sens = {g["degradation"]: g["endpoint"]["shift"]
                    for g in pr.get("family_gap_sensitivity", [])
                    if "endpoint" in g}
            ok("on the real data the two subtrahends disagree in sign on fog "
               "(which is why one column is not enough)",
               sens.get("fog", 0.0) > 0 > got["fog"],
               f"greedy={sens.get('fog', float('nan')):+.4f} "
               f"dagrs={got['fog']:+.4f}")

    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="_scratch/severity_20260921")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="severity_curve")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--self-test", action="store_true",
                    help="run the guards on planted synthetic data and exit")
    ap.add_argument("--fig", action="store_true", help="also write results/figs/F3_severity.png")
    ap.add_argument("--published-root", default=None,
                    help="root holding the published level-3 rows to reconcile "
                         "against (default: the repo root). Point it at the P1 tree "
                         "(scripts/115_frame_denominator.py --emit) to reconcile a "
                         "P1 sweep against P1 published rows")
    args, _unknown = ap.parse_known_args(argv)
    if args.self_test:
        return self_test()

    src = Path(args.dir)
    if not src.is_absolute():
        src = ROOT / src
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid(src)
    recon = reconcile_level3(grid, args.published_root)

    # ---- aggregate: per (arm, degradation, level) over the 30 sequences -----
    seq_jf: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
    for arm, _ in ARMS:
        seq_jf[arm] = {}
        for deg in DEGRADATIONS:
            seq_jf[arm][deg] = {}
            for lv in LEVELS:
                seq_jf[arm][deg][lv] = per_sequence(grid[arm][(deg, lv)])

    cells: List[Dict[str, Any]] = []
    for deg in DEGRADATIONS:
        for lv in LEVELS:
            for arm, _ in ARMS:
                vals = seq_jf[arm][deg][lv]
                seqs = sorted(vals)
                point, lo, hi = bootstrap_ci([vals[s] for s in seqs],
                                             n_boot=args.n_boot, seed=0)
                cells.append({"degradation": deg, "level": lv, "mode": arm,
                              "J&F_mean": point, "ci_lo": lo, "ci_hi": hi,
                              "n": len(seqs)})

    def cell(deg: str, lv: float, arm: str) -> Dict[str, Any]:
        for c in cells:
            if c["degradation"] == deg and c["level"] == lv and c["mode"] == arm:
                return c
        raise SystemExit(f"internal: no cell {deg}@{lv}/{arm}")

    # ---- CL2: monotone response (paired within sequence) -------------------
    trend: Dict[str, Dict[str, Any]] = {}
    for deg in DEGRADATIONS:
        trend[deg] = {}
        for arm, _ in ARMS:
            trend[deg][arm] = per_sequence_rank_trend(seq_jf[arm][deg])
    p_adj = holm_bonferroni([trend[d][a]["rho_p"] for d in DEGRADATIONS for a, _ in ARMS])
    k = 0
    for d in DEGRADATIONS:
        for a, _ in ARMS:
            trend[d][a]["rho_p_holm"] = float(p_adj[k]); k += 1

    # ---- the family gap vs severity, for EVERY declared subtrahend ----------
    # `family_gap` (primary = the arm under test) keeps its exact old shape, so
    # every existing consumer is unaffected; `family_gap_sensitivity` is the same
    # quantity under Sec. 5.2's declared definition.  These are NOT redundant:
    # see SUBTRAHENDS for the two degradations on which they disagree in sign.
    gaps_by_sub = {arm: family_gap_block(seq_jf, arm, args.n_boot)
                   for arm, _role in SUBTRAHENDS}
    # Two ways for this to become decorative, both asserted rather than trusted.
    # The assertions run BEFORE any lookup: `gaps_by_sub` is keyed by arm, so a
    # duplicated declaration collapses it to one key and the lookup below would
    # die with a bare KeyError instead of naming the actual defect -- the failure
    # mode has to state the rule it is enforcing, not just crash near it.
    declared = [arm for arm, _role in SUBTRAHENDS]
    if len(set(declared)) != len(declared):
        raise SystemExit(
            f"ABORT: SUBTRAHENDS declares the same arm twice ({declared}) -- the "
            f"primary and sensitivity blocks would be the same object, so the "
            f"dual reporting would be decorative rather than evidential")
    gaps = gaps_by_sub[FAMILY_ARM]
    gaps_sens = gaps_by_sub[SENSITIVITY_ARM]
    # (b) two distinct arms whose blocks come out equal anyway -- impossible
    # unless the wiring ignores the subtrahend.
    if all(g["gap_mean"] == s["gap_mean"] for g, s in zip(gaps, gaps_sens)):
        raise SystemExit(
            "ABORT: the two declared subtrahends yielded identical gap sequences "
            "-- the sensitivity block is a no-op, so dual reporting would be "
            "decorative rather than evidential")

    payload = {
        "note": ("Severity curve (CL2 / FIG-3). Degradations fog, sensor_noise and "
                 "C1_fog_noise (= fog + sensor_noise) over levels 1-5, three arms. "
                 "Aggregation = sequence_level -> collapse_objects -> equal-weight "
                 "mean over 30 sequences; CIs are percentile bootstrap over "
                 "sequences. Level 3 is re-measured rather than reused and is "
                 "reconciled bit-for-bit against the published TABLE-9/10 rows "
                 "(see `level3_reconciliation`), so the 5-point curve carries "
                 "end-to-end evidence of sitting on the paper's protocol."),
        "protocol": {"config": "configs/_tau013.yaml", "frame_stride": 1,
                     "max_objects": 0, "levels": list(LEVELS),
                     "degradations": list(DEGRADATIONS),
                     "expect_instances": EXPECT_INSTANCES,
                     "expect_sequences": EXPECT_SEQUENCES},
        "level3_reconciliation": recon,
        "cells": cells,
        "monotonicity": trend,
        "family_gap": gaps,
        # The subtrahend is half of the estimand.  Sec. 5.2 defines the family gap
        # as `sam2video - greedy`; this section's question is about the arm under
        # test, so its primary block subtracts `dagrs`.  Both are emitted, with
        # the division of labour stated here rather than left to the reader.
        "family_gap_subtrahends": {
            "primary": FAMILY_ARM,
            "sensitivity": SENSITIVITY_ARM,
            "memory_arm": MEMORY_ARM,
            "primary_means": f"the arm under test: J&F({MEMORY_ARM}) - J&F({FAMILY_ARM})",
            "sensitivity_means": (f"the manuscript's declared definition: "
                                  f"J&F({MEMORY_ARM}) - J&F({SENSITIVITY_ARM})"),
            "why_two": ("the two disagree in sign for fog and C1_fog_noise, so a "
                        "'the gap does not widen' verdict is subtrahend-dependent "
                        "and must be reported as such"),
        },
        "family_gap_sensitivity": gaps_sens,
    }
    jpath = out_dir / f"{args.prefix}.json"
    jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    cpath = out_dir / f"{args.prefix}.csv"
    with cpath.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["degradation", "level", "mode",
                                           "J&F_mean", "ci_lo", "ci_hi", "n"])
        w.writeheader()
        w.writerows(cells)

    L: List[str] = []
    L.append("# 严重度曲线（CL2 / FIG-3；脚本生成，勿手改）")
    L.append("")
    L.append(f"- 协议：`configs/_tau013.yaml`，stride 1，max_objects 0"
             f"（{EXPECT_SEQUENCES} 段 / {EXPECT_INSTANCES} 实例）")
    L.append("- 聚合：`sequence_level` → `collapse_objects` → 段等权平均；"
             "CI = 段级自助法 95% 百分位区间")
    L.append("- **level 3 是重测而非复用**，并与已发布的 TABLE-9/10 原始行逐位对账：")
    for ln in recon:
        L.append(f"  - `{ln}`")
    L.append("")
    L.append("| 退化 | 严重度 | " + " | ".join(a for a, _ in ARMS)
             + " | 家族缺口（memory − family） |")
    L.append("|---|---|" + "---|" * (len(ARMS) + 1))
    gmap = {(g["degradation"], g["level"]): g for g in gaps}
    for deg in DEGRADATIONS:
        for lv in LEVELS:
            cs = [cell(deg, lv, a) for a, _ in ARMS]
            g = gmap[(deg, lv)]
            L.append(f"| {deg} | {lv:g} | "
                     + " | ".join(f"{c['J&F_mean']:.4f} "
                                  f"[{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]" for c in cs)
                     + f" | {g['gap_mean']:+.4f} [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}] |")
    L.append("")
    L.append("## CL2：单调性（逐序列排序相关，n = 30 条独立 ρ）")
    L.append("")
    L.append("| 退化 | 臂 | ρ 均值 | ρ 范围 | 负号段数 | Holm p | Cliff δ | "
             "level1→5 位移 | 位移 CI | 下降段数 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for deg in DEGRADATIONS:
        for arm, _ in ARMS:
            t = trend[deg][arm]
            L.append(f"| {deg} | {arm} | {t['rho_mean']:+.3f} | "
                     f"[{t['rho_min']:+.3f}, {t['rho_max']:+.3f}] | "
                     f"{t['n_negative']}/{t['n_sequences']} | {t['rho_p_holm']:.2g} | "
                     f"{t['rho_cliff']:+.3f} | {t['drop_mean']:+.4f} | "
                     f"[{t['drop_ci_lo']:+.4f}, {t['drop_ci_hi']:+.4f}] | "
                     f"{t['drop_wins']}/{t['drop_losses']} |")
    L.append("")
    L.append("## 缺口随严重度：端点配对检验（level 5 − level 1，n = 30 段）")
    L.append("")
    L.append(f"**两个减数都报。** 主口径 = `{MEMORY_ARM} − {FAMILY_ARM}`（**被测臂**："
             f"本节的问句是「记忆库相对被测臂的优势是否随严重度变化」）；"
             f"敏感性 = `{MEMORY_ARM} − {SENSITIVITY_ARM}`（**§5.2 的明文定义**："
             f"「完全不重锚定的臂」）。两列**不可互相替代**：`fog` 与 `C1_fog_noise` 上"
             "位移**符号相反**，因此「缺口不随严重度扩大」是一个**减数依赖**的结论。")
    L.append("")
    for sub_arm, role in SUBTRAHENDS:
        gm = {(g["degradation"], g["level"]): g for g in gaps_by_sub[sub_arm]}
        L.append(f"### 减数 = `{sub_arm}`（{role}）")
        L.append("")
        L.append("| 退化 | 缺口@1 | 缺口@5 | 位移 | 95% CI | p | Cliff δ | 扩大/收窄 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for deg in DEGRADATIONS:
            e = gm[(deg, LEVELS[-1])]["endpoint"]
            L.append(f"| {deg} | {e['gap_l1']:+.4f} | {e['gap_l5']:+.4f} | "
                     f"{e['shift']:+.4f} | [{e['shift_ci_lo']:+.4f}, "
                     f"{e['shift_ci_hi']:+.4f}] | {e['p']:.3g} | {e['cliff']:+.3f} | "
                     f"{e['wider']}/{e['narrower']} |")
        L.append("")
    mpath = out_dir / f"{args.prefix}.md"
    mpath.write_text("\n".join(L) + "\n", encoding="utf-8")

    print()
    print(f"{'degradation':<16}{'lvl':>4}" + "".join(f"{a:>13}" for a, _ in ARMS)
          + f"{'gap':>10}")
    for g in gaps:
        row = "".join(f"{cell(g['degradation'], g['level'], a)['J&F_mean']:>13.4f}"
                      for a, _ in ARMS)
        print(f"{g['degradation']:<16}{g['level']:>4.0f}{row}{g['gap_mean']:>+10.4f}")
    print()
    for sub_arm, role in SUBTRAHENDS:
        gm = {(g["degradation"], g["level"]): g for g in gaps_by_sub[sub_arm]}
        print(f"-- gap = {MEMORY_ARM} - {sub_arm}  ({role})")
        for deg in DEGRADATIONS:
            e = gm[(deg, LEVELS[-1])]["endpoint"]
            print(f"   gap shift {deg:<16} {e['shift']:+.4f} "
                  f"[{e['shift_ci_lo']:+.4f}, {e['shift_ci_hi']:+.4f}] p={e['p']:.3g} "
                  f"cliff={e['cliff']:+.3f} {e['wider']} wider / {e['narrower']} narrower")
    print()
    print(f"json -> {jpath}")
    print(f"csv  -> {cpath}")
    print(f"md   -> {mpath}")

    if args.fig:
        make_fig(cells, gaps, out_dir / "figs")
    return 0


def make_fig(cells: List[Dict[str, Any]], gaps: List[Dict[str, Any]], fig_dir: Path) -> None:
    """FIG-3: J&F vs severity, one panel per degradation, three arms."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(figures skipped: {e})")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(DEGRADATIONS), figsize=(4.6 * len(DEGRADATIONS), 3.6),
                             sharey=True)
    if len(DEGRADATIONS) == 1:
        axes = [axes]
    for ax, deg in zip(axes, DEGRADATIONS):
        for arm, _ in ARMS:
            cs = [c for c in cells if c["degradation"] == deg and c["mode"] == arm]
            cs.sort(key=lambda c: c["level"])
            xs = [c["level"] for c in cs]
            ax.plot(xs, [c["J&F_mean"] for c in cs], marker="o", label=arm)
            ax.fill_between(xs, [c["ci_lo"] for c in cs], [c["ci_hi"] for c in cs],
                            alpha=0.16)
        ax.set_title(deg)
        ax.set_xlabel("severity level")
        ax.grid(alpha=0.3)
    # NOTE: mathtext has no `\&`; the ampersand must sit OUTSIDE math mode or
    # matplotlib raises ParseException (this crashed the first --fig run).
    axes[0].set_ylabel(r"$\mathcal{J}$&$\mathcal{F}$")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "F3_severity.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"fig  -> {p}")


if __name__ == "__main__":
    raise SystemExit(main())
