#!/usr/bin/env python
"""114 - Cross-SOURCE table builder: CL12 / PAPER_FRAMEWORK 5.5b / [[TABLE-15]].

WHY
---
The paper's title promises "Cross-Domain" and, until 2026-09-22, `data/` held a
single VOS source (DAVIS-2017 val).  Every "domain shift" in the paper was in
fact produced by the synthetic degradation model.  `run_mose_xsrc.sh` now
evaluates the same protocol on a second, independent source (`data/MOSE_mini`,
MOSEv2 train subset, 24 videos / 1450 frames / 58 instances).

This script turns those three arm CSVs into the table, and it is built the same
way `111_severity_table.py` is built, for the same reason: a number that a human
copies is a number that can silently drift from its source.

THE TWO QUANTITIES THE SECTION ASKS FOR
---------------------------------------
  gap(arm, deg)      = mean_seq [sam2video - arm]          (family gap)
  extra(arm, deg)    = gap(arm, deg) - gap(arm, clean)

`extra` is the point of the section.  A second source differs from DAVIS in
general, so `gap(arm, deg)` alone cannot distinguish
  (a) "this route is a constant 0.16 worse on this source's content"  from
  (b) "degradation widens the gap further on this source".
`extra` isolates (b) by differencing out the source-level offset.  Both answers
are publishable; they demand different wording, which is why the number must
come from a script rather than from reading the table with the eye.

GUARD RAILS (all raise -- none warn-and-continue)
-------------------------------------------------
1. COMPLETENESS. Every (arm, degradation) cell must hold exactly
   `instance_count` rows covering exactly the `sequence_count` sequences named
   by `data/MOSE_mini/manifest.csv`, with no duplicate (seq, obj_id).  A
   partially written sweep must never be tabulated.  The expected counts come
   from the MANIFEST, so they cannot go stale behind the data.
2. ARM CONSISTENCY. All three arms must cover the SAME sequence set.  If the
   sam2video wave aborted (the 120-frame clip risk documented in
   run_mose_xsrc.sh) this aborts instead of quietly emitting a two-column
   "table" that a reader would take as three-way.
3. SELF-TEST (`--self-test`) exercises every guard above on synthetic data with
   a PLANTED effect, and adds a NEGATIVE CONTROL: when the arms are identical,
   `extra` must be ~0 AND its CI must contain 0.  A gap detector that cannot
   report "no gap" is not evidence of anything.

Usage
-----
    python scripts/114_xsource_table.py --dir _scratch/mose_xsrc \\
        --out-dir results --prefix xsource
    python scripts/114_xsource_table.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from xdrp.stats import bootstrap_ci, paired_report  # noqa: E402

#: arm -> file written by run_mose_xsrc.sh
ARMS: Tuple[Tuple[str, str], ...] = (
    ("dagrs", "xsrc_dagrs.csv"),
    ("greedy", "xsrc_greedy.csv"),
    ("sam2video", "xsrc_sam2video.csv"),
)
#: the arm the family gap is measured against (the memory bank), and the two
#: members of the gated family.
MEMORY_ARM = "sam2video"
FAMILY_ARMS: Tuple[str, ...] = ("dagrs", "greedy")

DEGRADATIONS: Tuple[str, ...] = ("clean", "fog", "sensor_noise", "C1_fog_noise")
LEVELS: Tuple[float, ...] = (3.0,)

DEFAULT_MANIFEST = "data/MOSE_mini/manifest.csv"


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


def read_manifest(path: Path) -> Dict[str, int]:
    """-> {seq: n_instances}.  The external anchor for the completeness guard."""
    rows = read_csv(path, "MOSE manifest")
    out: Dict[str, int] = {}
    for r in rows:
        v = r["video_id"]
        ids = [x for x in str(r["object_ids"]).split("|") if x]
        out[v] = len(ids)
    if not out:
        raise SystemExit(f"ABORT: manifest {path} names no sequences")
    return out


def per_sequence(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    """Standard collapse -> {seq: J&F} (instances averaged in, sequences equal)."""
    return {str(r["seq"]): float(r["J&F"])
            for r in a05.collapse_objects(a05.sequence_level(rows))}


def load_grid(src: Path,
              manifest: Dict[str, int]
              ) -> Dict[str, Dict[Tuple[str, float], List[Dict[str, str]]]]:
    """-> {arm: {(deg, level): rows}}, with the completeness + arm guards."""
    want_seqs = set(manifest)
    n_inst = sum(manifest.values())
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
                if len(got) != n_inst:
                    problems.append(
                        f"{arm}: {deg}@{lv:g} has {len(got)} instance rows, "
                        f"expected {n_inst} (incomplete unit)")
                keys = [(str(r["seq"]), str(r["obj_id"])) for r in got]
                if len(set(keys)) != len(keys):
                    problems.append(f"{arm}: {deg}@{lv:g} has duplicate (seq,obj)")
                seqs = {str(r["seq"]) for r in got}
                if seqs != want_seqs:
                    problems.append(
                        f"{arm}: {deg}@{lv:g} covers {len(seqs)} sequences, the "
                        f"manifest names {len(want_seqs)} "
                        f"(missing={sorted(want_seqs - seqs)[:3]}, "
                        f"extra={sorted(seqs - want_seqs)[:3]})")
    if problems:
        raise SystemExit(
            "ABORT: the cross-source grid is not complete (this is the guard "
            "that stops a partially written sweep from being tabulated, and it "
            "is also what refuses to emit a two-arm 'table' when the sam2video "
            "wave aborted):\n  " + "\n  ".join(problems[:20])
            + (f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else ""))
    return grid


def build(grid, n_boot: int) -> Dict[str, Any]:
    """Per-degradation per-arm stats, family gaps, and the `extra` contrast."""
    out: Dict[str, Any] = {"arity": list(ARMS), "degradations": {}, "gaps": {}}
    for deg in DEGRADATIONS:
        cell = {arm: per_sequence(grid[arm][(deg, LEVELS[0])]) for arm, _ in ARMS}
        seqs = sorted(cell[ARMS[0][0]])
        arm_stats = {}
        for arm, vals in cell.items():
            v = [vals[s] for s in seqs]
            point, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=0)
            arm_stats[arm] = {"mean": point, "ci_lo": lo, "ci_hi": hi,
                              "n_sequences": len(v)}
        out["degradations"][deg] = arm_stats
        out["_cell_" + deg] = {arm: {s: cell[arm][s] for s in seqs} for arm, _ in ARMS}

    # family gaps and the source-level offset / extra-gap contrast
    clean_gap: Dict[str, Dict[str, float]] = {}
    for fam in FAMILY_ARMS:
        gap_clean = {s: out["_cell_clean"][MEMORY_ARM][s] - out["_cell_clean"][fam][s]
                     for s in sorted(out["_cell_clean"][MEMORY_ARM])}
        clean_gap[fam] = gap_clean
        entry: Dict[str, Any] = {}
        p, lo, hi = bootstrap_ci(list(gap_clean.values()), n_boot=n_boot, seed=0)
        entry["at_clean"] = {"mean": p, "ci_lo": lo, "ci_hi": hi}
        for deg in DEGRADATIONS:
            if deg == "clean":
                continue
            g = {s: out["_cell_" + deg][MEMORY_ARM][s] - out["_cell_" + deg][fam][s]
                 for s in sorted(out["_cell_" + deg][MEMORY_ARM])}
            p2, lo2, hi2 = bootstrap_ci(list(g.values()), n_boot=n_boot, seed=0)
            pr = paired_report([out["_cell_" + deg][fam][s] for s in sorted(g)],
                               [out["_cell_" + deg][MEMORY_ARM][s] for s in sorted(g)],
                               baseline_name=fam, method_name=MEMORY_ARM)
            # the headline: does degradation ADD to the source-level offset?
            extra = [g[s] - gap_clean[s] for s in sorted(g)]
            p3, lo3, hi3 = bootstrap_ci(extra, n_boot=n_boot, seed=0)
            entry[deg] = {
                "gap": {"mean": p2, "ci_lo": lo2, "ci_hi": hi2},
                "paired": pr,
                "extra_vs_clean": {"mean": p3, "ci_lo": lo3, "ci_hi": hi3,
                                   "ci_includes_zero": bool(lo3 <= 0.0 <= hi3)},
            }
        out["gaps"][fam] = entry
    return out


def write_outputs(res: Dict[str, Any], out_dir: Path, prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    payload = {k: v for k, v in res.items() if not k.startswith("_cell_")}
    jp = out_dir / f"{prefix}.json"
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(jp)

    rows = []
    for deg in DEGRADATIONS:
        for arm, _ in ARMS:
            st = res["degradations"][deg][arm]
            rows.append({"degradation": deg, "arm": arm,
                         "mean_JF": round(st["mean"], 6),
                         "ci_lo": round(st["ci_lo"], 6),
                         "ci_hi": round(st["ci_hi"], 6),
                         "n_sequences": st["n_sequences"]})
    cp = out_dir / f"{prefix}.csv"
    with cp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    written.append(cp)

    L: List[str] = []
    L.append(f"# Cross-source table ({prefix})")
    L.append("")
    L.append("Aggregation = `sequence_level` -> `collapse_objects` -> equal-weight "
             "mean over sequences; CIs are percentile bootstrap over sequences.")
    L.append("")
    L.append("| degradation | " + " | ".join(a for a, _ in ARMS) + " |")
    L.append("|---|" + "---|" * len(ARMS))
    for deg in DEGRADATIONS:
        cells = []
        for arm, _ in ARMS:
            st = res["degradations"][deg][arm]
            cells.append(f"{st['mean']:.4f} [{st['ci_lo']:.4f}, {st['ci_hi']:.4f}]")
        L.append(f"| {deg} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## family gap (sam2video - family) and the extra-gap contrast")
    L.append("")
    L.append("| family | at clean | degradation | gap | extra vs clean | CI contains 0 |")
    L.append("|---|---|---|---|---|---|")
    for fam in FAMILY_ARMS:
        e = res["gaps"][fam]
        L.append(f"| {fam} | {e['at_clean']['mean']:+.4f} | — | — | — | — |")
        for deg in DEGRADATIONS:
            if deg == "clean":
                continue
            d = e[deg]
            L.append(f"| {fam} | | {deg} | {d['gap']['mean']:+.4f} "
                     f"[{d['gap']['ci_lo']:+.4f}, {d['gap']['ci_hi']:+.4f}] | "
                     f"{d['extra_vs_clean']['mean']:+.4f} "
                     f"[{d['extra_vs_clean']['ci_lo']:+.4f}, "
                     f"{d['extra_vs_clean']['ci_hi']:+.4f}] | "
                     f"{'yes' if d['extra_vs_clean']['ci_includes_zero'] else 'no'} |")
    mp = out_dir / f"{prefix}.md"
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    written.append(mp)
    return written


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

#: the real MOSE mini shape, taken from data/MOSE_mini/manifest.csv:
#: 24 videos holding 1,2,3,4,5,6 instances in the counts 9,6,3,3,2,1 = 58.
_SYNTH_OBJ_HIST: Tuple[Tuple[int, int], ...] = ((1, 9), (2, 6), (3, 3),
                                                (4, 3), (5, 2), (6, 1))


def _synth_dir(dst: Path, memory_offset_clean: float, extra_at_deg: float,
               identical: bool = False, null_noise: float = 0.0,
               seed: int = 0) -> Path:
    """Write a synthetic cross-source grid with KNOWN planted values.

    sam2video = base + memory_offset_clean * jitter   on `clean`
    sam2video = base + (memory_offset_clean + extra) * jitter  on degraded levels
    family arms = base

    so gap(clean) should be recovered as `memory_offset_clean` and
    extra_vs_clean as `extra_at_deg`.  With `identical=True` every arm gets the
    same values, which is the negative control: nothing may be reported.

    The per-sequence `jitter` (mean 1, spread +-40%) is NOT decoration: with a
    constant planted offset every per-sequence difference is identical, the
    bootstrap CI degenerates to a point, and "the CI excludes 0" passes for
    free.  The jitter forces the CI machinery to actually be exercised.
    """
    if dst.exists():
        _best_effort_rmtree(dst)
    dst.mkdir(parents=True)

    rng = random.Random(seed)
    seqs: List[str] = []
    objects: Dict[str, List[int]] = {}
    i = 0
    for n_obj, n_seq in _SYNTH_OBJ_HIST:
        for _ in range(n_seq):
            v = f"synth{i:03d}"
            i += 1
            seqs.append(v)
            objects[v] = list(range(1, n_obj + 1))
    seqs.sort()

    base: Dict[Tuple[str, int], float] = {}
    for v in seqs:
        for o in objects[v]:
            base[(v, o)] = rng.uniform(0.30, 0.90)
    jitter = {v: rng.uniform(0.6, 1.4) for v in seqs}
    #: per (arm, seq, object) perturbation for the null world; drawn up-front so
    #: that the three arms of one replication are independent of each other.
    noise: Dict[Tuple[str, str, int], float] = {}
    if null_noise > 0.0:
        for arm, _ in ARMS:
            for v in seqs:
                for o in objects[v]:
                    noise[(arm, v, o)] = rng.uniform(-null_noise, null_noise)

    def val(arm: str, deg: str, v: str, o: int) -> float:
        if identical:
            return base[(v, o)]
        if null_noise > 0.0:
            # null world with real noise: every arm is the same base plus its
            # own independent perturbation, so per-sequence gaps are non-zero
            # but their MEAN must stay indistinguishable from zero.
            return min(1.0, max(0.0, base[(v, o)] + noise[(arm, v, o)]))
        if arm != MEMORY_ARM:
            return base[(v, o)]
        off = memory_offset_clean + (0.0 if deg == "clean" else extra_at_deg)
        return min(1.0, max(0.0, base[(v, o)] + off * jitter[v]))

    for arm, fname in ARMS:
        with (dst / fname).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["seq", "obj_id", "mode", "degradation", "level", "J&F",
                        "frame_stride"])
            for deg in DEGRADATIONS:
                for lv in LEVELS:
                    for v in seqs:
                        for o in objects[v]:
                            w.writerow([v, o, arm, deg, lv,
                                        f"{val(arm, deg, v, o):.6f}", 1])
    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "n_frames", "n_objects", "object_ids"])
        for v in seqs:
            w.writerow([v, 60, len(objects[v]),
                        "|".join(str(o) for o in objects[v])])
    return dst


def _best_effort_rmtree(path) -> None:
    """Remove a self-test tree, never turning a cleanup failure into a verdict.

    The self-test builds a ~600-file synthetic grid.  Deleting such a tree in one
    call is a BULK delete, and an environment-level safe-delete guard may refuse
    it (observed: `SAFE_DELETE_BULK_CONFIRM_REQUIRED`, count 600 > threshold 50,
    budget scoped to the turn).  When that guard fires the caller sees a
    non-zero exit even though every check passed -- the self-test then looks
    broken while the code under test is fine.

    Cleanup is therefore decoupled: a refusal is swallowed here, and the run is
    made indifferent to leftovers by giving each invocation its own directory.
    """
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except BaseException:  # noqa: BLE001 - sandbox guards may exit, not raise
        pass


def self_test() -> int:
    # Unique per process.  A FIXED path forced every run to delete the previous
    # run's ~600-file tree first, which is exactly the bulk delete the guard
    # blocks -> the self-test failed inside the smoke test (where earlier tests
    # had already spent the turn's delete budget) but passed standalone.
    d = ROOT / f"_scratch/114_selftest_{os.getpid()}"
    print("114 self-test")
    print("=" * 68)
    fails: List[str] = []
    checks = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  [{'ok' if cond else 'FAIL'}]   {msg}")
        if not cond:
            fails.append(msg)

    # ---- 1. planted effect must be recovered ------------------------------
    # Tolerances are generous relative to the sampling error of a mean over 24
    # sequences (se ~ 0.005 at these effect sizes) -- they exist to catch a
    # wrong sign or a wrong differencing, not to test the RNG.
    off, extra = 0.10, -0.05
    tol = 0.03
    _synth_dir(d / "planted", off, extra)
    man = read_manifest(d / "planted" / "manifest.csv")
    grid = load_grid(d / "planted", man)
    res = build(grid, n_boot=2000)
    g = res["gaps"]["dagrs"]
    check(abs(g["at_clean"]["mean"] - off) < tol,
          f"planted clean offset recovered: {g['at_clean']['mean']:+.6f} "
          f"(planted {off:+.6f}, tol {tol})")
    check(g["at_clean"]["ci_hi"] - g["at_clean"]["ci_lo"] > 0,
          "clean offset CI is non-degenerate (bootstrap actually exercised)")
    for deg in ("fog", "sensor_noise", "C1_fog_noise"):
        e = g[deg]["extra_vs_clean"]
        check(abs(e["mean"] - extra) < tol,
              f"planted extra gap recovered on {deg}: {e['mean']:+.6f} "
              f"(planted {extra:+.6f}, tol {tol})")
        check(not e["ci_includes_zero"],
              f"{deg}: CI excludes 0 for a planted non-zero effect "
              f"[{e['ci_lo']:+.6f}, {e['ci_hi']:+.6f}]")
        check(e["ci_hi"] - e["ci_lo"] > 0,
              f"{deg}: extra-gap CI is non-degenerate")

    # ---- 2. NEGATIVE CONTROL: identical arms must report nothing ----------
    _synth_dir(d / "identical", off, extra, identical=True)
    man2 = read_manifest(d / "identical" / "manifest.csv")
    grid2 = load_grid(d / "identical", man2)
    res2 = build(grid2, n_boot=2000)
    g2 = res2["gaps"]["dagrs"]
    check(abs(g2["at_clean"]["mean"]) < 1e-9,
          f"negative control: clean gap is ~0 ({g2['at_clean']['mean']:+.9f})")
    for deg in ("fog", "sensor_noise", "C1_fog_noise"):
        e = g2[deg]["extra_vs_clean"]
        check(abs(e["mean"]) < 1e-9 and e["ci_includes_zero"],
              f"negative control on {deg}: extra={e['mean']:+.9f}, "
              f"CI [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] contains 0")

    # ---- 3. NULL WORLD WITH NOISE, calibrated as a FALSE-POSITIVE RATE ----
    # A single no-effect draw must NOT be required to produce a CI containing 0:
    # a 95% CI excludes 0 about 5% of the time under the null, so that assertion
    # would fail on ~1 run in 20 for purely statistical reasons.  (The first
    # version of this check did exactly that and failed on the first try --
    # which is the same uncalibrated-threshold mistake this project has already
    # paid for once.)  What is actually testable is the RATE: over many
    # independent null replications the CI must rarely exclude 0, and the point
    # estimates must stay centred.
    n_rep = 30
    excl_clean = excl_extra = 0
    pts_clean: List[float] = []
    pts_extra: List[float] = []
    for k in range(n_rep):
        dr = d / f"null_{k:02d}"
        _synth_dir(dr, off, extra, null_noise=0.05, seed=100 + k)
        r = build(load_grid(dr, read_manifest(dr / "manifest.csv")),
                  n_boot=1000)
        e = r["gaps"]["dagrs"]
        ac = e["at_clean"]
        pts_clean.append(ac["mean"])
        excl_clean += int(not (ac["ci_lo"] <= 0 <= ac["ci_hi"]))
        for deg in DEGRADATIONS:
            if deg == "clean":
                continue
            x = e[deg]["extra_vs_clean"]
            pts_extra.append(x["mean"])
            excl_extra += int(not x["ci_includes_zero"])
        _best_effort_rmtree(dr)
    rate_clean = excl_clean / n_rep
    rate_extra = excl_extra / (n_rep * 3)
    check(rate_clean <= 0.20,
          f"noisy null: clean-gap CI excluded 0 in {excl_clean}/{n_rep} "
          f"({rate_clean:.1%}; nominal 5%, bound 20%)")
    check(rate_extra <= 0.20,
          f"noisy null: extra-gap CI excluded 0 in {excl_extra}/"
          f"{n_rep * 3} ({rate_extra:.1%}; nominal 5%, bound 20%)")
    bias_clean = abs(float(np.mean(pts_clean)))
    bias_extra = abs(float(np.mean(pts_extra)))
    check(bias_clean < 0.02,
          f"noisy null: clean-gap estimates centred on 0 "
          f"(mean {float(np.mean(pts_clean)):+.4f} over {n_rep} draws)")
    check(bias_extra < 0.02,
          f"noisy null: extra-gap estimates centred on 0 "
          f"(mean {float(np.mean(pts_extra)):+.4f} over {n_rep * 3} draws)")

    # ---- 4. guards must fire ---------------------------------------------
    # (a) a missing arm file must abort, not yield a two-column table
    #     NOTE: built by copying WITHOUT that file, not by deleting it.  A
    #     sandbox safe-delete guard budgets deletions per turn; once spent, every
    #     later delete in this turn is refused (`SAFE_DELETE_BULK_CONFIRM_REQUIRED`,
    #     count 733 > threshold 50).  A self-test that *requires* a delete then
    #     fails inside the full smoke run while passing standalone -- a false
    #     alarm that hides the very signal it was meant to guard.
    d3 = d / "missing_arm"
    _best_effort_rmtree(d3)
    shutil.copytree(d / "planted", d3,
                    ignore=shutil.ignore_patterns("xsrc_sam2video.csv"))
    try:
        load_grid(d3, read_manifest(d3 / "manifest.csv"))
        check(False, "missing arm file: guard did NOT fire")
    except SystemExit:
        check(True, "missing arm file aborts (no silent two-arm table)")

    # (b) a half-written cell must abort
    d4 = d / "half_cell"
    if d4.exists():
        _best_effort_rmtree(d4)
    shutil.copytree(d / "planted", d4)
    src = (d4 / "xsrc_dagrs.csv").read_text(encoding="utf-8").splitlines()
    (d4 / "xsrc_dagrs.csv").write_text("\n".join(src[:-3]) + "\n", encoding="utf-8")
    try:
        load_grid(d4, read_manifest(d4 / "manifest.csv"))
        check(False, "half-written cell: guard did NOT fire")
    except SystemExit:
        check(True, "half-written cell aborts")

    _best_effort_rmtree(d)
    print()
    if fails:
        print(f"VERDICT: FAIL ({len(fails)} of {checks})")
        for f in fails:
            print("  -", f)
        return 1
    print(f"VERDICT: PASS ({checks} checks, 0 failures)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="_scratch/mose_xsrc")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="xsource")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    src = ROOT / args.dir
    man = read_manifest(ROOT / args.manifest)
    print("114 :: cross-source table")
    print("=" * 68)
    print(f"  manifest      : {args.manifest}  "
          f"({len(man)} sequences, {sum(man.values())} instances)")
    grid = load_grid(src, man)
    res = build(grid, n_boot=args.n_boot)
    for p in write_outputs(res, ROOT / args.out_dir, args.prefix):
        print(f"  wrote {p}")

    print()
    print("family gap (sam2video - family), and whether degradation ADDS to it:")
    for fam in FAMILY_ARMS:
        e = res["gaps"][fam]
        print(f"  {fam:<10} clean {e['at_clean']['mean']:+.4f}")
        for deg in DEGRADATIONS:
            if deg == "clean":
                continue
            d = e[deg]
            flag = "0 in CI -> no extra gap" if d["extra_vs_clean"]["ci_includes_zero"] \
                else "CI excludes 0 -> extra gap"
            print(f"    {deg:<14} gap {d['gap']['mean']:+.4f} "
                  f"extra {d['extra_vs_clean']['mean']:+.4f} "
                  f"[{d['extra_vs_clean']['ci_lo']:+.4f}, "
                  f"{d['extra_vs_clean']['ci_hi']:+.4f}]  {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
