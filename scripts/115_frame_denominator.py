"""115 - Make the evaluation denominator a PROTOCOL, not a side effect of the model.

THE DEFECT THIS CLOSES
----------------------
`xdrp/metrics.py::seq_metrics` skips a frame only when the ground truth AND the
prediction are both empty.  Nothing in that rule mentions the arm, but the arm
decides the outcome, because arms disagree about what to draw once the target
leaves the scene:

    sam2video   emits an EMPTY mask once the object is gone
                -> both empty -> frame skipped -> n_frames == GT-present count
    greedy      keep painting the object
    dagrs       -> empty GT with a non-empty prediction -> frame scored
                   (J = F = 0) -> n_frames == CLIP LENGTH

So on any instance whose target leaves the frame, the three arms average over
DIFFERENT frame sets, and two means over different frames may not be subtracted
(`docs/BASELINE_DIAGNOSIS.md` 13.4.3 states this rule; the rule was violated by
the code that produced the very tables it guards).

Ground truth for how often this happens, counted from the annotations:

    MOSE   22 of 58 instances   (38%, spanning 15 of 24 sequences)
    DAVIS   4 of 61 instances   (india 1/2/3, kite-surf 2)

WHY THIS IS NOT "sam2video DROPS FRAMES"
----------------------------------------
`xdrp/video_backend.py::run_video_tracking` raises on a genuinely missing mask
(lines 471-477), and the 2026-09-23 MOSE wave B exited 0 -- nothing was dropped.
The deviation is the skip rule, so the fix is a protocol decision, not a re-run.

THE TWO PROTOCOLS
-----------------
P0  status quo.  Denominator = frames NOT (GT empty AND prediction empty).
    Arm-dependent  ->  invalid for any cross-arm comparison.

P1  **primary** (this project's choice, 2026-09-23).  Denominator = frames where
    the GT object is present.  Arm-independent, needs no arbitrary score for a
    correctly-empty prediction, and it is the CONSERVATIVE direction: it moves
    the headline family gap DOWN, against this paper's own claim.

P2  full clip length, with a convention for a correctly-empty prediction.
    Rejected: it needs an arbitrary score (1.0 vs 0 swings the result hard) and
    it moves the headline gap UP, i.e. in the direction that flatters our own
    thesis.  Recorded here so the choice is auditable, not implied.

THE EXACT CORRECTION (why adopting P1 costs zero GPU)
-----------------------------------------------------
A frame outside the GT-present set always scores exactly 0: J = 0 because the
intersection is empty and the union is not, F = 0 because the GT boundary is
empty and the predicted boundary is not.  A GT-present frame is never skipped
(skipping requires an empty GT).  Hence, for every arm,

    raw  =  (sum over GT-present frames) / n_frames

so the P1 value is

    corrected  =  raw * n_frames / n_present          (exact, no re-run)

`calibrate_formula()` proves that identity on synthetic masks whose per-frame J
is known, and refuses to let the correction run if it does not hold.

Run:
    python scripts/115_frame_denominator.py --self-test
    python scripts/115_frame_denominator.py --source davis --emit
    python scripts/115_frame_denominator.py --source mose  --emit
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

#: (source, path relative to ROOT).  Outputs MIRROR this relative path under
#: `--emit-dir`, so a consumer that hard-codes `results/cl4_full_v2.csv` only has
#: to be pointed at a different root instead of being patched.
REGISTRY = [
    ("mose",  "_scratch/mose_xsrc/xsrc_dagrs.csv"),
    ("mose",  "_scratch/mose_xsrc/xsrc_greedy.csv"),
    ("mose",  "_scratch/mose_xsrc/xsrc_sam2video.csv"),
    ("davis", "results/cl4_full_v2.csv"),
    ("davis", "results/singles_dagrs.csv"),
    ("davis", "results/singles_greedy.csv"),
    ("davis", "results/singles_sam2video.csv"),
    ("davis", "results/c1c4_a_base.csv"),
    ("davis", "results/c1c4_b_base.csv"),
    ("davis", "results/c1c4_a_dagrs.csv"),
    ("davis", "results/c1c4_b_dagrs.csv"),
    ("davis", "_scratch/severity_20260921/severity_dagrs.csv"),
    ("davis", "_scratch/severity_20260921/severity_greedy.csv"),
    ("davis", "_scratch/severity_20260921/severity_sam2video.csv"),
]

MOSE_ANN = ROOT / "data" / "MOSE_mini" / "Annotations"

#: appended to every corrected row, so P0 and P1 stay distinguishable downstream
P1_COLUMNS = ["n_present", "n_clip", "p1_scale", "protocol", "J_raw", "F_raw", "J&F_raw"]

_fail: list[str] = []
_checks = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _checks
    _checks += 1
    print(("  PASS  " if ok else "  FAIL  ") + name + ((" | " + detail) if detail else ""))
    if not ok:
        _fail.append(name)
    return ok


# --------------------------------------------------------------------------- #
# the exact correction, and its calibration
# --------------------------------------------------------------------------- #
def p1_scale(n_frames: int, n_present: int) -> float:
    """The factor that turns a P0 mean into the P1 mean.  Guards, not hope."""
    if n_present <= 0:
        raise RuntimeError(
            f"n_present={n_present}: the object never appears, so the P1 "
            "denominator is empty and the ratio is undefined")
    if n_frames < n_present:
        raise RuntimeError(
            f"n_frames={n_frames} < n_present={n_present}: a GT-present frame was "
            "lost, so the exact correction is invalid")
    return n_frames / float(n_present)


def _jf(pred: np.ndarray, gt: np.ndarray) -> float:
    from xdrp.metrics import jf
    _j, _f, m = jf(pred, gt)
    return float(m)


def calibrate_formula(verbose: bool = True) -> tuple[bool, str]:
    """Prove `corrected = raw * T / P` on synthetic masks with a known truth.

    Two regimes, because the two arm families behave differently:
      (a) predict-everywhere      -> every frame is scored, extras score 0
      (b) predict-empty-when-gone -> extras are skipped, n_frames == P (identity)

    Both are checked against the ground truth computed frame by frame, which is
    the only way to know the closed form is the right one.
    """
    from xdrp.metrics import seq_metrics

    rng = np.random.default_rng(0)
    T, P = 40, 17
    H = W = 32
    present_at = np.sort(rng.choice(T, size=P, replace=False))
    ok = True
    detail = ""
    for tag, fill_extras in (("predict-everywhere", True),
                             ("predict-empty-when-gone", False)):
        gts, preds = [], []
        for t in range(T):
            g = np.zeros((H, W), bool)
            p = np.zeros((H, W), bool)
            if t in present_at:
                g[4:20, 4:20] = True                      # a fixed square GT
                p[4:20, 4:20] = True
                p[:8, :8] = True                          # deliberate error
            elif fill_extras:
                p[10:26, 10:26] = True                    # paints the absent object
            gts.append(g)
            preds.append(p)
        m = seq_metrics(preds, gts)
        truth = float(np.mean([_jf(preds[t], gts[t]) for t in present_at]))
        raw = float(m["J&F"])
        corr = raw * p1_scale(int(m["n_frames"]), P)
        d = abs(corr - truth)
        expect_n = T if fill_extras else P
        good = (int(m["n_frames"]) == expect_n) and d < 1e-12
        ok &= bool(good)
        # Negative control: a WRONG denominator must NOT reproduce the truth.
        # It uses P-1, not P+1, on purpose: in regime (b) n_frames == P, so P+1
        # would trip the `n_frames < n_present` guard instead of exercising the
        # comparison -- the control has to sit on the same code path it audits.
        wrong = raw * p1_scale(int(m["n_frames"]), P - 1)
        ctrl_ok = abs(wrong - truth) > 1e-3
        ok &= bool(ctrl_ok)
        if verbose:
            print(f"  [{tag}] n_frames={m['n_frames']} (expect {expect_n})  "
                  f"raw={raw:.6f} corrected={corr:.6f} truth={truth:.6f} |d|={d:.2e}"
                  f"   [wrong-denominator control |d|={abs(wrong - truth):.2e}]")
        detail = f"|d|={d:.2e}"
        if not (good and ctrl_ok):
            return False, detail
    return bool(ok), detail


# --------------------------------------------------------------------------- #
# ground truth: how many frames is each (sequence, object) actually present?
# --------------------------------------------------------------------------- #
def present_from_davis() -> dict[str, dict[str, int]]:
    from _common import base_parser, preset, open_src

    a = base_parser("x").parse_args(["--config", "configs/_tau013.yaml"])
    src = open_src(preset(a))
    out: dict[str, dict[str, int]] = {}
    for q in src.sequences():
        cnt: dict[str, int] = {}
        n = src.n_frames(q)
        for t in range(n):
            lab = src.label(q, t)
            for i in np.unique(lab):
                if i:
                    cnt[str(int(i))] = cnt.get(str(int(i)), 0) + 1
        for o in src.object_ids(q):
            cnt.setdefault(str(o), 0)
        out[q] = cnt
    return out


def present_from_mose() -> dict[str, dict[str, int]]:
    if not MOSE_ANN.is_dir():
        raise SystemExit(f"missing {MOSE_ANN}: rebuild with "
                         "scripts/112_prepare_mose_mini.py --convert")
    out: dict[str, dict[str, int]] = {}
    for q in sorted(os.listdir(MOSE_ANN)):
        cnt: dict[str, int] = {}
        for f in sorted(glob.glob(str(MOSE_ANN / q / "*.png"))):
            from PIL import Image
            arr = np.asarray(Image.open(f))
            for i in np.unique(arr):
                if i:
                    cnt[str(int(i))] = cnt.get(str(int(i)), 0) + 1
        out[q] = cnt
    return out


def clip_lengths(source: str) -> dict[str, int]:
    from _common import base_parser, preset, open_src

    cfg = "configs/_tau013_mose.yaml" if source == "mose" else "configs/_tau013.yaml"
    a = base_parser("x").parse_args(["--config", cfg])
    src = open_src(preset(a))
    return {q: int(src.n_frames(q)) for q in src.sequences()}


def present_table(source: str, refresh: bool = False) -> dict:
    """{ (seq, obj): {'n_clip': T, 'n_present': P} } with a disk cache."""
    cache = ROOT / "results" / f"frame_present_counts_{source}.json"
    if cache.exists() and not refresh:
        return json.load(io.open(cache, encoding="utf-8"))
    present = present_from_mose() if source == "mose" else present_from_davis()
    clips = clip_lengths(source)
    out: dict[str, dict[str, int]] = {}
    for q, cnt in present.items():
        T = clips.get(q)
        if T is None:
            raise RuntimeError(f"{q}: present counts exist but the dataset has no "
                               "clip length -- refusing to guess a denominator")
        for o, P in cnt.items():
            out[f"{q}|{o}"] = {"n_clip": int(T), "n_present": int(P)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with io.open(cache, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return out


# --------------------------------------------------------------------------- #
# correction
# --------------------------------------------------------------------------- #
def correct_rows(rows: list[dict], table: dict, where: str) -> list[dict]:
    """Copy of `rows` whose J / F / J&F are P1 values.  Never drops a row."""
    if not rows:
        raise RuntimeError(f"{where}: zero rows to correct -- a guard that compares "
                           "nothing must not report success")
    n_bad_key = n_raised = 0
    out: list[dict] = []
    for r in rows:
        key = f"{r['seq']}|{r['obj_id']}"
        t = table.get(key)
        if t is None:
            n_bad_key += 1
            continue
        try:
            s = p1_scale(int(float(r["n_frames"])), int(t["n_present"]))
        except RuntimeError:
            n_raised += 1
            continue
        d = dict(r)
        if s != 1.0:
            # Only rows whose arm scored frames beyond the GT-present set are
            # rewritten.  Re-formatting the others would be a no-op that still
            # costs decimal digits -- and would make "how many rows moved?" an
            # unanswerable question.
            for col in ("J", "F", "J&F"):
                if col in r:
                    d[f"{col}_raw"] = r[col]
                    d[col] = f"{float(r[col]) * s:.10g}"
        else:
            for col in ("J", "F", "J&F"):
                if col in r:
                    d[f"{col}_raw"] = r[col]
        d["n_present"] = int(t["n_present"])
        d["n_clip"] = int(t["n_clip"])
        d["p1_scale"] = f"{s:.10g}"
        d["protocol"] = "P1"
        out.append(d)
    if n_bad_key or n_raised:
        raise RuntimeError(f"{where}: {n_bad_key} rows had no ground-truth present "
                           f"count and {n_raised} rows violated n_frames >= n_present"
                           " -- refusing to emit a partially corrected table")
    if len(out) != len(rows):
        raise RuntimeError(f"{where}: {len(rows)} rows in, {len(out)} out")
    return out


def emit(source: str, emit_dir: Path, refresh: bool, which: list[str] | None) -> int:
    table = present_table(source, refresh=refresh)
    todo = [e for e in REGISTRY
            if e[0] == source and (not which or Path(e[1]).stem in which)]
    if not todo:
        print(f"  [115] nothing to emit for source={source} (filter={which})")
        return 1
    n_touched = 0
    for src_rel, rel in todo:
        p = ROOT / rel
        if not p.exists():
            print(f"  [skip] {rel} (absent)")
            continue
        rows = list(csv.DictReader(io.open(p, encoding="utf-8", newline="")))
        fixed = correct_rows(rows, table, rel)
        dst = emit_dir / rel                      # mirrors the relative path
        dst.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys()) + [c for c in P1_COLUMNS if c not in rows[0]]
        with io.open(dst, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(fixed)
        n_moved = sum(1 for a, b in zip(rows, fixed)
                      if float(a["J&F"]) != float(b["J&F"]))
        print(f"  [ok] {rel:<52} rows={len(rows):<5} changed={n_moved:<5} -> {dst.relative_to(ROOT)}")
        n_touched += 1
    if n_touched == 0:
        raise RuntimeError(f"source={source}: no registered file could be emitted")
    return 0


# --------------------------------------------------------------------------- #
# audit (no writes): how far does the defect reach?
# --------------------------------------------------------------------------- #
def audit() -> dict:
    print("\n" + "=" * 74)
    print("DEFECT FOOTPRINT (n_frames vs GT-present count)")
    print("=" * 74)
    report = {}
    for source in ("mose", "davis"):
        table = present_table(source)
        aff = sum(1 for v in table.values() if v["n_present"] < v["n_clip"])
        print(f"  {source:<6} instances={len(table):<4} with an absent-object frame={aff}")
        report[source] = {"instances": len(table), "affected": aff}
        for _src, rel in [e for e in REGISTRY if e[0] == source]:
            p = ROOT / rel
            if not p.exists():
                continue
            rows = list(csv.DictReader(io.open(p, encoding="utf-8", newline="")))
            over = under = 0
            seqs: set[str] = set()
            for r in rows:
                t = table.get(f"{r['seq']}|{r['obj_id']}")
                if t is None:
                    continue
                nf = int(float(r["n_frames"]))
                if nf > t["n_present"]:
                    over += 1
                    seqs.add(r["seq"])
                elif nf < t["n_present"]:
                    under += 1
            print(f"    {rel:<50} rows={len(rows):<5} n_frames>present={over:<5} "
                  f"({over / len(rows):5.1%})  n_frames<present={under}"
                  + (f"  seqs={sorted(seqs)}" if 0 < len(seqs) <= 4 else ""))
            report.setdefault("files", []).append(
                {"file": rel, "rows": len(rows), "over": over, "under": under,
                 "sequences": sorted(seqs)})
    n_under = sum(f["under"] for f in report.get("files", []))
    check("no row has n_frames < GT-present count (the exact correction is legal "
          "everywhere it is applied)", n_under == 0, f"{n_under} violations")
    check("the audit compared at least one file", len(report.get("files", [])) > 0,
          f"{len(report.get('files', []))} files")
    out = ROOT / "results" / "frame_denominator_audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  -> {out.relative_to(ROOT)}")
    return report


# --------------------------------------------------------------------------- #
# summary: P0 vs P1, so the manuscript never quotes the two interchangeably
# --------------------------------------------------------------------------- #
def _load05():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "a05_for_115", ROOT / "scripts" / "05_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seq_equal_means(rows: list[dict], a05) -> dict[str, dict[str, float]]:
    """{degradation: {mode: mean J&F}} with instances collapsed per sequence."""
    out: dict[str, dict[str, float]] = {}
    if not rows:
        return out
    per: dict[tuple[str, str], dict[str, float]] = {}
    for r in a05.collapse_objects(a05.sequence_level(rows)):
        per.setdefault((str(r["degradation"]), str(r["mode"])), {})[str(r["seq"])] = \
            float(r["J&F"])
    for (deg, mode), vals in per.items():
        out.setdefault(deg, {})[mode] = float(np.mean(list(vals.values())))
    return out


def summary(emit_dir: Path) -> int:
    a05 = _load05()
    print("=" * 88)
    print("P0 (status quo) vs P1 (primary) -- sequence-equal mean J&F")
    print("=" * 88)
    md = ["# Protocol comparison: P0 (status quo) vs P1 (primary)", "",
          "Generated by `scripts/115_frame_denominator.py --summary`. "
          "P1 = denominator restricted to the frames where the GT object is "
          "present (arm-independent). P0 = the `xdrp/metrics.py` skip rule "
          "(arm-dependent).", ""]
    n_cmp = 0
    for _src, rel in REGISTRY:
        p0_path = ROOT / rel
        p1_path = emit_dir / rel
        if not (p0_path.exists() and p1_path.exists()):
            continue
        p0 = list(csv.DictReader(io.open(p0_path, encoding="utf-8", newline="")))
        p1 = list(csv.DictReader(io.open(p1_path, encoding="utf-8", newline="")))
        m0, m1 = seq_equal_means(p0, a05), seq_equal_means(p1, a05)
        if not m0:
            continue
        print(f"\n{rel}")
        print(f"  {'degradation':<18}{'mode':<14}{'P0':>9}{'P1':>9}{'delta':>10}")
        md += [f"## `{rel}`", "", "| degradation | mode | P0 | P1 | P1-P0 |",
               "|---|---|---|---|---|"]
        for deg in sorted(m0):
            for mode in sorted(m0[deg]):
                a, b = m0[deg][mode], m1.get(deg, {}).get(mode, float("nan"))
                print(f"  {deg:<18}{mode:<14}{a:>9.4f}{b:>9.4f}{b - a:>+10.4f}")
                md.append(f"| {deg} | {mode} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
                n_cmp += 1
        md.append("")
    if n_cmp == 0:
        raise RuntimeError("summary compared nothing -- refusing to write a "
                           "report that would read as 'no differences'")
    check("the summary compared at least one (degradation, mode) pair",
          n_cmp > 0, f"{n_cmp} pairs")

    # the headline: the family gap on the four-degradation main table
    comp = ROOT / "results" / "cl4_full_v2.csv"
    comp_p1 = emit_dir / "results" / "cl4_full_v2.csv"
    if comp.exists() and comp_p1.exists():
        a05_ = a05
        m0 = seq_equal_means(list(csv.DictReader(io.open(comp, encoding="utf-8", newline=""))), a05_)
        m1 = seq_equal_means(list(csv.DictReader(io.open(comp_p1, encoding="utf-8", newline=""))), a05_)
        degs0 = ", ".join(sorted(m0))
        print("\n" + "-" * 88)
        print(f"HEADLINE family gap (sam2video - family), cl4_full_v2  [{degs0}]")
        md += ["## Headline family gap (`sam2video` minus the family arm)", "",
               f"Source file `results/cl4_full_v2.csv`; degradations: {degs0}.", "",
               "| reference arm | degradation | P0 gap | P1 gap | change |",
               "|---|---|---|---|---|"]
        for fam in ("dagrs", "greedy"):
            g0 = g1 = 0.0
            for deg in sorted(m0):
                if fam not in m0[deg] or "sam2video" not in m0[deg]:
                    continue
                d0 = m0[deg]["sam2video"] - m0[deg][fam]
                d1 = m1[deg]["sam2video"] - m1[deg][fam]
                g0 += d0 / len(m0)
                g1 += d1 / len(m1)
                print(f"  vs {fam:<8} {deg:<20} P0 {d0:+.4f}  P1 {d1:+.4f}  "
                      f"({d1 - d0:+.4f})")
                md.append(f"| {fam} | {deg} | {d0:+.4f} | {d1:+.4f} | {d1 - d0:+.4f} |")
            print(f"  vs {fam:<8} {'MEAN over degradations':<20} P0 {g0:+.4f}  "
                  f"P1 {g1:+.4f}  ({g1 - g0:+.4f})")
            md.append(f"| {fam} | **mean** | **{g0:+.4f}** | **{g1:+.4f}** | "
                      f"**{g1 - g0:+.4f}** |")
        md.append("")

    out = emit_dir / "protocol_p1_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  -> {out.relative_to(ROOT)}")
    return 1 if _fail else 0


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def self_test() -> int:
    print("=" * 74)
    print("115 self-test: the P1 correction cannot pass on wrong input")
    print("=" * 74)
    print("\nCALIBRATION: closed form == ground truth on synthetic masks")
    ok, detail = calibrate_formula()
    check("corrected = raw * n_frames / n_present reproduces the GT-present mean "
          "(and a wrong denominator does not)", ok, detail)

    print("\nNEGATIVE CONTROLS on the guards")
    raised = False
    try:
        p1_scale(n_frames=10, n_present=0)
    except RuntimeError:
        raised = True
    check("an empty P1 denominator raises instead of dividing by zero", raised)

    raised = False
    try:
        p1_scale(n_frames=9, n_present=10)
    except RuntimeError:
        raised = True
    check("n_frames < n_present raises (a lost present frame invalidates the "
          "correction)", raised)

    raised = False
    try:
        correct_rows([], {}, "empty")
    except RuntimeError:
        raised = True
    check("an EMPTY input raises instead of reporting 'zero differences'", raised)

    raised = False
    try:
        correct_rows([{"seq": "s", "obj_id": "1", "n_frames": "50", "J&F": "0.5"}],
                     {}, "missing-truth")
    except RuntimeError:
        raised = True
    check("a row with no ground-truth present count raises (no silent drop)", raised)

    # a planted inconsistency must be visible in the corrected value
    rows = [{"seq": "s", "obj_id": "1", "n_frames": "100", "J&F": "0.25"}]
    fixed = correct_rows(rows, {"s|1": {"n_clip": 100, "n_present": 50}}, "planted")
    good = abs(float(fixed[0]["J&F"]) - 0.5) < 1e-12 and fixed[0]["protocol"] == "P1"
    check("P0 0.25 over 100 frames becomes P1 0.50 over 50 present frames",
          good, f"got {fixed[0]['J&F']}")

    return 1 if _fail else 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["davis", "mose", "both"], default="both")
    ap.add_argument("--emit", action="store_true",
                    help="write P1-corrected copies under --emit-dir")
    ap.add_argument("--emit-dir", default="results/p1")
    ap.add_argument("--files", default="",
                    help="comma-separated stems to restrict --emit to")
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the ground-truth present counts (ignore cache)")
    ap.add_argument("--audit", action="store_true", help="print the footprint report")
    ap.add_argument("--summary", action="store_true",
                    help="compare P0 against the emitted P1 tables and write "
                         "results/p1/protocol_p1_summary.md")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    sources = ["mose", "davis"] if args.source == "both" else [args.source]
    which = [s.strip() for s in args.files.split(",") if s.strip()] or None
    emit_dir = Path(args.emit_dir)
    if not emit_dir.is_absolute():
        emit_dir = ROOT / emit_dir

    if args.audit:
        audit()
    if args.emit:
        for src in sources:
            print(f"\n=== emitting P1 tables for source={src} -> "
                  f"{emit_dir.relative_to(ROOT)} ===")
            emit(src, emit_dir, args.refresh, which)
    if args.summary:
        summary(emit_dir)
    if not (args.audit or args.emit or args.summary):
        ap.print_help()
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
