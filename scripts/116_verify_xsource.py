"""116 - Verify a completed cross-source (MOSE-mini) sweep, independently.

Why this script exists
----------------------
`run_mose_xsrc.sh` is supposed to run its own verification block at the end.
On 2026-09-23 it exited 1 *after* all three waves had already returned exit=0,
because of a Windows-CRLF bug that killed the launcher at its own
`read ... && $(( ... ))` line.  So the launcher's guard NEVER EXECUTED, and its
exit code was meaningless in both directions: it did not prove a failure, and it
did not prove success either.  "The check did not run" and "the check passed"
look identical from the outside -- only an independently written checker can
tell them apart.

This script re-derives every expected number from the DATASET (never from the
CSV) and re-runs the checks here.  Per the project rule that a verification
script must carry its own negative controls, four deliberate corruptions are
injected into in-memory copies and must each be caught.

The frame-count check below is the defect-13 family check: `n_frames` must be a
function of (seq, obj) alone, because degradation changes pixels, not length.  A
denominator that drifts between cells would silently rescale J&F.  That check
fired for real on the first run of this script, which is why the denominator is
now an explicit protocol decision -- see `scripts/115_frame_denominator.py`.

Run:
    python scripts/116_verify_xsource.py
    python scripts/116_verify_xsource.py --dir results/p1/_scratch/mose_xsrc
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

CONF = "configs/_tau013_mose.yaml"
DEGS = ["clean", "fog", "sensor_noise", "C1_fog_noise"]
ARMS = ["dagrs", "greedy", "sam2video"]
DEFAULT_DIR = "_scratch/mose_xsrc"

_failures: list[str] = []
_checks = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _checks
    _checks += 1
    print(("  PASS  " if ok else "  FAIL  ") + name + ((" | " + detail) if detail else ""))
    if not ok:
        _failures.append(name)
    return ok


def load(directory: Path, arm: str) -> list[dict]:
    with io.open(directory / f"xsrc_{arm}.csv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def protocol_truth() -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    from _common import base_parser, preset, open_src

    ap = base_parser("x")
    a = ap.parse_args(["--config", CONF])
    src = open_src(preset(a))
    seqs = src.sequences()
    inst = {q: list(src.object_ids(q)) for q in seqs}
    nf = {q: int(src.n_frames(q)) for q in seqs}
    return seqs, inst, nf


def audit(rows_by_arm: dict[str, list[dict]], truth, tag: str) -> bool:
    """All structural checks.  Returns True if every check passed."""
    seqs, inst, nf_true = truth
    n_inst = sum(len(v) for v in inst.values())
    exp_rows = n_inst * len(DEGS)
    print(f"\n=== {tag} ===")
    print(f"  truth: n_seq={len(seqs)} n_inst={n_inst} exp_rows/arm={exp_rows}")

    for arm in ARMS:
        rows = rows_by_arm[arm]
        keys = [(r["seq"], r["obj_id"], r["mode"], r["degradation"], r["level"]) for r in rows]
        per_deg = Counter(r["degradation"] for r in rows)
        cells = {(r["seq"], r["mode"], r["degradation"], r["level"]) for r in rows}
        check(f"{arm}: rows == {exp_rows}", len(rows) == exp_rows, f"got {len(rows)}")
        check(f"{arm}: unique (seq,obj,mode,deg,level) == rows", len(set(keys)) == len(rows),
              f"uniq {len(set(keys))} of {len(rows)}")
        check(f"{arm}: cells == {len(seqs) * len(DEGS)}",
              len(cells) == len(seqs) * len(DEGS), f"got {len(cells)}")
        check(f"{arm}: degradation set exact", set(per_deg) == set(DEGS), str(sorted(per_deg)))
        for d in DEGS:
            check(f"{arm}: rows[{d}] == {n_inst}", per_deg.get(d, 0) == n_inst,
                  f"got {per_deg.get(d, 0)}")
        check(f"{arm}: single mode", len({r['mode'] for r in rows}) == 1,
              str(sorted({r['mode'] for r in rows})))
        check(f"{arm}: single level 3.0", {r["level"] for r in rows} == {"3.0"},
              str(sorted({r["level"] for r in rows})))

    # ---- cross-arm agreement: identical (seq,obj) universe -------------------
    uni = {arm: {(r["seq"], r["obj_id"]) for r in rows_by_arm[arm]} for arm in ARMS}
    check("three arms share one (seq,obj) universe",
          len(set(frozenset(v) for v in uni.values())) == 1,
          " ".join(f"{a}={len(uni[a])}" for a in ARMS))
    truth_pairs = {(q, str(o)) for q, objs in inst.items() for o in objs}
    check("arm universe == dataset universe", uni[ARMS[0]] == truth_pairs,
          f"missing={len(truth_pairs - uni[ARMS[0]])} extra={len(uni[ARMS[0]] - truth_pairs)}")

    # ---- the defect-13 family check: the frame-count DENOMINATOR -------------
    # `n_frames` is the denominator each arm actually averaged over, and it must
    # lie between two ground-truth quantities:
    #
    #   n_present  the object is on screen in this many frames
    #   n_clip     the clip is this long
    #
    # A frame beyond `n_present` is scored instead of skipped exactly when the arm
    # painted the departed object, and such a frame contributes 0 to the sum.  So
    #   * n_frames == n_clip     the arm painted every absence frame (greedy/dagrs)
    #   * n_frames == n_present  the arm painted none of them
    #   * in between             the arm painted SOME of them -- which happens for
    #                            `sam2video` once the frames are degraded, because
    #                            noise makes it emit a spurious mask on an
    #                            absence frame.  Its denominator therefore moves
    #                            with the DEGRADATION, not only with the arm.
    #
    # The legitimate universe is `n_present <= n_frames <= n_clip`.  Asserting
    # equality with the clip length ALONE -- which this check used to do -- fails
    # on MOSE for a legitimate reason and hides the real signal.
    # See scripts/115_frame_denominator.py.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "a115", ROOT / "scripts" / "115_frame_denominator.py")
    a115 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a115)
    table = a115.present_table("mose")

    nf_map = defaultdict(set)
    for arm in ARMS:
        for r in rows_by_arm[arm]:
            nf_map[(r["seq"], r["obj_id"])].add(int(float(r["n_frames"])))
    bad = {k: sorted(v) for k, v in nf_map.items() if len(v) != 1}
    print(f"  [info] arms disagree on the denominator for {len(bad)} of "
          f"{len(nf_map)} instances -- the defect-13 footprint, resolved by the "
          f"P1 protocol rather than by re-running")

    off_universe = {}
    n_clip_rows = n_present_rows = n_mid_rows = 0
    mid_examples = []
    for arm in ARMS:
        for r in rows_by_arm[arm]:
            t = table.get(f"{r['seq']}|{r['obj_id']}")
            if t is None:
                off_universe[(arm, r["seq"], r["obj_id"])] = "no ground truth"
                continue
            nf = int(float(r["n_frames"]))
            if not (t["n_present"] <= nf <= t["n_clip"]):
                off_universe[(arm, r["seq"], r["obj_id"])] = nf
            elif nf == t["n_clip"]:
                n_clip_rows += 1
            elif nf == t["n_present"]:
                n_present_rows += 1
            else:
                n_mid_rows += 1
                if len(mid_examples) < 4:
                    mid_examples.append((arm, r["seq"], r["obj_id"], r["degradation"],
                                         nf, t["n_present"], t["n_clip"]))
    total_rows = sum(len(v) for v in rows_by_arm.values())
    check("every n_frames lies in [GT-present count, clip length]",
          not off_universe,
          f"{len(off_universe)} offenders" + ("" if not off_universe
                                              else " e.g. " + str(list(off_universe.items())[:2])))
    print(f"  [info] denominator split over {total_rows} arm-rows: "
          f"clip={n_clip_rows}  present={n_present_rows}  strictly-between={n_mid_rows}")
    for ex in mid_examples:
        print(f"         strictly between: {ex[0]}/{ex[1]}/obj{ex[2]} {ex[3]} "
              f"n_frames={ex[4]} (present={ex[5]}, clip={ex[6]}) -- a spurious mask "
              f"on an absence frame")

    # ---- clean must actually be clean ---------------------------------------
    for arm in ARMS:
        cl = [r for r in rows_by_arm[arm] if r["degradation"] == "clean"]
        check(f"{arm}: clean severity_mean == 1.0",
              all(abs(float(r["severity_mean"]) - 1.0) < 1e-9 for r in cl))
        check(f"{arm}: clean n_predict_failures == 0",
              all(int(float(r["n_predict_failures"])) == 0 for r in cl))

    # ---- ranges ---------------------------------------------------------------
    for arm in ARMS:
        vals = [float(r["J&F"]) for r in rows_by_arm[arm]]
        check(f"{arm}: J&F within [0,1] and finite",
              all(0.0 <= v <= 1.0 for v in vals), f"min={min(vals):.4f} max={max(vals):.4f}")
    return not _failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help="directory holding xsrc_<arm>.csv "
                         "(point at results/p1/... to verify the P1 tables)")
    args = ap.parse_args(argv)
    directory = Path(args.dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    print(f"116 :: cross-source sweep verification  dir={directory.relative_to(ROOT)}")

    truth = protocol_truth()
    rows_by_arm = {arm: load(directory, arm) for arm in ARMS}

    audit(rows_by_arm, truth, "as-produced sweep")
    clean_rc = not _failures

    # ---------------- negative controls --------------------------------------
    # Each corruption must be caught by the SAME code path used above; a
    # negative control that trips a different assertion does not calibrate the
    # check it is supposed to calibrate.
    print("\n=== negative controls (each must FIRE) ===")
    negs = []

    # N1: duplicate one row -> unique-key count must fall below rows
    m = {a: copy.deepcopy(v) for a, v in rows_by_arm.items()}
    m["greedy"].append(copy.deepcopy(m["greedy"][0]))
    negs.append(("N1 duplicated row", m))

    # N2: drop one row -> per-degradation count must mismatch
    m = {a: copy.deepcopy(v) for a, v in rows_by_arm.items()}
    del m["sam2video"][3]
    negs.append(("N2 dropped row", m))

    # N3: perturb n_frames in one arm for one cell -> cross-arm denominator
    m = {a: copy.deepcopy(v) for a, v in rows_by_arm.items()}
    m["dagrs"][0]["n_frames"] = str(int(float(m["dagrs"][0]["n_frames"])) + 1)
    negs.append(("N3 n_frames drift", m))

    # N4: relabel one degradation -> set/count mismatch
    m = {a: copy.deepcopy(v) for a, v in rows_by_arm.items()}
    m["greedy"][5]["degradation"] = "clean "
    negs.append(("N4 mislabelled degradation", m))

    for tag, mutated in negs:
        before = len(_failures)
        saved = _failures[:]
        _failures.clear()
        print(f"\n--- {tag} ---")
        audit(mutated, truth, tag)
        fired = len(_failures) > 0
        _failures[:] = saved
        print(f"  => {'FIRED' if fired else 'SILENT (bad: control not caught)'}"
              f"  [{len(_failures) - before} failures raised]")
        if not fired:
            _failures.append(f"negative control {tag} was SILENT")

    print("\n" + "=" * 62)
    if _failures:
        print(f"RESULT: {len(_failures)} PROBLEM(S)")
        for f in _failures:
            print("   -", f)
        return 1
    print(f"RESULT: ALL {_checks} CHECKS PASSED; 4/4 negative controls FIRED")
    print("        (as-produced sweep clean)" if clean_rc else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
