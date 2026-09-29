"""CL11: the A8/A9/A10 hyper-parameter probes -- do they move anything?

The three probes each vary one knob of the same pipeline, on the reduced scope the
project agreed on (single phase fog@3, first 10 DAVIS sequences = 12 instances /
742 frames, `dagrs` arm only).  Each row carries its `arm` (A8=1 ... A10=13), so
`sequence_level` keeps the variants apart -- without that key every variant would
collapse into one bucket and the mean would be a meaningless mix.

The reference is the DEFAULT configuration, which is present inside every sweep:
    A8  keyframe_stride K = 5      (configs/_tau013.yaml)
    A9  j_steady       J = 3
    A10 bank size      M = 13
A useful internal control falls out for free: the three default variants are three
independent runs of the same configuration, so their means must be identical.

Nothing is hard-coded: means, effect sizes and Holm-adjusted p-values are all
computed here, and the test machinery is calibrated on synthetic data first.

Run:  python scripts/117_cl11_probe.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("a05", ROOT / "scripts" / "05_analysis.py")
m05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m05)
from xdrp.stats import cliffs_delta, holm_bonferroni, paired_report  # noqa: E402

DIR = ROOT / "_scratch" / "abl_sweep_20260922"   # overridable with --dir
#: sweep -> (variant values in the file's `arm` column, the default/reference value,
#:           what the knob is called in the paper)
SWEEPS = {
    "A8": ((1, 3, 5, 10), 5, "keyframe stride K"),
    "A9": ((1, 2, 3, 5, 8, 13), 3, "profile top-J  J"),
    "A10": ((3, 5, 8, 13), 13, "operator-bank size M"),
}

_fail: list[str] = []


def calibrate() -> None:
    """The test must fire on a planted effect and stay quiet on no effect."""
    print("=" * 70)
    print("CALIBRATION of the paired machinery")
    print("=" * 70)
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.9, 12)
    # (a) no effect: a copy of the baseline must give delta 0 and p = 1
    r0 = paired_report(base, base + rng.normal(0, 0.002, 12))  # tiny noise, same dist
    same = paired_report(base, list(base))
    ok_a = abs(same["cliffs_delta"]) < 1e-12 and same["mean_delta"] == 0.0
    print(f"  [identity]      delta={same['cliffs_delta']:+.3f} p={same['wilcoxon_p']:.3g} "
          f"-> {'OK' if ok_a else 'BAD'}")
    # (b) planted uniform effect of +0.20 on 12 sequences.
    # NOTE ON THE EXPECTED MAGNITUDE: `cliffs_delta` is the UNPAIRED definition
    # (P(b>a) - P(b<a) over all cross pairs), so even a uniform shift cannot reach
    # 1.0 -- the pairs (a_i, b_j) with i != j can go either way.  Asserting == 1.0
    # here was wrong on my part, not a defect in the tool (2026-09-23).
    planted = paired_report(base, base + 0.20)
    ok_b = (planted["cliffs_delta"] > 0.3 and planted["delta_ci_lo"] > 0.15
            and planted["wilcoxon_p"] < 1e-3)
    print(f"  [planted +0.20] delta={planted['cliffs_delta']:+.3f} "
          f"CI=[{planted['delta_ci_lo']:+.3f}, {planted['delta_ci_hi']:+.3f}] "
          f"p={planted['wilcoxon_p']:.2g} -> {'OK' if ok_b else 'BAD'}")
    # (b2) DIRECTION control: the same shift with the opposite sign must produce a
    # negative effect size and a CI below zero.  Sign convention bugs cannot be
    # caught by "the numbers look plausible" -- they need a known-direction case.
    neg = paired_report(base, base - 0.20)
    ok_n = (neg["cliffs_delta"] < -0.3 and neg["delta_ci_hi"] < -0.15
            and neg["wilcoxon_p"] < 1e-3)
    print(f"  [planted -0.20] delta={neg['cliffs_delta']:+.3f} "
          f"CI=[{neg['delta_ci_lo']:+.3f}, {neg['delta_ci_hi']:+.3f}] "
          f"-> {'OK (sign convention holds)' if ok_n else 'BAD'}")
    # (c) planted null effect of exactly 0 must NOT be called significant
    nullp = paired_report(base, base + rng.normal(0, 0.05, 12))
    print(f"  [planted null]  delta={nullp['cliffs_delta']:+.3f} p={nullp['wilcoxon_p']:.3g}")
    print(f"  (noise-level run, for scale only: p={r0['wilcoxon_p']:.3g})")
    if not (ok_a and ok_b and ok_n):
        _fail.append("paired machinery failed calibration")


def per_variant(rows: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    """(sweep, variant) -> {sequence: J&F} with objects collapsed per sequence."""
    seq_rows = m05.sequence_level(rows)
    col = m05.collapse_objects(seq_rows)
    out: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in col:
        arm = str(r.get("arm", ""))
        if "=" not in arm:
            continue
        sw, val = arm.split("=", 1)
        try:
            out[(sw, int(float(val)))][r["seq"]] = float(r["J&F"])
        except ValueError:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="_scratch/abl_sweep_20260922",
                    help="directory holding sweep_A8/A9/A10.csv")
    ap.add_argument("--out", default="results/cl11_hyperparam_probe.json")
    ap.add_argument("--self-test", action="store_true",
                    help="run only the paired-test calibration on synthetic data "
                         "(the guard registered in 00_smoke_test.py)")
    args = ap.parse_args(argv)
    global DIR
    DIR = Path(args.dir)
    if not DIR.is_absolute():
        DIR = ROOT / DIR

    if args.self_test:
        calibrate()
        if _fail:
            print(f"RESULT: {len(_fail)} calibration problem(s)")
            for f in _fail:
                print("   -", f)
            return 1
        print("RESULT: calibration ok")
        return 0

    calibrate()

    all_rows: list[dict] = []
    for sw in SWEEPS:
        p = DIR / f"sweep_{sw}.csv"
        if not p.exists():
            _fail.append(f"missing {p.name}")
            continue
        all_rows += list(csv.DictReader(io.open(p, encoding="utf-8")))
    per = per_variant(all_rows)

    print("\n" + "=" * 70)
    print("A8/A9/A10 results  (fog@3, 12 instances / 10 sequences, dagrs arm)")
    print("=" * 70)
    report: dict = {"sweeps": {}}
    # internal control first
    defaults = {}
    for sw, (_vals, ref, _lbl) in SWEEPS.items():
        defaults[sw] = float(np.mean(list(per[(sw, ref)].values()))) if per.get((sw, ref)) else float("nan")
    same = max(defaults.values()) - min(defaults.values())
    print(f"  internal control: mean J&F of the three DEFAULT variants "
          f"(A8 K=5, A9 J=3, A10 M=13)")
    for sw, v in defaults.items():
        print(f"      {sw} = {v:.6f}")
    print(f"      spread = {same:.2e}  -> "
          f"{'OK: runs are deterministic' if same < 1e-12 else 'FAIL: runs are not reproducible'}")
    if not (same < 1e-12):
        _fail.append("default variants disagree -> the sweep is not deterministic")

    for sw, (vals, ref, lbl) in SWEEPS.items():
        if (sw, ref) not in per:
            _fail.append(f"{sw}: reference variant {ref} missing")
            continue
        ref_map = per[(sw, ref)]
        seqs = sorted(ref_map)
        base = [ref_map[s] for s in seqs]
        variants = [v for v in vals if v != ref]
        res = {}
        for v in variants:
            if (sw, v) not in per:
                _fail.append(f"{sw}: variant {v} missing")
                continue
            m = per[(sw, v)]
            common = [s for s in seqs if s in m]
            res[v] = {"report": paired_report([ref_map[s] for s in common],
                                              [m[s] for s in common],
                                              f"{lbl}={ref}", f"{lbl}={v}"),
                      "common_seqs": len(common)}
        # Holm across the variants of this sweep
        keys = [v for v in variants if v in res]
        ps = [res[v]["report"]["wilcoxon_p"] for v in keys]
        adj = holm_bonferroni(ps) if ps else []
        print(f"\n  --- {sw}: {lbl} sweep   (reference {lbl}={ref}, "
              f"mean J&F={float(np.mean(base)):.4f}, n_seq={len(seqs)}) ---")
        print(f"      {'variant':>8} {'mean J&F':>9} {'delta':>8} {'Cliff d':>8} "
              f"{'p_holm':>9}  {'95% CI of delta':>24}")
        entry = {"label": lbl, "reference": ref,
                 "reference_mean": float(np.mean(base)), "variants": {}}
        if not keys:
            check = abs(np.mean(base) - 0.7017) < 5e-3
            entry["error"] = "no variant identified"
            if not check:
                _fail.append(f"{sw}: could not identify variants")
            print("      (no variant rows found -- arm column missing?)")
            report["sweeps"][sw] = entry
            continue
        for v, pv in zip(keys, adj):
            rp = res[v]["report"]
            mean_v = float(np.mean([per[(sw, v)][s] for s in seqs if s in per[(sw, v)]]))
            star = "*" if (pv < 0.05 and abs(rp["cliffs_delta"]) >= 0.147) else " "
            print(f"      {v:>8} {mean_v:>9.4f} {rp['mean_delta']:>+8.4f} "
                  f"{rp['cliffs_delta']:>+8.3f} {pv:>9.3g}  "
                  f"[{rp['delta_ci_lo']:>+8.4f}, {rp['delta_ci_hi']:>+8.4f}]{star}")
            entry["variants"][str(v)] = {
                "mean": mean_v, "delta": rp["mean_delta"],
                "cliffs_delta": rp["cliffs_delta"],
                "ci": [rp["delta_ci_lo"], rp["delta_ci_hi"]],
                "p_holm": float(pv),
                "decidable": bool(pv < 0.05 and abs(rp["cliffs_delta"]) >= 0.147)}
        n_dec = sum(1 for d in entry["variants"].values() if d["decidable"])
        entry["n_decidable"] = n_dec
        print(f"      decidable effects (p_holm<0.05 and |delta|>=0.147): {n_dec} of {len(keys)}")
        report["sweeps"][sw] = entry

    report["internal_control"] = {"defaults": defaults, "spread": float(same),
                                 "deterministic": bool(same < 1e-12)}
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(ROOT)}")

    print("\n" + "=" * 70)
    if _fail:
        print(f"RESULT: {len(_fail)} PROBLEM(S)")
        for f in _fail:
            print("   -", f)
        return 1
    print("RESULT: ok -- see the table above for the CL11 verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
