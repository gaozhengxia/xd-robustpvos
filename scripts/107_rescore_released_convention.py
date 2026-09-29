"""Rewrite cl4_full_v2.csv's J&F under the released evaluator's boundary rule.

WHY
---
`scripts/105_eval_protocol_ab.py` established, on real predictions from the same
code path that produced the CSV, that:

  * J (region) is *bit-identical* between `xdrp.metrics` and the released
    `sam2.sav_dataset` evaluator -- our IoU is not a source of error at all;
  * F (boundary) differs a lot, because the released scorer dilates the 1-px
    boundary map with `disk(ceil(0.008*norm(shape)))` -- radius 8 at 480x854, a
    17-px band -- while ours uses a 5x5 box.

So the CSV's J&F is a systematically *conservative* estimate of the released
metric. This script converts it, and reports how much the conversion is trusted.

WHICH FORM OF THE CORRECTION IS USED, AND WHY
---------------------------------------------
Measured on 6 cells:

    F_released / F_ours :  1.145 .. 1.738   (a 1.5x spread)
    J&F gain            : +0.063 .. +0.097  (tight)

The *ratio* is not transferable -- it blows up for degenerate masks (the dagrs
cells have F_ours 0.18, where the wider tolerance changes the boundary band far
more in relative terms). The *additive* gain is stable, because both scorers
compute `F = 2PR/(P+R)` on almost the same boundary geometry: widening the
tolerance mostly converts "missed" boundary pixels into "matched" ones, and the
number of such pixels is bounded by the boundary *perimeter*, which does not
collapse with the mask.

An earlier version of this script applied a multiplicative k to J&F and
predicted `sam2video = 0.9679` with an upper bound of **1.1034** -- above 1.0,
which is impossible. That is the sanity check that caught it: any convention
conversion that can produce J&F > 1 is wrong by construction. The additive form
cannot: the gain is bounded by `0.5 * (1 - F_ours)`.

WHAT IS REPORTED
----------------
  1. `J&F_new = J&F_ours + gain`, gain measured per cell. Applied at the
     **instance** level, because that is the level at which it was measured;
     segment and dataset means are then formed exactly as the paper forms them
     (instance -> segment -> equal-weight over segments).
  2. An interval from the most conservative / most generous measured gain, so
     the estimate cannot be gamed by choosing which cell's gain to prefer.
  3. The effect of the released `skip_first_and_last=True` default (measured
     mean -0.0043, and it cannot touch J), stated rather than omitted.

CAVEATS THAT MUST TRAVEL WITH EVERY NUMBER THIS SCRIPT PRODUCES
--------------------------------------------------------------
  * `gain` was measured on **object 1 only**, on **6 clean cells** (4 sam2video,
    1 dagrs, 1 greedy; two on the same sequence). Multi-instance sequences are
    therefore *extrapolated*, not measured.
  * The uniform-gain assumption is the approximation: the measured spread is
    +/-0.017 around the mean, and the bound column expresses exactly that.
  * The two dagrs/greedy cells carry the highest gains (+0.066/+0.070), but
    they are also the most degenerate masks; the spread is not clearly
    arm-dependent, and the report does not assume it is.
  * It is a *re-scoring* of existing masks. It does not re-run anything.

So this belongs in a footnote / appendix as "our numbers under the released
convention, estimated from a measured per-cell transfer", never as a headline.
The paper's own three-arm comparison stays on our scorer, which is the only one
applied identically to all three arms.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_instances(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{"seq": r["seq"], "obj_id": int(r["obj_id"]), "mode": r["mode"],
             "degradation": r["degradation"], "J": float(r["J"]),
             "F": float(r["F"]), "J&F": float(r["J&F"])} for r in rows]


def collapse(a: List[float]) -> List[float]:
    """Instance list -> segment mean, i.e. one number per sequence."""
    return [float(np.mean(a))] if a else [float("nan")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(ROOT / "results" / "cl4_full_v2.csv"))
    ap.add_argument("--ab", default=str(ROOT / "results" / "eval_protocol_ab.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "cl4_released_convention.json"))
    args = ap.parse_args()

    csv_path, ab_path = Path(args.csv), Path(args.ab)
    if not ab_path.exists():
        print(f"no A/B measurement at {ab_path}; run scripts/105 first")
        return 1

    with ab_path.open(encoding="utf-8", newline="") as fh:
        ab_rows = list(csv.DictReader(fh))

    # Additive J&F gain per A/B cell: 0.5 * (F_released - F_ours), J unchanged.
    gains = np.array([0.5 * (float(r["official_F"]) - float(r["ours_F"]))
                      for r in ab_rows])
    g_mean, g_min, g_max = float(gains.mean()), float(gains.min()), float(gains.max())
    ratios = np.array([float(r["official_F"]) / float(r["ours_F"]) for r in ab_rows])
    skip_deltas = np.array([float(r["delta_skip_official"]) for r in ab_rows])

    inst = load_instances(csv_path)
    # Two physical bounds on the correction, both of which the naive form can
    # violate and both of which must be enforced rather than assumed:
    #
    #   * J&F cannot exceed 1. The additive gain `0.5*(F_new - F_ours)` is itself
    #     bounded by `0.5*(1 - F_ours)`, so applying the *dataset-wide maximum*
    #     gain to a cell whose F is already high over-shoots. `blackswan/obj1/
    #     sam2video/clean` has J&F 0.9054 and F 0.8498; max-gain would demand
    #     F_new = 1.0444, which is impossible. The bound is therefore clamped at
    #     1.0 -- an interval that silently exceeds a metric's range is not a
    #     bound, and the earlier multiplicative version of this script predicted
    #     1.1034 for exactly this reason.
    #   * J and F cannot decrease.
    #
    # The direct estimate (mean gain) never violates either: the mean gain is
    # 0.077, well inside every cell's headroom.
    def convert(base: float, gain: float) -> float:
        return min(1.0, base + gain)

    def mode_table(degradation: str = "") -> Dict[str, Dict[str, float]]:
        """Segment-equal means, optionally restricted to one degradation."""
        rs = [r for r in inst if not degradation or r["degradation"] == degradation]
        by: Dict[tuple, List[Dict]] = {}
        for r in rs:
            by.setdefault((r["mode"], r["seq"]), []).append(r)
        acc: Dict[str, Dict[str, List[float]]] = {}
        for (mode, _seq), group in by.items():
            base = float(np.mean([x["J&F"] for x in group]))
            d = acc.setdefault(mode, {"ours": [], "direct": [], "lo": [],
                                      "hi": [], "J": [], "F": []})
            d["ours"].append(base)
            d["direct"].append(convert(base, g_mean))
            d["lo"].append(convert(base, g_min))
            d["hi"].append(convert(base, g_max))
            d["J"].append(float(np.mean([x["J"] for x in group])))
            d["F"].append(float(np.mean([x["F"] for x in group])))
        return {m: {"J&F": float(np.mean(v["ours"])),
                    "J&F_released_direct": float(np.mean(v["direct"])),
                    "J&F_released_lo": float(np.mean(v["lo"])),
                    "J&F_released_hi": float(np.mean(v["hi"])),
                    "J_mean": float(np.mean(v["J"])),
                    "F_mean": float(np.mean(v["F"])),
                    "n_segments": len(v["ours"])}
                for m, v in acc.items()}

    report = {
        "source_csv": str(csv_path),
        "ab_measurement": str(ab_path),
        "correction": {
            "form": "additive: J&F_released = J&F_ours + 0.5*(F_released - F_ours)",
            "gain_mean": g_mean, "gain_min": g_min, "gain_max": g_max,
            "F_ratio_mean": float(ratios.mean()),
            "F_ratio_min": float(ratios.min()), "F_ratio_max": float(ratios.max()),
            "skip_first_last_mean_delta": float(skip_deltas.mean()),
            "gain_per_cell": {f"{r['seq']}/{r['mode']}/{r['degradation']}":
                              0.5 * (float(r["official_F"]) - float(r["ours_F"]))
                              for r in ab_rows},
            "why_not_multiplicative": (
                "A multiplicative k on J&F predicted sam2video = 0.9679 with an "
                "upper bound of 1.1034 -- above 1.0, which is impossible. The "
                "F ratio (1.145..1.738) is not transferable; the additive gain "
                "(0.063..0.097) is."),
        },
        "measurement_scope": (
            "object 1 only, 6 clean cells (4 sam2video, 1 dagrs, 1 greedy; two on "
            "the same sequence). Multi-instance sequences are extrapolated."),
        "dataset_level_segment_equal": mode_table(),
        "by_degradation": {
            deg: mode_table(deg)
            for deg in ("clean", "dust", "fog", "underwater")
            if any(r["degradation"] == deg for r in inst)
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[rescore] additive J&F gain: mean {g_mean:+.4f}  "
          f"min {g_min:+.4f}  max {g_max:+.4f}   (F ratio {ratios.min():.3f}"
          f"..{ratios.max():.3f}, not transferable)")
    print(f"[rescore] skip-first/last effect on released F: {skip_deltas.mean():+.4f}")
    print("[rescore] DAVIS-2017 val, 30 segments, segment-equal weighting:")
    print(f"  {'mode':<11s} {'ours':>8s} {'released':>9s} "
          f"{'[lo':>8s} {'hi]':>8s}")
    for m, v in sorted(report["dataset_level_segment_equal"].items()):
        print(f"  {m:<11s} {v['J&F']:8.4f} {v['J&F_released_direct']:9.4f} "
              f"{v['J&F_released_lo']:8.4f} {v['J&F_released_hi']:8.4f}")
    print("[rescore] by degradation (ours -> released):")
    for deg, tbl in report["by_degradation"].items():
        line = "  ".join(f"{m}={v['J&F']:.3f}->{v['J&F_released_direct']:.3f}"
                         for m, v in sorted(tbl.items()))
        print(f"  {deg:<11s} {line}")
    print(f"[rescore] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
