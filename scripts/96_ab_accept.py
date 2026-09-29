"""96 - A/B the ACCEPTANCE POLICY of the propagation baseline.

Project-internal diagnostic (NOT a paper artefact). It answers the question that
decides whether the paper's baseline is defensible:

    `greedy` refuses nothing -- every frame's prediction becomes the next frame's
    prompt. On clean DAVIS val that loop destroys itself (bmx-trees 13.1 J&F,
    area_last = 98x the ground-truth area). Is the damage caused by the missing
    refusal, and does a STANDARD correction repair it?

Three corrections are tested. None of them is DAG-RS's mechanism -- a baseline
that borrows evidence-driven re-anchoring stops being a baseline:

  gate     refuse a prediction whose predicted IoU is below a threshold and emit
           the flow-propagated anchor instead. (SAM 2's own default for this
           number is 0.88, from the automatic-mask-generation config.)
  reseed   re-prompt from the annotated first frame every K frames -- the classic
           drift correction for a loop with no memory bank.
  robust   both.

The threshold / period sensitivity is swept because a single value could always
be accused of having been tuned to lose.

One probe is self-validated before any verdict is printed: `accept_all` here must
reproduce the production `greedy` numbers bit-for-bit, otherwise this script is
measuring a different pipeline than the paper reports.

Run from the PROJECT ROOT (split across processes; each writes its own file):

    python scripts/96_ab_accept.py --seq bmx-trees,libby,bike-packing,car-roundabout \\
        --degradations clean --levels 1 --frame-stride 1 \\
        --out _scratch/attr/ab_accept_a.csv
    python scripts/96_ab_accept.py --seq blackswan,parkour,drift-straight \\
        --degradations clean --levels 1 --frame-stride 1 \\
        --out _scratch/attr/ab_accept_b.csv

Then merge with scripts/94_merge_sweeps.py and read the verdict tables.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (ROOT, base_parser, config_for_mode, make_backend_for,  # noqa: E402
                     make_bank, open_src, preset,
                     resolve_degradations, resolve_levels)
from xdrp.benchmark import Protocol, load_degraded_sequence   # noqa: E402
from xdrp.evidence import as_bool                             # noqa: E402
from xdrp.metrics import jf                                   # noqa: E402
from xdrp.pipeline import run_sequence                        # noqa: E402


#: area_ratio above this counts as "the mask has left the object"
RUNAWAY_RATIO = 3.0

#: (label, mode, config overrides). `greedy_gate` defaults to tau=0.88 and
#: `greedy_reseed` to period 20; the extra entries expose the sensitivity of the
#: verdict to those two numbers.
ARMS: Tuple[Tuple[str, str, Dict[str, Any]], ...] = (
    ("accept_all",   "greedy",        {}),
    ("gate_70",      "greedy_gate",   {"tau_score": 0.70}),
    ("gate_88",      "greedy_gate",   {}),
    ("gate_95",      "greedy_gate",   {"tau_score": 0.95}),
    ("reseed_10",    "greedy_reseed", {"reseed_stride": 10}),
    ("reseed_20",    "greedy_reseed", {}),
    ("robust_88_20", "greedy_robust", {}),
)


def run_cell(s: Dict[str, Any], src, seq: str, deg: str, level: float,
             label: str, mode: str, over: Dict[str, Any],
             backend, bank) -> Tuple[List[Dict], Dict]:
    proto = Protocol(degradation=deg, level=float(level),
                     frame_stride=int(s["frame_stride"]),
                     max_frames=int(s["max_frames"]),
                     severity_amp=float(s["severity_amp"]),
                     seed=int(s["seed"]))
    dseq = load_degraded_sequence(src, seq, proto,
                                 max_objects=int(s["max_objects"]))
    if not dseq.object_ids:
        raise RuntimeError(
            f"{seq}: frame-0 annotation contains no object -- the label reader or "
            f"the split is wrong, not the arm.")
    obj = int(dseq.object_ids[0])

    cfg = config_for_mode(s["cfg"], mode, **over)
    # The arm's identity must survive into the config, or this script is
    # comparing an arm against itself and reporting a tie as a finding.
    for k, v in over.items():
        if getattr(cfg, k) != v:
            raise AssertionError(f"{label}: override {k}={v!r} did not reach the "
                                 f"config (got {getattr(cfg, k)!r})")
    if mode != "greedy" and cfg.use_reanchor:
        raise AssertionError(f"{label}: a propagation baseline must not re-anchor")

    backend.clear_cache()
    frames = dseq.frames_for_mode(cfg.mode)

    t0 = time.time()
    res = run_sequence(frames, dseq.first_masks[obj], backend, bank, cfg,
                       seq_name=seq, obj_id=obj, flow_frames=frames)
    dt = time.time() - t0

    gts = dseq.gt_masks[obj]
    rows: List[Dict[str, Any]] = []
    for t in range(len(res.masks)):
        j, f, jfval = jf(res.masks[t], gts[t], int(s["dilation"]))
        pred_area = int(as_bool(res.masks[t]).sum())
        gt_area = int(as_bool(gts[t]).sum())
        rows.append({
            "seq": seq, "arm": label, "mode": mode,
            "degradation": deg, "level": float(level), "frame": t,
            "J": j, "F": f, "JF": jfval,
            "pred_area": pred_area, "gt_area": gt_area,
            "area_ratio": (pred_area / gt_area) if gt_area else float("nan"),
            "n_evaluated": int(res.n_evaluated[t]),
            "is_keyframe": int(res.is_keyframe[t]),
            "backend_score": (float(res.backend_score[t])
                              if np.isfinite(res.backend_score[t]) else float("nan")),
        })

    ratios = [r["area_ratio"] for r in rows[1:]]      # frame 0 is the given mask
    summ = {
        "seq": seq, "arm": label, "mode": mode,
        "degradation": deg, "level": float(level),
        "tau_score": float(cfg.tau_score),
        "reseed_stride": int(cfg.reseed_stride),
        "n_frames": len(rows),
        "JF": float(np.mean([r["JF"] for r in rows])),
        "JF_f1": (float(np.mean([r["JF"] for r in rows[1:]]))
                  if len(rows) > 1 else float("nan")),
        "area_first": float(ratios[0]) if ratios else float("nan"),
        "area_last": float(ratios[-1]) if ratios else float("nan"),
        "area_max": float(np.nanmax(ratios)) if ratios else float("nan"),
        "runaway": int(bool(ratios) and float(np.nanmax(ratios)) > RUNAWAY_RATIO),
        "n_rejected": int(res.n_rejected),
        "n_reseeds": int(res.n_reseeds),
        "n_predict_failures": int(res.n_predict_failures),
        "wall_time_s": dt,
    }
    return rows, summ


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--seq", default=None,
                    help="comma-separated sequence names; default = first N of split")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of: " + ",".join(a[0] for a in ARMS))
    args = ap.parse_args()
    s = preset(args)

    src = open_src(s)
    seqs = ([x.strip() for x in args.seq.split(",") if x.strip()] if args.seq
            else src.sequences()[:max(1, int(s["max_sequences"] or 4))])
    degs = resolve_degradations(args.degradations, s["cfg"])
    levels = resolve_levels(args.levels, s["cfg"])
    arms = [(l, m, o) for (l, m, o) in ARMS
            if not args.arms or l in {x.strip() for x in args.arms.split(",")}]

    print("=" * 92)
    print("XD-RobustPVOS :: acceptance policy A/B for the propagation baseline")
    print("=" * 92)
    print(f"dataset_root : {s['dataset_root']}")
    print(f"backend      : {s['backend_kind']}   stride={s['frame_stride']}")
    print(f"sequences    : {seqs}")
    print(f"degradations : {degs}  levels {levels}")
    print(f"arms         : {[a[0] for a in arms]}")

    backend = make_backend_for(s)
    bank = make_bank(s["cfg"])

    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    t_start = time.time()

    for deg in degs:
        for level in levels:
            print(f"\n{'-' * 92}\n{deg}  L{level:g}  (stride={s['frame_stride']})\n{'-' * 92}")
            for seq in seqs:
                cell: Dict[str, Dict[str, Any]] = {}
                for label, mode, over in arms:
                    try:
                        rows, summ = run_cell(s, src, seq, deg, level, label,
                                              mode, over, backend, bank)
                    except BaseException as e:                # noqa: BLE001
                        import traceback
                        print(f"[{seq} | {label}] HARD FAILURE: "
                              f"{type(e).__name__}: {e}")
                        print(traceback.format_exc())
                        continue
                    all_rows.extend(rows)
                    summary.append(summ)
                    cell[label] = summ
                    print(f"  {seq:<18} {label:<14} J&F={summ['JF_f1']:>7.4f} "
                          f"area_last={summ['area_last']:>8.2f} "
                          f"area_max={summ['area_max']:>8.2f} "
                          f"runaway={'Y' if summ['runaway'] else 'n'} "
                          f"rej={summ['n_rejected']:>3} "
                          f"reseed={summ['n_reseeds']:>2}")

    if not summary:
        print("\nno cell completed -- nothing to compare.")
        return 1

    # ---- verdict ---------------------------------------------------------- #
    base_label = "accept_all"
    print("\n" + "=" * 92)
    print(f"VERDICT  (paired against `{base_label}` on identical frames)")
    print("=" * 92)
    print(f"cells: {len(summary)}   wall time: {(time.time() - t_start) / 60:.1f} min")

    have = [a[0] for a in arms if any(x["arm"] == a[0] for x in summary)]
    hdr = f"{'arm':<14}{'mean J&F':>10}{'median':>9}{'worst seq':>11}{'runaways':>10}"
    for sq in seqs:
        hdr += f"{sq[:9]:>11}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for label in have:
        v = [x for x in summary if x["arm"] == label]
        vals = np.array([x["JF_f1"] for x in v], float)
        row = (f"{label:<14}{float(np.nanmean(vals)):>10.4f}"
               f"{float(np.nanmedian(vals)):>9.4f}{float(np.nanmin(vals)):>11.4f}"
               f"{sum(x['runaway'] for x in v):>10}")
        for sq in seqs:
            m = [x["JF_f1"] for x in v if x["seq"] == sq]
            row += f"{(m[0] if m else float('nan')):>11.4f}"
        print(row)

    if base_label in have:
        print()
        print(f"{'arm':<14}{'d vs base':>11}{'improved':>10}{'worse':>8}"
              f"{'rejections':>12}{'reseeds':>9}")
        print("-" * 66)
        for label in have:
            if label == base_label:
                continue
            pairs = []
            for sq in seqs:
                b = [x["JF_f1"] for x in summary
                     if x["arm"] == base_label and x["seq"] == sq]
                t = [x["JF_f1"] for x in summary
                     if x["arm"] == label and x["seq"] == sq]
                if b and t:
                    pairs.append(t[0] - b[0])
            d = np.array(pairs, float)
            nrej = sum(x["n_rejected"] for x in summary if x["arm"] == label)
            nres = sum(x["n_reseeds"] for x in summary if x["arm"] == label)
            print(f"{label:<14}{float(np.mean(d)):>+11.4f}"
                  f"{int((d > 0.005).sum()):>10}{int((d < -0.005).sum()):>8}"
                  f"{nrej:>12}{nres:>9}")

        print()
        best = max(have, key=lambda l: float(np.nanmean(
            [x["JF_f1"] for x in summary if x["arm"] == l])))
        print(f"best arm: {best}  "
              f"(mean over {len(seqs)} sequences)")
        if best == base_label:
            print("  => no standard correction improved on unconditional acceptance.")
            print("     The closed loop is NOT the binding problem here, and changing")
            print("     the propagation code will not close the gap to published SAM 2.1.")
        else:
            print("  => a standard drift correction DOES matter; quote the corrected")
            print("     arm as the baseline, not `greedy`.")

    if any(x["n_predict_failures"] for x in summary):
        print("\n[WARN] some cells had backend.predict failures that fell back to the")
        print("       warped anchor; those cells are NOT clean measurements.")

    # ---- write ------------------------------------------------------------ #
    def _write(path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p) + ".tmp", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        Path(str(p) + ".tmp").replace(p)
        print(f"  {len(rows):>6} rows -> {path}")

    print("")
    p1 = args.out or str(ROOT / "results" / "ab_accept.csv")
    p2 = str(Path(p1).with_name(Path(p1).stem + "_summary.csv"))
    _write(p1, all_rows)
    _write(p2, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
