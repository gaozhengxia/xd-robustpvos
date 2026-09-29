"""92 - hard vs soft re-prompting, head to head on byte-identical frames.

Project-internal diagnostic (NOT a paper artefact). It answers exactly one
question, with the real backend and the real production config:

    Does carrying the segmenter's own probability across frames (`soft`) stop
    the closed-loop runaway that the hard +-8 logit wall produces?

`hard` (previous behaviour) re-prompts every frame with a BINARISED warp of the
anchor mask turned into a +-8 logit wall: the boundary is *asserted*. `soft`
carries the segmenter's own low-resolution probability, warped by the same flow,
so the boundary can be *re-decided* from the current image. See
docs/BASELINE_DIAGNOSIS.md section 2.3 for the failure this targets.

Both arms run in the SAME process on the SAME degraded sequence object, so the
comparison cannot be confounded by re-generating the degradation or by a
different operator bank.

Differences from scripts/90_diagnose_greedy.py, deliberately:
  * the config comes from `_common.config_for_mode`, i.e. exactly the config the
    production runners use. 90_* used `make_config` directly, so its yaml
    tunables (keyframe stride, J, kappa, ...) were NOT the production ones.
    A diagnostic that measures a different configuration than the paper is the
    same class of bug as the ones catalogued in docs/BASELINE_DIAGNOSIS.md.
  * it reports the runaway signature directly (area_ratio last / max) instead of
    leaving the reader to diff two text dumps by eye.

Run from the PROJECT ROOT:

    python scripts/92_ab_mask_prompt.py --seq lab-coat,drift-straight,bmx-trees,shooting \\
        --degradations clean --levels 1 --frame-stride 1 --mode greedy

Output: per-frame rows -> results/ab_mask_prompt.csv
        per-cell summary -> results/ab_mask_prompt_summary.csv
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
                     make_bank, open_src, out_path, preset,
                     resolve_degradations, resolve_levels)
from xdrp.benchmark import Protocol, load_degraded_sequence   # noqa: E402
from xdrp.evidence import as_bool                             # noqa: E402
from xdrp.metrics import jf                                   # noqa: E402
from xdrp.pipeline import run_sequence                        # noqa: E402


#: area_ratio above this counts as "the mask has left the object". 3x is well
#: outside anything the dilation-2 boundary metric can absorb.
RUNAWAY_RATIO = 3.0


# --------------------------------------------------------------------------- #
# one (sequence, degradation, prompt mode) cell
# --------------------------------------------------------------------------- #

def run_cell(s: Dict[str, Any], src, seq: str, deg: str, level: float,
             mode: str, mask_prompt: str, backend, bank) -> Tuple[List[Dict], Dict]:
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
            f"the split is wrong, not the method.")
    obj = int(dseq.object_ids[0])

    cfg = config_for_mode(s["cfg"], mode, mask_prompt=mask_prompt)
    # The whole point of the A/B is that this switch reached the config. The
    # defect that motivated this script was a yaml knob that never did.
    if str(cfg.mask_prompt).lower() != mask_prompt:
        raise AssertionError(
            f"mask_prompt did not reach the config: asked {mask_prompt!r}, "
            f"config says {cfg.mask_prompt!r}")

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
            "seq": seq, "mode": mode, "mask_prompt": mask_prompt,
            "degradation": deg, "level": float(level), "frame": t,
            "J": j, "F": f, "JF": jfval,
            "pred_area": pred_area, "gt_area": gt_area,
            "area_ratio": (pred_area / gt_area) if gt_area else float("nan"),
            "n_evaluated": int(res.n_evaluated[t]),
            "is_keyframe": int(res.is_keyframe[t]),
            "backend_score": (float(res.backend_score[t])
                              if np.isfinite(res.backend_score[t]) else float("nan")),
        })

    ratios = [r["area_ratio"] for r in rows[1:]]          # frame 0 is the given mask
    summ = {
        "seq": seq, "mode": mode, "mask_prompt": mask_prompt,
        "degradation": deg, "level": float(level),
        "n_frames": len(rows),
        "JF": float(np.mean([r["JF"] for r in rows])),
        "JF_f1": float(np.mean([r["JF"] for r in rows[1:]])) if len(rows) > 1 else float("nan"),
        "area_first": float(ratios[0]) if ratios else float("nan"),
        "area_last": float(ratios[-1]) if ratios else float("nan"),
        "area_max": float(np.nanmax(ratios)) if ratios else float("nan"),
        "area_mean": float(np.nanmean(ratios)) if ratios else float("nan"),
        "runaway": int(bool(ratios) and float(np.nanmax(ratios)) > RUNAWAY_RATIO),
        "n_predict_failures": int(res.n_predict_failures),
        "n_encoder_calls": int(res.n_encoder_calls),
        "wall_time_s": dt,
    }
    return rows, summ


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--mode", default="greedy",
                    help="segmentation policy to A/B; 'greedy' isolates the prompt "
                         "question, 'dagrs' measures the full method")
    ap.add_argument("--seq", default=None,
                    help="comma-separated sequence names; default = first N of split")
    ap.add_argument("--modes", default="hard,soft",
                    help="prompt modes to compare")
    args = ap.parse_args()
    s = preset(args)

    src = open_src(s)
    seqs = ([x.strip() for x in args.seq.split(",") if x.strip()] if args.seq
            else src.sequences()[:max(1, int(s["max_sequences"] or 4))])
    degs = resolve_degradations(args.degradations, s["cfg"])
    levels = resolve_levels(args.levels, s["cfg"])
    prompts = [x.strip() for x in args.modes.split(",") if x.strip()]

    print("=" * 78)
    print("XD-RobustPVOS :: hard vs soft re-prompting A/B")
    print("=" * 78)
    print(f"cwd            : {Path.cwd()}")
    print(f"dataset_root   : {s['dataset_root']}")
    print(f"backend        : {s['backend_kind']}")
    print(f"sequences      : {seqs}")
    print(f"degradations   : {degs}")
    print(f"levels         : {levels}")
    print(f"frame_stride   : {s['frame_stride']}   max_frames={s['max_frames']}")
    print(f"mode           : {args.mode}   prompt modes: {prompts}")

    backend = make_backend_for(s)
    bank = make_bank(s["cfg"])

    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    t_start = time.time()

    for deg in degs:
        for level in levels:
            print(f"\n{'-' * 78}\n{deg}  L{level:g}  (stride={s['frame_stride']})\n{'-' * 78}")

            for seq in seqs:
                cell: Dict[str, Dict[str, Any]] = {}
                for mp in prompts:
                    try:
                        rows, summ = run_cell(s, src, seq, deg, level, args.mode,
                                              mp, backend, bank)
                    except BaseException as e:            # noqa: BLE001
                        import traceback
                        print(f"[{seq} | {deg} L{level:g} | {mp}] HARD FAILURE: "
                              f"{type(e).__name__}: {e}")
                        print(traceback.format_exc())
                        continue
                    all_rows.extend(rows)
                    summary.append(summ)
                    cell[mp] = summ

                if not cell:
                    continue
                h, so = cell.get("hard"), cell.get("soft")
                if h and so:
                    print(f"{seq:<16} J&F {h['JF_f1']:>7.4f} -> {so['JF_f1']:>7.4f} "
                          f"({so['JF_f1'] - h['JF_f1']:+.4f}) | "
                          f"area last {h['area_last']:>9.2f} -> {so['area_last']:>9.2f} | "
                          f"area max {h['area_max']:>9.2f} -> {so['area_max']:>9.2f} | "
                          f"runaway {'Y' if h['runaway'] else 'n'}"
                          f"->{'Y' if so['runaway'] else 'n'} | "
                          f"fails {h['n_predict_failures']}/{so['n_predict_failures']}")
                else:
                    for mp, sm in cell.items():
                        print(f"{seq:<16} [{mp}] J&F(>=1)={sm['JF_f1']:.4f} "
                              f"area_last={sm['area_last']:.2f} "
                              f"fails={sm['n_predict_failures']}")

    # ---- verdict ---------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not summary:
        print("no cell completed -- nothing to compare.")
        return 1

    print(f"cells: {len(summary)}   total wall time: "
          f"{(time.time() - t_start) / 60:.1f} min")
    print(f"\n{'prompt':<8}{'cells':>7}{'mean J&F>=1':>13}{'median':>10}"
          f"{'runaways':>10}{'failures':>10}")
    for mp in prompts:
        v = [x for x in summary if x["mask_prompt"] == mp]
        if not v:
            continue
        vals = np.array([x["JF_f1"] for x in v], dtype=float)
        print(f"{mp:<8}{len(v):>7}{float(np.nanmean(vals)):>13.4f}"
              f"{float(np.nanmedian(vals)):>10.4f}"
              f"{sum(x['runaway'] for x in v):>10}"
              f"{sum(x['n_predict_failures'] for x in v):>10}")

    if len(prompts) >= 2:
        # paired comparison per (seq, degradation, level)
        pairs = []
        for key in {(x["seq"], x["degradation"], x["level"]) for x in summary}:
            h = [x for x in summary if (x["seq"], x["degradation"], x["level"]) == key
                 and x["mask_prompt"] == "hard"]
            so = [x for x in summary if (x["seq"], x["degradation"], x["level"]) == key
                  and x["mask_prompt"] == "soft"]
            if h and so:
                pairs.append((key, h[0]["JF_f1"], so[0]["JF_f1"]))
        if pairs:
            d = np.array([b - a for _, a, b in pairs], dtype=float)
            print(f"\npaired cells: {len(d)}   mean delta (soft - hard) on J&F >=1 "
                  f"= {float(np.mean(d)):+.4f}   improved "
                  f"{int((d > 1e-9).sum())} / unchanged {int((abs(d) <= 1e-9).sum())}"
                  f" / worse {int((d < -1e-9).sum())}")
            worst = sorted(pairs, key=lambda p: p[2] - p[1])[0]
            best = sorted(pairs, key=lambda p: p[1] - p[2])[0]
            print(f"  biggest gain : {best[0]}  {best[1]:.4f} -> {best[2]:.4f}")
            print(f"  biggest loss : {worst[0]}  {worst[1]:.4f} -> {worst[2]:.4f}")

    if any(x["n_predict_failures"] for x in summary):
        print("\n[WARN] some cells contain backend.predict failures that fell back to "
              "the warped anchor; those cells are NOT clean measurements.")

    # ---- write ------------------------------------------------------------ #
    def _write(path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path + ".tmp", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        Path(path + ".tmp").replace(path)
        print(f"  {len(rows):>6} rows -> {path}")

    print("")
    p1 = out_path(s, args, "ab_mask_prompt.csv")
    # Derive the summary path from the per-frame path so `--out foo.csv` cannot
    # make the two writes collide on the same file.
    p2 = str(Path(p1).with_name(Path(p1).stem + "_summary.csv"))
    _write(p1, all_rows)
    _write(p2, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
