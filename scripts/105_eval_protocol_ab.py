"""Close the evaluator residual risk: our J&F scorer vs the released SAM 2 evaluator.

WHY THIS EXISTS
---------------
`results/cl4_full_v2.csv` reports `sam2video` clean at J&F 0.8393 (segment-equal
weights, DAVIS-2017 val). The SAM 2.1 paper reports a higher DAVIS-2017 val
number for its largest checkpoint. Three arms share our scorer, so the internal
`dagrs` vs `sam2video` comparison is unaffected -- but a reviewer comparing our
`sam2video` row against the published table will ask why it is lower, and
"our evaluator is different" is not an answer until the difference is
*quantified*. This script quantifies it.

WHAT IT DOES
------------
Re-runs a small, explicitly listed set of (sequence, mode, degradation) cells
through the *same* code path that produced the CSV (`xdrp.benchmark` ->
`xdrp.datasets.DavisReader` -> `xdrp.degradations` -> `xdrp.pipeline.run_sequence`),
keeps the per-frame masks, and scores each cell with four scorers over the
identical masks:

  ours                 xdrp.metrics.seq_metrics (dilation=2)        -- what the CSV used
  ours_skip            same, but dropping frames 0 and T-1
  official             a faithful port of sam2/sav_dataset/utils/sav_benchmark.py
                       (`_seg2bmap` + `disk(ceil(0.008*norm(shape)))` + `get_iou`)
  official_skip        the same, with `skip_first_and_last=True` -- the released
                       default, and DAVIS semi-supervised evaluation practice

The `official` scorer is *also* cross-checked against the real released code by
importing `sam2.sav_dataset.utils.sav_benchmark` and feeding it the same masks
through its public `Evaluator.feed_frame` / `.conclude` API. If the port and the
released implementation disagree on any cell, the script exits non-zero: a port
that is only *claimed* faithful is exactly the kind of unverified probe this
repository has been burned by before (see docs/BASELINE_DIAGNOSIS.md).

The critical property of this design is that both scoring paths see the SAME
MASK ARRAYS. Any measured gap is therefore attributable to the scorer, not to
re-running a stochastic model.

CACHING
-------
Re-inference is the expensive part (each `sam2video` cell loads SAM 2.1 and
tracks the clip). Masks are cached to
`_scratch/eval_protocol_ab/cache/<seq>__<mode>__<deg>.npz`, so re-running the
script after changing only the scorer costs nothing. Delete the cache to force
a fresh inference.

USAGE
-----
    python scripts/105_eval_protocol_ab.py                  # default 6 cells
    python scripts/105_eval_protocol_ab.py --cells bike-packing:sam2video:clean
    python scripts/105_eval_protocol_ab.py --all-defaults   # 12 cells, ~25 min
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import (  # noqa: E402
    ROOT, base_parser, config_for_mode, deep_get, make_backend_for, make_bank,
    open_src, preset,
)
from scripts.eval_protocol_common import (  # noqa: E402
    official_seq_metrics, ours_seq_metrics, released_check,
)

CACHE_DIR = ROOT / "_scratch" / "eval_protocol_ab" / "cache"

#: Cells scored by default. Chosen to span the phenomenon rather than to flatter
#: it: `sam2video` on clean is the row a reviewer compares with the published
#: table; the rest add a second sequence, a DAG-RS-family arm and one degraded
#: condition so the delta is not measured on a single lucky clip.
DEFAULT_CELLS: List[Tuple[str, str, str]] = [
    ("bike-packing", "sam2video", "clean"),
    ("car-shadow", "sam2video", "clean"),
    ("gold-fish", "sam2video", "clean"),
    ("drift-chicane", "sam2video", "clean"),
    ("bike-packing", "dagrs", "clean"),
    ("bike-packing", "greedy", "clean"),
]

#: Extended set for `--all-defaults`.
EXTENDED_CELLS: List[Tuple[str, str, str]] = DEFAULT_CELLS + [
    ("bike-packing", "sam2video", "fog"),
    ("bike-packing", "sam2video", "dust"),
    ("bike-packing", "sam2video", "underwater"),
    ("car-shadow", "sam2video", "fog"),
    ("drift-chicane", "sam2video", "fog"),
    ("gold-fish", "sam2video", "fog"),
]

# --------------------------------------------------------------------------- #
# mask acquisition
# --------------------------------------------------------------------------- #

def cell_key(seq: str, mode: str, deg: str) -> str:
    return f"{seq}__{mode}__{deg}"


def get_masks(s: Dict, seq: str, mode: str, deg: str, obj_id: int,
              force: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Per-frame (preds, gts) for one cell, via the CSV's own code path.

    `Protocol` and `load_degraded_sequence` are constructed exactly as
    `run_sweep` does (`xdrp/benchmark.py:372`), so the degraded frames and the
    prompt mask are bit-identical to the ones behind `cl4_full_v2.csv`. The
    backend and the per-mode config come from `scripts/_common`, the single
    entry point for arm definitions.
    """
    from xdrp.benchmark import Protocol, load_degraded_sequence
    from xdrp.pipeline import run_sequence

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{cell_key(seq, mode, deg)}.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return list(z["preds"]), list(z["gts"])

    proto = Protocol(degradation=deg, level=float(s["level"]),
                     frame_stride=int(s["frame_stride"]),
                     max_frames=int(s["max_frames"]), seed=int(s["seed"]))
    dseq = load_degraded_sequence(s["src"], seq, proto, max_objects=0)
    if obj_id not in dseq.object_ids:
        # Frame 0's annotation decides which instances exist; a missing id means
        # the (sequence, object) pair does not exist, not that the run failed.
        raise KeyError(f"{seq}: object {obj_id} absent (available="
                       f"{dseq.object_ids})")

    backend = make_backend_for(s)
    cfg = config_for_mode(s["cfg"], mode)
    # `make_bank` is the same constructor `scripts/02/03/04` use, so the operator
    # bank here cannot silently drift from the one behind the CSV.
    bank = make_bank(s["cfg"])

    backend.clear_cache()
    frames = dseq.frames_for_mode(cfg.mode)
    res = run_sequence(frames, dseq.first_masks[obj_id], backend, bank, cfg,
                       seq_name=seq, obj_id=int(obj_id), flow_frames=frames)
    if int(res.n_predict_failures) > 0:
        raise RuntimeError(
            f"{seq}/{mode}/{deg}: {res.n_predict_failures} backend.predict call(s) "
            f"fell back to the warped anchor, so these masks are not a clean "
            f"segmentation measurement -- refusing to score them.")
    preds = [np.asarray(m, bool) for m in res.masks]
    gts = [np.asarray(m, bool) for m in dseq.gt_masks[obj_id]]
    if len(preds) != len(gts):
        raise RuntimeError(f"{seq}: {len(preds)} preds vs {len(gts)} gts -- "
                           f"scoring a length mismatch changes the denominator.")
    np.savez_compressed(cache, preds=np.stack(preds), gts=np.stack(gts))
    return preds, gts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_cell(text: str) -> Tuple[str, str, str]:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--cells entries must be seq:mode:degradation, got {text!r}")
    return parts[0], parts[1], parts[2]


def main() -> int:
    ap = base_parser(__doc__ or "eval protocol A/B")
    ap.add_argument("--cells", nargs="*", type=parse_cell, default=None,
                    help="seq:mode:degradation triples (default: a fixed 6-cell set)")
    ap.add_argument("--all-defaults", action="store_true",
                    help="use the 12-cell extended set")
    ap.add_argument("--obj-id", type=int, default=1,
                    help="which instance to score (default 1)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--force", action="store_true", help="ignore the mask cache")
    ap.add_argument("--no-released-check", action="store_true",
                    help="skip the cross-check against sam2.sav_dataset")
    args = ap.parse_args()

    global CACHE_DIR
    s = preset(args)
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)
    s["dilation"] = int(deep_get(s["cfg"], "protocol.dilation", 2) or 2)
    # `level`/`seed` must match the sweep that produced cl4_full_v2.csv.
    s["level"] = float(deep_get(s["cfg"], "protocol.level", 3.0))
    s["src"] = open_src(s)

    cells = (EXTENDED_CELLS if args.all_defaults else DEFAULT_CELLS)
    if args.cells:
        cells = list(args.cells)

    print(f"[eval-ab] {len(cells)} cell(s) backend={s['backend_kind']} "
          f"dilation={s['dilation']} level={s['level']} "
          f"stride={s['frame_stride']} max_frames={s['max_frames']}")
    print(f"[eval-ab] cache={CACHE_DIR}")

    rows: List[Dict] = []
    mismatches: List[str] = []
    for seq, mode, deg in cells:
        t0 = time.time()
        try:
            preds, gts = get_masks(s, seq, mode, deg, args.obj_id, force=args.force)
        except Exception as e:  # noqa: BLE001
            print(f"[eval-ab] SKIP {seq}/{mode}/{deg}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        row: Dict = {"seq": seq, "mode": mode, "degradation": deg,
                     "obj_id": args.obj_id, "n_frames": len(preds),
                     "gpu_seconds": round(time.time() - t0, 1)}

        ours = ours_seq_metrics(preds, gts, s["dilation"], False)
        ours_skip = ours_seq_metrics(preds, gts, s["dilation"], True)
        off = official_seq_metrics(preds, gts, False)
        off_skip = official_seq_metrics(preds, gts, True)
        for tag, m in (("ours", ours), ("ours_skip", ours_skip),
                       ("official", off), ("official_skip", off_skip)):
            row[f"{tag}_J"] = m["J"]
            row[f"{tag}_F"] = m["F"]
            row[f"{tag}_J&F"] = m["J&F"]

        if not args.no_released_check:
            for tag, skip in (("official", False), ("official_skip", True)):
                rel = released_check(preds, gts, skip)
                if rel is None:
                    break
                row[f"released_{tag}_J&F"] = rel["J&F"]
                if not np.isclose(rel["J&F"], row[f"{tag}_J&F"], atol=1e-9):
                    mismatches.append(
                        f"{seq}/{mode}/{deg} {tag}: port J&F "
                        f"{row[f'{tag}_J&F']:.6f} != released {rel['J&F']:.6f}")

        row["delta_official_minus_ours"] = off["J&F"] - ours["J&F"]
        row["delta_officialskip_minus_ourskip"] = off_skip["J&F"] - ours_skip["J&F"]
        row["delta_skip_ours"] = ours_skip["J&F"] - ours["J&F"]
        row["delta_skip_official"] = off_skip["J&F"] - off["J&F"]
        rows.append(row)
        print(f"[eval-ab] {seq:<16s} {mode:<10s} {deg:<11s} "
              f"ours={ours['J&F']:.4f} official={off['J&F']:.4f} "
              f"(d={row['delta_official_minus_ours']:+.4f})  "
              f"skip: ours={ours_skip['J&F']:.4f} official={off_skip['J&F']:.4f}")

    if mismatches:
        print("\n[eval-ab] FATAL: the ported scorer disagrees with the released "
              "sam2.sav_dataset code on:", file=sys.stderr)
        for m in mismatches:
            print(f"    {m}", file=sys.stderr)
        return 2

    if not rows:
        print("[eval-ab] no cells scored", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (ROOT / "results" / "eval_protocol_ab.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["seq", "mode", "degradation", "obj_id", "n_frames", "gpu_seconds",
            "ours_J", "ours_F", "ours_J&F", "ours_skip_J", "ours_skip_F",
            "ours_skip_J&F", "official_J", "official_F", "official_J&F",
            "official_skip_J", "official_skip_F", "official_skip_J&F",
            "released_official_J&F", "released_official_skip_J&F",
            "delta_official_minus_ours", "delta_officialskip_minus_ourskip",
            "delta_skip_ours", "delta_skip_official"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    a = np.array([r["delta_official_minus_ours"] for r in rows], float)
    b = np.array([r["delta_officialskip_minus_ourskip"] for r in rows], float)
    c = np.array([r["delta_skip_ours"] for r in rows], float)
    print(f"\n[eval-ab] ---- summary over {len(rows)} cell(s), object "
          f"{args.obj_id} (J&F units) ----")
    print(f"  official      - ours       : mean {a.mean():+.4f}  "
          f"min {a.min():+.4f}  max {a.max():+.4f}")
    print(f"  official_skip - ours_skip  : mean {b.mean():+.4f}  "
          f"min {b.min():+.4f}  max {b.max():+.4f}")
    print(f"  skip-first/last effect     : mean {c.mean():+.4f}  "
          f"min {c.min():+.4f}  max {c.max():+.4f}")
    print(f"  wrote {out}")

    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(
        {"cells": [list(x) for x in cells],
         "obj_id": args.obj_id, "dilation": s["dilation"],
         "level": s["level"], "seed": s["seed"],
         "frame_stride": s["frame_stride"], "max_frames": s["max_frames"],
         "n_cells": len(rows),
         "mean_delta_official_minus_ours": float(a.mean()),
         "mean_delta_officialskip_minus_ourskip": float(b.mean()),
         "mean_delta_skip_ours": float(c.mean()),
         "port_matches_released": not mismatches},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
