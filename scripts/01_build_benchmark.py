"""01 - Build (and optionally materialise) the XD-RobustPVOS benchmark.

Two things happen here:
  1. a sanity report on the dataset and the degradation protocol
  2. optional materialisation of degraded frames to disk

Materialising is RECOMMENDED for the camera-ready run: it makes every later
experiment byte-reproducible and independent of OpenCV version differences.

    # inspect only
    python scripts/01_build_benchmark.py --quick

    # materialise Tier-1 single degradations at 5 severities
    python scripts/01_build_benchmark.py --degradations single --levels 1,2,3,4,5 \
        --frame-stride 4 --materialize data/XD-RobustPVOS

    # materialise the compound set (the paper's headline setting)
    python scripts/01_build_benchmark.py --degradations compound --levels 1,2,3,4,5 \
        --materialize data/XD-RobustPVOS
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (ROOT, base_parser, deep_get, load_yaml, make_bank, open_src,
                     out_path, preset, resolve_degradations, resolve_levels)
from xdrp.benchmark import (Protocol, load_degraded_sequence, materialize_sequence,
                            write_csv)
from xdrp.degradations import (ALL_DEGRADATION_NAMES, COMPOUND_DEGRADATIONS,
                              DEGRADATIONS, is_compound, members)
from xdrp.iqa import compute_iqa


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--materialize", default=None,
                    help="write degraded frames under this directory")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    args = ap.parse_args()
    s = preset(args)

    print("XD-RobustPVOS benchmark builder")
    print("=" * 68)

    # ---- operator bank table (TABLE-5) ------------------------------------ #
    bank = make_bank(s["cfg"])
    print(f"\nOperator bank: M = {len(bank)}")
    print(f"  {'idx':>3}  {'name':<12} {'family':<10} {'target':<26} cost")
    for row in bank.table():
        print(f"  {row['idx']:>3}  {row['name']:<12} {row['family']:<10} "
              f"{row['target']:<26} {row['cost']}")
    print(f"  families: {bank.family_map()}")

    # ---- degradation table (TABLE-1 / TABLE-2) ---------------------------- #
    d = resolve_degradations(args.degradations, s["cfg"])
    lv = resolve_levels(args.levels, s["cfg"])
    print(f"\nDegradations ({len(d)}): {d}")
    print(f"Severities  ({len(lv)}): {lv}")
    print(f"Cells per sequence: {len(d) * len(lv)}")

    # ---- dataset ------------------------------------------------------------ #
    print("\nDataset")
    src = None
    try:
        src = open_src(s)
        desc = src.describe()
        seqs = src.sequences()
        print(f"  kind={s['dataset_kind']} root={s['dataset_root']} split={s['split']}")
        print(f"  sequences={desc['n_sequences']} frames={desc['n_frames']}")
        print(f"  first 5: {seqs[:5]}")
        n_obj = [len(src.object_ids(q)) for q in seqs[:10]]
        print(f"  objects per sequence (first 10): {n_obj}")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] dataset unavailable: {type(e).__name__}: {e}")
        print("  The protocol description below is still valid; download DAVIS-2017 to")
        print("  <root>/JPEGImages/480p and <root>/Annotations/480p and re-run.")

    # ---- protocol report --------------------------------------------------- #
    print("\nProtocol")
    print(f"  frame_stride={s['frame_stride']} max_frames={s['max_frames'] or 'all'} "
          f"max_sequences={s['max_sequences'] or 'all'} "
          f"max_objects={s['max_objects'] or 'all'}")
    print(f"  severity_amp={s['severity_amp']} (temporal variation) seed={s['seed']}")
    print(f"  dilation={s['dilation']}px for boundary F")

    if args.materialize and src is not None:
        out_root = args.materialize
        print(f"\nMaterialising to {out_root} ...")
        seqs = src.sequences()
        if s["max_sequences"]:
            seqs = seqs[:int(s["max_sequences"])]
        made, iqa_rows = 0, []
        for q in seqs:
            for deg in d:
                for level in lv:
                    proto = Protocol(degradation=deg, level=float(level),
                                     frame_stride=int(s["frame_stride"]),
                                     max_frames=int(s["max_frames"]),
                                     severity_amp=float(s["severity_amp"]),
                                     seed=int(s["seed"]))
                    dseq = load_degraded_sequence(src, q, proto,
                                                  max_objects=int(s["max_objects"]))
                    materialize_sequence(dseq, out_root, obj_ids=dseq.object_ids)
                    made += 1
                    if s["collect_iqa"]:
                        for t in range(dseq.n_frames):
                            row = {"seq": q, "degradation": deg, "level": float(level),
                                   "frame": t, "severity": float(dseq.severity[t]),
                                   "is_compound": int(is_compound(deg))}
                            row.update(compute_iqa(dseq.degraded[t], clean=dseq.clean[t]))
                            iqa_rows.append(row)
                    print(f"  [{made}] {q} | {proto.key()} | {dseq.n_frames} frames")
        print(f"  wrote {made} sequence cells")
        if iqa_rows:
            p = write_csv(iqa_rows, str(ROOT / s["results_dir"] / "benchmark_iqa.csv"))
            print(f"  IQA table -> {p}")

        # append to the shared IQA report used by the CL3 study
        idx_path = ROOT / s["results_dir"] / "benchmark_index.json"
        index = json.loads(idx_path.read_text()) if idx_path.exists() else {"cells": []}
        for deg in d:
            for level in lv:
                index["cells"].append({"degradation": deg, "level": float(level),
                                       "is_compound": int(is_compound(deg)),
                                       "members": members(deg)})
        idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(f"  index -> {idx_path}")

    if args.report:
        rep = {
            "operators": bank.table(),
            "degradations": d,
            "levels": lv,
            "is_compound": {k: is_compound(k) for k in d},
            "compound_members": {k: members(k) for k in d},
            "protocol": {"frame_stride": s["frame_stride"], "max_frames": s["max_frames"],
                         "severity_amp": s["severity_amp"], "seed": s["seed"],
                         "dilation": s["dilation"]},
        }
        Path(args.report).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
