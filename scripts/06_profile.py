"""06 - Cost and memory profiling (paper TABLE-7: Complexity analysis).

Measures encoder calls, wall-clock time and peak GPU memory for each mode on the
same sequence, so the accuracy/cost trade-off in the paper is measured, not
estimated.

    python scripts/06_profile.py --quick
    python scripts/06_profile.py --degradation fog --level 4 --seqs 3
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (ROOT, base_parser, config_for_mode, make_backend_for,
                     make_bank, make_dagrs_config,
                     open_src, out_path, preset)
from xdrp.benchmark import Protocol, evaluate_cell, load_degraded_sequence, write_csv
from xdrp.pipeline import make_config

MODES = ["greedy", "cascade", "equal_tta", "random_op", "dagrs"]


def gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def reset_gpu_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--degradation", default="fog")
    ap.add_argument("--level", type=float, default=4.0)
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args()
    s = preset(args)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print("XD-RobustPVOS :: cost profile")
    print("=" * 68)

    src = open_src(s)
    backend = make_backend_for(s)
    bank = make_bank(s["cfg"])

    seqs = src.sequences()
    n = args.seqs or len(seqs)
    rows = []
    for q in seqs[:n]:
        proto = Protocol(degradation=args.degradation, level=float(args.level),
                         frame_stride=int(s["frame_stride"]),
                         max_frames=int(s["max_frames"]), seed=int(s["seed"]))
        dseq = load_degraded_sequence(src, q, proto, max_objects=int(s["max_objects"]))
        for mode in modes:
            cfg = config_for_mode(s["cfg"], mode)
            backend.clear_cache()
            reset_gpu_peak()
            t0 = time.time()
            recs = evaluate_cell(dseq, backend, bank, cfg,
                                 dilation=int(s["dilation"]), collect_iqa=False,
                                 max_objects=int(s["max_objects"]))
            dt = time.time() - t0
            calls = float(np.mean([r["n_encoder_calls"] for r in recs])) if recs else 0.0
            jf = float(np.mean([r["J&F"] for r in recs])) if recs else float("nan")
            frames = float(np.mean([r["n_frames"] for r in recs])) if recs else 1.0
            rows.append({
                "seq": q, "mode": mode, "degradation": args.degradation,
                "level": float(args.level),
                "n_frames": frames,
                "encoder_calls": calls,
                "walls_s": dt,
                "fps": frames / dt if dt > 0 else float("nan"),
                "gpu_peak_mb": gpu_peak_mb(),
                "J&F": jf,
            })
            r = rows[-1]
            print(f"  {q:<22}{mode:<12} frames={frames:5.0f} calls={calls:7.1f} "
                  f"time={dt:7.2f}s fps={r['fps']:6.2f} peak={r['gpu_peak_mb']:8.1f}MB "
                  f"J&F={jf:.4f}")

    # relative cost vs greedy
    for r in rows:
        g = [x for x in rows if x["seq"] == r["seq"] and x["mode"] == "greedy"]
        r["rel_cost"] = (r["encoder_calls"] / g[0]["encoder_calls"]) if g and g[0]["encoder_calls"] else np.nan

    p = write_csv(rows, out_path(s, args, "table7_profile.csv"))
    print(f"\nwrote {len(rows)} rows -> {p}")

    print(f"\n{'mode':<14}{'calls':>9}{'rel':>8}{'fps':>9}{'peakMB':>10}{'J&F':>9}")
    print("-" * 60)
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        print(f"{mode:<14}{np.mean([r['encoder_calls'] for r in sub]):>9.1f}"
              f"{np.nanmean([r['rel_cost'] for r in sub]):>8.2f}"
              f"{np.nanmean([r['fps'] for r in sub]):>9.2f}"
              f"{np.nanmean([r['gpu_peak_mb'] for r in sub]):>10.1f}"
              f"{np.nanmean([r['J&F'] for r in sub]):>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
