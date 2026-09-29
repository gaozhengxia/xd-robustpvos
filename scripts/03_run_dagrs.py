"""03 - Run DAG-RS (the proposed method) and its supporting arms.

    python scripts/03_run_dagrs.py --quick
    python scripts/03_run_dagrs.py --degradations compound --levels 1,2,3,4,5

  --ood : CANCELLED EXPERIMENT (2026-09-23) -- kept for audit only, do NOT run.
          The flag still works, but nothing consumes its output any more.  It was
          meant to answer "does the family gap vary with unfamiliarity", but the
          level 1-5 severity sweep already answers that with a stronger protocol
          (scripts/111_severity_table.py, PAPER_FRAMEWORK 5.6 / CL2), and --ood
          would merely sample INTERIOR points (2.5 / 4.5) of a range whose two
          ENDPOINTS are already flat.  Real cross-domain evidence is now the
          second VOS source (scripts/114_xsource_table.py, PAPER_FRAMEWORK 5.5b).
          Rationale + the Limitations wording it forces: PAPER_FRAMEWORK 5.5.1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (ROOT, base_parser, config_for_mode, make_backend_for,
                     make_bank, make_dagrs_config,
                     open_src, out_path, preset, resolve_degradations, resolve_levels)
from xdrp.benchmark import SweepConfig, run_sweep, write_csv
from xdrp.pipeline import make_config

#: Degradations held out for the OOD study (never used to set hyperparameters).
OOD_DEGRADATIONS = ["underwater", "dust", "C2_lowlight_blur", "C4_fog_blur_noise"]
OOD_LEVELS = [2.5, 4.5]


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--ood", action="store_true",
                    help="run the out-of-distribution transfer study")
    ap.add_argument("--modes", default="greedy,equal_tta,dagrs")
    args = ap.parse_args()
    s = preset(args)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    degs = OOD_DEGRADATIONS if args.ood else resolve_degradations(args.degradations, s["cfg"])
    levels = OOD_LEVELS if args.ood else resolve_levels(args.levels, s["cfg"])
    tag = "ood" if args.ood else "main"

    print(f"XD-RobustPVOS :: DAG-RS ({tag})")
    print("=" * 68)
    print(f"  modes      : {modes}")
    print(f"  degradations: {degs}")
    print(f"  levels     : {levels}")

    src = open_src(s)
    backend = make_backend_for(s)
    bank = make_bank(s["cfg"])

    cfgs = {}
    for m in modes:
        # the hand-written field list that used to live here included
        # `use_reanchor`, which silently turned `dagrs_no_reanchor` into a copy
        # of `dagrs`. The shared helper deliberately excludes arm-defining switches.
        cfgs[m] = config_for_mode(s["cfg"], m)

    out = out_path(s, args, f"dagrs_{tag}_raw.csv")
    sweep = SweepConfig(
        dataset_root=s["dataset_root"], dataset_kind=s["dataset_kind"],
        dataset_split=s["split"],
        degradations=degs, levels=levels, modes=modes,
        frame_stride=int(s["frame_stride"]),
        max_frames=int(s["max_frames"]),
        max_sequences=int(s["max_sequences"]),
        seq_offset=int(s["seq_offset"]),
        max_objects=int(s["max_objects"]),
        seed=int(s["seed"]), dilation=int(s["dilation"]),
        collect_iqa=bool(s["collect_iqa"]),
        # DAG-RS is the most expensive arm set in the project (J candidates at
        # every keyframe plus re-anchoring, times six modes times 30 sequences
        # times 40 cells), so it is exactly the sweep that cannot finish inside
        # one shell call. Without these two the whole run had to start over from
        # zero after any interruption -- the defect 02_run_baselines.py was
        # already fixed for.
        resume_file=out if args.resume else "",
        append_file=out if args.resume else "",
    )

    try:
        from tqdm import tqdm
        state = {"bar": None}

        def progress(step, total, msg):
            if state["bar"] is None:
                state["bar"] = tqdm(total=total, unit="cell")
            state["bar"].set_description(msg[:70])
            state["bar"].update(1)
    except Exception:  # noqa: BLE001
        def progress(step, total, msg):
            print(f"  [{step}/{total}] {msg}")

    rows = run_sweep(src, backend, bank, cfgs, sweep, progress=progress)
    if args.resume:
        # each finished cell was appended as it completed, so rewriting the file
        # with this invocation's rows only would throw away every earlier chunk
        p = out
        print(f"\nappended this run's cells -> {p} "
              f"({len(rows)} rows this invocation)")
    else:
        p = write_csv(rows, out)
    print(f"\nwrote {len(rows)} per-frame records -> {p}")

    # quick on-the-spot summary so you can see whether it worked immediately
    import numpy as np
    by = {}
    for r in rows:
        by.setdefault((r["degradation"], r["mode"]), []).append(float(r["J&F"]))
    header = f"{'degradation':<20}" + "".join(f"{m:>12}" for m in modes)
    print("\n" + header)
    print("-" * len(header))
    for deg in sorted({r["degradation"] for r in rows}):
        line = f"{deg:<20}"
        for m in modes:
            v = by.get((deg, m), [])
            line += f"{np.mean(v):>12.4f}" if v else f"{'-':>12}"
        print(line)
    print("\nNext: python scripts/05_analysis.py --study stats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
