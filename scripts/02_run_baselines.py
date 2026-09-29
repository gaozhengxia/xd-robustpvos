"""02 - Run the baselines (paper TABLE-9 / TABLE-10 baseline columns).

Produces one CSV of per-frame records, which 05_analysis.py turns into the
sequence-level tables plus the error bars and significance tests.

    python scripts/02_run_baselines.py --quick
    python scripts/02_run_baselines.py --degradations single --levels 1,2,3,4,5
    python scripts/02_run_baselines.py --degradations compound --out results/baselines_compound.csv

Baselines run:
  greedy        SAM2 warp-and-reprompt propagation with UNCONDITIONAL acceptance.
                Kept as a reference point; on clean DAVIS val its closed loop is
                self-destructive (13.1 J&F on bmx-trees), so it is NOT the
                number to quote as "SAM 2.1" -- see docs/BASELINE_DIAGNOSIS.md 9.
                The three arms below are the same loop with a standard drift
                correction, and one of them is the honest baseline.
  greedy_gate   refuse a propagated mask whose predicted IoU is below SAM 2's own
                default (0.88) and emit the flow-propagated anchor instead
  greedy_reseed re-prompt from the annotated first frame every 20 frames
  greedy_robust both corrections together
  cascade       always apply one fixed restoration (DCP dehazing) then propagate
  fixed_op      always apply one domain-prior operator (gamma_hi) then propagate
  equal_tta     evaluate ALL operators every frame, fuse with EQUAL weights
  random_op     evaluate ONE randomly chosen operator per frame
  oracle_clean  fed the CLEAN frames (upper-bound reference, not a competitor)
  sam2video     SAM 2.1's released WHOLE-VIDEO predictor (memory bank). Not a
                per-frame loop at all: it consumes the whole clip at once, so it
                is the only arm here that carries state across frames beyond a
                binary mask. It is the comparator the paper's central hypothesis
                ("downstream gating can replace a memory bank") was missing --
                see docs/BASELINE_DIAGNOSIS.md and xdrp/video_backend.py.

None of the propagation arms re-anchor on evidence: that is DAG-RS's mechanism,
and a baseline that borrows it stops being a baseline. `sam2video` carries none
of the DAG-RS switches either (it is registered as a whole-video arm, and
`run_sequence` refuses it if any mechanism switch is left on).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (ROOT, base_parser, config_for_mode, make_backend_for,
                     make_bank, open_src, out_path, preset,
                     resolve_degradations, resolve_levels)
from xdrp.benchmark import SweepConfig, run_sweep, write_csv
from xdrp.pipeline import make_config

MODES = ["greedy", "greedy_gate", "greedy_reseed", "greedy_robust",
         "cascade", "fixed_op", "equal_tta", "random_op", "oracle_clean",
         "sam2video"]


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--modes", default=None,
                    help="comma-separated subset of: " + ",".join(MODES))
    args = ap.parse_args()
    s = preset(args)

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else MODES

    print("XD-RobustPVOS :: baselines")
    print("=" * 68)
    src = open_src(s)
    backend = make_backend_for(s)
    bank = make_bank(s["cfg"])

    cfgs = {}
    for m in modes:
        # yaml `dagrs:` tunables (incl. mask_prompt) applied via the shared helper,
        # so this script cannot drift from configs/default.yaml
        cfgs[m] = config_for_mode(s["cfg"], m)

    out = out_path(s, args, "baselines_raw.csv")
    sweep = SweepConfig(
        dataset_root=s["dataset_root"], dataset_kind=s["dataset_kind"],
        dataset_split=s["split"],
        degradations=resolve_degradations(args.degradations, s["cfg"]),
        levels=resolve_levels(args.levels, s["cfg"]),
        modes=modes,
        frame_stride=int(s["frame_stride"]),
        max_frames=int(s["max_frames"]),
        max_sequences=int(s["max_sequences"]),
        seq_offset=int(s["seq_offset"]),
        max_objects=int(s["max_objects"]),
        seed=int(s["seed"]),
        dilation=int(s["dilation"]),
        collect_iqa=bool(s["collect_iqa"]),
        resume_file=out if args.resume else "",
        append_file=out if args.resume else "",
    )

    try:
        from tqdm import tqdm
        bar = {"n": 0}

        def progress(step, total, msg):
            if bar["n"] == 0:
                bar["pbar"] = tqdm(total=total, unit="cell")
            bar["pbar"].set_description(msg[:70])
            bar["pbar"].update(1)
            bar["n"] += 1
    except Exception:  # noqa: BLE001
        def progress(step, total, msg):
            print(f"  [{step}/{total}] {msg}")

    rows = run_sweep(src, backend, bank, cfgs, sweep, progress=progress)
    if args.resume:
        # cells were appended as they finished; do NOT rewrite the file with only
        # the rows produced by this invocation
        p = out
        print(f"\nappended this run's cells -> {p} "
              f"({len(rows)} rows this invocation)")
    else:
        p = write_csv(rows, out)
        print(f"\nwrote {len(rows)} per-frame records -> {p}")
    print("\nNext: python scripts/05_analysis.py --study iqa   "
          "(THIS validates the paper's central claim before you run DAG-RS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
