"""04 - Ablations.

Runs one or more ablation arms (paper TABLE-13). Every arm reuses the exact same
inference code path as the full method and differs only in configuration flags,
so no arm can accidentally benefit from a different implementation.

    python scripts/04_ablation.py --list
    python scripts/04_ablation.py --variants A1,A3 --quick
    python scripts/04_ablation.py --variants all

Arm legend
  A0  full DAG-RS
  A1  equal-weight averaging over the SAME operator set  (isolates the agreement criterion)
  A2  time-consistency term removed from the agreement score
  A3  random operator choice per frame (same cost, no criterion)
  A4  single fixed domain-prior operator (no adaptivity)
  A5  no degradation profile (evaluate the full bank every decision frame)
  A6  no vacuity gating (always update the anchor)
  A7  no global re-anchoring
  A8  keyframe stride sweep        K in {1,3,5,10}
  A9  profile top-J sweep          J in {1,2,3,5,8,13}
  A10 operator-bank size sweep     M in {3,5,8,13}

The former A11 ("softmax instead of Dirichlet") was REMOVED on 2026-09-16. Its
mechanism was never implemented: the `_softmax` marker only decorated the printed
note, so the arm silently reproduced A0 and would have put a duplicate row in
TABLE-13. It was also dropped from docs/PAPER_FRAMEWORK.md. Selecting it now
raises KeyError instead of producing a fake measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (ROOT, base_parser, config_for_mode, make_backend_for,
                     make_bank, make_dagrs_config,
                     open_src, out_path, preset, resolve_degradations, resolve_levels)
from xdrp.benchmark import SweepConfig, run_sweep, write_csv
from xdrp.operators import OperatorBank, build_default_bank
from xdrp.pipeline import make_config

ARM_LEGEND = {
    "A0": ("full DAG-RS", "dagrs", {}),
    "A1": ("equal-weight fusion over the SAME operator set", "dagrs",
           {"fuse_uniform": True, "top_k_fuse": 99}),
    "A2": ("no time-consistency term (w1 = 0)", "dagrs", {"agreement_weights": (0.0, 0.64, 0.36)}),
    "A3": ("random operator per frame", "random_op", {}),
    "A4": ("single fixed domain-prior operator", "fixed_op", {}),
    "A5": ("no degradation profile (full bank every decision)",
           "dagrs_no_profile", {"j_cold": 0}),
    "A6": ("no vacuity gating", "dagrs_no_gating", {}),
    "A7": ("no global re-anchoring", "dagrs_no_reanchor", {}),
    "A8": ("keyframe stride sweep", "dagrs", {"_sweep": "keyframe_stride",
                                              "_values": [1, 3, 5, 10]}),
    "A9": ("profile top-J sweep", "dagrs", {"_sweep": "j_steady",
                                            "_values": [1, 2, 3, 5, 8, 13]}),
    "A10": ("operator-bank size sweep", "dagrs", {"_sweep": "_bank_size",
                                                 "_values": [3, 5, 8, 13]}),
}


#: Arms that are declared in the paper framework but whose mechanism is NOT
#: implemented in the pipeline. They stay in `ARM_LEGEND` so `--list` documents
#: them, but selecting one raises instead of silently measuring nothing.
#: Currently EMPTY: A11 used to be listed here (its `_softmax` marker never
#: reached the config, so it duplicated A0). It has since been deleted from both
#: the code and the paper; the guard mechanism is kept so the next arm of that
#: kind fails loudly instead of producing a fake measurement.
NOT_IMPLEMENTED_ARMS: set = set()


def build_arms(keys):
    """Return a list of (arm_key, mode, cfg, bank_or_None, note)."""
    arms = []
    for k in keys:
        if k not in ARM_LEGEND:
            raise KeyError(f"unknown variant '{k}'; choose from {list(ARM_LEGEND)}")
        note, mode, over = ARM_LEGEND[k]
        if k in NOT_IMPLEMENTED_ARMS:
            raise NotImplementedError(
                f"ablation arm {k} ({note}) is declared but not implemented: "
                f"selecting it would silently reproduce the full method and put a "
                f"duplicate row in the ablation table. Implement the mechanism "
                f"(or drop the arm from docs/PAPER_FRAMEWORK.md) first.")
        if "_sweep" in over:
            field = over["_sweep"]
            for v in over["_values"]:
                o = {kk: vv for kk, vv in over.items() if not kk.startswith("_")}
                if field != "_bank_size":
                    o[field] = v
                arms.append((f"{k}={v}", mode, o, (v if field == "_bank_size" else None),
                             f"{note}: {field}={v}"))
        else:
            o = {kk: vv for kk, vv in over.items() if not kk.startswith("_")}
            meta = set(over) - set(o)
            if meta:
                raise ValueError(
                    f"arm {k} carries metadata-only key(s) {sorted(meta)}: these "
                    f"never reach the config, which is exactly how A11 silently "
                    f"became a duplicate of A0. Implement the mechanism or drop "
                    f"the arm.")
            arms.append((k, mode, o, None, note))
    return arms


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--variants", default="A0,A1,A3,A5,A6,A7")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    s = preset(args)

    if args.list:
        print("Ablation arms:")
        for k, (note, mode, _) in ARM_LEGEND.items():
            flag = "  [NOT IMPLEMENTED]" if k in NOT_IMPLEMENTED_ARMS else ""
            print(f"  {k:<4} [{mode:<18}] {note}{flag}")
        return 0

    keys = [k for k in ARM_LEGEND if k not in NOT_IMPLEMENTED_ARMS] \
        if args.variants == "all" else \
        [x.strip() for x in args.variants.split(",") if x.strip()]

    print("XD-RobustPVOS :: ablations")
    print("=" * 68)
    src = open_src(s)
    backend = make_backend_for(s)
    default_bank = make_bank(s["cfg"])

    degs = resolve_degradations(args.degradations, s["cfg"])
    levels = resolve_levels(args.levels, s["cfg"])

    all_rows = []
    summary = []
    arms = build_arms(keys)
    for arm_key, mode, over, bank_size, note in arms:
        # The previous hand-written field list copied `use_reanchor` back from the
        # yaml, so arm A7 (`dagrs_no_reanchor`) was silently identical to A0.
        # `config_for_mode` excludes every arm-defining switch.
        cfg = config_for_mode(s["cfg"], mode, **over)
        if cfg.mode == "random_op":
            cfg.random_seed = int(s["seed"])

        bank = default_bank
        if bank_size is not None:
            ops = build_default_bank()[:int(bank_size)]
            if not any(o.name == "id" for o in ops):
                ops.insert(0, build_default_bank()[0])
            bank = OperatorBank(ops)

        print(f"\n[{arm_key}] {note}  (mode={mode}, M={len(bank)}, "
              f"K={cfg.keyframe_stride}, J={cfg.j_steady})")

        sweep = SweepConfig(
            dataset_root=s["dataset_root"], dataset_kind=s["dataset_kind"],
            dataset_split=s["split"],
            degradations=degs, levels=levels, modes=[cfg.mode],
            frame_stride=int(s["frame_stride"]), max_frames=int(s["max_frames"]),
            max_sequences=int(s["max_sequences"]),
            seq_offset=int(s["seq_offset"]),
            max_objects=int(s["max_objects"]),
            seed=int(s["seed"]), dilation=int(s["dilation"]),
            collect_iqa=False,
        )
        rows = run_sweep(src, backend, bank, {cfg.mode: cfg}, sweep,
                         progress=lambda a, b, m: None)
        for r in rows:
            r["arm"] = arm_key
            r["arm_note"] = note
            r["bank_size"] = len(bank)
        all_rows.extend(rows)
        if rows:
            summary.append((arm_key, float(np.mean([r["J&F"] for r in rows])),
                            float(np.mean([r["n_encoder_calls"] for r in rows])),
                            note))
            print(f"      J&F={summary[-1][1]:.4f}  encoder_calls={summary[-1][2]:.1f}")

    p = write_csv(all_rows, out_path(s, args, "ablation_raw.csv"))
    print(f"\nwrote {len(all_rows)} records -> {p}")

    print(f"\n{'arm':<10}{'J&F':>9}{'calls':>10}  note")
    print("-" * 70)
    for k, jf, calls, note in summary:
        print(f"{k:<10}{jf:>9.4f}{calls:>10.1f}  {note}")

    if summary:
        ref = next((jf for k, jf, _, _ in summary if k == "A0"), None)
        if ref:
            print(f"\ndelta vs A0 (full DAG-RS, J&F={ref:.4f}):")
            for k, jf, _, _ in summary:
                print(f"  {k:<10} {jf - ref:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
