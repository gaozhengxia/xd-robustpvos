"""104 - Does `offload_state_to_cpu` actually buy the VRAM it is turned on for?

    python scripts/104_video_offload_ab.py [--seq cows] [--obj 1] [--stride 1]

`xdrp/video_backend.py` turns SAM 2's `offload_state_to_cpu` ON (module constant
`OFFLOAD_STATE_TO_CPU`) so that two concurrent full-protocol workers fit inside
16 GB of VRAM. That decision rested on an argument -- "the memory bank is what
grows with the clip" -- plus two readings that did NOT agree with each other: one
process on 51-frame clips sampled at 8767 MiB, while two processes on 84- and
50-frame clips had earlier sampled 15.6 GB together (~7.8 GiB each). Neither
reading was controlled for clip length, so neither can settle the question.

This probe removes both problems. It runs the SAME clip twice inside ONE process
and reads the allocator's own high-water mark (`torch.cuda.max_memory_allocated`)
instead of sampling a moving target from outside:

  off   OFFLOAD_STATE_TO_CPU = False   memory bank resident on the GPU
  on    OFFLOAD_STATE_TO_CPU = True    the shipped setting

The two J&F values must be BIT-IDENTICAL: the switch moves where tensors live, it
does not change what is computed. A difference here means the setting is not safe
to publish baselines with, whatever it saves.

The default clip is the longest one in the val split, which is the worst case for
a memory bank whose footprint grows with the number of frames. Run order is
off-then-on; if the gap is only a few hundred MiB that ordering could matter, so
anything under ~1 GiB should be re-run with the legs swapped before being quoted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xdrp.datasets import open_dataset                              # noqa: E402
from xdrp.metrics import seq_metrics                                # noqa: E402
from xdrp.sam_backend import SAM2Config                             # noqa: E402
from xdrp import video_backend as vb                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="cows",
                    help="default is the longest clip in the val split "
                         "(104 frames) -- the worst case for the memory bank")
    ap.add_argument("--obj", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1,
                    help="must stay 1 to match the CL4 protocol; a larger stride "
                         "shortens the clip and with it the very quantity measured")
    ap.add_argument("--swap", action="store_true",
                    help="run the legs on-then-off instead of off-then-on. Two "
                         "identical-clip runs of this probe have already disagreed "
                         "(7044 vs 16340 MiB peak), so the ordering is part of what "
                         "is under test, not a detail")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("this probe measures GPU memory; no CUDA device is available")
        return 1

    from xdrp.benchmark import Protocol, load_degraded_sequence
    src = open_dataset("davis", "data/DAVIS", split="val")
    if args.seq not in src.sequences():
        print(f"sequence {args.seq!r} not found in {src.name} val split")
        return 1
    proto = Protocol(degradation="clean", level=1.0,
                     frame_stride=int(args.stride), max_frames=0)
    dseq = load_degraded_sequence(src, args.seq, proto, max_objects=0)
    obj = int(args.obj)
    frames = dseq.frames_for_mode("sam2video")
    gts = dseq.gt_masks[obj]
    print(f"{args.seq}/obj{obj} stride={args.stride} frames={len(frames)}")

    cfg = SAM2Config()
    predictor = vb.predictor_for(cfg, autocast_dtype=cfg.autocast_dtype)
    print(f"predictor image_size={int(predictor.image_size)} "
          f"fill_hole_area={int(getattr(predictor, 'fill_hole_area', 0))}")
    floor = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"model-only floor (weights, before any clip): {floor:.1f} MiB")
    shipped = bool(vb.OFFLOAD_STATE_TO_CPU)

    # The legs below drive the switch by rebinding the module constant, which is
    # the same path production code takes (the argument defaults to None). That
    # indirection is exactly where a probe can fool itself: if the rebinding did
    # nothing, both legs would run the same setting and the "no saving" verdict
    # would be a property of the probe, not of the setting. So prove the toggle
    # flips before trusting a negative result -- on a 2-frame slice, which costs
    # nothing.
    for flag in (True, False):
        vb.OFFLOAD_STATE_TO_CPU = flag
        st = vb.build_inference_state(predictor, frames[:2])
        got_cpu = st["storage_device"].type == "cpu"
        if got_cpu != flag:
            vb.OFFLOAD_STATE_TO_CPU = shipped
            print(f"probe self-check FAILED: OFFLOAD_STATE_TO_CPU={flag} gave "
                  f"storage_device={st['storage_device']}; the switch is not live, "
                  f"so any measurement below would be meaningless")
            return 1
    print("probe self-check: the offload switch flips storage_device as intended")
    vb.OFFLOAD_STATE_TO_CPU = shipped

    results = {}
    order = (("on", True), ("off", False)) if args.swap else (("off", False), ("on", True))
    for tag, flag in order:
        vb.OFFLOAD_STATE_TO_CPU = flag
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        masks, info = vb.run_video_tracking(
            predictor, frames, gts[0], autocast_dtype=cfg.autocast_dtype)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        m = seq_metrics(masks, gts, dilation=2)
        results[tag] = (m["J&F"], peak, reserved, info["n_encoder_calls"])
        print(f"  offload {tag:>3}: J&F={m['J&F']:.6f}  peak VRAM={peak:8.1f} MiB  "
              f"peak reserved={reserved:8.1f} MiB  backbone passes={info['n_encoder_calls']}")

    vb.OFFLOAD_STATE_TO_CPU = shipped

    off, on = results["off"], results["on"]
    saved = off[1] - on[1]
    identical = off[0] == on[0]
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"\nshipped OFFLOAD_STATE_TO_CPU={shipped}  device total={total:.0f} MiB  "
          f"legs ran {'on,off' if args.swap else 'off,on'}")
    print(f"  peak VRAM saved by the switch : {saved:+.1f} MiB "
          f"({saved / off[1] * 100:+.1f}% of {off[1]:.0f} MiB)")
    print(f"  J&F off vs on                 : {off[0]:.17f} vs {on[0]:.17f} "
          f"-> {'IDENTICAL' if identical else 'DIFFERENT'}")
    over = max(off[2], on[2]) - total
    if over > 0:
        print(f"  !! peak reserved exceeds the device by {over:.0f} MiB: the run "
              f"spilled into host memory (Windows CUDA sysmem fallback). Host RAM "
              f"is being spent on GPU overcommit, which is invisible to a VRAM "
              f"reading and to that process's own allocator.")
    print(f"  per-worker peak, shipped setting: {on[1]:.0f} MiB allocated / "
          f"{on[2]:.0f} MiB reserved")
    if not identical:
        print("\nReading: the switch CHANGED the score, so it is not a pure device "
              "placement in this environment and must not be shipped silently.")
    elif saved < 1024:
        print("\nReading: the switch buys nothing measurable (well under 1 GiB) while "
              "hiding the memory bank in host RAM, so it is NOT the VRAM fix it was "
              "turned on for; its value here is only that the numbers stay identical.")
    else:
        print("\nReading: the switch is numerically neutral and worth the VRAM.")
    print("\nThis number belongs in the appendix: the baseline arm's memory "
          "footprint is part of what makes it comparable or not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
