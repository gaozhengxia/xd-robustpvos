"""103 - How much does the video arm's hole-filling post-process actually matter?

    python scripts/103_video_postproc_ab.py [--seq bike-packing] [--frames 0]

The released SAM 2.1 *video* predictor builds with `fill_hole_area=8`
(`sam2/build_sam.py`), i.e. it fills small holes in the low-resolution mask
logits. That step calls into `sam2._C`, a CUDA extension this environment does not
build. `xdrp/video_backend.py` re-implements the operator with OpenCV so the arm
really is the released model -- but the CUDA kernel's CONNECTIVITY (4 vs 8) is not
visible in the Python source, so it is the one parameter we cannot read off.

This script measures that ambiguity instead of assuming it away. It runs the same
clip three times:

  off       post-process disabled (`fill_hole_area=0`)
  4-conn    re-implementation with 4-connectivity (the default in this repo)
  8-conn    re-implementation with 8-connectivity

and reports J&F for each. Two numbers matter: `4-conn` vs `off` (does the
post-process do anything at all, and does reproducing it change the baseline?)
and `4-conn` vs `8-conn` (how much does the unreadable parameter cost us?).

No dataset? Pass `--frames N` to run a synthetic clip instead; the point here is
the operator's effect on the mask geometry, not the score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xdrp.datasets import open_dataset                              # noqa: E402
from xdrp.metrics import seq_metrics                                # noqa: E402
from xdrp.sam_backend import SAM2Config                             # noqa: E402
from xdrp import video_backend as vb                                # noqa: E402


def _synthetic(n: int):
    """A moving rectangle with a small hole punched in it, on a noisy background.

    The hole is the whole point: a post-process that fills holes must change the
    mask on a clip that has one.
    """
    rng = np.random.default_rng(0)
    frames, gts = [], []
    for t in range(n):
        img = np.clip(rng.normal(0.35, 0.06, (160, 220, 3)), 0, 1).astype(np.float32)
        img[40:120, 30:180] = 0.75                       # bright rectangle
        img[70:80, 90:100] = 0.90                        # a "hole" that is NOT object
        gt = np.zeros((160, 220), bool)
        gt[40:120, 30:180] = True
        gt[70:80, 90:100] = False                        # the hole belongs to background
        frames.append(img)
        gts.append(gt)
    return frames, gts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bike-packing")
    ap.add_argument("--frames", type=int, default=0,
                    help="0 = use the real clip; >0 = synthetic clip of this length")
    ap.add_argument("--obj", type=int, default=1)
    ap.add_argument("--stride", type=int, default=4,
                    help="frame stride for the real dataset (the video predictor "
                         "is O(T); 4 keeps this probe to a couple of minutes)")
    args = ap.parse_args()

    if args.frames:
        frames, gts = _synthetic(int(args.frames))
        name = f"synthetic({len(frames)})"
    else:
        # Same construction path as the sweep (`load_degraded_sequence` with a
        # clean Protocol), so this probe cannot drift from the numbers it is
        # meant to qualify.
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
        name = (f"{args.seq}/obj{obj} stride={args.stride} "
                f"frames={len(frames)}")

    cfg = SAM2Config()
    predictor = vb.predictor_for(cfg, autocast_dtype=cfg.autocast_dtype)
    base_fill = int(getattr(predictor, "fill_hole_area", 0))

    results = {}
    for tag, fill, conn in (("off", 0, vb.POSTPROC_CONNECTIVITY),
                            ("4-conn", 8, 4),
                            ("8-conn", 8, 8)):
        predictor.fill_hole_area = int(fill)
        if fill > 0:
            vb.install_postprocessing_patch(connectivity=conn,
                                            device=str(cfg.device))
        else:
            # Restore SAM 2's own (falling-back) implementation so "off" really is
            # off rather than "our patch with nothing to do".
            import sam2.utils.misc as _misc
            orig = vb._POSTPROC_PATCHED.get("original")
            if orig is not None:
                _misc.get_connected_components = orig
            vb._POSTPROC_PATCHED["installed"] = False
        masks, info = vb.run_video_tracking(
            predictor, frames, gts[0], autocast_dtype=cfg.autocast_dtype)
        m = seq_metrics(masks, gts, dilation=2)
        px = float(np.mean([int(a.sum()) for a in masks]))
        results[tag] = (m["J&F"], px, info["n_encoder_calls"])
        print(f"  {tag:>7}: J&F={m['J&F']:.4f}  mean mask area={px:8.1f}  "
              f"backbone passes={info['n_encoder_calls']}")

    predictor.fill_hole_area = base_fill
    off, c4, c8 = (results["off"][0], results["4-conn"][0], results["8-conn"][0])
    print(f"\n{name}  predictor.fill_hole_area={base_fill}")
    print(f"  post-process ON vs OFF        : {c4 - off:+.4f} J&F")
    print(f"  4- vs 8-connectivity          : {c8 - c4:+.4f} J&F")
    print(f"  mask area ON(4-conn) vs OFF   : {results['4-conn'][1] - results['off'][1]:+.1f} px")
    print("\nReading: the first number says whether reproducing the post-process at "
          "all changes the baseline we publish. The second bounds the error from the "
          "parameter we could not read out of the CUDA kernel. Both belong in the "
          "appendix if this arm's numbers are published.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
