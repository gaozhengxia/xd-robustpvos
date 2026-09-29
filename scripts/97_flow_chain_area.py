"""97 - does CHAINED optical-flow propagation inflate a mask on its own?

Project-internal diagnostic. It needs no SAM 2 and no GPU: pure flow propagation
is deterministic, so this isolates the geometric half of the baseline's failure
from every segmentation question.

Why it matters
--------------
When the acceptance gate refuses a frame, the arm emits `FlowCache.warp_mask`,
i.e. the first frame's mask chained through `t` consecutive warps. On bmx-trees
that propagated anchor alone reached 79x the ground-truth area (see the
`gate_88` cell of scripts/96_ab_accept.py). If chaining inflates a mask by itself,
then part of what looked like "the closed loop lost the object" is a property of
the propagation primitive, not of the segmenter -- and any baseline built on that
primitive is damaged before the model is even asked.

`warp_with_flow` samples with INTER_NEAREST, so plain interpolation cannot be
the cause. Chaining can still spread the support: nearest-neighbour sampling
scatters each boundary pixel according to the local flow error, and the errors
accumulate over the chain. This script measures whether that happens, and how
fast -- and it validates the probe first on a synthetic PURE TRANSLATION, where
the correct answer (area preserved) is known in advance.

Run from the PROJECT ROOT:

    python scripts/97_flow_chain_area.py --seq bmx-trees,libby,blackswan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ROOT, base_parser, open_src, preset            # noqa: E402
from xdrp.benchmark import Protocol, load_degraded_sequence         # noqa: E402
from xdrp.flow import FlowCache, compute_flow, warp_with_flow       # noqa: E402

#: a chain that multiplies the mask area by more than this has left the object
RUNAWAY_RATIO = 3.0

#: the synthetic control's known content velocity (pixels per frame)
SYN_DX, SYN_DY, SYN_N = 2.0, 1.0, 24


def _synth_translation(n: int = SYN_N, h: int = 160, w: int = 208,
                       dx: float = SYN_DX, dy: float = SYN_DY):
    """A textured frame sliding at a CONSTANT velocity.

    Ground truth for the probe: the mask is translated by the same constant, so a
    correct flow + a correct warp must preserve its area to within the
    discretisation. Anything else means the probe (or the warp) is broken, and a
    negative result downstream would be worthless.
    """
    rng = np.random.default_rng(0)
    # The scratch buffer must be large enough for the LAST displacement, or the
    # crops silently come out short and every frame has a different width.
    pad_x, pad_y = int(round(dx * (n - 1))) + 2, int(round(dy * (n - 1))) + 2
    base = rng.integers(0, 255, (h + pad_y, w + pad_x), dtype=np.uint8)
    base = base.astype(np.float32) / 255.0
    frames, masks = [], []
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(n):
        ox, oy = int(round(dx * t)), int(round(dy * t))
        frames.append(base[oy:oy + h, ox:ox + w].copy())
        cx, cy, r = 0.5 * w, 0.5 * h, 0.18 * min(h, w)
        masks.append(((xx - (cx + dx * t)) ** 2 + (yy - (cy + dy * t)) ** 2)
                     <= r * r)
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise AssertionError(f"the synthetic frames are not all the same size: "
                             f"{shapes}")
    return frames, masks


def validate_probe() -> bool:
    """Known-positive control: pure translation must preserve area."""
    frames, masks = _synth_translation()
    flows = FlowCache(frames)
    a0 = int(masks[0].sum())
    worst = 0.0
    err = []
    for t in range(1, len(frames)):
        m = flows.warp_mask(masks[0], 0, t)
        err.append(abs(int(m.sum()) - int(masks[t].sum())) / max(a0, 1))
        worst = max(worst, abs(int(m.sum()) - a0) / max(a0, 1))
        # the flow itself should be ~the known displacement, with the pipeline's
        # convention `prev(x) ~= cur(x + flow(x))`: content moving by +dx means
        # flow = -dx. Check x and y separately -- averaging them hides an error
        # in one axis behind a correct value in the other.
        if t == 1:
            fl = compute_flow(frames[0], frames[1])
            med = np.median(fl.reshape(-1, 2), axis=0)
            if (abs(med[0] + SYN_DX) > 0.25
                    or abs(med[1] + SYN_DY) > 0.25):
                print(f"    [WARN] median flow on a known "
                      f"({SYN_DX:+.0f},{SYN_DY:+.0f}) content shift = "
                      f"({med[0]:+.2f},{med[1]:+.2f}); "
                      f"expected ({-SYN_DX:+.2f},{-SYN_DY:+.2f})")
    print(f"    area error vs the translated ground truth: "
          f"mean {100 * float(np.mean(err)):.2f}%, max {100 * worst:.2f}%")
    return worst < 0.05


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--seq", default=None,
                    help="comma-separated sequences; default = first N of split")
    args = ap.parse_args()
    s = preset(args)

    print("=" * 88)
    print("XD-RobustPVOS :: does chained flow propagation inflate a mask by itself?")
    print("=" * 88)
    print("\nPROBE SELF-VALIDATION (synthetic pure translation; known answer: "
          "area preserved)")
    ok = validate_probe()
    print(f"    -> {'ok' if ok else 'BROKEN'}")
    if not ok:
        print("\n[STOP] the probe does not reproduce a known-correct answer. "
              "Nothing below is interpretable.")
        return 1

    src = open_src(s)
    seqs = ([x.strip() for x in args.seq.split(",") if x.strip()] if args.seq
            else src.sequences()[:max(1, int(s["max_sequences"] or 6))])

    print("\n" + "-" * 88)
    print("CHAINED PROPAGATION OF THE FIRST-FRAME MASK (no segmenter involved)")
    print("-" * 88)
    print("  AREA alone is not enough: a mask can be the right size in the wrong")
    print("  place, and that is exactly what a re-seeding prompt must not be. IoU")
    print("  against the current ground truth is the positional check.")
    hdr = (f"{'seq':<20}{'n':>5}{'a1':>7}{'a20':>7}{'a_max':>7}"
           f"{'iou1':>7}{'iou10':>7}{'iou20':>7}{'iou_last':>9}{'iou_min':>8}")
    print(hdr)
    print("-" * len(hdr))

    worst_area: Dict[str, float] = {}
    worst_iou: Dict[str, float] = {}
    checked = False
    for seq in seqs:
        proto = Protocol(degradation="clean", level=1.0, frame_stride=1,
                         max_frames=0, severity_amp=1.0, seed=0)
        dseq = load_degraded_sequence(src, seq, proto, max_objects=0)
        if not dseq.object_ids:
            print(f"{seq:<20}  no frame-0 annotation -- skipped")
            continue
        obj = int(dseq.object_ids[0])
        gts = dseq.gt_masks[obj]
        frames = list(dseq.clean)
        flows = FlowCache(frames)

        # Incremental chaining: `warp_mask(m0, 0, t)` re-walks the whole chain
        # every call, which is O(T^2) and cost minutes on 100-frame sequences.
        # Applying one flow step per frame is the same sequence of warps and
        # therefore produces identical masks -- which is checked once, below,
        # rather than assumed.
        if not checked:
            tt = min(20, len(frames) - 1)
            if tt >= 1:
                slow = flows.warp_mask(gts[0].astype(bool), 0, tt)
                fast = gts[0].astype(bool).astype(np.uint8)
                for k in range(tt):
                    fast = warp_with_flow(fast, flows.flow_between(k)).astype(np.uint8)
                same = bool(np.array_equal(slow, fast > 0))
                print(f"    [check] incremental chain == warp_mask(t=0->{tt}): "
                      f"{same}")
                if not same:
                    print("[STOP] the incremental chain is not equivalent to "
                          "warp_mask; the numbers below are from a different "
                          "operator than the pipeline uses.")
                    return 1
                checked = True

        ratios: List[float] = []
        ious: List[float] = []
        cur = gts[0].astype(bool).astype(np.uint8)
        for t in range(1, len(frames)):
            cur = warp_with_flow(cur, flows.flow_between(t - 1)).astype(np.uint8)
            m = cur > 0
            gt = gts[t].astype(bool)
            gt_area = int(gt.sum())
            ratios.append(int(m.sum()) / gt_area if gt_area else float("nan"))
            inter = int((m & gt).sum())
            union = int((m | gt).sum())
            ious.append(inter / union if union else float("nan"))
        if not ratios:
            continue

        def at(v: List[float], i: int) -> float:
            return v[i] if i < len(v) else float("nan")

        amx = float(np.nanmax(ratios))
        imin = float(np.nanmin(ious))
        worst_area[seq] = amx
        worst_iou[seq] = imin
        print(f"{seq:<20}{len(ratios) + 1:>5}{at(ratios, 0):>7.2f}"
              f"{at(ratios, 19):>7.2f}{amx:>7.2f}"
              f"{at(ious, 0):>7.3f}{at(ious, 9):>7.3f}{at(ious, 19):>7.3f}"
              f"{at(ious, len(ious) - 1):>9.3f}{imin:>8.3f}")

    if worst_area:
        n_bad = sum(1 for v in worst_area.values() if v > RUNAWAY_RATIO)
        print(f"\n  {n_bad}/{len(worst_area)} sequences inflate past "
              f"{RUNAWAY_RATIO}x from chained warping ALONE.")
        print("  A column here above the threshold means the emitted mask would be")
        print("  wrong even if the segmenter were perfect: the propagation primitive")
        print("  is losing the object on its own, and the gate's fallback output")
        print("  (the warped anchor) cannot be trusted either.")
        bad_iou = sorted(s for s, v in worst_iou.items() if v < 0.5)
        print(f"\n  sequences whose chained prompt ever drops below IoU 0.5 "
              f"({len(bad_iou)}/{len(worst_iou)}): {bad_iou}")
        print("  On those frames, re-prompting from the first frame through flow")
        print("  alone hands the segmenter a mask in the WRONG PLACE, so a periodic")
        print("  re-seed built on it injects an error rather than correcting one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
