"""91 - audit the optical-flow propagation stage.

Project-internal diagnostic (NOT a paper artefact, but it produces the evidence
for the propagation claim in the method section).

Why this exists: the propagation step of DAG-RS is `warp(mask, t -> t+1)` along
Farneback flow, and `warp_with_flow` used to add the flow where the forward warp
requires subtracting it. Nothing in the pipeline noticed, because a mask that is
pushed *away* from its object still looks like a mask. This tool measures the
stage in isolation so a sign or scale error cannot hide again.

It scores two warps against the next ground truth:

    warp(+f)   the library applied to the raw flow
    warp(-f)   the library applied to the negated flow

Only one of them can be the forward warp, and which one it is is a property of
the convention, not of the data -- so the tool reports the winner rather than
assuming it. As of 2026-09-17 the answer is `warp(+f)`.

Baseline, 48 pairs over 16 val sequences, float32 frames from
`DegradedSequence.frames_for_mode` (exactly the frames the pipeline sees):

    mean IoU  no warp     0.6282
    mean IoU  warp(+f)    0.8054      <- correct: forward in all 48/48 pairs
    mean IoU  warp(-f)    0.5166      <- what the code did: worse than nothing

Reference input quality is also reported: the flow is not weak (p95|flow| inside
the object is ~1.4x the object's true displacement), so a low `warp` score means
a convention or warp problem, not a Farneback tuning problem.

`scripts/00_smoke_test.py::test_flow` pins the same invariant on a synthetic
translation, so a regression fails the self-test instead of quietly halving the
baseline. Run this when the flow backend, its parameters, or the warp convention
changes.

    python scripts/91_flow_audit.py --pairs 3
    python scripts/91_flow_audit.py --pairs 3 --seqs drift-straight,bmx-trees
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import base_parser, open_src, preset                       # noqa: E402
from xdrp.benchmark import Protocol, load_degraded_sequence            # noqa: E402
from xdrp.evidence import as_bool                                      # noqa: E402
from xdrp.flow import compute_flow, to_gray, warp_with_flow            # noqa: E402
from xdrp.metrics import j_region                                      # noqa: E402


def main() -> int:
    ap = base_parser("optical-flow propagation audit")
    ap.add_argument("--pairs", type=int, default=3,
                    help="frame pairs to test per sequence")
    ap.add_argument("--seqs", default="",
                    help="comma-separated; default = every sequence in val.txt")
    a = ap.parse_args()
    s = preset(a)
    src = open_src(s)

    seqs = [x.strip() for x in a.seqs.split(",") if x.strip()]
    if not seqs:
        split = ROOT / "data" / "DAVIS" / "ImageSets" / "2017" / "val.txt"
        seqs = [x.strip() for x in split.read_text(encoding="utf-8").split()
                if x.strip()]

    print(f"{'seq':<18}{'t':>3}{'true move':>16}{'|f| p95 in gt':>15}"
          f"{'|f| img':>9}{'nowarp':>8}{'warp(+f)':>10}{'warp(-f)':>10}")
    acc = {"nowarp": [], "plus": [], "minus": []}
    ratio = []
    for seq in seqs:
        proto = Protocol(degradation="clean", level=1.0, frame_stride=1,
                         max_frames=a.pairs + 1,
                         severity_amp=float(s["severity_amp"]), seed=int(s["seed"]))
        dseq = load_degraded_sequence(src, seq, proto, max_objects=1)
        frames = dseq.frames_for_mode("greedy")
        obj = dseq.object_ids[0] if dseq.object_ids else 1
        gts = dseq.gt_masks[obj]
        if len(frames) < 2:
            continue

        # Guard the dtype trap: feeding a uint8 frame to `to_gray` used to
        # saturate it to a constant, and Farneback then returned an identically
        # zero field with no error at all.
        g = to_gray(frames[0])
        if int(g.max()) - int(g.min()) < 8:
            print(f"{seq:<18} SKIPPED: grayscale is (near) constant "
                  f"[{int(g.min())}, {int(g.max())}] -- frame dtype or to_gray")
            continue

        for t in range(len(frames) - 1):
            ma, mb = as_bool(gts[t]), as_bool(gts[t + 1])
            if ma.sum() == 0 or mb.sum() == 0:
                continue
            ca = np.array(np.where(ma)).mean(axis=1)
            cb = np.array(np.where(mb)).mean(axis=1)
            move = np.array([cb[1] - ca[1], cb[0] - ca[0]])
            f = compute_flow(frames[t], frames[t + 1])
            mag = np.sqrt((f ** 2).sum(-1))

            i0 = j_region(ma, mb)
            ip = j_region(warp_with_flow(ma, f), mb)
            im = j_region(warp_with_flow(ma, -f), mb)
            acc["nowarp"].append(i0)
            acc["plus"].append(ip)
            acc["minus"].append(im)
            gt_mag = float(np.percentile(mag[ma], 95)) if ma.any() else 0.0
            ratio.append(gt_mag / max(float(np.linalg.norm(move)), 1e-6))
            print(f"{seq:<18}{t:>3}{f'({move[0]:+.1f},{move[1]:+.1f})':>16}"
                  f"{gt_mag:>15.2f}{mag.mean():>9.2f}{i0:>8.3f}{ip:>10.3f}{im:>10.3f}")

    n = len(acc["nowarp"])
    if not n:
        print("\nno usable pairs")
        return 0
    print(f"\npairs: {n}")
    for k in ("nowarp", "plus", "minus"):
        print(f"  mean IoU {k:<7} = {np.mean(acc[k]):.4f}")
    wins = sum(1 for p, m in zip(acc["plus"], acc["minus"]) if p > m + 1e-9)
    same = sum(1 for p, m in zip(acc["plus"], acc["minus"]) if abs(p - m) <= 1e-9)
    print(f"  warp(+f) strictly better than warp(-f): {wins}/{n} (ties {same})")
    print(f"  delta vs no-warp:  warp(+f) "
          f"{np.mean(acc['plus']) - np.mean(acc['nowarp']):+.4f}   warp(-f) "
          f"{np.mean(acc['minus']) - np.mean(acc['nowarp']):+.4f}")
    if wins < n:
        print("  [WARN] warp(+f) does not win unanimously. Either the warp "
              "convention in xdrp/flow.py changed, or the frame dtype reaching "
              "to_gray did. Do not trust any propagation number until resolved.")
    if np.mean(acc["plus"]) <= np.mean(acc["nowarp"]):
        print("  [WARN] warping is no better than not warping: propagation "
              "carries no signal on this data.")
    r = np.array(ratio)
    print(f"  p95|flow| in gt / |true displacement|: median={np.median(r):.2f}"
          f"  (<<1 means the flow itself is too weak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
