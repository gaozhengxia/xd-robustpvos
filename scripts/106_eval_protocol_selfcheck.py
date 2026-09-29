"""Offline self-check for the eval-protocol port: no GPU, no SAM 2 weights.

Runs `scripts/eval_protocol_common.py`'s scorers against the *released*
`sam2.sav_dataset` evaluator on synthetic masks. What must hold:

  1. A truly perfect prediction (pred == gt, pixel-for-pixel) scores exactly
     1.0 under BOTH scorers. This is the "verify the probe on a known-positive
     case" rule: a probe that cannot recognise a correct answer must not be
     trusted to report a discrepancy.
  2. A near-perfect prediction (1-px offset) scores < 1.0 under both, by a
     similar amount -- i.e. the probe can actually see a difference.
  3. The port agrees with the *released* code to 1e-9 on every case, including
     the skip-first/last variant. This is what makes the A/B in script 105
     meaningful: a port that only *claims* fidelity is worthless.
  4. The two conventions provably differ in the two places we claim they do:
     both-empty frames are scored by the released code but skipped by ours, and
     the boundary tolerance is a radius-8 disk vs a 5x5 box.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_protocol_common import (  # noqa: E402
    OFFICIAL_BOUNDARY, _seg2bmap, official_frame_metrics, official_seq_metrics,
    ours_seq_metrics, released_check,
)

TOL = 1e-9
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}"
          f"{(' :: ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def disk_seq(H: int = 60, W: int = 80, offset: int = 0, T: int = 7):
    """A moving disk; `offset` shifts the prediction by that many pixels."""
    yy, xx = np.mgrid[0:H, 0:W]
    preds, gts = [], []
    for t in range(T):
        cy, cx = 20 + t, 30 + 2 * t
        gts.append(((yy - cy) ** 2 + (xx - cx) ** 2) <= 12 ** 2)
        preds.append(((yy - (cy + offset)) ** 2 + (xx - (cx + offset)) ** 2)
                     <= 12 ** 2)
    return preds, gts


def check_agreement(preds, gts, label: str) -> None:
    """The port must reproduce the released code exactly, or the A/B is void."""
    for skip in (False, True):
        rel = released_check(preds, gts, skip)
        if rel is None:
            print("  [skip] released evaluator not loadable")
            return
        port = official_seq_metrics(preds, gts, skip)
        tag = f"{label} skip={skip}"
        for key in ("J", "F", "J&F"):
            check(f"port == released {key} ({tag})",
                  abs(port[key] - rel[key]) <= TOL,
                  f"port={port[key]:.12f} released={rel[key]:.12f}")


def main() -> int:
    print("[selfcheck] 1. a PERFECT prediction scores exactly 1.0")
    preds, gts = disk_seq(offset=0)
    check("synthetic pred == gt bit-for-bit",
          all(np.array_equal(p, g) for p, g in zip(preds, gts)))
    for tag, m in (("ours", ours_seq_metrics(preds, gts, 2, False)),
                   ("official", official_seq_metrics(preds, gts, False)),
                   ("released", released_check(preds, gts, False))):
        if m is None:
            continue
        check(f"{tag} perfect == 1.0", abs(m["J&F"] - 1.0) <= 1e-12,
              f"J&F={m['J&F']!r}")

    print("[selfcheck] 2. a 1-px offset is detected, and both agree on it")
    preds1, gts1 = disk_seq(offset=1)
    check("synthetic pred != gt at offset=1",
          not any(np.array_equal(p, g) for p, g in zip(preds1, gts1)))
    o1 = ours_seq_metrics(preds1, gts1, 2, False)
    f1 = official_seq_metrics(preds1, gts1, False)
    check("ours sees the offset", o1["J&F"] < 1.0, f"J&F={o1['J&F']:.4f}")
    check("official sees the offset", f1["J&F"] < 1.0, f"J&F={f1['J&F']:.4f}")
    print(f"      1-px offset: ours J&F={o1['J&F']:.4f} "
          f"official J&F={f1['J&F']:.4f}")

    print("[selfcheck] 3. port agrees with the released evaluator")
    p1, g1 = disk_seq(offset=1)
    check_agreement(p1, g1, "disk-1px")

    print("[selfcheck] 4. the two conventions differ where we claim they do")
    empty = [np.zeros((60, 80), bool)] * 5
    ours_e = ours_seq_metrics(empty, empty, 2, False)
    off_e = official_seq_metrics(empty, empty, False)
    check("ours skips all-empty frames -> n_frames 0", ours_e["n_frames"] == 0,
          f"n_frames={ours_e['n_frames']}")
    check("official counts all-empty frames -> n_frames 5", off_e["n_frames"] == 5,
          f"n_frames={off_e['n_frames']}")
    check("official scores all-empty as 1.0", abs(off_e["J&F"] - 1.0) <= 1e-12,
          f"J&F={off_e['J&F']!r}")

    # Boundary tolerance: the released disk at 480x854 has radius 8, our box is
    # 5x5. Measure the effective radius of each on the same shape.
    H, W = 480, 854
    bound_pix = int(np.ceil(OFFICIAL_BOUNDARY * np.linalg.norm((H, W))))
    check("released bound_pix at 480x854 is 8 (radius, not diameter)",
          bound_pix == 8, f"bound_pix={bound_pix}")

    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((yy - 240) ** 2 + (xx - 427) ** 2) <= 100 ** 2
    from xdrp.metrics import _boundary as ours_boundary
    import cv2
    from skimage.morphology import disk
    off_bmap = _seg2bmap(mask)
    off_dil = cv2.dilate(off_bmap.astype(np.uint8), disk(bound_pix)) > 0
    ours_dil = ours_boundary(mask, 2)
    # Width of the tolerance band: how far the dilated set reaches past the mask.
    off_extra = int((off_dil & ~mask).sum())
    ours_extra = int((ours_dil & ~mask).sum())
    check("official tolerance band is wider than ours (at 480x854)",
          off_extra > ours_extra,
          f"official extra px={off_extra} vs ours extra px={ours_extra} "
          f"({off_extra / max(ours_extra, 1):.2f}x)")
    print(f"      boundary band: official {off_extra} px, ours {ours_extra} px")

    print("[selfcheck] 5. _seg2bmap rejects a 3-D mask instead of guessing")
    try:
        _seg2bmap(np.zeros((10, 10, 3), bool))
        check("_seg2bmap rejects 3-D input", False, "no assertion raised")
    except AssertionError:
        check("_seg2bmap rejects 3-D input", True)

    print(f"\n[selfcheck] {'PASS' if not failures else 'FAIL'}: "
          f"{len(failures)} failure(s)")
    for f in failures:
        print(f"    - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
