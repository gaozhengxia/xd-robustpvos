"""Shared scorers for the evaluator-protocol investigation (scripts 105 / 106).

Kept in its own module so the offline self-check (106) can import the scorers
without pulling in the SAM 2 backend, CUDA or any dataset on disk.

Three scorers live here:

``official_*``
    A faithful port of the released SAM 2 evaluator
    (`sam2/sav_dataset/utils/sav_benchmark.py`, itself adapted from
    hkchengrex/vos-benchmark and davis2017-evaluation). Boundary tolerance is
    ``disk(ceil(0.008 * norm(shape)))`` -- at 480x854 that is ``disk(8)``, a
    radius-8 disk, i.e. a 17-pixel-wide band. Boundary maps are 1-pixel wide
    (``_seg2bmap``), and an all-empty (pred, gt) pair scores 1.0.

``ours_*``
    ``xdrp.metrics.seq_metrics``, the scorer behind every number in
    ``results/cl4_full_v2.csv``. Boundary tolerance is a 5x5 square
    (``cv2.erode(3x3)`` then ``dilate(5x5)``), and frames where both masks are
    empty are skipped rather than scored.

``released_check``
    Calls the *actual* released code, when importable, so the port can be
    validated rather than merely asserted. Returns ``None`` when
    ``sam2.sav_dataset`` is unavailable -- a mismatch is never swallowed, but an
    absence is tolerated so the offline path still runs.

The port existing as a port (rather than us calling the released code
everywhere) is deliberate: the released evaluator consumes on-disk PNG folders
and hashes/loads per frame, which is far slower than scoring arrays we already
hold, and it cannot express the intermediate convention (official boundary rule
+ our frame filtering) that the A/B needs. The self-check in script 106 is what
keeps the port honest.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Released default. Relative to the image diagonal: `bound_pix` is
#: `ceil(0.008 * norm((H, W)))`, so it is resolution-independent by design.
OFFICIAL_BOUNDARY = 0.008

_RELEASED_CLS = None
_RELEASED_TRIED = False


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Faithful copy of `sam2/sav_dataset/utils/sav_benchmark.py::_seg2bmap`.

    Emits a 1-pixel-wide boundary; the last row/column are special-cased and the
    bottom-right pixel is forced to 0. Kept behaviour-for-behaviour identical
    (not merely "equivalent") so a mismatch against the released code is a real
    signal rather than a difference of taste.
    """
    seg = np.asarray(seg)
    assert np.atleast_3d(seg).shape[2] == 1, "expected a single-channel mask"
    seg = seg.astype(bool).copy()
    seg[seg > 0] = 1

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def _get_iou(intersection: int, pixel_sum: int) -> float:
    """`get_iou` from the released evaluator: an all-empty pair scores 1."""
    if intersection == pixel_sum:
        assert intersection == 0, "intersection == pixel_sum with non-zero area"
        return 1.0
    return intersection / (pixel_sum - intersection)


def official_frame_metrics(pred: np.ndarray, gt: np.ndarray,
                           boundary: float = OFFICIAL_BOUNDARY
                           ) -> Tuple[float, float]:
    """J and F for one object in one frame, following the released evaluator."""
    import cv2
    from skimage.morphology import disk

    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)

    j = _get_iou(int((p & g).sum()), int(p.sum()) + int(g.sum()))

    mask_boundary = _seg2bmap(p)
    gt_boundary = _seg2bmap(g)
    # The disk depends only on the image shape, exactly as in the released code.
    bound_pix = np.ceil(boundary * np.linalg.norm(p.shape))
    boundary_disk = disk(bound_pix)
    mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

    gt_match = gt_boundary.astype(np.uint8) * mask_dilated
    fg_match = mask_boundary.astype(np.uint8) * gt_dilated

    n_fg = int(np.sum(mask_boundary))
    n_gt = int(np.sum(gt_boundary))
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = float(np.sum(fg_match)) / float(n_fg)
        recall = float(np.sum(gt_match)) / float(n_gt)
    f = (0.0 if precision + recall == 0
         else 2 * precision * recall / (precision + recall))
    return float(j), float(f)


def frame_index(n: int, skip_first_and_last: bool) -> range:
    """Released convention: drop the first and last frame of the *clip*."""
    if skip_first_and_last and n > 2:
        return range(1, n - 1)
    return range(n)


def official_seq_metrics(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                         skip_first_and_last: bool = False,
                         boundary: float = OFFICIAL_BOUNDARY) -> Dict[str, float]:
    """Per-instance J / F / J&F, released-evaluator style.

    Two conventions differ from `xdrp.metrics` and both are deliberate:
      * the first and last frames are dropped when `skip_first_and_last` (the
        released default, and DAVIS semi-supervised practice);
      * frames where BOTH masks are empty are still scored and score 1.0 -- the
        released code never filters them (`get_iou` returns 1 for the all-empty
        case). Our scorer skips such frames, which changes the denominator of
        the per-frame mean.
    """
    idx = list(frame_index(len(preds), skip_first_and_last))
    if not idx:
        return {"J": float("nan"), "F": float("nan"), "J&F": float("nan"),
                "n_frames": 0}
    js = np.empty(len(idx), np.float64)
    fs = np.empty(len(idx), np.float64)
    for i, t in enumerate(idx):
        js[i], fs[i] = official_frame_metrics(preds[t], gts[t], boundary)
    mj, mf = float(js.mean()), float(fs.mean())
    return {"J": mj, "F": mf, "J&F": 0.5 * (mj + mf), "n_frames": len(idx)}


def ours_seq_metrics(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                     dilation: int = 2, skip_first_and_last: bool = False
                     ) -> Dict[str, float]:
    """`xdrp.metrics.seq_metrics`, optionally with skip-first/last on top.

    The skip is applied here rather than inside `metrics.py` so that the CSV
    convention is untouched: this measures a *hypothetical* protocol, it does
    not change the one the results were produced under.
    """
    from xdrp.metrics import seq_metrics
    idx = list(frame_index(len(preds), skip_first_and_last))
    if not idx:
        return {"J": float("nan"), "F": float("nan"), "J&F": float("nan"),
                "n_frames": 0}
    return seq_metrics([preds[t] for t in idx], [gts[t] for t in idx],
                       dilation=dilation)


def _released_evaluator_class():
    """Import the released `Evaluator`, by any route that actually works here.

    The released file lives inside the SAM 2 checkout but is *not* part of the
    installed `sam2` package (there is no `sam2.sav_dataset` subpackage in the
    wheel), so `import sam2.sav_dataset...` fails. Load it from its path
    instead. Cached at module level: re-executing the file per cell would be
    pure overhead.
    """
    global _RELEASED_CLS, _RELEASED_TRIED
    if _RELEASED_TRIED:
        return _RELEASED_CLS
    _RELEASED_TRIED = True

    import importlib.util
    from pathlib import Path as _P

    candidates = [
        # A sibling `sam2/` checkout (the layout 1.5 produces), addressed
        # relative to this repository -- never through an absolute path from
        # whichever machine a run happened on.
        _P(__file__).resolve().parents[2] / "sam2" / "sav_dataset" / "utils" / "sav_benchmark.py",
        # The same file reached from the working directory, for a process
        # started from that parent instead of from inside the repository.
        _P.cwd() / "sam2" / "sav_dataset" / "utils" / "sav_benchmark.py",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_released_sav_benchmark", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _RELEASED_CLS = mod.Evaluator
            return _RELEASED_CLS
        except Exception:  # noqa: BLE001
            continue
    return None


def released_check(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                   skip_first_and_last: bool) -> Optional[Dict[str, float]]:
    """Score via the actual released evaluator, if it can be loaded.

    Returns None when the released module is unavailable so the offline path can
    still run; a *mismatch* is never swallowed by the caller.
    """
    Evaluator = _released_evaluator_class()
    if Evaluator is None:
        return None

    idx = list(frame_index(len(preds), skip_first_and_last))
    ev = Evaluator(name="probe", obj_id=1)
    for t in idx:
        # The released evaluator works on an object-id map; our masks are
        # binary, so object 1 is wherever the mask is True.
        ev.feed_frame(mask=np.asarray(preds[t]).astype(np.uint8),
                      gt=np.asarray(gts[t]).astype(np.uint8))
    iou, bf = ev.conclude()
    if 1 not in iou:
        return {"J": float("nan"), "F": float("nan"), "J&F": float("nan"),
                "n_frames": len(idx)}
    # `conclude()` already multiplies by 100.
    j, f = float(iou[1]) / 100.0, float(bf[1]) / 100.0
    return {"J": j, "F": f, "J&F": 0.5 * (j + f), "n_frames": len(idx)}
