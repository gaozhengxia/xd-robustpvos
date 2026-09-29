"""Whole-video, memory-based tracking -- SAM 2.1's released video predictor.

Why this module exists
----------------------
Every other arm in this repository is a *frame-by-frame* loop: encode the current
frame, prompt it with the previous mask, take the prediction. Nothing survives a
frame except the binary mask, so once the object is lost it stays lost
(`docs/BASELINE_DIAGNOSIS.md` section 9 measured this: the flow chain that would
have to carry the position dies within ~10 frames). SAM 2.1's released *video*
predictor is a different architecture -- a memory bank of past frames' features
read by the mask decoder -- and until now the paper had never measured one.
The central hypothesis ("downstream gating can replace a memory bank") was
therefore a comparative claim with no comparator (CL4).

How it plugs in
---------------
`pipeline.run_sequence` dispatches any mode listed in `pipeline.VIDEO_MODES` to
`run_video_sequence` below. Because the dispatch sits in `run_sequence`, every
runner (`02`/`03`/`04`/`06`) and `benchmark.evaluate_cell` work unchanged, and
`J`/`F`/`J&F` stay computed by the same code as every other arm.

Two choices that exist only to avoid uncontrolled variables
-----------------------------------------------------------
* **No JPEG staging.** SAM 2's own loader only accepts an MP4 or a directory of
  JPEGs. Writing the degraded frames out first would re-encode them a second
  time, so this arm would be scored on slightly different pixels than every other
  arm -- a silent difference of exactly the kind that has cost this project
  before. The frames are converted to the model's input tensor in memory
  instead, replicating `sam2.utils.misc._load_img_as_tensor` exactly (PIL, RGB,
  `resize((image_size, image_size))` with PIL's own default resample, `/255`,
  then the SAM 2 mean/std).
* **The same prompt as every other arm**: the annotated first-frame mask, passed
  as a hard binary mask prompt (`add_new_mask`). `mask_prompt: soft` has no
  counterpart here -- the public video API takes binary masks -- so this arm
  always uses the hard prompt, which is also the released convention.

Object mapping: the protocol scores per (sequence, object) and this runs one
object per call, so the internal SAM 2 object id is always 1 and the caller's
`obj_id` is carried in the result row, not in the tracker.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .degradations import to_u8

#: ImageNet statistics SAM 2 normalizes with (same constants as
#: `sam2.utils.misc.load_video_frames`). Duplicated rather than imported so this
#: module stays importable without loading torch/sam2.
_IMG_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMG_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Keys `inference_state` must carry. `init_state` builds this same dict; we
#: build it directly because that is the only way to feed in-memory frames.
#: Checked explicitly below so a SAM 2 upgrade that renames a key fails loudly
#: here instead of producing a cryptic `KeyError` deep inside the tracker.
_STATE_KEYS: Tuple[str, ...] = (
    "images", "num_frames", "offload_video_to_cpu", "offload_state_to_cpu",
    "video_height", "video_width", "device", "storage_device",
    "point_inputs_per_obj", "mask_inputs_per_obj", "cached_features",
    "constants", "obj_id_to_idx", "obj_idx_to_id", "obj_ids",
    "output_dict_per_obj", "temp_output_dict_per_obj", "frames_tracked_per_obj",
)

#: One predictor per (checkpoint, model_cfg, device, autocast). Loading the
#: weights per sequence would dominate the runtime.
_PREDICTORS: Dict[Tuple, Any] = {}

#: Connectivity used by the re-implemented hole-filling post-process (see
#: `install_postprocessing_patch`). SAM 2's CUDA kernel hides this choice inside
#: `_C.get_connected_componnets`, so it cannot be read off the source; 4-connectivity
#: is OpenCV's default for `connectedComponents`. The residual ambiguity is
#: MEASURED rather than assumed -- `scripts/103_video_postproc_ab.py` reports the
#: J&F difference between 4- and 8-connectivity on a real clip.
POSTPROC_CONNECTIVITY: int = 4

#: Set once the replacement is installed, so callers can tell "the post-process
#: ran" from "the library silently skipped it".
_POSTPROC_PATCHED: Dict[str, bool] = {"installed": False, "connectivity": 0}

#: Whether the inference state (memory-bank features and per-frame mask logits)
#: is staged in CPU memory and moved to the GPU frame by frame. This is the same
#: switch SAM 2's own `init_state` exposes as `offload_state_to_cpu`.
#:
#: SHIPPED OFF, on a measurement rather than on an argument. It was turned on on
#: 2026-09-17 to make two concurrent full-protocol workers fit inside 16 GB, and
#: `scripts/104_video_offload_ab.py` then measured the saving on the longest clip
#: in the val split (cows, 104 frames): **+0.5 MiB, 0.0%** -- nothing. J&F was
#: bit-identical across both legs AND both leg orders, so the switch is harmless,
#: but it is not the VRAM fix it was turned on for, and carrying an extra
#: host-resident copy is not worth a deviation from the released default. Kept as
#: a parameter so the A/B stays runnable; the invariant `storage_device follows
#: the flag` is asserted in `00_smoke_test.py`.
#:
#: The real constraint this measurement exposed is elsewhere: peak GPU footprint
#: grows with clip length (104 frames -> 16340 MiB allocated / 16852 MiB reserved
#: against a 16275 MiB device), so long clips SPILL into host RAM through the
#: Windows CUDA sysmem fallback. That is invisible to a `nvidia-smi` reading and
#: to the process's own allocator, and it is what actually exhausted the machine
#: when three workers ran at once. Budget host RAM per worker, not just VRAM.
OFFLOAD_STATE_TO_CPU: bool = False


def connected_components_cv2(mask: Any, connectivity: int = POSTPROC_CONNECTIVITY):
    """Pure-Python stand-in for `sam2._C.get_connected_componnets`.

    The released SAM 2.1 *video* predictor builds with `fill_hole_area=8`
    (`sam2/build_sam.py`), i.e. it fills small holes in the low-resolution mask
    logits before resizing to the video resolution. That post-process needs the
    CUDA extension `sam2._C`, which is not built here. Without a replacement the
    library catches the ImportError and **silently skips the step**, emitting only
    a `UserWarning` that says results are "not affected in most cases".

    "Most cases" is not a measurement, and a baseline that quietly differs from
    the released model is precisely the failure this project keeps paying for. So
    the operator is re-implemented with OpenCV and the result is pinned by
    `00_smoke_test.py`; the one remaining free parameter (connectivity) is
    quantified by `scripts/103_video_postproc_ab.py` instead of being assumed
    away.

    Contract (matching the docstring of the function it replaces): for an input
    of shape `(N, 1, H, W)` where nonzero marks the region to label, return
    `labels` (component id per pixel, 0 on the input's zeros) and `areas` (the
    component's pixel count, 0 on the input's zeros).
    """
    import cv2
    import torch

    arr = mask
    if hasattr(arr, "detach"):
        arr = arr.detach()
    # Keep the input's device and hand it back on the outputs. Returning CPU
    # tensors here is NOT harmless: `fill_holes_in_mask_scores` feeds them to
    # `torch.where(is_hole, 0.1, mask)` with `mask` on the GPU, which raises a
    # device-mismatch error -- and that call site catches every exception and
    # silently skips the post-process. The first version of this function did
    # exactly that, so the patch looked installed while the post-process still
    # never ran.
    dev = getattr(arr, "device", None)
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr)
    single = arr.ndim == 2
    if single:
        arr = arr[None, None]
    if arr.ndim == 3:
        arr = arr[:, None]
    if arr.ndim != 4:
        raise ValueError(f"expected (N, 1, H, W), got {arr.shape}")

    n, _, h, w = arr.shape
    labels = np.zeros((n, 1, h, w), np.int32)
    areas = np.zeros((n, 1, h, w), np.int32)
    for i in range(n):
        binary = (arr[i, 0] > 0).astype(np.uint8)
        if not binary.any():
            continue
        num, lab, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=int(connectivity))
        # `lab` is 0 on background and 1..num-1 on the components; `stats[k].area`
        # is the component size. Both outputs only carry the component
        # information on the input's nonzeros, 0 elsewhere.
        comp_area = np.zeros(num, np.int32)
        comp_area[1:] = stats[1:, cv2.CC_STAT_AREA]
        labels[i, 0] = lab.astype(np.int32)
        areas[i, 0] = comp_area[lab]
    labels_t = torch.from_numpy(labels)
    areas_t = torch.from_numpy(areas)
    if dev is not None:
        labels_t = labels_t.to(dev)
        areas_t = areas_t.to(dev)
    if single:
        # A 2-D input is a convenience for tests and callers with one frame; the
        # outputs then keep the input's own rank, not the (N,1,H,W) that the
        # batched path produces. SAM 2 itself always passes (N,1,H,W).
        return labels_t[0, 0], areas_t[0, 0]
    return labels_t, areas_t


def install_postprocessing_patch(connectivity: int = POSTPROC_CONNECTIVITY,
                                 device: str = "cpu") -> None:
    """Make SAM 2's hole-filling post-process actually run.

    Replaces `sam2.utils.misc.get_connected_components` so
    `fill_holes_in_mask_scores` stops falling into its silent `except`. Raises if
    the symbol is not where it is expected -- a patch that silently fails to
    install is worse than no patch, because the run then looks post-processed.

    The self-check deliberately runs on `device`, i.e. in the same device context
    as the real call, and **fails on the fallback warning** rather than only
    checking values. The first version of this function checked values on the CPU,
    where everything agreed, while the GPU path raised a device-mismatch error
    that the library swallowed -- so the post-process still never ran. A
    self-check that does not reproduce the real calling context is not a check.
    """
    import warnings

    import sam2.utils.misc as _misc
    if not hasattr(_misc, "get_connected_components"):
        raise RuntimeError(
            "sam2.utils.misc.get_connected_components disappeared; the hole-filling "
            "post-process of the released video predictor cannot be reproduced, so "
            "this arm would silently differ from the published model.")
    if "original" not in _POSTPROC_PATCHED:
        # Kept so a caller (e.g. scripts/103) can restore SAM 2's own
        # implementation and measure the post-process's effect by disabling it.
        _POSTPROC_PATCHED["original"] = _misc.get_connected_components
    _misc.get_connected_components = (
        lambda m: connected_components_cv2(m, connectivity=int(connectivity)))
    _POSTPROC_PATCHED["installed"] = True
    _POSTPROC_PATCHED["connectivity"] = int(connectivity)

    # Prove the post-process now runs, on a case with a known answer: a
    # background island of area 4 inside a filled square must be filled (to 0.1),
    # and one of area 900 must not. Any fallback warning is a failure, not a note.
    import torch
    logits = torch.full((1, 1, 64, 64), 1.0, device=device)
    logits[0, 0, 10:12, 10:12] = -1.0            # 2x2 hole   -> area 4,   filled
    logits[0, 0, 30:60, 30:60] = -1.0            # 30x30 hole -> area 900, kept
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filled = _misc.fill_holes_in_mask_scores(logits, 8)
    fallbacks = [str(w.message) for w in caught
                 if "Skipping the post-processing" in str(w.message)]
    if fallbacks:
        raise RuntimeError(
            "the hole-filling post-process fell back to a no-op on device "
            f"{device!r} (SAM 2 catches every exception there and continues). "
            f"Warning was: {fallbacks[0][:200]}. The memory-based arm would "
            f"silently differ from the released SAM 2.1 model; refusing to run.")
    ok = (abs(float(filled[0, 0, 10, 10]) - 0.1) < 1e-6
          and float(filled[0, 0, 40, 40]) < 0.0)
    if not ok:
        raise RuntimeError(
            "the hole-filling post-process did not behave as the released SAM 2.1 "
            "video predictor specifies (small hole filled with 0.1, large hole "
            "kept). The memory-based arm would silently differ from the published "
            "model; refusing to run.")


def frames_to_images(frames: Sequence[np.ndarray],
                     image_size: int) -> Tuple[Any, int, int]:
    """Frames -> `(T, 3, S, S)` float32 tensor (CPU), normalized like SAM 2.

    Mirrors `sam2.utils.misc._load_img_as_tensor` operation for operation. The
    `resize` call deliberately passes no `resample` argument, because neither
    does SAM 2 -- hardcoding an interpolation here could silently differ from
    the library's default after a Pillow upgrade.

    Returns `(images, video_height, video_width)`; the height/width are the
    *original* frame size, which is what the tracker resizes its output masks
    back to.
    """
    import torch
    from PIL import Image

    T = len(frames)
    if T == 0:
        raise ValueError("empty sequence")
    h, w = frames[0].shape[:2]
    images = torch.empty(T, 3, int(image_size), int(image_size), dtype=torch.float32)
    mean = torch.tensor(_IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(_IMG_STD, dtype=torch.float32)[:, None, None]
    for i, f in enumerate(frames):
        u8 = to_u8(f)
        if u8.shape[:2] != (h, w):
            raise ValueError(
                f"frame {i} has shape {u8.shape[:2]} but frame 0 has {(h, w)}; "
                f"SAM 2's video predictor tracks one fixed resolution")
        # `frames` are already RGB (degradations.py documents float32 RGB), so
        # this is the same conversion the loader performs on a JPEG.
        arr = np.array(Image.fromarray(u8).convert("RGB")
                       .resize((int(image_size), int(image_size))))
        images[i] = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)
    images.div_(255.0)
    images.sub_(mean)
    images.div_(std)
    return images, int(h), int(w)


def build_inference_state(predictor, frames: Sequence[np.ndarray],
                          offload_video_to_cpu: bool = True,
                          offload_state_to_cpu: Optional[bool] = None,
                          ) -> Dict[str, Any]:
    """The equivalent of `predictor.init_state(...)` for in-memory frames."""
    import torch

    from collections import OrderedDict

    if offload_state_to_cpu is None:
        offload_state_to_cpu = OFFLOAD_STATE_TO_CPU
    images, video_h, video_w = frames_to_images(frames, int(predictor.image_size))
    device = predictor.device
    # `init_state` derives the storage device from the offload flag, and the
    # tracker reads `storage_device` (not `device`) when it parks per-frame
    # outputs. Setting one without the other therefore sends tensors to a device
    # the propagation loop does not expect -- an error at best, a silent CPU/GPU
    # round trip at worst. So the two are derived together here and the
    # invariant is asserted in `00_smoke_test.py`.
    storage_device = torch.device("cpu") if offload_state_to_cpu else device
    state: Dict[str, Any] = {
        "images": images,
        "num_frames": len(frames),
        "offload_video_to_cpu": bool(offload_video_to_cpu),
        "offload_state_to_cpu": bool(offload_state_to_cpu),
        "video_height": video_h,
        "video_width": video_w,
        "device": device,
        "storage_device": storage_device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    missing = [k for k in _STATE_KEYS if k not in state]
    if missing:
        raise KeyError(
            f"this module's inference_state is missing {missing}. SAM 2's "
            f"`init_state` is the reference; the installed version may have "
            f"changed its state layout -- update `_STATE_KEYS` here after "
            f"reading sam2/sam2_video_predictor.py.")
    # Same warm-up `init_state` performs: encode frame 0 so the first prompt is
    # not paying for the backbone pass.
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    del torch  # imported only to fail early if torch is absent
    return state


def mask_from_logits(logits: Any, shape: Tuple[int, int],
                     where: str = "SAM 2 video predictor output") -> np.ndarray:
    """Threshold a logit map to a boolean mask, asserting the final geometry.

    SAM 2's video predictor returns *logits* at the original video resolution
    (`> 0` is the released convention). The shape is asserted against the caller's
    frame size rather than trusted: a silent resolution mismatch would still
    produce a plausible-looking `J&F`, which is how a whole class of bugs has
    survived in this repository before.
    """
    t = logits
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "float"):
        t = t.float().cpu().numpy()
    arr = np.asarray(t)
    # (B, 1, H, W) -> (H, W); (1, H, W) -> (H, W)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.shape != tuple(shape):
        raise ValueError(
            f"{where} returned a mask of shape {arr.shape}, but the frame is "
            f"{tuple(shape)}. The tracker is supposed to resize to the original "
            f"video resolution; a mismatch here would be scored silently.")
    return arr > 0.0


def predictor_for(sam2_cfg, autocast_dtype: Optional[str] = None):
    """Build (once) and return a SAM 2.1 video predictor for `sam2_cfg`."""
    if sam2_cfg is None:
        raise RuntimeError(
            "the video arm needs a checkpoint/model_cfg; pass the SAM 2 backend's "
            "config (the runner already builds one) -- a DummyBackend cannot "
            "stand in for a memory-based tracker.")
    key = (str(sam2_cfg.checkpoint), str(sam2_cfg.model_cfg),
           str(sam2_cfg.device), str(autocast_dtype))
    if key in _PREDICTORS:
        return _PREDICTORS[key]
    from sam2.build_sam import build_sam2_video_predictor
    from pathlib import Path
    if not Path(str(sam2_cfg.checkpoint)).exists():
        raise FileNotFoundError(
            f"checkpoint not found: {sam2_cfg.checkpoint}\n"
            "Run:  cd sam2/checkpoints && bash download_ckpts.sh")
    # Before building, make the released post-processing runnable: the default
    # build turns on `fill_hole_area=8`, which needs the CUDA extension we do not
    # have. See `install_postprocessing_patch`.
    if not _POSTPROC_PATCHED["installed"]:
        install_postprocessing_patch(device=str(sam2_cfg.device))
    predictor = build_sam2_video_predictor(str(sam2_cfg.model_cfg),
                                          str(sam2_cfg.checkpoint),
                                          device=str(sam2_cfg.device))
    predictor.eval()
    _POSTPROC_PATCHED["fill_hole_area"] = int(getattr(predictor, "fill_hole_area", 0))
    if _POSTPROC_PATCHED["fill_hole_area"] <= 0:
        # The build did not enable it (e.g. apply_postprocessing=False). Then the
        # arm IS the released model *with* post-processing off -- a declared
        # configuration, not a silent deficit -- and that must be visible.
        import warnings
        warnings.warn(
            "the video predictor was built with fill_hole_area=0, so the "
            "low-resolution hole-filling post-process is OFF. This is a declared "
            "configuration, not the released default; record it if the arm's "
            "numbers are published.", stacklevel=2)
    _PREDICTORS[key] = predictor
    return predictor


def tracker_context(predictor, autocast_dtype: Optional[str]):
    """Autocast context matching the image backend, so both arms run at the same
    precision (`bfloat16` on cuda by default)."""
    import torch
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "none": None}.get(str(autocast_dtype), None)
    if dtype is None:
        return torch.inference_mode()
    return torch.autocast("cuda", dtype=dtype)


def run_video_tracking(predictor, frames: Sequence[np.ndarray],
                       first_mask: np.ndarray,
                       offload_video_to_cpu: bool = True,
                       autocast_dtype: Optional[str] = None,
                       ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Track one object across `frames` given the annotated first-frame mask.

    Returns `(masks, info)` where `masks` has exactly one entry per input frame
    and `info` carries the mechanism counters (`n_encoder_calls`, `n_frames`).
    A frame the tracker failed to emit is an error, not a gap to fill: every
    reported metric is a mean over frames, so a silently missing frame would
    change the denominator.
    """
    shape = first_mask.shape[:2]
    T = len(frames)
    # Count backbone passes rather than asserting them: "the memory bank was
    # actually used" is a mechanism claim, and this repository's rule is that a
    # mechanism's counter must be measured and reported, not assumed.
    counts = {"n": 0}
    _orig = predictor._get_image_feature

    def _counted(*a, **kw):
        counts["n"] += 1
        return _orig(*a, **kw)

    predictor._get_image_feature = _counted
    try:
        # The hole-filling fallback is a *warning*, not an exception, at a call
        # site that catches everything -- so a run can be missing the released
        # post-process and look perfectly healthy. Turn it into a hard failure
        # for the duration of the tracking loop.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with tracker_context(predictor, autocast_dtype):
                state = build_inference_state(predictor, frames, offload_video_to_cpu)
                predictor.add_new_mask(state, 0, 1, to_u8(first_mask) > 0)
                masks: List[Optional[np.ndarray]] = [None] * T
                for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(state):
                    if 0 <= int(frame_idx) < T:
                        masks[int(frame_idx)] = mask_from_logits(
                            video_res_masks, shape,
                            where=f"SAM 2 video predictor, frame {int(frame_idx)}")
                predictor.reset_state(state)
        fallbacks = [str(w.message) for w in caught
                     if "Skipping the post-processing" in str(w.message)]
        if fallbacks:
            raise RuntimeError(
                "SAM 2 skipped its post-processing during tracking (it catches the "
                "exception and only warns). The track would not be the released "
                f"model's output. Warning was: {fallbacks[0][:200]}")
    finally:
        predictor._get_image_feature = _orig
    missing = [i for i, m in enumerate(masks) if m is None]
    if missing:
        raise RuntimeError(
            f"the video predictor emitted no mask for frames {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''} of {T}. Refusing to score a "
            f"partial track -- a missing frame silently changes the denominator "
            f"of the per-frame means.")
    info = {"n_frames": T, "obj_id_internal": 1,
            "n_encoder_calls": int(counts["n"])}
    return [np.asarray(m, bool) for m in masks], info
