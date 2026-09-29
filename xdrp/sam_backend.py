"""Promptable segmentation backends.

Two implementations:

`SAM2Backend`  -- wraps SAM 2.1's *image* predictor. We deliberately use the image
                  predictor rather than the streaming video predictor because
                  DAG-RS needs, at each decision keyframe, to run the encoder on
                  **several different observations** (one per restoration
                  operator). This keeps DAG-RS model-agnostic and free of any
                  dependency on SAM 2's internal memory implementation -- a
                  deliberate design choice that separates it from the memory-
                  engineering line of work (UAMP / SAM2Long / MA-SAM2).

`DummyBackend` -- a deterministic, degradation-sensitive colour-template
                  segmenter. It exists ONLY so the full pipeline can be smoke
                  tested without downloading weights. It is never used to
                  produce any number that goes into the paper.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

from .evidence import as_bool, bbox_of, prompt_logits, color_descriptor
from .degradations import to_u8

EPS = 1e-8


def as_prediction(out) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Normalise whatever a backend's `predict` returned into (mask, score, soft).

    Backends return either a 2-tuple `(mask, score)` or a 3-tuple
    `(mask, score, soft)`. Callers inside `pipeline.run_sequence` are wrapped in
    a broad `except Exception`, so unpacking a fixed arity silently degrades the
    whole run to "propagate the warped anchor" instead of raising -- a failure
    mode that produces plausible-looking but meaningless numbers. Funnelling
    every call through this helper removes that class of bug entirely.
    """
    if isinstance(out, tuple):
        m, s = out[0], float(out[1])
        soft = out[2] if len(out) > 2 else None
        return m, s, soft
    return np.asarray(out), 1.0, None


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #

class SegBackend:
    name = "base"
    supports_mask_prompt = True

    def set_reference(self, img: np.ndarray, mask: np.ndarray) -> None:
        """Called once with the first-frame prompt to seed object state."""

    def predict(self, img: np.ndarray, box: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                multimask: bool = False,
                want_soft: bool = False,
                ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Return `(mask, score, soft)`.

        `soft` is the segmenter's own continuous mask probability map, and is
        `None` unless `want_soft=True`. Implementations MUST return all three
        positions (pass `None` for `soft`); callers use `as_prediction` so that
        a 2-tuple from a third-party backend still works.
        """
        raise NotImplementedError

    def descriptor(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return color_descriptor(img, mask)

    def clear_cache(self) -> None:
        pass

    @property
    def n_encoder_calls(self) -> int:
        return 0


# --------------------------------------------------------------------------- #
# SAM 2.1
# --------------------------------------------------------------------------- #

@dataclass
class SAM2Config:
    checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device: str = "cuda"
    autocast_dtype: str = "bfloat16"     # bfloat16 | float16 | none
    cache_embeddings: bool = True
    cache_size: int = 4


class SAM2Backend(SegBackend):
    """Model-agnostic wrapper around SAM 2.1's image predictor."""

    name = "sam2"

    def __init__(self, cfg: SAM2Config):
        self.cfg = cfg
        self._predictor = None
        self._cache: "Dict[int, Any]" = {}
        self._cache_order: List[int] = []
        self._n_calls = 0
        self._cache_key: Optional[int] = None
        #: width of the pooled-embedding block, fixed by the first successful
        #: extraction so `descriptor` has an input-independent output shape.
        self._pooled_dim: Optional[int] = None
        self._init_model()

    # -- setup ---------------------------------------------------------- #

    def _init_model(self) -> None:
        import torch
        try:
            from sam2.build_sam import build_sam2
        except ImportError as e:      # pragma: no cover
            raise ImportError(
                "SAM 2 is not installed.\n"
                "  git clone https://github.com/facebookresearch/sam2.git\n"
                "  cd sam2 && pip install -e .\n"
                "On Windows the CUDA extension build may fail; that is fine -- SAM 2\n"
                "falls back to a Python-only path and inference results are unaffected.\n"
                f"Original error: {e}"
            ) from e
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not Path(self.cfg.checkpoint).exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.cfg.checkpoint}\n"
                "Run:  cd sam2/checkpoints && bash download_ckpts.sh\n"
                "or download sam2.1_hiera_large.pt manually into ./checkpoints/"
            )
        model = build_sam2(self.cfg.model_cfg, self.cfg.checkpoint,
                           device=self.cfg.device)
        model.eval()
        self._predictor = SAM2ImagePredictor(model)
        self._torch = torch
        self._autocast_dtype = {
            "bfloat16": torch.bfloat16, "float16": torch.float16, "none": None
        }[self.cfg.autocast_dtype]

    # -- helpers -------------------------------------------------------- #

    def _ctx(self):
        torch = self._torch
        if self._autocast_dtype is None:
            return torch.inference_mode()
        return torch.autocast("cuda", dtype=self._autocast_dtype)

    @staticmethod
    def _hash_image(img: np.ndarray) -> int:
        """Cheap content hash so identical restorations reuse the encoder pass."""
        h = to_u8(img)
        return hash((h.shape, int(h[::13, ::13].sum()), int(h[:8, :8].sum())))

    def _set_image(self, img: np.ndarray, cache_token: Optional[Any] = None) -> None:
        """Encode an image, reusing a cached encoder pass when the caller can
        supply an exact identity token for (sequence, frame, operator).

        We deliberately require an explicit token rather than hashing the pixels:
        a hash collision would silently reuse the wrong features, which is a
        correctness hazard we are not willing to accept in research code.
        """
        # The cache-hit path below never touches `img_u8`: encoding only happens
        # on a miss. Converting first therefore charged every HIT the conversion
        # (measured 10.45 ms) for a value that was then discarded. Converting on
        # the miss path is bit-identical -- the cached branch is exactly the one
        # that does not read the converted image.
        if cache_token is not None and self.cfg.cache_embeddings \
                and cache_token in self._cache:
            self._predictor._features = self._cache[cache_token]
            self._predictor._is_image_set = True
            self._cache_key = cache_token
            return
        img_u8 = to_u8(img) if img.dtype != np.uint8 else img
        with self._ctx():
            self._predictor.set_image(img_u8)
        self._n_calls += 1
        if cache_token is not None and self.cfg.cache_embeddings:
            feat = {k: (v.clone() if hasattr(v, "clone") else v)
                    for k, v in getattr(self._predictor, "_features", {}).items()}
            self._cache[cache_token] = feat
            self._cache_order.append(cache_token)
            while len(self._cache_order) > int(self.cfg.cache_size):
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        self._cache_key = cache_token

    @property
    def n_encoder_calls(self) -> int:
        return self._n_calls

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_order.clear()

    # -- API ------------------------------------------------------------ #

    def predict(self, img: np.ndarray, box: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                multimask: bool = False,
                cache_token: Optional[Any] = None,
                want_soft: bool = False,
                ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Run one prediction.

        `mask` may be a binary mask (prompted as a hard wall) or a float array in
        [0, 1] (prompted as soft logits) -- see `evidence.prompt_logits`.

        With `want_soft=True` the returned tuple has a third element: the model's
        own mask probability map at the input resolution, i.e. the sigmoid of the
        low-resolution logits that the image predictor already computes and the
        previous version of this wrapper used to discard. Callers that propagate
        an anchor across frames should carry this instead of a binarised mask.
        """
        self._set_image(img, cache_token=cache_token)
        kwargs: Dict[str, Any] = {"multimask_output": bool(multimask)}
        if box is not None:
            kwargs["box"] = np.asarray(box, np.float32).reshape(4)
        if mask is not None and self.supports_mask_prompt:
            kwargs["mask_input"] = prompt_logits(mask, size=256)
        if box is None and mask is None:
            raise ValueError("at least one prompt (box or mask) is required")

        with self._ctx():
            try:
                masks, scores, low_res = self._predictor.predict(**kwargs)
            except Exception:
                # Degrade gracefully to a box-only prompt if mask prompting is
                # unsupported by the installed SAM2 revision.
                if "mask_input" in kwargs:
                    kwargs.pop("mask_input", None)
                    masks, scores, low_res = self._predictor.predict(**kwargs)
                else:
                    raise

        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        if masks.ndim == 4:
            masks = masks[:, 0] if masks.shape[1] == 1 else masks[:, 0]
        if masks.ndim == 3 and masks.shape[0] > 1:
            best = int(np.argmax(scores))
        elif masks.ndim == 3:
            best = 0
        else:
            masks = masks[None]
            best = 0

        soft: Optional[np.ndarray] = None
        if want_soft:
            soft = self._soft_from_logits(low_res, best, img.shape[:2])
        return as_bool(masks[best]), float(scores[best]), soft

    @staticmethod
    def _soft_from_logits(low_res: Any, best: int,
                          shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Sigmoid of the model's own 256x256 logits, upsampled to image size."""
        if low_res is None:
            return None
        lr = np.asarray(low_res, np.float32)
        if lr.ndim == 4:
            lr = lr[0]
        if lr.ndim == 2:
            lr = lr[None]
        if best >= lr.shape[0]:
            best = 0
        logit = np.clip(lr[best], -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return cv2.resize(prob.astype(np.float32), (int(shape[1]), int(shape[0])),
                          interpolation=cv2.INTER_LINEAR)

    def set_reference(self, img: np.ndarray, mask: np.ndarray) -> None:
        """Seed the encoder on the annotated reference frame.

        The base class documents this call as the one that seeds object state,
        but the image predictor only encodes inside `_set_image` -- which
        `predict` reaches. Without this override nothing has been encoded yet
        when `pipeline` asks for the anchor descriptor, so `descriptor` fell
        back to the colour histogram alone. That silently made the anchor
        descriptor 24-wide instead of 280, froze the appearance EMA on its size
        guard, and made re-anchoring raise ValueError on its first invocation
        (24-dim appearance vs 280-dim candidate).
        """
        self._set_image(img)

    def descriptor(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Colour histogram + (if available) mask-pooled SAM image embedding.

        Width contract: once an embedding has been extracted the returned
        length is constant and independent of the input. When the embedding is
        unavailable -- no `set_image` yet, an empty mask, or an encoder failure
        -- the embedding block is zero-padded rather than dropped, so callers
        may carry a descriptor across frames and compare it with `np.dot`.
        Returning a short vector here is exactly what crashed re-anchoring.
        """
        col = color_descriptor(img, mask)
        pooled = self._pooled_embedding(img, mask)
        if pooled is not None:
            if self._pooled_dim is None:
                self._pooled_dim = int(pooled.size)
            elif pooled.size != self._pooled_dim:
                pooled = self._fit_width(pooled, self._pooled_dim)
        if self._pooled_dim is None:
            # No embedding has ever been obtained, so its width is unknown: the
            # colour histogram is all we can honestly return.
            return col
        if pooled is None:
            pooled = np.zeros(self._pooled_dim, np.float32)
        return np.concatenate([col, pooled.astype(np.float32)])

    @staticmethod
    def _fit_width(v: np.ndarray, width: int) -> np.ndarray:
        """Truncate or zero-pad `v` to `width` so descriptors stay comparable."""
        out = np.zeros(int(width), np.float32)
        flat = np.asarray(v, np.float32).reshape(-1)
        n = min(int(flat.size), int(width))
        out[:n] = flat[:n]
        return out

    def _pooled_embedding(self, img: np.ndarray,
                          mask: np.ndarray) -> Optional[np.ndarray]:
        """L2-normalised mask-pooled image embedding, or None if unavailable."""
        if self._predictor is None:
            return None
        if not getattr(self._predictor, "_is_image_set", False):
            # Explicit, checkable state instead of a swallowed exception: the
            # encoder has not run, so any embedding we could read would belong
            # to a different frame than `img`.
            return None
        try:
            feat = self._predictor.get_image_embedding()      # 1 x C x h x w
        except Exception:
            return None
        if feat is None:
            return None
        f = feat[0].float().cpu().numpy()
        h, w = f.shape[-2:]
        ms = cv2.resize(as_bool(mask).astype(np.float32), (w, h),
                        interpolation=cv2.INTER_AREA)
        ms = (ms > 0.5).astype(np.float32)
        if ms.sum() <= 0:
            return None
        pooled = (f * ms[None]).sum(axis=(1, 2)) / (ms.sum() + EPS)
        return (pooled / (np.linalg.norm(pooled) + EPS)).astype(np.float32)


# --------------------------------------------------------------------------- #
# dummy backend (smoke tests only)
# --------------------------------------------------------------------------- #

class DummyBackend(SegBackend):
    """Degradation-sensitive colour-template segmenter.

    FOR PIPELINE VALIDATION ONLY. Never used in reported experiments.
    It is designed to be sensitive to colour shift and contrast loss so that the
    smoke test exercises the full DAG-RS decision path meaningfully.
    """

    name = "dummy"

    def __init__(self, tau: float = 0.28, spatial_sigma: float = 0.45):
        self.tau = float(tau)
        self.spatial_sigma = float(spatial_sigma)
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_std: Optional[np.ndarray] = None
        self._n_calls = 0

    def set_reference(self, img: np.ndarray, mask: np.ndarray) -> None:
        lab = self._lab(img)
        m = as_bool(mask)
        if m.sum() < 4:
            self._ref_mean = lab.reshape(-1, 3).mean(axis=0)
            self._ref_std = lab.reshape(-1, 3).std(axis=0) + 1e-3
        else:
            px = lab[m]
            self._ref_mean = px.mean(axis=0)
            self._ref_std = px.std(axis=0) + 1e-3

    @staticmethod
    def _lab(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2LAB).astype(np.float32)

    @property
    def n_encoder_calls(self) -> int:
        return self._n_calls

    def predict(self, img: np.ndarray, box: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                multimask: bool = False,
                cache_token: Optional[Any] = None,
                want_soft: bool = False,
                ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        self._n_calls += 1
        if self._ref_mean is None:
            raise RuntimeError("DummyBackend.set_reference must be called first")
        lab = self._lab(img)
        d = np.linalg.norm((lab - self._ref_mean[None, None, :]) /
                           self._ref_std[None, None, :], axis=2)
        soft = np.exp(-d / (self.tau * 10.0)).astype(np.float32)

        h, w = soft.shape
        if box is not None:
            x0, y0, x1, y1 = [float(v) for v in np.asarray(box).reshape(4)]
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            diag = max(4.0, np.hypot(x1 - x0, y1 - y0))
            ys, xs = np.mgrid[0:h, 0:w]
            r2 = ((xs - cx) / (self.spatial_sigma * diag)) ** 2 + \
                 ((ys - cy) / (self.spatial_sigma * diag)) ** 2
            soft = soft * np.exp(-0.5 * r2)
        elif mask is not None:
            m = as_bool(mask)
            if m.sum() > 0:
                ys, xs = np.mgrid[0:h, 0:w]
                cx, cy = xs[m].mean(), ys[m].mean()
                diag = max(4.0, np.hypot(xs[m].max() - xs[m].min(),
                                         ys[m].max() - ys[m].min()))
                r2 = ((xs - cx) / (self.spatial_sigma * diag)) ** 2 + \
                     ((ys - cy) / (self.spatial_sigma * diag)) ** 2
                soft = soft * np.exp(-0.5 * r2)

        pred = soft >= 0.5
        score = float(soft[soft >= 0.5].mean()) if pred.sum() > 0 else 0.0
        return pred, score, (soft if want_soft else None)


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #

def build_backend(kind: str = "sam2", **kw) -> SegBackend:
    kind = kind.lower()
    if kind in ("sam2", "sam", "auto"):
        return SAM2Backend(SAM2Config(**{k: v for k, v in kw.items()
                                         if k in SAM2Config.__dataclass_fields__}))
    if kind == "dummy":
        return DummyBackend(**{k: v for k, v in kw.items()
                               if k in ("tau", "spatial_sigma")})
    raise KeyError(f"unknown backend '{kind}'")
