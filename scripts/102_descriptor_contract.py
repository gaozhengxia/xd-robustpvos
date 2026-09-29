"""102 - `descriptor` dimension contract: is the output shape input-independent?

    python scripts/102_descriptor_contract.py

READ-ONLY, GPU (loads SAM 2.1 once). Walks the descriptor call sites that
`pipeline.run_sequence` actually reaches and prints the shape at each one.

WHY THIS EXISTS
---------------
`SAM2Backend.descriptor` returns `concat(colour_histogram(24), pooled_embedding)`
when the image embedding is available and the colour histogram ALONE otherwise.
`pipeline.py:267` built `appearance` BEFORE any `predict` (hence before
`set_image`), so it landed on the short branch. The main-loop EMA update at
`pipeline.py:427` is guarded by `d_new.size == appearance.size` and therefore
never fired, and `global_reanchor` later compared that frozen 24-vector against
a 280-vector and raised

    ValueError: shapes (24,) and (280,) not aligned

The fix has two layers, and this script checks both:

  1. `SAM2Backend.set_reference` really encodes the reference frame, so the
     call sequence `set_reference -> descriptor` (pipeline.py:265 -> 267) gets
     the full-width descriptor instead of the colour-only fallback.
  2. `descriptor` fixes its width once an embedding has been obtained
     (`_pooled_dim`) and zero-pads when the embedding is unavailable (empty
     mask, encoder failure) instead of returning a shorter vector.

The one remaining short output is the honest fallback when NO embedding has
ever been extracted (`_pooled_dim is None`): the width is then genuinely
unknown, so the colour histogram is all that can be returned. Step (1) below
measures it; after fix (1) `run_sequence` never reaches that state, because
`set_reference` runs before the first descriptor call.

A probe that reports a NEGATIVE result must first prove itself on a known
positive, so the script asserts the colour part is 24 dims and exits non-zero
with a verdict line when the contract is broken.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xdrp.evidence import color_descriptor          # noqa: E402
from xdrp.sam_backend import SAM2Backend, SAM2Config  # noqa: E402


def synth(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((h, w, 3), dtype=np.float32)
    img[80:160, 100:200] += 0.6                     # a bright blob to segment
    img = np.clip(img, 0, 1)
    mask = np.zeros((h, w), bool)
    mask[80:160, 100:200] = True
    return img, mask


def main() -> int:
    print("=" * 78)
    print("descriptor dimension contract (SAM 2.1 Hiera-L)")
    img, mask = synth()

    # ---- known-positive anchor: the colour part really is 24 dims ---------- #
    col = color_descriptor(img, mask)
    ok_colour = (col.shape == (24,))
    print(f"  known-positive: colour descriptor alone   : shape {col.shape}")

    backend = SAM2Backend(SAM2Config(checkpoint=str(ROOT / "checkpoints"
                                                    / "sam2.1_hiera_large.pt"),
                                     model_cfg=str(ROOT / "configs" / "sam2.1"
                                                   / "sam2.1_hiera_l.yaml"),
                                     device="cuda"))

    def shape_of(m):
        try:
            return backend.descriptor(img, m).shape
        except Exception as e:                      # pragma: no cover
            return f"raised {type(e).__name__}: {e}"

    # ---- (1) cold backend: no embedding has ever been extracted ----------- #
    s_cold = shape_of(mask)
    print(f"  (1) descriptor, nothing encoded yet       : shape {s_cold}"
          f"   <- honest fallback, width unknown")

    # ---- (2) the pipeline's init sequence, post-fix ----------------------- #
    backend.set_reference(img, mask)
    s_ref = shape_of(mask)
    print(f"  (2) descriptor AFTER set_reference        : shape {s_ref}"
          f"   <- pipeline.py:265 -> 267 (the fix)")

    # ---- (3) after a real predict ---------------------------------------- #
    try:
        backend.predict(img, box=None, mask=mask.astype(np.uint8) * 255)
    except Exception as e:                          # pragma: no cover
        print(f"  predict() raised {type(e).__name__}: {e}")
    s_after = shape_of(mask)
    print(f"  (3) descriptor AFTER a predict            : shape {s_after}")

    # ---- (4) empty mask while the embedding is warm ---------------------- #
    s_empty = shape_of(np.zeros_like(mask))
    print(f"  (4) descriptor, EMPTY mask, warm encoder  : shape {s_empty}"
          f"   <- must be zero-padded, not shortened")

    # ---- (5) the comparison global_reanchor performs --------------------- #
    warm = [s_ref, s_after, s_empty]
    contract = (all(isinstance(s, tuple) for s in warm)
                and len(set(warm)) == 1)
    if contract:
        a = backend.descriptor(img, mask)
        b = backend.descriptor(img, np.zeros_like(mask))
        dot_ok = True
        try:
            np.dot(a, b)
        except ValueError as e:                     # pragma: no cover
            dot_ok = False
            print(f"  (5) np.dot(appearance, descriptor) -> {type(e).__name__}: {e}")
        else:
            print(f"  (5) np.dot(appearance, descriptor) -> ok "
                  f"({a.shape[0]}-dim)")
    else:
        dot_ok = False
        print(f"  (5) warm shapes differ: {warm}")

    # ---- verdict ---------------------------------------------------------- #
    print("-" * 78)
    print(f"  known-positive anchor (colour == 24 dims): "
          f"{'ok' if ok_colour else 'FAIL'}")
    if not ok_colour:
        print("  ==> PROBE ITSELF IS WRONG (colour part is not 24 dims); "
              "refusing to interpret.")
        return 2
    if contract and dot_ok:
        print(f"  ==> CONTRACT HOLDS: every warm path returns {warm[0][0]} dims; "
              f"the EMA\n      guard and `global_reanchor` comparison are both "
              f"well-typed.\n"
              f"      (cold fallback alone returns {s_cold[0] if isinstance(s_cold, tuple) else s_cold} "
              f"dims -- unreachable from `run_sequence`\n"
              f"      now that `set_reference` encodes the reference frame.)")
        return 0
    print("  ==> CONTRACT VIOLATED: descriptor shape depends on whether the "
          "image embedding\n      is warm. Any code that caches a descriptor "
          "from one call site and compares it\n      against another call site "
          "is broken -- which is exactly what `appearance`\n      (pipeline.py:"
          "267) vs `global_reanchor` (pipeline.py:324) does.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
