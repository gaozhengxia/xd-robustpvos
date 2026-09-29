"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xdrp.operators import OperatorBank, build_default_bank      # noqa: E402
from xdrp.pipeline import DAGRSConfig, make_config               # noqa: E402
from xdrp.sam_backend import build_backend                       # noqa: E402


# --------------------------------------------------------------------------- #
# yaml
# --------------------------------------------------------------------------- #

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_get(d: Dict[str, Any], dotted: str, default=None):
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--backend", default=None, choices=["sam2", "dummy"],
                    help="override backend kind")
    ap.add_argument("--dataset", default=None, help="override dataset root")
    ap.add_argument("--split", default=None)
    ap.add_argument("--degradations", default=None,
                    help="comma-separated; 'single' / 'compound' / 'all' shortcuts accepted")
    ap.add_argument("--levels", default=None, help="comma-separated severities, e.g. 1,3,5")
    ap.add_argument("--frame-stride", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-sequences", type=int, default=None)
    ap.add_argument("--seq-offset", type=int, default=None,
                    help="skip the first N sequences (chunk a long sweep; "
                         "does not change the protocol)")
    ap.add_argument("--max-objects", type=int, default=None,
                    help="cap the number of instances scored per sequence; "
                         "omit or 0 = all instances (official DAVIS protocol)")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--resume", action="store_true",
                    help="append each finished cell to --out as it completes and "
                         "skip cells already present. Makes a long sweep safe to "
                         "interrupt and cheap to restart")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mask-prompt", default=None, choices=["hard", "soft"],
                    help="override dagrs.mask_prompt: 'hard' re-prompts with a "
                         "binarised mask as a +-8 logit wall; 'soft' carries the "
                         "segmenter's own probability across frames so the "
                         "boundary is re-decided from the current image")
    ap.add_argument("--no-iqa", action="store_true",
                    help="skip per-frame IQA (much faster)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny setting for a first end-to-end check")
    return ap


def preset(ap_args) -> Dict[str, Any]:
    """Return an effective setting dict merging file config and CLI overrides."""
    cfg = load_yaml(ap_args.config)
    # Fold the CLI override back into the in-memory yaml dict so that every
    # caller of `config_for_mode` picks it up without a per-script parameter.
    mp = getattr(ap_args, "mask_prompt", None)
    if mp:
        cfg.setdefault("dagrs", {})["mask_prompt"] = str(mp)
    if ap_args.quick:
        ap_args.frame_stride = ap_args.frame_stride or 8
        ap_args.max_frames = ap_args.max_frames or 6
        ap_args.max_sequences = ap_args.max_sequences or 2
        ap_args.levels = ap_args.levels or "3"
        ap_args.degradations = ap_args.degradations or "fog,motion_blur"

    kind = ap_args.backend or deep_get(cfg, "backend.kind", "sam2")
    return {
        "dataset_kind": deep_get(cfg, "dataset.kind", "davis"),
        "dataset_root": ap_args.dataset or deep_get(cfg, "dataset.root", "data/DAVIS"),
        "split": ap_args.split or deep_get(cfg, "dataset.split", "val"),
        "backend_kind": kind,
        "frame_stride": ap_args.frame_stride if ap_args.frame_stride is not None
        else deep_get(cfg, "protocol.frame_stride", 4),
        "max_frames": ap_args.max_frames if ap_args.max_frames is not None
        else deep_get(cfg, "protocol.max_frames", 0),
        "max_sequences": ap_args.max_sequences if ap_args.max_sequences is not None
        else deep_get(cfg, "protocol.max_sequences", 0),
        "seq_offset": ap_args.seq_offset if ap_args.seq_offset is not None
        else deep_get(cfg, "protocol.seq_offset", 0),
        # 0 = every annotated instance, i.e. the official DAVIS protocol (per
        # sequence J&F is the mean over its objects). The fallback matches the
        # yaml and SweepConfig; all three were `1` before 2026-09-17, which
        # silently scored 17 of the 30 val sequences on a single -- sometimes
        # 18-pixel -- instance. See docs/BASELINE_DIAGNOSIS.md section 6.
        "max_objects": ap_args.max_objects if ap_args.max_objects is not None
        else deep_get(cfg, "protocol.max_objects", 0),
        "dilation": deep_get(cfg, "protocol.dilation", 2),
        "severity_amp": deep_get(cfg, "protocol.severity_amp", 0.30),
        "seed": ap_args.seed if ap_args.seed is not None
        else deep_get(cfg, "protocol.seed", 0),
        "collect_iqa": not ap_args.no_iqa,
        "results_dir": deep_get(cfg, "output.results_dir", "results"),
        "figs_dir": deep_get(cfg, "output.figs_dir", "results/figs"),
        "cfg": cfg,
    }


def resolve_degradations(spec: Optional[str], cfg: Dict[str, Any],
                         which: str = "sweep") -> list:
    from xdrp.degradations import ALL_DEGRADATION_NAMES, COMPOUND_DEGRADATIONS
    if spec is None:
        return list(deep_get(cfg, f"{which}.degradations", ["clean", "fog"]))
    s = spec.strip().lower()
    if s == "single":
        return [d for d in ALL_DEGRADATION_NAMES if d not in COMPOUND_DEGRADATIONS]
    if s == "compound":
        return list(COMPOUND_DEGRADATIONS.keys())
    if s == "all":
        return list(ALL_DEGRADATION_NAMES)
    return [x.strip() for x in spec.split(",") if x.strip()]


def resolve_levels(spec: Optional[str], cfg: Dict[str, Any]) -> list:
    if spec is None:
        return [float(v) for v in deep_get(cfg, "sweep.levels", [3.0])]
    return [float(x) for x in spec.split(",") if x.strip()]


# --------------------------------------------------------------------------- #
# objects
# --------------------------------------------------------------------------- #

def make_bank(cfg: Dict[str, Any]) -> OperatorBank:
    return OperatorBank(build_default_bank(
        include_slow=bool(deep_get(cfg, "operators.include_slow", False))))


def make_dagrs_config(cfg: Dict[str, Any], **overrides) -> DAGRSConfig:
    d = deep_get(cfg, "dagrs", {}) or {}
    kw = {}
    for k in DAGRSConfig.__dataclass_fields__:
        if k in d:
            kw[k] = d[k]
    if "agreement_weights" in kw:
        kw["agreement_weights"] = tuple(kw["agreement_weights"])
    if "reanchor_scales" in kw:
        kw["reanchor_scales"] = tuple(kw["reanchor_scales"])
    kw.update(overrides)
    c = DAGRSConfig(**kw)
    return c


#: Fields taken from the `dagrs:` block of the yaml config and applied on top of
#: every mode-specific config.
#:
#: The `use_*` switches that define the ablation arms are deliberately EXCLUDED.
#: `make_config` sets them per arm, and copying them back from the yaml silently
#: turns an ablation into a duplicate of the full method. That defect had already
#: shipped: `use_reanchor: true` in `configs/default.yaml` made the
#: `dagrs_no_reanchor` arm (ablation A7) behave exactly like `dagrs` in scripts
#: 03 and 04, because both copied a hand-written field list that included
#: `use_reanchor`.
YAML_DAGRS_TUNABLES: Tuple[str, ...] = (
    "keyframe_stride", "j_steady", "j_cold", "kappa", "gamma", "top_k_fuse",
    "fuse_uniform", "tau_vacuity", "max_consecutive_uncertain",
    "output_when_rejected", "profile_lambda", "reanchor_scales",
    "reanchor_top_ops", "box_pad", "agreement_weights", "mask_prompt",
)


def config_for_mode(cfg: Dict[str, Any], mode: str, **overrides) -> DAGRSConfig:
    """`make_config(mode)` with the yaml `dagrs:` tunables applied on top.

    Use this instead of copying a hand-maintained field list at each call site.
    Those lists had drifted apart (02/03/04/06 each carried a *different*
    subset), so a new yaml knob would silently fail to reach some runners, and
    arm-defining switches leaked through and erased the arm being measured.
    """
    c = make_config(mode)
    base = make_dagrs_config(cfg)
    for k in YAML_DAGRS_TUNABLES:
        if hasattr(base, k):
            setattr(c, k, getattr(base, k))
    for k, v in overrides.items():
        if not hasattr(c, k):
            raise KeyError(f"unknown config field '{k}'")
        setattr(c, k, v)
    return c


def make_backend_for(s: Dict[str, Any]):
    cfg = s["cfg"]
    if s["backend_kind"] == "dummy":
        return build_backend("dummy")
    return build_backend("sam2",
                         checkpoint=deep_get(cfg, "backend.checkpoint"),
                         model_cfg=deep_get(cfg, "backend.model_cfg"),
                         device=deep_get(cfg, "backend.device", "cuda"),
                         autocast_dtype=deep_get(cfg, "backend.autocast_dtype", "bfloat16"),
                         cache_embeddings=bool(deep_get(cfg, "backend.cache_embeddings", True)),
                         cache_size=int(deep_get(cfg, "backend.cache_size", 4)))


def open_src(s: Dict[str, Any]):
    from xdrp.datasets import open_dataset
    return open_dataset(s["dataset_kind"], s["dataset_root"], split=s["split"])


def out_path(s: Dict[str, Any], ap_args, name: str) -> str:
    if ap_args.out:
        return ap_args.out
    return str(ROOT / s["results_dir"] / name)
