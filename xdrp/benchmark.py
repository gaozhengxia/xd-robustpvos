"""Benchmark driver: builds the XD-RobustPVOS protocol and runs any config on it.

The driver is the single place where degraded frames are produced, so every mode
sees byte-identical inputs. Degraded frames can be materialised to disk
(recommended: reproducible across runs and cheap to re-use) or generated on the
fly (saves disk, costs CPU).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2

from .datasets import FrameSource
from .degradations import (ALL_DEGRADATION_NAMES, COMPOUND_DEGRADATIONS,
                           DEGRADATIONS, _stable_hash, apply_degradation,
                           is_compound, members, severity_schedule, to_float, to_u8)
from .iqa import compute_iqa, ALL_IQA_METRICS
from .metrics import jf, seq_metrics
from .operators import OperatorBank
from .pipeline import DAGRSConfig, SequenceResult, run_sequence
from .sam_backend import SegBackend


# --------------------------------------------------------------------------- #
# protocol description
# --------------------------------------------------------------------------- #

@dataclass
class Protocol:
    """One evaluation cell: (dataset, corruption, severity, sequence subset)."""
    degradation: str
    level: float                 # nominal severity in [1, 5]
    frame_stride: int = 1        # temporal subsampling of the sequence
    max_frames: int = 0          # 0 = no cap
    temporal_variation: bool = True
    severity_amp: float = 0.30
    seed: int = 0

    def key(self) -> str:
        return f"{self.degradation}_L{self.level:g}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def protocol_grid(degradations: Sequence[str], levels: Sequence[float],
                  frame_stride: int = 1, max_frames: int = 0,
                  seed: int = 0, **kw) -> List[Protocol]:
    return [Protocol(degradation=d, level=float(l), frame_stride=frame_stride,
                     max_frames=max_frames, seed=seed, **kw)
            for d in degradations for l in levels]


# --------------------------------------------------------------------------- #
# degraded sequence loading / materialisation
# --------------------------------------------------------------------------- #

class DegradedSequence:
    """Holds the frames for one (dataset sequence, protocol) pair."""

    def __init__(self, clean: List[np.ndarray], degraded: List[np.ndarray],
                 first_masks: Dict[int, np.ndarray],
                 gt_masks: Dict[int, List[np.ndarray]],
                 severity: np.ndarray, protocol: Protocol, seq_name: str):
        self.clean = clean
        self.degraded = degraded
        self.first_masks = first_masks
        self.gt_masks = gt_masks
        self.severity = severity
        self.protocol = protocol
        self.seq_name = seq_name

    @property
    def n_frames(self) -> int:
        return len(self.degraded)

    @property
    def object_ids(self) -> List[int]:
        return sorted(self.first_masks.keys())

    def clean_float(self) -> List[np.ndarray]:
        """Clean frames as float32 in [0,1] (used by the `oracle_clean` arm)."""
        return [to_float(c) for c in self.clean]

    def frames_for_mode(self, mode: str) -> List[np.ndarray]:
        """Which observation the segmenter sees for a given experiment arm.

        `oracle_clean` intentionally receives the CLEAN frames: it is the upper
        bound that tells a reader how much of the gap is recoverable at all.
        Every other arm sees the degraded frames.
        """
        if mode == "oracle_clean":
            return self.clean_float()
        return self.degraded


def _frame_indices(n: int, stride: int, max_frames: int) -> List[int]:
    idx = list(range(0, n, max(1, int(stride))))
    if max_frames and len(idx) > int(max_frames):
        idx = idx[:int(max_frames)]
    return idx


def load_degraded_sequence(src: FrameSource, seq: str, proto: Protocol,
                           max_objects: int = 0) -> DegradedSequence:
    n = src.n_frames(seq)
    idx = _frame_indices(n, proto.frame_stride, proto.max_frames)
    clean = [src.frame(seq, t) for t in idx]
    T = len(clean)

    if proto.degradation == "clean":
        degraded = [c.astype(np.float32) / 255.0 for c in clean]
        sev = np.ones(T, np.float32)
    else:
        seq_seed = int(proto.seed) + _stable_hash(seq) % 100000
        if proto.temporal_variation:
            sev = severity_schedule(T, base=float(proto.level),
                                    amp=float(proto.severity_amp),
                                    seed=seq_seed)
        else:
            sev = np.full(T, float(proto.level), np.float32)
        degraded = [apply_degradation(c, proto.degradation, float(sev[i]),
                                      seed=seq_seed,
                                      stream_id=int(idx[i]))
                    for i, c in enumerate(clean)]

    obj_ids = src.object_ids(seq)
    if max_objects:
        obj_ids = obj_ids[:int(max_objects)]
    first = {o: src.mask(seq, idx[0], o) for o in obj_ids}
    gts = {o: [src.mask(seq, t, o) for t in idx] for o in obj_ids}

    return DegradedSequence(clean, degraded, first, gts, sev, proto, seq)


def materialize_sequence(dseq: DegradedSequence, out_dir: str,
                         obj_ids: Optional[Sequence[int]] = None) -> str:
    """Write degraded frames and GT masks as PNG/JPG so a run is byte-reproducible."""
    root = Path(out_dir) / dseq.seq_name / dseq.protocol.key()
    img_dir = root / "JPEGImages"
    ann_dir = root / "Annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(dseq.degraded):
        cv2.imwrite(str(img_dir / f"{i:05d}.jpg"), to_u8(fr)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 98])
    for o in (obj_ids or dseq.object_ids):
        for i, m in enumerate(dseq.gt_masks[o]):
            cv2.imwrite(str(ann_dir / f"{o}_{i:05d}.png"),
                        (m.astype(np.uint8) * 255))
    (root / "protocol.json").write_text(json.dumps(
        {"protocol": dseq.protocol.to_dict(),
         "severity": [float(v) for v in dseq.severity],
         "n_frames": dseq.n_frames,
         "objects": [int(o) for o in (obj_ids or dseq.object_ids)]},
        indent=2), encoding="utf-8")
    return str(root)


# --------------------------------------------------------------------------- #
# one evaluation cell
# --------------------------------------------------------------------------- #

def evaluate_cell(dseq: DegradedSequence, backend: SegBackend, bank: OperatorBank,
                  cfg: DAGRSConfig, dilation: int = 2,
                  collect_iqa: bool = True,
                  max_objects: int = 0,
                  iqa_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                  ) -> List[Dict[str, Any]]:
    """Run one config on one degraded sequence; return one record per object.

    `iqa_cache` lets a caller that evaluates several modes on the SAME degraded
    sequence reuse the IQA panel across them. The key embeds the mode, so a mode
    that sees different frames (``oracle_clean``) can never hit another mode's
    entry. Pass ``None`` (the default) for a standalone call.
    """
    recs: List[Dict[str, Any]] = []
    obj_ids = dseq.object_ids
    if max_objects:
        obj_ids = obj_ids[:int(max_objects)]

    if not obj_ids:
        # A cell that finds no object returns zero rows and therefore vanishes
        # from every downstream table with no error. That is how a broken label
        # reader went unnoticed for a week (see xdrp/datasets.py:read_label_png).
        # Fail loudly instead.
        import sys
        print(f"[WARN] {dseq.seq_name} / {dseq.protocol.key()}: frame-0 annotation "
              f"contains no object -> this cell contributes ZERO rows and will be "
              f"missing from the results. Check the label reader.",
              file=sys.stderr)
        return recs

    # IQA describes the OBSERVATION (a degraded frame against its clean
    # reference), never the object, so it is identical for every instance of a
    # sequence -- but this function is entered once per object. Computing it
    # inside the loop below therefore repeated the whole panel (measured 354
    # ms/frame, of which SSIM alone is 243 ms) 2-5 times per frame: DAVIS val has
    # 61 instances over 30 sequences, so the sweep was ~2x longer than necessary.
    # Hoisted, every value is bit-identical except for SSIM's separable-window
    # rounding (|diff| = 1e-16, see xdrp/iqa.py::ssim).
    iqa_by_frame: Optional[List[Dict[str, Any]]] = None
    if collect_iqa:
        obs = dseq.frames_for_mode(cfg.mode)
        # The panel is a property of the observation, so within one call every
        # instance must reuse it -- the key deliberately omits obj_id. It DOES
        # embed the mode, because `frames_for_mode` can hand a mode different
        # frames (`oracle_clean` sees the clean ones), so no cross-mode reuse is
        # possible or wanted. Net effect: computed once per observation instead
        # of once per (instance, mode) pair.
        if iqa_cache is None:
            iqa_cache = {}
        _iqa_key = f"{dseq.seq_name}|{dseq.protocol.key()}|{cfg.mode}"
        iqa_by_frame = iqa_cache.get(_iqa_key)
        if iqa_by_frame is None:
            iqa_by_frame = [compute_iqa(obs[t], clean=dseq.clean[t])
                            for t in range(len(obs))]
            iqa_cache[_iqa_key] = iqa_by_frame

    for o in obj_ids:
        backend.clear_cache()
        # The `oracle_clean` arm must genuinely see the clean frames; feeding it
        # degraded ones would silently make it a duplicate of `greedy`.
        frames = dseq.frames_for_mode(cfg.mode)
        res: SequenceResult = run_sequence(
            frames, dseq.first_masks[o], backend, bank, cfg,
            seq_name=dseq.seq_name, obj_id=int(o),
            flow_frames=frames)

        gts = dseq.gt_masks[o]
        preds = res.masks
        m = seq_metrics(preds, gts, dilation=dilation)

        rec: Dict[str, Any] = {
            "seq": dseq.seq_name,
            "obj_id": int(o),
            "mode": cfg.mode,
            "degradation": dseq.protocol.degradation,
            "is_compound": int(is_compound(dseq.protocol.degradation)),
            "level": float(dseq.protocol.level),
            "severity_mean": float(np.mean(dseq.severity)),
            "severity_max": float(np.max(dseq.severity)),
            "frame_stride": int(dseq.protocol.frame_stride),
            "n_frames": int(m.get("n_frames", 0)),
            "J": m["J"], "F": m["F"], "J&F": m["J&F"],
            "n_decisions": int(res.n_decisions),
            "n_reanchors": int(res.n_reanchors),
            # The acceptance policy is part of an arm's identity: two rows that
            # agree on (seq, mode, degradation, level) but differ in `tau_score`
            # or `reseed_stride` are different experiments. Without these
            # columns the CSV cannot tell them apart, which is exactly how the
            # config-whitelist drift in scripts/_common.py stayed invisible.
            "tau_score": float(cfg.tau_score),
            "reseed_stride": int(cfg.reseed_stride),
            # Counted, not assumed: `n_rejected == 0` on a gated arm means the
            # gate never fired and the arm is really the ungated one.
            "n_rejected": int(res.n_rejected),
            "n_reseeds": int(res.n_reseeds),
            "n_encoder_calls": int(res.n_encoder_calls),
            # Counted, not assumed: a non-zero value means `backend.predict`
            # raised and the candidate mask silently became the warped anchor,
            # so this row measures optical-flow propagation, not segmentation.
            # Without it in the CSV that degradation is invisible downstream.
            "n_predict_failures": int(res.n_predict_failures),
            "wall_time_s": float(res.wall_time_s),
        }

        if int(res.n_predict_failures) > 0:
            print(f"[WARN] {dseq.seq_name}/obj{o} mode={cfg.mode}: "
                  f"{res.n_predict_failures} backend.predict call(s) fell back to the "
                  f"warped anchor -> this row is NOT a clean segmentation "
                  f"measurement.", file=sys.stderr)

        # per-frame diagnostics + IQA correlation rows (CL3)
        if collect_iqa and iqa_by_frame is not None:
            for t in range(min(len(preds), len(gts))):
                j, f, jf_val = jf(preds[t], gts[t], dilation)
                # IQA is measured on the observation the segmenter actually saw
                # (hoisted above; it does not depend on the object).
                iqa = iqa_by_frame[t]
                rec_t = dict(rec)
                rec_t.update({"frame": int(t), "J_frame": j, "F_frame": f,
                              "JF_frame": jf_val,
                              "vacuity": float(res.vacuity[t]),
                              "selected_op": int(res.selected_op[t]),
                              "n_evaluated": int(res.n_evaluated[t]),
                              "agree_a": float(res.agree_a[t]),
                              "agree_c": float(res.agree_c[t]),
                              "agree_u": float(res.agree_u[t]),
                              "is_keyframe": int(res.is_keyframe[t]),
                              "clean_reference": int(dseq.protocol.degradation == "clean")})
                rec_t.update(iqa)
                recs.append(rec_t)
        else:
            recs.append(rec)
    return recs


# --------------------------------------------------------------------------- #
# full sweep
# --------------------------------------------------------------------------- #

@dataclass
class SweepConfig:
    dataset_root: str = "data/DAVIS"
    dataset_kind: str = "davis"
    dataset_split: str = "val"
    degradations: List[str] = field(default_factory=lambda: ["clean", "fog"])
    levels: List[float] = field(default_factory=lambda: [3.0])
    modes: List[str] = field(default_factory=lambda: ["greedy", "dagrs"])
    frame_stride: int = 4
    max_frames: int = 0
    max_sequences: int = 0
    #: Skip the first N sequences. Only purpose: split a long sweep into chunks
    #: that each fit inside a single shell call, without changing the protocol.
    seq_offset: int = 0
    #: 0 = score every annotated instance of a sequence, then average -- the
    #: official DAVIS protocol and the default since 2026-09-17. It used to be 1,
    #: which is NOT a valid sub-protocol: `:1` takes the lowest label id, and on
    #: DAVIS that is often a sliver (lab-coat's instance 1 is 18 px). Comparing
    #: such a number against published DAVIS J&F is meaningless.
    max_objects: int = 0
    seed: int = 0
    dilation: int = 2
    collect_iqa: bool = True
    materialize_dir: str = ""
    resume_file: str = ""          # append results to this CSV and skip done cells
    #: Append every finished cell to this CSV as soon as it is computed. Set
    #: together with `resume_file` (same path) this makes a long sweep safe to
    #: interrupt: a cell is written only when complete, so a killed run loses at
    #: most the cell in flight, and re-running skips what is done. Without it a
    #: 3-hour sweep that dies at hour 2 keeps nothing, because the CSV is written
    #: once at the very end.
    append_file: str = ""


def run_sweep(src: FrameSource, backend: SegBackend, bank: OperatorBank,
              cfg_by_mode: Dict[str, DAGRSConfig], sweep: SweepConfig,
              progress=None) -> List[Dict[str, Any]]:
    from .pipeline import make_config

    seqs = src.sequences()
    if sweep.seq_offset:
        seqs = seqs[int(sweep.seq_offset):]
    if sweep.max_sequences:
        seqs = seqs[:int(sweep.max_sequences)]

    all_recs: List[Dict[str, Any]] = []
    done = set()
    if sweep.resume_file and Path(sweep.resume_file).exists():
        import csv
        with open(sweep.resume_file, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                # Cell-level key, deliberately WITHOUT obj_id: the old marker
                # hardcoded obj_id "1", so with max_objects > 1 (the default
                # since 2026-09-17) every non-first instance would be re-run
                # while the cell still looked finished. A cell counts as done as
                # soon as it has any row -- if a run was killed mid-cell, delete
                # that cell's rows from the CSV before resuming.
                done.add((row.get("seq"), row.get("mode"), row.get("degradation"),
                          str(row.get("level"))))
    total = len(seqs) * len(sweep.degradations) * len(sweep.levels) * len(sweep.modes)
    step = 0

    for seq in seqs:
        for deg in sweep.degradations:
            for lvl in sweep.levels:
                lvl_s = str(float(lvl))
                todo = [m for m in sweep.modes
                        if (seq, m, deg, lvl_s) not in done]
                # Synthesising the degraded frames is pure CPU work that a
                # resumed run would otherwise redo for every finished cell. Load
                # lazily, inside the mode loop, so a fully-done cell costs
                # nothing but the counter update.
                proto = None
                dseq = None
                # One IQA panel per (sequence, protocol, mode); reset with the
                # cell so it can never carry another cell's panel. The key inside
                # `evaluate_cell` embeds seq/protocol/mode as well, so a hit is
                # only ever a genuine identity.
                iqa_cache: Dict[str, List[Dict[str, Any]]] = {}
                for mode in sweep.modes:
                    step += 1
                    if progress:
                        tag = "" if mode in todo else "  (already done)"
                        progress(step, total, f"{seq} | {deg}@{lvl} | {mode}{tag}")
                    if mode not in todo:
                        continue
                    if dseq is None:
                        proto = Protocol(degradation=deg, level=float(lvl),
                                         frame_stride=sweep.frame_stride,
                                         max_frames=sweep.max_frames,
                                         seed=sweep.seed)
                        dseq = load_degraded_sequence(src, seq, proto,
                                                      max_objects=sweep.max_objects)
                        if sweep.materialize_dir:
                            materialize_sequence(dseq, sweep.materialize_dir)
                    cfg = cfg_by_mode.get(mode) or make_config(mode)
                    recs = evaluate_cell(dseq, backend, bank, cfg,
                                         dilation=sweep.dilation,
                                         collect_iqa=sweep.collect_iqa,
                                         max_objects=sweep.max_objects,
                                         iqa_cache=iqa_cache)
                    if sweep.append_file:
                        maybe_append_csv(recs, sweep.append_file)
                    all_recs.extend(recs)
    return all_recs


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return str(p)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return str(p)


def maybe_append_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return str(p)
    keys = list(rows[0].keys())
    exists = p.exists() and p.stat().st_size > 0
    with open(p, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return str(p)
