"""The DAG-RS inference loop, plus every baseline and ablation variant.

A single code path serves the method AND all baselines/ablations, so that the
comparison in the paper differs only in the configuration flags. This is
deliberate: it removes the most common source of unfair comparisons.

Modes (cfg.mode)
----------------
greedy       : no restoration, warp-and-reprompt propagation        (main baseline)
oracle_clean : same, but fed the CLEAN frames (upper bound reference)
cascade      : always apply one fixed restoration operator, then greedy
fixed_op     : always apply a specified operator (domain prior), then greedy
equal_tta    : evaluate ALL operators every frame, fuse with EQUAL weights
random_op    : evaluate ONE randomly chosen operator per frame
dagrs        : the proposed method (full)
dagrs_no_agree   : DAG-RS but with uniform weights inside the chosen operator set
dagrs_no_profile : DAG-RS but always evaluating all M operators (no profile)
dagrs_no_gating  : DAG-RS but the anchor is always updated (no vacuity gate)
dagrs_no_reanchor: DAG-RS but without global re-anchoring
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .degradations import to_float
from .evidence import (DegradationProfile, agreement_scores, as_bool, bbox_of,
                       descriptor_similarity, dirichlet_from_agreement,
                       fuse_candidates, area)
from .flow import FlowCache
from .operators import OperatorBank
from .sam_backend import SegBackend, as_prediction

EPS = 1e-8


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass
class DAGRSConfig:
    mode: str = "dagrs"

    # decision scheduling
    keyframe_stride: int = 5          # K
    decide_at_first_transition: bool = True

    # operator set
    use_restoration: bool = True
    j_steady: int = 3                 # J: operators evaluated once the profile is warm
    j_cold: int = 0                   # 0 -> all M at the first decision frame
    fixed_operator: str = "id"        # for cascade / fixed_op modes

    # agreement + evidence
    use_agreement: bool = True
    agreement_weights: Tuple[float, float, float] = (0.45, 0.35, 0.20)
    kappa: float = 8.0
    gamma: float = 2.0
    top_k_fuse: int = 2
    #: fuse with EQUAL weights over the same candidate set (ablation A1: isolates
    #: whether the gain comes from the agreement criterion or merely from having
    #: several restored hypotheses at all).
    fuse_uniform: bool = False

    # gating
    use_gating: bool = True
    tau_vacuity: float = 0.55
    max_consecutive_uncertain: int = 2
    output_when_rejected: str = "fused"     # fused | anchor

    # profile
    use_profile: bool = True
    profile_lambda: float = 0.45

    # re-anchoring
    use_reanchor: bool = True
    reanchor_scales: Tuple[float, ...] = (1.0, 1.3, 1.8)
    reanchor_top_ops: int = 2

    # propagation
    tau_score: float = 0.0            # min backend score to accept a propagated mask
    #: Periodic re-seeding from the annotated first frame: the classic drift
    #: correction for a mask-prompt propagation loop that has no memory bank.
    #: `0` disables it. This is a BASELINE knob, not a method component -- the
    #: DAG-RS family handles uncertainty by evidence-driven re-anchoring, and
    #: combining the two would blur the one mechanism difference the paper
    #: rests on (see `_check_reseed_scope`).
    reseed_stride: int = 0
    box_pad: float = 0.12
    random_seed: int = 0
    #: Re-prompt convention:
    #:   "hard" -- warp the binary anchor and prompt SAM 2 with a hard +-8 logit
    #:             wall. Faithful to the released baselines; the default.
    #:   "soft" -- carry the segmenter's own mask probability across frames and
    #:             prompt with its logit, so the boundary can be re-decided from
    #:             the current image instead of being asserted.
    #: See `evidence.soft_mask_to_logits` and docs/BASELINE_DIAGNOSIS.md.
    mask_prompt: str = "hard"

    def is_dagrs_family(self) -> bool:
        return self.mode.startswith("dagrs")


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class SequenceResult:
    seq: str
    obj_id: int
    mode: str
    masks: List[np.ndarray] = field(default_factory=list)
    # per-frame diagnostics (length T, NaN-padded where not applicable)
    vacuity: List[float] = field(default_factory=list)
    selected_op: List[int] = field(default_factory=list)
    n_evaluated: List[int] = field(default_factory=list)
    agree_a: List[float] = field(default_factory=list)
    agree_c: List[float] = field(default_factory=list)
    agree_u: List[float] = field(default_factory=list)
    backend_score: List[float] = field(default_factory=list)
    is_keyframe: List[int] = field(default_factory=list)
    n_decisions: int = 0
    n_reanchors: int = 0
    #: Frames where the propagated mask was refused and the flow-warped anchor
    #: was emitted instead (`tau_score` gate). 0 means the gate never fired, so
    #: a "gated" arm that counts 0 rejections is really the ungated arm.
    n_rejected: int = 0
    #: Times the anchor was reset to the annotated first frame.
    n_reseeds: int = 0
    n_encoder_calls: int = 0
    #: Number of backend calls that raised and silently fell back to the warped
    #: anchor. Non-zero means the run is NOT a valid measurement -- see
    #: `as_prediction` in sam_backend.py for why this must stay visible.
    n_predict_failures: int = 0
    wall_time_s: float = 0.0

    def diag_dict(self) -> Dict[str, List[float]]:
        return {
            "frame": list(range(len(self.masks))),
            "vacuity": self.vacuity,
            "selected_op": self.selected_op,
            "n_evaluated": self.n_evaluated,
            "agree_a": self.agree_a,
            "agree_c": self.agree_c,
            "agree_u": self.agree_u,
            "backend_score": self.backend_score,
            "is_keyframe": self.is_keyframe,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def expand_box(box: np.ndarray, scale: float, shape: Tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in np.asarray(box).reshape(4)]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw, hh = max(1.0, 0.5 * (x1 - x0)), max(1.0, 0.5 * (y1 - y0))
    h, w = int(shape[0]), int(shape[1])
    nx0 = max(0.0, cx - hw * float(scale))
    nx1 = min(w - 1.0, cx + hw * float(scale))
    ny0 = max(0.0, cy - hh * float(scale))
    ny1 = min(h - 1.0, cy + hh * float(scale))
    return np.array([nx0, ny0, nx1, ny1], np.float32)


#: --- arm registries ------------------------------------------------------- #
#: Which structural branch an arm takes used to be decided by enumerating mode
#: strings at each site (`cfg.mode == "greedy" or cfg.mode == "oracle_clean"`, ...).
#: Adding an arm then meant finding every site; missing one sent the new arm down
#: the DAG-RS branch, where `operator_list` hands it the restoration bank it was
#: never supposed to have. The arm would still run and still print plausible J&F,
#: so the mistake is invisible in the results -- the same failure class as the
#: config-whitelist drift in scripts/_common.py. `run_sequence` now refuses a mode
#: that is in none of these groups.

#: Single-mask prompt propagation, no restoration operator, no keyframe schedule.
PROPAGATION_MODES: Tuple[str, ...] = (
    "greedy", "greedy_gate", "greedy_reseed", "greedy_robust", "oracle_clean",
)

#: Always apply one fixed restoration operator, then propagate.
FIXED_OPERATOR_MODES: Tuple[str, ...] = ("cascade", "fixed_op")

#: Evaluate the whole bank every frame, no decision schedule.
TTA_MODES: Tuple[str, ...] = ("equal_tta", "random_op")

#: Whole-video arms: a memory-based tracker consumes the ENTIRE clip at once and
#: cannot be expressed as a per-frame backend, so `run_sequence` dispatches these
#: to `xdrp/video_backend.run_video_tracking` instead of the loop below. They are
#: registered here for the same reason the other groups are: an arm that is in no
#: group used to fall through to the DAG-RS branch and borrow the restoration bank
#: it was never meant to have, while still printing a plausible J&F.
VIDEO_MODES: Tuple[str, ...] = ("sam2video",)

#: Arms that never take a keyframe decision.
NO_DECISION_MODES: Tuple[str, ...] = PROPAGATION_MODES + FIXED_OPERATOR_MODES


def _nan(v: float) -> float:
    return float(v) if v is not None else float("nan")


def _run_video_sequence(frames: Sequence[np.ndarray],
                        first_mask: np.ndarray,
                        backend: SegBackend,
                        cfg: DAGRSConfig,
                        seq_name: str,
                        obj_id: int) -> SequenceResult:
    """Whole-clip dispatch target for `VIDEO_MODES` (see `xdrp/video_backend.py`).

    `backend` is used only as the carrier of the SAM 2 configuration (checkpoint,
    model_cfg, device, autocast dtype), so the memory-based arm runs the same
    weights on the same device as the frame-by-frame arm it is compared against.
    That is a fairness requirement rather than a convenience -- a comparison
    across two different checkpoints or precisions would not isolate the memory
    bank. A backend that carries no SAM 2 config (e.g. `DummyBackend`) is refused
    instead of being silently substituted.

    Frame 0 is reported as the prompt itself, exactly like every other arm
    (`preds = [anchor.copy()]` in `run_sequence`). Uniformity matters here: if
    this arm alone reported the tracker's own frame-0 output, the two arms would
    disagree at frame 0 for a reason that has nothing to do with memory.
    """
    from .video_backend import predictor_for, run_video_tracking

    sam2_cfg = getattr(backend, "cfg", None)
    live = [k for k in ("use_restoration", "use_agreement", "use_gating",
                        "use_profile", "use_reanchor") if getattr(cfg, k, False)]
    if live or float(getattr(cfg, "tau_score", 0.0)) != 0.0 \
            or int(getattr(cfg, "reseed_stride", 0)) != 0:
        # Refused, not warned: this arm exists to isolate the memory bank. With
        # any DAG-RS machinery left on it would no longer be the comparator the
        # claim needs, and it would still print a plausible J&F. Checked BEFORE
        # the backend, because it is a property of the config alone -- putting it
        # second would have made it unreachable in exactly the tests that need it.
        raise ValueError(
            f"mode {cfg.mode!r} is a whole-video arm but carries live mechanism "
            f"settings: {live}, tau_score={getattr(cfg, 'tau_score', None)}, "
            f"reseed_stride={getattr(cfg, 'reseed_stride', None)}. A memory-based "
            f"tracker has none of these; configure it through "
            f"`make_config('sam2video')`.")
    if sam2_cfg is None or not hasattr(sam2_cfg, "checkpoint"):
        raise RuntimeError(
            f"mode {cfg.mode!r} is a whole-video memory-based arm and needs a "
            f"SAM 2 backend to supply checkpoint/model_cfg/device; got "
            f"{type(backend).__name__} with cfg="
            f"{None if sam2_cfg is None else type(sam2_cfg).__name__}. "
            f"Run with the SAM 2 backend (i.e. --backend sam2), not the dummy one.")

    res = SequenceResult(seq=seq_name, obj_id=int(obj_id), mode=cfg.mode)
    t_start = time.time()
    predictor = predictor_for(sam2_cfg,
                              autocast_dtype=getattr(sam2_cfg, "autocast_dtype", None))
    masks, info = run_video_tracking(
        predictor, frames, first_mask,
        autocast_dtype=getattr(sam2_cfg, "autocast_dtype", None))

    T = len(masks)
    if T != len(frames):
        raise RuntimeError(
            f"the video arm returned {T} masks for {len(frames)} frames; refusing "
            f"to score a track whose length does not match the clip.")
    preds = [as_bool(m).copy() for m in masks]
    preds[0] = as_bool(first_mask).copy()
    res.masks = preds

    # This arm has no operator selection, no agreement terms and no gate, so its
    # per-frame diagnostics are NaN. They are padded to the full length T on
    # purpose: a shorter list would silently misalign every diagnostic column of
    # the CSV against the frame index.
    nan = float("nan")
    res.vacuity = [nan] * T
    res.selected_op = [0] * T
    res.n_evaluated = [0] * T
    res.agree_a = [nan] * T
    res.agree_c = [nan] * T
    res.agree_u = [nan] * T
    res.backend_score = [nan] * T
    # Frame 0 is the prompt, so it is the one "keyframe" the arm has -- the same
    # flag every other arm sets for frame 0.
    res.is_keyframe = [0] * T
    res.is_keyframe[0] = 1
    res.n_decisions = 0
    res.n_reanchors = 0
    res.n_rejected = 0
    res.n_reseeds = 0
    res.n_encoder_calls = int(info.get("n_encoder_calls", 0))
    res.n_predict_failures = 0
    res.wall_time_s = float(time.time() - t_start)
    return res


# --------------------------------------------------------------------------- #
# the main loop
# --------------------------------------------------------------------------- #

def run_sequence(frames: Sequence[np.ndarray],
                 first_mask: np.ndarray,
                 backend: SegBackend,
                 bank: OperatorBank,
                 cfg: DAGRSConfig,
                 seq_name: str = "seq",
                 obj_id: int = 1,
                 flow_frames: Optional[Sequence[np.ndarray]] = None,
                 ) -> SequenceResult:
    """Run one configuration over one (sequence, object).

    Parameters
    ----------
    frames : the frames actually fed to the segmenter (degraded, or clean for
        the `oracle_clean` mode).
    first_mask : ground-truth mask on frame 0 (the prompt).
    flow_frames : frames used for optical flow. Defaults to `frames`.
    """
    T = len(frames)
    if T == 0:
        raise ValueError("empty sequence")
    if not cfg.is_dagrs_family() and cfg.mode not in (
            PROPAGATION_MODES + FIXED_OPERATOR_MODES + TTA_MODES + VIDEO_MODES):
        raise KeyError(
            f"mode {cfg.mode!r} is not registered as a propagation arm "
            f"({PROPAGATION_MODES}), a fixed-operator arm ({FIXED_OPERATOR_MODES}), "
            f"a TTA arm ({TTA_MODES}) or a whole-video arm ({VIDEO_MODES}), and it "
            f"is not a DAG-RS arm either. "
            f"Unregistered names used to fall through to the DAG-RS branch, which "
            f"silently handed the arm the restoration bank it was written to not "
            f"have. Register the arm in xdrp/pipeline.py and pick its branch "
            f"explicitly.")
    if cfg.mode in VIDEO_MODES:
        if cfg.reseed_stride:
            raise ValueError(
                f"reseed_stride={cfg.reseed_stride} requested for the whole-video "
                f"arm {cfg.mode!r}. Periodic re-seeding is a property of a "
                f"frame-by-frame loop that has no memory; a memory-based tracker "
                f"does not re-prompt from frame 0, and accepting the setting would "
                f"imply it does.")
        return _run_video_sequence(frames, first_mask, backend, cfg,
                                   seq_name=seq_name, obj_id=obj_id)
    if cfg.reseed_stride and cfg.is_dagrs_family():
        # Refused rather than warned: an arm that silently does both is not a
        # measurement of either. Periodic re-seeding is the baseline's drift
        # correction; the DAG-RS family's answer to drift is evidence-driven
        # re-anchoring, and that difference is the paper's central mechanism
        # claim (docs/NOVELTY_BOUNDARY.md).
        raise ValueError(
            f"reseed_stride={cfg.reseed_stride} requested for mode={cfg.mode!r}. "
            f"Periodic re-seeding is a BASELINE drift correction and must not be "
            f"combined with the DAG-RS family -- running both would stop the arm "
            f"from isolating either mechanism.")
    flow_frames = list(flow_frames) if flow_frames is not None else list(frames)
    flows = FlowCache([to_float(f) for f in flow_frames])

    M = len(bank)
    rng = np.random.default_rng(int(cfg.random_seed) * 10007 + obj_id)

    res = SequenceResult(seq=seq_name, obj_id=int(obj_id), mode=cfg.mode)
    t_start = time.time()

    anchor = as_bool(first_mask).copy()
    anchor_idx = 0
    #: continuous companion of `anchor`, only maintained in "soft" prompt mode.
    soft_anchor = anchor.astype(np.float32)
    use_soft = (str(cfg.mask_prompt).lower() == "soft")
    profile = DegradationProfile(M, lam=cfg.profile_lambda, min_j=cfg.j_steady)

    f0 = to_float(frames[0])
    backend.set_reference(f0, anchor)
    try:
        appearance = backend.descriptor(f0, anchor)
    except Exception:
        appearance = np.zeros(1, np.float32)

    preds: List[np.ndarray] = [anchor.copy()]
    res.vacuity.append(float("nan"))
    res.selected_op.append(0)
    res.n_evaluated.append(0)
    res.agree_a.append(float("nan"))
    res.agree_c.append(float("nan"))
    res.agree_u.append(float("nan"))
    res.backend_score.append(float("nan"))
    res.is_keyframe.append(1)

    consec_uncertain = 0
    n_predict_failures = 0
    enc0 = backend.n_encoder_calls

    # ---- decide which operators to evaluate at this frame ---------------- #
    def operator_list(t: int, is_decision: bool) -> List[int]:
        if not cfg.use_restoration:
            return [0]
        if cfg.mode in FIXED_OPERATOR_MODES:
            return [bank.index_of(cfg.fixed_operator)]
        if cfg.mode in PROPAGATION_MODES:
            return [0]
        if cfg.mode == "equal_tta":
            return profile.all_operators()
        if cfg.mode == "random_op":
            if not is_decision:
                return [int(rng.integers(0, M))]
            return [int(rng.integers(0, M))]
        # dagrs family
        if not is_decision:
            return [profile.argmax() if profile.ready else 0]
        if cfg.mode == "dagrs_no_profile" or not cfg.use_profile or not profile.ready:
            return profile.all_operators() if cfg.j_cold <= 0 else profile.select(cfg.j_steady)
        return profile.select(cfg.j_steady)

    # ---- global re-anchoring --------------------------------------------- #
    def global_reanchor(t: int, anchor_warped: np.ndarray):
        base_box = bbox_of(anchor_warped)
        h, w = anchor_warped.shape[:2]
        ops = profile.select(cfg.reanchor_top_ops) if profile.ready else [0]
        best_mask, best_gain = anchor_warped, -1.0
        for op in ops:
            img_op = bank.apply(op, frames[t])
            for s in cfg.reanchor_scales:
                b = expand_box(base_box, s, (h, w))
                try:
                    out = backend.predict(img_op, box=b, mask=None,
                                          cache_token=(seq_name, t, int(op), float(s)))
                    m, sc, _ = as_prediction(out)
                except Exception:
                    continue
                if m.sum() == 0:
                    continue
                sim = descriptor_similarity(appearance, backend.descriptor(img_op, m))
                a_new, a_ref = area(m), max(area(anchor), 1)
                ratio = min(a_new, a_ref) / float(max(a_new, a_ref))
                gain = 0.6 * float(sim) + 0.4 * float(ratio)
                if gain > best_gain:
                    best_gain, best_mask = gain, m
        return best_mask

    # ---- main loop -------------------------------------------------------- #
    for t in range(1, T):
        if cfg.reseed_stride and (t % int(cfg.reseed_stride) == 0):
            # Drift correction: throw the propagated state away and re-prompt
            # from the annotated first frame. `anchor_idx = 0` (not `t`) so the
            # next warp is measured from the frame the mask actually belongs to.
            # No backend call, so this costs nothing but the flow it displaces.
            anchor = as_bool(first_mask).copy()
            anchor_idx = 0
            soft_anchor = anchor.astype(np.float32)
            try:
                # Re-encode the reference frame first: `descriptor` pools the
                # image the encoder *currently* holds, and by t >= 1 that is a
                # later frame -- which mixed f0's colour histogram with another
                # frame's embedding. `set_reference` makes the two agree.
                backend.set_reference(f0, anchor)
                appearance = backend.descriptor(f0, anchor)
            except Exception:
                pass
            res.n_reseeds += 1

        anchor_warped = flows.warp_mask(anchor, anchor_idx, t)
        soft_warped = (flows.warp_soft_mask(soft_anchor, anchor_idx, t)
                       if use_soft else None)

        if cfg.mode in TTA_MODES:
            is_decision = True
        elif cfg.mode in NO_DECISION_MODES:
            is_decision = False
        else:
            is_decision = (t % int(cfg.keyframe_stride) == 0) or \
                          (cfg.decide_at_first_transition and t == 1)

        op_list = operator_list(t, is_decision)

        # evaluate candidates
        cands: List[np.ndarray] = []
        scores: List[float] = []
        softs: List[Optional[np.ndarray]] = []
        prompt_mask = soft_warped if use_soft else anchor_warped
        prompt_box = bbox_of(prompt_mask, cfg.box_pad)
        for op in op_list:
            img_op = bank.apply(op, frames[t]) if (cfg.use_restoration and M > 0) \
                else to_float(frames[t])
            try:
                out = backend.predict(img_op, box=prompt_box, mask=prompt_mask,
                                      cache_token=(seq_name, t, int(op)),
                                      want_soft=use_soft)
                m, s, sf = as_prediction(out)
            except Exception:
                m, s, sf = anchor_warped, 0.0, soft_warped
                n_predict_failures += 1
                if n_predict_failures == 1:
                    print(f"[WARN] backend.predict failed on {seq_name} "
                          f"obj{obj_id} t={t}; falling back to the warped anchor. "
                          f"Further occurrences are counted, not printed.",
                          file=sys.stderr)
            cands.append(as_bool(m))
            scores.append(float(s))
            softs.append(sf)

        # agreement + fusion
        if len(cands) == 1:
            fused = cands[0]
            vac = float("nan")
            a_, c_, u_ = float("nan"), float("nan"), float("nan")
            evidence = None
        else:
            if cfg.use_agreement and cfg.mode != "dagrs_no_agree":
                A, comps = agreement_scores(cands, anchor_warped, cfg.agreement_weights)
            else:
                A = np.ones(len(cands), np.float64)
                comps = {"a": A.copy(), "c": A.copy(), "u": A.copy()}
            ev = dirichlet_from_agreement(A, cfg.kappa, cfg.gamma)
            evidence = ev
            a_, c_, u_ = (float(comps["a"].mean()), float(comps["c"].mean()),
                          float(comps["u"].mean()))
            top_k = cfg.top_k_fuse
            if cfg.mode in ("equal_tta", "dagrs_no_agree"):
                top_k = len(cands)
            fused, _order = fuse_candidates(cands, ev.belief, top_k=top_k)
            vac = float(ev.vacuity)

        # acceptance logic
        accepted = True
        if is_decision and len(cands) > 1 and cfg.use_gating and cfg.mode.startswith("dagrs"):
            if not np.isnan(vac) and vac >= cfg.tau_vacuity:
                accepted = False
        if not is_decision and cfg.tau_score > 0:
            accepted = accepted and (scores[0] >= cfg.tau_score)

        if accepted:
            anchor = fused.copy()
            anchor_idx = t
            if is_decision:
                # Reset only on decision frames. The gate is evaluated only
                # there, and non-decision frames are accepted by construction
                # when `tau_score == 0`, so resetting on *every* accepted frame
                # pinned this counter to 0/1 and made
                # `max_consecutive_uncertain = 2` unreachable for ANY tau: the
                # re-anchoring mechanism could never fire, however the
                # threshold was moved (see `scripts/98_vac_gate_audit.py`).
                consec_uncertain = 0
            if evidence is not None:
                profile.update(evidence.e, idxs=op_list)
            try:
                d_new = backend.descriptor(bank.apply(op_list[0], frames[t]) if cfg.use_restoration
                                          else to_float(frames[t]), fused)
                if d_new.size == appearance.size and np.linalg.norm(d_new) > 0:
                    appearance = 0.7 * appearance + 0.3 * d_new
                    appearance = appearance / (np.linalg.norm(appearance) + EPS)
            except Exception:
                pass
        else:
            res.n_rejected += 1
            if is_decision:
                consec_uncertain += 1
            if (is_decision and cfg.use_reanchor
                    and consec_uncertain >= int(cfg.max_consecutive_uncertain)):
                new_anchor = global_reanchor(t, anchor_warped)
                anchor = as_bool(new_anchor).copy()
                anchor_idx = t
                consec_uncertain = 0
                res.n_reanchors += 1
                fused = anchor.copy()
            elif cfg.output_when_rejected == "anchor":
                fused = anchor_warped

        if use_soft:
            # Carry the segmenter's own probability forward. Only the
            # single-candidate (pure propagation) path can hand us a soft map
            # that belongs to the accepted observation; the multi-candidate
            # fusion path has no soft equivalent, so it falls back to the fused
            # binary mask -- which is what every mode already used.
            if accepted and len(softs) == 1 and softs[0] is not None:
                soft_anchor = np.asarray(softs[0], np.float32)
            else:
                soft_anchor = as_bool(fused).astype(np.float32)

        preds.append(as_bool(fused).copy())
        res.vacuity.append(_nan(vac))
        res.selected_op.append(int(op_list[int(np.argmax(scores))]) if scores else 0)
        res.n_evaluated.append(len(op_list))
        res.agree_a.append(a_)
        res.agree_c.append(c_)
        res.agree_u.append(u_)
        res.backend_score.append(float(max(scores)) if scores else float("nan"))
        res.is_keyframe.append(int(is_decision))
        if is_decision:
            res.n_decisions += 1

    res.masks = preds
    res.n_encoder_calls = int(backend.n_encoder_calls - enc0)
    res.n_predict_failures = int(n_predict_failures)
    res.wall_time_s = float(time.time() - t_start)
    return res


# --------------------------------------------------------------------------- #
# config factory for the paper's comparison table
# --------------------------------------------------------------------------- #

def make_config(mode: str, **overrides) -> DAGRSConfig:
    """Build a config for a named experiment arm (paper TABLE-9 / TABLE-13)."""
    base = DAGRSConfig(mode=mode)
    if mode == "oracle_clean":
        base.use_restoration = False
    elif mode == "greedy":
        # Unconditional acceptance. Kept as the reference the other propagation
        # arms are measured against, NOT as the paper's baseline: this closed
        # loop is self-destructive and the damage scales with how many feedback
        # steps it takes (docs/BASELINE_DIAGNOSIS.md section 9).
        pass
    elif mode in ("greedy_gate", "greedy_reseed", "greedy_robust"):
        # The two standard drift corrections for a mask-prompt loop that has no
        # memory bank, plus their combination. All three switch `use_reanchor`
        # off: evidence-driven re-anchoring is DAG-RS's mechanism, and a baseline
        # that borrows it stops being a baseline.
        base.use_reanchor = False
        # On a refusal, emit the flow-propagated anchor -- "keep the last thing
        # we believed" -- rather than the fused multi-candidate output.
        base.output_when_rejected = "anchor"
        if mode in ("greedy_gate", "greedy_robust"):
            # SAM 2's own published default for accepting a predicted mask
            # (`pred_iou_thresh: 0.88` in the automatic-mask-generation config).
            # Using the vendor default rather than a value tuned on DAVIS keeps
            # the baseline immune to "you tuned it to lose"; the sensitivity to
            # this number is reported by scripts/96_ab_accept.py.
            base.tau_score = 0.88
        if mode in ("greedy_reseed", "greedy_robust"):
            # Re-prompt from the annotated first frame every 20 frames. DAVIS val
            # runs 34-104 frames, so this is 1-4 resets per sequence.
            base.reseed_stride = 20
    elif mode == "cascade":
        base.fixed_operator = "dcp"
    elif mode == "fixed_op":
        base.fixed_operator = "gamma_hi"
    elif mode == "equal_tta":
        base.use_agreement = False
        base.use_gating = False
        base.use_profile = False
    elif mode == "random_op":
        base.use_agreement = False
        base.use_gating = False
        base.use_profile = False
    elif mode == "sam2video":
        # SAM 2.1's released whole-video tracker (memory bank). It has no operator
        # bank, no agreement terms, no keyframe schedule and no re-anchoring, so
        # every mechanism switch is off and the arm cannot be mistaken for a
        # DAG-RS variant. Its purpose is to be the COMPARATOR the "gating can
        # replace a memory bank" claim was missing -- if any switch here were left
        # on, the arm would stop isolating the memory bank.
        base.use_restoration = False
        base.use_agreement = False
        base.use_gating = False
        base.use_profile = False
        base.use_reanchor = False
        base.tau_score = 0.0
        base.reseed_stride = 0
    elif mode == "dagrs_no_agree":
        # keep the profile and the vacuity gate, but fuse the evaluated operator
        # set with equal weights: isolates the fusion criterion alone.
        base.use_agreement = True
        base.fuse_uniform = True
    elif mode == "dagrs_no_profile":
        base.use_profile = False
        base.j_cold = 0
    elif mode == "dagrs_no_gating":
        base.use_gating = False
    elif mode == "dagrs_no_reanchor":
        base.use_reanchor = False
    for k, v in overrides.items():
        if not hasattr(base, k):
            raise KeyError(f"unknown config field '{k}'")
        setattr(base, k, v)
    return base


#: Modes that appear in the main comparison table (TABLE-9).
MAIN_MODES = ["greedy", "cascade", "equal_tta", "random_op", "dagrs"]

#: Modes that appear in the ablation table (TABLE-13).
ABLATION_MODES = ["dagrs", "dagrs_no_agree", "dagrs_no_profile", "dagrs_no_gating",
                  "dagrs_no_reanchor"]


def config_summary(cfg: DAGRSConfig) -> Dict[str, Any]:
    return asdict(cfg)
