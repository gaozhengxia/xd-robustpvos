"""Diagnose why the `greedy` baseline collapses (project-internal debugging tool).

The 2026-09-14 baseline sweep produced greedy J&F = 0.095 on DAVIS -- roughly
seven times below the published 0.7-0.85 range for single-frame-propagation
baselines. Before touching any method code, three questions must be answered
with measurements rather than inference:

  Q1  Is SAM 2 actually being called, and does it succeed?
      `xdrp/pipeline.py:281` catches every exception from `backend.predict` and
      substitutes the warped anchor. If SAM 2 were failing on every frame, the
      pipeline would silently degrade to pure optical-flow propagation with no
      visible error. This tool counts escaped exceptions at TWO levels: the
      backend call and the underlying `SAM2ImagePredictor.predict` (which has
      its own internal box-only fallback at `xdrp/sam_backend.py:190`).

  Q2  Is the predicted region eroding over time?
      Reports pred_area / gt_area per frame plus the mask-prompt area that was
      fed in. Monotone decay of the ratio is the signature of the
      "mask prompt is a re-statement, not a correction" failure mode.

  Q3  Is the collapse caused by the degradation, or already present on clean
      input? Runs the same loop on `clean`.

  Q4  Which prompt is responsible?
      `--prompt box_mask|box|mask` edits the prompt set in flight (without
      modifying the pipeline) so the culprit can be isolated in one run.

Run from the PROJECT ROOT -- `configs/default.yaml` stores the checkpoint and
dataset paths relative to the working directory:

    python scripts/90_diagnose_greedy.py --degradations clean,fog --level 1 --sequences 3
    python scripts/90_diagnose_greedy.py --degradations clean --prompt box
    python scripts/90_diagnose_greedy.py --degradations clean --prompt mask

Output: a per-frame table on stdout plus results/diag_greedy.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xdrp.benchmark import Protocol, load_degraded_sequence       # noqa: E402
from xdrp.datasets import open_dataset                            # noqa: E402
from xdrp.evidence import as_bool                                 # noqa: E402
from xdrp.sam_backend import as_prediction                        # noqa: E402
from xdrp.metrics import jf                                       # noqa: E402
from xdrp.operators import OperatorBank, build_default_bank       # noqa: E402
from xdrp.pipeline import make_config, run_sequence               # noqa: E402
from xdrp.sam_backend import build_backend                        # noqa: E402


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #

class Spy:
    """Everything the production pipeline hides from the caller."""

    def __init__(self, prompt: str = "box_mask"):
        self.prompt = prompt
        self.calls: List[Dict[str, Any]] = []
        self.inner: List[Dict[str, Any]] = []
        self.n_outer_exc = 0
        self.n_inner_exc = 0
        self.n_empty_out = 0

    # -- reporting ------------------------------------------------------ #

    def token_frame(self, rec: Dict[str, Any]) -> Optional[int]:
        tok = rec.get("token")
        if isinstance(tok, tuple) and len(tok) >= 2 and isinstance(tok[1], int):
            return int(tok[1])
        return None


def instrument(backend, prompt: str = "box_mask") -> Spy:
    """Wrap the backend and (if present) the SAM 2 image predictor in place."""
    spy = Spy(prompt=prompt)
    orig_predict = backend.predict

    def spy_predict(img, box=None, mask=None, multimask=False, cache_token=None,
                    want_soft=False):
        rec: Dict[str, Any] = {
            "token": cache_token,
            "in_mask_area": int(as_bool(mask).sum()) if mask is not None else -1,
            "box": ([float(v) for v in np.asarray(box).ravel()]
                    if box is not None else None),
            "n_inner_before": len(spy.inner),
            "dropped": [],
        }
        if prompt == "box" and mask is not None:
            mask = None
            rec["dropped"].append("mask_input")
        elif prompt == "mask" and box is not None:
            box = None
            rec["dropped"].append("box")
        try:
            out = orig_predict(img, box=box, mask=mask, multimask=multimask,
                               cache_token=cache_token, want_soft=want_soft)
            m, s, soft = as_prediction(out)
        except BaseException as e:                       # noqa: BLE001
            rec["outer_exc"] = f"{type(e).__name__}: {e}"
            spy.n_outer_exc += 1
            spy.calls.append(rec)
            raise
        area = int(as_bool(m).sum())
        rec["out_mask_area"] = area
        rec["score"] = float(s)
        rec["n_inner"] = len(spy.inner) - rec["n_inner_before"]
        if area == 0:
            spy.n_empty_out += 1
        spy.calls.append(rec)
        return m, s, soft

    backend.predict = spy_predict

    pred = getattr(backend, "_predictor", None)
    if pred is not None:
        orig_inner = pred.predict

        def spy_inner(**kw):
            mi = kw.get("mask_input")
            rec: Dict[str, Any] = {
                "has_mask_input": mi is not None,
                "mask_input_shape": (tuple(np.asarray(mi).shape)
                                     if mi is not None else None),
                "has_box": kw.get("box") is not None,
                "multimask": bool(kw.get("multimask_output", False)),
            }
            try:
                out = orig_inner(**kw)
            except BaseException as e:                   # noqa: BLE001
                rec["exc"] = f"{type(e).__name__}: {e}"
                spy.n_inner_exc += 1
                spy.inner.append(rec)
                raise
            spy.inner.append(rec)
            return out

        pred.predict = spy_inner
    return spy


# --------------------------------------------------------------------------- #
# one (sequence, degradation) cell
# --------------------------------------------------------------------------- #

def run_cell(src, seq: str, degradation: str, level: float, stride: int,
             max_frames: int, backend, bank: OperatorBank, mode: str,
             dilation: int, spy: Spy, start_call: int,
             mask_prompt: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    proto = Protocol(degradation=degradation, level=float(level),
                     frame_stride=int(stride), max_frames=int(max_frames), seed=0)
    dseq = load_degraded_sequence(src, seq, proto, max_objects=1)
    obj = int(dseq.object_ids[0]) if dseq.object_ids else 1
    over = {"mask_prompt": mask_prompt} if mask_prompt else {}
    cfg = make_config(mode, **over)
    backend.clear_cache()
    frames = dseq.frames_for_mode(cfg.mode)

    t0 = time.time()
    res = run_sequence(frames, dseq.first_masks[obj], backend, bank, cfg,
                       seq_name=seq, obj_id=obj, flow_frames=frames)
    dt = time.time() - t0

    # map spy calls to frames via the cache token
    per_frame_mask_area: Dict[int, int] = {}
    per_frame_box: Dict[int, List[float]] = {}
    for rec in spy.calls[start_call:]:
        t = spy.token_frame(rec)
        if t is None:
            continue
        if rec.get("in_mask_area", -1) >= 0:
            per_frame_mask_area.setdefault(t, rec["in_mask_area"])
        if rec.get("box"):
            per_frame_box.setdefault(t, rec["box"])

    gts = dseq.gt_masks[obj]
    rows: List[Dict[str, Any]] = []
    for t in range(len(res.masks)):
        j, f, jfval = jf(res.masks[t], gts[t], dilation)
        pred_area = int(as_bool(res.masks[t]).sum())
        gt_area = int(as_bool(gts[t]).sum())
        rows.append({
            "seq": seq, "mode": mode, "prompt": spy.prompt,
            "mask_prompt": str(cfg.mask_prompt),
            "degradation": degradation, "level": float(level), "frame": t,
            "J": j, "F": f, "JF": jfval,
            "pred_area": pred_area, "gt_area": gt_area,
            "area_ratio": (pred_area / gt_area) if gt_area else float("nan"),
            "anchor_area": per_frame_mask_area.get(t, -1),
            "n_evaluated": int(res.n_evaluated[t]),
            "is_keyframe": int(res.is_keyframe[t]),
            "backend_score": (float(res.backend_score[t])
                              if np.isfinite(res.backend_score[t]) else float("nan")),
        })

    summary = {
        "seq": seq, "degradation": degradation, "level": float(level),
        "mode": mode, "prompt": spy.prompt, "mask_prompt": str(cfg.mask_prompt),
        "n_frames": len(res.masks),
        "J": float(np.mean([r["J"] for r in rows])),
        "F": float(np.mean([r["F"] for r in rows])),
        "JF": float(np.mean([r["JF"] for r in rows])),
        "JF_f1": float(np.mean([r["JF"] for r in rows[1:]])),
        "area_ratio_mean": float(np.nanmean([r["area_ratio"] for r in rows[1:]])),
        "area_ratio_last": float(rows[-1]["area_ratio"]),
        "wall_time_s": dt,
        "n_encoder_calls": int(res.n_encoder_calls),
    }
    return rows, summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model-cfg", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--degradations", default="clean,fog")
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--sequences", type=int, default=3)
    ap.add_argument("--seq", default=None,
                    help="pin sequences by name; comma-separated for several")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--prompt", default="box_mask",
                    choices=["box_mask", "box", "mask"],
                    help="which prompts to pass to the segmenter")
    ap.add_argument("--mask-prompt", default=None, choices=["hard", "soft"],
                    help="override dagrs.mask_prompt: 'hard' re-prompts with a "
                         "binarised mask, 'soft' carries the segmenter's own mask "
                         "probability (see docs/BASELINE_DIAGNOSIS.md)")
    ap.add_argument("--dilation", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-frames", type=int, default=12)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_root = args.dataset or cfg["dataset"]["root"]
    ckpt = args.checkpoint or cfg["backend"]["checkpoint"]
    mcfg = args.model_cfg or cfg["backend"]["model_cfg"]

    print("=" * 78)
    print("XD-RobustPVOS greedy-collapse diagnostic")
    print("=" * 78)
    print(f"cwd          : {Path.cwd()}")
    print(f"dataset_root : {dataset_root}   (exists={Path(dataset_root).is_dir()})")
    print(f"checkpoint   : {ckpt}   (exists={Path(ckpt).exists()})")
    print(f"model_cfg    : {mcfg}   (exists={Path(mcfg).exists()})")
    print(f"prompt set   : {args.prompt}")
    if args.mask_prompt:
        print(f"mask_prompt  : {args.mask_prompt}")

    src = open_dataset(cfg["dataset"]["kind"], dataset_root, split=args.split)
    seqs = src.sequences()
    print(f"sequences    : {len(seqs)} visible in split '{args.split}'")
    if args.seq:
        seqs = [s.strip() for s in args.seq.split(",") if s.strip()]
    else:
        seqs = seqs[:max(1, int(args.sequences))]
    print(f"using        : {seqs}")

    bank = OperatorBank(build_default_bank())
    print(f"operator bank: M={len(bank)}  index0={bank.names[0]!r}")

    backend = build_backend("sam2", checkpoint=ckpt, model_cfg=mcfg,
                            device=args.device, autocast_dtype="bfloat16",
                            cache_embeddings=True, cache_size=4)
    n_par = sum(p.numel() for p in backend._predictor.model.parameters())
    print(f"backend      : SAM2 on {args.device}, {n_par/1e6:.1f} M params, bfloat16 autocast")
    spy = instrument(backend, prompt=args.prompt)

    degs = [d.strip() for d in args.degradations.split(",") if d.strip()]
    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for deg in degs:
        for seq in seqs:
            start = len(spy.calls)
            n_exc0, n_inner_exc0, n_empty0 = spy.n_outer_exc, spy.n_inner_exc, spy.n_empty_out
            try:
                rows, summ = run_cell(src, seq, deg, args.level, args.stride,
                                      args.max_frames, backend, bank, args.mode,
                                      args.dilation, spy, start,
                                      mask_prompt=args.mask_prompt)
            except BaseException as e:                    # noqa: BLE001
                import traceback
                print(f"\n[{seq} | {deg} L{args.level:g}] HARD FAILURE: "
                      f"{type(e).__name__}: {e}")
                print(traceback.format_exc())
                summaries.append({"seq": seq, "degradation": deg, "level": args.level,
                                  "mode": args.mode, "prompt": args.prompt,
                                  "hard_failure": f"{type(e).__name__}: {e}"})
                continue
            summ["outer_exc"] = spy.n_outer_exc - n_exc0
            summ["inner_exc"] = spy.n_inner_exc - n_inner_exc0
            summ["empty_out"] = spy.n_empty_out - n_empty0
            all_rows.extend(rows)
            summaries.append(summ)

            print(f"\n--- {seq} | {deg} | L{args.level:g} | {args.mode} | "
                  f"{args.prompt} ---")
            print(f"{'t':>4}{'J':>9}{'F':>9}{'J&F':>9}{'pred_px':>10}"
                  f"{'gt_px':>10}{'ratio':>8}{'anchor_px':>11}{'score':>8}")
            for r in rows[:max(1, int(args.show_frames))]:
                print(f"{r['frame']:>4}{r['J']:>9.4f}{r['F']:>9.4f}"
                      f"{r['JF']:>9.4f}{r['pred_area']:>10}{r['gt_area']:>10}"
                      f"{r['area_ratio']:>8.3f}{r['anchor_area']:>11}"
                      f"{r['backend_score']:>8.3f}")
            if len(rows) > args.show_frames:
                tail = rows[-3:]
                for r in tail:
                    print(f"{r['frame']:>4}{r['J']:>9.4f}{r['F']:>9.4f}"
                          f"{r['JF']:>9.4f}{r['pred_area']:>10}{r['gt_area']:>10}"
                          f"{r['area_ratio']:>8.3f}{r['anchor_area']:>11}"
                          f"{r['backend_score']:>8.3f}")
            print(f"    mean J&F={summ['JF']:.4f}  (from frame 1: {summ['JF_f1']:.4f})"
                  f"  area_ratio: first={rows[1]['area_ratio']:.3f} "
                  f"last={summ['area_ratio_last']:.3f}"
                  f"  encoder_calls={summ['n_encoder_calls']}"
                  f"  {summ['wall_time_s']:.1f}s")
            print(f"    escaped exceptions: backend={summ['outer_exc']} "
                  f"sam2_predict={summ['inner_exc']}  empty_masks={summ['empty_out']}")

    # ---- verdict ---------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    tot_outer = sum(s.get("outer_exc", 0) for s in summaries)
    tot_inner = sum(s.get("inner_exc", 0) for s in summaries)
    print(f"escaped exceptions at the backend boundary : {tot_outer}")
    print(f"escaped exceptions inside SAM2Predictor    : {tot_inner}")
    if tot_inner:
        seen = set()
        for rec in spy.inner:
            if "exc" in rec and rec["exc"] not in seen:
                seen.add(rec["exc"])
                print(f"   -> {rec['exc'][:160]}")
    if tot_outer == 0 and tot_inner == 0:
        print("   => SAM 2 ran successfully on every frame. The collapse is NOT a")
        print("      swallowed exception; it is a propagation/prompting behaviour.")

    clean = [s for s in summaries if s.get("degradation") == "clean" and "JF" in s]
    if clean:
        cjf = float(np.mean([s["JF_f1"] for s in clean]))
        cmean = float(np.mean([s["area_ratio_mean"] for s in clean]))
        print(f"\nclean input   : mean J&F (frames>=1) = {cjf:.4f}   "
              f"mean area_ratio = {cmean:.3f}")
        if cjf >= 0.7:
            print("   => propagation is healthy on clean input. The degradation")
            print("      sweep is the problem (severity calibration).")
        elif cjf <= 0.25:
            print("   => propagation is ALREADY broken on clean input. Do not run")
            print("      any more degradation cells until this is fixed.")
        else:
            print("   => partial: propagation works but is prompt-limited.")

    print("\nreading the area_ratio column: a value decaying toward 0 means the")
    print("predicted region is eroding frame by frame. Compare the three --prompt")
    print("runs to see whether the mask prompt is what drives the erosion.")

    rows_out = args.out or str(ROOT / "results" / "diag_greedy.csv")
    p = Path(rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        keys: List[str] = []
        for r in all_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nper-frame rows -> {p}  ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
