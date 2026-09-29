"""128 - FIG-4: the qualitative panel (clean / degraded / memory bank / arm).

Why this script exists
----------------------
`docs/MANUSCRIPT.md` Sec. 5.8 has said since the first draft that Figure 4 "is the
one figure this paper cannot do without" and that it "needs a pipeline run that
keeps the masks it already computes but does not persist".  `03_run_dagrs.py`
scores each cell and drops `SequenceResult.masks`, so no artifact in `results/`
can be turned into a panel after the fact.  This script performs that run.

It is deliberately separate from `124_fig_build.py`: every other figure is a
plot of a table that already exists, whereas FIG-4 requires new GPU output.  The
two therefore have different failure modes, and `124` keeps refusing to draw
anything it cannot re-derive from a landed artifact.

Design (stated, not eyeballed)
------------------------------
* **Phase** -- the single degradation `fog` at level 3.0, stride 1, seed 0: the
  phase Sec. 6.3 quotes.  Its P1 rows live in `cl4_full_v2.csv` (see `PHASE_CSV`
  below for why).  Note `C1_fog_noise` is a *different*, **compound** phase and
  appears here only as the `--self-test` negative control, never as the plate's
  phase.
* **Columns** -- instances ranked by `dJ = J&F(dagrs) - J&F(greedy)` in that
  phase, taken from the landed P1 per-instance CSV (not re-measured here):
  column 1 = argmax (the case the arm rescues), column 2 = the instance nearest
  the median, column 3 = argmin (the case the arm loses).  Sec. 6.3 says the
  last column is reserved for the failure material, so the rule puts the worst
  instance there by construction rather than by selection after the fact.
* **Frame** -- for each column, `t* = argmax_t |J_t(dagrs) - J_t(sam2video)|`,
  i.e. the frame where the two predicted rows differ most.  One rule for all
  three columns.
* **Rows** -- clean input, degraded input, memory-bank prediction, arm under
  test.  Ground truth is drawn as a thin outline on both prediction rows so a
  reader can see what "wrong" means without a colour legend.

Usage
-----
    python scripts/128_fig4_qualitative.py --select-only   # no GPU, prints picks
    python scripts/128_fig4_qualitative.py                 # run, render, write
    python scripts/128_fig4_qualitative.py --check          # landed vs sources
    python scripts/128_fig4_qualitative.py --self-test      # guards on synthetic input
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:  # a check that crashes on a legal non-ASCII string is worse than no check
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from _common import (base_parser, config_for_mode, make_backend_for, make_bank,
                     open_src, preset)

PHASE = "fog"
LEVEL = 3.0
ARMS = ("sam2video", "dagrs")
ROW_LABELS = ("clean input", "degraded input", "memory bank (sam2video)",
              "arm under test (dagrs)")
SELECTION_RULE = (
    "single-degradation fog @ level 3.0, stride 1, seed 0; the 61 instances of "
    "the 30-sequence DAVIS val set ranked by dJ = J&F(dagrs) - J&F(greedy); "
    "column 1 = argmax dJ, column 2 = nearest the median dJ, column 3 = argmin dJ"
)
FRAME_RULE = ("per column, the frame t* = argmax_t |J_t(dagrs) - J_t(sam2video)|, "
              "i.e. where the two predicted rows differ most")

#: Both arms' per-instance P1 rows for `fog` live in `cl4_full_v2.csv`: the four
#: single degradations it covers (clean / fog / dust / underwater) are the
#: complement of the five in `singles_*.csv`, and `fog` -- the phase Sec. 6.3
#: quotes -- is only here.
PHASE_CSV = "results/p1/results/cl4_full_v2.csv"
#: The reference arm the ranking subtracts.  Sec. 5.2 defines the family gap
#: against `greedy`, and the failure material in Sec. 6.3 is stated the same way
#: ("a target that the ungated arm tracked is lost after a re-anchoring
#: decision"), so the ranking uses the same subtrahend as the prose.
RANK_SUBTRAHEND = "greedy"

#: Sec. 6.3 names both of these instances by hand, so the plate must land on
#: them -- otherwise the figure would illustrate a different claim from the one
#: the text makes: `bike-packing` is "the early qualitative observation that
#: re-anchoring *rescues* a target that the ungated arm loses", and `bmx-trees`
#: object 2 at -0.2106 is "the largest single instance loss in the fog phase".
PROSE_ANCHORS = (
    ("best (rescued)", "bike-packing", 1, +0.3233, 5e-5),
    ("worst (lost)", "bmx-trees", 2, -0.2106, 5e-5),
)

OUT_PNG = ROOT / "results" / "figs" / "F4_qualitative.png"
OUT_PDF = ROOT / "results" / "figs" / "F4_qualitative.pdf"
MANIFEST = ROOT / "results" / "figs" / "FIG4.json"
SELECTION = ROOT / "results" / "figs" / "fig4_selection.json"


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable without a GPU)
# --------------------------------------------------------------------------- #

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p: Path) -> str:
    return sha1_bytes(p.read_bytes())


def read_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"ABORT: {p} is missing -- the ranking is taken "
                             "from the landed P1 per-instance results, so it "
                             "cannot be computed without them")
        with p.open(encoding="utf-8") as fh:
            rows.extend(dict(r) for r in csv.DictReader(fh))
    return rows


def collect_deltas(base_rows: List[Dict[str, str]],
                   dagrs_rows: List[Dict[str, str]],
                   phase: str, level: float,
                   subtrahend: str) -> List[Tuple[Tuple[str, int], float, int]]:
    """[(seq, obj)] -> (delta, n_contributing_rows), over the requested phase.

    Rows repeat across the published halves (`a`/`b`), so the mean over the rows
    that share a key is what enters the ranking -- and the count is returned so
    a caller can prove no key is a single half.
    """
    def index(rows, mode):
        out: Dict[Tuple[str, int], List[float]] = {}
        for r in rows:
            if r.get("mode") != mode:
                continue
            if r.get("degradation") != phase:
                continue
            if abs(float(r.get("level", "nan")) - float(level)) > 1e-9:
                continue
            key = (str(r["seq"]), int(float(r["obj_id"])))
            out.setdefault(key, []).append(float(r["J&F"]))
        return out

    a = index(base_rows, subtrahend)
    b = index(dagrs_rows, "dagrs")
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit(
            f"ABORT: no instance has both a {subtrahend} row and a dagrs row in "
            f"{phase} @ {level:g} -- the ranking would be over an empty set and "
            "would silently pass")
    out: List[Tuple[Tuple[str, int], float, int]] = []
    for k in keys:
        d = sum(b[k]) / len(b[k]) - sum(a[k]) / len(a[k])
        out.append((k, d, min(len(a[k]), len(b[k]))))
    return out


def pick_three(ranked: Sequence[Tuple[Tuple[str, int], float, int]]
               ) -> List[Tuple[Tuple[str, int], float, int]]:
    """argmax / nearest-median / argmin, with the tie-breaks stated.

    Ties are broken by sequence name then object id so the plate is reproducible
    on a rerun rather than dependent on dict order.
    """
    if len(ranked) < 3:
        raise SystemExit(
            f"ABORT: the ranking holds {len(ranked)} instance(s); a 3-column "
            "plate needs at least 3 -- a smaller panel would silently repeat or "
            "drop an instance")
    srt = sorted(ranked, key=lambda x: (-x[1], x[0][0], x[0][1]))
    top = srt[0]
    bot = srt[-1]
    med_val = sorted(x[1] for x in ranked)[len(ranked) // 2]
    mid_cands = sorted((x for x in ranked if x[0] not in (top[0], bot[0])),
                       key=lambda x: (abs(x[1] - med_val), x[0][0], x[0][1]))
    if not mid_cands:
        raise SystemExit(
            "ABORT: after removing the extreme instances no candidate remains "
            "for the middle column -- the panel would be degenerate")
    return [top, mid_cands[0], bot]


def pick_frame(j_dagrs: Sequence[float], j_mem: Sequence[float]) -> int:
    """t* = argmax_t |J_t(dagrs) - J_t(sam2video)|; earliest frame wins a tie."""
    if not j_dagrs or len(j_dagrs) != len(j_mem):
        raise SystemExit(
            f"ABORT: the two per-frame traces differ in length "
            f"({len(j_dagrs)} vs {len(j_mem)}); the frame rule compares them "
            "element-wise and cannot proceed")
    best, best_i = -1.0, 0
    for i, (a, b) in enumerate(zip(j_dagrs, j_mem)):
        d = abs(float(a) - float(b))
        if d > best + 1e-15:
            best, best_i = d, i
    return best_i


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def select_instances(level: float = LEVEL) -> Dict[str, Any]:
    rows = read_rows([ROOT / PHASE_CSV])
    ranked = collect_deltas(rows, rows, PHASE, level, RANK_SUBTRAHEND)
    picks = pick_three(ranked)
    if any(n < 1 for _k, _d, n in picks):
        raise SystemExit("ABORT: a picked instance rests on no contributing "
                         "row -- the ranking is not evidential")
    columns = [
        {"role": role, "seq": k[0], "obj_id": k[1],
         "delta_vs_greedy": round(d, 10), "n_rows": n}
        for role, (k, d, n) in zip(("best (rescued)", "median", "worst (lost)"),
                                   picks)
    ]
    verify_prose_anchors(columns, ranked)
    return {
        "phase": PHASE,
        "level": float(level),
        "ranking_subtrahend": RANK_SUBTRAHEND,
        "n_instances_ranked": len(ranked),
        "delta_range": [round(min(d for _k, d, _n in ranked), 10),
                        round(max(d for _k, d, _n in ranked), 10)],
        "selection_rule": SELECTION_RULE,
        "frame_rule": FRAME_RULE,
        "source": PHASE_CSV,
        "source_sha1": {PHASE_CSV: sha1_file(ROOT / PHASE_CSV)},
        "prose_anchors": [{"role": r, "seq": s, "obj_id": o,
                           "expected_delta": e, "tol": t}
                          for r, s, o, e, t in PROSE_ANCHORS],
        "ranked_all": [[k[0], k[1], round(d, 10), n] for k, d, n in
                       sorted(ranked, key=lambda x: -x[1])],
        "columns": columns,
    }


def verify_prose_anchors(columns: List[Dict[str, Any]],
                         ranked: Sequence[Tuple[Tuple[str, int], float, int]]
                         ) -> None:
    """The plate must land on the instances Sec. 6.3 names, at its numbers.

    Both are stated in the manuscript as hand-written facts about single
    instances.  A figure that illustrated different instances would be a silent
    contradiction between the plate and the text, which no other guard reads.
    """
    by_role = {c["role"]: c for c in columns}
    delta_of = {k: d for k, d, _n in ranked}
    for role, seq, obj, expect, tol in PROSE_ANCHORS:
        col = by_role.get(role)
        if col is None:
            raise SystemExit(f"ABORT: no column has role {role!r}; the "
                             "prose anchor cannot be checked")
        if (col["seq"], int(col["obj_id"])) != (seq, obj):
            raise SystemExit(
                f"ABORT: the {role} column is {col['seq']} object {col['obj_id']}"
                f", but Sec. 6.3 names {seq} object {obj}. The figure and the "
                "text would be illustrating different instances")
        got = delta_of.get((seq, obj))
        if got is None:
            raise SystemExit(f"ABORT: {seq} object {obj} is absent from the "
                             "ranking, so its published delta cannot be checked")
        if abs(got - expect) > tol:
            raise SystemExit(
                f"ABORT: Sec. 6.3 quotes {expect:+.4f} for {seq} object {obj}, "
                f"but this phase gives {got:+.4f} -- the text and the data "
                "disagree, and the plate must not paper over that")


# --------------------------------------------------------------------------- #
# GPU run
# --------------------------------------------------------------------------- #

def run_masks(s: Dict[str, Any], sel: Dict[str, Any]) -> Dict[str, Any]:
    """Run the two arms on the three selected instances, keeping the masks."""
    from xdrp.benchmark import Protocol, load_degraded_sequence
    from xdrp.metrics import jf, seq_metrics
    from xdrp.pipeline import run_sequence

    backend = make_backend_for(s)
    bank = make_bank(s["cfg"])
    src = open_src(s)
    have = set(src.sequences())

    out: Dict[str, Any] = {}
    for col in sel["columns"]:
        seq, obj = col["seq"], int(col["obj_id"])
        if seq not in have:
            raise SystemExit(f"ABORT: selected sequence {seq!r} is not in the "
                             f"{s['dataset_kind']} {s['split']} split")
        proto = Protocol(degradation=PHASE, level=float(sel["level"]),
                         frame_stride=int(s["frame_stride"]),
                         max_frames=int(s["max_frames"]), seed=int(s["seed"]))
        dseq = load_degraded_sequence(src, seq, proto)
        if obj not in dseq.object_ids:
            raise SystemExit(f"ABORT: {seq} has no object {obj} at frame 0 "
                             f"(has {dseq.object_ids})")
        entry: Dict[str, Any] = {"seq": seq, "obj_id": obj,
                                 "role": col["role"], "arms": {},
                                 "pred": {}}
        for mode in ARMS:
            cfg = config_for_mode(s["cfg"], mode)
            backend.clear_cache()
            frames = dseq.frames_for_mode(mode)
            res = run_sequence(frames, dseq.first_masks[obj], backend, bank, cfg,
                               seq_name=seq, obj_id=obj, flow_frames=frames)
            gts = dseq.gt_masks[obj]
            per_frame = [float(jf(p, g, dilation=int(s["dilation"]))[2])
                         for p, g in zip(res.masks, gts)]
            agg = seq_metrics(res.masks, gts, dilation=int(s["dilation"]))
            entry["arms"][mode] = {
                "J&F": float(agg["J&F"]), "n_frames": int(agg.get("n_frames", 0)),
                "per_frame": per_frame, "n_reanchors": int(res.n_reanchors),
                "n_rejected": int(res.n_rejected),
            }
            # keep this pass's masks: re-running the arm just to fetch frame t*
            # would double the GPU cost of the plate for no extra information
            entry["_masks"] = entry.get("_masks", {})
            entry["_masks"][mode] = res.masks
        t = pick_frame(entry["arms"]["dagrs"]["per_frame"],
                       entry["arms"]["sam2video"]["per_frame"])
        if t >= min(len(entry["_masks"][m]) for m in ARMS):
            raise SystemExit(
                f"ABORT: the frame rule returned t={t}, outside the {seq} obj "
                f"{obj} clip; the masks and the traces are out of step")
        entry["frame"] = int(t)
        entry["clean"] = dseq.clean[t]
        entry["degraded"] = dseq.degraded[t]
        entry["gt"] = dseq.gt_masks[obj][t]
        for mode in ARMS:
            entry["pred"][mode] = entry["_masks"][mode][t]
        entry.pop("_masks")
        out[f"{seq}|{obj}"] = entry
    return out


def render(runs: Dict[str, Any], sel: Dict[str, Any]) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.measure import find_contours

    def outline(ax, mask, color, lw, ls="-"):
        m = np.asarray(mask)
        if m.ndim > 2:
            m = m[..., 0]
        for c in find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=lw, ls=ls)

    cols = sel["columns"]
    fig, axes = plt.subplots(4, len(cols), figsize=(3.15 * len(cols), 9.6))
    if len(cols) == 1:
        axes = axes.reshape(4, 1)

    for j, col in enumerate(cols):
        e = runs[f"{col['seq']}|{int(col['obj_id'])}"]
        t = e["frame"]
        deg = np.clip(np.asarray(e["degraded"], np.float32), 0, 1)
        clean = np.asarray(e["clean"])
        if clean.dtype != np.uint8:
            clean = np.clip(clean * 255.0, 0, 255).astype(np.uint8)

        axes[0, j].imshow(clean)
        axes[1, j].imshow(deg)
        for row, mode in ((2, "sam2video"), (3, "dagrs")):
            axes[row, j].imshow(deg)
            outline(axes[row, j], e["gt"], "#ffffff", 1.0, ls="--")
            outline(axes[row, j], e["pred"][mode], "#ff2d2d", 2.0)
        for i in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_linewidth(0.6)
        axes[0, j].set_title(
            "%s  obj %d\n%s = %+.4f   frame %d\nJ&F  dagrs %.4f / sam2video %.4f"
            % (e["seq"], e["obj_id"], RANK_SUBTRAHEND,
               col["delta_vs_greedy"], t,
               e["arms"]["dagrs"]["J&F"], e["arms"]["sam2video"]["J&F"]),
            fontsize=8.0)
    for i in range(4):
        axes[i, 0].set_ylabel(ROW_LABELS[i], fontsize=8.5)

    fig.suptitle(
        "FIG-4  qualitative panel -- %s @ level %g, stride 1\n"
        "rows: clean / degraded / memory bank / arm under test   "
        "(red = predicted outline, white dashed = ground truth)"
        % (PHASE, sel["level"]), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    drawn = [[c["seq"], c["obj_id"], round(float(c["delta_vs_greedy"]), 10),
              int(runs[f"{c['seq']}|{int(c['obj_id'])}"]["frame"])]
             for c in cols]
    return {"columns_drawn": drawn,
            "value_digest": sha1_bytes(json.dumps(drawn).encode("utf-8")),
            "png": str(OUT_PNG.relative_to(ROOT)).replace("\\", "/"),
            "pdf": str(OUT_PDF.relative_to(ROOT)).replace("\\", "/"),
            "png_sha1": sha1_file(OUT_PNG)}


def build(s: Dict[str, Any]) -> Dict[str, Any]:
    sel = select_instances()
    runs = run_masks(s, sel)
    art = render(runs, sel)
    man = {
        "id": "FIG-4",
        "title": "Qualitative panel: clean / degraded / memory bank / arm",
        "generated_by": "scripts/128_fig4_qualitative.py",
        "table": "Sec. 5.8, Sec. 6.3",
        "convention": ("one frame per column, chosen by the stated frame rule; "
                       "rows are the two observations and the two predictions; "
                       "ground truth drawn as a white dashed outline so the "
                       "prediction rows are readable without a legend"),
        "selection": sel,
        "frame_rule": FRAME_RULE,
        "arms": list(ARMS),
        "phase": PHASE,
        "level": float(sel["level"]),
        "per_column": {
            f"{c['seq']}|{int(c['obj_id'])}": {
                "role": c["role"], "seq": c["seq"], "obj_id": int(c["obj_id"]),
                "delta_vs_greedy": c["delta_vs_greedy"],
                "frame": runs[f"{c['seq']}|{int(c['obj_id'])}"]["frame"],
                "n_frames": runs[f"{c['seq']}|{int(c['obj_id'])}"
                                 ]["arms"]["dagrs"]["n_frames"],
                "J&F": {m: runs[f"{c['seq']}|{int(c['obj_id'])}"]["arms"][m]["J&F"]
                        for m in ARMS},
                "n_reanchors_dagrs": runs[f"{c['seq']}|{int(c['obj_id'])}"
                                          ]["arms"]["dagrs"]["n_reanchors"],
            }
            for c in sel["columns"]
        },
        "art": art,
    }
    MANIFEST.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return man


# --------------------------------------------------------------------------- #
# --check / --self-test
# --------------------------------------------------------------------------- #

def check() -> int:
    if not MANIFEST.is_file():
        print(f"[128] FAIL: no {MANIFEST.name} -- FIG-4 has not been built")
        return 1
    if not OUT_PNG.is_file():
        print(f"[128] FAIL: no {OUT_PNG.name}")
        return 1
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sel_now = select_instances()
    old = man["selection"]
    if sel_now["columns"] != old["columns"]:
        print("[128] FAIL: the ranking no longer selects the same instances -- "
              "the landed plate is stale")
        return 1
    if sel_now["source_sha1"] != old["source_sha1"]:
        print("[128] FAIL: a source CSV changed after the plate was drawn")
        return 1
    if man["art"]["png_sha1"] != sha1_file(OUT_PNG):
        print("[128] FAIL: the PNG changed after the manifest was written")
        return 1
    n = len(man["per_column"])
    if n != 3:
        print(f"[128] FAIL: the plate records {n} column(s), expected 3")
        return 1
    print(f"[128] --check: 3 columns, {len(man['selection']['ranked_all'])} "
          f"instances ranked, sources and PNG digest unchanged")
    print("[128] VERDICT: PASS")
    return 0


def self_test() -> int:
    fails: List[str] = []
    checks = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(f"{name}  {detail}")
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))

    def must_abort(fn, name: str, needle: str) -> None:
        try:
            fn()
        except SystemExit as exc:
            ok(name, needle in str(exc), str(exc)[:100])
        else:
            ok(name, False, "no abort was raised")

    print("128 self-test")
    print("-" * 66)

    mk = lambda name, d, n=1: ((name, 1), float(d), n)
    ranked = [mk("a", -0.20), mk("b", 0.01), mk("c", 0.05), mk("d", 0.15),
              mk("e", -0.05), mk("f", 0.02)]
    picks = pick_three(ranked)
    ok("the extremes and a middle instance are picked, in order",
       picks[0][0][0] == "d" and picks[2][0][0] == "a"
       and picks[1][0][0] not in ("a", "d"),
       f"{[p[0][0] for p in picks]}")
    ok("the picks are distinct instances",
       len({p[0] for p in picks}) == 3)
    med_val = sorted(x[1] for x in ranked)[len(ranked) // 2]
    ok("the middle pick is the candidate nearest the count-median, extremes excluded",
       picks[1][0][0] not in ("a", "d")
       and abs(picks[1][1] - med_val)
       <= min(abs(x[1] - med_val) for x in ranked
              if x[0] not in ("a", "d")) + 1e-12,
       f"median value = {med_val:+.3f}, picked {picks[1][0][0]}")

    must_abort(lambda: pick_three(ranked[:2]),
               "a two-instance ranking refuses to make a 3-column plate",
               "at least 3")
    ties = pick_three([mk("a", 0.0), mk("b", 0.0), mk("c", 0.0)])
    ok("an all-tie ranking still yields three distinct columns",
       len({p[0] for p in ties}) == 3, f"{[p[0][0] for p in ties]}")

    ok("the frame rule takes the largest disagreement",
       pick_frame([0.1, 0.9, 0.2], [0.5, 0.1, 0.5]) == 1)
    ok("the frame rule resolves a tie to the earliest frame",
       pick_frame([0.0, 1.0], [1.0, 0.0]) == 0
       and pick_frame([0.0, 1.0, 0.2], [1.0, 0.0, 0.2]) == 0,
       "a two-way tie must not pick whichever index the loop happens to keep")
    must_abort(lambda: pick_frame([0.1, 0.2], [0.1]),
               "traces of unequal length abort instead of truncating",
               "differ in length")

    # deliberately a phase that is NOT the selected one, so the "wrong phase"
    # fixture below is genuinely excluded rather than accidentally matching
    PH_ = "C1_fog_noise"
    base = [{"mode": "greedy", "degradation": PH_, "level": "3.0",
             "seq": "s1", "obj_id": "1", "J&F": "0.10"},
            {"mode": "sam2video", "degradation": PH_, "level": "3.0",
             "seq": "s1", "obj_id": "1", "J&F": "0.90"},
            {"mode": "greedy", "degradation": PHASE, "level": "3.0",
             "seq": "s1", "obj_id": "1", "J&F": "0.99"},
            {"mode": "greedy", "degradation": PH_, "level": "4.0",
             "seq": "s1", "obj_id": "1", "J&F": "0.77"}]
    dagrs = [{"mode": "dagrs", "degradation": PH_, "level": "3.0",
              "seq": "s1", "obj_id": "1", "J&F": "0.30"}]
    d = collect_deltas(base, dagrs, PH_, LEVEL, "greedy")
    ok("the delta uses the declared phase, level and subtrahend only",
       len(d) == 1 and abs(d[0][1] - 0.20) < 1e-12,
       str([(k[0], round(v, 4)) for k, v, _n in d]))
    ok("the contributing row count is returned for each key", d[0][2] == 1)

    two = [dict(r) for r in base if r["mode"] == "greedy"]
    d2 = collect_deltas(two + two, dagrs + dagrs, PH_, LEVEL, "greedy")
    ok("duplicated published halves are averaged, not double-counted",
       abs(d2[0][1] - 0.20) < 1e-12 and d2[0][2] == 2,
       f"delta={d2[0][1]:+.4f} n={d2[0][2]}")
    must_abort(lambda: collect_deltas([], dagrs, PH_, LEVEL, "greedy"),
               "a ranking over an empty key set aborts (no silent PASS)",
               "empty set")

    # the prose anchors: Sec. 6.3 names two instances and one of their numbers,
    # so the plate is only allowed to land on those
    anch = [{"role": "best (rescued)", "seq": "bike-packing", "obj_id": 1,
             "delta_vs_greedy": +0.3233},
            {"role": "median", "seq": "x", "obj_id": 1,
             "delta_vs_greedy": 0.0},
            {"role": "worst (lost)", "seq": "bmx-trees", "obj_id": 2,
             "delta_vs_greedy": -0.2106}]
    rank = [(("bike-packing", 1), +0.3233, 1), (("x", 1), 0.0, 1),
            (("bmx-trees", 2), -0.2106, 1)]
    try:
        verify_prose_anchors(anch, rank)
        ok("the prose anchors pass when the plate lands on the named instances",
           True)
    except SystemExit as exc:
        ok("the prose anchors pass when the plate lands on the named instances",
           False, str(exc)[:100])
    wrong_inst = [dict(a) for a in anch]
    wrong_inst[2] = dict(wrong_inst[2], seq="pigs", obj_id=3)
    must_abort(lambda: verify_prose_anchors(wrong_inst, rank),
               "a plate that lands on a different instance than the text aborts",
               "different instances")
    off_num = [rank[0], rank[1], (("bmx-trees", 2), -0.1900, 1)]
    must_abort(lambda: verify_prose_anchors(anch, off_num),
               "a plate whose delta contradicts Sec. 6.3 aborts",
               "text and the data")

    if (ROOT / PHASE_CSV).is_file():
        real = select_instances()
        ok("the real ranking lands on both instances Sec. 6.3 names",
           [(c["seq"], int(c["obj_id"])) for c in real["columns"][::2]]
           == [(s, o) for _r, s, o, _e, _t in PROSE_ANCHORS],
           ", ".join(f"{c['role']}={c['seq']} obj{c['obj_id']}"
                     for c in real["columns"]))
        ok("the real ranking reproduces Sec. 6.3's quoted loss",
           abs(real["columns"][2]["delta_vs_greedy"]
               - PROSE_ANCHORS[1][3]) <= PROSE_ANCHORS[1][4]
           and abs(real["columns"][0]["delta_vs_greedy"]
                   - PROSE_ANCHORS[0][3]) <= PROSE_ANCHORS[0][4],
           f"range {real['delta_range'][0]:+.4f} .. {real['delta_range'][1]:+.4f}")

    print("-" * 66)
    if fails:
        print(f"128 --self-test: FAIL ({len(fails)}/{checks})")
        for f in fails:
            print("   " + f)
        return 1
    print(f"128 --self-test: PASS ({checks} controls)")
    return 0


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--select-only", action="store_true",
                    help="compute and record the selection, run nothing on GPU")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.check:
        return check()

    sel = select_instances()
    if args.select_only:
        SELECTION.parent.mkdir(parents=True, exist_ok=True)
        SELECTION.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
        print("FIG-4 selection (no GPU run)")
        print("=" * 66)
        print(f"  ranked instances : {sel['n_instances_ranked']}")
        print(f"  phase            : {sel['phase']} @ {sel['level']:g}")
        print(f"  subtrahend       : {sel['ranking_subtrahend']}")
        for c in sel["columns"]:
            print(f"  {c['role']:<14} {c['seq']:<16} obj {c['obj_id']:<3} "
                  f"dJ = {c['delta_vs_greedy']:+.4f}  (n rows {c['n_rows']})")
        print(f"  wrote            : {SELECTION}")
        return 0

    if args.frame_stride is None:
        args.frame_stride = 1
        print("[128] defaulting --frame-stride to 1 (the paper's protocol; the "
              "config default is 4)")
    s = preset(args)
    if int(s["frame_stride"]) != 1:
        raise SystemExit(
            f"ABORT: frame_stride is {s['frame_stride']}, but every table this "
            "plate accompanies was produced at stride 1 -- pass "
            "--frame-stride 1")
    man = build(s)
    print("XD-RobustPVOS :: FIG-4 qualitative panel")
    print("=" * 66)
    for key, col in man["per_column"].items():
        print(f"  {col['role']:<14} {key:<20} dJ={col['delta_vs_greedy']:+.4f} "
              f"frame {col['frame']:<4} J&F dagrs {col['J&F']['dagrs']:.4f} / "
              f"sam2video {col['J&F']['sam2video']:.4f}")
    print(f"  wrote            : {man['art']['png']}")
    print(f"  wrote            : {man['art']['pdf']}")
    print(f"  wrote            : {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
