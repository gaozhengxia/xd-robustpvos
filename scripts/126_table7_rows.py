"""126 - Table 7 (Sec. 4.8, complexity): owns the measured cost rows.

Why this script exists
----------------------
Sec. 4.8's Table 7 sat at `[[TODO]]` from the first draft, and the paragraph under
it explained that the FPS and memory columns "may not be filled by estimate".
When `scripts/06_profile.py` was finally run it produced a CSV with one row per
(sequence, mode) -- which is a *measurement*, not a table.  Something has to turn
it into rows, decide which measured mode is which configuration, and be able to
say afterwards that the numbers in the manuscript are still the numbers the
profiler produced.  That is this script.

What it owns
------------
* the **caption** (which carries the scope: stride, condition, sequence count);
* the **pipe table** between the caption and `<!-- MS-TABLE: table7 -->`;
* the **scope footnote** under the table, which is where the unmeasurable row is
  explained -- see below;
* the **framework mirror** between `<!-- P1-SYNC: table7 -->` and the next `##`
  heading, so `scripts/122_ms_doc_guard.py` rule A (every manuscript number must
  occur verbatim in the framework) holds for numbers this run introduced.

Two things it refuses to take on trust
--------------------------------------
* **The configuration labels.**  Table 7's rows are described by `(K, J)`, and the
  profiler measures *modes*.  Which mode is which configuration is therefore an
  inference, so it is checked against the analytic per-frame encoder-call counts
  instead of being asserted by eye: `sam2video` must sit at ~1 call/frame,
  `dagrs_no_reanchor` at ~1+3/5, and the deployed `dagrs` above that because
  re-anchoring adds passes.  A mapping that disagrees with the arithmetic aborts.
* **The protocol.**  `table7_profile.csv` does not record its own `frame_stride`,
  and FPS at stride 4 is not comparable with FPS at stride 1.  Rather than trust
  the launcher, the profiled frame counts are checked against the stride-1 frame
  counts that the P1 per-instance artifacts record for the same sequences; a
  profiler run at another stride cannot pass that check.

The unmeasurable row
--------------------
`J = M = 13` is a configuration of the mechanism, not a mode the profiler can be
asked for: no arm evaluates the bank over all $M$ operators at every decision
frame.  Its measured cells are therefore written as an em dash with the reason
stated in the footnote, never filled from the neighbouring row -- a filled cell
would be indistinguishable from a measurement.

Usage
-----
    python scripts/126_table7_rows.py            # print the rows
    python scripts/126_table7_rows.py --emit     # print the three owned blocks
    python scripts/126_table7_rows.py --check    # reconcile manuscript + framework
    python scripts/126_table7_rows.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROFILE = ROOT / "results" / "p1" / "table7_profile.csv"
MS = ROOT / "docs" / "MANUSCRIPT.md"
FRAMEWORK = ROOT / "docs" / "PAPER_FRAMEWORK.md"

#: `FRAMEWORK` is an internal working document and is deliberately NOT
#: distributed; `_release_gate` turns its absence in a clone into a reported
#: SKIP instead of a crash.  Only `--check` reads it, so only `--check` is gated.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _release_gate import gate as release_gate                       # noqa: E402
MS_ANCHOR = "<!-- MS-TABLE: table7 -->"
FW_ANCHOR = "<!-- P1-SYNC: table7 -->"
#: The P1 per-instance artifacts, all at stride 1; used only for their frame counts.
P1_INSTANCE_GLOBS = ("results/p1/results/*.csv",)

#: The em dash, U+2014: this document's mark for "not applicable / not measured".
DASH = "\u2014"

#: Table 7's rows, in the manuscript's order, with the labels spelt the way the
#: manuscript's other tables spell them (math for $K$, $J$, $M$), so the block this
#: script emits can be installed verbatim instead of re-typed with a typo.
#:   (row label, measured mode or None, analytic calls per frame)
#: `None` = no arm in the profiler corresponds to this configuration, so its
#: measured cells must stay unmeasured rather than be filled from a neighbour.
ROWS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("SAM 2.1-L baseline", "sam2video", "1"),
    ("DAG-RS ($K{=}5$, $J{=}3$)", "dagrs_no_reanchor", "1+3/5"),
    ("DAG-RS ($K{=}5$, $J{=}M{=}13$)", None, "1+13/5"),
    ("DAG-RS + re-anchor", "dagrs", "-"),
)

#: Which measured mode is the denominator of "relative to baseline".
BASELINE_MODE = "sam2video"

#: Analytic per-frame encoder calls, as (numerator, denominator) of a fraction.
ANALYTIC = {"sam2video": (1, 1), "dagrs_no_reanchor": (8, 5), "dagrs": None}
#: Tolerance on the ratio of measured to analytic calls.  Generous enough for the
#: cold-frame term (all M operators at the first decision frame) and for the
#: decision-frame count not dividing the clip evenly, tight enough to separate
#: 1.0 from 1.6 from "more than 1.6".
RATIO_TOL = 0.35
#: A mode's per-sequence peak memory counts as "flat" within this relative band.
FLAT_TOL = 0.005


# --------------------------------------------------------------------------- #

def read_profile(path: Path = PROFILE) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(
            f"ABORT: {path} is missing. Table 7's measured cells come from "
            "`scripts/06_profile.py --config configs/_tau013.yaml --frame-stride 1`; "
            "nothing may be filled by estimate.")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"ABORT: {path} holds no data rows")
    return rows


def stride1_frame_counts() -> Dict[str, int]:
    """Frame count per sequence as recorded by the stride-1 P1 artifacts."""
    out: Dict[str, int] = {}
    seen: set[Tuple[str, int]] = set()
    for pat in P1_INSTANCE_GLOBS:
        for f in sorted(glob.glob(str(ROOT / pat))):
            with open(f, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    if int(float(r.get("frame_stride", "1"))) != 1:
                        continue
                    seq, n = str(r["seq"]), int(float(r["n_frames"]))
                    if (seq, n) in seen:
                        continue
                    seen.add((seq, n))
                    out[seq] = max(out.get(seq, 0), n)
    if not out:
        raise SystemExit("ABORT: no stride-1 P1 per-instance rows were found, so "
                         "the profiler's protocol cannot be cross-checked")
    return out


def scope(rows: Sequence[Dict[str, str]], stride: int) -> Dict[str, Any]:
    """The one experiment the profile describes, stated once.

    A CSV that mixed severities would make a column called "Mean FPS" the mean of
    two different experiments, so mixing is an abort rather than a footnote.
    """
    degs = sorted({r["degradation"] for r in rows})
    levs = sorted({float(r["level"]) for r in rows})
    if len(degs) != 1 or len(levs) != 1:
        raise SystemExit(
            f"ABORT: the profile mixes conditions (degradation={degs}, level={levs}); "
            "a mean over a mixed grid is not one experiment")
    modes: List[str] = []
    for r in rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    seqs = sorted({r["seq"] for r in rows})
    frames = [float(r["n_frames"]) for r in rows]
    return {"n_seq": len(seqs), "seqs": seqs, "modes": modes,
            "degradation": degs[0], "level": levs[0], "stride": int(stride),
            # The memory column's meaning depends on how long the clips were: a
            # bank that has not yet grown is cheap for the wrong reason.
            "frames_lo": int(min(frames)), "frames_hi": int(max(frames))}


def summarise(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """Per mode: mean encoder calls per frame, mean FPS, mean peak memory."""
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["mode"], []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for mode, sub in by.items():
        frames = [float(r["n_frames"]) for r in sub]
        calls = [float(r["encoder_calls"]) for r in sub]
        fps = [float(r["fps"]) for r in sub]
        peak = [float(r["gpu_peak_mb"]) for r in sub]
        if any(f <= 0 for f in frames):
            raise SystemExit(f"ABORT: mode {mode!r} reports a non-positive frame "
                             "count, so calls per frame is undefined")
        if any(c < 0 for c in calls):
            raise SystemExit(f"ABORT: mode {mode!r} reports a negative encoder-call "
                             "count")
        out[mode] = {
            "n_seq": float(len(sub)),
            # calls per frame is a ratio of sums, not a mean of ratios: a short
            # clip must not weigh as much as a long one
            "calls_per_frame": sum(calls) / sum(frames),
            "fps": sum(fps) / len(fps),
            "fps_lo": min(fps),
            "fps_hi": max(fps),
            "peak_mb": sum(peak) / len(peak),
            "peak_lo": min(peak),
            "peak_hi": max(peak),
            "frames": sum(frames),
        }
    if BASELINE_MODE not in out:
        raise SystemExit(
            f"ABORT: the profiler did not measure {BASELINE_MODE!r}, so there is "
            "no baseline for the relative-cost column")
    return out


def check_protocol(rows: Sequence[Dict[str, str]], stride: int) -> str:
    """The measured clip lengths must be the stride-1 lengths."""
    if int(stride) != 1:
        raise SystemExit(
            f"ABORT: --frame-stride is {stride}; every table Table 7 accompanies "
            "was produced at stride 1, and FPS is not comparable across strides")
    want = stride1_frame_counts()
    seqs = sorted({r["seq"] for r in rows})
    bad: List[str] = []
    for s in seqs:
        got = sorted({int(float(r["n_frames"])) for r in rows if r["seq"] == s})
        if s not in want:
            bad.append(f"{s}: not present in any stride-1 P1 artifact")
            continue
        if got != [want[s]]:
            bad.append(f"{s}: profiled {got} frame(s), stride-1 artefact says "
                       f"{want[s]}")
    if bad:
        raise SystemExit(
            "ABORT: the profiler's clip lengths do not match the stride-1 P1 "
            "artifacts, so this FPS is not on the paper's protocol:\n  "
            + "\n  ".join(bad))
    return (f"{len(seqs)} sequence(s) profiled at their stride-1 lengths "
            f"(mean clip {sum(want[s] for s in seqs) / len(seqs):.1f} frames)")


def check_mapping(summary: Dict[str, Dict[str, float]]) -> List[str]:
    """Do the measured call rates agree with the labels Table 7 puts on them?"""
    notes: List[str] = []
    bad: List[str] = []
    for mode, frac in ANALYTIC.items():
        if frac is None or mode not in summary:
            continue
        num, den = frac
        want = num / den
        got = summary[mode]["calls_per_frame"]
        rel = got / want
        notes.append(f"{mode}: {got:.3f} calls/frame vs analytic {want:.3f} "
                     f"(x{rel:.3f})")
        if abs(rel - 1.0) > RATIO_TOL:
            bad.append(f"{mode}: {got:.3f} calls/frame is {rel:.2f}x the analytic "
                       f"{want:.3f}, outside the {RATIO_TOL:.2f} tolerance")
    if "dagrs" in summary and "dagrs_no_reanchor" in summary:
        re_cost = summary["dagrs"]["calls_per_frame"]
        no_re_cost = summary["dagrs_no_reanchor"]["calls_per_frame"]
        # Reported whichever way it falls, so the check's output states the fact
        # it is judging rather than only complaining when the fact is wrong.
        notes.append(f"re-anchor ordering: dagrs {re_cost:.3f} vs "
                     f"dagrs_no_reanchor {no_re_cost:.3f} calls/frame")
        if not re_cost > no_re_cost:
            bad.append("the deployed dagrs arm does not cost more than "
                       "dagrs_no_reanchor, yet Table 7 calls one of them "
                       "'+ re-anchor' and the re-anchoring it adds is real "
                       "(Sec. 4.6 fires in 100/120 compound cells)")
    if bad:
        raise SystemExit(
            "ABORT: the measured encoder-call rates do not support the "
            "configuration labels Table 7 puts on these modes:\n  "
            + "\n  ".join(bad) + "\n  " + "; ".join(notes))
    return notes


def build_rows(summary: Dict[str, Dict[str, float]]
               ) -> List[Dict[str, Optional[str]]]:
    base = summary[BASELINE_MODE]["calls_per_frame"]
    out: List[Dict[str, Optional[str]]] = []
    for label, mode, analytic in ROWS:
        rec: Dict[str, Optional[str]] = {"label": label, "mode": mode,
                                         "analytic": analytic}
        if mode is None or mode not in summary:
            rec.update({"relative": None, "fps": None, "peak_gb": None})
        else:
            s = summary[mode]
            rec.update({
                "relative": f"{s['calls_per_frame'] / base:.2f}",
                "fps": f"{s['fps']:.2f}",
                "peak_gb": f"{s['peak_mb'] / 1024.0:.2f}",
                "calls_per_frame": f"{s['calls_per_frame']:.3f}",
            })
        out.append(rec)
    return out


# --------------------------------------------------------------- owned blocks #

def caption_line(sc: Dict[str, Any]) -> str:
    return (f"**Table 7. Cost (analytic per-frame encoder calls; measured FPS and peak "
            f"GPU memory at frame stride {sc['stride']}, single-degradation "
            f"`{sc['degradation']}` at level {sc['level']:.1f}, {sc['n_seq']} DAVIS "
            f"validation sequences).**")


def analytic_cell(a: str) -> str:
    """The analytic column: arithmetic is set as math, a bare count stays plain."""
    if a == "-":
        return DASH
    if "+" in a or "/" in a:
        return f"${a}$"
    return a


def md_table(rows: Sequence[Dict[str, Optional[str]]]) -> str:
    lines = ["| Configuration | Encoder calls / frame | Relative to baseline | "
             "Mean FPS | Peak memory |",
             "|---|---|---|---|---|"]
    for r in rows:
        rel = f"{r['relative']}$\\times$" if r["relative"] else DASH
        fps = r["fps"] or DASH
        mem = f"{r['peak_gb']} GB" if r["peak_gb"] else DASH
        lines.append(f"| {r['label']} | {analytic_cell(r['analytic'] or '')} | "
                     f"{rel} | {fps} | {mem} |")
    lines.append(MS_ANCHOR)
    return "\n".join(lines)


def footprint_clause(rows: Sequence[Dict[str, str]],
                     summary: Dict[str, Dict[str, float]]) -> str:
    """The memory sentence, in the strongest form the per-sequence data supports.

    Checked per sequence, not on the means: "the memory-free arms use less memory"
    has to be true of every profiled sequence, or it is a statement about an
    average that no single run exhibits.  When the comparison is not available
    (no memory-free arm, no sequence, an arm missing from some sequence) the
    clause is *omitted* -- an all() over an empty set is vacuously true, which is
    how a comparison guard silently passes on nothing.
    """
    free = [m for m in summary if m != BASELINE_MODE]
    seqs = sorted({r["seq"] for r in rows})
    per: Dict[Tuple[str, str], float] = {(r["seq"], r["mode"]): float(r["gpu_peak_mb"])
                                         for r in rows}
    if not free or not seqs:
        return ""
    comparable = True
    base_bigger = True
    for s in seqs:
        if (s, BASELINE_MODE) not in per:
            comparable = False
            break
        for m in free:
            if (s, m) not in per:
                comparable = False
                break
            if not per[(s, BASELINE_MODE)] > per[(s, m)]:
                base_bigger = False
    if not comparable:
        return ""
    lo_free = min(summary[m]["peak_mb"] for m in free)
    flat_free = all(
        (summary[m]["peak_hi"] - summary[m]["peak_lo"]) <= FLAT_TOL * summary[m]["peak_mb"]
        for m in free)
    flat_base = (summary[BASELINE_MODE]["peak_hi"]
                 - summary[BASELINE_MODE]["peak_lo"]
                 ) <= FLAT_TOL * summary[BASELINE_MODE]["peak_mb"]
    if not base_bigger:
        return (f"At this scope the memory-free arms' peak allocator footprint "
                f"averages {lo_free / 1024.0:.2f} GB against the baseline's "
                f"{summary[BASELINE_MODE]['peak_mb'] / 1024.0:.2f} GB, but the "
                f"baseline is not the larger one on every profiled sequence, so "
                f"no per-sequence ordering is claimed.")
    tail = (f" The memory-free arms hold the same footprint on every profiled "
            f"sequence while the baseline's varies with the clip, which is the "
            f"deployment difference this column exists to record."
            if (flat_free and not flat_base) else "")
    return (f"At this scope the deployed arm costs "
            f"{summary['dagrs']['calls_per_frame'] / summary[BASELINE_MODE]['calls_per_frame']:.2f}"
            f"$\\times$ the baseline, and its peak allocator footprint is "
            f"{lo_free / 1024.0:.2f} GB against the baseline's "
            f"{summary[BASELINE_MODE]['peak_mb'] / 1024.0:.2f} GB "
            f"({summary[BASELINE_MODE]['peak_mb'] / lo_free:.2f}$\\times$ lower); the "
            f"baseline is the larger of the two on every profiled sequence." + tail)


def footnote_line(rows: Sequence[Dict[str, str]], sc: Dict[str, Any],
                  summary: Dict[str, Dict[str, float]],
                  notes: Sequence[str]) -> str:
    """The sentence(s) that make the table legible, owned here so they cannot drift.

    Two things a reader cannot recover from the table itself: the conditions the
    measured columns were taken at, and why one row's measured cells are empty.
    """
    got = ana = ""
    for n in notes:
        if n.startswith("dagrs_no_reanchor"):
            got = n.split(": ")[1].split(" calls/frame")[0]
            ana = n.split(" vs analytic ")[1].split(" ")[0]
    meas = (f" The measured call rates reproduce the analytic column ({got} vs {ana} "
            f"and {summary[BASELINE_MODE]['calls_per_frame']:.3f} vs 1.000)."
            if got and ana else "")
    fps_lo = min(summary[m]["fps_lo"] for m in summary)
    fps_hi = max(summary[m]["fps_hi"] for m in summary)
    foot = footprint_clause(rows, summary)
    b = summary[BASELINE_MODE]
    d = summary["dagrs"] if "dagrs" in summary else b
    head = (f"`scripts/06_profile.py` measured these rows at frame stride "
            f"{sc['stride']} on single-degradation `{sc['degradation']}` at level "
            f"{sc['level']:.1f} over the first {sc['n_seq']} DAVIS validation "
            f"sequences (clip lengths {sc['frames_lo']}\u2013{sc['frames_hi']} frames), "
            f"with equal weight per sequence. FPS is a single-machine wall-clock "
            f"measurement and varies widely inside an arm (the baseline spans "
            f"{b['fps_lo']:.2f}\u2013{b['fps_hi']:.2f} across these clips, the deployed "
            f"arm {d['fps_lo']:.2f}\u2013{d['fps_hi']:.2f}), so the mean is a summary, "
            f"not a constant. The $J{{=}}M{{=}}13$ row's three measured cells are left "
            f"as an em dash: no profiled arm evaluates the bank over all $M$ operators "
            f"at every decision frame, so that entry is the analytic price of the "
            f"mechanism's own formula at $J=M$, not a measurement.")
    return head + meas + (" " + foot if foot else "")


def fw_block(rows: Sequence[Dict[str, Optional[str]], ], sc: Dict[str, Any],
             summary: Dict[str, Dict[str, float]]) -> str:
    """The framework mirror: every token the manuscript gained must appear here."""
    b = summary[BASELINE_MODE]
    lines = [FW_ANCHOR,
             f"[[TABLE-7]] 时间/显存开销（**已测**；口径 = `scripts/06_profile.py`，"
             f"stride {sc['stride']}，单档 `{sc['degradation']}` @ {sc['level']:.1f}，"
             f"前 {sc['n_seq']} 条 DAVIS val（片段 {sc['frames_lo']}\u2013{sc['frames_hi']} 帧）；"
             f"FPS/显存为**按序列等权均值**，FPS 逐序列跨度：基线 "
             f"{b['fps_lo']:.2f}\u2013{b['fps_hi']:.2f}、部署臂 "
             f"{summary['dagrs']['fps_lo']:.2f}\u2013{summary['dagrs']['fps_hi']:.2f}）",
             ""]
    lines.append("| 配置 | 每帧编码次数 | 相对基线 | 平均 FPS | 峰值显存 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        rel = f"{r['relative']}×" if r["relative"] else DASH
        fps = r["fps"] or DASH
        mem = f"{r['peak_gb']} GB" if r["peak_gb"] else DASH
        label = (r["label"] or "").replace("$K{=}5$", "K=5").replace("$J{=}3$", "J=3") \
                                 .replace("$J{=}M{=}13$", "J=M=13")
        lines.append(f"| {label} | {analytic_cell(r['analytic'] or '')} | {rel} | "
                     f"{fps} | {mem} |")
    lines.append("")
    lines.append(f"（launcher：`python scripts/06_profile.py --config configs/_tau013.yaml "
                 f"--frame-stride {sc['stride']} --degradation {sc['degradation']} "
                 f"--level {sc['level']:.1f} --seqs {sc['n_seq']} "
                 f"--modes {','.join(sc['modes'])} --out results/p1/table7_profile.csv`）")
    lines.append(f"（实测每帧编码次数复现解析列：sam2video "
                 f"{summary[BASELINE_MODE]['calls_per_frame']:.3f} vs 1.000、"
                 f"dagrs_no_reanchor {summary['dagrs_no_reanchor']['calls_per_frame']:.3f} "
                 f"vs 1.600；发布比 "
                 f"{summary['dagrs']['calls_per_frame'] / summary[BASELINE_MODE]['calls_per_frame']:.2f}"
                 f" = dagrs/sam2video。）")
    lines.append(f"（`J=M=13` 行**无对应臂** ⇒ 三格留 `{DASH}`，不得用邻行填数；"
                 f"该行只是机制自身代价式在 $J=M$ 处的解析价。）")
    free = [m for m in summary if m != BASELINE_MODE]
    if free:
        lo_free = min(summary[m]["peak_mb"] for m in free)
        lines.append(f"（峰值显存：记忆自由臂 {lo_free / 1024.0:.2f} GB vs 基线 "
                     f"{summary[BASELINE_MODE]['peak_mb'] / 1024.0:.2f} GB ⇒ "
                     f"{summary[BASELINE_MODE]['peak_mb'] / lo_free:.2f}×，"
                     f"逐序列对比见下表脚注。）")
    lines.append(f"（另有一个**不同口径**的代价估计：P1 网格段等权 `n_encoder_calls` "
                 f"⇒ 1.97（见 §5.1b 与 §4.8 正文）。两者**分母同为 `sam2video`**，"
                 f"差别只在条件与帧数；上表 {summary['dagrs']['calls_per_frame'] / summary[BASELINE_MODE]['calls_per_frame']:.2f}"
                 f" 是 `dagrs`/`sam2video`，1.97 是同一比值在复合网格上的值。）")
    return "\n".join(lines)


# ------------------------------------------------------------------- checking #

def ms_block(text: str) -> List[str]:
    """The pipe rows of the Sec. 4.8 table (the block ending at MS_ANCHOR)."""
    if MS_ANCHOR not in text:
        raise SystemExit(f"ABORT: {MS_ANCHOR} is not in the manuscript, so the "
                         "Sec. 4.8 table cannot be located")
    head = text[:text.index(MS_ANCHOR)]
    block: List[str] = []
    for line in reversed(head.splitlines()):
        if line.lstrip().startswith("|"):
            block.append(line.strip())
        elif block:
            break
    block.reverse()
    if len(block) < 3:
        raise SystemExit("ABORT: fewer than three pipe rows precede the Sec. 4.8 "
                         "anchor, so no table was found")
    return block


def cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def ms_caption(text: str) -> str:
    """The line that introduces Table 7."""
    for line in text.splitlines():
        if line.startswith("**Table 7."):
            return line
    raise SystemExit("ABORT: no line starting with '**Table 7.' in the manuscript")


def fw_region(text: str) -> str:
    """The framework block: from FW_ANCHOR to the next `## ` heading."""
    if FW_ANCHOR not in text:
        raise SystemExit(f"ABORT: {FW_ANCHOR} is not in the framework, so the "
                         "mirror of Sec. 4.8 cannot be located")
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if FW_ANCHOR in ln)
    j = i + 1
    while j < len(lines) and not lines[j].startswith("## "):
        j += 1
    return "\n".join(lines[i:j]).strip("\n")


def check_documents(rows: Sequence[Dict[str, str]],
                    built: Sequence[Dict[str, Optional[str]]],
                    sc: Dict[str, Any], summary: Dict[str, Dict[str, float]],
                    text: str) -> List[str]:
    """Every measured cell, the caption and the footnote must equal what we emit."""
    problems: List[str] = []
    want_cap = caption_line(sc)
    got_cap = ms_caption(text)
    if got_cap != want_cap:
        problems.append(f"caption does not match the profile's scope:\n"
                        f"     manuscript: {got_cap}\n     expected  : {want_cap}")
    block = ms_block(text)
    if len(block) != len(built) + 2:
        problems.append(f"the manuscript's Sec. 4.8 table has {len(block)} rows, "
                        f"expected {len(built) + 2}")
        return problems
    for line, want in zip(block[2:], built):
        got = cells(line)
        if len(got) != 5:
            problems.append(f"row {want['label']!r} has {len(got)} cells")
            continue
        if got[0] != want["label"]:
            problems.append(f"row label {got[0]!r} != {want['label']!r}")
            continue
        exp = [want["relative"], want["fps"], want["peak_gb"]]
        for col, e in zip((2, 3, 4), exp):
            cell = got[col].replace("$\\times$", "").replace("GB", "").strip()
            if e is None:
                if cell != DASH:
                    problems.append(f"{want['label']}: column {col} is {cell!r} "
                                    f"but no measured arm exists for this row, so "
                                    f"it must stay {DASH!r}")
            else:
                if cell != e:
                    problems.append(f"{want['label']}: column {col} is {cell!r} "
                                    f"but the profiler gives {e!r}")
    fn = footnote_line(rows, sc, summary, check_mapping(summary))
    if fn not in text:
        problems.append("the scope footnote generated for the profile is not in "
                        "the manuscript (the table would then be quoted without "
                        "its scope or its explanation of the unmeasured row):\n"
                        f"     {fn[:160]}...")
    return problems


def check_framework(rows: Sequence[Dict[str, Optional[str]]], sc: Dict[str, Any],
                    summary: Dict[str, Dict[str, float]], text: str) -> List[str]:
    """The framework mirror must be exactly what this run would emit."""
    want = fw_block(rows, sc, summary).strip("\n")
    got = fw_region(text).strip("\n")
    if got == want:
        return []
    wl, gl = want.splitlines(), got.splitlines()
    problems = []
    for k in range(max(len(wl), len(gl))):
        a = wl[k] if k < len(wl) else "<missing>"
        b = gl[k] if k < len(gl) else "<missing>"
        if a != b:
            problems.append(f"framework mirror line {k + 1}:\n"
                            f"     framework: {b}\n     expected : {a}")
    return problems


# --------------------------------------------------------------------------- #

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
            ok(name, needle in str(exc), str(exc)[:110])
        else:
            ok(name, False, "no abort was raised")

    print("126 self-test")
    print("-" * 66)

    ok("four Table 7 rows are declared, in the manuscript's order",
       len(ROWS) == 4 and ROWS[0][0].startswith("SAM 2.1-L")
       and ROWS[-1][0].startswith("DAG-RS + re-anchor"))
    ok("exactly one row is analytic-only, and it is the J=M=13 row",
       sum(1 for _l, m, _a in ROWS if m is None) == 1
       and ROWS[2][1] is None)
    ok("the analytic column is the documented arithmetic",
       ROWS[0][2] == "1" and ROWS[1][2] == "1+3/5" and ROWS[2][2] == "1+13/5")

    fake = [{"seq": s, "mode": m, "degradation": "fog", "level": "4.0",
             "n_frames": "100", "encoder_calls": c, "fps": f,
             "gpu_peak_mb": p}
            for s in ("a", "b")
            for m, c, f, p in (("sam2video", "100", "4.0", "8192"),
                               ("dagrs_no_reanchor", "160", "2.5", "2048"),
                               ("dagrs", "197", "0.8", "2048"))]
    s = summarise(fake)
    sc = scope(fake, 1)
    ok("calls per frame is computed as a ratio of sums, not a mean of ratios",
       abs(s["sam2video"]["calls_per_frame"] - 1.0) < 1e-12
       and abs(s["dagrs_no_reanchor"]["calls_per_frame"] - 1.6) < 1e-12,
       f"{s['sam2video']['calls_per_frame']:.3f} / "
       f"{s['dagrs_no_reanchor']['calls_per_frame']:.3f}")
    ok("the mapping agrees with the analytic call rates",
       len(check_mapping(s)) == 3, "; ".join(check_mapping(s)))
    rws = build_rows(s)
    ok("the baseline row is 1.00x by construction",
       rws[0]["relative"] == "1.00", str(rws[0]))
    ok("the re-anchor row is priced above the K=5,J=3 row",
       float(rws[3]["relative"]) > float(rws[1]["relative"]),
       f"{rws[3]['relative']} vs {rws[1]['relative']}")
    ok("the analytic-only row keeps its measured cells unmeasured",
       all(rws[2][k] is None for k in ("relative", "fps", "peak_gb")),
       str(rws[2]))
    ok("memory is reported in GB, not MB",
       rws[0]["peak_gb"] == "8.00", str(rws[0]["peak_gb"]))
    ok("the caption carries the scope, not just the words 'Table 7'",
       all(t in caption_line(sc) for t in ("Table 7", "stride 1", "fog", "4.0",
                                           "2 DAVIS")),
       caption_line(sc)[:96])
    ok("the unmeasured cells are emitted as an em dash, not a placeholder",
       DASH in md_table(rws).splitlines()[4]
       and "TODO" not in md_table(rws)
       and md_table(rws).splitlines()[4].count(DASH) == 3,
       md_table(rws).splitlines()[4])
    fn = footnote_line(fake, sc, s, check_mapping(s))
    ok("the footnote states the em-dash reason and the scope it was measured at",
       "no profiled arm evaluates the bank over all" in fn
       and "frame stride 1" in fn and "2 DAVIS validation sequences" in fn, fn[:96])
    ok("the footnote carries the profile's own cost ratio",
       "1.97" in fn and "costs 1.97" in fn, fn[-120:])
    ok("the footnote reports the footprint when the baseline is the larger one",
       "peak allocator footprint is 2.00 GB" in fn and "8.00 GB" in fn, fn[-160:])
    ok("the framework mirror names its anchor and records the launcher",
       fw_block(rws, sc, s).startswith(FW_ANCHOR)
       and "--seqs 2" in fw_block(rws, sc, s)
       and "results/p1/table7_profile.csv" in fw_block(rws, sc, s),
       fw_block(rws, sc, s).splitlines()[0])

    # a profile where the baseline is NOT the larger one on every sequence: the
    # per-sequence ordering claim must go, rather than become false
    swapped = json.loads(json.dumps(fake))
    for r in swapped:
        if r["seq"] == "b" and r["mode"] == "sam2video":
            r["gpu_peak_mb"] = "1024"
    s2 = summarise(swapped)
    fc = footprint_clause(swapped, s2)
    ok("the footprint sentence drops the per-sequence ordering claim when it is false",
       "no per-sequence ordering is claimed" in fc
       and "larger of the two on every profiled sequence" not in fc, fc[:110])
    ok("an empty comparison set yields no footprint claim at all, not a vacuous one",
       footprint_clause([], s2) == "", repr(footprint_clause([], s2)))

    bad = json.loads(json.dumps(fake))
    for r in bad:
        if r["mode"] == "dagrs_no_reanchor":
            r["encoder_calls"] = "400"          # 4.0 calls/frame, not 1.6
    must_abort(lambda: check_mapping(summarise(bad)),
               "a mode whose call rate contradicts its label aborts",
               "do not support the configuration labels")
    swapped2 = json.loads(json.dumps(fake))
    for r in swapped2:
        if r["mode"] == "dagrs_no_reanchor":
            r["encoder_calls"], r["fps"] = "197", "0.9"
        elif r["mode"] == "dagrs":
            r["encoder_calls"] = "160"
    must_abort(lambda: check_mapping(summarise(swapped2)),
               "a profiler where +re-anchor is cheaper than -re-anchor aborts",
               "does not cost more")
    mixed = json.loads(json.dumps(fake))
    mixed[0]["degradation"] = "dust"
    must_abort(lambda: scope(mixed, 1),
               "a profile that mixes conditions aborts rather than being averaged",
               "mixes conditions")
    no_base = [r for r in fake if r["mode"] != "sam2video"]
    must_abort(lambda: summarise(no_base),
               "a profile with no baseline arm aborts", "no baseline")
    zero = json.loads(json.dumps(fake))
    zero[0]["n_frames"] = "0"
    must_abort(lambda: summarise(zero),
               "a zero frame count aborts instead of dividing by zero",
               "non-positive")
    must_abort(lambda: check_protocol(fake, 4),
               "a non-stride-1 profiler setting aborts", "stride")
    must_abort(lambda: check_protocol(fake, 1),
               "profiled clip lengths that are not the stride-1 lengths abort",
               "not on the paper's protocol")
    must_abort(lambda: ms_block("no anchor here"), "a manuscript with no anchor aborts",
               "cannot be located")
    must_abort(lambda: fw_region("no anchor here either"),
               "a framework with no mirror anchor aborts", "cannot be located")
    must_abort(lambda: read_profile(ROOT / "_scratch" / "does_not_exist.csv"),
               "a missing profile aborts rather than reporting zero rows",
               "missing")

    ok("the emitted markdown is a five-column pipe table + anchor",
       md_table(rws).splitlines()[0].startswith("| Configuration |")
       and md_table(rws).splitlines()[-1] == MS_ANCHOR
       and len(md_table(rws).splitlines()) == len(ROWS) + 3,
       f"{len(md_table(rws).splitlines())} lines, last = "
       f"{md_table(rws).splitlines()[-1]}")

    print("-" * 66)
    if fails:
        print(f"126 --self-test: FAIL ({len(fails)}/{checks})")
        for f in fails:
            print("   " + f)
        return 1
    print(f"126 --self-test: PASS ({checks} controls)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Table 7 rows from the cost profile")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = read_profile()
    proto = check_protocol(rows, args.frame_stride)
    sc = scope(rows, args.frame_stride)
    summary = summarise(rows)
    notes = check_mapping(summary)
    built = build_rows(summary)
    table = md_table(built)
    foot = footnote_line(rows, sc, summary, notes)
    fw = fw_block(built, sc, summary)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"rows": built, "summary": summary, "mapping_notes": notes,
                        "protocol": proto, "scope": {k: v for k, v in sc.items()
                                                     if k != "seqs"},
                        "sequences": sc["seqs"], "caption": caption_line(sc),
                        "footnote": foot, "framework_block": fw}, indent=2) + "\n",
            encoding="utf-8")
        print(f"wrote {args.json}")
        return 0

    if args.emit:
        print("### manuscript: caption")
        print(caption_line(sc))
        print()
        print("### manuscript: table")
        print(table)
        print()
        print("### manuscript: footnote (one paragraph, directly after the table)")
        print(foot)
        print()
        print("### framework: mirror")
        print(fw)
        return 0

    if args.check:
        st, msg = release_gate(
            "126", ROOT, [MS.relative_to(ROOT).as_posix(),
                          FRAMEWORK.relative_to(ROOT).as_posix()],
            "the Sec. 4.8 Table 7 document check")
        if st != "ok":
            print(msg)
            return 1 if st == "fail" else 0
        ms_text = MS.read_text(encoding="utf-8")
        fw_text = FRAMEWORK.read_text(encoding="utf-8")
        problems = check_documents(rows, built, sc, summary, ms_text)
        problems += check_framework(built, sc, summary, fw_text)
        for n in notes:
            print(f"[126] mapping: {n}")
        print(f"[126] protocol: {proto}")
        if problems:
            for p in problems:
                print(f"[126] FAIL: {p}")
            return 1
        print("[126] --check: caption, every measured cell and the scope footnote "
              "match the profiler, and the framework mirror is current")
        print("[126] VERDICT: PASS")
        return 0

    print("Table 7 (Sec. 4.8) from scripts/06_profile.py --frame-stride 1")
    print("=" * 74)
    print(f"  protocol : {proto}")
    print(f"  scope    : {sc['n_seq']} sequences, {sc['degradation']} @ "
          f"{sc['level']:.1f}, modes {','.join(sc['modes'])}")
    for n in notes:
        print(f"  mapping  : {n}")
    print()
    for line in table.splitlines():
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
