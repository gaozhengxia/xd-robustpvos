"""TABLE-13 ablation table builder (CL6-CL8, CL11).

`04_ablation.py` has no `--resume`, so the TABLE-13 sweep is run one
(arm, phase) per invocation (`_scratch/ablation_20260920/run_ablation2.sh`) and
each unit lands its own CSV.  This script turns those 16 files into the table.

Aggregation is the project's standard one, not a new convention:
`05_analysis.sequence_level` -> `05_analysis.collapse_objects`, i.e. instances
are averaged into their sequence first and the 30 sequences are then weighted
equally (DAVIS `max_objects: 0` scores 61 instances).

Per arm it reports, paired against A0 over the same 30 sequences:

  * J&F           : mean over sequences (the number that goes in the paper)
  * delta vs A0   : paired mean, percentile bootstrap 95% CI
  * Wilcoxon p    : two-sided signed-rank, Holm-corrected over the 7 arms
  * Cliff's delta : effect size, POSITIVE => the arm beats A0
  * win / loss    : sequences where the arm is above / below A0

The two phases are reported SEPARATELY, never pooled: A0 scores ~0.650 on `fog`
and ~0.459 on the compound `C1_fog_noise`, so a pooled column would be an
average over two different difficulty regimes.  The A0 pairing is only
meaningful within a phase.

A0 IS the full method under `configs/_tau013.yaml`, so it must reproduce the
`dagrs` column of `results/table9_main.csv`.  That guard is re-run here and
aborts on mismatch, because a silent protocol drift would invalidate the whole
table -- the exact failure mode this project has already paid for twice.

Usage
-----
    python scripts/110_ablation_table.py --dir results/ablation \\
        --out-dir results --prefix ablation_table13

Guard rails (all raise, none warn-and-continue)
-----------------------------------------------
* every expected (phase, arm) file must exist and hold exactly 61 instance rows;
* a file must contain ONLY its own phase (mixed phases raise);
* every arm of a phase must cover the same 30 sequences;
* A0 must match `results/table9_main.csv` to 1e-6.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from xdrp.stats import holm_bonferroni, paired_report  # noqa: E402

EXPECT_INSTANCES = 61   # 30 DAVIS-2017 val sequences, 61 annotated objects
EXPECT_SEQUENCES = 30
JTOL = 1e-6

#: Blackwell/Cohen-style thresholds for Cliff's delta.  THIS PROJECT JUDGES BY
#: EFFECT SIZE, NOT BY p (n = 30 paired sequences makes tiny shifts significant
#: on their own).  An arm whose |delta| is negligible is reported as "no material
#: contribution" even when its p-value clears 0.05 -- writing "the component
#: contributes" off a significant p with cliff = 0.08 is exactly the misreading
#: the CL3 write-up rules forbid.
CLIFF_NEGLIGIBLE = 0.147
CLIFF_SMALL = 0.330

#: (file tag, degradation, level) -- tag is what `run_ablation2.sh` prefixes.
PHASES: Tuple[Dict[str, Any], ...] = (
    {"tag": "fog", "degradation": "fog", "level": 3.0},
    {"tag": "C1", "degradation": "C1_fog_noise", "level": 3.0},
)

ARMS: Tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")

#: The hypothesis each arm is the control for (mirrors PAPER_FRAMEWORK TABLE-13).
ARM_QUESTION: Dict[str, str] = {
    "A0": "reference",
    "A1": "is the gain only TTA?",
    "A2": "does the method depend on optical flow?",
    "A3": "is the SELECTION rule itself doing work?",
    "A4": "is adaptivity necessary?",
    "A5": "efficiency/performance trade-off (CL8)",
    "A6": "contribution of the vacuity gate",
    "A7": "contribution of global re-anchoring",
}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a05 = _load("a05", "scripts/05_analysis.py")


def read_arm(path: Path, degradation: str, level: float) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"ABORT: {path} is empty")
    others = sorted({r["degradation"] for r in rows} - {degradation})
    if others:
        raise SystemExit(
            f"ABORT: {path} contains degradations {others} besides "
            f"'{degradation}' -- one file must hold exactly one (arm, phase)")
    return rows


def sequence_values(rows: Sequence[Dict[str, str]], path: Path) -> Tuple[
        Dict[str, float], Dict[str, float]]:
    """-> ({seq: J&F}, {seq: mean n_encoder_calls}) using the standard collapse."""
    col = a05.collapse_objects(a05.sequence_level(rows))
    jf: Dict[str, float] = {}
    enc_acc: Dict[str, List[float]] = {}
    for r in col:
        jf[str(r["seq"])] = float(r["J&F"])
    # encoder calls: per-instance rows, averaged to the sequence the same way
    buckets: Dict[str, List[float]] = {}
    for r in rows:
        v = r.get("n_encoder_calls", "")
        if v not in ("", None):
            buckets.setdefault(str(r["seq"]), []).append(float(v))
    enc = {s: sum(v) / len(v) for s, v in buckets.items()}
    return jf, enc


def a0_guard(jf_by_phase: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """A0 must equal the authoritative `dagrs` cell of results/table9_main.csv."""
    want: Dict[Tuple[str, float], float] = {}
    with (ROOT / "results" / "table9_main.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["mode"].strip() == "dagrs":
                want[(r["degradation"], float(r["level"]))] = float(r["J&F_mean"])
    out: Dict[str, float] = {}
    for ph in PHASES:
        seqs = sorted(jf_by_phase[ph["tag"]])
        got = sum(jf_by_phase[ph["tag"]][s] for s in seqs) / len(seqs)
        key = (ph["degradation"], float(ph["level"]))
        if key not in want:
            raise SystemExit(f"ABORT: table9_main.csv has no dagrs row for {key}")
        d = abs(got - want[key])
        print(f"A0 guard {ph['tag']}@{ph['level']:.0f}: A0={got:.6f} "
              f"table9_main={want[key]:.6f}  |d|={d:.3e}")
        if d > JTOL:
            raise SystemExit(
                "ABORT: A0 does not reproduce the authoritative dagrs value "
                f"({d:.3e} > {JTOL:.0e}) -> protocol/config drift; the ablation "
                "grid is NOT on the paper's protocol.")
        out[ph["tag"]] = got
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="_scratch/ablation_20260920",
                    help="directory holding <tag>_<arm>.csv")
    ap.add_argument("--out-dir", default="_scratch/ablation_20260920/table13_dry",
                    help="where the table lands (use an isolated dir to preview)")
    ap.add_argument("--prefix", default="ablation_table13")
    args = ap.parse_args(argv)

    src = Path(args.dir)
    if not src.is_absolute():
        src = ROOT / src
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_phase: Dict[str, Dict[str, Dict[str, float]]] = {}
    enc_phase: Dict[str, Dict[str, Dict[str, float]]] = {}
    noop_phase: Dict[str, Dict[str, Tuple[int, float]]] = {}
    notes: Dict[str, str] = {}

    for ph in PHASES:
        tag = ph["tag"]
        per_phase[tag] = {}
        enc_phase[tag] = {}
        noop_phase[tag] = {}
        #: A0's per-instance J&F, needed by the no-op guard below.  ARMS[0] is
        #: "A0", so it is always populated before any other arm is read.
        a0_inst: Dict[Tuple[str, str], float] = {}
        for arm in ARMS:
            path = src / f"{tag}_{arm}.csv"
            if not path.exists():
                raise SystemExit(f"ABORT: missing {path}")
            rows = read_arm(path, ph["degradation"], float(ph["level"]))
            if len(rows) != EXPECT_INSTANCES:
                raise SystemExit(
                    f"ABORT: {path.name} has {len(rows)} instance rows, expected "
                    f"{EXPECT_INSTANCES} -- incomplete unit, do NOT tabulate")
            jf, enc = sequence_values(rows, path)
            if len(jf) != EXPECT_SEQUENCES:
                raise SystemExit(
                    f"ABORT: {path.name} covers {len(jf)} sequences, expected "
                    f"{EXPECT_SEQUENCES}")
            per_phase[tag][arm] = jf
            enc_phase[tag][arm] = enc
            inst = {(r["seq"], r["obj_id"]): float(r["J&F"]) for r in rows}
            if arm == "A0":
                a0_inst = inst
            if not a0_inst:
                raise SystemExit("internal: A0 must be read before any other arm")
            noop_phase[tag][arm] = (
                sum(1 for k, v in inst.items()
                    if k in a0_inst and abs(v - a0_inst[k]) > 0.0),
                max((abs(v - a0_inst[k]) for k, v in inst.items()
                     if k in a0_inst), default=0.0),
            )
            note = rows[0].get("arm_note", "").strip()
            if note:
                notes.setdefault(arm, note)
            print(f"  read {tag}/{arm}: {len(rows)} instances, {len(jf)} seqs, "
                  f"J&F={sum(jf.values()) / len(jf):.6f}")

        seqsets = {a: frozenset(per_phase[tag][a]) for a in ARMS}
        ref = seqsets["A0"]
        bad = {a: s for a, s in seqsets.items() if s != ref}
        if bad:
            raise SystemExit(
                f"ABORT: {tag}: arms {sorted(bad)} do not cover the same 30 "
                "sequences as A0 -- the pairing would silently drop sequences")

    # ---- NO-OP GUARD -------------------------------------------------------
    # An arm that is bit-identical to A0 on all 61 instances is an arm whose
    # override never reached the pipeline.  Its zero delta would then be a
    # measurement artifact, and writing it up as "this component is
    # unnecessary" would be a fabrication -- so refuse to tabulate instead.
    dead = [(t, a) for t, dd in noop_phase.items() for a, (n, _m) in dd.items()
            if a != "A0" and n == 0]
    if dead:
        raise SystemExit(
            f"ABORT: arm(s) {dead} are bit-identical to A0 on every instance "
            f"-> the override never reached the pipeline; a zero delta here is "
            f"an artifact, NOT a finding")
    for t, dd in sorted(noop_phase.items()):
        worst = min(n for a, (n, _m) in dd.items() if a != "A0")
        print(f"effect check {t}: every arm changes at least {worst}/{EXPECT_INSTANCES} "
              "instances (overrides reach the pipeline)")

    a0_guard({t: per_phase[t]["A0"] for t in per_phase})

    # ---- per-arm statistics, paired against A0 within each phase ------------
    recs: List[Dict[str, Any]] = []
    for ph in PHASES:
        tag = ph["tag"]
        seqs = sorted(per_phase[tag]["A0"])
        base = [per_phase[tag]["A0"][s] for s in seqs]
        rows_here: List[Dict[str, Any]] = []
        for arm in ARMS:
            if arm == "A0":
                continue
            vals = [per_phase[tag][arm][s] for s in seqs]
            rep = paired_report(base, vals, baseline_name="A0",
                                method_name=arm, seed=0)
            diffs = [b - a for a, b in zip(base, vals)]
            rows_here.append({
                "phase": tag,
                "degradation": ph["degradation"],
                "level": float(ph["level"]),
                "arm": arm,
                "arm_note": notes.get(arm, ""),
                "question": ARM_QUESTION.get(arm, ""),
                "jf": sum(vals) / len(vals),
                "jf_a0": sum(base) / len(base),
                "delta": rep["mean_delta"],
                "median_delta": rep["median_delta"],
                "delta_ci_lo": rep["delta_ci_lo"],
                "delta_ci_hi": rep["delta_ci_hi"],
                "wilcoxon_p": rep["wilcoxon_p"],
                "cliffs_delta": rep["cliffs_delta"],
                "n_sequences": len(seqs),
                "wins": sum(1 for d in diffs if d > 0),
                "losses": sum(1 for d in diffs if d < 0),
                "n_encoder_calls": (sum(enc_phase[tag][arm].values())
                                    / max(1, len(enc_phase[tag][arm]))),
            })
        adj = holm_bonferroni([r["wilcoxon_p"] for r in rows_here])
        for r, p in zip(rows_here, adj):
            r["holm_p"] = float(p)
        recs.extend(rows_here)

    for ph in PHASES:
        tag = ph["tag"]
        base_enc = (sum(enc_phase[tag]["A0"].values())
                    / max(1, len(enc_phase[tag]["A0"])))
        recs.append({
            "phase": tag, "degradation": ph["degradation"],
            "level": float(ph["level"]), "arm": "A0", "arm_note": notes.get("A0", ""),
            "question": ARM_QUESTION["A0"], "jf": None, "jf_a0": None,
            "delta": 0.0, "median_delta": 0.0, "delta_ci_lo": None,
            "delta_ci_hi": None, "wilcoxon_p": None, "holm_p": None,
            "cliffs_delta": None, "n_sequences": EXPECT_SEQUENCES,
            "wins": None, "losses": None, "n_encoder_calls": base_enc,
        })
    recs.sort(key=lambda r: (0 if r["phase"] == "fog" else 1, ARMS.index(r["arm"])))

    # fill A0's J&F back in (it is the phase's own mean)
    for r in recs:
        if r["arm"] == "A0":
            seqs = sorted(per_phase[r["phase"]]["A0"])
            r["jf"] = sum(per_phase[r["phase"]]["A0"][s] for s in seqs) / len(seqs)
            r["jf_a0"] = r["jf"]

    # ---- verdict text: EFFECT SIZE FIRST -----------------------------------
    # n = 30 paired sequences makes a 0.02 shift "significant" on its own, so a
    # bare p < 0.05 must never be written up as "the component contributes".
    # Magnitude is decided by Cliff's delta; p only qualifies a non-negligible
    # magnitude.  Same discipline the CL3 write-up follows.
    for r in recs:
        if r["arm"] == "A0":
            r["reading"] = "reference"
            continue
        d = abs(r["cliffs_delta"])
        sig = r["holm_p"] is not None and r["holm_p"] < 0.05
        if d < CLIFF_NEGLIGIBLE:
            r["reading"] = (f"no material contribution (delta={r['delta']:+.4f}, "
                            f"cliff={r['cliffs_delta']:+.3f} < {CLIFF_NEGLIGIBLE}; "
                            f"{r['wins']}/{r['losses']} seqs, Holm p={r['holm_p']:.2g})")
        elif sig:
            r["reading"] = (f"{'WORSE' if r['delta'] < 0 else 'BETTER'} than A0, "
                            f"small effect (delta={r['delta']:+.4f}, "
                            f"cliff={r['cliffs_delta']:+.3f}, Holm p={r['holm_p']:.2g})")
        else:
            r["reading"] = (f"non-negligible but NOT significant "
                            f"(delta={r['delta']:+.4f}, "
                            f"cliff={r['cliffs_delta']:+.3f}, Holm p={r['holm_p']:.2g})")

    payload = {
        "note": ("TABLE-13 ablation. Peak memory / no, this is behavior: each arm "
                 "is the full method with one component removed or replaced, run "
                 "on the paper's own protocol (configs/_tau013.yaml, stride 1, "
                 "max_objects 0). Aggregation = sequence_level -> collapse_objects "
                 "-> equal-weight mean over the 30 sequences. Phases are reported "
                 "separately and are NEVER pooled: A0 is 0.650 on fog and 0.459 on "
                 "the compound C1, so a pooled column would mix difficulty regimes. "
                 "The A0 guard against results/table9_main.csv ran here and passed, "
                 "so the grid is on the paper's protocol."),
        "input_dir": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "protocol": {"config": "configs/_tau013.yaml", "frame_stride": 1,
                     "levels": sorted({float(p["level"]) for p in PHASES}),
                     "expect_instances": EXPECT_INSTANCES,
                     "expect_sequences": EXPECT_SEQUENCES},
        "statistics": {"paired_unit": "sequence",
                       "delta": "arm - A0",
                       "ci": "percentile bootstrap 95% over the 30 paired deltas",
                       "p": "two-sided Wilcoxon signed-rank vs A0, Holm over the 7 arms of the phase",
                       "cliffs_delta": "positive => arm beats A0"},
        "arm_questions": ARM_QUESTION,
        "arm_effect_check": {
            t: {a: {"instances_changed": n, "max_abs_jf_diff": m}
                for a, (n, m) in sorted(dd.items())}
            for t, dd in sorted(noop_phase.items())},
        "reading_rule": (f"effect size first: |Cliff's delta| < {CLIFF_NEGLIGIBLE} "
                         "=> 'no material contribution' regardless of p"),
        "rows": recs,
    }

    jpath = out_dir / f"{args.prefix}.json"
    jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    cpath = out_dir / f"{args.prefix}.csv"
    with cpath.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    L: List[str] = []
    L.append("# TABLE-13 消融矩阵（脚本生成，勿手改）")
    L.append("")
    L.append(f"- 协议：`configs/_tau013.yaml`，stride 1，level 3，max_objects 0"
             f"（{EXPECT_SEQUENCES} 段 / {EXPECT_INSTANCES} 实例）")
    L.append("- 聚合：`sequence_level` → `collapse_objects` → 段等权平均")
    L.append("- Δ = 臂 − A0（按序列配对）；CI = 配对差的自助法 95% 百分位区间；"
             "p = 双侧 Wilcoxon 符号秩（同相位 7 臂 Holm 校正）")
    L.append("- **两个相位分别报告、不合并**：A0 在 `fog` 与复合 `C1_fog_noise` 上"
             "难度不同（0.650 / 0.459）")
    L.append(f"- **读数按效应量优先**：|Cliff's δ| < {CLIFF_NEGLIGIBLE} 一律写"
             "「无可检出贡献」，不看 p（n = 30 时 0.02 的位移天然显著）")
    for t, dd in sorted(noop_phase.items()):
        worst = min(n for a, (n, _m) in dd.items() if a != "A0")
        L.append(f"- **臂生效检查（{t}）**：每个臂至少改变 {worst}/{EXPECT_INSTANCES} "
                 "个实例 ⇒ 开关确实到达了 pipeline；与 A0 逐位相同的臂会直接报错")
    L.append("")
    L.append("| 相位 | # | 变体 | J&F | Δ vs A0 | 95% CI | p (Holm) | Cliff's δ | 胜/负 | 编码次数 | 读数 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        if r["arm"] == "A0":
            ci = "—"
            p = "—"
            cl = "—"
            wl = "—"
            dd = "—"
        else:
            ci = f"[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]"
            p = f"{r['holm_p']:.2g}"
            cl = f"{r['cliffs_delta']:+.3f}"
            wl = f"{r['wins']}/{r['losses']}"
            dd = f"{r['delta']:+.4f}"
        L.append(f"| {r['degradation']}@{r['level']:.0f} | {r['arm']} | "
                 f"{r['arm_note']} | {r['jf']:.4f} | {dd} | {ci} | {p} | {cl} | "
                 f"{wl} | {r['n_encoder_calls']:.1f} | {r['reading']} |")
    mpath = out_dir / f"{args.prefix}.md"
    mpath.write_text("\n".join(L) + "\n", encoding="utf-8")

    print()
    print(f"{'phase':<16}{'arm':<5}{'J&F':>8}{'delta':>10}{'holm_p':>10}"
          f"{'cliff':>8}  reading")
    for r in recs:
        d = "     -" if r["arm"] == "A0" else f"{r['delta']:+.4f}"
        p = "     -" if r["arm"] == "A0" else f"{r['holm_p']:.2g}"
        c = "    -" if r["arm"] == "A0" else f"{r['cliffs_delta']:+.3f}"
        print(f"{r['degradation']:<16}{r['arm']:<5}{r['jf']:>8.4f}{d:>10}{p:>10}"
              f"{c:>8}  {r['reading']}")
    print()
    print(f"json -> {jpath}")
    print(f"csv  -> {cpath}")
    print(f"md   -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
