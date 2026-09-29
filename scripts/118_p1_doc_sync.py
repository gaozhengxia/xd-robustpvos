"""118 - keep `docs/PAPER_FRAMEWORK.md` consistent with the PRIMARY (P1) protocol.

Why this exists
---------------
`115_frame_denominator.py` produces the mirror tree `results/p1/` (P1 = denominator
restricted to the frames where the GT object is present, arm-independent).  Section
5.1b of the framework then fixes P1 as THE manuscript protocol: *"表内数字一律为
P1；P0 只在 §5.1b / §5.5b / §5.6 的敏感性一句里出现"*.

Before this script the framework's main tables still carried the P0 sweep values,
so the document was internally inconsistent (Intro/§5.2 in P0, §5.6 in P1).  A
reviewer reads those two sections against each other.

How it stays honest
-------------------
Nothing here re-implements the aggregation.  Every expected value is produced by
the SAME production scripts that produced the published P0 tables, just pointed at
the P1 mirror:

    python scripts/05_analysis.py --study tables --inputs <P1 mirror>  -> table9_{main,pivot}.csv, table9c_retention.csv
    python scripts/108_compound_vs_member.py --compound-files/--single-files <P1 mirror>
                                                                      -> c1c4_vs_members_p1.json

so `--check` can never "drift" from the sweep, only disagree with the document.

Modes
-----
    --check      parse every anchored table in the doc, compare cell by cell, exit 1 on any mismatch
    --emit       print the P1 replacement blocks (ready to paste, same styling as the doc)
    --self-test  perturb one cell in a temp copy of the doc and assert --check FIRES on it

Doc anchors
-----------
Each checked table is preceded by `<!-- P1-SYNC: <name> -->` in the markdown.  The
anchor makes the mapping table<->spec explicit instead of relying on line numbers.

Usage
-----
    python scripts/118_p1_doc_sync.py --check
    python scripts/118_p1_doc_sync.py --emit
    python scripts/118_p1_doc_sync.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
P1 = ROOT / "results" / "p1"
DERIVED = P1 / "_derived"
DOC = ROOT / "docs" / "PAPER_FRAMEWORK.md"

#: The framework is an internal working document and is deliberately NOT
#: distributed.  `_release_gate` turns its absence in a clone into a reported
#: SKIP instead of a crash (which would read as "the repository is broken").
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _release_gate import gate as release_gate                       # noqa: E402

#: 4 decimals, U+2212 minus (the document's typography).
MINUS = "\u2212"


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def f4(x: float) -> str:
    return f"{x:.4f}".replace("-", MINUS)


def sgn4(x: float) -> str:
    """Signed 4-dp, the way the doc writes family gaps (always shows +/-)."""
    return ("+" if x >= 0 else MINUS) + f"{abs(x):.4f}"


def ci4(lo: float, hi: float) -> str:
    return f"[{('+' if lo >= 0 else MINUS)}{abs(lo):.4f}, {('+' if hi >= 0 else MINUS)}{abs(hi):.4f}]"


def num(cell: str) -> float:
    """Parse a doc cell back to a float (strips styling, unicode minus)."""
    t = cell.replace(MINUS, "-").replace("\u00d7", "x")
    t = t.replace("*", "").replace("`", "").strip()
    t = t.replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+", t)
    if not m:
        raise ValueError(f"no number in cell {cell!r}")
    return float(m.group(0))


def plain(cell: str) -> str:
    return cell.replace("*", "").replace("`", "").replace(" ", "").strip()


# --------------------------------------------------------------------------- #
# step 1 - regenerate the P1 artefacts with the production scripts
# --------------------------------------------------------------------------- #
TABLES_INPUTS = [
    "results/p1/results/cl4_full_v2.csv",
    "results/p1/results/singles_greedy.csv",
    "results/p1/results/singles_dagrs.csv",
    "results/p1/results/singles_sam2video.csv",
    "results/p1/results/c1c4_a_base.csv",
    "results/p1/results/c1c4_a_dagrs.csv",
    "results/p1/results/c1c4_b_base.csv",
    "results/p1/results/c1c4_b_dagrs.csv",
]
COMPOUND_INPUTS = [
    "results/p1/results/c1c4_a_dagrs.csv",
    "results/p1/results/c1c4_b_dagrs.csv",
    "results/p1/results/c1c4_a_base.csv",
    "results/p1/results/c1c4_b_base.csv",
]
SINGLE_INPUTS = [
    "results/p1/results/singles_dagrs.csv",
    "results/p1/results/singles_greedy.csv",
    "results/p1/results/singles_sam2video.csv",
    "results/p1/results/cl4_full_v2.csv",
]

#: the nine single degradations that make the "9-class" column set (clean included)
NINE = ["clean", "fog", "dust", "underwater", "lowlight", "motion_blur", "rain", "snow",
        "sensor_noise"]
#: the four CL4 degradations (the 120-pair headline set)
FOUR = ["clean", "dust", "fog", "underwater"]
#: the eight degraded singles (retention table excludes clean, whose retention is 1 by construction)
EIGHT = [d for d in NINE if d != "clean"]
COMPOUNDS = ["C1_fog_noise", "C2_lowlight_blur", "C3_rain_snow", "C4_fog_blur_noise"]
COMPOUND_MEMBERS = {
    "C1_fog_noise": ["fog", "sensor_noise"],
    "C2_lowlight_blur": ["lowlight", "motion_blur"],
    "C3_rain_snow": ["rain", "snow"],
    "C4_fog_blur_noise": ["fog", "motion_blur", "sensor_noise"],
}
#: how the document writes each compound's member list
COMPOUND_LABEL = {
    "C1_fog_noise": "fog+noise",
    "C2_lowlight_blur": "lowlight+blur",
    "C3_rain_snow": "rain+snow",
    "C4_fog_blur_noise": "fog+blur+noise",
}
#: Chinese row labels used by TABLE-9b / TABLE-9c
CN = {
    "干净": "clean", "雾": "fog", "沙尘": "dust", "水下/浊度": "underwater",
    "低光": "lowlight", "运动模糊": "motion_blur", "雨": "rain", "雪": "snow",
    "传感器噪声": "sensor_noise",
}


def run(cmd: Sequence[str], label: str, capture: bool = False) -> str:
    p = subprocess.run(list(cmd), cwd=str(ROOT), text=True,
                       capture_output=capture)
    if p.returncode != 0:
        raise SystemExit(f"[118] {label} failed with exit {p.returncode}")
    return p.stdout or ""


def build_artefacts(verbose: bool = True) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("[118] regenerating P1 artefacts with the production scripts ...")
    run([PY, "scripts/05_analysis.py", "--study", "tables",
         "--inputs", *TABLES_INPUTS, "--out-dir", str(DERIVED)],
        "05_analysis --study tables", capture=True)
    run([PY, "scripts/108_compound_vs_member.py",
         "--compound-files", *COMPOUND_INPUTS,
         "--single-files", *SINGLE_INPUTS,
         "--out", str(P1 / "c1c4_vs_members_p1.json"),
         "--md", str(P1 / "c1c4_vs_members_p1.md")],
        "108_compound_vs_member", capture=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    import csv
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# step 2 - expected values, keyed exactly like the doc's rows
# --------------------------------------------------------------------------- #
def expected() -> Dict[str, Dict[str, Dict[str, str]]]:
    """{table_name: {row_key: {column_name: rendered_cell}}}"""
    piv = {r["degradation"]: r for r in read_csv(DERIVED / "table9_pivot.csv")}
    # table9c_retention.csv is LONG format (degradation, level, mode, J&F, retention)
    ret: Dict[str, Dict[str, str]] = {}
    for r in read_csv(DERIVED / "table9c_retention.csv"):
        ret.setdefault(r["degradation"], {})[r["mode"]] = r["retention"]
    comp = json.loads((P1 / "c1c4_vs_members_p1.json").read_text(encoding="utf-8"))
    gaps = comp["gaps"]

    def arm(deg: str, a: str) -> float:
        return float(piv[deg][a])

    def fam(deg: str) -> float:
        return float(gaps[deg]["family_gap"])

    def grp(deg: str) -> float:
        return float(gaps[deg]["gated_gap"])

    out: Dict[str, Dict[str, Dict[str, str]]] = {}

    # -- TABLE-9 (a0 four CL4 degradations + POOLED), columns greedy/dagrs/video/delta
    t9 = {}
    for d in FOUR:
        t9[d] = {"greedy": f4(arm(d, "greedy")), "dagrs": f4(arm(d, "dagrs")),
                 "video": f4(arm(d, "sam2video")),
                 "delta": f4(arm(d, "dagrs") - arm(d, "sam2video"))}
    t9["POOLED"] = {
        "greedy": f4(sum(arm(d, "greedy") for d in FOUR) / 4),
        "dagrs": f4(sum(arm(d, "dagrs") for d in FOUR) / 4),
        "video": f4(sum(arm(d, "sam2video") for d in FOUR) / 4),
        "delta": f4(sum(arm(d, "dagrs") - arm(d, "sam2video") for d in FOUR) / 4),
    }
    out["table9"] = t9

    # -- the full 9-single + 2-POOLED table, columns greedy/dagrs/video/family-gap
    t9a = {}
    for d in NINE:
        t9a[d] = {"greedy": f4(arm(d, "greedy")), "dagrs": f4(arm(d, "dagrs")),
                  "video": f4(arm(d, "sam2video")), "gap": sgn4(arm(d, "sam2video") - arm(d, "greedy"))}
    for key, ds in (("POOLED4", FOUR), ("POOLED9", NINE)):
        t9a[key] = {
            "greedy": f4(sum(arm(d, "greedy") for d in ds) / len(ds)),
            "dagrs": f4(sum(arm(d, "dagrs") for d in ds) / len(ds)),
            "video": f4(sum(arm(d, "sam2video") for d in ds) / len(ds)),
            "gap": sgn4(sum(arm(d, "sam2video") - arm(d, "greedy") for d in ds) / len(ds)),
        }
    out["table9all"] = t9a

    # -- TABLE-9b (Chinese labels), columns video/greedy/dagrs/delta
    t9b = {}
    for cn, d in CN.items():
        t9b[cn] = {"video": f4(arm(d, "sam2video")), "greedy": f4(arm(d, "greedy")),
                   "dagrs": f4(arm(d, "dagrs")), "delta": f4(arm(d, "dagrs") - arm(d, "sam2video"))}
    t9b["9类等权"] = {
        "video": f4(sum(arm(d, "sam2video") for d in NINE) / 9),
        "greedy": f4(sum(arm(d, "greedy") for d in NINE) / 9),
        "dagrs": f4(sum(arm(d, "dagrs") for d in NINE) / 9),
        # Δ(dagrs - video) on the equal-weight means, NOT the mean of per-cell deltas
        "delta": f4((sum(arm(d, "dagrs") for d in NINE) - sum(arm(d, "sam2video") for d in NINE)) / 9),
    }
    out["table9b"] = t9b

    # -- TABLE-9c retention, columns video/greedy/dagrs
    t9c = {}
    for cn, d in CN.items():
        r = ret[d]
        t9c[cn] = {"video": f4(float(r["sam2video"])), "greedy": f4(float(r["greedy"])),
                   "dagrs": f4(float(r["dagrs"]))}
    for label, ds in (("C1", "C1_fog_noise"), ("C2", "C2_lowlight_blur"),
                      ("C3", "C3_rain_snow"), ("C4", "C4_fog_blur_noise")):
        r = ret[ds]
        t9c[label] = {"video": f4(float(r["sam2video"])), "greedy": f4(float(r["greedy"])),
                      "dagrs": f4(float(r["dagrs"]))}
    def _rng(mode: str) -> str:
        vals = [float(ret[d][mode]) for d in EIGHT]
        return f"{f4(min(vals))}\u2013{f4(max(vals))}"

    t9c["minmax8"] = {"video": _rng("sam2video"), "greedy": _rng("greedy"),
                      "dagrs": _rng("dagrs")}
    out["table9c"] = t9c

    # -- TABLE-10 (single 9 + means + compound 4 + mean)
    t10 = {}
    for d in NINE:
        t10[d] = {"kind": "单档", "greedy": f4(arm(d, "greedy")), "dagrs": f4(arm(d, "dagrs")),
                  "video": f4(arm(d, "sam2video")), "gap": sgn4(fam(d))}
    for key, ds in (("MEAN9", NINE), ("MEAN4", COMPOUNDS)):
        t10[key] = {"kind": "\u2014",
                    "greedy": f4(sum(arm(d, "greedy") for d in ds) / len(ds)),
                    "dagrs": f4(sum(arm(d, "dagrs") for d in ds) / len(ds)),
                    "video": f4(sum(arm(d, "sam2video") for d in ds) / len(ds)),
                    "gap": sgn4(sum(fam(d) for d in ds) / len(ds))}
    for c in COMPOUNDS:
        t10[c] = {"kind": "复合", "greedy": f4(arm(c, "greedy")), "dagrs": f4(arm(c, "dagrs")),
                  "video": f4(arm(c, "sam2video")), "gap": sgn4(fam(c))}
    out["table10"] = t10

    # -- §5.3.1 layer 2: per (compound, member) gap shift
    # the tag rule is copied from 108 (mean_delta vs WIDEN_TOL, one-sided in the
    # widening direction): WIDENS if mean > +tol, NARROWS if mean < -tol, else FLAT
    tol = float(comp.get("widening_tolerance", 0.02))
    shifts = {}
    for k, v in comp["gap_shift_paired"].items():
        c, m = k.split("|vs|")
        md = float(v["mean_delta"])
        tag = "WIDENS" if md > tol else ("NARROWS" if md < -tol else "FLAT")
        shifts[f"{c}|{m}"] = {
            "delta": f4(md),
            "ci": ci4(v["delta_ci_lo"], v["delta_ci_hi"]),
            "p": (f"{v['wilcoxon_p']:.3f}" if v["wilcoxon_p"] >= 1e-3 else f"{v['wilcoxon_p']:.1e}"),
            "cliff": (("+" if v["cliffs_delta"] >= 0 else MINUS) + f"{abs(v['cliffs_delta']):.3f}"),
            "winlose": f"{v['n_widen']}/{v['n_narrow']}",
            "verdict": tag,
        }
    out["widen_pairs"] = shifts

    # -- §5.3.1 layer 3: pooled across compounds
    pf = comp["pooled_family_gap"]
    out["widen_pool"] = {"family": {
        "single": f4(pf["mean_gap_single"]), "compound": f4(pf["mean_gap_compound"]),
        "shift": f4(pf["mean_delta"]), "ci": ci4(pf["delta_ci_lo"], pf["delta_ci_hi"]),
        "p": f"{pf['wilcoxon_p']:.3f}",
        "cliff": ("+" if pf["cliffs_delta"] >= 0 else MINUS) + f"{abs(pf['cliffs_delta']):.3f}",
        "winlose": f"{pf['n_widen']} / {pf['n_narrow']}",
    }}
    return out


# --------------------------------------------------------------------------- #
# step 3 - doc parsing
# --------------------------------------------------------------------------- #
ANCHOR = re.compile(r"<!--\s*P1-SYNC:\s*([A-Za-z0-9_]+)\s*-->")

#: table -> ordered (match_token, expected row key, [cell indices that carry numbers])
SPEC: Dict[str, Tuple[List[Tuple[str, str, List[int]]], Dict[str, List[str]]]] = {
    # (rows), {column role: [cell indices in order]}
    "table9": ([
        ("clean", "clean", [1, 2, 3, 4]),
        ("dust", "dust", [1, 2, 3, 4]),
        ("fog", "fog", [1, 2, 3, 4]),
        ("underwater", "underwater", [1, 2, 3, 4]),
        ("POOLED", "POOLED", [1, 2, 3, 4]),
    ], {"cols": ["greedy", "dagrs", "video", "delta"]}),
    "table9all": ([
        ("clean", "clean", [1, 2, 3, 4]),
        ("dust", "dust", [1, 2, 3, 4]),
        ("fog", "fog", [1, 2, 3, 4]),
        ("underwater", "underwater", [1, 2, 3, 4]),
        ("lowlight", "lowlight", [1, 2, 3, 4]),
        ("motion_blur", "motion_blur", [1, 2, 3, 4]),
        ("rain", "rain", [1, 2, 3, 4]),
        ("snow", "snow", [1, 2, 3, 4]),
        ("sensor_noise", "sensor_noise", [1, 2, 3, 4]),
        ("POOLED（CL4 4 档", "POOLED4", [1, 2, 3, 4]),
        ("POOLED（9 单档全表）", "POOLED9", [1, 2, 3, 4]),
    ], {"cols": ["greedy", "dagrs", "video", "gap"]}),
    "table9b": ([
        ("干净", "干净", [2, 3, 4, 5]),
        ("雾", "雾", [2, 3, 4, 5]),
        ("沙尘", "沙尘", [2, 3, 4, 5]),
        ("水下/浊度", "水下/浊度", [2, 3, 4, 5]),
        ("低光", "低光", [2, 3, 4, 5]),
        ("运动模糊", "运动模糊", [2, 3, 4, 5]),
        ("雨", "雨", [2, 3, 4, 5]),
        ("雪", "雪", [2, 3, 4, 5]),
        ("传感器噪声", "传感器噪声", [2, 3, 4, 5]),
        ("9 类等权", "9类等权", [2, 3, 4, 5]),
    ], {"cols": ["video", "greedy", "dagrs", "delta"]}),
    "table9c": ([
        ("雾", "雾", [2, 3, 4]),
        ("沙尘", "沙尘", [2, 3, 4]),
        ("水下/浊度", "水下/浊度", [2, 3, 4]),
        ("低光", "低光", [2, 3, 4]),
        ("运动模糊", "运动模糊", [2, 3, 4]),
        ("雨", "雨", [2, 3, 4]),
        ("雪", "雪", [2, 3, 4]),
        ("传感器噪声", "传感器噪声", [2, 3, 4]),
        ("**C1**", "C1", [2, 3, 4]),
        ("**C2**", "C2", [2, 3, 4]),
        ("**C3**", "C3", [2, 3, 4]),
        ("**C4**", "C4", [2, 3, 4]),
        ("单档 8 类 min", "minmax8", [2, 3, 4]),
    ], {"cols": ["video", "greedy", "dagrs"]}),
    "table10": ([
        ("clean", "clean", [2, 3, 4, 5]),
        ("fog", "fog", [2, 3, 4, 5]),
        ("dust", "dust", [2, 3, 4, 5]),
        ("underwater", "underwater", [2, 3, 4, 5]),
        ("lowlight", "lowlight", [2, 3, 4, 5]),
        ("motion_blur", "motion_blur", [2, 3, 4, 5]),
        ("rain", "rain", [2, 3, 4, 5]),
        ("snow", "snow", [2, 3, 4, 5]),
        ("sensor_noise", "sensor_noise", [2, 3, 4, 5]),
        ("单档均值", "MEAN9", [2, 3, 4, 5]),
        ("**C1**", "C1_fog_noise", [2, 3, 4, 5]),
        ("**C2**", "C2_lowlight_blur", [2, 3, 4, 5]),
        ("**C3**", "C3_rain_snow", [2, 3, 4, 5]),
        ("**C4**", "C4_fog_blur_noise", [2, 3, 4, 5]),
        ("复合均值", "MEAN4", [2, 3, 4, 5]),
    ], {"cols": ["greedy", "dagrs", "video", "gap"]}),
    "widen_pairs": ([
        ("C1|fog", "C1_fog_noise|fog", [2, 3, 4, 5, 6, 7]),
        ("C1|sensor_noise", "C1_fog_noise|sensor_noise", [2, 3, 4, 5, 6, 7]),
        ("C2|lowlight", "C2_lowlight_blur|lowlight", [2, 3, 4, 5, 6, 7]),
        ("C2|motion_blur", "C2_lowlight_blur|motion_blur", [2, 3, 4, 5, 6, 7]),
        ("C3|rain", "C3_rain_snow|rain", [2, 3, 4, 5, 6, 7]),
        ("C3|snow", "C3_rain_snow|snow", [2, 3, 4, 5, 6, 7]),
        ("C4|fog", "C4_fog_blur_noise|fog", [2, 3, 4, 5, 6, 7]),
        ("C4|motion_blur", "C4_fog_blur_noise|motion_blur", [2, 3, 4, 5, 6, 7]),
        ("C4|sensor_noise", "C4_fog_blur_noise|sensor_noise", [2, 3, 4, 5, 6, 7]),
    ], {"cols": ["delta", "ci", "p", "cliff", "winlose", "verdict"]}),
    "widen_pool": ([("家族缺口", "family", [1, 2, 3, 4, 5, 6, 7])],
                   {"cols": ["single", "compound", "shift", "ci", "p", "cliff", "winlose"]}),
}


def doc_tables(text: str) -> Dict[str, List[List[str]]]:
    """{anchor_name: [row cells, ...]} for every `<!-- P1-SYNC: x -->` anchored table.

    A table runs from its anchor to the first line that is not a `|` row, so an
    anchor can never silently swallow a neighbouring table.
    """
    lines = text.splitlines()
    out: Dict[str, List[List[str]]] = {}
    name = None
    for ln in lines:
        m = ANCHOR.search(ln)
        if m:
            name = m.group(1)
            out.setdefault(name, [])
            continue
        if name is None:
            continue
        if ln.lstrip().startswith("|"):
            out[name].append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif out[name]:
            name = None   # table ended
    return {k: v for k, v in out.items() if v}


def row_match(row: List[str], token: str) -> bool:
    """`a|b` means cells[0] startswith a AND cells[1] startswith b (disambiguates C4|fog vs C4|snow)."""
    for i, part in enumerate(token.split("|")):
        if i >= len(row) or not plain(row[i]).startswith(plain(part)):
            return False
    return True


def check(text: str, exp: Dict[str, Dict[str, Dict[str, str]]], verbose: bool = True) -> List[str]:
    problems: List[str] = []
    found = doc_tables(text)
    for tname, (rows, meta) in SPEC.items():
        if tname not in found:
            problems.append(f"{tname}: anchor `<!-- P1-SYNC: {tname} -->` not found in the doc")
            continue
        table = [r for r in found[tname] if not set("".join(r)) <= set("-\u2014 :")]
        cols = meta["cols"]
        for token, key, idxs in rows:
            hit = None
            for r in table:
                if row_match(r, token):
                    hit = r
                    break
            if hit is None:
                problems.append(f"{tname}: no row starting with {token!r}")
                continue
            want_row = exp[tname].get(key)
            if want_row is None:
                problems.append(f"{tname}: no expected values for key {key!r}")
                continue
            for ci_, role in zip(idxs, cols):
                if ci_ >= len(hit):
                    problems.append(f"{tname}/{token}: row has {len(hit)} cells, need index {ci_}")
                    continue
                got_s = hit[ci_]
                want_s = want_row[role]
                if "\u2013" in want_s:                       # range cell: compare both endpoints
                    gv = [float(x) for x in re.findall(r"\d+\.\d+", got_s.replace(MINUS, "-"))]
                    wv = [float(x) for x in re.findall(r"\d+\.\d+", want_s)]
                    if len(gv) != len(wv) or any(abs(a - b) > 5e-5 for a, b in zip(gv, wv)):
                        problems.append(f"{tname}/{token}/{role}: doc says {got_s!r} want {want_s!r}")
                    continue
                try:
                    got = num(got_s)
                except ValueError:
                    if plain(got_s) != plain(want_s):
                        problems.append(f"{tname}/{token}/{role}: got {got_s!r} want {want_s!r}")
                    continue
                want = num(want_s)
                if abs(got - want) > 5e-5:
                    problems.append(
                        f"{tname}/{token}/{role}: doc says {got_s} ({got:.4f}), P1 data says {want_s} ({want:.4f})")
    if verbose:
        n_rows = sum(len(v) for v in found.values())
        print(f"[118] anchored tables found: {len(found)}  rows parsed: {n_rows}")
        print(f"[118] mismatches: {len(problems)}")
        for p in problems:
            print("   -", p)
    return problems


# --------------------------------------------------------------------------- #
# step 4 - emit replacement blocks
# --------------------------------------------------------------------------- #
def emit(exp: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    t9 = exp["table9"]
    print("\n---------- TABLE-9 (5.2) ----------")
    for k in ["clean", "dust", "fog", "underwater", "POOLED"]:
        r = t9[k]
        lab = "**POOLED**" if k == "POOLED" else k
        v = f"**{r['video']}**" if k != "POOLED" else f"**{r['video']}**"
        d = f"**{r['delta']}**" if k == "POOLED" else r["delta"]
        print(f"| {lab} | {r['greedy']} | {r['dagrs']} | {v} | {d} |")

    print("\n---------- 9-degradation table (5.2) ----------")
    for k in NINE:
        r = exp["table9all"][k]
        print(f"| {k} | {r['greedy']} | {r['dagrs']} | **{r['video']}** | {r['gap']} |")
    for k, lab in (("POOLED4", "**POOLED（CL4 4 档，120 配对格）**"),
                   ("POOLED9", "**POOLED（9 单档全表）**")):
        r = exp["table9all"][k]
        print(f"| {lab} | {r['greedy']} | {r['dagrs']} | **{r['video']}** | **{r['gap']}** |")

    print("\n---------- TABLE-9b (9 types) ----------")
    for cn in list(CN) + ["9类等权"]:
        r = exp["table9b"][cn]
        lab = "**9 类等权**" if cn == "9类等权" else cn
        v = f"**{r['video']}**" if cn == "9类等权" else r["video"]
        print(f"| {lab} | 3.0 | {v} | {r['greedy']} | {r['dagrs']} | {r['delta']} |")

    print("\n---------- TABLE-9c (retention) ----------")
    order = ["雾", "沙尘", "水下/浊度", "低光", "运动模糊", "雨", "雪", "传感器噪声",
             "C1", "C2", "C3", "C4"]
    for cn in order:
        r = exp["table9c"][cn]
        lab = f"**{cn}**" if cn.startswith("C") else cn
        extra = {"C1": " fog+noise", "C2": " lowlight+blur", "C3": " rain+snow",
                 "C4": " fog+blur+noise"}.get(cn, "")
        print(f"| {lab}{extra} | 3.0 | {r['video']} | {r['greedy']} | {r['dagrs']} |")
    r = exp["table9c"]["minmax8"]
    print(f"| **单档 8 类 min\u2013max** | \u2014 | **{r['video']}** | **{r['greedy']}** | **{r['dagrs']}** |")

    print("\n---------- TABLE-10 (5.3) ----------")
    order = [(d, "单档") for d in NINE] + [("MEAN9", None)] + \
            [(c, "复合") for c in COMPOUNDS] + [("MEAN4", None)]
    for k, kind in order:
        r = exp["table10"][k]
        if k == "MEAN9":
            print(f"| *单档均值（9 档等权）* | \u2014 | *{r['greedy']}* | *{r['dagrs']}* | ***{r['video']}*** | ***{r['gap']}*** |")
        elif k == "MEAN4":
            print(f"| *复合均值（4 档等权）* | \u2014 | *{r['greedy']}* | *{r['dagrs']}* | ***{r['video']}*** | ***{r['gap']}*** |")
        elif kind == "复合":
            print(f"| **{k[:2]}** {COMPOUND_LABEL[k]} | 复合 | {r['greedy']} | {r['dagrs']} | **{r['video']}** | {r['gap']} |")
        else:
            print(f"| {k} | 单档 | {r['greedy']} | {r['dagrs']} | **{r['video']}** | {r['gap']} |")

    print("\n---------- 5.3.1 layer-2 (widen_pairs) ----------")
    for k, v in exp["widen_pairs"].items():
        c, m = k.split("|")
        print(f"| {c[:2]} | {m} | {v['delta']} | {v['ci']} | {v['p']} | {v['cliff']} | {v['winlose']} | {v['verdict']} |")
    print("\n---------- 5.3.1 layer-3 (widen_pool) ----------")
    r = exp["widen_pool"]["family"]
    print(f"| 家族缺口 | {r['single']} | {r['compound']} | **{r['shift']}** | {r['ci']} | **{r['p']}** | {r['cliff']} | **{r['winlose']}** |")


# --------------------------------------------------------------------------- #
def self_test(exp: Dict[str, Dict[str, Dict[str, str]]]) -> int:
    """Perturb one cell and assert the checker FIRES (a checker that cannot fail is not a check)."""
    text = DOC.read_text(encoding="utf-8")
    base = check(text, exp, verbose=False)
    if base:
        print(f"[118][self-test] WARNING: the live document already has {len(base)} mismatch(es);")
        print("                 the perturbation control is run on top of that baseline.")
    target = f"| clean | {exp['table9']['clean']['greedy']} |"
    if target not in text:
        print("[118][self-test] FAIL: could not locate the TABLE-9 clean row to perturb")
        return 1
    bad = text.replace(target, "| clean | 0.0001 |", 1)
    got = check(bad, exp, verbose=False)
    fired = any("table9/clean/greedy" in p for p in got)
    print(f"[118][self-test] perturbed TABLE-9/clean/greedy -> mismatches {len(base)} -> {len(got)}")
    print(f"[118][self-test] {'PASS' if fired else 'FAIL'}: the checker {'detected' if fired else 'MISSED'} the injected error")
    return 0 if fired else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-regen", action="store_true",
                    help="reuse the artefacts instead of re-running 05_analysis/108")
    args = ap.parse_args()
    if not (args.check or args.emit or args.self_test):
        args.check = True

    status, msg = release_gate("118", ROOT, [DOC.relative_to(ROOT).as_posix()],
                               "the framework-mirror sync check")
    if status != "ok":
        print(msg)
        return 1 if status == "fail" else 0

    if not args.no_regen:
        build_artefacts()
    exp = expected()

    rc = 0
    if args.emit:
        emit(exp)
    if args.check:
        rc = 1 if check(DOC.read_text(encoding="utf-8"), exp) else 0
    if args.self_test:
        rc = max(rc, self_test(exp))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
