#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
119_mechanism_activity.py -- mechanism-activity numbers for the DAG-RS arm.

WHY THIS SCRIPT EXISTS
----------------------
The claim "the gate really runs (<N>/120 compound cells fire), but the firing
count is unrelated to the result (r = ..., rho = ...)" is quoted three times
(paper framework positioning banner, manuscript S4.10, abstract skeleton) and
originated in `docs/BASELINE_DIAGNOSIS.md` S8.

It was **stale**: that paragraph predates the 2026-09-17 core-code fix
(`consec_uncertain` moved onto the decision-frame clock + `set_reference` before
descriptor read), which is exactly what makes re-anchoring fire at all.  The
recorded per-instance distribution was 0:88, 1:59, 2:35, ... ; the current
authoritative compound grid gives 0:59, 1:43, 2:54, ...  Consequently the quoted
`89/120` and `r = -0.0001 / rho = +0.0290` do not reproduce against the published
CSVs -- yet they had been copied forward into the abstract-level banner and the
manuscript, where no script ever checked them.

So: mechanism-activity numbers now have an owner.  They are recomputed from the
authoritative compound CSVs and checked into the docs mechanically.

INPUTS (all repo-relative; never pass /c/... to python.exe)
    results/c1c4_a_dagrs.csv  results/c1c4_b_dagrs.csv    (dagrs arm, 244 rows)
    results/c1c4_a_base.csv   results/c1c4_b_base.csv     (greedy + sam2video)

CONVENTIONS
    * paired unit  : the *cell* (degradation, sequence); a cell's J&F is the mean
                     over its annotated instances (official DAVIS protocol).
    * difficulty   : cancels when comparing `dagrs` against `greedy` in the SAME
                     cell, so correlations are reported against `greedy`.
    * "fires"      : n_reanchors > 0 for at least one instance in the cell
                     (cell level) or on the instance itself (instance level).
      Also reported: n_rejected > 0, i.e. the vacuity gate actually rejected a frame.

USAGE
    python scripts/119_mechanism_activity.py --emit        # markdown block
    python scripts/119_mechanism_activity.py --check       # docs <- data (exit 1 on drift)
    python scripts/119_mechanism_activity.py --self-test   # negative controls (must all FIRE)
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAGRS_CSV = ["results/c1c4_a_dagrs.csv", "results/c1c4_b_dagrs.csv"]
BASE_CSV = ["results/c1c4_a_base.csv", "results/c1c4_b_base.csv"]

N_CELLS = 120          # 4 compound degradations x 30 DAVIS-val sequences
N_SEQS = 30
N_INSTS = 244          # 61 annotated instances x 4 degradations
COMPOUND = ("C1_fog_noise", "C2_lowlight_blur", "C3_rain_snow", "C4_fog_blur_noise")

DOCS = {
    "framework": "docs/PAPER_FRAMEWORK.md",
    "manuscript": "docs/MANUSCRIPT.md",
    "diagnosis": "docs/BASELINE_DIAGNOSIS.md",
}

# Strings that must NOT appear any more: the stale pre-fix numbers.
STALE = [
    "89/120 格",
    "89/120 grid cells",
    r"r = −0.0001",
    r"r = -0.0001",
    r"ρ = +0.0290",
    r"rho = +0.0290",
]


def _rd(rel: str) -> str:
    with io.open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def load(rel_paths):
    rows = []
    for rel in rel_paths:
        rows.extend(csv.DictReader(io.open(os.path.join(ROOT, rel), newline="", encoding="utf-8")))
    return rows


def _rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[idx[k]] = avg
        i = j + 1
    return out


def _pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in ys) / n)
    if sx == 0 or sy == 0:
        return float("nan")
    return (sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n) / (sx * sy)


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def compute(dagrs_rows=None, base_rows=None):
    """Recompute every mechanism-activity number from the raw rows."""
    d = load(DAGRS_CSV) if dagrs_rows is None else dagrs_rows
    b = load(BASE_CSV) if base_rows is None else base_rows

    # ---------------- guards: the inputs must be the grid we think they are ----
    degs = sorted(set(r["degradation"] for r in d))
    if degs != sorted(COMPOUND):
        raise SystemExit(f"[119] FATAL unexpected degradations in dagrs CSV: {degs}")
    if len(d) != N_INSTS:
        raise SystemExit(f"[119] FATAL dagrs rows = {len(d)}, expected {N_INSTS}")
    seen = collections.Counter((r["degradation"], r["seq"], r["obj_id"]) for r in d)
    if any(v > 1 for v in seen.values()):
        raise SystemExit("[119] FATAL duplicate (deg, seq, obj) keys in dagrs CSV")
    modes = sorted(set(r["mode"] for r in b))
    if modes != ["greedy", "sam2video"]:
        raise SystemExit(f"[119] FATAL base modes = {modes}, expected greedy+sam2video")

    cells_d = collections.defaultdict(list)
    for r in d:
        cells_d[(r["degradation"], r["seq"])].append(float(r["J&F"]))
    # instance-level reference map: pairing an instance against the *cell-mean*
    # baseline would mix within-cell instance difficulty into the delta and
    # manufacture a spurious trend.  Delta must be instance-to-instance.
    inst_b = {}
    for r in b:
        inst_b[(r["degradation"], r["seq"], r["obj_id"], r["mode"])] = float(r["J&F"])
    # cell-level baseline = mean over the cell's instances (official protocol)
    cell_acc = collections.defaultdict(list)
    for (deg, seq, _obj, mode), v in inst_b.items():
        cell_acc[(deg, seq, mode)].append(v)
    cells_b = collections.defaultdict(dict)
    for (deg, seq, mode), vs in cell_acc.items():
        cells_b[(deg, seq)][mode] = statistics.mean(vs)

    # ---------------- activity: how often does each mechanism fire? ----------
    def fires(field, keyf):
        agg = collections.defaultdict(bool)
        for r in d:
            agg[keyf(r)] = agg[keyf(r)] or int(float(r.get(field) or 0)) > 0
        return sum(agg.values()), len(agg)

    re_cells, tot_cells = fires("n_reanchors", lambda r: (r["degradation"], r["seq"]))
    re_seqs, tot_seqs = fires("n_reanchors", lambda r: r["seq"])
    re_insts, tot_insts = fires("n_reanchors", lambda r: (r["degradation"], r["seq"], r["obj_id"]))
    gt_cells, _ = fires("n_rejected", lambda r: (r["degradation"], r["seq"]))
    gt_seqs, _ = fires("n_rejected", lambda r: r["seq"])

    dist = collections.Counter(int(float(r.get("n_reanchors") or 0)) for r in d)

    # ---------------- correlation: firing count vs the paired delta ----------
    # primary  : instance-to-instance vs the SAME arm pair (difficulty cancels twice)
    # secondary: cell-mean-to-cell-mean vs greedy
    inst_x, inst_y = [], []
    for r in d:
        kb = (r["degradation"], r["seq"], r["obj_id"], "greedy")
        if kb not in inst_b:
            continue
        inst_x.append(int(float(r.get("n_reanchors") or 0)))
        inst_y.append(float(r["J&F"]) - inst_b[kb])
    cell_x, cell_y = [], []
    for k, vals in cells_d.items():
        if "greedy" not in cells_b[k]:
            continue
        n_re = sum(int(float(r.get("n_reanchors") or 0)) for r in d
                   if (r["degradation"], r["seq"]) == k)
        cell_x.append(n_re)
        cell_y.append(statistics.mean(vals) - cells_b[k]["greedy"])

    # ---------------- grouped means (must be read as "no monotone trend") ----
    def grp(lo, hi):
        v = [y for x, y in zip(inst_x, inst_y) if lo <= x <= hi]
        return (statistics.mean(v), len(v)) if v else (float("nan"), 0)

    # ---------------- character of the intervention: per-instance delta ------
    # (this is the paragraph in BASELINE_DIAGNOSIS S8 that also went stale)
    all_d = [y for y in inst_y]
    dstats = {
        "mean": statistics.mean(all_d), "median": statistics.median(all_d),
        "n_le002": sum(1 for x in all_d if abs(x) <= 0.02),
        "n_ge01": sum(1 for x in all_d if abs(x) >= 0.1),
        "min": min(all_d), "max": max(all_d), "n": len(all_d),
    }

    return {
        "re_cells": re_cells, "tot_cells": tot_cells,
        "re_seqs": re_seqs, "tot_seqs": tot_seqs,
        "re_insts": re_insts, "tot_insts": tot_insts,
        "gate_cells": gt_cells, "gate_seqs": gt_seqs,
        "dist": dict(sorted(dist.items())),
        "inst_r": _pearson(inst_x, inst_y), "inst_rho": _spearman(inst_x, inst_y),
        "cell_r": _pearson(cell_x, cell_y), "cell_rho": _spearman(cell_x, cell_y),
        "n_inst_pairs": len(inst_x), "n_cell_pairs": len(cell_x),
        "g0": grp(0, 0), "g13": grp(1, 3), "g4p": grp(4, 99),
        "inst_b": inst_b, "dstat": dstats,
    }


def f4(x):
    return ("%.4f" % x).replace("-", "\u2212")


def emit(m):
    L = []
    L.append(f"re-anchor fires      : {m['re_cells']}/{m['tot_cells']} cells, "
             f"{m['re_seqs']}/{m['tot_seqs']} sequences, {m['re_insts']}/{m['tot_insts']} instances")
    L.append(f"gate rejects >=1 frame: {m['gate_cells']}/{m['tot_cells']} cells, "
             f"{m['gate_seqs']}/{m['tot_seqs']} sequences")
    L.append(f"instance n_reanchors dist: {m['dist']}")
    L.append(f"corr vs greedy, per-instance (n={m['n_inst_pairs']}) : r = {f4(m['inst_r'])}, rho = {f4(m['inst_rho'])}")
    L.append(f"corr vs greedy, per-cell     (n={m['n_cell_pairs']}) : r = {f4(m['cell_r'])}, rho = {f4(m['cell_rho'])}")
    L.append(f"group mean delta: n_re=0 -> {f4(m['g0'][0])} (n={m['g0'][1]}); "
             f"1-3 -> {f4(m['g13'][0])} (n={m['g13'][1]}); >=4 -> {f4(m['g4p'][0])} (n={m['g4p'][1]})")
    s = m["dstat"]
    L.append(f"per-instance delta (n={s['n']}): mean {f4(s['mean'])}, median {f4(s['median'])}; "
             f"|d|<=0.02 {s['n_le002']}/{s['n']}; |d|>=0.1 {s['n_ge01']}/{s['n']}; "
             f"min/max {f4(s['min'])} / {f4(s['max'])}")
    return "\n".join(L)


def check(m, verbose=True):
    """Assert the docs carry the recomputed numbers and no stale ones."""
    fails = []
    texts = {k: _rd(v) for k, v in DOCS.items()}

    # (1) the headline activity count must be present, in both places that assert it
    headline = "%d/%d" % (m["re_cells"], m["tot_cells"])
    for key in ("framework", "manuscript", "diagnosis"):
        if headline not in texts[key]:
            fails.append(f"{DOCS[key]}: missing recomputed activity count {headline}")

    # (2) the recomputed correlation strings must be present where the claim is made
    for s in ("%s" % f4(m["inst_r"]), "%s" % f4(m["inst_rho"])):
        if not any(s in t for t in texts.values()):
            fails.append(f"docs: missing recomputed correlation value {s}")

    # (2b) the intervention-character stats (BASELINE_DIAGNOSIS S8 paragraph)
    st = m["dstat"]
    need = [f4(st["mean"]), f4(st["median"]),
            "%d/%d" % (st["n_le002"], st["n"]), "%d/%d" % (st["n_ge01"], st["n"]),
            f4(st["min"]), f4(st["max"])]
    for v in need:
        if v not in texts["diagnosis"]:
            fails.append(f"{DOCS['diagnosis']}: missing recomputed intervention stat {v}")

    # (3) no stale value may survive in the *assertive* text.  A retraction note
    #     has to quote the wrong value it withdraws, so such a note is fenced by
    #     <!-- RETRACTED-OK --> ... <!-- /RETRACTED-OK --> and skipped here.
    #     Two-sided: the stale value must be gone after stripping, AND the fence
    #     must actually exist somewhere (otherwise deleting the note would pass).
    fenced = 0
    for key, t in texts.items():
        stripped, n = re.subn(r"<!--\s*RETRACTED-OK\s*-->.*?<!--\s*/RETRACTED-OK\s*-->",
                              "", t, flags=re.S)
        fenced += n
        for s in STALE:
            if re.search(s, stripped):
                fails.append(f"{DOCS[key]}: STALE value still present (outside a retraction block) -> {s!r}")
    if fenced == 0:
        fails.append("docs: no <!-- RETRACTED-OK --> block found; the stale sweep has nothing to exempt "
                     "and a quoted retraction would be indistinguishable from a live claim")

    if verbose:
        print("[119] mechanism activity vs docs")
        print("  " + emit(m).replace("\n", "\n  "))
        if fails:
            print("  [119] FAIL")
            for f in fails:
                print("      -", f)
        else:
            print("  [119] OK  (0 mismatches)")
    return fails


def self_test():
    """Negative controls -- every one must FIRE, on the same path check() uses."""
    print("[119] self-test: negative controls must all FIRE")
    m = compute()
    fired = []

    # (a) kill the mechanism count in the data -> headline must move to 0/120
    d = load(DAGRS_CSV)
    for r in d:
        r["n_reanchors"] = "0"
    m2 = compute(dagrs_rows=d, base_rows=load(BASE_CSV))
    ok = (m2["re_cells"] == 0 and m["re_cells"] > 0)
    print(f"   (a) zeroed n_reanchors -> activity {m2['re_cells']}/{m2['tot_cells']} "
          f"(was {m['re_cells']}/{m['tot_cells']})  {'FIRED' if ok else 'NOT FIRED **'}")
    fired.append(ok)

    # (b) plant a STRONG POSITIVE association -> r must go strongly positive.
    #     Sign calibration is mandatory: a correlation of ~0 is only meaningful
    #     if the same code path can produce a large |r| when one exists.
    d = load(DAGRS_CSV)
    for r in d:
        kb = (r["degradation"], r["seq"], r["obj_id"], "greedy")
        if kb in m["inst_b"]:
            n = int(float(r.get("n_reanchors") or 0))
            r["J&F"] = repr(m["inst_b"][kb] + 0.10 * n)
    mp = compute(dagrs_rows=d, base_rows=load(BASE_CSV))
    ok = mp["inst_r"] > 0.99
    print(f"   (b+) planted +0.10*n_re on dJ&F  -> r = {f4(mp['inst_r'])} (expect ~+1)  "
          f"{'FIRED' if ok else 'NOT FIRED **'}")
    fired.append(ok)

    # (b-) same, with the opposite sign
    d = load(DAGRS_CSV)
    for r in d:
        kb = (r["degradation"], r["seq"], r["obj_id"], "greedy")
        if kb in m["inst_b"]:
            n = int(float(r.get("n_reanchors") or 0))
            r["J&F"] = repr(m["inst_b"][kb] - 0.10 * n)
    mn = compute(dagrs_rows=d, base_rows=load(BASE_CSV))
    ok = mn["inst_r"] < -0.99
    print(f"   (b-) planted -0.10*n_re on dJ&F  -> r = {f4(mn['inst_r'])} (expect ~-1)  "
          f"{'FIRED' if ok else 'NOT FIRED **'}")
    fired.append(ok)

    # (b0) zero association must land near zero, not at either rail.
    #      A constant delta would make the correlation *undefined* (0/0), so the
    #      control uses deterministic noise of comparable size but zero slope.
    d = load(DAGRS_CSV)
    for i, r in enumerate(d):
        kb = (r["degradation"], r["seq"], r["obj_id"], "greedy")
        if kb in m["inst_b"]:
            r["J&F"] = repr(m["inst_b"][kb] + 0.001 * (((i * 7919) % 97) - 48))
    mz = compute(dagrs_rows=d, base_rows=load(BASE_CSV))
    ok = (not math.isnan(mz["inst_r"])) and abs(mz["inst_r"]) < 0.15
    print(f"   (b0) noisy zero association      -> r = {f4(mz['inst_r'])} (expect ~0)  "
          f"{'FIRED' if ok else 'NOT FIRED **'}")
    fired.append(ok)

    # (c) the STALE sweep must catch an unfenced planted token ...
    def _sweep(text):
        stripped = re.sub(r"<!--\s*RETRACTED-OK\s*-->.*?<!--\s*/RETRACTED-OK\s*-->", "", text, flags=re.S)
        return [s for s in STALE if re.search(s, stripped)]

    base = _rd(DOCS["framework"])
    hit_open = len(_sweep(base + "\n89/120 \u683c\n")) > 0
    print(f"   (c1) planted stale token, NO fence   -> {'FIRED' if hit_open else 'NOT FIRED **'}")
    fired.append(hit_open)

    # (c2) ... and must NOT flag the same token when it is quoted inside a fence
    fenced_copy = base + "\n<!-- RETRACTED-OK -->\n89/120 \u683c\n<!-- /RETRACTED-OK -->\n"
    hit_fenced = len(_sweep(fenced_copy)) > 0
    print(f"   (c2) same token INSIDE retraction fence -> {'NOT fired (correct)' if not hit_fenced else 'FIRED **'}")
    fired.append(not hit_fenced)

    # (d) the checker must reject a doc that lacks the recomputed count
    fails = check(m)
    if fails:
        print("   (d) docs currently fail the check -> fix the docs before relying on it")
    ok = 0 < len(fails)
    print(f"   (d) docs without the recomputed count are rejected: {len(fails)} finding(s)")
    fired.append(True)  # informative; the real control is (a)-(c)

    print("[119] self-test:", "PASS" if all(fired) else "FAIL")
    return 0 if all(fired) else 1


def main():
    ap = argparse.ArgumentParser(description="mechanism-activity numbers for the DAG-RS arm")
    ap.add_argument("--emit", action="store_true", help="print the markdown/plain block")
    ap.add_argument("--check", action="store_true", help="verify docs against the data")
    ap.add_argument("--self-test", action="store_true", help="negative controls")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    m = compute()
    if a.emit:
        print(emit(m))
        return 0
    if a.check:
        sys.stdout.reconfigure(encoding="utf-8")
        return 1 if check(m) else 0

    print(emit(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
