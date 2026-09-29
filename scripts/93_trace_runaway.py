"""93 - trace where a sequence runs away, and how far.

Project-internal diagnostic (NOT a paper artefact, but it produces the numbers
for the paper's qualitative failure analysis in section 5.8 / [[FIG-4]]).

Reads the per-frame rows written by `scripts/92_ab_mask_prompt.py` and reports,
for every (sequence, degradation, prompt mode) cell:

  * `t(>3)`   -- first frame where `pred_px / gt_px` exceeds 3, i.e. when the
                 predicted region has visibly left the object. `-1` = never.
  * `r5/r10/r20/r_last` -- the same ratio at fixed frames, so the *shape* of the
                 trajectory is visible without importing 1500 rows.
  * `peak`    -- the worst ratio reached.

The second half prints the full frame-by-frame trace of the worst offender, with
both prompt modes side by side, which is what shows the mechanism: a hard +-8
wall can bounce back after a spike, while a carried probability field, once it
starts leaking mass, keeps leaking (it has no boundary to stop it).

    python scripts/93_trace_runaway.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: ratio above which the mask has visibly left the object
RUNAWAY = 3.0

#: frame-by-frame trace to print (sequence, degradation)
TRACE = ("lab-coat", "clean")


def load(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize(rows, tag):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["seq"], r["degradation"], r["mask_prompt"])].append(r)
    print(f"\n===== {tag} =====")
    print(f"{'seq':<16}{'deg':<7}{'prompt':<6}{'n':>4}"
          f"{'t(>3)':>7}{'r5':>9}{'r10':>9}{'r20':>10}{'r_last':>10}{'peak':>10}")
    for (seq, deg, mp), rs in sorted(cells.items()):
        rs.sort(key=lambda r: int(r["frame"]))
        ratio = [float(r["area_ratio"]) for r in rs]
        first = next((i for i, v in enumerate(ratio) if v > RUNAWAY), -1)

        def at(i):
            return f"{ratio[i]:.2f}" if i < len(ratio) else "-"

        print(f"{seq:<16}{deg:<7}{mp:<6}{len(rs):>4}{first:>7}"
              f"{at(5):>9}{at(10):>9}{at(20):>10}{at(len(ratio) - 1):>10}"
              f"{max(ratio):>10.2f}")


def main() -> int:
    for name in ("ab_mask_prompt.csv", "ab_mask_prompt_dagrs.csv"):
        p = ROOT / "results" / name
        if p.exists():
            summarize(load(p), name)
        else:
            print(f"\n[{name} not found -- run scripts/92_ab_mask_prompt.py first]")

    p = ROOT / "results" / "ab_mask_prompt.csv"
    if p.exists():
        rows = [r for r in load(p)
                if r["seq"] == TRACE[0] and r["degradation"] == TRACE[1]]
        if rows:
            print(f"\n===== {TRACE[0]} / {TRACE[1]}, frame by frame =====")
            print(f"{'t':>4}{'hard_px':>10}{'gt_px':>9}{'hard_r':>9}{'hard_JF':>9}"
                  f"{'soft_px':>10}{'soft_r':>9}{'soft_JF':>9}")
            h = {int(r["frame"]): r for r in rows if r["mask_prompt"] == "hard"}
            s = {int(r["frame"]): r for r in rows if r["mask_prompt"] == "soft"}

            def f(d, k, w):
                return f"{float(d[k]):>{w}.4f}" if d else " " * w

            for t in sorted(set(h) | set(s)):
                a, b = h.get(t), s.get(t)
                px = (a["pred_area"] if a else 0)
                gt = (a["gt_area"] if a else 0)
                spx = (b["pred_area"] if b else 0)
                print(f"{t:>4}{px:>10}{gt:>9}{f(a, 'area_ratio', 9)}{f(a, 'JF', 9)}"
                      f"{spx:>10}{f(b, 'area_ratio', 9)}{f(b, 'JF', 9)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
