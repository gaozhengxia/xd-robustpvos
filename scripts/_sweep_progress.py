"""Progress report for a resume-able, possibly CHUNKED baseline sweep.

Reads every `results/baselines_raw*.csv` (the merged file plus any per-chunk files
produced by the parallel workflow in `94_merge_sweeps.py`) and reports how much is
done, so the remaining work is visible without parsing tqdm's carriage-return log.

Deliberately read-only and GPU-free: it is meant to be run WHILE the sweep is
writing, so it tolerates the torn last line a concurrent writer can leave behind.
"""
from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "baselines_raw"

DEGS = ["fog", "lowlight", "rain", "snow", "motion_blur", "sensor_noise",
        "underwater", "dust"]
LEVELS = ["1.0", "2.0", "3.0", "4.0", "5.0"]
N_SEQS = 30
REQUIRED = ("seq", "mode", "degradation", "level", "n_frames")


def main() -> int:
    parts = sorted(Path(p) for p in glob.glob(str(BASE) + "*.csv"))
    if not parts:
        print("no results/baselines_raw*.csv yet")
        return 1

    cells = set()
    objs = defaultdict(set)
    rows = 0
    for p in parts:
        n = 0
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                # skip the torn line a concurrent writer can expose
                if any(r.get(c) in (None, "") for c in REQUIRED):
                    continue
                n += 1
                rows += 1
                cells.add((r["seq"], r["mode"], r["degradation"], str(r["level"])))
                objs[r["seq"]].add(r["obj_id"])
        print(f"  {p.name:<32} {n:>7} rows")

    total = N_SEQS * len(DEGS) * len(LEVELS)
    print(f"\nrows                : {rows}")
    print(f"completed cells     : {len(cells)} / {total}  ({len(cells)/total*100:.1f}%)")
    print(f"sequences touched   : {len(objs)}")
    print(f"instances seen      : {sum(len(v) for v in objs.values())}")
    per_seq = defaultdict(int)
    for s, _m, _d, _l in cells:
        per_seq[s] += 1
    done_seq = sorted(s for s, n in per_seq.items() if n == len(DEGS) * len(LEVELS))
    part_seq = sorted(s for s, n in per_seq.items() if n < len(DEGS) * len(LEVELS))
    print(f"sequences finished  : {len(done_seq)}")
    print(f"sequences in flight : {len(part_seq)}  {part_seq[:6]}")
    remaining = total - len(cells)
    for rate, tag in ((5.45, "4-way parallel"), (13.0, "serial")):
        print(f"remaining cells     : {remaining}  (~{remaining * rate / 60:.0f} min "
              f"at {rate} s/cell, {tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
