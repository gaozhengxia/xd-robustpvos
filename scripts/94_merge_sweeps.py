"""94 - Audit and merge chunked sweep outputs.

WHY THIS EXISTS
---------------
The CL3 sweep is 1200 cells and it is **CPU-bound**: during a run `nvidia-smi`
reports the GPU at ~3%, because a cell spends its time in Farneback optical flow,
the IQA panel and the degradation renderer -- not in SAM 2. The machine has 24
cores, so the sweep can be split into disjoint sequence ranges that run in
PARALLEL, turning a ~4 h serial run into well under an hour.

Parallelism is only safe if the jobs do NOT share an output file:
`maybe_append_csv` opens the CSV once per finished cell, which is not atomic
across processes, so two writers can splice two rows into one corrupt line. Each
chunk therefore writes its own CSV and this tool merges them afterwards.

It also closes the hole a killed chunk otherwise leaves open. `maybe_append_csv`
writes a cell only after it is computed, so a process killed *during the compute*
loses nothing (the cell was never marked done). But a process killed *during the
append itself* leaves a truncated cell, and the resume marker is cell-level -- "a
cell counts as done as soon as it has any row" -- so that cell would then be
skipped forever and its partial mean would enter the tables. `--trim` removes those
rows from the chunk file itself, which is the only place the fix can live: dropping
them from the merged copy alone would not stop the next resume from skipping the
cell.

    python scripts/94_merge_sweeps.py --status               # read-only audit
    python scripts/94_merge_sweeps.py --trim                 # drop partial cells
    python scripts/94_merge_sweeps.py --out results/baselines_raw.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: A chunk file is `<base>.csv` or `<base>_s<offset>.csv`, so one glob picks up
#: every chunk while `clean_baseline_all.csv` never matches.
DEFAULT_BASE = "results/baselines_raw"
DEFAULT_OUT = "results/baselines_raw.csv"

#: Protocol the coverage report is checked against (greedy CL3 on DAVIS-2017 val).
EXPECTED_SEQS = 30
EXPECTED_DEGS = 8
EXPECTED_LEVELS = 5


def fnum(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def discover(base: str) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(str(ROOT / f"{base}*.csv")))


REQUIRED_COLS = ("seq", "obj_id", "mode", "degradation", "level", "n_frames")

#: The sweep writes ONE ROW PER FRAME when `collect_iqa` is on (CL3 needs the
#: per-frame IQA correlation) and ONE ROW PER CELL when it is off (`--no-iqa`,
#: which is what the CL4 / baseline grids run). The two formats need different
#: bookkeeping: only the per-frame one can be checked for a torn append by
#: comparing the row count against `n_frames`. Reading a per-cell chunk with the
#: per-frame rules used to drop EVERY row (no `frame` column) and report an empty
#: audit -- a silent no-op that reads like "no data". So the format is detected
#: from the header instead of assumed.
FRAME_COLS = ("frame", "J_frame", "F_frame", "JF_frame")
CELL_COLS = ("J", "F", "J&F")


def read_csv(p: Path) -> Tuple[List[str], List[Dict[str, str]], int]:
    """Read a chunk. Rows missing a required column are dropped, not crashed on.

    `--status` is expected to be run WHILE a chunk is writing, and a reader can
    catch the last line after `w.writerows` has flushed part of it. Such a row
    parses as a dict with `None` values; it is a torn line, not data, and the
    cell it belongs to is reported as partial by `partial_cells` anyway.

    Returns the dropped-row count so a torn line is *reported* rather than
    silently subtracted. Either way the cell was never marked done, so a resume
    re-runs it -- but only if the user knows to resume.
    """
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        header = list(rd.fieldnames or [])
        need = tuple(c for c in REQUIRED_COLS if c in header)
        rows, dropped = [], 0
        for r in rd:
            if all(r.get(c) not in (None, "") for c in need):
                rows.append(r)
            else:
                dropped += 1
    return header, rows, dropped


def write_csv(p: Path, header: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in header})
    tmp.replace(p)


def cell_of(r: Dict[str, str]) -> tuple:
    return (r["seq"], r["mode"], r["degradation"], str(r["level"]))


def partial_cells(rows: Sequence[Dict[str, str]], has_frame: bool = True) -> Set[tuple]:
    """Cells where some instance does not have exactly `n_frames` rows.

    `evaluate_cell` writes one row per frame and stamps `n_frames` (constant within
    a cell), so a short or oversized group means the cell was killed mid-append.
    A cell whose `n_frames` is missing/zero is also reported -- it cannot be
    verified, and an unverifiable cell should not be trusted.

    Without `collect_iqa` there are no per-frame rows at all: a cell IS one row,
    so a torn append cannot leave a *partial* cell -- it leaves a dropped line
    (`read_csv`) and a cell that was never marked done. `has_frame=False` returns
    an empty set for that reason; it is not "no check", it is "this check does
    not apply to this format".
    """
    if not has_frame:
        return set()
    counts: Dict[tuple, int] = {}
    declared: Dict[tuple, int] = {}
    for r in rows:
        k = (r["seq"], r["obj_id"], r["mode"], r["degradation"], str(r["level"]))
        counts[k] = counts.get(k, 0) + 1
        declared.setdefault(k, int(fnum(r.get("n_frames"))))
    bad: Set[tuple] = set()
    for k, n in counts.items():
        if declared[k] <= 0 or n != declared[k]:
            bad.add((k[0], k[2], k[3], k[4]))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help=f"chunk-file base path (default: {DEFAULT_BASE})")
    ap.add_argument("--out", default=None,
                    help=f"write the merged CSV here (the pipeline's name is "
                         f"{DEFAULT_OUT}); omit to audit only")
    ap.add_argument("--status", action="store_true", help="audit only, never writes")
    ap.add_argument("--trim", action="store_true",
                    help="rewrite each chunk file without the rows of cells that were "
                         "killed mid-append, so the next --resume re-runs them")
    args = ap.parse_args()

    parts = discover(args.base)
    # The merged file matches the chunk glob (same base), so re-running the merge
    # would otherwise read its own previous output as if it were a chunk.
    out_path: Optional[Path] = None
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path = out_path.resolve()
        parts = [p for p in parts if p.resolve() != out_path]
    print("XD-RobustPVOS :: audit / merge chunked sweeps")
    print("=" * 68)
    if not parts:
        print(f"  no chunk files match {args.base}*.csv")
        return 1

    header: List[str] = []
    all_rows: List[Dict[str, str]] = []
    per_part: Dict[str, List[Dict[str, str]]] = {}
    dropped_total = 0
    for p in parts:
        h, rows, dropped = read_csv(p)
        dropped_total += dropped
        if not header:
            header = h
        elif h != header:
            raise SystemExit(
                f"header mismatch in {p.name}: "
                f"{sorted(set(h) ^ set(header))[:8]}")
        per_part[p.name] = rows
        for r in rows:
            r["_src"] = p.name
        all_rows.extend(rows)

    # One row per frame only when the sweep collected IQA; otherwise one row per
    # cell. Detected from the header, never assumed -- see FRAME_COLS.
    has_frame = "frame" in header
    cmp_cols = FRAME_COLS[1:] if has_frame else CELL_COLS
    fmt = "per-frame (collect_iqa on)" if has_frame else "per-cell (--no-iqa)"
    print(f"  format: {fmt}")
    if not has_frame:
        print("  [note] per-cell chunks carry no per-frame rows, so the "
              "torn-append check is not applicable; a torn line is simply "
              "dropped and its cell re-run by --resume.")
    if dropped_total:
        print(f"  [note] {dropped_total} torn/partial line(s) dropped while "
              f"reading; their cells were never marked done -> --resume re-runs them")

    for p in parts:
        rows = per_part[p.name]
        cells = len({cell_of(r) for r in rows})
        seqs = len({r["seq"] for r in rows})
        bad = len(partial_cells(rows, has_frame))
        flag = f"  <-- {bad} PARTIAL CELL(S)" if bad else ""
        print(f"  {p.name:<32} {len(rows):>7} rows  {cells:>4} cells  "
              f"{seqs:>2} seqs{flag}")

    # ---- conflicting duplicates across chunks: fatal, nothing is written ----
    # Identity columns differ per format: the per-frame chunks must agree on the
    # per-frame scores, the per-cell chunks on the cell aggregate.
    val_col = cmp_cols[-1]
    seen: Dict[tuple, Dict[str, str]] = {}
    agreed = 0
    conflicts: List[str] = []
    for r in all_rows:
        k = (r["seq"], r["obj_id"], r["mode"], r["degradation"],
             str(r["level"]), str(r.get("frame")))
        if k in seen:
            a, b = seen[k], r
            if all(a.get(c) == b.get(c) for c in cmp_cols):
                agreed += 1
            else:
                conflicts.append(
                    f"{k}: {val_col} {a.get(val_col)} ({a['_src']}) vs "
                    f"{b.get(val_col)} ({b['_src']})")
        else:
            seen[k] = r

    cells_all = {cell_of(r) for r in all_rows}
    unit = "frame" if has_frame else "cell"
    if has_frame:
        expected = EXPECTED_SEQS * EXPECTED_DEGS * EXPECTED_LEVELS
        head = f"\n  union: {len(cells_all)}/{expected} cells, "
    else:
        head = f"\n  union: {len(cells_all)} cells, "
    print(f"{head}{len(seen)} distinct (cell, obj, {unit}) rows, "
          f"{len({(r['seq'], r['obj_id']) for r in all_rows})} instances")
    if agreed:
        print(f"  [note] {agreed} rows appeared in >1 chunk and agreed exactly")
    for c in conflicts[:10]:
        print(f"  [PROBLEM] conflicting duplicate {c}")
    if conflicts:
        print("\n  refusing to write: the chunks were not produced by the same "
              "protocol, so merging them would be meaningless.")
        return 1

    total_bad = sum(len(partial_cells(v, has_frame)) for v in per_part.values())
    if args.trim and not has_frame:
        print("\n  --trim: not applicable to per-cell chunks (a cell is one row, "
              "so there is no half-written cell to drop)")
    elif args.trim:
        if not total_bad:
            print("\n  --trim: nothing to do, every cell is complete")
        else:
            print()
            for p in parts:
                rows = per_part[p.name]
                bad = partial_cells(rows, has_frame)
                if not bad:
                    continue
                kept = [r for r in rows if cell_of(r) not in bad]
                write_csv(p, header, kept)
                print(f"  trimmed {p.name}: dropped {len(rows) - len(kept)} rows "
                      f"from {len(bad)} cell(s) -> re-run with --resume to redo them")
                for c in sorted(bad):
                    print(f"      {c[0]} | {c[1]}@{c[2]} | {c[3]}")
    elif total_bad:
        print(f"\n  [note] {total_bad} partial cell(s) present; run --trim before "
              f"merging, otherwise resume will skip them forever")

    if not args.out or args.status:
        print("\n  (audit only -- nothing written)")
        return 0

    dst = out_path if out_path is not None else Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, str]] = []
    taken: Set[tuple] = set()
    for r in all_rows:
        k = (r["seq"], r["obj_id"], r["mode"], r["degradation"],
             str(r["level"]), str(r.get("frame")))
        if k in taken:
            continue
        taken.add(k)
        out_rows.append({c: r.get(c, "") for c in header})
    write_csv(dst, header, out_rows)
    print(f"\n  wrote {len(out_rows)} rows -> {dst}")
    print(f"  cells: {len({cell_of(r) for r in out_rows})}")
    print("\n  Reminder: the chunk files still exist and now duplicate the merged")
    print("  file. Continue with --resume --out " + str(args.out) + " and delete the")
    print("  chunk files, so a cell cannot be re-run and appended twice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
