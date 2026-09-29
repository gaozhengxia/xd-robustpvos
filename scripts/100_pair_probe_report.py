"""100 - Aggregate a PAIRED probe between two arms (read-only, safe mid-sweep).

    python scripts/100_pair_probe_report.py [csv] [deg_filter] [modeA] [modeB]

Defaults to `greedy` vs `dagrs`; any two arms present in the CSV work (e.g.
`greedy sam2video` for the memory-bank comparator).

Reports per-sequence delta, plus the **median** and the min/max. The mean alone is
not enough: a 21-cell probe where 19 cells sit within +/-0.02 and the remaining
two are -0.024 / +0.150 has a mean of +0.008 and a median of +0.001, and only
the median describes what the method actually does. It also prints the
mechanism counters (`n_reanchors`), because a gain attributed to a mechanism
whose counter is 0 everywhere is not a gain.

INSTANCE COLLAPSE (2026-09-17 fix). The ranked instances are scored AND written
per frame, so a sequence with k objects contributes k * n_frames rows. This
script used to key on (degradation, seq) and keep the FIRST row per mode, which
silently scored obj1 only -- `bike-packing` and `bmx-trees` have 2 objects each,
and `bike-packing` is precisely the sequence with the largest delta, so the
headline number was computed on a subset of the protocol. The aggregation now
mirrors `05_analysis.collapse_objects()`: average the instances of a sequence
first, then give every sequence equal weight. `collapse_pair_rows()` is a pure
function so `00_smoke_test.py` can pin that semantics on synthetic rows.

NOTE: the sweep writes the per-sequence aggregate on EVERY frame row
(there are no dedicated summary rows), so the first row per
(seq, obj_id, degradation, mode) already carries the authoritative J / F / J&F
and the mechanism counters.
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

_OBJ_SORT = lambda x: (len(str(x)), str(x))  # noqa: E731  ("1" < "2" < "10")


def read_rows(path):
    """Read the sweep CSV into plain dicts (kept separate so the logic is testable)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collapse_pair_rows(rows, deg_filter=None, arm_a="greedy", arm_b="dagrs"):
    """(deg, seq) -> per-sequence collapse of the instance rows, per arm.

    Returns (seqs, pairs, multi, reanch):
      * `seqs`   sorted list of (deg, seq) that have BOTH arms on at least one instance
      * `pairs`  (deg, seq, obj) -> {mode: row}, the per-instance first row
      * `multi`  (deg, seq, objs, arms, delta) for sequences with >1 instance
      * `reanch` (deg, seq) -> n_reanchors summed over instances

    Instances are averaged (not first-row-wins) because a sequence's official
    DAVIS value is the mean over its annotated objects.
    """
    inst = defaultdict(dict)
    for r in rows:
        if deg_filter and r["degradation"] != deg_filter:
            continue
        key = (r["degradation"], r["seq"], r["obj_id"])
        if r["mode"] not in inst[key]:
            inst[key][r["mode"]] = r

    pairs = {}
    reanch = {}
    multi = []
    seqs = []
    for (deg, seq) in sorted({(d, s) for (d, s, _) in inst}):
        objs = sorted([o for (d, s, o) in inst if (d, s) == (deg, seq)], key=_OBJ_SORT)
        arms = {arm_a: [], arm_b: []}
        n_re = 0
        for o in objs:
            row = inst[(deg, seq, o)]
            pairs[(deg, seq, o)] = row
            for m in arms:
                if m in row:
                    arms[m].append(float(row[m]["J&F"]))
            if arm_b in row:
                n_re += int(float(row[arm_b]["n_reanchors"]))
        if not arms[arm_a] or not arms[arm_b]:
            continue
        seqs.append((deg, seq))
        reanch[(deg, seq)] = n_re
        if len(objs) > 1:
            multi.append((deg, seq, objs, arms,
                          st.mean(arms[arm_b]) - st.mean(arms[arm_a])))
    return seqs, pairs, multi, reanch


def instance_deltas(pairs, arm_a="greedy", arm_b="dagrs"):
    """Per-instance (deg, seq, obj, n_reanchors, delta) for the arms present.

    Reported alongside the sequence means because a sequence mean can hide the
    fact that the mechanism's *count* carries no information about the outcome:
    the interesting question is not "how many times did it fire" but "does more
    firing mean a better result", and only the instance-level scatter answers it.
    """
    keys = sorted({(d, s, o) for (d, s, o) in pairs})
    out = []
    for (deg, seq, obj) in keys:
        row = pairs[(deg, seq, obj)]
        if arm_a not in row or arm_b not in row:
            continue
        out.append((deg, seq, obj, int(float(row[arm_b]["n_reanchors"])),
                    float(row[arm_b]["J&F"]) - float(row[arm_a]["J&F"])))
    return out


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
    sx = (sum((a - mx) ** 2 for a in xs) / n) ** 0.5
    sy = (sum((b - my) ** 2 for b in ys) / n) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default = os.path.join(ROOT, "_scratch", "attr_20260917", "dagrs_probe.csv")
    arg = sys.argv[1] if len(sys.argv) > 1 else _default
    PATH = arg if os.path.isabs(arg) else os.path.join(os.getcwd(), arg)
    DEG_FILTER = sys.argv[2] if len(sys.argv) > 2 else None
    # `ALL` means "no filter". Treating it as a literal degradation name filters
    # every row out and prints an empty table, which reads like "no data yet"
    # instead of "you asked for a degradation that does not exist".
    if DEG_FILTER and DEG_FILTER.strip().upper() == "ALL":
        DEG_FILTER = None
    ARM_A = sys.argv[3] if len(sys.argv) > 3 else "greedy"
    ARM_B = sys.argv[4] if len(sys.argv) > 4 else "dagrs"

    if not os.path.exists(PATH):
        print("MISSING", PATH)
        return 1

    seqs, pairs, multi, reanch = collapse_pair_rows(
        read_rows(PATH), DEG_FILTER, arm_a=ARM_A, arm_b=ARM_B)

    print(f"{os.path.basename(PATH)}  degradation filter={DEG_FILTER or 'ALL'}  "
          f"arms: {ARM_A} -> {ARM_B}")
    print("instance-collapsed per (seq, degradation); n_obj = ranked objects")
    print(f"{'deg':<12}{'seq':<16}{ARM_A:>10}{ARM_B:>10}{'delta':>10}"
          f"{'n_obj':>6}{'d_re':>6}{'a_enc':>7}{'b_enc':>7}{'b_fail':>7}")
    print("-" * 95)

    by_deg = defaultdict(list)
    for (deg, seq) in seqs:
        objs = sorted([o for (d, s, o) in pairs if (d, s) == (deg, seq)], key=_OBJ_SORT)
        gvals = [float(pairs[(deg, seq, o)][ARM_A]["J&F"])
                 for o in objs if ARM_A in pairs[(deg, seq, o)]]
        dvals = [float(pairs[(deg, seq, o)][ARM_B]["J&F"])
                 for o in objs if ARM_B in pairs[(deg, seq, o)]]
        if not gvals or not dvals:
            continue
        first = pairs[(deg, seq, objs[0])]
        gd, dd = first.get(ARM_A), first.get(ARM_B)
        gj, dj = st.mean(gvals), st.mean(dvals)
        by_deg[deg].append((seq, dj - gj))
        print(f"{deg:<12}{seq:<16}{gj:>10.4f}{dj:>10.4f}{dj - gj:>+10.4f}"
              f"{len(objs):>6}{reanch[(deg, seq)]:>6}"
              f"{gd['n_encoder_calls'] if gd else '-':>7}"
              f"{dd['n_encoder_calls'] if dd else '-':>7}"
              f"{dd['n_predict_failures'] if dd else '-':>7}")

    print("-" * 95)

    # ---- multi-instance split: the mean must not hide a single broken instance ----
    if multi:
        print("multi-instance sequences (per-object delta; the sequence value is their mean)")
        print(f"{'deg':<12}{'seq':<16}{'obj':>4}{ARM_A:>10}{ARM_B:>10}{'delta':>10}")
        print("-" * 95)
        for (deg, seq, objs, arms, _) in multi:
            for i, o in enumerate(objs):
                gv = arms[ARM_A][i] if i < len(arms[ARM_A]) else float("nan")
                dv = arms[ARM_B][i] if i < len(arms[ARM_B]) else float("nan")
                print(f"{deg:<12}{seq:<16}{o:>4}{gv:>10.4f}{dv:>10.4f}{dv - gv:>+10.4f}")
        print("-" * 95)

    def _summ(tag, ds):
        n = len(ds)
        if not n:
            return
        near = sum(1 for x in ds if abs(x) <= 0.02)
        # the gain the mean claims once any single dramatic sequence is removed --
        # if the mean collapses, the headline was one sequence, not the method
        trimmed = sorted(ds)[:-1] or ds
        print(f"[{tag}] paired={n}  mean D={st.mean(ds):+.4f}  "
              f"median D={st.median(ds):+.4f}  wins={sum(1 for x in ds if x > 0)}/{n}  "
              f"|D|<=0.02 in {near}/{n}")
        print(f"        min/max={min(ds):+.4f}/{max(ds):+.4f}   "
              f"mean without the single largest D = {st.mean(trimmed):+.4f}")

    all_d = []
    for deg in sorted(by_deg):
        ds = [x[1] for x in by_deg[deg]]
        all_d += ds
        _summ(deg, ds)
    if all_d:
        _summ("POOLED", all_d)
        if ARM_B.startswith("dagrs"):
            # Only meaningful for an arm that HAS the mechanism. Printing
            # "the mechanism never ran" for a memory-bank arm would be a
            # category error: that arm has no re-anchoring by construction.
            zeros = sum(1 for v in reanch.values() if v == 0)
            print(f"[MECHANISM] n_reanchors == 0 in {zeros}/{len(reanch)} cells"
                  + ("  ==> THE MECHANISM NEVER RAN; any 'gain' is not from re-anchoring."
                     if zeros == len(reanch) else ""))
            print(f"        n_reanchors > 0 in {len(reanch) - zeros}/{len(reanch)} cells")

    # ---- is the mechanism's firing COUNT a usable knob? ---------------------- #
    # A mechanism that helps would show up as "more firing -> larger delta".
    # A mechanism that is merely *unconditional* shows r ~ 0: the count then says
    # nothing about the outcome, and no threshold tuning can rescue the headline.
    per_inst = instance_deltas(pairs, arm_a=ARM_A, arm_b=ARM_B)
    if per_inst and any(x[3] for x in per_inst):
        xs = [x[3] for x in per_inst]
        ys = [x[4] for x in per_inst]
        r = _pearson(xs, ys)
        fired = [x[4] for x in per_inst if x[3] > 0]
        quiet = [x[4] for x in per_inst if x[3] == 0]
        print(f"[PER-INSTANCE] n={len(per_inst)} instances  "
              f"r(n_reanchors, delta) = {r:+.3f}"
              + ("   ==> firing count carries no signal about the outcome"
                 if abs(r) < 0.3 else ""))
        if fired:
            print(f"        n_re>0: n={len(fired)} cells  meanD={st.mean(fired):+.4f}  "
                  f"medianD={st.median(fired):+.4f}  wins={sum(1 for v in fired if v > 0)}/{len(fired)}")
        if quiet:
            print(f"        n_re=0 : n={len(quiet)} cells  meanD={st.mean(quiet):+.4f}  "
                  f"wins={sum(1 for v in quiet if v > 0)}/{len(quiet)}"
                  "   (expect ~0: only the re-anchor path reads `appearance`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
