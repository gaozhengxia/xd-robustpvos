# -*- coding: utf-8 -*-
"""Owner script for the §5.9 paired-test rows, plus a mirror check for the §5
result tables.

Two jobs, both about the *same class of defect*: a number that exists verbatim
somewhere in the repository but no longer means what the text says it means.

(A) **Paired tests on the P1 convention.**  The rows previously written into
    TABLE-14 of `docs/PAPER_FRAMEWORK.md` were computed on the P0 inputs: the
    family gap read +0.1591 there while the P1 convention (§5.1b) gives +0.1561.
    Table-internal numbers must all be P1, so this script recomputes the rows from
    `results/p1/results/*.csv`, per sequence over all 13 degradation cells, and
    `--check` requires the result to appear in both documents.

    As of 2026-09-28 the table has **five** rows: the two comparisons against the
    ablation arms A1 (equal-weight TTA) and A3 (random operator) are no longer
    withheld, because those arms have now been run on all 13 cells.  Their inputs
    live in two scratch sweeps, not in the P1 grid, and they are recorded in the
    **unscaled** metric -- see `_load_ablation` / `lift_to_p1`, and the two guards
    in `ablation_diagnostics` that admit them.  Note the side effect: the Holm
    correction is over the table's comparisons, so m went from 3 to 5 and the
    three original rows' adjusted p changed (their means, medians, CIs, effect
    sizes and win/loss counts did not).

(B) **Manuscript mirror check.**  The tables in `docs/MANUSCRIPT.md` §5 are
    transcriptions of the script-generated blocks in `docs/PAPER_FRAMEWORK.md`
    (which `scripts/118_p1_doc_sync.py` already reconciles against `results/p1/`).
    A transcription is a place where a correct number can acquire a wrong
    neighbour, so every manuscript table carries its own `<!-- MS-TABLE: name -->`
    anchor and is compared cell-by-cell with the framework block of the same
    name.  Comparison is by numeric tuple per row: it catches a wrong value and
    a swapped column, but not, e.g., two columns with equal values.

Usage
-----
    python scripts/121_sec5_table_audit.py --emit        # authoritative §5.9 rows
    python scripts/121_sec5_table_audit.py --check       # verify both documents
    python scripts/121_sec5_table_audit.py --self-test   # prove the checker can fail
"""
import io
import os
import re
import sys
import csv
import random
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANUSCRIPT = "docs/MANUSCRIPT.md"
FRAMEWORK = "docs/PAPER_FRAMEWORK.md"

#: `FRAMEWORK` and the two sweep directories are deliberately NOT distributed
#: (see `scripts/_release_gate.py`); in a clone this guard reports a SKIP rather
#: than dying on the first `open()`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _release_gate import gate as release_gate                       # noqa: E402

#: manuscript mirror tables: anchor name -> the manuscript's row labels (its first
#: cell), **in the framework block's row order**.  Labels repeat where the framework
#: block's first cell repeats (the composition test has one row per member pair), so
#: the label check is a multiset check and the cell check carries the real weight.
#: The name is the one used by `scripts/118_p1_doc_sync.py` in its
#: `<!-- P1-SYNC: name -->` anchor.
MIRROR_TABLES = {
    "table9all": {"rows": [
        "clean", "fog", "dust", "underwater", "lowlight", "motion_blur",
        "rain", "snow", "sensor_noise",
        "POOLED (CL4, 4 degradations)", "POOLED (all 9 single degradations)"]},
    "table9c": {"rows": [
        "fog", "dust", "underwater", "lowlight", "motion_blur", "rain",
        "snow", "sensor_noise",
        "C1 (fog+noise)", "C2 (lowlight+blur)", "C3 (rain+snow)",
        "C4 (fog+blur+noise)", "single-degradation min-max"]},
    "table10": {"rows": [
        "clean", "fog", "dust", "underwater", "lowlight", "motion_blur",
        "rain", "snow", "sensor_noise",
        "single-degradation mean (9, equal weight)",
        "C1 (fog+noise)", "C2 (lowlight+blur)", "C3 (rain+snow)",
        "C4 (fog+blur+noise)",
        "compound mean (4, equal weight)"]},
    "widen_pairs": {"rows": [
        "C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4", "C4"]},
    "widen_pool": {"rows": ["family gap"]},
}

#: the §5.9 paired-test rows this script owns.  The last two were unreportable
#: until 2026-09-28, when A1 and A3 finished on all 13 degradation cells.
STAT_PAIRS = [("sam2video", "greedy"), ("dagrs", "greedy"), ("dagrs", "sam2video"),
              ("dagrs", "A1"), ("dagrs", "A3")]
P1_INPUTS = ["singles_greedy.csv", "singles_dagrs.csv", "singles_sam2video.csv",
             "cl4_full_v2.csv", "c1c4_a_dagrs.csv", "c1c4_a_base.csv",
             "c1c4_b_dagrs.csv", "c1c4_b_base.csv"]
ARM_ALIAS = {"base": "greedy", "greedy": "greedy", "dagrs": "dagrs",
             "sam2video": "sam2video"}

#: The 13 cells of a table row are the P1 grid's (`_load_p1`) plus the eleven cells
#: A1/A3 were never run on.  Those eleven come from the 2026-09-24 completion sweep
#: and the remaining two (fog, C1_fog_noise) from the 2026-09-20 two-phase sweep --
#: the tiling of the 13 cells across the two runs is asserted in `_load_ablation`.
#: (`tag in the file name`, `degradation value inside the CSV`)
ABL_SOURCES = [
    ("_scratch/abl13_20260924", [
        ("clean", "clean"), ("dust", "dust"), ("underwater", "underwater"),
        ("lowlight", "lowlight"), ("motion_blur", "motion_blur"), ("rain", "rain"),
        ("sensor_noise", "sensor_noise"), ("snow", "snow"),
        ("C2", "C2_lowlight_blur"), ("C3", "C3_rain_snow"),
        ("C4", "C4_fog_blur_noise")]),
    ("_scratch/ablation_20260920", [("fog", "fog"), ("C1", "C1_fog_noise")]),
]
N_CELLS = 13
N_INSTANCES = 61
#: file-name arm -> the arm key inside this table.  A0 *is* the full method under
#: `configs/_tau013.yaml`, i.e. the same thing the P1 grid calls `dagrs`; mapping it
#: to `dagrs` is what makes the drift guard below a like-for-like comparison.
ABL_ARM_KEY = {"A0": "dagrs", "A1": "A1", "A3": "A3"}
#: The `dagrs` column rebuilt from the two new runs must reproduce the P1 grid's
#: `dagrs` column, or the two new rows would be compared against a different
#: computation than the three existing ones.  1e-6 is the tolerance `110` uses.
ABL_DRIFT_TOL = 1e-6

N_BOOT = 10000
SEED = 7
MS_ANCHOR = "MS-TABLE:"

#: §5.9 rows as they must be written; format is pinned by the checker
stat_row_tpl = ("| {label} | {n} | **{mean:+.4f}** | {median:+.4f} | {p} | {holm} | "
                "**{cliff:+.3f}** | [{lo:+.4f}, {hi:+.4f}] | **{win} / {lose}** |")


# --------------------------------------------------------------------------- #
# markdown table parsing
# --------------------------------------------------------------------------- #
NUM_RE = re.compile(r"[-+\u2212]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _norm_num(tok):
    tok = tok.replace("\u2212", "-")
    return float(tok)


def _nums(text):
    return tuple(_norm_num(t) for t in NUM_RE.findall(text.replace("\u2212", "-")))


def parse_tables(text):
    """Return {anchor_name: [(row_label, [cells])]} for `<!-- anchor: name -->`.

    A table is a maximal run of lines starting with `|`.  The label is the first
    cell; the separator row is dropped.
    """
    out = {}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.search(re.escape(MS_ANCHOR) + r"\s*([A-Za-z0-9_]+)\s*-->", line)
        if not m:
            m2 = re.search(r"<!--\s*P1-SYNC:\s*([A-Za-z0-9_]+)\s*-->", line)
            if not m2:
                continue
            m = m2
        name = m.group(1)
        rows = []
        j = i + 1
        while j < len(lines) and not lines[j].lstrip().startswith("|"):
            j += 1
        header = None
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
            if header is None:
                header = cells
            elif not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                rows.append((cells[0], cells[1:]))
            j += 1
        out[name] = (header, rows)
    return out


# --------------------------------------------------------------------------- #
# (A) paired tests on the P1 convention
# --------------------------------------------------------------------------- #
def _load_p1(root):
    inst = {}
    for fn in P1_INPUTS:
        p = os.path.join(root, "results", "p1", "results", fn)
        if not os.path.exists(p):
            raise IOError("missing P1 input " + p)
        with io.open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                arm = ARM_ALIAS.get(r["mode"])
                if arm is None:
                    continue
                jf = r.get("J&F")
                if jf in ("", None):
                    jf = r.get("jf")
                if jf in ("", None):
                    raise KeyError("no J&F column in " + fn)
                inst[(arm, r["seq"], r["obj_id"], r["degradation"])] = float(jf)
    if not inst:
        raise AssertionError("no P1 rows loaded")
    return inst


def _load_ablation(root):
    """Arm-tagged rows for the 13 cells, from the two runs that produced A1/A3.

    Two traps, both silent:

    * **The arm is the FILE NAME, never the `mode` column.**  The driver writes
      the equal-weight arm (A1) and the full method (A0) both as `mode = dagrs`
      (only A3 differs, `random_op`), so keying arms off `mode` -- correct for the
      P1 grid, where the three arms do differ -- would fold A1 into A0 here and
      report a fabricated zero delta.
    * **These files carry the UNSCALED metric** in `J&F`.  They must be lifted
      into the P1 convention before they can sit beside P1 rows; see `lift_to_p1`.

    Returns `(rows, n_frames)`, keyed by (arm, seq, obj_id, degradation).
    """
    inst, nframes = {}, {}
    cells = collections.defaultdict(set)
    for rel, tags in ABL_SOURCES:
        for tag, deg in tags:
            for arm in sorted(ABL_ARM_KEY):
                key = ABL_ARM_KEY[arm]
                p = os.path.join(root, rel, "%s_%s.csv" % (tag, arm))
                if not os.path.exists(p):
                    raise IOError("missing ablation input " + p)
                n = 0
                with io.open(p, encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        if r.get("degradation") != deg:
                            raise AssertionError(
                                "%s: degradation %r, expected %r"
                                % (p, r.get("degradation"), deg))
                        jf = r.get("J&F")
                        if jf in ("", None):
                            jf = r.get("jf")
                        if jf in ("", None):
                            raise KeyError("no J&F column in " + p)
                        k = (key, r["seq"], r["obj_id"], deg)
                        inst[k] = float(jf)
                        nframes[k] = int(float(r["n_frames"]))
                        n += 1
                if n != N_INSTANCES:
                    raise AssertionError("%s: %d instance rows, expected %d"
                                         % (p, n, N_INSTANCES))
                cells[key].add(deg)
    for key in sorted(set(ABL_ARM_KEY.values())):
        if len(cells[key]) != N_CELLS:
            raise AssertionError(
                "ablation arm %s covers %d cell(s), expected %d: %s"
                % (key, len(cells[key]), N_CELLS, sorted(cells[key])))
    if set(cells["A1"]) != set(cells["A3"]):
        raise AssertionError("A1 and A3 do not cover the same cell set")
    return inst, nframes


def _p1_scale(root):
    """(seq, obj_id, degradation) -> (n_clip, n_present, p1_scale) from the P1 grid.

    §5.1b makes the P1 denominator the target-present frame count -- a property of
    the ground truth, not of the arm -- so a single multiplier serves every arm on
    a cell.  Reading it off the grid's `dagrs` rows is therefore not a choice
    between arms, and `lift_to_p1` checks the consequence rather than trusting it.
    """
    out = {}
    for fn in P1_INPUTS:
        p = os.path.join(root, "results", "p1", "results", fn)
        with io.open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if ARM_ALIAS.get(r["mode"]) != "dagrs":
                    continue
                k = (r["seq"], r["obj_id"], r["degradation"])
                v = (int(float(r["n_clip"])), int(float(r["n_present"])),
                     float(r["p1_scale"]))
                if k in out and out[k] != v:
                    raise AssertionError("the P1 scale disagrees with itself on %s"
                                         % (k,))
                out[k] = v
    if len(out) != N_CELLS * N_INSTANCES:
        raise AssertionError("P1 scale map has %d instances, expected %d"
                             % (len(out), N_CELLS * N_INSTANCES))
    return out


def lift_to_p1(rows, nframes, scale):
    """Ablation raw metric -> P1, refusing unless the data licenses the lift.

    `P1 = raw * n_clip / n_present` holds only because these arms score **every**
    frame of the clip: with the ground truth empty they still emit a mask, so those
    frames score J = F = 0 and the mean over scored frames is the mean over the
    clip.  That is what makes the sum over scored frames equal the sum over present
    frames, and it is asserted per instance against the clip length the P1 grid
    recorded -- never assumed from the arm's name.  (The 2026-09-28 first attempt
    at this table compared the ablation's raw column against the P1 grid's scaled
    one and reported a 8.3e-02 "protocol drift" that was purely a unit mismatch.)
    """
    out, bad_cov, n_scaled = {}, [], 0
    for k, raw in rows.items():
        inst = (k[1], k[2], k[3])
        if inst not in scale:
            raise AssertionError("no P1 scale for instance %s" % (inst,))
        n_clip, n_present, sc = scale[inst]
        if nframes[k] != n_clip:
            bad_cov.append((k, nframes[k], n_clip))
        if abs(sc - n_clip / float(n_present)) > 1e-9:
            raise AssertionError("p1_scale %r != n_clip/n_present %r on %s"
                                 % (sc, n_clip / float(n_present), inst))
        if abs(sc - 1.0) > 1e-12:
            n_scaled += 1
        out[k] = raw * sc
    if bad_cov:
        raise AssertionError(
            "%d instance(s) were not scored over the whole clip, so `raw * scale` "
            "is not their P1 value (first: %s)" % (len(bad_cov), bad_cov[0]))
    return out, n_scaled


def ablation_diagnostics(root, p1_inst, rows=None, nframes=None, scale=None):
    """Guards that must hold before the two new rows may be reported.

    `rows` / `nframes` / `scale` may be injected so that `--self-test` can drive
    **this** function with planted faults instead of a re-implementation of it.

    1. **Protocol drift, like for like.**  `dagrs` lifted from the two ablation
       runs must equal the P1 grid's own `dagrs` **instance by instance**, on all
       793 of them.  That is the comparison `_scratch/abl13_20260924/check_drift.py`
       makes (which is what cleared the eleven new cells), and it is also what
       validates the lift: instances where `n_present < n_clip` cannot pass with a
       wrong multiplier, so the guard is not vacuous.
    2. **The arm switch actually fired.**  A1/A3 are tabulated against A0, so if
       the driver had ignored the arm flag every instance would be identical and a
       delta of ~0 would be read as "this component does not matter" when it is
       really "the override never reached the code" (§8 of the sweep skill).
    """
    if rows is None:
        rows, nframes = _load_ablation(root)
    if scale is None:
        scale = _p1_scale(root)
    lifted, n_scaled = lift_to_p1(rows, nframes, scale)
    worst, n = 0.0, 0
    for k, v in sorted(lifted.items()):
        if k[0] != "dagrs":
            continue
        ref = p1_inst.get(k)
        if ref is None:
            raise AssertionError("P1 grid has no dagrs row for %s" % (k,))
        worst = max(worst, abs(v - ref))
        n += 1
    if n != N_CELLS * N_INSTANCES:
        raise AssertionError("drift guard compared %d instances, expected %d"
                             % (n, N_CELLS * N_INSTANCES))
    if worst > ABL_DRIFT_TOL:
        raise AssertionError(
            "protocol drift: the ablation runs' `dagrs` column, lifted into the P1 "
            "convention, differs from the P1 grid by up to %.3e (tolerance %.0e) "
            "-- the new cells were not run on the P1 protocol and must not be "
            "pooled into this table" % (worst, ABL_DRIFT_TOL))
    moved = {}
    for arm in ("A1", "A3"):
        d = sum(1 for k, v in lifted.items()
                if k[0] == arm and v != lifted[("dagrs", k[1], k[2], k[3])])
        if d == 0:
            raise AssertionError(
                "arm %s is identical to A0 on every instance -- the arm flag was "
                "not applied, so its delta must not be tabulated" % arm)
        moved[arm] = (d, N_CELLS * N_INSTANCES)
    return lifted, worst, n, moved, n_scaled


def _cell_and_seq_means(inst):
    acc = collections.defaultdict(lambda: [0.0, 0])
    for (arm, seq, obj, deg), v in inst.items():
        a = acc[(arm, seq, deg)]
        a[0] += v
        a[1] += 1
    cell = {k: v[0] / v[1] for k, v in acc.items()}
    per = collections.defaultdict(list)
    for (arm, seq, deg), v in cell.items():
        per[(arm, seq)].append(v)
    return {k: sum(v) / len(v) for k, v in per.items()}, cell


def _cliff(x, y):
    gt = lt = 0
    for a in x:
        for b in y:
            if a > b:
                gt += 1
            elif a < b:
                lt += 1
    n = len(x) * len(y)
    return (gt - lt) / n if n else float("nan")


def _boot(x, y, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(x[i] - y[i] for i in idx) / n)
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def _wilcoxon(x, y):
    from scipy.stats import wilcoxon
    d = [a - b for a, b in zip(x, y) if a != b]
    if not d:
        return 1.0
    return float(wilcoxon(d, alternative="two-sided").pvalue)


def paired_rows(inst):
    per_seq, _ = _cell_and_seq_means(inst)
    seqs = sorted(set(s for (_, s) in per_seq))
    seqs = [s for s in seqs if all((a, s) in per_seq for a, _ in STAT_PAIRS)]
    if len(seqs) < 2:
        raise AssertionError("need >=2 sequences with all three arms, got %d"
                             % len(seqs))
    rows = []
    raw = {}
    for a, b in STAT_PAIRS:
        x = [per_seq[(a, s)] for s in seqs]
        y = [per_seq[(b, s)] for s in seqs]
        d = [u - v for u, v in zip(x, y)]
        lo, hi = _boot(x, y)
        raw[(a, b)] = dict(n=len(seqs), mean=sum(d) / len(d),
                           median=sorted(d)[len(d) // 2], lo=lo, hi=hi,
                           p=_wilcoxon(x, y), cliff=_cliff(x, y),
                           win=sum(1 for v in d if v > 0),
                           lose=sum(1 for v in d if v < 0))
    # Holm-Bonferroni across THIS TABLE's comparisons.  The correction family is
    # the table, so adding the two A1/A3 rows raises m from 3 to 5 and moves the
    # existing rows' adjusted p on its own -- the manuscript's Holm column is
    # therefore re-emitted, not preserved.
    order = sorted((raw[k]["p"], k) for k in STAT_PAIRS)
    prev = 0.0
    holm = {}
    m = len(order)
    for i, (p, k) in enumerate(order):
        adj = min(1.0, max(prev, p * (m - i)))
        holm[k] = adj
        prev = adj
    for a, b in STAT_PAIRS:
        r = raw[(a, b)]
        rows.append((a, b, r, holm[(a, b)]))
    if len(rows) != len(STAT_PAIRS):
        raise AssertionError("row build lost a comparison")
    return rows, len(seqs)


def fmt_p(p):
    """ASCII minus only: both documents are normalised to ASCII before matching."""
    if p == 0:
        return "0"
    if p >= 0.001:
        return "%.3f" % p
    mant, exp = ("%.1e" % p).split("e")
    return "%se%s" % (mant, exp)


def table_instances(root):
    """Everything the table pairs on: the P1 grid plus arms A1 and A3.

    Returns `(inst, diag)`; `diag` carries the two guards' diagnostics.  Only the
    two NEW arms are merged, so the comparator stays the P1 grid's own `dagrs` and
    the three original rows keep exactly the values they had before the table grew
    -- the drift guard is what licenses that.
    """
    inst = _load_p1(root)
    abl, worst, n_inst, moved, n_scaled = ablation_diagnostics(root, inst)
    for k, v in abl.items():
        if k[0] in ("A1", "A3"):
            inst[k] = v
    diag = {"drift_worst": worst, "drift_n": n_inst, "moved": moved,
            "n_comparisons": len(STAT_PAIRS), "n_scaled": n_scaled}
    return inst, diag


def build_stat_rows(root):
    """The table's rows, plus the diagnostics of the two guards that admit A1/A3."""
    inst, diag = table_instances(root)
    rows, n = paired_rows(inst)
    return [("STAT|%s|%s" % (a, b), r, h) for a, b, r, h in rows], n, diag


def emit_stat_rows(root):
    want, n, _ = build_stat_rows(root)
    return want, n


def _normalise(text):
    """Backticks off, unicode minus to ASCII, retracted fences removed.

    The fence matters: the superseded P0 rows are archived in the framework inside
    `<!-- RETRACTED-OK -->`, and a checker that reads them would report the *old*
    value as the live one (or, worse, accept it).
    """
    text = re.sub(r"<!--\s*RETRACTED-OK\s*-->.*?<!--\s*/RETRACTED-OK\s*-->", "",
                  text, flags=re.S)
    return text.replace("`", "").replace("\u2212", "-")


def check_stat_rows(root, docs):
    want, n = emit_stat_rows(root)
    fails = []
    blob = {}
    for rel in docs:
        p = os.path.join(root, rel)
        blob[rel] = _normalise(io.open(p, encoding="utf-8").read()) \
            if os.path.exists(p) else ""
    for key, r, h in want:
        _, a, b = key.split("|")
        label = "%s vs %s" % (a, b)
        need = [
            "%+.4f" % r["mean"], "%+.4f" % r["median"],
            "%+.3f" % r["cliff"],
            "%+.4f" % r["lo"], "%+.4f" % r["hi"],
            "%d / %d" % (r["win"], r["lose"]),
            fmt_p(r["p"]), fmt_p(h),
        ]
        want_line = re.compile(re.escape(label) + r"[^\n]*")
        for rel, text in blob.items():
            hit = None
            for line in text.split("\n"):
                if want_line.search(line) and ("| %d |" % r["n"]) in line:
                    hit = line
                    break
            if hit is None:
                fails.append("stat row missing from %s (no line with %r and n = %d): %s"
                             % (rel, label, r["n"], label))
                continue
            for token in need:
                if token not in hit:
                    fails.append("%s: %r not on the %s row" % (rel, token, label))
    if not want:
        fails.append("no stat rows produced (nothing compared)")
    return fails, want, n


# --------------------------------------------------------------------------- #
# (B) manuscript mirror check
# --------------------------------------------------------------------------- #
def check_mirror(root, docs=None, tables=None):
    """Compare each manuscript mirror table with the framework block of the same
    name, positionally (row order is fixed on the framework side by 118).

    Three assertions per table, so that neither "empty set" nor "labels drifted"
    can pass silently: the row counts are equal, the manuscript's label multiset
    equals the documented one, and every numeric cell matches in position.
    """
    fw = io.open(os.path.join(root, FRAMEWORK), encoding="utf-8").read()
    ms = io.open(os.path.join(root, MANUSCRIPT), encoding="utf-8").read()
    tmap = tables if tables is not None else MIRROR_TABLES
    fw_t = parse_tables(fw)
    ms_t = parse_tables(ms)
    fails = []
    compared = 0
    for name, spec in sorted(tmap.items()):
        if name not in fw_t:
            fails.append("framework block %s not found" % name)
            continue
        if name not in ms_t:
            fails.append("manuscript table %s not found (missing MS anchor?)" % name)
            continue
        _, fw_rows = fw_t[name]
        _, ms_rows = ms_t[name]
        want_labels = list(spec["rows"])
        if len(fw_rows) != len(want_labels):
            fails.append("%s: framework has %d data rows, mapping documents %d"
                         % (name, len(fw_rows), len(want_labels)))
            continue
        if len(ms_rows) != len(fw_rows):
            fails.append("%s: manuscript has %d data rows, framework has %d"
                         % (name, len(ms_rows), len(fw_rows)))
            continue
        got_labels = [l.strip() for l, _ in ms_rows]
        if sorted(got_labels) != sorted(want_labels):
            fails.append("%s: manuscript row labels differ from the mapping; "
                         "missing %s, unexpected %s"
                         % (name,
                            sorted(set(want_labels) - set(got_labels)),
                            sorted(set(got_labels) - set(want_labels))))
            continue
        for (fw_label, cells), (ms_label, mcells) in zip(fw_rows, ms_rows):
            compared += 1
            a = _nums(" ".join(cells))
            b = _nums(" ".join(mcells))
            if len(a) != len(b):
                fails.append("%s/%s: %d framework numbers vs %d manuscript numbers"
                             % (name, fw_label, len(a), len(b)))
                continue
            for x, y in zip(a, b):
                if abs(x - y) > 1e-9:
                    fails.append("%s/%s: framework %g vs manuscript %g"
                                 % (name, fw_label, x, y))
    if compared == 0:
        fails.append("mirror check compared 0 rows (nothing to compare)")
    return fails, compared


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def run_check(root=ROOT, verbose=True):
    fails = []
    f1, want, n = check_stat_rows(root, [MANUSCRIPT, FRAMEWORK])
    f2, compared = check_mirror(root)
    fails += f1 + f2
    if verbose:
        if fails:
            for f in fails:
                print("  FAIL  " + f)
        else:
            print("[121] --check: %d paired row(s) on n = %d sequences verified in "
                  "both documents; %d mirrored table row(s) verified"
                  % (len(want), n, compared))
    return fails


def run_emit(root=ROOT):
    want, n, diag = build_stat_rows(root)
    print("# authoritative rows for the §5.9 paired-test table (P1 convention,")
    print("# per sequence over all 13 degradation cells, n = %d, m = %d comparisons)"
          % (n, diag["n_comparisons"]))
    print("# guard 1 (protocol drift, like for like): the ablation runs' `dagrs`,")
    print("#   lifted into P1, vs the P1 grid's `dagrs` on %d instances; worst "
          "|d| = %.3e (tolerance %.0e)"
          % (diag["drift_n"], diag["drift_worst"], ABL_DRIFT_TOL))
    print("#   the lift is exercised, not vacuous: %d of the %d lifted rows carry a "
          "non-unit n_clip/n_present" % (diag["n_scaled"], 3 * diag["drift_n"]))
    for arm in sorted(diag["moved"]):
        d, tot = diag["moved"][arm]
        print("# guard 2 (arm switch fired): %s differs from A0 on %d/%d instances"
              % (arm, d, tot))
    print("| comparison | n | mean delta | median delta | Wilcoxon p | Holm | "
          "Cliff's delta | 95% CI | win / lose |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, r, h in want:
        _, a, b = key.split("|")
        print("| %s vs %s | %d | **%+.4f** | %+.4f | %s | %s | **%+.3f** | "
              "[%+.4f, %+.4f] | **%d / %d** |"
              % (a, b, r["n"], r["mean"], r["median"], fmt_p(r["p"]), fmt_p(h),
                 r["cliff"], r["lo"], r["hi"], r["win"], r["lose"]))


def run_self_test(root=ROOT):
    checks = []
    fw_path = os.path.join(root, FRAMEWORK)
    ms_path = os.path.join(root, MANUSCRIPT)
    fw = io.open(fw_path, encoding="utf-8").read()
    ms = io.open(ms_path, encoding="utf-8").read()
    tmpdir = os.path.join(root, "_scratch", "121_selftest")
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    fw_tmp_rel = os.path.relpath(os.path.join(tmpdir, "fw.md"), root)
    ms_tmp_rel = os.path.relpath(os.path.join(tmpdir, "ms.md"), root)

    # 0. an untouched pair of copies must PASS (null control -- a checker that
    #    always fires is as useless as one that never does)
    io.open(os.path.join(root, fw_tmp_rel), "w", encoding="utf-8").write(fw)
    io.open(os.path.join(root, ms_tmp_rel), "w", encoding="utf-8").write(ms)
    globals()["FRAMEWORK"], globals()["MANUSCRIPT"] = fw_tmp_rel, ms_tmp_rel
    try:
        clean = run_check(root, verbose=False)
        checks.append(("untouched copy passes", not clean))

        # 1. perturb one manuscript cell -> must fire
        m = re.search(r"\| clean \| 0\.6803 \|", ms)
        if not m:
            raise AssertionError("self-test cannot find the clean row to perturb")
        bad = ms[:m.start()] + "| clean | 0.6804 |" + ms[m.end():]
        io.open(os.path.join(root, ms_tmp_rel), "w", encoding="utf-8").write(bad)
        f = run_check(root, verbose=False)
        checks.append(("perturbing a manuscript cell fires",
                       any("manuscript 0.6804" in x or "manuscript 0.680" in x
                           for x in f)))

        # 2. perturb one framework cell -> must fire.  The perturbation must land
        #    inside the anchored block: `0.6803` for `clean` also occurs in the
        #    un-mirrored 4-row block of 118, and perturbing *that* would print a
        #    green self-test while leaving the checked block untouched.
        anchor_at = fw.find("P1-SYNC: table9all")
        if anchor_at < 0:
            raise AssertionError("self-test cannot find the table9all anchor")
        m = re.compile(r"\| clean \| 0\.6803 \| 0\.6830 \|").search(fw, anchor_at)
        if not m:
            raise AssertionError("self-test cannot find the framework clean row "
                                 "inside table9all")
        badfw = fw[:m.start()] + "| clean | 0.6805 | 0.6830 |" + fw[m.end():]
        if badfw == fw:
            raise AssertionError("framework perturbation was a no-op")
        io.open(os.path.join(root, ms_tmp_rel), "w", encoding="utf-8").write(ms)
        io.open(os.path.join(root, fw_tmp_rel), "w", encoding="utf-8").write(badfw)
        f = run_check(root, verbose=False)
        checks.append(("perturbing a framework cell fires",
                       any("framework 0.6805" in x for x in f)))

        # 3. removing the manuscript anchor -> must fire
        io.open(os.path.join(root, fw_tmp_rel), "w", encoding="utf-8").write(fw)
        io.open(os.path.join(root, ms_tmp_rel), "w", encoding="utf-8").write(
            ms.replace("<!-- MS-TABLE: table9all -->", ""))
        f = run_check(root, verbose=False)
        checks.append(("removing an MS anchor fires",
                       any("not found" in x for x in f)))

        # 4. statistic positive control: a planted +0.05 shift on one arm must
        #    move the family-gap mean by about +0.05.  Use the table's own merged
        #    instance set -- `paired_rows` now needs all five arms.
        inst, _ = table_instances(root)
        base, n = paired_rows(inst)
        shifted = {(a, s, o, d): (v + 0.05 if a == "sam2video" else v)
                   for (a, s, o, d), v in inst.items()}
        got, _ = paired_rows(shifted)
        got_map = {(a, b): r for a, b, r, _ in got}
        base_map = {(a, b): r for a, b, r, _ in base}
        d_shift = (got_map[("sam2video", "greedy")]["mean"]
                   - base_map[("sam2video", "greedy")]["mean"])
        checks.append(("planted +0.05 shift moves the gap by +0.05",
                       abs(d_shift - 0.05) < 1e-9))

        # 5. statistic null control: recomputation is deterministic
        again, _ = paired_rows(table_instances(root)[0])
        same = all(abs(r["mean"] - r2["mean"]) == 0
                   for (_, _, r, _), (_, _, r2, _) in zip(base, again))
        checks.append(("recomputation is deterministic", same))

        # 6. the two new rows rest on two guards, and each needs its own control.
        def raises(fn, needle):
            try:
                fn()
            except AssertionError as e:
                return needle in str(e)
            return False

        rows, nframes = _load_ablation(root)
        scale = _p1_scale(root)
        checks.append((
            "ablation load: 3 arms x 13 cells x 61 instances",
            len(rows) == 3 * N_CELLS * N_INSTANCES
            and all(len(set(k[3] for k in rows if k[0] == a)) == N_CELLS
                    for a in ("dagrs", "A1", "A3"))))

        # 6a. the lift is not a no-op.  Comparing the UN-lifted column against the
        #     P1 grid must exceed the tolerance -- this encodes the 2026-09-28 bug,
        #     where raw-vs-scaled was read as an 8.3e-02 "protocol drift".
        raw_d = max(abs(rows[k] - inst[k]) for k in rows if k[0] == "dagrs")
        checks.append(("comparing raw against P1 fires (the lift is not a no-op)",
                       raw_d > ABL_DRIFT_TOL))

        # 6b. null control: the real inputs must not fire
        checks.append(("the real ablation inputs pass both new guards",
                       not raises(lambda: ablation_diagnostics(root, inst),
                                  "protocol drift")
                       and not raises(lambda: ablation_diagnostics(root, inst),
                                      "not scored over the whole clip")))

        kdg = sorted(k for k in rows if k[0] == "dagrs")[0]

        # 6c. drift positive control: a planted 1e-3 shift in one dagrs value
        bad = dict(rows)
        bad[kdg] = bad[kdg] + 1e-3
        checks.append(("a planted 1e-3 shift in one ablation dagrs value fires",
                       raises(lambda: ablation_diagnostics(
                           root, inst, rows=bad, nframes=nframes, scale=scale),
                           "protocol drift")))

        # 6d. coverage control: an instance not scored over the whole clip
        badn = dict(nframes)
        badn[kdg] = badn[kdg] + 1
        checks.append(("an instance not scored over the whole clip fires",
                       raises(lambda: ablation_diagnostics(
                           root, inst, rows=rows, nframes=badn, scale=scale),
                           "not scored over the whole clip")))

        # 6e. arm-switch control: A1 made identical to A0 must fire, not tabulate 0
        flat = dict(rows)
        for k in list(flat):
            if k[0] == "A1":
                flat[k] = flat[("dagrs", k[1], k[2], k[3])]
        checks.append(("an arm identical to A0 fires instead of tabulating a zero",
                       raises(lambda: ablation_diagnostics(
                           root, inst, rows=flat, nframes=nframes, scale=scale),
                           "identical to A0")))
    finally:
        globals()["FRAMEWORK"], globals()["MANUSCRIPT"] = FRAMEWORK, MANUSCRIPT

    print("[121] === --self-test ===")
    ok = True
    for name, good in checks:
        print("  %-46s %s" % (name, "PASS" if good else "FAIL"))
        ok = ok and good
    return 0 if ok else 1


def main(argv):
    status, msg = release_gate("121", ROOT,
                               [MANUSCRIPT, FRAMEWORK] + [s for s, _ in ABL_SOURCES],
                               "the section-5 table audit")
    if status != "ok":
        print(msg)
        return 1 if status == "fail" else 0
    if "--self-test" in argv:
        return run_self_test()
    if "--emit" in argv:
        run_emit()
        return 0
    if "--check" in argv:
        return 1 if run_check() else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
