# -*- coding: utf-8 -*-
"""120 - CL10 / TABLE-16 owner: the vacuity<->error association, on the unit it is
claimed about.

Why this script exists (red line #15: every number in the text needs an owner)
------------------------------------------------------------------------------
`05_analysis.py --study vacuity` produces ONE number, and it is the number the
section would like to quote:

    Spearman(vacuity, 1 - JF_frame) = +0.2487  [+0.216, +0.281]  p=6.37e-47  n=3244

Two things make that number unsafe to quote as-is, and both need a script that
keeps checking them after any future change:

  1. `n = 3244` is FRAMES from 244 object instances inside 120 cells.  Frames in a
     cell share the sequence, the degradation realisation, the object and the memory
     bank, so they are not independent.  This project has already published one
     consequence of getting the paired unit wrong (PAPER_FRAMEWORK 5.9: a family-gap
     p-value inflated by ~50 orders of magnitude).

  2. The CI is a percentile bootstrap that resamples FRAMES.  Its width is 0.37x the
     cell-clustered width -- i.e. it behaves as if there were ~544 independent
     observations instead of 244 instances / 120 cells.

So this script re-derives the claim from `results/dagrs_main_raw.csv` and reports it
on four axes (reproducibility / unit / decomposition / calibration), then keeps the
document in sync via `--check`.

Calibration is not optional (red line #9): every interval and every test quoted here
is measured against a synthetic null whose TRUE pooled rho is 0 by construction
(i.i.d. nu per frame, error column and grouping untouched), N draws, and the empirical
false-positive rate is asserted rather than assumed.  Two planted controls make both
tests prove their own discriminating power:

    P[within]  nu = error centred inside each instance + independent per-instance
               jitter  -> WITHIN test fires, BETWEEN test must stay silent.
               (NOT nu = error + noise: that also lifts the instance means and would
                fire BOTH tests, i.e. would prove nothing.)
    P[between] nu = the instance's own mean error, constant within instance
               -> BETWEEN test fires, WITHIN test must stay silent.

Modes
-----
    --audit      full report (default; the exploratory view)
    --emit       print the canonical TABLE-16 block, numbers filled in
    --check      recompute and verify every quoted number appears in the documents
    --self-test  planted controls + prove --check FIRES on a perturbed document

Usage
-----
    python scripts/120_cl10_rho_audit.py --audit
    python scripts/120_cl10_rho_audit.py --check
    python scripts/120_cl10_rho_audit.py --self-test
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import random
import re
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    # the PROJECT's own frame-level routine -- imported so the frame CI quoted in the
    # documents is byte-for-byte the one `05_analysis.py --study vacuity` prints,
    # instead of a look-alike produced by a second implementation.
    from xdrp.stats import spearman_with_ci as _prod_spearman_ci
except Exception:                                    # pragma: no cover
    _prod_spearman_ci = None

TARGET = "results/dagrs_main_raw.csv"

DOCS = ["docs/PAPER_FRAMEWORK.md", "docs/MANUSCRIPT.md"]
#: the block this script owns in the documents
ANCHOR = "<!-- CL10-SYNC -->"

#: pinned so that --check is deterministic (the quantities are stochastic by nature)
N_BOOT = 2000
N_PERM = 2000
N_NULL = 100
N_NULL_BOOT = 300
N_NULL_PERM = 1000
DROP = 10

LEVEL_KEY = {"instance": 0, "cell": 1}


# --------------------------------------------------------------------------- #
# data + statistics
# --------------------------------------------------------------------------- #

def load(path):
    with io.open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def spearman(x, y):
    """Rank -> Pearson, with scipy rankdata (independent of 05_analysis.py)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def observations(rows):
    """[(instance_key, cell_key, nu, err)] over the `dagrs` decision frames."""
    out = []
    for r in rows:
        if not str(r.get("mode", "")).startswith("dagrs"):
            continue
        if str(r.get("is_keyframe", "0")) not in ("1", "1.0"):
            continue
        nu = fnum(r.get("vacuity"))
        jf = fnum(r.get("JF_frame", r.get("J&F", "")))
        if not (np.isfinite(nu) and np.isfinite(jf)):
            continue
        inst = (str(r.get("seq")), str(r.get("degradation")),
                str(r.get("level")), str(r.get("obj_id")))
        out.append((inst, inst[:3], nu, 1.0 - jf))
    return out


def group(obs, level):
    i = LEVEL_KEY[level]
    g = collections.defaultdict(list)
    for o in obs:
        g[o[i]].append((o[2], o[3]))
    return g


def boot_ci(obs, level, n_boot, seed=0):
    """Resample the declared unit.  Frames resample (nu, err) PAIRS -- drawing two
    independent index vectors silently yields rho ~ 0 and a meaningless tight CI."""
    rng = random.Random(seed)
    if level == "frame":
        pairs = [(o[2], o[3]) for o in obs]
        n = len(pairs)
        vals = []
        for _ in range(n_boot):
            pick = [pairs[rng.randrange(n)] for _ in range(n)]
            vals.append(spearman([p[0] for p in pick], [p[1] for p in pick]))
    else:
        g = group(obs, level)
        keys = sorted(g)
        vals = []
        for _ in range(n_boot):
            a, b = [], []
            for k in (keys[rng.randrange(len(keys))] for _ in range(len(keys))):
                for x, y in g[k]:
                    a.append(x)
                    b.append(y)
            vals.append(spearman(a, b))
    v = np.asarray([z for z in vals if np.isfinite(z)])
    if not len(v):
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def unit_means(obs, level, drop=1):
    g = group(obs, level)
    nus, errs = [], []
    for k in sorted(g):
        if len(g[k]) < drop:
            continue
        nus.append(np.mean([p[0] for p in g[k]]))
        errs.append(np.mean([p[1] for p in g[k]]))
    return np.asarray(nus), np.asarray(errs)


def between(obs, level, n_perm, seed=0):
    nus, errs = unit_means(obs, level, 1)
    if len(nus) < 5:
        return float("nan"), float("nan"), 0
    rho = spearman(nus, errs)
    rng = random.Random(seed)
    null = np.asarray([spearman(rng.sample(list(nus), len(nus)), errs)
                       for _ in range(n_perm)])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1))
    return rho, p, len(nus)


def within(obs, level, drop):
    g = group(obs, level)
    out = []
    for k in sorted(g):
        if len(g[k]) < drop:
            continue
        v = spearman([p[0] for p in g[k]], [p[1] for p in g[k]])
        if np.isfinite(v):
            out.append(v)
    return np.asarray(out)


def within_p(v):
    if len(v) < 2 or len(set(np.round(v, 12))) < 2:
        return float("nan")
    return float(wilcoxon(v).pvalue)


def plant(obs, mode, seed=0):
    """Rewrite nu only; the error column and all grouping stay intact."""
    rng = random.Random(seed)
    er = np.array([o[3] for o in obs], dtype=float)
    if mode == "null":
        nu = np.array([rng.random() for _ in obs])
    elif mode == "within":
        g = group(obs, "instance")
        mu = {k: float(np.mean([p[1] for p in g[k]])) for k in g}
        jit = {k: rng.gauss(0.0, 1.0) for k in g}
        s = er.std() or 1.0
        nu = np.array([(er[i] - mu[obs[i][0]]) / s + jit[obs[i][0]]
                       + rng.gauss(0, 0.05) for i in range(len(obs))])
    elif mode == "between":
        g = group(obs, "instance")
        m = {k: float(np.mean([p[1] for p in g[k]])) for k in g}
        nu = np.array([m[o[0]] for o in obs])
    else:
        raise ValueError(mode)
    return [(o[0], o[1], float(nu[i]), o[3]) for i, o in enumerate(obs)]


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #

def compute(csv_path=None, verbose=True, calibrate=True):
    path = csv_path or os.path.join(ROOT, TARGET)
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} not found")
    obs = observations(load(path))
    if len(obs) < 50:
        raise SystemExit(f"FATAL: only {len(obs)} usable observations in {path}")

    m = {}
    m["n_frames"] = len(obs)
    m["n_instances"] = len(set(o[0] for o in obs))
    m["n_cells"] = len(set(o[1] for o in obs))
    m["rho"] = spearman([o[2] for o in obs], [o[3] for o in obs])

    # the frame interval the documents must quote = the production one (10000 reps)
    nus = [o[2] for o in obs]
    errs = [o[3] for o in obs]
    if _prod_spearman_ci is None:
        raise SystemExit("FATAL: xdrp.stats.spearman_with_ci not importable; run from "
                         "the repository root")
    prod = _prod_spearman_ci(nus, errs, n_boot=10000, seed=0)
    if abs(prod["rho"] - m["rho"]) > 1e-12:
        raise SystemExit(f"FATAL: production rho {prod['rho']!r} != independent rho "
                         f"{m['rho']!r}; the two implementations disagree on the data")
    m["ci_frame"] = (prod["lo"], prod["hi"])
    m["p_prod"] = prod["p"]
    own_frame = boot_ci(obs, "frame", N_BOOT, seed=7)     # cross-check, not quoted
    m["ci_frame_own"] = own_frame
    m["ci_instance"] = boot_ci(obs, "instance", N_BOOT, seed=7)
    m["ci_cell"] = boot_ci(obs, "cell", N_BOOT, seed=7)
    wf = m["ci_frame"][1] - m["ci_frame"][0]
    wi = m["ci_instance"][1] - m["ci_instance"][0]
    wc = m["ci_cell"][1] - m["ci_cell"][0]
    m["ratio_inst"] = wf / wi
    m["ratio_cell"] = wf / wc
    m["n_eff"] = int(round(m["n_frames"] * m["ratio_inst"] ** 2))

    m["between_instance"], m["p_between_instance"], m["n_between_instance"] = \
        between(obs, "instance", N_PERM, seed=11)
    m["between_cell"], m["p_between_cell"], m["n_between_cell"] = \
        between(obs, "cell", N_PERM, seed=11)
    vw_i = within(obs, "instance", DROP)
    vw_c = within(obs, "cell", DROP)
    m["within_instance"] = float(vw_i.mean())
    m["within_instance_pos"] = float(np.mean(vw_i > 0))
    m["p_within_instance"] = within_p(vw_i)
    m["n_within_instance"] = len(vw_i)
    m["within_cell"] = float(vw_c.mean())
    m["within_cell_pos"] = float(np.mean(vw_c > 0))
    m["p_within_cell"] = within_p(vw_c)
    m["n_within_cell"] = len(vw_c)

    # The threshold sweep that `--study vacuity` writes alongside the correlation is
    # quoted with it, so its sample sizes are owned here too: a "better tau" read off
    # 7 frames is exactly the kind of unowned number this script exists to prevent.
    sweep = os.path.join(ROOT, "results", "vacuity_threshold_sweep.csv")
    if os.path.exists(sweep):
        hi = [int(float(r["n_high"])) for r in load(sweep)]
        m["sweep_n_high_min"] = min(hi)
        m["sweep_n_high_max"] = max(hi)
    else:
        m["sweep_n_high_min"] = -1
        m["sweep_n_high_max"] = -1

    # calibration: true pooled rho = 0 by construction.  `calibrate=False` skips it
    # (the --self-test path), which is why CLAIMS is split into light/calibrated sets.
    if not calibrate:
        for lvl in ("frame", "instance", "cell"):
            m[f"fp_ci_{lvl}"] = float("nan")
            m[f"fp_perm_{lvl}"] = float("nan")
            m[f"fp_within_{lvl}"] = float("nan")
        m["null_realised"] = float("nan")
        return m

    fp_ci = {"frame": 0, "instance": 0, "cell": 0}
    fp_perm = {"instance": 0, "cell": 0}
    fp_within = {"instance": 0, "cell": 0}
    n_w = {"instance": 0, "cell": 0}
    realised = []
    for t in range(N_NULL):
        nb = plant(obs, "null", seed=2000 + t)
        realised.append(spearman([o[2] for o in nb], [o[3] for o in nb]))
        for lvl in ("frame", "instance", "cell"):
            lo, hi = boot_ci(nb, lvl, N_NULL_BOOT, seed=t)
            if np.isfinite(lo) and (lo > 0 or hi < 0):
                fp_ci[lvl] += 1
        for lvl in ("instance", "cell"):
            _, p, _ = between(nb, lvl, N_NULL_PERM, seed=500 + t)
            if np.isfinite(p) and p < 0.05:
                fp_perm[lvl] += 1
            v = within(nb, lvl, DROP)
            if len(v) > 1:
                n_w[lvl] += 1
                p = within_p(v)
                if np.isfinite(p) and p < 0.05:
                    fp_within[lvl] += 1
    m["null_realised"] = float(np.mean(realised))
    for lvl in ("frame", "instance", "cell"):
        m[f"fp_ci_{lvl}"] = fp_ci[lvl] / N_NULL
    for lvl in ("instance", "cell"):
        m[f"fp_perm_{lvl}"] = fp_perm[lvl] / N_NULL
        m[f"fp_within_{lvl}"] = (fp_within[lvl] / n_w[lvl]) if n_w[lvl] else float("nan")

    if verbose:
        report(m)
    return m


def report(m):
    print(f"[120] arm        : single `dagrs` (compound C1-C4 @ level 3.0)")
    print(f"[120] units      : {m['n_frames']} decision frames / {m['n_instances']} "
          f"instances / {m['n_cells']} cells")
    print(f"[120] Q1 pooled rho = {m['rho']:+.4f}  (both implementations agree; "
          f"production p = {m['p_prod']:.2e})")
    print(f"[120] Q2 CI frame(prod, quoted) [{m['ci_frame'][0]:+.4f}, "
          f"{m['ci_frame'][1]:+.4f}]  [own 2000-rep {m['ci_frame_own'][0]:+.4f}, "
          f"{m['ci_frame_own'][1]:+.4f}]")
    print(f"[120]    CI instance [{m['ci_instance'][0]:+.4f}, {m['ci_instance'][1]:+.4f}]")
    print(f"[120]    CI cell     [{m['ci_cell'][0]:+.4f}, {m['ci_cell'][1]:+.4f}]")
    print(f"[120]    width ratio frame/instance={m['ratio_inst']:.2f}x "
          f"frame/cell={m['ratio_cell']:.2f}x -> n_eff ~ {m['n_eff']}")
    print(f"[120] Q3 between-instance rho={m['between_instance']:+.4f} "
          f"p={m['p_between_instance']:.4f} n={m['n_between_instance']}")
    print(f"[120]    between-cell     rho={m['between_cell']:+.4f} "
          f"p={m['p_between_cell']:.4f} n={m['n_between_cell']}")
    print(f"[120]    within-instance  mean rho={m['within_instance']:+.4f} "
          f"share>0={m['within_instance_pos']:.1%} n={m['n_within_instance']} "
          f"p={m['p_within_instance']:.3g}")
    print(f"[120]    within-cell      mean rho={m['within_cell']:+.4f} "
          f"share>0={m['within_cell_pos']:.1%} n={m['n_within_cell']} "
          f"p={m['p_within_cell']:.3g}")
    print(f"[120] Q4 calibration (true rho=0, {N_NULL} draws, "
          f"realised {m['null_realised']:+.4f})")
    print(f"[120]    CI FP   frame {m['fp_ci_frame']:.0%}  instance "
          f"{m['fp_ci_instance']:.0%}  cell {m['fp_ci_cell']:.0%}   (nominal 5%)")
    print(f"[120]    perm FP instance {m['fp_perm_instance']:.0%}  "
          f"cell {m['fp_perm_cell']:.0%}")
    print(f"[120]    wilcox FP instance {m['fp_within_instance']:.0%}  "
          f"cell {m['fp_within_cell']:.0%}")


# --------------------------------------------------------------------------- #
# emit / check
# --------------------------------------------------------------------------- #

#: (label, format template) -- every value the documents are allowed to quote
CLAIMS = [
    ("rho", "**ρ = {rho:+.4f}**"),
    ("p_prod", "p = **{p_prod:.2e}**"),
    ("n_frames", "**{n_frames}** 个决策帧"),
    ("ci_frame", "`[{ci_frame[0]:+.3f}, {ci_frame[1]:+.3f}]`"),
    ("ci_instance", "`[{ci_instance[0]:+.3f}, {ci_instance[1]:+.3f}]`"),
    ("ci_cell", "`[{ci_cell[0]:+.3f}, {ci_cell[1]:+.3f}]`"),
    ("n_cells", "{n_cells} 格"),
    ("n_instances", "{n_instances} 实例"),
    ("ratio_inst", "**{ratio_inst:.2f}×**"),
    ("ratio_cell", "**{ratio_cell:.2f}×**"),
    ("n_eff", "等效样本量 ≈ **{n_eff}**"),
    ("fp_ci_frame", "帧 **{fp_ci_frame:.0%}**"),
    ("fp_ci_instance", "实例 **{fp_ci_instance:.0%}**"),
    ("fp_ci_cell", "格 **{fp_ci_cell:.0%}**"),
    ("between_cell", "格级聚合 ρ = **{between_cell:+.4f}**"),
    ("p_between_cell", "置换检验 p = **{p_between_cell:.4f}**"),
    ("n_between_cell", "n = {n_between_cell} 格"),
    ("fp_perm_cell", "假阳性率 {fp_perm_cell:.0%}"),
    ("within_cell", "格内平均 ρ = **{within_cell:+.4f}**"),
    ("p_within_cell", "Wilcoxon p = **{p_within_cell:.1e}**"),
    ("n_within_cell", "n = {n_within_cell} 格"),
    ("fp_within_cell", "假阳性率 {fp_within_cell:.0%}"),
    ("between_instance", "格间 ρ = {between_instance:+.4f}（n = {n_between_instance}）"),
    ("within_instance", "格内平均 ρ = {within_instance:+.4f}"),
    ("within_instance_pos", "{within_instance_pos:.1%} 的实例为正"),
    ("sweep_upper_tail", "上尾样本量 {sweep_n_high_min}–{sweep_n_high_max}"),
]

#: the claims that do NOT depend on the null simulation, so `--self-test` can check
#: the checking machinery without paying for the 100-draw calibration.
CALIBRATED_CLAIMS = {"fp_ci_frame", "fp_ci_instance", "fp_ci_cell",
                     "fp_perm_cell", "fp_within_cell"}

#: The manuscript is English, so its sentences cannot contain the Chinese claim
#: strings above.  Without this set the manuscript would sail through `--check` on the
#: strength of the framework file alone -- i.e. the document a reviewer reads would be
#: the one document nobody checked.
MS_CLAIMS = [
    ("rho_en", "rho = {rho:+.4f}"),
    ("p_en", "p = {p_prod:.2e}"),
    ("ci_cell_en", "[{ci_cell[0]:+.3f}, {ci_cell[1]:+.3f}]"),
    ("n_eff_en", "about {n_eff}"),
    ("between_cell_en", "rho = {between_cell:+.4f}"),
    ("p_between_cell_en", "p = {p_between_cell:.4f}"),
    ("within_cell_en", "rho = {within_cell:+.4f}"),
    ("p_within_cell_en", "p = {p_within_cell:.1e}"),
]


def emit(m):
    print(ANCHOR)
    print(f"| CL10 / TABLE-16 | 值 |")
    print(f"|---|---|")
    for name, tpl in CLAIMS:
        print(f"| `{name}` | {tpl.format(**m)} |")


def _strip_retracted(text):
    return re.sub(r"<!--\s*RETRACTED-OK\s*-->.*?<!--\s*/RETRACTED-OK\s*-->", "",
                  text, flags=re.S)


def _norm(s):
    """U+2212 -> ASCII hyphen, so the typography of the doc does not matter."""
    return s.replace("\u2212", "-")


def block_of(text):
    """The text this script owns: from the anchor to the next markdown heading.

    Searching the WHOLE document instead would let a value that is correct in another
    context (red line #13: the same digits mean different things in different tables)
    silently satisfy the check.
    """
    i = text.find(ANCHOR)
    if i < 0:
        return None
    rest = text[i + len(ANCHOR):]
    m = re.search(r"\n#{1,3} ", rest)
    return rest[:m.start()] if m else rest


def check(m, verbose=True, docs=None, names=None):
    """Every claim string must appear inside this script's anchored block.

    `names` restricts the check to a subset (used by --self-test, which runs without
    the calibration and therefore has no FP rates to compare).
    """
    if docs is None:
        docs = DOCS
    claims = [(n, t) for n, t in CLAIMS if names is None or n in names]
    fails = []
    texts = {}
    for rel in docs:
        p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        texts[rel] = _strip_retracted(io.open(p, encoding="utf-8").read()) \
            if os.path.exists(p) else None
    present = [r for r, t in texts.items() if t is not None]
    if not present:
        return ["no document found at " + ", ".join(docs)]
    blocks = {}
    for r, t in texts.items():
        if t is None:
            continue
        b = block_of(t)
        if b is not None:
            blocks[r] = b
    if not blocks:
        fails.append(f"docs: anchor {ANCHOR} not found in any of {present}; the "
                     f"checked block has nothing to attach to")
        if verbose:
            for f in fails:
                print(f"  FAIL  {f}")
        return fails

    blob = _norm("\n".join(blocks.values()))
    for name, tpl in claims:
        want = _norm(tpl.format(**m))
        if want not in blob:
            fails.append(f"{name}: {want!r} not found in the anchored block")

    # the manuscript must carry its OWN copy of the headline numbers
    ms_blocks = {r: b for r, b in blocks.items() if "MANUSCRIPT" in r.upper()}
    if ms_blocks and names is None:
        for name, tpl in MS_CLAIMS:
            want = _norm(tpl.format(**m))
            if not any(want in _norm(b) for b in ms_blocks.values()):
                fails.append(f"manuscript {name}: {want!r} not found in the "
                             f"manuscript CL10 block")
    if verbose:
        if fails:
            for f in fails:
                print(f"  FAIL  {f}")
        else:
            print(f"[120] --check: {len(claims)} claim(s) verified in "
                  f"{', '.join(sorted(blocks))}")
    return fails


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def self_test():
    ok = True
    print("[120] === planted controls (each must fire only where planted) ===")
    obs = observations(load(os.path.join(ROOT, TARGET)))

    pw = plant(obs, "within", seed=17)
    rb, pb, _ = between(pw, "instance", 500, seed=2)
    vw = within(pw, "instance", DROP)
    wm = float(vw.mean())
    fires_b, fires_w = pb < 0.05, wm > 0.05
    print(f"  P[within]  BETWEEN rho={rb:+.4f} p={pb:.3f} -> "
          f"{'FIRES' if fires_b else 'silent'}  (expected silent)")
    print(f"  P[within]  WITHIN  mean rho={wm:+.4f} -> "
          f"{'FIRES' if fires_w else 'silent'}  (expected FIRES)")
    if fires_b or not fires_w:
        ok = False
        print("  -> FAIL: a within-only signal must not reach the between test, and "
              "must be caught by the within test")

    pb_ = plant(obs, "between", seed=17)
    rb2, pb2, _ = between(pb_, "instance", 500, seed=2)
    vw2 = within(pb_, "instance", DROP)
    wm2 = float(vw2.mean()) if len(vw2) else float("nan")
    fires_b2 = pb2 < 0.05
    fires_w2 = np.isfinite(wm2) and wm2 > 0.05
    print(f"  P[between] BETWEEN rho={rb2:+.4f} p={pb2:.3f} -> "
          f"{'FIRES' if fires_b2 else 'silent'}  (expected FIRES)")
    print(f"  P[between] WITHIN  mean rho={wm2:+.4f} -> "
          f"{'FIRES' if fires_w2 else 'silent'}  (expected silent)")
    if not fires_b2 or fires_w2:
        ok = False
        print("  -> FAIL: a between-only signal must be caught by the between test "
              "and must not reach the within test")

    print("\n[120] === --check must FIRE on a perturbed document ===")
    # only claims that do not need the null simulation: --self-test must stay cheap
    # enough to live in the smoke suite.
    names = ["rho", "ci_cell", "between_cell"]
    if set(names) & CALIBRATED_CLAIMS:
        print("  FAIL: a calibrated claim was put in the cheap subset")
        return 1
    m = compute(verbose=False, calibrate=False)
    tmp = os.path.join(ROOT, "_scratch", "120_selftest_doc.md")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    src = os.path.join(ROOT, DOCS[0])
    text = io.open(src, encoding="utf-8").read()
    with io.open(tmp, "w", encoding="utf-8") as fh:      # first: unchanged copy
        fh.write(text)
    rel = os.path.relpath(tmp, ROOT)
    f_clean = check(m, verbose=False, docs=[rel], names=names)
    # a copy with the anchor removed must also fire (proves the anchor is load-bearing)
    noanchor = text.replace(ANCHOR, "")
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(noanchor)
    f_noanchor = check(m, verbose=False, docs=[rel], names=names)
    # one number perturbed inside the anchored block must be caught.  Two separate
    # perturbations, because a check can be blind to one kind and not the other.
    want = CLAIMS[0][1].format(**m)
    wrong = want.replace(f"{m['rho']:+.4f}", f"{m['rho'] + 0.05:+.4f}")
    assert wrong != want
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text.replace(want, wrong))
    f_perturbed = check(m, verbose=False, docs=[rel], names=names)
    want2 = dict(CLAIMS)["ci_cell"].format(**m)
    wrong2 = want2.replace(f"{m['ci_cell'][1]:+.3f}", f"{m['ci_cell'][1] + 0.01:+.3f}")
    assert wrong2 != want2
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text.replace(want2, wrong2))
    f_perturbed2 = check(m, verbose=False, docs=[rel], names=names)
    # Deliberately NOT deleted.  An environment-level safe-delete guard budgets deletions per turn
    # and *terminates the process* when the budget runs out -- observed here as
    # SAFE_DELETE_BULK_CONFIRM_REQUIRED targeting exactly this file, once the earlier tests in the
    # smoke suite had spent the turn's budget.  `try/except OSError` cannot catch a termination, so
    # this self-test used to die *after* all four planted controls had run but *before* it printed
    # them: the guard looked broken while the code under test was fine, and the smoke suite scored
    # 5 failures off one refused unlink.
    # The path is fixed, so it is overwritten on every run and cannot accumulate.  `114` reaches the
    # same place from the other side: it keeps cleanup best-effort *and* gives each run its own
    # directory, so leftovers never turn into a verdict.
    # Same-discipline note for `--check`: read-only, so it stays safe in a restricted environment.
    if os.environ.get("DAGRS_CLEAN_SCRATCH") == "1":
        try:
            os.remove(tmp)
        except OSError:
            pass

    checks = [
        ("unchanged copy passes", not f_clean),
        ("removing the anchor fires", bool(f_noanchor)),
        ("perturbing rho fires", bool(f_perturbed)),
        ("perturbing a CI bound fires", bool(f_perturbed2)),
    ]
    for name, good in checks:
        print(f"  {name:<26} {'ok' if good else 'NOT DETECTED'}")
        if not good:
            ok = False
    if f_noanchor:
        print(f"    (anchor-less copy reported: {f_noanchor[0][:80]})")
    if f_perturbed:
        print(f"    (perturbed copy reported: {f_perturbed[0][:80]})")

    print(f"\n[120] VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="full report (default)")
    ap.add_argument("--emit", action="store_true", help="print the TABLE-16 block")
    ap.add_argument("--check", action="store_true", help="verify the documents")
    ap.add_argument("--self-test", action="store_true", help="planted controls")
    ap.add_argument("--csv", default=None, help="override the input CSV")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.emit:
        emit(compute(csv_path=args.csv, verbose=False))
        return 0
    m = compute(csv_path=args.csv, verbose=not args.check)
    if args.check:
        fails = check(m)
        print(f"[120] VERDICT: {'FAIL' if fails else 'PASS'} "
              f"({len(fails)} problem(s))")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
