"""98 - DAG-RS gate audit: is the re-anchoring trigger reachable AND informative?

    python scripts/98_vac_gate_audit.py                       # default corpora
    python scripts/98_vac_gate_audit.py path/to/probe*.csv

READ-ONLY, NO GPU. It consumes CSVs that `03_run_dagrs.py` already wrote and
answers the three questions that decide whether the `dagrs` arm is measuring
the mechanism it claims to measure:

  (1) REACHABILITY -- the gate fires only on `vacuity >= tau_vacuity`. We
      histogram the observed vacuity on decision frames and report the hit rate
      for a range of candidate thresholds, plus the `n_reanchors` counter
      itself. If `n_reanchors` is 0 everywhere, the `dagrs` arm is numerically
      the same arm as `dagrs_no_reanchor` and any "gain" attributed to
      re-anchoring is spurious.

  (2) STRUCTURAL REACHABILITY -- a hit rate says nothing about whether the
      CONSECUTIVE requirement can ever be met, because the counter and the gate
      can live on different clocks. Until 2026-09-17 `pipeline.run_sequence`
      evaluated the gate only on decision frames (every K-th frame) but reset
      `consec_uncertain` on *every accepted frame* -- and every non-decision
      frame is accepted (the dagrs family leaves `tau_score = 0`, which disables
      the only other rejection path). The counter could therefore only reach 1
      and `max_consecutive_uncertain >= 2` was unreachable **for any tau**, which
      is why `n_reanchors` was 0 in all 28 cells of the A3 probe.
      That is fixed: the counter now lives on the decision grid, like the gate
      that drives it. This script keeps BOTH readings -- it still measures the
      legacy cadence (for the record, and to keep the old diagnosis checkable)
      and reports whether the shipped cadence can now reach the threshold.
      A `max run` below `max_consecutive_uncertain` means the mechanism cannot
      fire however the threshold is moved.
      We reproduce the pipeline's counter exactly and *validate it against the
      `n_rejected` column the pipeline itself wrote* -- a known-positive check
      that the simulation is faithful before any verdict is printed.

  (3) INFORMATIVENESS -- even a reachable gate is useless if the statistic
      carries no information. We pair each decision frame's `vacuity` with its
      real error (1 - J_frame) and report Spearman rho. The usable direction is
      rho > 0 (high uncertainty == high error). A rho near 0 means the
      threshold is untunable, not merely mistuned -- a different conclusion.

CLOSED FORM, for telling "mistuned" apart from "structurally impossible":
    e_m = kappa * A_m^gamma ; S = sum(alpha) ; vacuity = M / S
so vacuity is a function of the MEAN candidate-vs-anchor agreement A:
    A=0.90 -> 0.134 | A=0.70 -> 0.203 | A=0.50 -> 0.333 | A=0.32 -> 0.550
Reaching the shipped 0.55 therefore requires mean agreement <= 0.32, i.e. the
mask is already close to collapse. Note the counter-intuitive consequence:
COLLAPSE IS A HIGH-AGREEMENT STATE -- all operators share the same (already
wrong) prompt, so they fail identically and disagree little. We print this band
so the two situations are never confused.

The Spearman routine is hand-rolled, so it is SELF-VALIDATED before any verdict
is printed (project convention: never trust a probe that has not reproduced a
known-positive synthetic case).
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_shipped():
    """(tau, max_consec, K) as shipped, read from the config, not hardcoded.

    Hardcoding these is exactly how a probe drifts away from the pipeline it is
    supposed to describe, so they are loaded from the single config entry point.
    """
    try:
        import yaml
        d = yaml.safe_load(open(ROOT / "configs" / "default.yaml",
                                encoding="utf-8"))["dagrs"]
        return (float(d["tau_vacuity"]),
                int(d["max_consecutive_uncertain"]),
                int(d["keyframe_stride"]))
    except Exception:                                   # pragma: no cover
        return 0.55, 2, 5


TAU_SHIPPED, MAX_CONSEC, K = read_shipped()
KAPPA, GAMMA = 8.0, 2.0

CANDIDATE_TAUS = (0.10, 0.13, 0.15, 0.18, 0.22, 0.25, 0.30, 0.40, 0.55)


def spearman(xs, ys) -> float:
    """Rank correlation, average ranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    n = len(xs)
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def self_test() -> bool:
    """Known-positive synthetic cases. Returns True only if ALL pass."""
    ok = True
    mono = [1.0, 2.0, 3.0, 4.0, 5.0]
    checks = [
        ("perfectly increasing -> +1.000", spearman(mono, mono), 1.0),
        ("perfectly decreasing -> -1.000", spearman(mono, list(reversed(mono))), -1.0),
        ("increasing with ties   -> +1.000",
         spearman([1, 1, 2, 2, 3], [1, 1, 2, 2, 3]), 1.0),
    ]
    print("probe self-validation (hand-rolled Spearman):")
    for name, got, want in checks:
        good = abs(got - want) < 1e-9
        ok = ok and good
        print(f"  [{'ok ' if good else 'FAIL'}] {name}: got {got:+.6f} want {want:+.6f}")
    return ok


def load(path):
    """-> (vacuity, 1 - J_frame) on dagrs decision frames, plus n_reanchors map."""
    pts, reanch = [], {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mode"] != "dagrs":
                continue
            reanch.setdefault((r["seq"], r["degradation"]), r["n_reanchors"])
            if r["frame"] == "" or r["is_keyframe"] != "1":
                continue
            try:
                v, j = float(r["vacuity"]), float(r["J_frame"])
            except (TypeError, ValueError):
                continue
            if v != v or j != j:       # nan
                continue
            pts.append((v, 1.0 - j))
    return pts, reanch


def load_frames(path):
    """-> {(degradation, seq, obj): [(t, is_keyframe, vacuity)]} (per-frame rows).

    Reconstructs the FULL frame timeline, non-decision frames included. That is
    the whole point: the counter lives on the full timeline while the gate lives
    on the decision grid, and that mismatch is the defect this script measures.
    """
    cells = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mode"] != "dagrs" or r["frame"] == "":
                continue
            try:
                t = int(r["frame"])
            except (TypeError, ValueError):
                continue
            try:
                v = float(r["vacuity"])
            except (TypeError, ValueError):
                v = float("nan")
            key = (r["degradation"], r["seq"], r["obj_id"])
            cells.setdefault(key, []).append((t, r["is_keyframe"] == "1", v))
    for k in cells:
        cells[k].sort()
    return cells


def load_counters(path, col):
    """-> {(degradation, seq, obj): <col>} for the dagrs arm."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mode"] != "dagrs":
                continue
            out.setdefault((r["degradation"], r["seq"], r["obj_id"]), r[col])
    return out


def simulate(rows, tau, max_consec, mode="shipped"):
    """Count re-anchor events under one reading of `consec_uncertain`.

    mode="shipped" -- the counter lives on the decision grid, like the gate that
                      drives it: it advances on a rejected DECISION frame and
                      resets on an accepted one. This is what
                      `pipeline.run_sequence` does since the 2026-09-17 fix.
    mode="legacy"  -- reset on EVERY accepted frame (the pre-fix behaviour).
                      Non-decision frames are always accepted because the dagrs
                      family leaves `tau_score = 0`, disabling the score gate, so
                      the counter could not exceed 1 and any threshold >= 2 was
                      unreachable for any tau.

    Returns (events, rejections, max_run_seen). `rejections` is what makes the
    simulation checkable against the `n_rejected` column the pipeline wrote.
    """
    consec = ev = rej = mx = 0
    for t, kf, v in rows:
        flagged = bool(kf) and v == v and v >= tau
        if mode == "legacy":
            if not flagged:
                consec = 0
                continue
        else:
            if not kf:
                continue
            if not flagged:
                consec = 0
                continue
        rej += 1
        consec += 1
        mx = max(mx, consec)
        if consec >= max_consec:
            ev += 1
            consec = 0
    return ev, rej, mx


def count_rejections(frames, tau) -> int:
    """Total gate rejections at `tau`. Independent of max_consecutive_uncertain,
    which is what lets us recover the tau a CSV was produced with."""
    return sum(simulate(rows, tau, 1, "shipped")[1] for rows in frames.values())


def infer_tau(frames, rej_map):
    """Recover the tau that produced this CSV from its `n_rejected` column.

    Rejections do not depend on `max_consecutive_uncertain`, so a 1-D scan over
    tau is exact. This is the known-positive check that licenses the verdict:
    if no tau reproduces the counter, the CSV is not from this code path and we
    must not interpret it.
    """
    try:
        want = sum(int(float(v)) for v in rej_map.values())
    except (TypeError, ValueError):
        return None, None, 0
    if want == 0:
        return None, want, 0
    best, best_err = None, None
    grid = [round(0.06 + 0.005 * i, 3) for i in range(int((0.65 - 0.06) / 0.005) + 1)]
    for tau in grid:
        err = abs(count_rejections(frames, tau) - want)
        if best_err is None or err < best_err:
            best, best_err = tau, err
    return best, want, best_err


def structural_audit(frames, rej_map, ra_map) -> None:
    """Reproduce the pipeline's own counter and prove what it can never do."""
    tau_csv, want, err = infer_tau(frames, rej_map)
    if tau_csv is None:
        if want == 0:
            print("      (no rejections in this CSV: every frame was accepted, "
                  "so the counter never even\n       left 0 -- the mechanism "
                  "is dead before the CONSECUTIVE rule is reached)")
        else:
            print("      (could not read `n_rejected`; skipping the structural "
                  "check)")
        return
    got = count_rejections(frames, tau_csv)
    faithful = (err == 0)
    try:
        ra_csv = sum(int(float(v)) for v in ra_map.values())
    except (TypeError, ValueError):
        ra_csv = -1
    print(f"      tau recovered from `n_rejected`: {tau_csv:.3f} "
          f"(simulated {got} vs csv {want}) -> "
          f"{'FAITHFUL' if faithful else f'APPROXIMATE, off by {err}'}")
    if ra_csv == 0 and got > 0:
        print(f"      csv `n_reanchors` = 0 while the gate rejected {got} frames "
              f"=> ran with the\n      unreachable CONSECUTIVE rule (see below).")
    elif ra_csv == got and got > 0:
        print(f"      csv `n_reanchors` = {ra_csv} == rejections => this CSV ran "
              f"with an EFFECTIVE\n      max_consecutive_uncertain of 1 (a "
              f"reachability probe, not the shipped config):\n      every "
              f"rejected frame re-anchored.")
    elif ra_csv >= 0:
        print(f"      csv `n_reanchors` = {ra_csv} (rejections = {got}).")
    print(f"      shipped cadence: counter on the K={K} decision grid "
          f"(post-2026-09-17 fix)")
    print(f"      {'tau':>5} | {'rejections':>11} | {'max run':>8} | "
          f"{'events (shipped)':>17} | {'events (legacy)':>16}")

    for tau in CANDIDATE_TAUS:
        ev_s = rej_s = run_s = ev_l = 0
        for rows in frames.values():
            e, r, m = simulate(rows, tau, MAX_CONSEC, "shipped")
            ev_s += e
            rej_s += r
            run_s = max(run_s, m)
            ev_l += simulate(rows, tau, MAX_CONSEC, "legacy")[0]
        tag = "  <== inferred" if abs(tau - tau_csv) < 1e-9 else ""
        print(f"      {tau:>5.2f} | {rej_s:>11} | {run_s:>8} | "
              f"{ev_s:>17} | {ev_l:>16}{tag}")

    worst = max((simulate(rows, tau_csv, MAX_CONSEC, "shipped")[2]
                 for rows in frames.values()), default=0)
    print(f"      ==> worst achievable run of consecutive rejections = {worst}, "
          f"threshold = {MAX_CONSEC}")
    if worst < MAX_CONSEC:
        print(f"      ==> UNREACHABLE for ANY tau on this K={K} cadence.\n"
              f"          The gate could flag 100% of decision frames and still "
              f"never re-anchor.\n"
              f"          If the `events (legacy)` column is non-zero while this "
              f"one is 0, the CSV\n          predates the 2026-09-17 counter fix "
              f"and the arm never ran its mechanism.")
    else:
        print(f"      ==> REACHABLE: a run of {worst} >= {MAX_CONSEC} is "
              f"attainable, so `max_consecutive_uncertain`\n"
              f"          is a meaningful threshold on this cadence. Whether a "
              f"given tau ACTUALLY fires\n          is the `events (shipped)` "
              f"column.")
    if not faithful:
        print("      (the recovery is only approximate: this CSV may come from a "
              "different config than\n       configs/default.yaml -- read the "
              "run column as indicative, the verdict on\n       "
              "unreachability as structural.)")


def report(path) -> None:
    pts, reanch = load(path)
    print("=" * 78)
    print(f"{os.path.basename(path)}")
    if not pts:
        print("  no usable dagrs decision frames -- nothing to judge")
        return

    # ---- (1) reachability ------------------------------------------------- #
    vacs = sorted(p[0] for p in pts)
    n = len(vacs)
    print(f"  (1) REACHABILITY   n(decision frames) = {n}"
          f"   [shipped gate: tau>={TAU_SHIPPED}, {MAX_CONSEC} consecutive, K={K}]")
    print(f"      vacuity  min/p25/med/p75/p95/max = "
          f"{vacs[0]:.3f} / {vacs[n//4]:.3f} / {vacs[n//2]:.3f} / "
          f"{vacs[3*n//4]:.3f} / {vacs[int(.95*n)]:.3f} / {vacs[-1]:.3f}")
    for t in CANDIDATE_TAUS:
        frac = sum(1 for v in vacs if v >= t) / n
        tag = "  <== shipped" if abs(t - TAU_SHIPPED) < 1e-9 else ""
        span = f"1 per {1/frac:6.1f} frames" if frac > 0 else "        never"
        print(f"      tau>={t:.2f}: {100*frac:5.1f}% flagged ({span}){tag}")
    zeros = sum(1 for v in reanch.values() if str(v).strip() in ("0", "0.0"))
    print(f"      n_reanchors == 0 in {zeros}/{len(reanch)} cells")
    if zeros == len(reanch):
        print("      ==> THE MECHANISM NEVER RAN: this arm is numerically "
              "`dagrs_no_reanchor`.")

    # ---- (2) structural reachability --------------------------------------- #
    print(f"  (2) STRUCTURAL REACHABILITY")
    structural_audit(load_frames(path), load_counters(path, "n_rejected"),
                     load_counters(path, "n_reanchors"))

    # ---- (3) informativeness ---------------------------------------------- #
    e = [p[1] for p in pts]
    rho = spearman([p[0] for p in pts], e)
    print(f"  (3) INFORMATIVENESS  Spearman rho(vacuity, 1 - J_frame) = {rho:+.3f}"
          f"   [usable direction: rho > 0]")
    lo = [p[1] for p in pts if p[0] < 0.13]
    hi = [p[1] for p in pts if p[0] >= 0.16]
    if lo and hi:
        print(f"      mean error @ vacuity <0.13 : {sum(lo)/len(lo):.4f} (n={len(lo)})")
        print(f"      mean error @ vacuity >=0.16: {sum(hi)/len(hi):.4f} (n={len(hi)})")
    fail = [p for p in pts if p[1] > 0.5]
    if fail:
        print(f"      collapsed frames (error>0.5): {len(fail)}, "
              f"their vacuity spans "
              f"{min(p[0] for p in fail):.3f}..{max(p[0] for p in fail):.3f}")
    print("      ==> high rho + unreachable tau = MISTUNED (fixable by re-picking "
          "tau);\n          low rho = the statistic itself is unusable.")


def print_band() -> None:
    print("=" * 78)
    print(f"closed form: vacuity = M / (M + kappa*sum(A^gamma)), "
          f"kappa={KAPPA}, gamma={GAMMA}  (M cancels)")
    for abar in (0.90, 0.70, 0.50, 0.32, 0.20, 0.10):
        m = 3
        print(f"   mean agreement A={abar:.2f} -> vacuity = "
              f"{m / (m + KAPPA * m * abar ** GAMMA):.3f}")
    print("   reaching the shipped 0.55 needs A <= 0.32 (mask already collapsing);"
          "\n   collapse is a HIGH-agreement state, so vacuity can miss it.")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not self_test():
        print("\nSELF-TEST FAILED -- refusing to interpret anything.")
        return 2
    print(f"shipped hyperparameters read from configs/default.yaml: "
          f"tau={TAU_SHIPPED}, max_consecutive_uncertain={MAX_CONSEC}, "
          f"keyframe_stride={K}")
    paths = [Path(a) for a in args] or sorted(
        Path(p) for p in glob.glob(str(ROOT / "_scratch" / "attr_20260917"
                                   / "dagrs_probe*.csv")))
    if not paths:
        print("\nno input CSVs found; pass them explicitly.")
        return 1
    for p in paths:
        if p.exists():
            report(p)
        else:
            print(f"MISSING {p}")
    print_band()
    return 0


if __name__ == "__main__":
    sys.exit(main())
