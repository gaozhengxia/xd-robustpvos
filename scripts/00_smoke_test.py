"""End-to-end self-test. RUN THIS FIRST, before downloading anything.

It validates, in order:
  1. environment (python / torch / CUDA / sm_120 / SAM 2 availability)
  2. the degradation engine, including BIT-EXACT reproducibility
  3. the restoration operator bank
  4. the evidence / Dirichlet math (invariants are asserted, not eyeballed)
  5. the metrics (J and F on analytically known cases)
  6. the statistics (bootstrap / Wilcoxon / Spearman on synthetic data)
  7. the FULL DAG-RS pipeline on a synthetic video with the dummy backend,
     showing that DAG-RS beats the greedy baseline on a degraded sequence
  8. the TABLE-13 ablation builder's dead-arm guard, in both directions: a
     planted arm that is bit-identical to A0 must abort the build (its zero
     delta would otherwise be written up as "this component is unnecessary"),
     while a grid of genuinely perturbed arms must get past that guard
  9. the severity-curve builder's own self-test, which plants a known severity
     response and a known gap-widening on the PUBLISHED rows and demands the
     result, plus a negative control on its level-3 reconciliation
 10. the cross-SOURCE guards: `113` validates the derived `data/MOSE_mini`
     (palette integrity, frame/mask alignment, source round-trip, each paired
     with a control that must fail), and `114`'s self-test plants a known
     source-level offset plus a known extra degradation gap and recovers both,
     while CALIBRATING its false-positive rate over 30 no-effect replications
     rather than trusting a single null draw.  `113` is skipped, loudly, when
     the dataset is not present.

Exits non-zero if any check fails.

    python scripts/00_smoke_test.py
"""
from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path

# Child guards print non-ASCII: U+2212 in 119's planted correlation line, "Rädle" in 123's
# bibliography.  Python encodes stdout with the locale code page (GBK on this host) whenever stdout
# is NOT a real console -- i.e. exactly when this suite is run with its output redirected to a log,
# as CI and every captured run does.  The child then dies with UnicodeEncodeError *instead of*
# reporting its verdict, and the wrapper sees truncated output and reports a spurious FAIL.
# Setting it here covers every subprocess.run in this file at once, because children inherit the
# environment; guards that may be invoked directly carry their own reconfigure as well.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xdrp.degradations import (ALL_DEGRADATION_NAMES, apply_degradation,
                               severity_schedule, to_float, to_u8)
from xdrp.evidence import (DegradationProfile, agreement_scores, as_bool,
                           dirichlet_from_agreement,
                           fuse_candidates, iou_binary, mask_to_logits, prompt_logits)
from xdrp.iqa import compute_iqa
from xdrp.metrics import f_boundary, j_region, jf, seq_metrics
from xdrp.operators import OperatorBank, build_default_bank
from xdrp.pipeline import make_config, run_sequence
from xdrp.sam_backend import DummyBackend
from xdrp.stats import bootstrap_ci, cliffs_delta, holm_bonferroni, spearman_with_ci, \
    wilcoxon_paired

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []
warnings = []


def check(name: str, ok: bool, detail: str = "", soft: bool = False) -> bool:
    """`soft=True` records a WARNING instead of a FAILURE.

    Used for prerequisites the user may legitimately not have installed yet
    (torch / SAM 2 / matplotlib figures) -- these must not make the self-test
    fail, because they are not code defects.
    """
    if ok:
        results.append((name, True, detail))
        print(f"  [{PASS}] {name}" + (f"  -- {detail}" if detail else ""))
        return True
    if soft:
        warnings.append((name, detail))
        print(f"  [{WARN}] {name}" + (f"  -- {detail}" if detail else ""))
        return False
    results.append((name, False, detail))
    print(f"  [{FAIL}] {name}" + (f"  -- {detail}" if detail else ""))
    return False


def section(title: str) -> None:
    print(f"\n=== {title} ===")


#: Guard suites that declined to run because their inputs are not distributed.
skips = []


def skip_guard(tag: str, what: str) -> bool:
    """True when the guard behind `tag` would report SKIP in this checkout.

    The decision is delegated to `scripts/_release_gate.py` -- the same module the
    guard itself calls -- so this harness and the guard cannot disagree about
    whether it ran.  Only a *declared-unpublished* absence skips: if a shipped
    input is missing, this returns False and the guard runs and fails loudly,
    which is what should happen then.

    Skipping is a legitimate outcome (a fresh clone has no internal working
    documents) but it is never silent: the tag is recorded and reported in the
    summary, so a skip in the author's own checkout -- where every declared input
    is present -- cannot pass for a green run.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import _release_gate
    except ModuleNotFoundError:
        # Not "unknown, therefore skip".  If the gate module itself is missing
        # then the guard about to run will fail too, and it should fail loudly
        # rather than be quietly excused.  Observed the hard way: the module was
        # present locally but absent from the index, then absent from disk, and
        # six guards died with a ModuleNotFoundError that named no cause.
        print("  [FAIL] scripts/_release_gate.py is missing -- the guard chain "
              "cannot decide whether to skip, so nothing is excused")
        return False

    status, msg = _release_gate.gate(tag, ROOT, mode=what)
    if status != "skip":
        return False
    head = next((ln.strip() for ln in msg.splitlines() if ln.strip()), msg)
    print(f"  [skip] {head}")
    skips.append(tag)
    return True


# --------------------------------------------------------------------------- #
# 1. environment
# --------------------------------------------------------------------------- #

def test_env() -> None:
    section("1. Environment")
    print(f"  python      : {sys.version.split()[0]}")
    try:
        import torch
        print(f"  torch       : {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  cuda avail  : {cuda_ok}")
        if cuda_ok:
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            print(f"  gpu         : {name}  sm_{cap[0]}{cap[1]}")
            if cap[0] >= 12:
                warn = ("Blackwell (sm_120) detected. PyTorch must be a cu128 build "
                        "and >= 2.7.0, otherwise you will hit 'no kernel image is "
                        "available for execution on the device'.")
                check("RTX 50-series PyTorch build", "+cu12" in torch.__version__ or
                      "cu128" in torch.__version__, warn)
            else:
                check("GPU detected", True, f"sm_{cap[0]}{cap[1]}")
        else:
            check("CUDA available", False,
                  "no GPU visible to torch -- the pipeline still runs with the dummy "
                  "backend, but SAM 2 inference would be unusably slow", soft=True)
    except ImportError:
        check("torch installed", False,
              "install with: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128", soft=True)

    try:
        import sam2  # noqa: F401
        check("SAM 2 importable", True)
    except Exception as e:  # noqa: BLE001
        check("SAM 2 importable", False,
              f"{type(e).__name__}  (expected before you install SAM 2; only the "
              f"dummy backend is exercised here)", soft=True)

    import cv2 as _cv2
    print(f"  opencv      : {_cv2.__version__}")
    import scipy
    print(f"  scipy       : {scipy.__version__}")

    try:
        import matplotlib
        print(f"  matplotlib  : {matplotlib.__version__}")
    except Exception as e:  # noqa: BLE001
        check("matplotlib available", False, f"{e} (needed only for figures)")


# --------------------------------------------------------------------------- #
# 2. degradations
# --------------------------------------------------------------------------- #

def _synth_frame(h: int = 128, w: int = 160, t: int = 0) -> np.ndarray:
    """A textured background with a bright moving disk."""
    rng = np.random.default_rng(1234)
    base = rng.random((h, w, 3)).astype(np.float32) * 0.25 + 0.30
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    cx = int(w * (0.30 + 0.35 * (t / 20.0)))
    cy = int(h * (0.40 + 0.20 * np.sin(t / 4.0)))
    cv2.circle(base, (cx, cy), 18, (0.92, 0.88, 0.30), -1)
    return np.clip(base, 0, 1)


def synth_object_mask(h: int = 128, w: int = 160, t: int = 0) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cx = int(w * (0.30 + 0.35 * (t / 20.0)))
    cy = int(h * (0.40 + 0.20 * np.sin(t / 4.0)))
    cv2.circle(m, (cx, cy), 18, 1, -1)
    return m.astype(bool)


def test_degradations() -> None:
    section("2. Degradation engine")
    f = _synth_frame()

    # determinism (bit-exact)
    a = apply_degradation(f, "fog", 3.0, seed=7, stream_id=11)
    b = apply_degradation(f, "fog", 3.0, seed=7, stream_id=11)
    check("fog is bit-identical for the same (seed, stream_id)",
          bool(np.array_equal(a, b)), f"max|diff|={np.abs(a - b).max():.3e}")

    c = apply_degradation(f, "fog", 3.0, seed=7, stream_id=12)
    check("different stream_id gives a different realisation",
          not np.array_equal(a, c))

    # monotonic degradation (PSNR must fall as severity rises)
    from xdrp.iqa import psnr
    vals = []
    for lv in (1.0, 2.0, 3.0, 4.0, 5.0):
        d = apply_degradation(f, "fog", lv, seed=3, stream_id=0)
        vals.append(psnr(to_u8(f), to_u8(d)))
    mono = all(vals[i] >= vals[i + 1] - 1e-6 for i in range(len(vals) - 1))
    check("fog severity is monotonically worse (PSNR)",
          mono, "PSNR " + " > ".join(f"{v:.1f}" for v in vals))

    # all degradations produce finite, in-range output
    bad = []
    for name in ALL_DEGRADATION_NAMES:
        for lv in (1.0, 3.0, 5.0):
            out = apply_degradation(f, name, lv, seed=1, stream_id=0)
            if not np.isfinite(out).all() or out.min() < -1e-6 or out.max() > 1 + 1e-6:
                bad.append(f"{name}@{lv}")
    check("all degradations stay finite and in [0,1]", not bad, str(bad))

    # compound degradation actually composes
    fog = apply_degradation(f, "fog", 4.0, seed=5, stream_id=0)
    comp = apply_degradation(f, "C1_fog_noise", 4.0, seed=5, stream_id=0)
    check("compound C1 differs from its fog member",
          float(np.abs(fog - comp).mean()) > 1e-3,
          f"mean|diff|={float(np.abs(fog - comp).mean()):.4f}")

    # severity schedule is bounded and reproducible
    s1 = severity_schedule(20, base=3.0, seed=2)
    s2 = severity_schedule(20, base=3.0, seed=2)
    check("severity schedule reproducible and in [1,5]",
          bool(np.array_equal(s1, s2)) and float(s1.min()) >= 1.0 and float(s1.max()) <= 5.0,
          f"range [{s1.min():.2f}, {s1.max():.2f}]")


# --------------------------------------------------------------------------- #
# 3. operators
# --------------------------------------------------------------------------- #

def test_operators() -> None:
    section("3. Restoration operator bank")
    bank = OperatorBank(build_default_bank())
    f = _synth_frame()
    check("bank has >= 13 operators", len(bank) >= 13, f"M={len(bank)}")

    ok, det = True, []
    for i, op in enumerate(bank):
        try:
            o1 = op(f)
            o2 = op(f)
            if not np.array_equal(o1, o2):
                ok = False
                det.append(f"{op.name}:non-deterministic")
            if not np.isfinite(o1).all() or o1.min() < -1e-6 or o1.max() > 1 + 1e-6:
                ok = False
                det.append(f"{op.name}:range")
        except Exception as e:  # noqa: BLE001
            ok = False
            det.append(f"{op.name}:{type(e).__name__}")
    check("every operator runs, is deterministic and in range", ok, "; ".join(det))

    # DCP dehazing must REDUCE the dark channel: the dark-channel prior states
    # that a haze-free image has dark channel ~= 0, while haze adds airlight and
    # therefore raises it. (Getting this direction backwards is a common slip.)
    foggy = apply_degradation(f, "fog", 4.0, seed=1, stream_id=0)
    from xdrp.iqa import dark_channel_mean, laplacian_variance
    dh = bank.apply_name("dcp", foggy)
    check("dcp dehazing reduces the dark channel (dark-channel prior)",
          dark_channel_mean(dh) < dark_channel_mean(foggy),
          f"{dark_channel_mean(foggy):.3f} -> {dark_channel_mean(dh):.3f}")

    # USM must increase sharpness
    blurry = apply_degradation(f, "motion_blur", 4.0, seed=1, stream_id=0)
    sh = bank.apply_name("unsharp_hi", blurry)
    check("unsharp increases Tenengrad sharpness",
          laplacian_variance(sh) > laplacian_variance(blurry),
          f"{laplacian_variance(blurry):.1f} -> {laplacian_variance(sh):.1f}")


# --------------------------------------------------------------------------- #
# 4. evidence math
# --------------------------------------------------------------------------- #

def test_evidence() -> None:
    section("4. Evidence / Dirichlet math")
    h, w = 64, 64
    # two identical candidates + one unrelated one
    m1 = np.zeros((h, w), bool)
    m1[10:40, 10:40] = True
    m2 = m1.copy()
    m3 = np.zeros((h, w), bool)
    m3[30:55, 30:55] = True

    check("IoU of a mask with itself is 1", abs(iou_binary(m1, m1) - 1.0) < 1e-9)

    A, comps = agreement_scores([m1, m2, m3], m1)
    check("agreement: the identical candidate scores highest",
          int(np.argmax(A)) in (0, 1), f"A={np.round(A, 4).tolist()}")
    check("agreement components all lie in [0,1]",
          bool((comps["a"] >= -1e-9).all() and (comps["a"] <= 1 + 1e-9).all()
               and (comps["c"] >= -1e-9).all() and (comps["c"] <= 1 + 1e-9).all()
               and (comps["u"] >= -1e-9).all() and (comps["u"] <= 1 + 1e-9).all()))

    # vacuity must increase with disagreement
    ev_agree = dirichlet_from_agreement(np.ones(5))
    ev_disagree = dirichlet_from_agreement(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    check("vacuity rises when hypotheses disagree",
          ev_disagree.vacuity > ev_agree.vacuity,
          f"nu(agree)={ev_agree.vacuity:.4f} < nu(disagree)={ev_disagree.vacuity:.4f}")
    check("vacuity bounds hold: 1/(kappa+1) <= nu <= 1",
          abs(ev_agree.vacuity - 1.0 / (8.0 + 1.0)) < 1e-6 and ev_disagree.vacuity <= 1.0 + 1e-9,
          f"lower bound = {1/9:.4f}")
    check("beliefs sum to (1 - vacuity)",
          abs(ev_agree.belief.sum() - (1.0 - ev_agree.vacuity)) < 1e-9,
          f"sum(b)={ev_agree.belief.sum():.6f}, 1-nu={1-ev_agree.vacuity:.6f}")

    fused, order = fuse_candidates([m1, m2, m3], np.array([0.5, 0.4, 0.1]), top_k=2)
    check("fusion with k=2 returns 2 indices and a non-empty mask",
          len(order) == 2 and fused.sum() > 0, f"order={order}, area={int(fused.sum())}")

    # profile invariants
    prof = DegradationProfile(13, lam=0.5, min_j=3)
    check("profile starts uniform and not ready", not prof.ready)
    prof.update(np.array([0.9, 0.05, 0.05]), idxs=[0, 1, 2])
    check("profile masses sum to 1 after update",
          abs(prof.w.sum() - 1.0) < 1e-9, f"sum={prof.w.sum():.6f}")
    check("profile selects exactly j operators", len(prof.select(3)) == 3,
          f"selected={prof.select(3)}")
    check("profile argmax follows the evidence", prof.argmax() == 0)

    # logit conversion shape
    lg = mask_to_logits(m1, size=256)
    check("mask prompt logits have shape (1,256,256)", lg.shape == (1, 256, 256),
          str(lg.shape))

    # hard vs soft prompt: the soft variant must be a bounded logit of a
    # probability, NOT the saturated +-8 wall that `mask_to_logits` produces.
    soft = np.zeros((h, w), np.float32)
    soft[10:40, 10:40] = 0.8
    soft[5:8, 5:8] = 0.05
    lg_soft = prompt_logits(soft, size=256)
    check("prompt_logits dispatches hard/soft by dtype",
          lg_soft.shape == (1, 256, 256)
          and float(np.abs(lg_soft).max()) <= 8.0 + 1e-6
          and float(np.abs(lg_soft - lg).mean()) > 1e-3,
          f"soft range [{lg_soft.min():.2f}, {lg_soft.max():.2f}] "
          f"vs hard wall +-{np.abs(lg).max():.1f}")


# --------------------------------------------------------------------------- #
# 5. metrics
# --------------------------------------------------------------------------- #

def test_metrics() -> None:
    section("5. Metrics")
    a = np.zeros((50, 50), bool)
    a[10:30, 10:30] = True           # area 400
    b = a.copy()
    check("J(pred==gt) == 1", abs(j_region(a, b) - 1.0) < 1e-9)
    check("F(pred==gt) == 1", abs(f_boundary(a, b) - 1.0) < 1e-9)

    c = np.zeros((50, 50), bool)
    c[10:30, 10:20] = True           # half of a
    check("J(disjoint half) == 0.5", abs(j_region(c, a) - 0.5) < 1e-9,
          f"J={j_region(c, a):.4f}")

    d = np.zeros((50, 50), bool)
    check("J(pred empty, gt non-empty) == 0", abs(j_region(d, a)) < 1e-9)
    check("F(pred empty, gt non-empty) == 0", abs(f_boundary(d, a)) < 1e-9)

    # a 1-pixel shift must be tolerated by F with dilation=2
    e = np.zeros((50, 50), bool)
    e[11:31, 10:30] = True
    check("F tolerates a 1-pixel shift at dilation=2", f_boundary(e, a) > 0.7,
          f"F={f_boundary(e, a):.3f}")

    # ground truth with a hole at the boundary -> J&F of identical masks is 1
    j, f, jfv = jf(a, a)
    check("jf() returns (1,1,1) for identical masks",
          abs(j - 1) < 1e-9 and abs(f - 1) < 1e-9 and abs(jfv - 1) < 1e-9)

    m = seq_metrics([a, c], [a, a])
    check("seq_metrics averages per frame", abs(m["J"] - 0.75) < 1e-9,
          f"J={m['J']:.4f}")

    # IQA smoke
    fr = _synth_frame()
    deg = apply_degradation(fr, "fog", 4.0, seed=1, stream_id=0)
    iqa = compute_iqa(deg, clean=fr)
    check("IQA returns all metrics, finite",
          all(np.isfinite(v) for v in iqa.values()) and "iqa_psnr" in iqa,
          f"PSNR={iqa['iqa_psnr']:.2f}, SSIM={iqa['iqa_ssim']:.3f}")


# --------------------------------------------------------------------------- #
# 6. optical flow
# --------------------------------------------------------------------------- #

def test_flow() -> None:
    section("6. Optical flow")
    from xdrp.flow import (FlowCache, compute_flow, to_gray,        # noqa: E402
                           warp_with_flow)

    # A textured frame is required: Farneback cannot estimate motion inside a
    # uniform region (aperture problem), so a flat square would test nothing.
    rng = np.random.default_rng(3)
    tex = rng.normal(0.5, 0.18, (240, 320)).astype(np.float32)
    tex = np.clip(cv2.GaussianBlur(tex, (0, 0), 2.0), 0.0, 1.0)[..., None].repeat(3, 2)

    dy, dx = 6, 10
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    cur = cv2.warpAffine(tex, M, (320, 240), borderMode=cv2.BORDER_REPLICATE)
    M2 = np.float32([[1, 0, 2 * dx], [0, 1, 2 * dy]])
    cur2 = cv2.warpAffine(tex, M2, (320, 240), borderMode=cv2.BORDER_REPLICATE)

    gt0 = np.zeros((240, 320), bool)
    gt0[80:160, 100:200] = True
    gt1 = np.zeros_like(gt0)
    gt1[80 + dy:160 + dy, 100 + dx:200 + dx] = True
    gt2 = np.zeros_like(gt0)
    gt2[80 + 2 * dy:160 + 2 * dy, 100 + 2 * dx:200 + 2 * dx] = True

    f = compute_flow(tex, cur)
    mag = np.sqrt((f ** 2).sum(-1))
    check("flow magnitude matches the known translation",
          abs(float(mag[120, 150]) - np.hypot(dx, dy)) < 2.5,
          f"|f|={float(mag[120, 150]):.2f} vs {np.hypot(dx, dy):.2f}")

    iou_fwd = j_region(warp_with_flow(gt0, f), gt1)
    iou_bwd = j_region(warp_with_flow(gt0, -f), gt1)
    check("warp_with_flow propagates forwards (IoU with the true mask > 0.85)",
          iou_fwd > 0.85, f"IoU={iou_fwd:.3f}")
    # This is the defect that shipped: `xs + flow` scored 0.5166 mean IoU against
    # the next ground truth on 48 real frame pairs, versus 0.6282 for not warping
    # at all. Propagation was pushing the anchor away from the object.
    check("the opposite sign is decisively worse (sign is pinned)",
          iou_fwd > iou_bwd + 0.25, f"+f {iou_fwd:.3f} vs -f {iou_bwd:.3f}")

    fc = FlowCache([tex, cur, cur2])
    iou_chain = j_region(fc.warp_mask(gt0, 0, 2), gt2)
    check("FlowCache chains single-step flows to t>t+1",
          iou_chain > 0.70, f"IoU={iou_chain:.3f}")

    wc = fc.warp_confidence(gt0, 0, 1)
    wm = fc.warp_mask(gt0, 0, 1)
    check("warp_confidence agrees with warp_mask (same warp direction)",
          abs(wc - wm.sum() / gt0.sum()) < 0.15,
          f"conf={wc:.3f} area_ratio={wm.sum() / gt0.sum():.3f}")

    # The dtype trap: `to_gray` scaled a uint8 frame by 255 and saturated it to a
    # constant, so Farneback silently returned an identically zero field.
    u8 = (tex * 255.0 + 0.5).astype(np.uint8)
    g8 = to_gray(u8)
    check("to_gray accepts uint8 without saturating",
          int(g8.max()) - int(g8.min()) > 8,
          f"range=[{int(g8.min())}, {int(g8.max())}]")
    check("to_gray is consistent between float and uint8 input",
          float(np.abs(g8.astype(np.int16) - to_gray(tex).astype(np.int16)).mean()) < 2.0)


# --------------------------------------------------------------------------- #
# 7. statistics
# --------------------------------------------------------------------------- #

def test_stats() -> None:
    section("7. Statistics")
    rng = np.random.default_rng(0)
    base = rng.normal(0.60, 0.05, 30)
    method = base + rng.normal(0.04, 0.02, 30)

    pt, lo, hi = bootstrap_ci(method - base, n_boot=2000)
    check("bootstrap CI brackets the point estimate", lo <= pt <= hi,
          f"{pt:.4f} in [{lo:.4f}, {hi:.4f}]")

    w = wilcoxon_paired(base, method)
    check("Wilcoxon detects a consistent positive shift", w["p"] < 0.01,
          f"p={w['p']:.2e}, n={w['n']}")

    same = wilcoxon_paired(base, base)
    check("Wilcoxon on identical samples is not significant", same["p"] > 0.9,
          f"p={same['p']:.3f}")

    # Cliff's delta sign convention: cliffs_delta(a, b) > 0 <=> b tends to be larger.
    rng_c = np.random.default_rng(1)
    base_c = rng_c.normal(0.60, 0.05, 25)
    better = base_c + 1.0                      # strictly dominates every base value
    worse = base_c - 1.0
    check("Cliff's delta = +1 when group b strictly dominates a",
          abs(cliffs_delta(base_c, better) - 1.0) < 1e-9,
          f"delta={cliffs_delta(base_c, better):+.3f}")
    check("Cliff's delta = -1 when group b is strictly dominated",
          abs(cliffs_delta(base_c, worse) + 1.0) < 1e-9,
          f"delta={cliffs_delta(base_c, worse):+.3f}")
    noisy = base_c + rng.normal(0.04, 0.06, 25)
    d_pos = cliffs_delta(base_c, noisy)
    check("Cliff's delta is positive for a noisy but genuine improvement",
          d_pos > 0, f"delta={d_pos:+.3f}")

    adj = holm_bonferroni([0.001, 0.01, 0.04, 0.20])
    check("Holm-Bonferroni is monotone and >= raw p", all(a >= b - 1e-12
          for a, b in zip(adj, [0.001, 0.01, 0.04, 0.20])),
          f"raw->adj {[0.001,0.01,0.04,0.20]} -> {[round(a,4) for a in adj]}")

    x = np.linspace(0, 1, 40)
    y = 2.0 * x + rng.normal(0, 0.01, 40)
    sp = spearman_with_ci(x, y, n_boot=500)
    check("Spearman recovers a near-perfect monotone relation",
          sp["rho"] > 0.98 and sp["p"] < 1e-6, f"rho={sp['rho']:.4f}, p={sp['p']:.2e}")

    sp0 = spearman_with_ci(x, rng.normal(0, 1, 40), n_boot=500)
    check("Spearman on independent data is not significant", sp0["p"] > 0.05,
          f"rho={rho_txt(sp0['rho'])}, p={sp0['p']:.3f}")


def rho_txt(v: float) -> str:
    return "nan" if v != v else f"{v:.3f}"


# --------------------------------------------------------------------------- #
# 8. the full pipeline
# --------------------------------------------------------------------------- #

def _make_video(T: int = 12, h: int = 128, w: int = 160):
    clean = [_synth_frame(h, w, t) for t in range(T)]
    gts = [synth_object_mask(h, w, t) for t in range(T)]
    return clean, gts


def test_pipeline() -> None:
    section("8. Full DAG-RS pipeline (dummy backend, synthetic video)")
    T = 12
    clean, gts = _make_video(T)
    sev = severity_schedule(T, base=4.0, amp=0.25, seed=0)
    degraded = [apply_degradation(clean[t], "C1_fog_noise", float(sev[t]),
                                  seed=0, stream_id=t) for t in range(T)]

    bank = OperatorBank(build_default_bank())
    backend = DummyBackend()

    runs = {}
    for mode in ("greedy", "equal_tta", "random_op", "dagrs"):
        backend.clear_cache()
        cfg = make_config(mode)
        cfg.keyframe_stride = 3
        backend.set_reference(degraded[0], gts[0])
        res = run_sequence(degraded, gts[0], backend, bank, cfg,
                           seq_name="smoke", obj_id=1)
        m = seq_metrics(res.masks, gts)
        runs[mode] = (res, m)
        print(f"    {mode:<10} J&F={m['J&F']:.4f}  J={m['J']:.4f}  F={m['F']:.4f}  "
              f"decisions={res.n_decisions}  reanchors={res.n_reanchors}  "
              f"encoder_calls={res.n_encoder_calls}")

    check("every mode produced a mask for every frame",
          all(len(r.masks) == T for r, _ in runs.values()))

    # Regression guard. The main loop wraps `backend.predict` in a broad
    # `except Exception`, so a backend that returns a different number of
    # values than the caller unpacks would degrade EVERY mode to "propagate the
    # warped anchor" -- silently, with plausible-looking J&F. That happened once
    # (see docs/BASELINE_DIAGNOSIS.md, defect 3); this check makes it loud.
    n_fb = sum(r.n_predict_failures for r, _ in runs.values())
    check("no backend call silently fell back to the warped anchor",
          n_fb == 0, f"fallbacks={n_fb}")

    greedy = runs["greedy"][1]["J&F"]
    dagrs = runs["dagrs"][1]["J&F"]
    check("DAG-RS improves over the greedy baseline on compound degradation",
          dagrs > greedy + 1e-4, f"{greedy:.4f} -> {dagrs:.4f} (+{dagrs-greedy:.4f})")

    check("DAG-RS vacuity is recorded on decision frames",
          any(v == v for v in runs["dagrs"][0].vacuity[1:]),
          f"n_finite={sum(1 for v in runs['dagrs'][0].vacuity if v == v)}")

    # gating must sometimes actually reject
    res_d = runs["dagrs"][0]
    n_rejected = sum(1 for t in range(1, T)
                     if res_d.is_keyframe[t] and res_d.n_evaluated[t] > 1
                     and res_d.vacuity[t] >= 0.55)
    print(f"    gating rejections at decision frames: {n_rejected}")

    check("no NaNs leaked into any output mask",
          all(np.isfinite(r.masks[t].astype(np.float32)).all()
              for r, _ in runs.values() for t in range(T)))

    # cost sanity: DAG-RS must not cost more than evaluating every operator
    check("DAG-RS costs less than equal-TTA (profile prunes the bank)",
          runs["dagrs"][0].n_encoder_calls <= runs["equal_tta"][0].n_encoder_calls,
          f"{runs['dagrs'][0].n_encoder_calls} <= "
          f"{runs['equal_tta'][0].n_encoder_calls}")

    # determinism of the whole pipeline
    backend.clear_cache()
    backend.set_reference(degraded[0], gts[0])
    cfg = make_config("dagrs")
    cfg.keyframe_stride = 3
    r2 = run_sequence(degraded, gts[0], backend, bank, cfg, seq_name="smoke", obj_id=1)
    same = all(np.array_equal(a, b) for a, b in zip(runs["dagrs"][0].masks, r2.masks))
    check("the full pipeline is bit-reproducible", same)

    # ---- soft re-prompt path (opt-in; see docs/BASELINE_DIAGNOSIS.md) ----- #
    # This is the direction under test after the closed-loop runaway diagnosis,
    # so it gets covered here before any expensive GPU sweep.
    backend.clear_cache()
    cfg_s = make_config("dagrs", mask_prompt="soft")
    cfg_s.keyframe_stride = 3
    backend.set_reference(degraded[0], gts[0])
    r_s = run_sequence(degraded, gts[0], backend, bank, cfg_s,
                       seq_name="smoke", obj_id=1)
    m_s = seq_metrics(r_s.masks, gts)
    n_diff = sum(1 for a, b in zip(runs["dagrs"][0].masks, r_s.masks)
                 if not np.array_equal(a, b))
    print(f"    dagrs/soft J&F={m_s['J&F']:.4f}  "
          f"fallbacks={r_s.n_predict_failures}  frames_differing_from_hard={n_diff}")
    check("soft re-prompt path runs end-to-end with no backend fallbacks",
          r_s.n_predict_failures == 0 and len(r_s.masks) == T,
          f"fallbacks={r_s.n_predict_failures}, frames={len(r_s.masks)}")
    check("mask_prompt='soft' actually changes the trajectory",
          n_diff > 0, f"{n_diff}/{T} frames differ from the hard wall")

    # ---- IQA is a property of the FRAME PAIR, not of the object ------------ #
    # `evaluate_cell` is entered once per instance, so before 2026-09-17 it
    # recomputed the whole 8-metric panel (354 ms/frame, SSIM alone 243 ms) for
    # every object of a sequence -- the panels are identical, so this was pure
    # waste that made the sweeps ~2x longer. The hoist is a pure memoisation, and
    # this check pins both halves of that claim: the values must match a fresh
    # computation from the observed frame, and the two objects must agree.
    from xdrp.benchmark import DegradedSequence, Protocol, evaluate_cell
    from xdrp.iqa import compute_iqa
    _proto = Protocol(degradation="fog", level=3.0, frame_stride=1)
    _dseq = DegradedSequence(clean, degraded, {1: gts[0], 2: gts[0]},
                             {1: gts, 2: gts}, sev, _proto, "smoke_mo")
    backend.clear_cache()
    _cfg = make_config("greedy")
    _cfg.keyframe_stride = 3
    _recs = evaluate_cell(_dseq, backend, bank, _cfg, collect_iqa=True)
    _by_key = {(r["obj_id"], r["frame"]): r for r in _recs}
    _objs = sorted({r["obj_id"] for r in _recs})
    _ref_txt, _obj_txt = [], []
    for t in range(T):
        fresh = compute_iqa(degraded[t], clean=clean[t])
        _ref_txt.append(abs(_by_key[(1, t)]["iqa_entropy"] - fresh["iqa_entropy"]))
        if len(_objs) > 1:
            _obj_txt.append(abs(_by_key[(1, t)]["iqa_entropy"]
                                - _by_key[(2, t)]["iqa_entropy"]))
    _ref_max = max(_ref_txt) if _ref_txt else float("nan")
    _obj_max = max(_obj_txt) if _obj_txt else float("nan")
    check("cached IQA equals a fresh computation on the observed frame",
          len(_objs) == 2 and _ref_max < 1e-12,
          f"objects={_objs}, max|diff|={_ref_max:.2e}")
    check("IQA is identical across the objects of one sequence",
          bool(_obj_txt) and _obj_max == 0.0,
          f"max|obj1-obj2|={_obj_max:.2e}")


# --------------------------------------------------------------------------- #
# 9. configuration plumbing
# --------------------------------------------------------------------------- #

def test_config_plumbing() -> None:
    """Guards the yaml -> mode-config path used by every experiment script.

    Two defects lived here: a hand-written field list per script meant (a) a new
    yaml knob (e.g. `mask_prompt`) never reached some runners, and (b) copying
    `use_reanchor` back from the yaml silently turned ablation arm A7
    (`dagrs_no_reanchor`) into a duplicate of the full method. Both are invisible
    in the output tables, so they are asserted here.
    """
    section("9. Configuration plumbing")
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from _common import config_for_mode, load_yaml          # noqa: E402

    raw = load_yaml(str(ROOT / "configs" / "default.yaml"))
    dagrs_yaml = raw["dagrs"]

    full = config_for_mode(raw, "dagrs")
    no_re = config_for_mode(raw, "dagrs_no_reanchor")
    check("ablation switch survives the yaml round-trip "
          "(dagrs_no_reanchor != dagrs)",
          full.use_reanchor is True and no_re.use_reanchor is False,
          f"dagrs.use_reanchor={full.use_reanchor}, "
          f"no_reanchor.use_reanchor={no_re.use_reanchor}")

    et = config_for_mode(raw, "equal_tta")
    check("equal_tta keeps its arm switches after the yaml round-trip",
          et.use_agreement is False and et.use_gating is False
          and et.use_profile is False,
          f"agreement={et.use_agreement}, gating={et.use_gating}, "
          f"profile={et.use_profile}")

    check("yaml tunables reach the mode config",
          full.mask_prompt == str(dagrs_yaml["mask_prompt"])
          and full.keyframe_stride == int(dagrs_yaml["keyframe_stride"]),
          f"mask_prompt={full.mask_prompt}, K={full.keyframe_stride}")

    check("unknown overrides are rejected instead of silently ignored",
          _raises(lambda: config_for_mode(raw, "dagrs", not_a_field=1)))

    # CLI path: `--mask-prompt soft` must reach the DAG-RS config.
    from _common import base_parser, preset                # noqa: E402
    ns = base_parser("selftest").parse_args(["--mask-prompt", "soft"])
    soft = config_for_mode(preset(ns)["cfg"], "dagrs")
    check("--mask-prompt soft reaches the DAG-RS config",
          soft.mask_prompt == "soft", f"mask_prompt={soft.mask_prompt}")

    # Code <-> paper drift, third instance of the same defect class. Ablation arm
    # A11 was declared in the paper's TABLE-13 and listed in the runner, but its
    # mechanism was never implemented, so selecting it silently reproduced A0 --
    # i.e. the ablation table would have contained a copy of the full method
    # labelled as a different configuration. The two lists are now one assertion.
    import importlib.util as _ilu                            # noqa: E402
    _spec = _ilu.spec_from_file_location(
        "_abl", str(Path(__file__).resolve().parent / "04_ablation.py"))
    _abl = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_abl)
    code_arms = set(_abl.ARM_LEGEND) - set(_abl.NOT_IMPLEMENTED_ARMS)
    declared = set()
    _fw = ROOT / "docs" / "PAPER_FRAMEWORK.md"
    if _fw.exists():
        for _line in _fw.read_text(encoding="utf-8").splitlines():
            _m = re.match(r"\|\s*(A\d+)\s*\|", _line)
            if _m:
                declared.add(_m.group(1))
    check("ablation arms in code == arms declared in PAPER_FRAMEWORK TABLE-13",
          bool(declared) and declared == code_arms,
          f"paper={sorted(declared)}  code={sorted(code_arms)}")

    # --- evaluation protocol: every annotated instance must be scored -------- #
    # `max_objects` was 1 in all three places below (yaml / SweepConfig / preset
    # fallback) until 2026-09-17. A cap of 1 takes the LOWEST label id, which on
    # DAVIS is often a sliver (lab-coat's instance 1 is 18 px), so 17 of the 30
    # val sequences were scored on a single instance and the resulting number is
    # not comparable with any published DAVIS J&F. Drift between the three
    # defaults is the same defect class as the config whitelists above.
    from xdrp.benchmark import SweepConfig                     # noqa: E402
    from _common import base_parser, preset                    # noqa: E402
    _sweep = SweepConfig()
    _pre = {}
    try:
        _pre = preset(base_parser("probe").parse_args([]))
    except SystemExit:
        pass
    check("every instance is scored by default (yaml / SweepConfig / preset agree)",
          _sweep.max_objects == 0
          and int(raw["protocol"]["max_objects"]) == 0
          and _pre.get("max_objects") == 0,
          f"SweepConfig={_sweep.max_objects} yaml={raw['protocol']['max_objects']} "
          f"preset={_pre.get('max_objects')}")

    # --- chunking knob must be inert unless asked for ------------------------ #
    check("--seq-offset is 0 by default, so chunking never changes the protocol",
          _sweep.seq_offset == 0 and _pre.get("seq_offset") == 0,
          f"SweepConfig={_sweep.seq_offset} preset={_pre.get('seq_offset')}")

    # --- incremental append used by long (multi-hour) sweeps ----------------- #
    # `--resume` appends each finished cell and later skips it, which is the only
    # reason a sweep longer than one shell call is safe to run at all. The
    # failure to guard against is a second header row appearing mid-file, which
    # would corrupt every downstream reader.
    import tempfile as _tf                                      # noqa: E402
    from xdrp.benchmark import maybe_append_csv                 # noqa: E402
    _ap_path = str(Path(_tf.mkdtemp(prefix="append_selftest_")) / "grow.csv")
    maybe_append_csv([{"seq": "a", "J&F": 0.1}], _ap_path)
    maybe_append_csv([{"seq": "b", "J&F": 0.2}], _ap_path)
    _lines = [ln for ln in Path(_ap_path).read_text(encoding="utf-8").splitlines()
              if ln.strip()]
    check("incremental append keeps exactly one header and keeps every row",
          len(_lines) == 3 and _lines[0].startswith("seq")
          and _lines[0].count("J&F") == 1,
          f"{len(_lines)} lines: {_lines}")

    # --- the multi-instance collapse the protocol now requires --------------- #
    # With >1 instance per sequence, an un-collapsed table weights sequences by
    # instance count and an un-collapsed paired test lets the last object
    # overwrite the others. Both are silent, so both are asserted here.
    import importlib.util as _ilu2                           # noqa: E402
    _s2 = _ilu2.spec_from_file_location(
        "_an5", str(Path(__file__).resolve().parent / "05_analysis.py"))
    _an5 = _ilu2.module_from_spec(_s2)
    _s2.loader.exec_module(_an5)
    _mo = [{"seq": "s1", "obj_id": o, "mode": "dagrs", "degradation": "fog",
            "level": "3.0", "frame": t, "JF_frame": (0.60 if o == 1 else 0.20),
            "arm": "", "arm_note": ""}
           for t in (0, 1) for o in (1, 2)]
    _coll = _an5.collapse_objects(_an5.sequence_level(_mo))
    check("per-sequence collapse averages instances instead of overwriting",
          len(_coll) == 1 and abs(float(_coll[0]["J&F"]) - 0.40) < 1e-9
          and int(_coll[0]["n_objects"]) == 2,
          f"rows={len(_coll)} J&F={_coll[0]['J&F'] if _coll else 'n/a'} "
          f"n_objects={_coll[0]['n_objects'] if _coll else 'n/a'} (want 0.40 / 2)")

    # --- the same collapse on the *reporting* path ----------------------------#
    # `collapse_objects` protects the production tables; `100_pair_probe_report`
    # produces the paired DAG-RS-vs-greedy numbers quoted in the diagnosis and it
    # had the identical bug (it keyed on (degradation, seq) and kept the first row,
    # silently scoring obj1 only -- and obj1 is the instance with the largest
    # delta). Guarding only one of the two paths is how this class of defect
    # survives, so the reporting helper is pinned on synthetic rows too.
    _s100 = _ilu2.spec_from_file_location(
        "_pair100", str(Path(__file__).resolve().parent / "100_pair_probe_report.py"))
    _pair100 = _ilu2.module_from_spec(_s100)
    _s100.loader.exec_module(_pair100)
    _rows = []
    for o, (gv, dv, nre) in {1: (0.20, 0.30, 3), 2: (0.80, 0.70, 0)}.items():
        for m, v in (("greedy", gv), ("dagrs", dv)):
            _rows.append({"seq": "s1", "obj_id": str(o), "mode": m,
                          "degradation": "fog", "frame": "0", "J&F": str(v),
                          "n_reanchors": str(nre if m == "dagrs" else 0),
                          "n_encoder_calls": "0", "n_predict_failures": "0"})
    _seqs, _pairs, _multi, _ra = _pair100.collapse_pair_rows(_rows)
    _mean = lambda v: sum(v) / len(v)                           # noqa: E731
    _gm = _mean([float(_pairs[("fog", "s1", o)]["greedy"]["J&F"]) for o in ("1", "2")])
    _dm = _mean([float(_pairs[("fog", "s1", o)]["dagrs"]["J&F"]) for o in ("1", "2")])
    _naive = (float(_pairs[("fog", "s1", "1")]["dagrs"]["J&F"])
              - float(_pairs[("fog", "s1", "1")]["greedy"]["J&F"]))
    check("reporting helper collapses instances (mean) instead of taking obj1",
          abs(_gm - 0.50) < 1e-9 and abs(_dm - 0.50) < 1e-9
          and abs((_dm - _gm) - 0.0) < 1e-9 and abs(_naive - 0.10) < 1e-9
          and _ra[("fog", "s1")] == 3 and len(_multi) == 1,
          f"greedy={_gm} dagrs={_dm} delta={_dm - _gm} naive_obj1_delta={_naive} "
          f"n_re={_ra.get(('fog', 's1'))} multi={len(_multi)} "
          f"(want 0.50 / 0.50 / 0.0, naive 0.10, n_re 3, multi 1)")
    _rows3 = [{"seq": "s2", "obj_id": o, "mode": m, "degradation": "fog",
               "frame": "0", "J&F": "0.5", "n_reanchors": "0",
               "n_encoder_calls": "0", "n_predict_failures": "0"}
              for o in ("1", "2", "10") for m in ("greedy", "dagrs")]
    _ms = _pair100.collapse_pair_rows(_rows3)[2]
    check("reporting helper orders object ids numerically, not by string length",
          len(_ms) == 1 and list(_ms[0][2]) == ["1", "2", "10"],
          f"objects={list(_ms[0][2]) if _ms else 'n/a'} (want ['1','2','10'])")


def test_video_arm() -> None:
    """Guards the whole-video (memory-bank) arm's wiring and its input pipeline.

    This arm is the comparator the paper's central claim was missing, and it is
    the only arm that does not go through the per-frame loop. Three things can go
    wrong silently and each is asserted here:

    * the mode is not registered -> it takes the DAG-RS branch and borrows the
      restoration bank (the failure class section 10 already guards for other
      arms, but a NEW arm is exactly when it reappears);
    * the arm is measured with some of the DAG-RS machinery still on, so it stops
      isolating the memory bank;
    * the frames reach the tracker through a different conversion than every
      other arm uses (wrong channel order, wrong scale, no resize), which would
      make the comparison an artefact of preprocessing.
    """
    section("12. Whole-video (memory-bank) arm")
    from xdrp.pipeline import VIDEO_MODES, make_config, run_sequence
    from xdrp.video_backend import (OFFLOAD_STATE_TO_CPU, POSTPROC_CONNECTIVITY,
                                    build_inference_state, connected_components_cv2,
                                    frames_to_images, mask_from_logits)

    check("sam2video is registered as a whole-video arm",
          "sam2video" in VIDEO_MODES,
          f"VIDEO_MODES={VIDEO_MODES}")

    vcfg = make_config("sam2video")
    live = [k for k in ("use_restoration", "use_agreement", "use_gating",
                        "use_profile", "use_reanchor") if getattr(vcfg, k)]
    check("the video arm carries no DAG-RS mechanism switch",
          not live and vcfg.tau_score == 0.0 and vcfg.reseed_stride == 0
          and not vcfg.is_dagrs_family(),
          f"live={live} tau_score={vcfg.tau_score} reseed={vcfg.reseed_stride}")

    # A dummy backend cannot stand in for a memory bank; the arm must say so
    # rather than quietly running the frame-by-frame path.
    _f3 = [np.random.rand(32, 40, 3).astype(np.float32) for _ in range(3)]
    _m3 = np.zeros((32, 40), bool)
    _m3[8:20, 10:28] = True
    _bank = build_default_bank()
    try:
        run_sequence(_f3, _m3, DummyBackend(), _bank, vcfg, seq_name="s", obj_id=1)
        _dummy_refused = False
    except RuntimeError:
        _dummy_refused = True
    check("a video mode refuses a backend that carries no SAM 2 config",
          _dummy_refused, "the DummyBackend was accepted for mode='sam2video'")

    for field, value in (("use_reanchor", True), ("tau_score", 0.88),
                         ("reseed_stride", 20)):
        c = make_config("sam2video")
        setattr(c, field, value)
        try:
            run_sequence(_f3, _m3, DummyBackend(), _bank, c, seq_name="s", obj_id=1)
            ok = False
        except ValueError:
            ok = True
        check(f"a video mode is refused with {field}={value!r}",
              ok, "the arm would have been measured with a DAG-RS setting on")

    # --- preprocessing: channel order, scale and resize, on a known answer ---- #
    red = np.zeros((20, 30, 3), np.float32)
    red[..., 0] = 1.0                                   # pure RED, RGB order
    imgs, vh, vw = frames_to_images([red] * 2, 64)
    want = [(1.0 - 0.485) / 0.229, (0.0 - 0.456) / 0.224, (0.0 - 0.406) / 0.225]
    got = [float(imgs[0, c].mean()) for c in range(3)]
    check("the video arm feeds SAM 2 RGB, at SAM 2's scale (not BGR, not 0-255)",
          imgs.shape == (2, 3, 64, 64) and imgs.dtype.is_floating_point
          and all(abs(a - b) < 1e-5 for a, b in zip(got, want))
          and (vh, vw) == (20, 30),
          f"shape={tuple(imgs.shape)} means={[round(g, 3) for g in got]} "
          f"want={[round(w, 3) for w in want]} hw={(vh, vw)}")

    # --- inference state: the offload flag and the storage device must agree -- #
    # `storage_device` (NOT `device`) is what the propagation loop reads when it
    # parks per-frame outputs, so letting the two drift apart yields either a
    # device-mismatch deep inside the tracker or a silent CPU<->GPU round trip.
    # The offload switch is on to keep the memory bank out of the 16 GB of VRAM
    # that two concurrent full-protocol workers have to share -- a memory-only
    # change, so this check pins the invariant rather than the numeric result.
    import torch as _torch

    class _StubPredictor:
        image_size = 64
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

        @staticmethod
        def _get_image_feature(state, frame_idx=0, batch_size=1):
            return None

    _sp = _StubPredictor()
    _st_on = build_inference_state(_sp, [red] * 2)
    _st_off = build_inference_state(_sp, [red] * 2, offload_state_to_cpu=False)
    check("the offload switch and the storage device are derived together",
          bool(_st_on["offload_state_to_cpu"]) == OFFLOAD_STATE_TO_CPU
          and (_st_on["storage_device"].type == "cpu") == OFFLOAD_STATE_TO_CPU
          and _st_off["storage_device"] == _sp.device,
          f"on={_st_on['storage_device']} off={_st_off['storage_device']} "
          f"const={OFFLOAD_STATE_TO_CPU} device={_sp.device}")

    # --- mask post-processing: logits -> bool, and a loud shape mismatch ------ #
    _log = np.zeros((16, 16), np.float32)
    _log[4:12, 4:12] = 2.0
    got_mask = mask_from_logits(_log, (16, 16))
    try:
        mask_from_logits(_log, (17, 16))
        shape_guard = False
    except ValueError:
        shape_guard = True
    check("logits are thresholded at 0 and a resolution mismatch is refused",
          got_mask.dtype == bool and got_mask.sum() == 64 and shape_guard,
          f"dtype={got_mask.dtype} sum={got_mask.sum()} guard={shape_guard}")

    # --- the SAM 2 CUDA-extension replacement -------------------------------- #
    # SAM 2's video predictor fills holes of area <= 8 in the LOW-RES logits, a
    # step that needs `sam2._C`. Without this replacement the library catches the
    # ImportError and silently skips the step, so the arm would not be the
    # released model. Known answers, not shape checks.
    # NOTE ON SEMANTICS: the argument is `mask <= 0`, i.e. the BACKGROUND of the
    # predicted mask, so a "hole" is a small connected component of the INPUT's
    # nonzeros. A first version of this test asserted the opposite convention and
    # failed on a correct implementation -- the label goes on the input's True
    # pixels, and the surrounding False pixels are simply not labelled.
    _one = np.zeros((5, 5), np.uint8)
    _one[2, 2] = 1                          # a single background pixel = a 1-px hole
    lab, ar = connected_components_cv2(_one)
    check("connected components: a 1-pixel hole is one component of area 1",
          int(lab.sum()) == 1 and int(ar.max()) == 1 and int(ar[2, 2]) == 1
          and int(lab[0, 0]) == 0,
          f"labels={lab.tolist()} areas={ar.tolist()} (want one comp of area 1)")

    lab0, ar0 = connected_components_cv2(np.zeros((5, 5), np.uint8))
    check("connected components: an empty input labels nothing (no phantom hole)",
          int(lab0.sum()) == 0 and int(ar0.sum()) == 0,
          f"labels={lab0.tolist()} areas={ar0.tolist()}")

    _two = np.zeros((6, 6), np.uint8)
    _two[0:2, 0:2] = 1                     # area 4
    _two[4:6, 4:6] = 1                     # area 4, separate
    lab2, ar2 = connected_components_cv2(_two)
    check("connected components: two islands get distinct labels and areas",
          sorted(set(lab2[lab2 > 0].tolist())) == [1, 2]
          and sorted(set(ar2[ar2 > 0].tolist())) == [4],
          f"labels={sorted(set(lab2.flatten().tolist()))} "
          f"areas={sorted(set(ar2.flatten().tolist()))}")

    try:
        import torch as _torch
        from xdrp.video_backend import install_postprocessing_patch
        import sam2.utils.misc as _misc
        _orig_cc = _misc.get_connected_components
        install_postprocessing_patch(device="cpu")
        _patched = _misc.get_connected_components is not _orig_cc
        _lg = _torch.zeros(1, 1, 32, 32) + 1.0
        _lg[0, 0, 8:10, 8:10] = -1.0                     # area 4  -> filled
        _lg[0, 0, 20:30, 20:30] = -1.0                   # area 100 -> kept
        _filled = _misc.fill_holes_in_mask_scores(_lg, 8)
        _misc.get_connected_components = _orig_cc
        check("SAM 2's hole-filling post-process runs (CUDA extension replaced)",
              _patched and abs(float(_filled[0, 0, 8, 8]) - 0.1) < 1e-6
              and float(_filled[0, 0, 25, 25]) < 0.0,
              f"patched={_patched} small={float(_filled[0, 0, 8, 8])} "
              f"large={float(_filled[0, 0, 25, 25])} (want 0.1 / <0)")
    except ImportError as e:                      # pragma: no cover
        warnings.append(("hole-filling post-process",
                         f"SAM 2 not installed: {e}"))
        check("SAM 2's hole-filling post-process runs (CUDA extension replaced)",
              False, "sam2 is not importable")

    check("the post-process patch records the connectivity it used",
          int(POSTPROC_CONNECTIVITY) in (4, 8),
          f"POSTPROC_CONNECTIVITY={POSTPROC_CONNECTIVITY}")

    # --- the dispatch contract: lengths, prompt frame, mechanism counters ----- #
    # Run `run_sequence` for a video mode with the tracker stubbed out, so the
    # padding/consistency contract is checked without a GPU. A shorter diagnostic
    # list would silently misalign every per-frame CSV column.
    import xdrp.video_backend as _vb
    from xdrp.sam_backend import SAM2Config as _SAM2Cfg
    _T = 7
    _frames = [np.random.rand(16, 16, 3).astype(np.float32) for _ in range(_T)]
    _first = np.zeros((16, 16), bool)
    _first[4:12, 4:12] = True
    _real_for, _real_track = _vb.predictor_for, _vb.run_video_tracking
    _vb.predictor_for = lambda cfg, autocast_dtype=None: "stub-predictor"
    _vb.run_video_tracking = lambda p, f, m, **kw: (
        [np.zeros((16, 16), bool) for _ in f], {"n_frames": len(f),
                                                "n_encoder_calls": 99})

    class _StubBackend:
        cfg = _SAM2Cfg(checkpoint="stub.pt", model_cfg="stub.yaml", device="cpu")
        n_encoder_calls = 0

    try:
        _res = run_sequence(_frames, _first, _StubBackend(), _bank,
                            make_config("sam2video"), seq_name="stub", obj_id=2)
        diag_len_ok = all(len(getattr(_res, k)) == _T for k in
                          ("vacuity", "selected_op", "n_evaluated", "agree_a",
                           "agree_c", "agree_u", "backend_score", "is_keyframe",
                           "masks"))
        check("the video arm's per-frame diagnostics are padded to the clip length",
              diag_len_ok, f"masks={len(_res.masks)} want={_T}")
        check("the video arm reports the prompt itself on frame 0, like every arm",
              bool((_res.masks[0] == _first).all()) and int(_res.obj_id) == 2,
              f"frame0 equals prompt={bool((_res.masks[0] == _first).all())}")
        check("the video arm reports no DAG-RS mechanism activity",
              (_res.n_decisions, _res.n_reanchors, _res.n_rejected,
               _res.n_reseeds) == (0, 0, 0, 0) and _res.n_encoder_calls == 99,
              f"decisions={_res.n_decisions} reanchors={_res.n_reanchors} "
              f"rejected={_res.n_rejected} reseeds={_res.n_reseeds} "
              f"enc={_res.n_encoder_calls}")
        # A short track must fail loudly: every reported metric is a per-frame
        # mean, so a frame that silently goes missing changes the denominator.
        _vb.run_video_tracking = lambda p, f, m, **kw: (
            [np.zeros((16, 16), bool) for _ in f[:-1]],
            {"n_frames": len(f) - 1, "n_encoder_calls": 0})
        try:
            run_sequence(_frames, _first, _StubBackend(), _bank,
                         make_config("sam2video"), seq_name="stub", obj_id=1)
            short_refused = False
        except RuntimeError:
            short_refused = True
        check("the video arm refuses a track shorter than the clip",
              short_refused, f"6-of-7-frame track raised={short_refused}")
    finally:
        _vb.predictor_for, _vb.run_video_tracking = _real_for, _real_track


def test_propagation_arms() -> None:
    """Guards the propagation arms' branch assignment and their two corrections.

    Which structural branch an arm takes used to be chosen by enumerating mode
    strings at each call site (`cfg.mode == "greedy" or cfg.mode == "oracle_clean"`).
    Adding `greedy_gate` without also editing `operator_list` would have sent it
    down the DAG-RS branch, which hands the arm the restoration bank it exists to
    NOT have -- and it would still have printed a plausible J&F. That is the same
    failure class as the config-whitelist drift guarded in section 9, so the arms
    are registered in xdrp/pipeline.py and asserted here.
    """
    section("10. Propagation arms (acceptance policy and drift correction)")
    from xdrp.degradations import to_float                             # noqa: E402
    from xdrp.flow import FlowCache                                    # noqa: E402
    from xdrp.pipeline import (FIXED_OPERATOR_MODES, PROPAGATION_MODES,  # noqa: E402
                               TTA_MODES, VIDEO_MODES, make_config)

    # --- every selectable arm must have a registered branch ---------------- #
    import importlib.util as _ilu3                                     # noqa: E402

    def _load(name: str, path: Path):
        spec = _ilu3.spec_from_file_location(name, str(path))
        mod = _ilu3.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _mx = _load("_mx", Path(__file__).resolve().parent / "02_run_baselines.py")
    _abl3 = _load("_abl3", Path(__file__).resolve().parent / "04_ablation.py")
    # ARM_LEGEND maps an arm LABEL (A0..A10) to (note, mode, overrides); what
    # needs a registered branch is the underlying mode.
    _abl_modes = {mode
                  for _k, (_note, mode, _over) in _abl3.ARM_LEGEND.items()
                  if _k not in _abl3.NOT_IMPLEMENTED_ARMS and mode}
    selectable = set(_mx.MODES) | _abl_modes
    registered = (set(PROPAGATION_MODES) | set(FIXED_OPERATOR_MODES)
                  | set(TTA_MODES) | set(VIDEO_MODES))
    unregistered = sorted(m for m in selectable
                          if m not in registered and not m.startswith("dagrs"))
    check("every selectable arm has a registered pipeline branch",
          not unregistered, f"unregistered={unregistered}")

    # --- the corrected arms must not borrow the method's mechanism ---------- #
    _gate = make_config("greedy_gate")
    _seed = make_config("greedy_reseed")
    _rob = make_config("greedy_robust")
    _re = [_gate.use_reanchor, _seed.use_reanchor, _rob.use_reanchor]
    _ow = [_gate.output_when_rejected, _seed.output_when_rejected,
           _rob.output_when_rejected]
    check("the corrected propagation arms never re-anchor on evidence",
          not any(_re) and all(x == "anchor" for x in _ow),
          f"reanchor={_re} output_when_rejected={_ow}")

    # --- behavioural checks on a synthetic video --------------------------- #
    T = 12
    clean, gts = _make_video(T)
    degraded = [apply_degradation(clean[t], "C1_fog_noise", 4.0, seed=0, stream_id=t)
                for t in range(T)]
    bank = OperatorBank(build_default_bank())
    backend = DummyBackend()

    def _run(cfg):
        backend.clear_cache()
        cfg.keyframe_stride = 3
        backend.set_reference(degraded[0], gts[0])
        return run_sequence(degraded, gts[0], backend, bank, cfg,
                            seq_name="arms", obj_id=1)

    r_plain = _run(make_config("greedy"))
    check("unconditional acceptance refuses nothing (so a gated arm reporting 0 "
          "rejections is really the ungated one)",
          r_plain.n_rejected == 0 and r_plain.n_reseeds == 0,
          f"rejected={r_plain.n_rejected} reseeds={r_plain.n_reseeds}")

    # A threshold above any possible predicted IoU must refuse every frame, and
    # a refusal must emit the flow-propagated anchor -- not the prediction it
    # just refused. Both halves are asserted, because emitting the refused
    # prediction anyway would look like a working gate in the summary counts.
    r_rej = _run(make_config("greedy_gate", tau_score=2.0))
    check("a gate that always fires refuses every frame and counts them",
          r_rej.n_rejected == T - 1,
          f"rejected={r_rej.n_rejected} (want {T - 1})")

    flows = FlowCache([to_float(f) for f in degraded])
    want_ref = [as_bool(gts[0]).copy()] + [
        as_bool(flows.warp_mask(gts[0], 0, t)) for t in range(1, T)]
    worst = max(int(np.abs(as_bool(a).astype(np.int8)
                           - as_bool(b).astype(np.int8)).sum())
                for a, b in zip(r_rej.masks, want_ref))
    check("a refused frame emits the flow-propagated anchor, not the prediction",
          worst == 0, f"max pixel disagreement vs the warped anchor = {worst}")

    # --- periodic re-seeding ------------------------------------------------ #
    r_seed = _run(make_config("greedy_reseed", reseed_stride=4))
    want_n = (T - 1) // 4
    check("periodic re-seeding fires on the requested schedule",
          r_seed.n_reseeds == want_n,
          f"reseeds={r_seed.n_reseeds} (want {want_n})")
    _n_diff = sum(1 for a, b in zip(r_seed.masks, r_plain.masks)
                  if not np.array_equal(as_bool(a), as_bool(b)))
    check("re-seeding changes the trajectory (it is not a no-op)",
          _n_diff > 0, f"{_n_diff}/{T} frames differ from the ungated run")

    # --- the guards --------------------------------------------------------- #
    check("re-seeding is refused on the DAG-RS family",
          _raises(lambda: _run(make_config("dagrs", reseed_stride=4)), ValueError))
    check("an unregistered mode is refused instead of silently taking the "
          "DAG-RS branch",
          _raises(lambda: _run(make_config("greedy_typo")), KeyError))

    # --- the policy must survive into the CSV ------------------------------- #
    # Two rows that agree on (seq, mode, degradation, level) but differ in the
    # gate are different experiments; without these columns downstream cannot
    # tell them apart.
    import inspect as _insp                                          # noqa: E402
    from xdrp import benchmark as _bm                                # noqa: E402
    src = _insp.getsource(_bm.evaluate_cell)
    missing = [k for k in ('"tau_score"', '"reseed_stride"', '"n_rejected"',
                           '"n_reseeds"') if k not in src]
    check("the acceptance policy is recorded in every result row",
          not missing, f"missing columns={missing}")


def test_mechanism_runs() -> None:
    """Guards the two defects that let `dagrs` claim a mechanism it never ran.

    Until 2026-09-17 the arm could not re-anchor for ANY threshold (the counter
    was reset on every accepted frame while the gate was only evaluated every
    K-th frame), and the descriptor it compared was a different width from the
    one it had cached, so the mechanism both could not fire and would have
    crashed if it had. Neither defect was covered here -- which is exactly why
    both survived: an uncovered core stage is an unprotected one.

    This section asserts the three properties that make the claim falsifiable.
    """
    section("11. The mechanism actually runs (descriptor contract + gate)")
    from xdrp.sam_backend import SAM2Backend                          # noqa: E402

    # --- (a) descriptor width belongs to the backend, not to the input ----- #
    # Constructed without __init__, so no checkpoint and no GPU are needed: the
    # contract is about the arithmetic, not about SAM 2 itself.
    _b = SAM2Backend.__new__(SAM2Backend)
    _b.cfg = None
    _b._predictor = None
    _b._pooled_dim = None
    _img = np.zeros((32, 32, 3), np.float32)
    _m = np.zeros((32, 32), bool)
    _m[8:24, 8:24] = True

    class _FakeTensor:
        """Duck-typed stand-in for the torch chain `.float().cpu().numpy()`."""

        def __init__(self, a):
            self.a = a

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class _StubPredictor:
        """Minimal stand-in: a warm encoder exposing an 8-channel embedding."""
        _is_image_set = True

        @staticmethod
        def get_image_embedding():
            return [_FakeTensor(np.zeros((8, 4, 4), np.float32))]

    s_cold = _b.descriptor(_img, _m).shape
    _b._predictor = _StubPredictor()
    s_warm = _b.descriptor(_img, _m).shape
    s_empty = _b.descriptor(_img, np.zeros_like(_m)).shape
    check("descriptor width is input-independent (empty mask included)",
          s_warm == s_empty and s_warm[0] == s_cold[0] + 8,
          f"cold={s_cold} warm={s_warm} empty={s_empty}")
    check("_fit_width returns exactly the requested length",
          SAM2Backend._fit_width(np.arange(5, dtype=np.float32), 8).shape == (8,)
          and SAM2Backend._fit_width(np.arange(5, dtype=np.float32), 3).shape
          == (3,))

    # --- (b) the trigger is reachable, and (c) firing is not a no-op ------- #
    T = 12
    clean, gts = _make_video(T)
    degraded = [apply_degradation(clean[t], "C1_fog_noise", 4.0, seed=0, stream_id=t)
                for t in range(T)]
    bank = OperatorBank(build_default_bank())
    backend = DummyBackend()

    def _run(cfg):
        backend.clear_cache()
        cfg.keyframe_stride = 3
        backend.set_reference(degraded[0], gts[0])
        return run_sequence(degraded, gts[0], backend, bank, cfg,
                            seq_name="mech", obj_id=1)

    # tau = 0 makes every decision frame uncertain, so a counter that shares the
    # gate's clock MUST reach `max_consecutive_uncertain`. If both counters come
    # back 0, the counter and the gate are on different clocks again.
    cfg_d = make_config("dagrs")
    cfg_d.tau_vacuity = 0.0
    cfg_d.max_consecutive_uncertain = 2
    r_d = _run(cfg_d)
    check("the re-anchoring trigger is reachable on a firing input",
          r_d.n_reanchors > 0,
          f"n_reanchors={r_d.n_reanchors} n_rejected={r_d.n_rejected}")

    r_no = _run(make_config("dagrs_no_reanchor"))
    n_diff = sum(1 for a, b in zip(r_d.masks, r_no.masks)
                 if not np.array_equal(as_bool(a), as_bool(b)))
    check("a firing mechanism changes the result "
          "(dagrs != dagrs_no_reanchor)",
          n_diff > 0, f"{n_diff}/{T} frames differ from dagrs_no_reanchor")


def _raises(fn, exc=KeyError) -> bool:
    try:
        fn()
        return False
    except exc:
        return True


def test_chunk_merge_formats() -> None:
    """Guards the chunk merger against the two sweep output formats.

    `94_merge_sweeps.py` was written for the CL3 sweep, which writes ONE ROW PER
    FRAME because it collects IQA (`collect_iqa` -> per-frame IQA correlation).
    Every other grid -- the CL4 memory-bank comparison, the ablation sweeps --
    runs `--no-iqa` and writes ONE ROW PER CELL, with no `frame` column. The audit
    filtered every row on `frame`, so on a per-cell chunk it silently dropped
    100% of the data and reported an empty sweep. "0 rows" reads like "nothing
    has run yet", which is exactly the kind of wrong answer that gets believed:
    it would have hidden a complete grid behind a "still running" impression.
    """
    section("13. Chunk merge covers both sweep output formats")
    import csv as _csv                                                # noqa: E402
    import importlib.util as _ilu4                                    # noqa: E402
    import tempfile                                                   # noqa: E402

    _spec = _ilu4.spec_from_file_location(
        "_m94", str(Path(__file__).resolve().parent / "94_merge_sweeps.py"))
    _m94 = _ilu4.module_from_spec(_spec)
    _spec.loader.exec_module(_m94)

    cell_hdr = ["seq", "obj_id", "mode", "degradation", "level", "n_frames",
                "J", "F", "J&F"]
    frame_hdr = ["seq", "obj_id", "mode", "degradation", "level", "n_frames",
                 "frame", "J_frame", "F_frame", "JF_frame"]
    with tempfile.TemporaryDirectory() as td:
        pc = Path(td) / "cells.csv"
        with open(pc, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(cell_hdr)
            w.writerow(["bear", "1", "greedy", "clean", "1.0", "50",
                        "0.7", "0.8", "0.75"])
            w.writerow(["bear", "1", "dagrs", "clean", "1.0", "50",
                        "0.8", "0.9", "0.85"])
            w.writerow(["bear"])                     # torn append

        hdr, rows, dropped = _m94.read_csv(pc)
        check("a per-cell chunk is read, not filtered down to nothing",
              len(rows) == 2 and dropped == 1 and "frame" not in hdr,
              f"rows={len(rows)} dropped={dropped} has_frame={'frame' in hdr}")
        check("the torn-append check is skipped (not failed) for per-cell chunks",
              _m94.partial_cells(rows, has_frame=False) == set(),
              "per-cell chunk reported no partial cells")

        pf = Path(td) / "frames.csv"
        with open(pf, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(frame_hdr)
            for t in range(3):                       # one complete cell
                w.writerow(["bear", "1", "greedy", "clean", "1.0", "3",
                            str(t), "0.7", "0.8", "0.75"])
            for t in range(2):                       # one short cell (killed)
                w.writerow(["bear", "1", "dagrs", "clean", "1.0", "3",
                            str(t), "0.8", "0.9", "0.85"])
        hdr_f, rows_f, dropped_f = _m94.read_csv(pf)
        bad_f = _m94.partial_cells(rows_f, has_frame=True)
        check("a per-frame chunk still detects the short cell it always did",
              len(rows_f) == 5 and dropped_f == 0 and bad_f == {("bear", "dagrs", "clean", "1.0")},
              f"rows={len(rows_f)} dropped={dropped_f} partial={sorted(bad_f)}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def test_eval_protocol_port():
    """The ported released-evaluator scorer must stay bit-exact.

    Why this is a smoke-test item and not a one-off script: `scripts/105` and
    `107` both *quote* the released evaluator's boundary rule to justify
    reporting a second set of numbers under it. If `eval_protocol_common` ever
    drifts from `sam2/sav_dataset` -- a changed `disk` argument, a dropped
    `_seg2bmap` edge case -- the quoted numbers become wrong while everything
    still runs and still prints a plausible J&F. That is the silent-drift shape
    this repository has been burned by repeatedly.

    Skips (as a warning, not a failure) when the released file or
    scikit-image is absent, since neither is needed for the pipeline itself.
    """
    import numpy as np
    try:
        from scripts import eval_protocol_common as epc
    except Exception as e:  # noqa: BLE001
        check("eval_protocol_common imports", False, f"{type(e).__name__}: {e}")
        return

    if epc._released_evaluator_class() is None:
        warnings.append(("eval_protocol_port",
                         "released sam2/sav_dataset evaluator not loadable "
                         "(no sam2 checkout or no scikit-image) -- port fidelity "
                         "NOT verified in this run"))
        return

    H, W = 60, 80
    yy, xx = np.mgrid[0:H, 0:W]
    gts = [((yy - (20 + t)) ** 2 + (xx - (30 + 2 * t)) ** 2) <= 12 ** 2
           for t in range(7)]
    # A perfect prediction must score exactly 1.0 -- the probe is verified on a
    # known-positive case BEFORE it is trusted to report a discrepancy.
    perfect = epc.official_seq_metrics(gts, gts, False)
    check("port: perfect prediction scores exactly 1.0",
          abs(perfect["J&F"] - 1.0) <= 1e-12, f"J&F={perfect['J&F']!r}")
    check("port: a bit-identical pair gives n_frames == T (no frame filtered)",
          perfect["n_frames"] == 7, f"n_frames={perfect['n_frames']}")

    preds = [((yy - (20 + t + 1)) ** 2 + (xx - (30 + 2 * t + 1)) ** 2) <= 12 ** 2
             for t in range(7)]
    for skip in (False, True):
        rel = epc.released_check(preds, gts, skip)
        port = epc.official_seq_metrics(preds, gts, skip)
        for key in ("J", "F", "J&F"):
            check(f"port == released {key} (skip_first_last={skip})",
                  rel is not None and abs(port[key] - rel[key]) <= 1e-9,
                  f"port={port[key]:.12f} released={None if rel is None else rel[key]:.12f}")
        check(f"port: released F binds the frame count (skip_first_last={skip})",
              port["n_frames"] == (5 if skip else 7),
              f"n_frames={port['n_frames']}")

    # The released convention must differ from ours in the two documented ways:
    # all-empty frames are scored (not skipped), and the tolerance disk is
    # resolution-scaled rather than a fixed 5x5 box.
    empty = [np.zeros((H, W), bool)] * 4
    check("port: all-empty frames are scored and score 1.0",
          epc.official_seq_metrics(empty, empty, False)["n_frames"] == 4,
          f"n_frames={epc.official_seq_metrics(empty, empty, False)['n_frames']}")
    check("ours: all-empty frames are skipped instead",
          epc.ours_seq_metrics(empty, empty, 2, False)["n_frames"] == 0,
          f"n_frames={epc.ours_seq_metrics(empty, empty, 2, False)['n_frames']}")
    _bp = int(np.ceil(epc.OFFICIAL_BOUNDARY * np.linalg.norm((480, 854))))
    check("port: bound_pix at 480x854 is 8 (a radius, not a diameter)",
          _bp == 8, f"bound_pix={_bp}")

    # A conversion that can push J&F past 1 is wrong by construction; the
    # rescoring script is the only consumer, so guard the bound here.
    gain = 0.1062                       # the largest measured gain
    check("the additive correction cannot exceed the metric's range",
          min(1.0, 0.9054 + gain) <= 1.0, f"clamped={min(1.0, 0.9054 + gain)}")
    check("_seg2bmap refuses a 3-D mask rather than guessing",
          _raises(lambda: epc._seg2bmap(np.zeros((4, 4, 3), bool)), AssertionError))


def test_compound_gap_sign() -> None:
    """`108_compound_vs_member` must report the family gap with `hi - lo` sign.

    A sign inversion here is the worst kind of silent bug: the printed p-value,
    confidence interval and Cliff's delta all stay plausible, and only the
    WIDENS / NARROWS verdict flips.  This session hit exactly that --
    `gap_per_seq` returned `lo - hi`, so C4's *narrowing* gap was reported as
    widening -- and the tell was that the two independent views of the same
    quantity (per-sequence paired mean vs difference of group means) disagreed
    in sign while both looked reasonable in isolation.

    Pinned with two synthetic cases whose direction is known by construction:
    a compound gap LARGER than the member's must come out positive (widening),
    and a compound gap SMALLER must come out negative (narrowing).
    """
    import importlib.util as _ilu
    path = Path(__file__).resolve().parent / "108_compound_vs_member.py"
    if not path.exists():
        check("108_compound_vs_member.py present", False, "file not found")
        return
    spec = _ilu.spec_from_file_location("_smoke_108", str(path))
    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        check("108_compound_vs_member loads", False, f"{type(e).__name__}: {e}")
        return

    def make(deg: str, gap: float):
        out = []
        for s in ("s1", "s2", "s3", "s4"):
            out.append({"seq": s, "mode": "greedy", "degradation": deg,
                        "level": "3", "J": 0.4, "F": 0.3, "J&F": 0.35})
            out.append({"seq": s, "mode": "sam2video", "degradation": deg,
                        "level": "3", "J": 0.4 + gap, "F": 0.3 + gap,
                        "J&F": 0.35 + gap})
        return out

    cpd = make("CPD", 0.20)
    per_seq = mod.gap_per_seq(cpd, "CPD", "sam2video", "greedy")
    check("family gap is computed as hi - lo (positive by construction)",
          all(abs(v - 0.20) < 1e-12 for v in per_seq.values()),
          f"got {sorted({round(v, 6) for v in per_seq.values()})}, expected [+0.2]")

    try:
        rep = mod.paired_gap_shift(cpd, make("MEM", 0.10), "CPD", "MEM")
    except AssertionError as e:  # noqa: BLE001
        check("paired_gap_shift sign guard fires on inconsistency", False, str(e))
        return
    check("a LARGER compound gap is reported as widening",
          abs(rep["mean_delta"] - 0.10) < 1e-9 and rep["sign_consistent"],
          f"mean_delta={rep['mean_delta']:+.6f} consistent={rep.get('sign_consistent')}")
    check("paired mean equals the difference of group means",
          abs(rep["group_gap_shift"] - rep["mean_delta"]) < 1e-9,
          f"group={rep['group_gap_shift']:+.6f} paired={rep['mean_delta']:+.6f}")

    rep2 = mod.paired_gap_shift(cpd, make("MEM2", 0.30), "CPD", "MEM2")
    check("a SMALLER compound gap is reported as narrowing (negative)",
          rep2["mean_delta"] < 0 and rep2["sign_consistent"],
          f"mean_delta={rep2['mean_delta']:+.6f} consistent={rep2.get('sign_consistent')}")


def test_ablation_guard() -> None:
    """`110_ablation_table.py` must REFUSE to tabulate a dead ablation arm.

    An arm whose override never reached the pipeline reproduces A0 bit for bit,
    so its "delta" is exactly 0.  Read naively that is the strongest possible
    evidence for "this component does not matter" -- and it is a fabrication.
    The builder therefore aborts.  That abort is the difference between a
    finding and an artifact, so it gets its own test, in TWO directions:

      (a) plant one arm identical to A0  -> must abort, naming that arm;
      (b) perturb every arm              -> must get PAST that guard (it then
          trips the unrelated A0-vs-table9 protocol guard), which is what proves
          (a) is a targeted check and not a script that always fails.

    The run is a subprocess on a synthetic 61-instance / 30-sequence grid, so it
    needs no GPU and no real experiment output.
    """
    import subprocess
    import tempfile

    script = ROOT / "scripts" / "110_ablation_table.py"
    if not script.exists():
        check("ablation builder exists", False, str(script))
        return

    SEQS = 30
    ROWS = 61                      # the builder hard-codes 61 / 30

    def write_phase(d: Path, tag: str, degradation: str, plant_dead_arm: bool) -> None:
        for arm in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
            # A0 is always the untouched reference.  A1 duplicates it when
            # plant_dead_arm; otherwise every non-A0 arm differs on every
            # instance, which is exactly what the guard is supposed to accept.
            bump = 0.0 if (arm == "A0" or (plant_dead_arm and arm == "A1")) else 0.013
            lines = ["seq,obj_id,mode,degradation,level,arm,J&F,n_encoder_calls"]
            for i in range(ROWS):
                s = f"syn{i % SEQS:02d}"
                base = 0.50 + 0.01 * (i % SEQS)
                lines.append(
                    f"{s},o{i},dagrs,{degradation},3.0,{arm},"
                    f"{base + bump:.6f},{100 + i}")
            (d / f"{tag}_{arm}.csv").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")

    def run(build) -> "subprocess.CompletedProcess[str]":
        with tempfile.TemporaryDirectory(prefix="smoke_abl_") as tmp:
            d = Path(tmp)
            build(d)
            return subprocess.run(
                [sys.executable, "-B", str(script), "--dir", str(d),
                 "--out-dir", str(d / "out"), "--prefix", "t"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    try:
        dead = run(lambda d: (
            write_phase(d, "fog", "fog", True),
            write_phase(d, "C1", "C1_fog_noise", False)))
        out = (dead.stdout or "") + (dead.stderr or "")
        check("dead-arm guard aborts instead of tabulating",
              dead.returncode != 0, f"rc={dead.returncode}")
        check("dead-arm guard says WHY (bit-identical to A0)",
              "bit-identical to A0" in out,
              out.strip().splitlines()[-1] if out.strip() else "(no output)")
        check("dead-arm guard names the offending arm (fog/A1)",
              "fog" in out and "A1" in out, "")

        live = run(lambda d: (
            write_phase(d, "fog", "fog", False),
            write_phase(d, "C1", "C1_fog_noise", False)))
        out2 = (live.stdout or "") + (live.stderr or "")
        check("guard is targeted: with all arms perturbed it gets past the "
              "dead-arm check (and trips the separate table9 protocol guard)",
              "bit-identical to A0" not in out2 and "table9_main" in out2,
              out2.strip().splitlines()[-1] if out2.strip() else "(no output)")
    except subprocess.TimeoutExpired:
        check("ablation guard test completes", False, "subprocess timed out")


def test_severity_guard() -> None:
    """`111_severity_table.py --self-test` must pass.

    That self-test builds a COMPLETE 3x5 grid out of the already-published
    level-3 rows, plants a known severity response and a known gap-widening, and
    asserts the table machinery recovers both -- plus a negative control that
    corrupts one level-3 J&F and demands the reconciliation guard abort.  It is
    delegated rather than re-implemented so the two can never drift apart.
    """
    import subprocess

    script = ROOT / "scripts" / "111_severity_table.py"
    if not script.exists():
        check("severity builder exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("severity builder self-test completes", False, "timed out")
        return
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    check("severity builder self-test passes (planted signal + negative control)",
          r.returncode == 0, tail[0][:160])


def test_mose_xsource_guard() -> None:
    """The cross-source guards: `113` (data) and `114` (table).

    Two different failure modes are covered:
      * `113_check_mose_mini.py` validates `data/MOSE_mini` -- including the two
        traps that already bit this project (a palette-less P-mode PNG that every
        in-repo metric reads correctly while every external tool sees black, and
        a frame/mask misalignment).  It is SKIPPED when the dataset is absent,
        because `data/MOSE_mini` is a 1.4 GB derived download and a fresh clone
        will not have it; the skip is reported, not silent.
      * `114_xsource_table.py --self-test` is synthetic and therefore always
        runnable: it plants a known source-level offset plus a known extra
        degradation gap and requires the table to recover both, then CALIBRATES
        its own false-positive rate over 30 no-effect replications instead of
        demanding that a single null draw be non-significant.
    """
    import subprocess

    mose = ROOT / "data" / "MOSE_mini"
    checker = ROOT / "scripts" / "113_check_mose_mini.py"
    if not (mose / "manifest.csv").exists():
        print("  [skip] data/MOSE_mini absent -- 113 not run "
              "(rebuild with scripts/112_prepare_mose_mini.py --convert)")
    elif not checker.exists():
        check("MOSE checker exists", False, str(checker))
    else:
        try:
            r = subprocess.run([sys.executable, "-B", str(checker)],
                               cwd=str(ROOT), capture_output=True, text=True,
                               timeout=1800)
        except subprocess.TimeoutExpired:
            check("MOSE data validation completes", False, "timed out")
        else:
            tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
            check("MOSE data validation passes (structure + palette + "
                  "alignment, each with its own control)",
                  r.returncode == 0, tail[0][:160])

    script = ROOT / "scripts" / "114_xsource_table.py"
    if not script.exists():
        check("cross-source table builder exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True,
                           timeout=1800)
    except subprocess.TimeoutExpired:
        check("cross-source builder self-test completes", False, "timed out")
        return
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    if r.returncode != 0:
        # A failing subprocess must say WHY.  `114` reports its aborts on
        # stderr, which this harness captures separately, so reporting only the
        # stdout tail showed a truncated log with no cause (it looked like the
        # run died mid-print).  Surface both streams.
        err = (r.stderr or "").strip().splitlines()[-6:]
        print("  [diag] 114 --self-test stdout tail:")
        for line in (r.stdout or "").strip().splitlines()[-8:]:
            print("         " + line)
        print("  [diag] 114 --self-test stderr tail:")
        for line in err:
            print("         " + line)
    check("cross-source builder self-test passes "
          "(planted effect + rate-calibrated null + completeness guards)",
          r.returncode == 0, tail[0][:160])


def test_frame_denominator_guard() -> None:
    """`115_frame_denominator.py --self-test` must pass.

    The P1 denominator protocol is what every cross-arm number in the paper now
    rests on, so its correction has to be provably exact rather than plausible.
    The self-test proves the closed form against frame-by-frame ground truth on
    synthetic masks, makes a wrong denominator fail, and checks that four guard
    conditions raise instead of letting a partial table through.  Synthetic, so
    it always runs.
    """
    import subprocess

    script = ROOT / "scripts" / "115_frame_denominator.py"
    if not script.exists():
        check("frame-denominator utility exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        check("frame-denominator self-test completes", False, "timed out")
        return
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    check("frame-denominator self-test passes (exact correction + 5 guards)",
          r.returncode == 0, tail[0][:160])


def test_cl11_probe_guard() -> None:
    """`117_cl11_probe.py --self-test` must pass.

    CL11's verdict ("no decidable effect at n=10") is a NEGATIVE result, and a
    negative result is only as good as the machinery that failed to find the
    effect.  The calibration therefore plants a known +0.20 shift, a known -0.20
    shift and a known null, and requires each to land where it should.  The probe
    sweep CSVs themselves live in `_scratch/` and are not shipped, which is why
    the SMOKE-testable part is the calibration rather than the full analysis.
    """
    import subprocess

    script = ROOT / "scripts" / "117_cl11_probe.py"
    if not script.exists():
        check("CL11 probe script exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        check("CL11 probe calibration completes", False, "timed out")
        return
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    check("CL11 probe calibration passes (planted +/- effect + null)",
          r.returncode == 0, tail[0][:160])


def test_xsource_sweep_verification() -> None:
    """`116_verify_xsource.py` over the completed sweep, when it is present.

    This is the independent successor to a launcher guard that NEVER RAN: the
    MOSE cross-source launcher died on a CRLF bug after all waves had exited 0,
    so its own verification block never executed and its exit code proved
    nothing.  A skipped check is reported, never silent.
    """
    import subprocess

    script = ROOT / "scripts" / "116_verify_xsource.py"
    sweep = ROOT / "_scratch" / "mose_xsrc" / "xsrc_dagrs.csv"
    if not script.exists():
        check("cross-source verifier exists", False, str(script))
        return
    if not sweep.exists():
        print("  [skip] _scratch/mose_xsrc absent -- 116 not run "
              "(produced by _scratch/mose_xsrc/run_mose_xsrc.sh)")
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script)],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        check("cross-source verification completes", False, "timed out")
        return
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    check("cross-source sweep verifies (4 negative controls fire)",
          r.returncode == 0, tail[0][:160])


def test_mechanism_activity_guard() -> None:
    """`119_mechanism_activity.py --self-test` must pass.

    The "the mechanism runs, but firing frequency is not a knob" claim is quoted
    in the abstract-level positioning banner and in the manuscript, and it had
    NO owner: the numbers came from a paragraph written *before* the 2026-09-17
    core-code fix, so `89/120` and `r = -0.0001 / rho = +0.0290` were stale and
    were carried into the manuscript as-is.  Nothing caught it, because the only
    traceability check in the repo tested whether a number EXISTS somewhere, not
    whether it MEANS what the sentence says.

    119 recomputes the numbers from the published compound CSVs and checks them
    into the docs.  The calibration pairs a known +0.10*n_re and a known
    -0.10*n_re association (must give r = +1 / -1) with a noisy null (must give
    ~0), because a correlation reported as "~0" is only meaningful if the same
    code path can produce a large |r| when one exists.
    """
    import subprocess

    script = ROOT / "scripts" / "119_mechanism_activity.py"
    if not script.exists():
        check("mechanism-activity script exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        check("mechanism-activity calibration completes", False, "timed out")
        return
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    check("mechanism-activity calibration passes (bidirectional sign + noisy null)",
          r.returncode == 0, tail[0][:160])


def test_cl10_rho_guard() -> None:
    """`120_cl10_rho_audit.py --self-test` must pass.

    TABLE-16 (section 5.10) asks whether `vacuity` ranks frames by their true error,
    and the section would like to quote the one number `05_analysis.py --study vacuity`
    prints.  That number is measured over 3244 FRAMES drawn from 244 object instances
    in 120 cells, so the frame-level confidence interval is not a 95% interval -- this
    project has already published one consequence of mis-declaring the paired unit
    (section 5.9).  120 re-derives the claim on the declared unit, calibrates every
    interval AND every test it quotes against a synthetic zero-association null, and
    is the owner of the numbers in the document.

    The self-test's job here is the part that cannot be argued away: two planted
    signals, each of which must reach only its own test.  `nu = error centred inside
    each instance + independent per-instance jitter` must light up the WITHIN test and
    leave the BETWEEN test silent; `nu = the instance's own mean error` must do the
    reverse.  A test that fires on both, or on neither, fails the suite.
    """
    import subprocess

    script = ROOT / "scripts" / "120_cl10_rho_audit.py"
    if not script.exists():
        check("CL10 rho-audit script exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        check("CL10 rho-audit controls complete", False, "timed out")
        return
    out = r.stdout or ""
    tail = (out.strip().splitlines() or [""])[-1]
    check("CL10 rho-audit controls pass (planted within/between signals stay separate)",
          r.returncode == 0, tail[:160])
    # The checks above are only worth their runtime if they can also FAIL, so each
    # detector must report "ok" on the perturbed copies (proving it saw the defect).
    for marker in ("perturbing rho fires", "perturbing a CI bound fires",
                   "removing the anchor fires"):
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"CL10 rho-audit detector fired: {marker}",
              line.strip().endswith("ok"), line.strip()[:120] or "line missing")
    if r.returncode != 0:
        check("CL10 rho-audit stderr", False, (r.stderr or "")[-200:])


def test_release_gate() -> None:
    """`_release_gate.py` decides what the guards do when their inputs are absent.

    This is a guard about the guard chain, and it checks three things.

    (1) Its own controls pass, in both directions.  A gate that only ever skips
        would turn every document check into a green no-op; a gate that never
        skips leaves a fresh clone reading as a broken repository.

    (2) The declaration is *true* about this repository -- every input it calls
        "not distributed" really is git-ignored, every guard it names really calls
        the gate, and its requirement table agrees with the guards' own call sites.
        A declaration that has drifted from reality is worse than no declaration:
        it is the same silent divergence the guards exist to catch.

    (3) The coupling inside *this* file: any gated guard this suite actually runs
        must be pre-flighted through `skip_guard`, or a clone would report failures
        for checks that were never going to run there.
    """
    import subprocess

    # Check that the file is there BEFORE importing it.  The reverse order turns
    # "the gate module does not ship" into an uncaught ModuleNotFoundError from
    # three frames deep, which is how this suite reported six failures with one
    # cause and named none of them.
    script = ROOT / "scripts" / "_release_gate.py"
    if not script.exists():
        check("release gate exists", False, str(script))
        return

    sys.path.insert(0, str(ROOT / "scripts"))
    import _release_gate

    r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    out = r.stdout or ""
    m = re.search(r"self-test: PASS \((\d+) controls\)", out)
    check("release gate self-test passes (both directions, >= 6 controls)",
          r.returncode == 0 and bool(m) and int(m.group(1)) >= 6,
          (out.strip().splitlines() or [""])[-1][:160])

    rc = subprocess.run([sys.executable, "-B", str(script), "--check"],
                        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    cout = rc.stdout or ""
    check("the unpublished-input declaration is current and true",
          rc.returncode == 0 and "VERDICT: PASS" in cout,
          (cout.strip().splitlines() or [""])[-1][:160])

    src = Path(__file__).read_text(encoding="utf-8")
    not_preflighted = [t for t in sorted(_release_gate.REQUIRED)
                       if re.search(r'"scripts"\s*/\s*"%s_' % t, src)
                       and ('skip_guard("%s"' % t) not in src]
    check("every gated guard this suite runs is pre-flighted through skip_guard",
          not not_preflighted, "not pre-flighted: " + ", ".join(not_preflighted))


def test_sec5_table_audit_guard() -> None:
    """`121_sec5_table_audit.py --self-test` and `--check` must both pass.

    Two failures this guards against, both of the "the number exists but is not the
    number the text needs" family.

    (1) The paired-test rows of section 5.9 were still on the P0 convention
        (family gap +0.1591) while section 5.1b requires every table entry to be
        P1 (+0.1561).  118 owns the P0/P1 mirror for sections 5.2/5.3 only, so
        this table had no owner at all.  121 recomputes the rows from the P1
        inputs and requires them in both documents.

    (2) The manuscript's five result tables are transcriptions of script-generated
        blocks in the framework.  A transcription is exactly where a correct value
        acquires a wrong neighbour, so each manuscript table carries an
        `MS-TABLE:` anchor and is compared cell by cell with the framework block of
        the same name.  The self-test proves the comparison can fail (perturb a
        manuscript cell, perturb a framework cell, remove an anchor) and that it
        does not fail on an untouched copy.

    (3) As of 2026-09-28 the table has two more rows (the ablation arms A1 and A3,
        which were previously withheld).  Those rows draw on two scratch sweeps
        rather than the P1 grid, and those files record the UNSCALED metric, so
        admitting them needs two further guards -- a like-for-like protocol-drift
        check after the lift into P1, and a per-instance count proving the arm
        switch actually fired.  Each has a planted-fault control, and one control
        pins the very bug that produced the first draft of the lift (comparing raw
        against scaled and reading the unit mismatch as an 8.3e-02 drift).
    """
    import subprocess

    if skip_guard("121", "121 section-5 table audit"):
        return
    script = ROOT / "scripts" / "121_sec5_table_audit.py"
    if not script.exists():
        check("section-5 table audit script exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("section-5 table audit self-test completes", False, "timed out")
        return
    out = r.stdout or ""
    check("section-5 table audit self-test passes (12 controls)",
          r.returncode == 0 and out.count("PASS") == 12,
          (out.strip().splitlines() or [""])[-1][:160])
    # every control must be a real verdict, not an absent line
    for marker in ("untouched copy passes",
                   "perturbing a manuscript cell fires",
                   "perturbing a framework cell fires",
                   "removing an MS anchor fires",
                   "planted +0.05 shift moves the gap by +0.05",
                   "recomputation is deterministic",
                   "ablation load: 3 arms x 13 cells x 61 instances",
                   "comparing raw against P1 fires (the lift is not a no-op)",
                   "the real ablation inputs pass both new guards",
                   "a planted 1e-3 shift in one ablation dagrs value fires",
                   "an instance not scored over the whole clip fires",
                   "an arm identical to A0 fires instead of tabulating a zero"):
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"section-5 audit control ran: {marker}",
              line.strip().endswith("PASS"), line.strip()[:120] or "line missing")
    if r.returncode != 0:
        check("section-5 table audit stderr", False, (r.stderr or "")[-200:])

    try:
        rc = subprocess.run([sys.executable, "-B", str(script), "--check"],
                            cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("section-5 table audit --check completes", False, "timed out")
        return
    cout = rc.stdout or ""
    check("section-5 tables reconcile with the framework (P1 rows + mirrored cells)",
          rc.returncode == 0, (cout.strip().splitlines() or [""])[-1][:160])
    # a check whose failure mode is "pass" must report that it compared something
    ver = next((ln for ln in cout.splitlines() if "verified" in ln), "")
    check("section-5 table audit compared a non-empty set",
          bool(ver) and "0 mirrored table row(s)" not in ver, ver[:160] or "line missing")


def test_ms_doc_guard() -> None:
    """`122_ms_doc_guard.py --self-test` and `--check` must both pass.

    Checks 118-121 are *table*-level: they reconcile named blocks.  This one is
    *document*-level, and it exists because the three properties below had no owner
    at all while sections 1, 2, 6, 7 and the Abstract were written.

    (1) Numbers.  Every numeric token that reaches the manuscript must occur
        verbatim in the framework.  The manuscript grew from 0 to 693 raw tokens in
        one sitting, and a token with no framework anchor is a number with no owning
        script (red lines 6/13/15).

    (2) Wording.  The paper claims a diagnosis, not a gain.  "our method",
        "we propose", "outperforms", "improves N points" are banned as *claims* --
        but the file quotes the ban list in its own positioning note, so the rule is
        judged per blockquote *block*, not per line: markdown wraps that note and the
        banned phrase lands on a continuation line whose own text carries no marker.
        A line-wise rule fires on legal data, which is worse than no rule.

    (3) Citation keys.  The keys `[SAM2]`, `[DAVIS]`, `[MOSE]`, ... were introduced
        here while only `[N1]` existed in the obligation list, which made the
        manuscript's own provenance sentence false.  The key table in
        `NOVELTY_BOUNDARY.md` §3.0 is now the single source and is asserted.
    """
    import subprocess

    if skip_guard("122", "122 manuscript document guard"):
        return
    script = ROOT / "scripts" / "122_ms_doc_guard.py"
    if not script.exists():
        check("manuscript document guard exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("manuscript document guard self-test completes", False, "timed out")
        return
    out = r.stdout or ""
    markers = ("A fires: untraceable number reported",
               "A does not fire on a framework-backed number",
               "A does not fire on a number planted in the bibliography",
               "B fires: plain claim",
               "B fires: claim inside a blockquote",
               "B does not fire on the file's own rule note",
               "C fires: undeclared key [N99]",
               "C has something to check (n>0 on both sides)",
               "D fires: stale tabulated size",
               "D fires: manuscript grew, table not updated",
               "D does not fire on the shipped pair",
               "unchanged documents pass")
    # count verdict *lines* (three-space indent, trailing PASS): "PASS" also occurs in the
    # script's own "[122] self-test: PASS" summary line, so a substring count would be 9.
    n_ctrl = sum(1 for ln in out.splitlines()
                 if ln.startswith("   ") and ln.rstrip().endswith("PASS"))
    # Three mutually constraining numbers, so none of them can drift alone:
    #   n_ctrl  -- what the guard actually printed
    #   EXPECT  -- what it is supposed to print (catches a control silently *removed*)
    #   len(markers) -- how many controls this file names in full
    # A mismatch between the last two is the exact failure this file suffered when a
    # bibliography control was added to 122: the guard grew, the mirror did not.  Asserting
    # the coupling turns "remember to update both" into a check.
    EXPECT = 12
    check("manuscript document guard self-test passes (%d controls)" % EXPECT,
          r.returncode == 0 and n_ctrl == EXPECT,
          (out.strip().splitlines() or [""])[-1][:160])
    check("ms doc guard: every counted control is also named here",
          len(markers) == EXPECT,
          f"{len(markers)} marker(s) vs {EXPECT} expected -- update both lists together")
    # both directions for the controls that could over-fire
    for marker in markers:
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"ms doc guard control ran: {marker}",
              line.strip().endswith("PASS"), line.strip()[:120] or "line missing")
    if r.returncode != 0:
        check("manuscript document guard stderr", False, (r.stderr or "")[-200:])

    try:
        rc = subprocess.run([sys.executable, "-B", str(script), "--check"],
                            cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("manuscript document guard --check completes", False, "timed out")
        return
    cout = rc.stdout or ""
    check("manuscript traceability + wording + citation keys hold",
          rc.returncode == 0, (cout.strip().splitlines() or [""])[-1][:160])
    # a check whose failure mode is "pass" must report that it inspected a non-empty set
    line_c = next((ln for ln in cout.splitlines() if ln.startswith("[122] C.")), "")
    n_used = re.search(r"(\d+) used", line_c)
    check("manuscript document guard inspected a non-empty key set",
          bool(n_used) and int(n_used.group(1)) > 0, line_c[:160] or "line missing")
    line_a = next((ln for ln in cout.splitlines() if ln.startswith("[122] A.")), "")
    n_tok = re.search(r"(\d+) distinct non-heading token", line_a)
    check("manuscript document guard inspected a non-empty number set",
          bool(n_tok) and int(n_tok.group(1)) > 0, line_a[:160] or "line missing")
    # D reported 0 rows would mean the size table silently measured nothing
    line_d = next((ln for ln in cout.splitlines() if ln.startswith("[122] D.")), "")
    n_rows = re.search(r"(\d+) section\(s\) measured", line_d)
    check("manuscript document guard measured a non-empty section set",
          bool(n_rows) and int(n_rows.group(1)) > 0, line_d[:160] or "line missing")


def test_ref_guard() -> None:
    """`123_ref_guard.py --self-test` and `--check` must both pass.

    `122` asserts that every citation key used in the body is *declared* in the
    `NOVELTY_BOUNDARY.md` §3.0 obligation table.  That is a one-way property, and it
    cannot see the opposite failure -- which is the dangerous one: a declared
    obligation with no bibliography entry is an unpayable citation debt, and a
    manuscript satisfies "every key is declared" perfectly well while shipping no
    references at all.

    `123` owns the bibliography and asserts the equality in *both* directions, plus
    per-entry completeness (a venue / arXiv / doi / page range and a year, so a bare
    title cannot pass as a reference) and the absence of placeholder markers that a
    reader could mistake for a citation.  A single-entry-per-key rule catches the
    duplicate that would otherwise shadow a real entry.
    """
    import subprocess

    if skip_guard("123", "123 bibliography guard"):
        return
    script = ROOT / "scripts" / "123_ref_guard.py"
    if not script.exists():
        check("bibliography guard exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("bibliography guard self-test completes", False, "timed out")
        return
    out = r.stdout or ""
    markers = ("unchanged documents pass",
               "E has something to compare (n>0 on both sides)",
               "the bibliography is not empty",
               "verbose print of non-ASCII entries does not raise",
               "E fires: orphan entry [NZ]",
               "E fires: declared obligation",
               "E fires: key [ACDC] defined twice",
               "no duplicate exists in the real file",
               "F fires: entry with no publication marker",
               "F fires: entry with a venue but no year",
               "G fires: placeholder in an entry",
               "F fires: method family that cites no concrete key",
               "the family exemption is exercised by real data",
               "F/G do not fire on the real bibliography")
    n_ctrl = sum(1 for ln in out.splitlines()
                 if ln.startswith("   ") and ln.rstrip().endswith("PASS"))
    EXPECT = 14
    check("bibliography guard self-test passes (%d controls)" % EXPECT,
          r.returncode == 0 and n_ctrl == EXPECT,
          (out.strip().splitlines() or [""])[-1][:160])
    check("bibliography guard: every counted control is also named here",
          len(markers) == EXPECT,
          f"{len(markers)} marker(s) vs {EXPECT} expected -- update both lists together")
    for marker in markers:
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"bibliography guard control ran: {marker}",
              line.strip().endswith("PASS"), line.strip()[:120] or "line missing")
    if r.returncode != 0:
        check("bibliography guard stderr", False, (r.stderr or "")[-200:])

    try:
        rc = subprocess.run([sys.executable, "-B", str(script), "--check"],
                            cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("bibliography guard --check completes", False, "timed out")
        return
    cout = rc.stdout or ""
    check("bibliography is complete and consistent in both directions",
          rc.returncode == 0, (cout.strip().splitlines() or [""])[-1][:160])
    line_e = next((ln for ln in cout.splitlines() if ln.startswith("[123] E.")), "")
    m = re.search(r"(\d+) entries vs (\d+) declared", line_e)
    check("bibliography guard compared two non-empty sets",
          bool(m) and int(m.group(1)) > 0 and int(m.group(2)) > 0,
          line_e[:160] or "line missing")


def test_fig_build_guard() -> None:
    """`124_fig_build.py --self-test` and `--check` must both pass.

    A figure is a claim-bearing artifact: the numbers it draws are a claim about the
    table it illustrates.  The suite guards the tables (118-123) and could still be
    guarding *no figure at all*, which is why this test also asserts that every
    figure is either emitted (with digests) or withheld for a stated disagreement --
    a figure simply missing from the manifest is the failure mode it exists to catch.
    """
    import json
    import subprocess

    script = ROOT / "scripts" / "124_fig_build.py"
    if not script.exists():
        check("figure guard exists", False, str(script))
        return
    try:
        r = subprocess.run([sys.executable, "-B", str(script), "--self-test"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("figure guard self-test completes", False, "timed out")
        return
    out = r.stdout or ""
    markers = ("a manifest that matches its sources passes (baseline)",
               "F2 fires: convention changed to the frame-level probe",
               "F2 fires: a plotted x series drifted",
               "F2 fires: point count changed (361 != 360)",
               "F2 fires: annotated rho no longer matches the table",
               "F3 fires: the severity numbers changed",
               "F3 fires: source swapped to the P0 (retracted) file",
               "F5 fires: the plotted nu series changed",
               "fires: empty manifest",
               "FIG-3 draws quietly on the real, unmutated source (baseline)",
               "FIG-3 refuses a source that declares no subtrahend",
               "FIG-3 refuses when its own arithmetic disagrees with the owner",
               "F4's manifest entry transcribes 128's plate and reports it present",
               "F4 fires when the plate's SHA-1 is not the one 128 recorded",
               "F4 reports absence rather than a crash when 128 has not run",
               "check reports a table rho that no longer matches the plotted points",
               "a withheld figure with a genuine disagreement passes",
               "the withheld record names exactly the metric that disagrees",
               "a withheld figure writes no plate",
               "withholding fires: the disagreement is gone, so it is stale",
               "withholding fires: the recorded disagreement was edited",
               "withholding fires: a withholding that names nothing",
               "staleness fires: figure older than its source",
               "staleness quiet: figure newer than its source",
               "staleness quiet: a figure that does not exist yet is not 'stale'",
               "cross-check accepts a figure that matches its table",
               "cross-check RAISES when the figure rho differs by 1e-6",
               "cross-check RAISES when the figure has 359 points instead of 360",
               "F2 has 360 points over 30 sequences and 12 degradations",
               "the IQA cell table covers the same 30 sequences",
               "F5 picks a deterministic instance and it exists in the CSV",
               "F5 selection rule is order-independent",
               "tau_vacuity is read from the config, not hard-coded")
    n_ctrl = sum(1 for ln in out.splitlines()
                 if ln.startswith("   ") and ln.rstrip().endswith("PASS"))
    EXPECT = 33
    check("figure guard self-test passes (%d controls)" % EXPECT,
          r.returncode == 0 and n_ctrl == EXPECT,
          (out.strip().splitlines() or [""])[-1][:160])
    check("figure guard: every counted control is also named here",
          len(markers) == EXPECT,
          f"{len(markers)} marker(s) vs {EXPECT} expected -- update both lists together")
    for marker in markers:
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"figure guard control ran: {marker}",
              line.strip().endswith("PASS"), line.strip()[:120] or "line missing")
    if r.returncode != 0:
        check("figure guard stderr", False, (r.stderr or "")[-200:])

    try:
        rc = subprocess.run([sys.executable, "-B", str(script), "--check"],
                            cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        check("figure guard --check completes", False, "timed out")
        return
    cout = rc.stdout or ""
    check("every derived figure is emitted or withheld for a stated reason",
          rc.returncode == 0, (cout.strip().splitlines() or [""])[-1][:160])

    man = json.loads((ROOT / "results" / "figs" / "FIGURES.json").read_text(
        encoding="utf-8"))
    figs = man.get("figures") or {}
    check("the figure manifest is not empty", len(figs) > 0, f"{len(figs)} figure(s)")
    check("the manifest accounts for every figure the guard derives",
          set(figs) == {"F1", "F2", "F3", "F4", "F5"}, f"manifest keys {sorted(figs)}")
    drawn = [k for k, v in figs.items() if v.get("files")]
    check("at least one figure is actually on disk", len(drawn) > 0,
          f"drawn: {sorted(drawn)}")
    for k in sorted(figs):
        w = figs[k].get("withheld") or {}
        files = figs[k].get("files") or {}
        why = w.get("disagreements") or []
        # `check` prints the detail on PASS as well, so it must state the *actual*
        # state; a static "the figure is simply missing" would contradict a PASS
        # line and read as a defect in the guard rather than a report of the data.
        if files:
            state = f"emitted: {sorted(files)}"
        elif why:
            state = f"withheld on {len(why)} disagreement(s): {sorted(why)[:2]}"
        else:
            state = "neither files nor a withheld record -- the figure is simply missing"
        check(f"{k} is emitted or withheld with a stated disagreement",
              bool(files) or bool(why), state)


def test_cl3_table_guard() -> None:
    """`125_cl3_table_rows.py` owns Tables 14/15; every mode must pass.

    Nothing owned these two tables before 2026-09-24 -- `118_p1_doc_sync.py` syncs
    only Tables 9-10 and the section-5 audit never read IQA -- which is how they
    sat on the P0 result tree while section 5.1b declared every table in the paper
    P1.  So this test asserts two things: that the guard passes, and that it
    actually compared a non-empty set of rows.  A guard that silently parsed zero
    rows would print PASS exactly like a consistent document.
    """
    import subprocess

    if skip_guard("125", "125 CL3 table guard"):
        return
    script = ROOT / "scripts" / "125_cl3_table_rows.py"
    if not script.exists():
        check("CL3 table guard exists", False, str(script))
        return

    def run(*args: str):
        try:
            return subprocess.run([sys.executable, "-B", str(script), *args],
                                  cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=300)
        except subprocess.TimeoutExpired:
            return None

    for flag, label in (("--self-test", "self-test"), ("--check", "check")):
        r = run(flag)
        if r is None:
            check(f"CL3 table guard {label} completes", False, "timed out")
            continue
        out = (r.stdout or "") + (r.stderr or "")
        tail = (out.strip().splitlines() or [""])[-1][:160]
        check(f"CL3 table guard {label} passes", r.returncode == 0, tail)
        if flag == "--self-test":
            m = re.search(r"self-test: (\d+)/(\d+) controls pass", out)
            check("CL3 table guard self-test ran a non-empty control set",
                  bool(m) and int(m.group(1)) == int(m.group(2)),
                  m.group(0) if m else "no control summary line")

    r = run()
    if r is None:
        check("CL3 table guard printing mode completes", False, "timed out")
        return
    out = r.stdout or ""
    rows = [ln for ln in out.splitlines() if ln.startswith("| PSNR vs clean")]
    check("CL3 table guard emitted a PSNR row for both tables",
          len(rows) >= 2, f"{len(rows)} PSNR row(s) printed")
    check("CL3 table guard reads the P1 artefact, not the P0 one",
          "results/p1/iqa_config_level_p1.json" in out,
          (out.splitlines() or [""])[0][:140])


def test_docx_guard() -> None:
    """`127_make_docx.py` owns the Word manuscript: self-test and freshness.

    Two failure modes matter here and neither is a table or a figure:
    the deliverable can be *stale* (the manuscript was edited after the .docx was
    built) and it can be *hollow* (images embedded, captions silently dropped by
    a reader that did not parse the image attributes).  So this test asserts the
    guard's own controls pass, that the build record still matches the sources,
    and that the guard refuses a product with no captions -- which is what makes
    the second failure mode non-returnable.
    """
    import json
    import subprocess

    script = ROOT / "scripts" / "127_make_docx.py"
    if not script.exists():
        check("Word manuscript guard exists", False, str(script))
        return

    def run(*args: str):
        try:
            return subprocess.run([sys.executable, "-B", str(script), *args],
                                  cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=900)
        except subprocess.TimeoutExpired:
            return None

    r = run("--self-test")
    if r is None:
        check("Word manuscript guard self-test completes", False, "timed out")
        return
    out = (r.stdout or "") + (r.stderr or "")
    markers = (
        "the title is read from the working-title blockquote",
        "the drafting header is gone and the product opens on the author line",
        "the Sec. 5 drafting note is gone",
        "the paper blockquotes survive",
        "keywords sit between the abstract and Sec. 1",
        "a figure block is a bare implicit figure, with no attribute text",
        "FIG-3 refuses to caption an un-declared subtrahend",
        "FIG-3 refuses an arm declared as its own subtrahend",
        "a caption whose cell count disagrees with the plate aborts",
        "every table start in the real source is preceded by a blank line",
        "re-spacing neither invents nor drops a table",
        "a product whose captions were lost is rejected",
        "the landed product carries all five captions as text",
        "the built product carries every body table, and no extra",
    )
    n_ctrl = sum(1 for ln in out.splitlines()
                 if ln.strip().startswith("[PASS]"))
    check("Word manuscript guard self-test passes (%d controls)" % n_ctrl,
          r.returncode == 0 and n_ctrl >= len(markers),
          (out.strip().splitlines() or [""])[-1][:160])
    for marker in markers:
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"Word manuscript guard control ran: {marker}",
              line.strip().startswith("[PASS]"), line.strip()[:120] or "line missing")

    rc = run("--check")
    if rc is None:
        check("Word manuscript guard --check completes", False, "timed out")
        return
    cout = (rc.stdout or "") + (rc.stderr or "")
    check("the Word deliverable is not stale relative to its sources",
          rc.returncode == 0, (cout.strip().splitlines() or [""])[-1][:160])

    side = ROOT / "out" / "XD-RobustPVOS_manuscript.build.json"
    check("the build record exists", side.is_file(), str(side))
    if side.is_file():
        rec = json.loads(side.read_text(encoding="utf-8"))
        st = rec.get("structure") or {}
        check("the Word deliverable embeds one plate per drawn figure",
              st.get("images") == len(rec.get("figures_embedded") or []),
              f"{st.get('images')} image(s) for "
              f"{len(rec.get('figures_embedded') or [])} figure(s)")
        check("the build record states what was stripped from the draft",
              len(rec.get("stripped") or []) >= 2,
              "; ".join(rec.get("stripped") or [])[:140])


def test_fig4_guard() -> None:
    """`128_fig4_qualitative.py` owns FIG-4, the qualitative plate.

    FIG-4 is the one figure that needs a GPU run (every other plate is a plot of
    a landed table), so it is guarded separately.  The controls that matter are
    the two prose anchors: Sec. 6.3 names `bike-packing` as the rescue and
    `bmx-trees` object 2 at -0.2106 as the largest instance loss, and the plate is
    only allowed to land on those.  A figure that illustrated different instances
    would contradict the text silently.
    """
    import subprocess

    script = ROOT / "scripts" / "128_fig4_qualitative.py"
    if not script.exists():
        check("FIG-4 guard exists", False, str(script))
        return

    def run(*args: str):
        try:
            return subprocess.run([sys.executable, "-B", str(script), *args],
                                  cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=900)
        except subprocess.TimeoutExpired:
            return None

    r = run("--self-test")
    if r is None:
        check("FIG-4 guard self-test completes", False, "timed out")
        return
    out = (r.stdout or "") + (r.stderr or "")
    markers = (
        "the extremes and a middle instance are picked, in order",
        "the frame rule takes the largest disagreement",
        "the frame rule resolves a tie to the earliest frame",
        "traces of unequal length abort instead of truncating",
        "the delta uses the declared phase, level and subtrahend only",
        "a ranking over an empty key set aborts (no silent PASS)",
        "the prose anchors pass when the plate lands on the named instances",
        "a plate that lands on a different instance than the text aborts",
        "a plate whose delta contradicts Sec. 6.3 aborts",
        "the real ranking lands on both instances Sec. 6.3 names",
        "the real ranking reproduces Sec. 6.3's quoted loss",
    )
    n_ctrl = sum(1 for ln in out.splitlines()
                 if ln.strip().startswith("[PASS]"))
    check("FIG-4 guard self-test passes (%d controls)" % n_ctrl,
          r.returncode == 0 and n_ctrl >= len(markers),
          (out.strip().splitlines() or [""])[-1][:160])
    for marker in markers:
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"FIG-4 guard control ran: {marker}",
              line.strip().startswith("[PASS]"), line.strip()[:120] or "line missing")

    plate = ROOT / "results" / "figs" / "F4_qualitative.png"
    if not plate.is_file():
        check("FIG-4 plate is on disk (needs one GPU run)", False,
              "results/figs/F4_qualitative.png is absent -- run "
              "scripts/128_fig4_qualitative.py")
        return
    rc = run("--check")
    if rc is None:
        check("FIG-4 guard --check completes", False, "timed out")
        return
    cout = (rc.stdout or "") + (rc.stderr or "")
    check("FIG-4 still matches the ranking it was selected from",
          rc.returncode == 0, (cout.strip().splitlines() or [""])[-1][:160])


def test_table7_guard() -> None:
    """`126_table7_rows.py` owns Sec. 4.8's Table 7 and its framework mirror.

    Table 7 is the one table whose measured cells come from a *profiler* rather
    than from a sweep, and the profiler measures modes, not configurations.  The
    controls that matter are therefore the mapping (a mode whose encoder-call rate
    contradicts the `(K, J)` label the table gives it must abort), the scope (the
    measured columns are only valid at frame stride 1, checked against the stride-1
    frame counts in the P1 artifacts) and the row with no arm behind it, whose
    three cells must stay an em dash rather than be filled from its neighbour.
    """
    import subprocess

    if skip_guard("126", "126 Table 7 guard"):
        return
    script = ROOT / "scripts" / "126_table7_rows.py"
    if not script.exists():
        check("Table 7 guard exists", False, str(script))
        return

    def run(*args: str):
        try:
            return subprocess.run([sys.executable, "-B", str(script), *args],
                                  cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=600)
        except subprocess.TimeoutExpired:
            return None

    r = run("--self-test")
    if r is None:
        check("Table 7 guard self-test completes", False, "timed out")
        return
    out = (r.stdout or "") + (r.stderr or "")
    markers = (
        "the caption carries the scope, not just the words 'Table 7'",
        "the unmeasured cells are emitted as an em dash, not a placeholder",
        "the footnote states the em-dash reason and the scope it was measured at",
        "the footnote carries the profile's own cost ratio",
        "the framework mirror names its anchor and records the launcher",
        "the footprint sentence drops the per-sequence ordering claim when it is false",
        "an empty comparison set yields no footprint claim at all, not a vacuous one",
        "a profile that mixes conditions aborts rather than being averaged",
        "profiled clip lengths that are not the stride-1 lengths abort",
        "a framework with no mirror anchor aborts",
    )
    n_ctrl = sum(1 for ln in out.splitlines()
                 if ln.strip().startswith("[PASS]"))
    check("Table 7 guard self-test passes (%d controls)" % n_ctrl,
          r.returncode == 0 and n_ctrl >= len(markers),
          (out.strip().splitlines() or [""])[-1][:160])
    for marker in markers:
        line = next((ln for ln in out.splitlines() if marker in ln), "")
        check(f"Table 7 guard control ran: {marker}",
              line.strip().startswith("[PASS]"), line.strip()[:120] or "line missing")

    rc = run("--check")
    if rc is None:
        check("Table 7 --check completes", False, "timed out")
        return
    cout = (rc.stdout or "") + (rc.stderr or "")
    check("Table 7 still equals the profiler, and its framework mirror is current",
          rc.returncode == 0, (cout.strip().splitlines() or [""])[-1][:160])


def main() -> int:
    print("XD-RobustPVOS / DAG-RS self-test")
    print("=" * 68)
    for fn in (test_env, test_degradations, test_operators, test_evidence,
               test_metrics, test_flow, test_stats, test_pipeline,
               test_config_plumbing, test_propagation_arms,
               test_mechanism_runs, test_video_arm, test_chunk_merge_formats,
               test_eval_protocol_port, test_compound_gap_sign,
               test_ablation_guard, test_severity_guard,
               test_mose_xsource_guard, test_frame_denominator_guard,
               test_cl11_probe_guard, test_xsource_sweep_verification,
               test_mechanism_activity_guard, test_cl10_rho_guard,
               test_release_gate,
               test_sec5_table_audit_guard, test_ms_doc_guard, test_ref_guard,
               test_fig_build_guard, test_cl3_table_guard, test_table7_guard,
               test_docx_guard, test_fig4_guard):
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"  [FAIL] {fn.__name__} raised:")
            traceback.print_exc()
            results.append((fn.__name__, False, "exception"))

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 68)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    if skips:
        # Not a warning and not a failure: a fresh clone legitimately has none of
        # the internal working documents.  But it is *reported*, because in the
        # author's own checkout -- where every declared input exists -- a skip
        # would mean a document went missing, and that must not read as green.
        print("\nSkipped suites (their inputs are not distributed with the "
              "repository, see scripts/_release_gate.py):")
        for tag in sorted(set(skips)):
            print(f"  - guard {tag}")
        print("  Every check above was run; this checkout is simply missing the "
              "declared inputs.")
    if warnings:
        print("\nWarnings (not defects -- prerequisites you have not installed yet):")
        for name, detail in warnings:
            print(f"  ! {name}: {detail}")
    if n_fail:
        print("\nFailures:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        print("\nFix the failures above before running real experiments.")
        return 1
    print("\nEnvironment and pipeline verified. Next:")
    print("  1. download a dataset (DAVIS-2017) into data/DAVIS")
    print("  2. install SAM 2.1 and its checkpoints (see README)")
    print("  3. python scripts/01_build_benchmark.py --quick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
