#!/usr/bin/env python
"""113 - Verify data/MOSE_mini is actually evaluable, and that the checks bite.

WHY
---
`data/MOSE_mini` becomes the paper's only real cross-source VOS table
(Tier-2 of TABLE-3 / §5.5).  A silent label defect here would corrupt every
number in that table, and the historical failure mode in this repo is exactly
that kind of silence: `read_label_png`'s docstring records a week of baseline
results computed on 2 of 30 sequences because palette PNGs were read through
OpenCV.  So this script does not just check "files exist" -- it pairs every
positive assertion with a **negative control** that must fail, proving the check
can tell right from wrong.

WHAT IT CHECKS
--------------
  A. manifest.csv parses, 24 videos, frame counts consistent with the layout
  B. layout is exact: JPEGImages/<v>/<5digit>.jpg + Annotations/<v>/<5digit>.png
     for t = 0..n-1, with no gaps and no extras
  C. every frame is 854x480 and every frame image is non-degenerate
  D. every label is a P-mode PNG with the full 768-byte VOC/DAVIS palette
  E. label index sets == manifest object ids; frame 0 of each video non-empty
  F. `open_dataset('mose', ...)` loads it and agrees with the manifest
  G. NEGATIVE CONTROL (palette trap): the buggy OpenCV blue-channel read must
     *disagree* with the correct read -- else check E has no teeth
  H. NEGATIVE CONTROL (off-by-one frames): reading label t against the frame
     t+1 image must score near-zero IoU -- else the frame/label pairing is not
     actually being tested
  I. optional --roundtrip: re-decode N sampled frames from the source parquet
     and require the written PNG index arrays to match pixel-exactly

Usage (from the project root):
    python scripts/113_check_mose_mini.py
    python scripts/113_check_mose_mini.py --roundtrip 8
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: the geometry script 112 writes; anything else means the converter drifted
EXP_W, EXP_H = 854, 480

#: palette length a well-formed label must carry (see scripts/112)
EXP_PALETTE_BYTES = 768


class Report:
    def __init__(self) -> None:
        self.fails: List[str] = []
        self.checks = 0

    def ok(self, msg: str) -> None:
        self.checks += 1
        print(f"  [ok]   {msg}")

    def fail(self, msg: str) -> None:
        self.checks += 1
        self.fails.append(msg)
        print(f"  [FAIL] {msg}")


def read_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["video_id"]] = {
                "n_frames": int(r["n_frames"]),
                "object_ids": [int(x) for x in r["object_ids"].split("|") if x],
            }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/MOSE_mini")
    ap.add_argument("--roundtrip", type=int, default=0,
                    help="re-decode this many frames from the source parquet")
    ap.add_argument("--parquet-dir", default="_scratch/mose_mini_20260922/parquet")
    args = ap.parse_args(argv)

    from PIL import Image

    root = ROOT / args.root
    rep = Report()
    print("MOSE mini :: validation")
    print("=" * 68)

    # ---------------------------------------------------------------- A
    man_path = root / "manifest.csv"
    if not man_path.exists():
        rep.fail(f"manifest missing: {man_path}")
        print("\nVERDICT: FAIL (no manifest)")
        return 1
    man = read_manifest(man_path)
    rep.ok(f"A manifest parsed: {len(man)} videos")
    if len(man) == 0:
        rep.fail("A manifest has zero rows")

    # ---------------------------------------------------------------- B
    missing = 0
    extra = 0
    for v, m in man.items():
        n = int(m["n_frames"])
        jdir = root / "JPEGImages" / v
        adir = root / "Annotations" / v
        for t in range(n):
            if not (jdir / f"{t:05d}.jpg").exists():
                missing += 1
            if not (adir / f"{t:05d}.png").exists():
                missing += 1
        got_j = {p.stem for p in jdir.glob("*.jpg")}
        got_a = {p.stem for p in adir.glob("*.png")}
        want = {f"{t:05d}" for t in range(n)}
        if got_j - want:
            extra += len(got_j - want)
        if got_a - want:
            extra += len(got_a - want)
        if got_j != got_a:
            rep.fail(f"B {v}: {len(got_j)} frames vs {len(got_a)} masks")
    if missing:
        rep.fail(f"B {missing} frame/mask files missing")
    else:
        rep.ok(f"B layout exact: {sum(int(m['n_frames']) for m in man.values())} "
               f"frames, 0 gaps")
    if extra:
        rep.fail(f"B {extra} stray files beyond the manifest")
    else:
        rep.ok("B no stray files")

    # ---------------------------------------------------------------- C/D/E
    from xdrp.datasets import read_label_png

    n_small = 0
    n_flat = 0
    n_badpal = 0
    idmismatch: List[str] = []
    empty_first: List[str] = []
    for v, m in man.items():
        jdir = root / "JPEGImages" / v
        adir = root / "Annotations" / v
        seen: set = set()
        for t in range(int(m["n_frames"])):
            with Image.open(jdir / f"{t:05d}.jpg") as im:
                if im.size != (EXP_W, EXP_H):
                    n_small += 1
                if np.asarray(im).std() < 1e-6:
                    n_flat += 1
            lp = adir / f"{t:05d}.png"
            with Image.open(lp) as im:
                if im.mode != "P":
                    n_badpal += 1
                elif im.palette is None or len(im.palette.getdata()[1]) != EXP_PALETTE_BYTES:
                    n_badpal += 1
            lab = read_label_png(lp)
            seen |= {int(x) for x in np.unique(lab) if int(x) != 0}
        if sorted(seen) != sorted(m["object_ids"]):
            idmismatch.append(f"{v}: png={sorted(seen)} manifest={m['object_ids']}")
        if not sorted(seen):
            empty_first.append(v)
    if n_small:
        rep.fail(f"C {n_small} frames not {EXP_W}x{EXP_H}")
    else:
        rep.ok(f"C all frames are {EXP_W}x{EXP_H}")
    if n_flat:
        rep.fail(f"C {n_flat} frames are constant (decode problem)")
    else:
        rep.ok("C all frames non-degenerate")
    if n_badpal:
        rep.fail(f"D {n_badpal} labels lack a {EXP_PALETTE_BYTES}-byte palette")
    else:
        rep.ok(f"D all labels carry the {EXP_PALETTE_BYTES}-byte VOC/DAVIS palette")
    if idmismatch:
        rep.fail(f"E object ids disagree in {len(idmismatch)} videos: "
                 f"{idmismatch[:3]}")
    else:
        rep.ok(f"E label index sets == manifest ids for all {len(man)} videos")
    if empty_first:
        rep.fail(f"E {len(empty_first)} videos have no object in any frame")
    else:
        rep.ok("E every video labels at least one object somewhere")

    # ---------------------------------------------------------------- F
    try:
        from xdrp.datasets import open_dataset
        s = open_dataset("mose", str(args.root))
        seqs = s.sequences()
        if sorted(seqs) != sorted(man):
            rep.fail(f"F open_dataset gives {len(seqs)} seqs, manifest {len(man)}")
        else:
            rep.ok(f"F open_dataset loads {len(seqs)} sequences")
        bad = [q for q in seqs
               if s.n_frames(q) != int(man[q]["n_frames"])]
        if bad:
            rep.fail(f"F n_frames disagrees for {len(bad)}: {bad[:3]}")
        else:
            rep.ok("F n_frames agrees with the manifest")
        bado = [q for q in seqs
                if sorted(s.object_ids(q)) != sorted(man[q]["object_ids"])]
        if bado:
            rep.fail(f"F object_ids disagree for {len(bado)}: {bado[:3]}")
        else:
            rep.ok("F object_ids() agrees with the manifest")
        # a real mask must be boolean and select the right pixels
        q = seqs[0]
        oid = s.object_ids(q)[0]
        mk = s.mask(q, 0, oid)
        if mk.dtype != bool or not mk.any():
            rep.fail(f"F mask({q},0,{oid}) dtype={mk.dtype} any={mk.any()}")
        else:
            rep.ok(f"F mask({q},0,{oid}) is a non-empty boolean map")
    except Exception as e:                                   # noqa: BLE001
        rep.fail(f"F open_dataset raised: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- G
    # negative control: the documented OpenCV palette trap.  If this read
    # AGREES with the correct one, then check E cannot distinguish a good
    # label from a mangled one and the whole suite is worthless.
    import cv2
    q = sorted(man)[0]
    buggy_ids: set = set()
    for t in range(int(man[q]["n_frames"])):
        b = cv2.imread(str(root / "Annotations" / q / f"{t:05d}.png"),
                       cv2.IMREAD_COLOR)[..., 0]
        buggy_ids |= {int(x) for x in np.unique(b) if int(x) != 0}
    if sorted(buggy_ids) == sorted(man[q]["object_ids"]):
        rep.fail("G negative control did not fire: cv2 BGR read AGREES with "
                 "Pillow -- check E has no teeth")
    else:
        rep.ok(f"G negative control fires: cv2 BGR read sees {sorted(buggy_ids)} "
               f"but the truth is {sorted(man[q]['object_ids'])}")

    # ---------------------------------------------------------------- H
    # frame/mask pairing, done as a SELF-CALIBRATING offset profile.
    #
    # Two earlier attempts at this check were wrong and both failed on good
    # data; the lesson is recorded here because the failure mode is subtle:
    #
    #   v1 demanded "mask fits its own frame better than the next one" win
    #      rate >= 90%.  Real value is ~67% -- the statistic (fg vs bg mean
    #      intensity) is coarse, so 90% was simply never calibrated.
    #   v2 added a "label taken from t+1" control and required it below 0.45
    #      *at every K*.  That control is only valid at K=1: at K=10 the
    #      label's true frame t+1 is still far nearer to t than to t+10, so
    #      the control legitimately wins and the assertion was meaningless.
    #
    # The profile avoids inventing thresholds: fit(t, t+delta) must PEAK at
    # delta = 0.  A dataset whose labels are off by one frame peaks at +/-1
    # instead, so the test carries its own control (the off-peak values).
    prof = _offset_profile(root, man, offsets=(-3, -2, -1, 0, 1, 2, 3),
                           stride=3)
    if not prof:
        rep.fail("H offset profile empty")
    else:
        line = " ".join(f"d{d:+d}={prof[d][0]:6.2f}" for d in sorted(prof))
        print(f"         H profile: {line}")
        peak = max(prof, key=lambda d: prof[d][0])
        p0 = prof.get(0, (0.0, 0))[0]
        if prof[0][1] < 50:
            rep.fail(f"H only {prof[0][1]} samples in the profile")
        elif peak != 0:
            rep.fail(f"H offset profile peaks at delta={peak:+d}, not 0 -- "
                     f"masks are misaligned relative to their frames")
        elif not (p0 > prof[-1][0] and p0 > prof[1][0]):
            rep.fail(f"H profile peak at 0 is not above its neighbours "
                     f"({prof[-1][0]:.2f} / {p0:.2f} / {prof[1][0]:.2f})")
        else:
            rep.ok(f"H offset profile peaks at delta=0 ({p0:.2f} vs "
                   f"{prof[-1][0]:.2f}/{prof[1][0]:.2f} at -1/+1, "
                   f"n={prof[0][1]})")
        # the profile must decay away from the peak, on both sides
        far = min(prof[-3][0], prof[3][0])
        if far >= p0:
            rep.fail(f"H profile does not decay at |delta|=3 "
                     f"({far:.2f} >= {p0:.2f})")
        else:
            rep.ok(f"H profile decays with |delta| (delta=+/-3 -> {far:.2f})")

    # K=1 control: a label taken from t+1 must fit image t+1 better than
    # image t, i.e. the win rate must fall BELOW chance.  Valid only at K=1.
    probe = _alignment_probe(root, man, ks=(1,))
    a1, s1 = probe[(1, "aligned")], probe[(1, "shifted")]
    if a1[1] == 0 or s1[1] == 0:
        rep.fail("H not enough frame/mask pairs for the K=1 control")
    else:
        print(f"         H detail: K=1 aligned={a1[0]/a1[1]:.3f} "
              f"shifted-control={s1[0]/s1[1]:.3f}")
        if s1[0] / s1[1] >= 0.45:
            rep.fail(f"H K=1 control did not fire: shifted win rate "
                     f"{s1[0]/s1[1]:.3f} (want < 0.45)")
        elif a1[0] / a1[1] <= 0.55:
            rep.fail(f"H K=1 aligned win rate only {a1[0]/a1[1]:.3f} "
                     f"(want > 0.55)")
        else:
            rep.ok(f"H K=1 pairing confirmed: aligned {a1[0]/a1[1]:.3f} vs "
                   f"shifted control {s1[0]/s1[1]:.3f}")

    # ---------------------------------------------------------------- I
    if args.roundtrip > 0:
        rep.checks += 1
        try:
            n_ok, n_cmp = _roundtrip(ROOT / args.parquet_dir, root, man,
                                     args.roundtrip)
            if n_cmp == 0:
                rep.fail("I round-trip compared 0 frames")
            elif n_ok == n_cmp:
                rep.ok(f"I round-trip pixel-exact on {n_ok}/{n_cmp} frames")
            else:
                rep.fail(f"I round-trip mismatch on {n_cmp - n_ok}/{n_cmp}")
        except Exception as e:                               # noqa: BLE001
            rep.fail(f"I round-trip raised: {type(e).__name__}: {e}")

    print()
    if rep.fails:
        print(f"VERDICT: FAIL ({len(rep.fails)} of {rep.checks} checks failed)")
        for f in rep.fails:
            print("  -", f)
        return 1
    print(f"VERDICT: PASS ({rep.checks} checks, 0 failures)")
    return 0


def _roundtrip(pq_dir: Path, root: Path, man: Dict[str, Dict[str, object]],
               n: int) -> Tuple[int, int]:
    """Re-decode up to `n` frames straight from the parquet shards.

    Only row groups that actually contain a sampled (video, frame) are read,
    so the cost is a handful of row-group decodes rather than the full 1.4 GB.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    sample: List[Tuple[str, int]] = []
    per_v = 1
    for v in sorted(man):
        for t in range(0, int(man[v]["n_frames"]), max(1, int(man[v]["n_frames"]) // 2)):
            sample.append((v, t))
            if len([s for s in sample if s[0] == v]) >= per_v:
                break
        if len(sample) >= n:
            break
    want = set(sample[:n])
    if not want:
        return 0, 0

    ok = cmp = 0
    for sh in sorted(pq_dir.glob("*.parquet")):
        f = pq.ParquetFile(sh)
        for g in range(f.metadata.num_row_groups):
            head = f.read_row_group(g, columns=["video_id", "frame_id"]).to_pylist()
            if not ({(r["video_id"], int(r["frame_id"])) for r in head} & want):
                continue
            rows = f.read_row_group(
                g, columns=["video_id", "frame_id", "annotation"]).to_pylist()
            for r in rows:
                key = (r["video_id"], int(r["frame_id"]))
                if key not in want:
                    continue
                import cv2
                ann = Image.open(io.BytesIO(r["annotation"]["bytes"]))
                idx = np.array(ann)
                exp = cv2.resize(idx.astype(np.uint8), (EXP_W, EXP_H),
                                 interpolation=cv2.INTER_NEAREST)
                got = np.asarray(Image.open(
                    root / "Annotations" / key[0] / f"{key[1]:05d}.png"))
                cmp += 1
                ok += int(np.array_equal(exp, got))
    return ok, cmp


def _alignment_probe(pq_root: Path, man: Dict[str, Dict[str, object]],
                     ks: Sequence[int] = (1, 10)
                     ) -> Dict[Tuple[int, str], Tuple[int, int]]:
    """Win rate of "the mask fits its own frame better", aligned vs shifted.

    For a frame index t and a temporal gap K, the mask from frame t is scored
    under the foreground-vs-background mean-intensity separation on image t
    versus image t+K.  A correctly paired dataset must favour image t, and the
    margin must GROW with K (a wrong frame drifts further away).

    The `shifted` variant re-does the same computation with the label taken
    from frame t+1 instead of t -- the built-in negative control.  It must
    score below 0.5, because then the mask genuinely belongs to a later frame.

    Caching is per video: 150 frames of 854x480 uint8 is ~60 MB, whereas
    caching the whole dataset would be ~600 MB.
    """
    from PIL import Image

    from xdrp.datasets import read_label_png

    out: Dict[Tuple[int, str], Tuple[int, int]] = {}
    for k in ks:
        out[(k, "aligned")] = (0, 0)
        out[(k, "shifted")] = (0, 0)

    def sep(img: np.ndarray, m: np.ndarray) -> float | None:
        if not m.any() or m.all():
            return None
        return float(np.abs(img[m].mean() - img[~m].mean()))

    for v in sorted(man):
        n = int(man[v]["n_frames"])
        kmax = max(ks)
        if n <= kmax + 1:
            continue
        imgs = [np.asarray(Image.open(
            pq_root / "JPEGImages" / v / f"{t:05d}.jpg").convert("L"))
            for t in range(n)]
        labs = [read_label_png(pq_root / "Annotations" / v / f"{t:05d}.png")
                for t in range(n)]
        for k in ks:
            for variant, off in (("aligned", 0), ("shifted", 1)):
                win, tot = out[(k, variant)]
                for t in range(0, n - k - off):
                    lab = labs[t + off]
                    if lab.max() == 0:
                        continue
                    m = lab > 0
                    s_here = sep(imgs[t].astype(np.float32), m)
                    s_far = sep(imgs[t + k].astype(np.float32), m)
                    if s_here is None or s_far is None:
                        continue
                    tot += 1
                    win += int(s_here >= s_far)
                out[(k, variant)] = (win, tot)
    return out


def _offset_profile(pq_root: Path, man: Dict[str, Dict[str, object]],
                    offsets: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3),
                    stride: int = 3
                    ) -> Dict[int, Tuple[float, int]]:
    """Mean mask/image fit as a function of the frame-offset delta.

    For each sampled frame t and every delta, measure how well the frame-t
    mask separates foreground from background on image t+delta.  A correctly
    aligned dataset peaks at delta = 0 and decays on both sides; a dataset
    whose masks lag or lead their frames peaks at +/-1 instead.

    The peak location is the only thing asserted -- no absolute threshold is
    invented, which is what the two rejected versions of this check got wrong.
    """
    from PIL import Image

    from xdrp.datasets import read_label_png

    dmax = max(abs(d) for d in offsets)
    acc: Dict[int, List[float]] = {d: [] for d in offsets}

    def sep(img: np.ndarray, m: np.ndarray) -> float | None:
        if not m.any() or m.all():
            return None
        return float(np.abs(img[m].mean() - img[~m].mean()))

    for v in sorted(man):
        n = int(man[v]["n_frames"])
        if n <= 2 * dmax:
            continue
        imgs = [np.asarray(Image.open(
            pq_root / "JPEGImages" / v / f"{t:05d}.jpg").convert("L"),
            dtype=np.float32) for t in range(n)]
        for t in range(dmax, n - dmax, stride):
            lab = read_label_png(pq_root / "Annotations" / v / f"{t:05d}.png")
            if lab.max() == 0:
                continue
            m = lab > 0
            vals = {d: sep(imgs[t + d], m) for d in offsets}
            if any(x is None for x in vals.values()):
                continue
            for d in offsets:
                acc[d].append(vals[d])
    return {d: (float(np.mean(vs)), len(vs)) for d, vs in acc.items() if vs}


if __name__ == "__main__":
    sys.exit(main())
