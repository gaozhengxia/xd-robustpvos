#!/usr/bin/env python
"""112 - Turn the MOSEv2 mini subset (parquet) into the DAVIS-style layout.

WHY
---
`data/` held only DAVIS-2017 val, so the paper's "Cross-Domain" promise had no
second, independent VOS source behind it (user decision 2026-09-22: add one).
Channel survey run from this machine:

    huggingface.co      unreachable (direct AND via proxy)  -> FudanCVL/MOSEv2 out
    drive.google.com    unreachable                         -> gdown out
    zenodo.org          unreachable
    acdc.vision.ee.ethz.ch  reachable, but ACDC is 4006 annotated *images* for
                        semantic/panoptic segmentation -- ACDC has NO video
                        instance masks, so "ACDC-Video" exists only as [N1]'s own
                        derived release and cannot be fetched from ACDC itself.
    modelscope.cn / opendatalab.com / openxlab.org.cn    reachable

MOSE's *val* split publishes first-frame annotations only (the test server scores
the rest), so it can never yield a whole-clip J&F.  `merve/mosev2-mini` is a
120-video subset of the **train** split -- whose rows are
`(video_id, frame_id, image, annotation)` with the annotation present for EVERY
frame -- which is exactly what `xdrp.datasets.MoseReader` needs.

WHAT THIS SCRIPT DOES
---------------------
1. walks the parquet shards and inventories each video (frame count, contiguity);
2. selects a subset by a DETERMINISTIC, PERFORMANCE-BLIND rule (see below);
3. writes  data/MOSE_mini/JPEGImages/<video_id>/<5-digit>.jpg
           data/MOSE_mini/Annotations/<video_id>/<5-digit>.png
   so that `MoseReader(data/MOSE_mini)` finds them.

SUBSET RULE (must be quoted in the paper -- it must not be performance-based)
---------------------------------------------------------------------------
    contiguity : frame ids must be exactly 0..n-1 (else the video is dropped:
                 a gap would silently misalign every mask after it)
    length     : MIN_FRAMES <= n <= MAX_FRAMES   (default 25..150)
                 the lower bound keeps the clip long enough for a meaningful J&F;
                 the upper bound stops a handful of very long clips from
                 dominating the compute budget
    order      : sorted by video_id and taken from the front
    count      : N_VIDEOS (default 24)
    final      : any video whose annotations are empty (no object ever labelled)
                 is dropped AFTER conversion-time decoding

RESOLUTION
----------
MOSE frames are 1920x1080; the rest of this benchmark (DAVIS-2017 val) is
480p = 854x480.  We downscale to 854x480 -- the aspect ratio is identical
(16:9), so this is a pure scale, and it keeps the per-pixel tolerance used by
`xdrp.metrics` (dilation = 2 px, defined AT 480p) meaningful.  Frames use
LANCZOS, masks use NEAREST (a mask must keep integer ids).

MASK ENCODING
-------------
Annotations are written as palette ("P") PNGs whose palette index IS the object
id, because that is what `read_label_png` documents and what DAVIS/MOSE ship.
The source PNGs are already P-mode; we resize the *index array* with NEAREST
rather than resizing the PIL image, so ids cannot be interpolated.

Usage (from the project root):
    python scripts/112_prepare_mose_mini.py --inventory
    python scripts/112_prepare_mose_mini.py --convert
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: geometry of the rest of the benchmark (DAVIS-2017 val 480p)
OUT_W, OUT_H = 854, 480

#: subset rule constants -- see the module docstring
MIN_FRAMES = 25
MAX_FRAMES = 150
N_VIDEOS = 24

#: licence inherited from the official dataset (CC BY-NC-SA 4.0)
LICENCE = "CC BY-NC-SA 4.0 (non-commercial academic use only)"


def voc_palette() -> bytes:
    """The 256-entry VOC/DAVIS colormap as 768 raw RGB bytes.

    Index 0 -> black, 1 -> (128,0,0), 2 -> (0,128,0), 4 -> (0,0,128), ...

    WHY THIS IS HERE (bug found 2026-09-22)
    ---------------------------------------
    `Image.fromarray(idx, mode="P")` produces an image whose *index array* is
    correct but which carries **no palette at all**.  Every metric in this repo
    reads labels with `read_label_png`, which takes `np.asarray(im)` -- i.e. the
    index array -- so the numbers were right and nothing raised.  But the files
    were unusable to anything else: `cv2.imread` saw all-black, so the
    qualitative panels and any reviewer tooling would have shown nothing.

    Verified against the in-repo DAVIS-2017 annotations
    (`data/DAVIS/Annotations/480p/*/*.png`): they carry this exact 768-byte
    palette, matching the classic VOC colour-map rule
    (bit-expand index i, 3 bits at a time, MSB first).  Generating it from the
    rule keeps the constant auditable instead of pasting a 768-byte blob.
    """
    def bitget(byteval: int, idx: int) -> int:
        return (byteval >> idx) & 1

    out = bytearray()
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        out += bytes((r, g, b))
    return bytes(out)


PALETTE = voc_palette()
assert len(PALETTE) == 768, len(PALETTE)


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _shards(pq_dir: Path) -> List[Path]:
    got = sorted(pq_dir.glob("*.parquet"))
    if not got:
        raise SystemExit(f"no parquet shards under {pq_dir}")
    return got


def inventory(pq_dir: Path) -> Tuple[Dict[str, List[int]], List[Path], Dict[int, List[str]]]:
    """Frame counts per video, without decoding any image bytes.

    Also returns, per row group, the video ids it contains, so the conversion
    pass can read only the row groups it needs instead of the whole shard.
    """
    import pyarrow.parquet as pq

    per_video: Dict[str, List[int]] = {}
    groups: Dict[int, List[str]] = {}
    shards = _shards(pq_dir)
    for sh in shards:
        f = pq.ParquetFile(sh)
        for g in range(f.metadata.num_row_groups):
            t = f.read_row_group(g, columns=["video_id", "frame_id"])
            vids = t.column("video_id").to_pylist()
            fids = t.column("frame_id").to_pylist()
            groups[(hash(sh.name), g)] = sorted(set(vids))   # type: ignore[index]
            for v, fid in zip(vids, fids):
                per_video.setdefault(v, []).append(int(fid))
    for v in per_video:
        per_video[v] = sorted(per_video[v])
    return per_video, shards, groups


def select(per_video: Dict[str, List[int]], n_videos: int,
           min_frames: int, max_frames: int) -> Tuple[List[str], List[Tuple[str, str, int]]]:
    """Deterministic, performance-blind subset selection."""
    picked: List[str] = []
    dropped: List[Tuple[str, str, int]] = []
    for v in sorted(per_video):
        fids = per_video[v]
        n = len(fids)
        if fids != list(range(n)):
            dropped.append((v, "non-contiguous frame ids", n))
            continue
        if n < min_frames:
            dropped.append((v, "too short", n))
            continue
        if n > max_frames:
            dropped.append((v, "too long", n))
            continue
        picked.append(v)
        if len(picked) >= n_videos:
            break
    return picked, dropped


def convert(pq_dir: Path, out_root: Path, picked: Sequence[str],
            per_video: Dict[str, List[int]]) -> List[Dict[str, object]]:
    import pyarrow.parquet as pq
    from PIL import Image

    want = set(picked)
    (out_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (out_root / "Annotations").mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    written: Dict[str, int] = {v: 0 for v in picked}
    labels: Dict[str, set] = {v: set() for v in picked}

    for sh in _shards(pq_dir):
        f = pq.ParquetFile(sh)
        for g in range(f.metadata.num_row_groups):
            # skip row groups with no selected video (cheap scalar read first)
            head = f.read_row_group(g, columns=["video_id"]).column("video_id").to_pylist()
            if not (set(head) & want):
                continue
            rows = f.read_row_group(g, columns=["video_id", "frame_id", "image",
                                                "annotation"]).to_pylist()
            for r in rows:
                v = r["video_id"]
                if v not in want:
                    continue
                t = int(r["frame_id"])
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                ann = Image.open(io.BytesIO(r["annotation"]["bytes"]))
                idx = np.array(ann)                       # palette index = object id
                if idx.ndim != 2:
                    raise SystemExit(f"{v}#{t}: annotation is not single-channel")
                labels[v].update(int(x) for x in np.unique(idx) if int(x) != 0)

                # resize: frames by area, masks by nearest on the INDEX array
                import cv2
                img_s = cv2.resize(np.array(img), (OUT_W, OUT_H),
                                   interpolation=cv2.INTER_AREA)
                idx_s = cv2.resize(idx.astype(np.uint8), (OUT_W, OUT_H),
                                   interpolation=cv2.INTER_NEAREST)

                idir = out_root / "JPEGImages" / v
                adir = out_root / "Annotations" / v
                idir.mkdir(parents=True, exist_ok=True)
                adir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img_s[:, :, ::-1]).save(      # BGR -> RGB
                    idir / f"{t:05d}.jpg", quality=95, subsampling=0)

                # P-mode label: index array carries the object id, and the
                # palette must be attached explicitly (see `voc_palette`).
                mask_png = Image.fromarray(idx_s, mode="P")
                mask_png.putpalette(PALETTE)
                lbl_path = adir / f"{t:05d}.png"
                mask_png.save(lbl_path)

                # Guard: the label must survive the file round-trip with its
                # integer ids intact AND a real palette attached.  Without the
                # palette the file is all-black to every non-Pillow reader,
                # which is exactly the failure this guard exists to catch.
                with Image.open(lbl_path) as back:
                    if back.palette is None or len(back.palette.getdata()[1]) != 768:
                        raise SystemExit(
                            f"{v}#{t}: written label has no 768-byte palette")
                    if not np.array_equal(np.asarray(back), idx_s):
                        raise SystemExit(
                            f"{v}#{t}: label index array changed on write")
                written[v] += 1

    for v in picked:
        if written[v] != len(per_video[v]):
            raise SystemExit(
                f"{v}: wrote {written[v]} frames but inventory says "
                f"{len(per_video[v])} -- aborting rather than shipping a video "
                f"whose frames and masks disagree")
        if not labels[v]:
            print(f"  [drop] {v}: annotations are empty in every frame")
            continue
        manifest.append({
            "video_id": v,
            "n_frames": written[v],
            "object_ids": sorted(labels[v]),
            "n_objects": len(labels[v]),
        })
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet-dir", default="_scratch/mose_mini_20260922/parquet")
    ap.add_argument("--out-root", default="data/MOSE_mini")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--n-videos", type=int, default=N_VIDEOS)
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--convert", action="store_true")
    args = ap.parse_args(argv)

    pq_dir = ROOT / args.parquet_dir
    out_root = ROOT / args.out_root
    if not pq_dir.is_dir():
        raise SystemExit(f"parquet dir not found: {pq_dir}")

    print("MOSE mini :: prepare")
    print("=" * 68)
    per_video, shards, _ = inventory(pq_dir)
    picked, dropped = select(per_video, args.n_videos, args.min_frames,
                             args.max_frames)
    print(f"  shards        : {len(shards)}")
    for sh in shards:
        print(f"    {sh.name}  {sh.stat().st_size:>12,} B  sha256={_sha256(sh)[:16]}...")
    print(f"  videos found  : {len(per_video)}")
    print(f"  length rule   : {args.min_frames} <= n <= {args.max_frames}")
    print(f"  selected      : {len(picked)}  -> {picked}")
    print(f"  dropped       : {len(dropped)}")
    for v, why, n in dropped[:12]:
        print(f"    {v}  n={n:<4} {why}")
    print(f"  frames total  : {sum(len(per_video[v]) for v in picked)}")

    if args.inventory and not args.convert:
        return 0

    manifest = convert(pq_dir, out_root, picked, per_video)
    prov = {
        "source": "FudanCVL/MOSEv2 (train split) via modelscope.cn/merve/mosev2-mini",
        "licence": LICENCE,
        "caveat": ("community 120-video subset, NOT the official 2149-video "
                   "release -- the paper must say so and cite MOSE (ICCV 2023) / "
                   "MOSEv2 (arXiv 2508.05630)"),
        "shards": [{"name": s.name, "bytes": s.stat().st_size,
                    "sha256": _sha256(s)} for s in shards],
        "subset_rule": {
            "contiguity": "frame ids must equal 0..n-1",
            "min_frames": args.min_frames, "max_frames": args.max_frames,
            "order": "sorted(video_id), take from the front",
            "n_videos": len(manifest),
        },
        "resolution": f"{OUT_W}x{OUT_H} (downscaled from 1920x1080; "
                      f"frame=INTER_AREA, mask=NEAREST)",
        "mask_encoding": ("P-mode PNG, palette index = object id, 768-byte "
                          "VOC/DAVIS colormap attached (same palette the "
                          "in-repo DAVIS-2017 annotations use)"),
        "videos": manifest,
    }
    (out_root / "PROVENANCE.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_id", "n_frames", "n_objects",
                                           "object_ids"])
        w.writeheader()
        for m in manifest:
            w.writerow({**m, "object_ids": "|".join(str(x) for x in m["object_ids"])})
    print()
    print(f"  wrote {len(manifest)} videos / "
          f"{sum(int(m['n_frames']) for m in manifest)} frames -> {out_root}")
    print(f"  provenance -> {out_root / 'PROVENANCE.json'}")
    print(f"  manifest   -> {out_root / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
