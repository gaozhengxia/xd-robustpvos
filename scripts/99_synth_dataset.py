"""99 (dev utility) - Generate a tiny synthetic dataset in DAVIS directory layout.

Purpose: let you smoke test the ENTIRE chain (01 -> 02 -> 03 -> 04 -> 05 -> 06)
without downloading DAVIS-2017. The dataset is deliberately trivial (moving
shapes on textured backgrounds) -- it validates the plumbing, NOT the science.

    python scripts/99_synth_dataset.py --root data/SYNTH --sequences 4 --frames 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_sequence(h: int, w: int, T: int, seed: int, shape: str):
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 2.5)
    bg = np.clip(bg * 0.30 + 0.28, 0, 1)
    frames, labels = [], []
    for t in range(T):
        f = bg.copy()
        m = np.zeros((h, w), np.uint8)
        cx = int(w * (0.22 + 0.55 * (t / max(1, T - 1))))
        cy = int(h * (0.50 + 0.22 * np.sin(t / 3.0)))
        if shape == "circle":
            cv2.circle(f, (cx, cy), 16, (0.90, 0.86, 0.25), -1)
            cv2.circle(m, (cx, cy), 16, 1, -1)
        else:
            cv2.rectangle(f, (cx - 14, cy - 11), (cx + 14, cy + 11),
                          (0.85, 0.20, 0.20), -1)
            cv2.rectangle(m, (cx - 14, cy - 11), (cx + 14, cy + 11), 1, -1)
        # a second distractor object (object id 2) so multi-object paths get exercised
        m2 = np.zeros((h, w), np.uint8)
        dx = int(w * (0.78 - 0.30 * (t / max(1, T - 1))))
        dy = int(h * 0.22)
        cv2.circle(f, (dx, dy), 9, (0.25, 0.45, 0.90), -1)
        cv2.circle(m2, (dx, dy), 9, 1, -1)
        lab = m.astype(np.uint8) + (m2.astype(np.uint8) * 2)
        frames.append(np.clip(f, 0, 1))
        labels.append(lab)
    return frames, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "SYNTH"))
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--width", type=int, default=208)
    args = ap.parse_args()

    base = Path(args.root)
    sets = base / "ImageSets" / "2017"
    (sets).mkdir(parents=True, exist_ok=True)
    names = [f"synth{i:02d}" for i in range(args.sequences)]
    for i, name in enumerate(names):
        shape = "circle" if i % 2 == 0 else "rect"
        frames, labels = make_sequence(args.height, args.width, args.frames,
                                       1000 + i, shape)
        idir = base / "JPEGImages" / "480p" / name
        adir = base / "Annotations" / "480p" / name
        idir.mkdir(parents=True, exist_ok=True)
        adir.mkdir(parents=True, exist_ok=True)
        for t, (f, l) in enumerate(zip(frames, labels)):
            cv2.imwrite(str(idir / f"{t:05d}.jpg"),
                        (f * 255).astype(np.uint8)[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(adir / f"{t:05d}.png"), l)
        print(f"  {name}: {args.frames} frames, 2 objects -> {idir}")
    (sets / "val.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (sets / "train.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"\nwrote synthetic DAVIS-layout dataset at {base}")
    print(f"use it with:  --dataset {base} --backend dummy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
