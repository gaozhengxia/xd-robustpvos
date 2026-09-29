"""Dataset readers for XD-RobustPVOS.

Supported layouts (all public, permissive for academic use):

DAVIS-2017
    <root>/JPEGImages/480p/<seq>/<5-digit>.jpg
    <root>/Annotations/480p/<seq>/<5-digit>.png      (0 = background, 1..N = objects)

MOSE
    <root>/JPEGImages/<seq>/<5-digit>.jpg
    <root>/Annotations/<seq>/<5-digit>.png

YouTube-VOS 2019
    <root>/<split>/JPEGImages/<seq>/<5-digit>.jpg
    <root>/<split>/Annotations/<seq>/<5-digit>.png   (palette-encoded; needs meta.json)
    <root>/<split>/meta.json

You only need ONE of these to start (DAVIS-2017 val is the recommended Tier-1 core:
30 sequences, small, clean licensing).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #

class FrameSource:
    """Minimal VOS reader interface."""

    name = "base"

    def sequences(self) -> List[str]:
        raise NotImplementedError

    def n_frames(self, seq: str) -> int:
        raise NotImplementedError

    def frame(self, seq: str, t: int) -> np.ndarray:
        raise NotImplementedError

    def label(self, seq: str, t: int) -> np.ndarray:
        raise NotImplementedError

    # ---- helpers shared by all readers ---------------------------------- #

    def object_ids(self, seq: str) -> List[int]:
        lab = self.label(seq, 0)
        ids = [int(v) for v in np.unique(lab) if int(v) != 0]
        return sorted(ids)

    def mask(self, seq: str, t: int, obj_id: int) -> np.ndarray:
        return self.label(seq, t) == int(obj_id)

    def first_mask(self, seq: str, obj_id: int) -> np.ndarray:
        return self.mask(seq, 0, obj_id)

    def describe(self) -> Dict[str, int]:
        seqs = self.sequences()
        return {"n_sequences": len(seqs),
                "n_frames": int(sum(self.n_frames(s) for s in seqs))}

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _list_subdirs(p: Path) -> List[str]:
    if not p.is_dir():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def _frames_in(p: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


def read_label_png(path: Path) -> np.ndarray:
    """Read a label PNG as an integer OBJECT-ID map (0 = background).

    Do NOT use `cv2.imread` for this. DAVIS/MOSE ship palette ("P" mode) PNGs
    whose *palette index* **is** the object id. OpenCV silently expands the
    palette into 3-channel BGR, and the obvious-looking `arr[..., 0]` does not
    recover the index:

        index 1 -> RGB (128, 0, 0) -> BGR (0, 0, 128) -> blue channel = 0
        index 4 -> RGB (0, 0, 128) -> BGR (128, 0, 0) -> blue channel = 128

    so objects 1..3 vanish entirely and object 4 is mistaken for "the" object.
    Measured on the DAVIS-2017 val download: a blue-channel read emptied the
    frame-0 annotation of 79/90 sequences, and of the 30 val sequences it
    produced non-empty labels for exactly 2 (`gold-fish`, `lab-coat`, the only
    two containing palette index 4). A week of baseline numbers was computed on
    those two sequences without any error being raised.

    Pillow returns the palette index directly, which is also what the official
    `davis2017-evaluation` package relies on, so we read labels with Pillow.
    """
    try:
        from PIL import Image
    except ImportError as e:      # pragma: no cover
        raise ImportError(
            "Reading DAVIS/MOSE labels requires Pillow: the annotations are "
            "palette PNGs whose palette INDEX is the object id, and OpenCV "
            "expands the palette instead of exposing the index.\n"
            "  pip install pillow\n"
            f"Original error: {e}"
        ) from e

    with Image.open(path) as im:
        mode = im.mode
        arr = np.asarray(im if mode == "P" else im.convert("L"))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int32)
    # A binary 0/255 map (DAVIS-2016-style) denotes a single object.
    if arr.size and arr.max() == 255 and not np.any((arr > 0) & (arr < 255)):
        arr = (arr > 0).astype(np.int32)
    return arr


# --------------------------------------------------------------------------- #
# DAVIS
# --------------------------------------------------------------------------- #

class DavisReader(FrameSource):
    name = "davis"

    def __init__(self, root: str, split: str = "val", resolution: str = "480p",
                 year: str = "2017"):
        self.root = Path(root)
        self.split = split
        self.res = resolution
        self.year = year
        # DAVIS 2017 trainval ships a single directory tree for both splits.
        cands = [self.root, self.root / "DAVIS", self.root / "davis"]
        self.base: Optional[Path] = None
        for c in cands:
            if (c / "JPEGImages").is_dir():
                self.base = c
                break
        if self.base is None:
            raise FileNotFoundError(
                f"DAVIS root not found under {root}. Expected <root>/JPEGImages/...")
        self._img_dir = self.base / "JPEGImages" / self.res
        self._ann_dir = self.base / "Annotations" / self.res
        if not self._img_dir.is_dir():
            raise FileNotFoundError(f"missing {self._img_dir}")
        if not self._ann_dir.is_dir():
            raise FileNotFoundError(f"missing {self._ann_dir}")
        self._split_file = self.base / "ImageSets" / f"{year}" / f"{split}.txt"
        self._cache: Dict[str, List[Path]] = {}

    def sequences(self) -> List[str]:
        all_seqs = _list_subdirs(self._img_dir)
        if self._split_file.is_file():
            want = [l.strip() for l in self._split_file.read_text().splitlines() if l.strip()]
            seqs = [s for s in all_seqs if s in set(want)]
            if seqs:
                return seqs
        return all_seqs

    def _files(self, seq: str) -> List[Path]:
        if seq not in self._cache:
            self._cache[seq] = _frames_in(self._img_dir / seq)
        return self._cache[seq]

    def n_frames(self, seq: str) -> int:
        return len(self._files(seq))

    def frame(self, seq: str, t: int) -> np.ndarray:
        return self._read_rgb(self._files(seq)[t])

    def label(self, seq: str, t: int) -> np.ndarray:
        p = self._ann_dir / seq / f"{t:05d}.png"
        if not p.exists():
            raise FileNotFoundError(f"cannot read annotation: {p}")
        return read_label_png(p)

    def frame_path(self, seq: str, t: int) -> Path:
        return self._files(seq)[t]


# --------------------------------------------------------------------------- #
# MOSE
# --------------------------------------------------------------------------- #

class MoseReader(FrameSource):
    name = "mose"

    def __init__(self, root: str, split: str = "valid"):
        self.root = Path(root)
        self.split = split
        cands = [self.root, self.root / split, self.root / ("val" if split == "valid" else split)]
        self.base: Optional[Path] = None
        for c in cands:
            if (c / "JPEGImages").is_dir():
                self.base = c
                break
        if self.base is None:
            raise FileNotFoundError(f"MOSE root not found under {root}")
        if not (self.base / "Annotations").is_dir():
            # MOSE ships a single annotation tree
            alt = self.base.parent / "Annotations"
            if alt.is_dir():
                self._ann_root = alt
            else:
                raise FileNotFoundError(f"missing MOSE annotations under {self.base}")
        else:
            self._ann_root = self.base / "Annotations"
        self._img_root = self.base / "JPEGImages"
        self._cache: Dict[str, List[Path]] = {}

    def sequences(self) -> List[str]:
        return _list_subdirs(self._img_root)

    def _files(self, seq: str) -> List[Path]:
        if seq not in self._cache:
            self._cache[seq] = _frames_in(self._img_root / seq)
        return self._cache[seq]

    def n_frames(self, seq: str) -> int:
        return len(self._files(seq))

    def frame(self, seq: str, t: int) -> np.ndarray:
        return self._read_rgb(self._files(seq)[t])

    def label(self, seq: str, t: int) -> np.ndarray:
        p = self._ann_root / seq / f"{t:05d}.png"
        if not p.exists():
            raise FileNotFoundError(f"cannot read annotation: {p}")
        return read_label_png(p)


# --------------------------------------------------------------------------- #
# YouTube-VOS
# --------------------------------------------------------------------------- #

class YouTubeVOSReader(FrameSource):
    name = "youtubevos"

    def __init__(self, root: str, split: str = "valid"):
        self.root = Path(root)
        self.split = split
        cands = [self.root / split, self.root]
        self.base: Optional[Path] = None
        for c in cands:
            if (c / "JPEGImages").is_dir():
                self.base = c
                break
        if self.base is None:
            raise FileNotFoundError(f"YouTube-VOS root not found under {root}")
        self._img_root = self.base / "JPEGImages"
        self._ann_root = self.base / "Annotations"
        self._palette: Dict[str, Dict[int, int]] = {}
        meta = self.base / "meta.json"
        if meta.is_file():
            data = json.loads(meta.read_text())
            for cat, obj in data.get("objects", {}).items():
                for oid, info in obj.items():
                    rgb = info.get("palette", [0, 0, 0])
                    key = (int(rgb[0]) << 16) + (int(rgb[1]) << 8) + int(rgb[2])
                    self._palette.setdefault(cat, {})[key] = int(oid)
        self._cache: Dict[str, List[Path]] = {}

    def sequences(self) -> List[str]:
        return _list_subdirs(self._img_root)

    def _files(self, seq: str) -> List[Path]:
        if seq not in self._cache:
            self._cache[seq] = _frames_in(self._img_root / seq)
        return self._cache[seq]

    def n_frames(self, seq: str) -> int:
        return len(self._files(seq))

    def frame(self, seq: str, t: int) -> np.ndarray:
        return self._read_rgb(self._files(seq)[t])

    def label(self, seq: str, t: int) -> np.ndarray:
        p = self._ann_root / seq / f"{t:05d}.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read annotation: {p}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.int64)
        key = (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + rgb[..., 2]
        cat = seq.split("/")[0] if "/" in seq else ""
        lut = self._palette.get(cat, {})
        if not lut:
            # No meta.json: treat distinct colours as distinct objects.
            uniq = {int(k): i + 1 for i, k in enumerate(np.unique(key)) if k != 0}
        else:
            uniq = {k: lut.get(int(k), 0) for k in np.unique(key)}
        out = np.zeros(key.shape, np.int32)
        for k, v in uniq.items():
            if v:
                out[key == k] = int(v)
        return out


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #

READERS = {"davis": DavisReader, "mose": MoseReader, "youtubevos": YouTubeVOSReader}


def open_dataset(kind: str, root: str, split: str = "val", **kw) -> FrameSource:
    kind = kind.lower()
    if kind not in READERS:
        raise KeyError(f"unknown dataset kind '{kind}'; known: {list(READERS)}")
    if kind == "davis":
        return DavisReader(root, split=split, **kw)
    return READERS[kind](root, split=split, **kw)


def try_open_first(available: Dict[str, str], split: str = "val") -> FrameSource:
    """Open the first dataset in `available` that actually exists on disk."""
    errs = []
    for kind, root in available.items():
        try:
            return open_dataset(kind, root, split=split)
        except Exception as e:      # noqa: BLE001
            errs.append(f"{kind}@{root}: {e}")
    raise RuntimeError("no usable dataset found:\n  " + "\n  ".join(errs))
