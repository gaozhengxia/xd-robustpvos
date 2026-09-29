"""127 - Build the complete Word (.docx) manuscript from `docs/MANUSCRIPT.md`.

`docs/MANUSCRIPT.md` stays the single source of truth.  This script only
*transforms* it and never re-enters, re-derives or re-types a number:

  1. front matter -> title    the drafting header (working title, file-order
                             note, status table, provenance note) is dropped,
                             and the paper title is READ out of the
                             working-title blockquote rather than retyped
  2. drafting notes removed   the Sec. 5 "Draft status" blockquote
  3. figures embedded         each drawn figure is inserted immediately after
                             the paragraph that introduces it, and every fact
                             in its caption is asserted against
                             `results/figs/FIGURES.json` -- the figure's owner
  4. markdown -> docx         through pandoc: TeX math becomes native Word
                             OMML (editable equations, not pictures), pipe
                             tables become real Word tables, images embedded

Fail-closed by construction: every structural assumption is asserted, the
product is re-opened and its structure compared with the source's, and a build
record is written.  A partial conversion therefore cannot be reported as a
build, and `--check` fails when the manuscript changes after the fact.

    python scripts/127_make_docx.py
    python scripts/127_make_docx.py --check      # rebuild to temp and compare
    python scripts/127_make_docx.py --self-test  # transform logic, no pandoc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "docs" / "MANUSCRIPT.md"
FIGMAN = ROOT / "results" / "figs" / "FIGURES.json"
OUTDIR = ROOT / "out"
DOCX = OUTDIR / "XD-RobustPVOS_manuscript.docx"
SIDE = OUTDIR / "XD-RobustPVOS_manuscript.build.json"
#: The submission copy: same content, journal numbering and typography.
DOCX_ELSEVIER = OUTDIR / "XD-RobustPVOS_manuscript_elsevier.docx"
SIDE_ELSEVIER = OUTDIR / "XD-RobustPVOS_manuscript_elsevier.build.json"

#: A general academic manuscript needs an author block and keywords; the
#: manuscript file carries neither.  The placeholder is deliberately explicit
#: so that it cannot be mistaken for real metadata at submission.
AUTHOR_PLACEHOLDER = ("*Authors and affiliations to be completed before "
                      "submission.*")
KEYWORDS = ("**Keywords:** promptable video object segmentation; robustness "
            "benchmark; compound degradation; memory bank; training-free "
            "restoration; evidential uncertainty.")

#: Every fact printed in a caption is asserted against the figure's own
#: manifest below, so a caption cannot drift away from the plate it describes.
FIG_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "F1",
        "anchor": "See Figure 1, which",
        "probe": "restoration operators; each restored frame is segmented",
        "png": "results/figs/F1_overview.png",
        "width": "6.0in",
        "caption": (
            "**Figure 1.** The DAG-RS arm. A degraded frame branches into "
            "$M = 13$ restoration operators; each restored frame is segmented, "
            "the $M$ candidate masks are scored for temporal consistency, "
            "candidate consensus and boundary stability, aggregated into a "
            "Dirichlet evidence vector and fused; the resulting vacuity "
            "$\\nu$ drives a gate (accept / reject / re-anchor) and updates "
            "the per-target degradation profile that is fed back at the next "
            "keyframe. The operator list drawn is read from "
            "`xdrp.operators.OperatorBank`, not transcribed, so the plate is "
            "reported stale if the bank changes."
        ),
    },
    {
        "key": "F2",
        "anchor": "Figure 2 and Table 14 are now generated from one code path",
        "probe": "The rank-binned median in each panel is shown",
        "png": "results/figs/F2_iqa_scatter.png",
        "width": "6.3in",
        "caption": (
            "**Figure 2.** Image quality against downstream "
            "$\\mathcal{J}\\&\\mathcal{F}$, eight metrics, one panel each. "
            "Unit: one point per (sequence, degradation) cell, arms pooled to "
            "the single `greedy` arm, clean cells excluded; $n = 360$ points "
            "(30 sequences $\\times$ 12 degradations). The rank-binned median "
            "in each panel is shown because a dense cloud otherwise reads as a "
            "trend; the medians are flat, the visual counterpart of "
            "$R^2 \\le 0.074$. Drawn by `scripts/124_fig_build.py`, which "
            "refuses to write the plate while it disagrees with Table 14."
        ),
    },
    {
        "key": "F3",
        "anchor": "**Table 17** and Figure 3 sweep severity",
        "probe": "levels 1-5, three arms",
        "png": "results/figs/F3_severity.png",
        "width": "6.3in",
        "caption": (
            "**Figure 3.** Severity curves for `fog`, `sensor_noise` and their "
            "compound `C1_fog_noise`, levels 1-5, three arms. Each cell is the "
            "sequence-equal mean over 30 sequences (45 cells); bands are 95% "
            "sequence bootstraps. The panel-title gap is the memory-bank arm "
            "minus the arm under test ($\\text{{{mem}}} - \\text{{{sub}}}$); "
            "the alternative subtrahend implied by the Sec. 5.2 definition "
            "($\\text{{{mem}}} - \\text{{{sens}}}$) is reported in Sec. 5.6 as "
            "a sensitivity, because the two disagree in sign on `fog` and "
            "`C1_fog_noise`."
        ),
    },
    {
        "key": "F4",
        "anchor": "the columns that actually appear",
        "probe": "white dashed outline in every row",
        "png": "results/figs/F4_qualitative.png",
        "width": "6.3in",
        # {n}, {sub} and {insts} are filled from FIGURES.json's F4 entry, so the
        # caption cannot name instances the plate does not show.
        "caption": (
            "**Figure 4.** Qualitative panel: clean input, degraded (`fog`, "
            "level 3.0) input, the memory-bank prediction and the arm under "
            "test, one frame per column. The three columns are the argmax, the "
            "median-nearest instance and the argmin of the `fog`-phase ranking "
            "of {n} instances by $\\Delta = "
            "\\mathcal{{J}}\\&\\mathcal{{F}}(\\text{{arm}}) - "
            "\\mathcal{{J}}\\&\\mathcal{{F}}(\\text{{{sub}}})$ "
            "(frame stride 1, seed 0): {insts}. Within each column the frame is "
            "the one where the two predictions differ most. Ground truth is "
            "drawn as a white dashed outline in every row, so the prediction "
            "rows can be read without a legend."
        ),
    },
    {
        "key": "F5",
        "anchor": ("which fails closed if the numbers it plots stop matching "
                   "the artifact they come from"),
        "probe": "Per-frame mechanism diagnostics",
        "png": "results/figs/F5_trajectory.png",
        "width": "6.3in",
        "caption": (
            "**Figure 5.** Per-frame mechanism diagnostics for one "
            "`C1_fog_noise` phase (instance `cows` object 1, 104 frames), "
            "chosen by a stated rule rather than by eye: the instance with the "
            "most decision frames. Top, the operator the degradation profile "
            "leads with, $\\arg\\max_m \\pi_{t,m}$. Middle, the vacuity "
            "$\\nu_t$ against the running threshold $\\tau_\\nu = 0.13$. "
            "Bottom, the frame's true error $1 - \\mathcal{J}_t$. **This is "
            "not** the profile's mass: the pipeline records "
            "$\\arg\\max_m \\pi$ and the number of operators evaluated, not the "
            "$\\pi$ vector, so the plate shows a one-dimensional *leader* "
            "trace -- weaker than migration of the profile's mass."
        ),
    },
)

#: (figure key, manifest field, expected value) -- read from the manifest, so
#: the caption's claims are checked against the plate's owner, not against the
#: caption itself.
CAPTION_FACT_CHECKS: Tuple[Tuple[str, str, Any], ...] = (
    ("F1", "n_operators", 13),
    ("F2", "n_points", 360),
    ("F2", "arms", ["greedy"]),
    ("F3", "n_cells", 45),
    ("F3", "n_sequences_per_cell", 30),
    ("F5", "n_frames", 104),
)

#: Numbers that must appear literally in the caption of a given figure.
CAPTION_NUMBER_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("F1", "13"), ("F2", "360"), ("F2", "30"), ("F3", "45"), ("F3", "30"),
    ("F5", "104"), ("F5", "0.13"),
)

PANDOC_ENV = "PANDOC"
PANDOC_FALLBACKS = (
    Path.home() / ".workbuddy" / "binaries" / "pandoc" / "pandoc.exe",
    Path.home() / ".workbuddy" / "binaries" / "pandoc" / "pandoc",
    ROOT / "_scratch" / "_pandoc_binary" / "pypandoc" / "files" / "pandoc.exe",
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p: Path) -> str:
    return sha1_bytes(p.read_bytes())


def norm_ws(s: str) -> str:
    """Collapse whitespace so an anchor survives a re-wrap of its paragraph."""
    return re.sub(r"\s+", " ", s).strip()


def strip_quote(line: str) -> str:
    t = line.lstrip()
    return t[1:].strip() if t.startswith(">") else t.strip()


def quote_runs(lines: List[str]) -> List[Tuple[int, int]]:
    """Half-open [start, end) index ranges of contiguous blockquote runs."""
    runs, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith(">"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith(">"):
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


# --------------------------------------------------------------------------- #
# 1 / 2 - transform
# --------------------------------------------------------------------------- #

def extract_title(lines: List[str]) -> str:
    """Read the paper title out of the working-title blockquote.

    Read, never retyped: if the blockquote is absent or no longer holds an
    italic span, the build stops instead of shipping a title nobody owns.
    """
    hits = [i for i, l in enumerate(lines) if "Working title" in l]
    if len(hits) != 1:
        raise SystemExit(
            "ABORT: expected exactly one 'Working title' line in the front "
            f"matter, found {len(hits)} -- the paper title is read from it, so "
            "it cannot be guessed")
    i = hits[0]
    run: List[str] = []
    j = i
    while j < len(lines) and lines[j].lstrip().startswith(">"):
        run.append(strip_quote(lines[j]))
        j += 1
    text = " ".join(run)
    k = text.find(":**")
    if k < 0:
        raise SystemExit("ABORT: the working-title blockquote has no ':**' "
                         "label terminator, so its italic span cannot be "
                         "isolated")
    rest = text[k + 3:].strip()
    # the blockquote continues past the title with the positioning paragraph,
    # so take the FIRST italic span rather than everything to the run's end
    m = re.match(r"^\*([^*]{20,})\*", rest)
    if not m or len(m.group(1)) < 60:
        raise SystemExit(
            "ABORT: the working-title blockquote does not open with a single "
            f"italic span of plausible length (got {rest[:70]!r})")
    return m.group(1).strip()


def split_front_matter(lines: List[str]) -> Tuple[str, List[str]]:
    hits = [i for i, l in enumerate(lines) if l.strip() == "## Abstract"]
    if len(hits) != 1:
        raise SystemExit(
            "ABORT: expected exactly one '## Abstract' heading, found "
            f"{len(hits)} -- the front matter cannot be located, so nothing "
            "may be stripped")
    i = hits[0]
    if not any("Working title" in l for l in lines[:i]):
        raise SystemExit(
            "ABORT: the block before '## Abstract' carries no working-title "
            "block, so it is not the drafting header and must not be deleted")
    return extract_title(lines[:i]), lines[i:]


def strip_draft_notes(lines: List[str]) -> Tuple[List[str], int]:
    """Drop the Sec. 5 drafting-status blockquote.  Returns (lines, removed)."""
    hits = [i for i, l in enumerate(lines)
            if l.lstrip().startswith("> **Draft status.")]
    if len(hits) != 1:
        raise SystemExit(
            "ABORT: expected exactly one '> **Draft status.' blockquote, found "
            f"{len(hits)} -- the drafting note must not silently stay in the "
            "product, nor must a paper blockquote be deleted in its place")
    i = hits[0]
    j = i
    while j < len(lines) and (lines[j].lstrip().startswith(">")
                              or not lines[j].strip()):
        j += 1
    while j > i and not lines[j - 1].strip():
        j -= 1
    return lines[:i] + lines[j:], j - i


def space_out_tables(lines: List[str]) -> List[str]:
    """Give every pipe table a blank line on both sides.

    Ten of the manuscript's tables are glued to the caption (or to the
    `<!-- MS-TABLE -->` anchor) directly above them.  A block-level construct
    cannot begin in the middle of a paragraph, so left as-is those tables are
    converted as literal text.  This is a purely mechanical re-spacing: no line
    is reordered, and no non-blank line is altered.
    """
    out: List[str] = []
    for l in lines:
        if out:
            prev = out[-1]
            prev_pipe = prev.lstrip().startswith("|")
            cur_pipe = l.lstrip().startswith("|")
            if cur_pipe and prev.strip() and not prev_pipe:
                out.append("")
            elif l.strip() and not cur_pipe and prev_pipe:
                out.append("")
        out.append(l)
    return out


def drop_rules(lines: List[str]) -> Tuple[List[str], int]:
    """Drop standalone `---` rules.

    The manuscript separates sections with them; in Word a rule becomes an
    empty bordered paragraph directly under the abstract.  Headings already do
    the separating, so they are removed rather than reproduced.  A table
    separator row is `|---|`, never a bare `---`, so no table is touched.
    """
    out = [l for l in lines if l.strip() != "---"]
    return out, len(lines) - len(out)


def insert_block_after(text: str, anchor: str, block: str, what: str) -> str:
    blocks = [b for b in re.split(r"\n{2,}", text)]
    hits = [i for i, b in enumerate(blocks) if anchor in norm_ws(b)]
    if len(hits) != 1:
        raise SystemExit(
            f"ABORT: {what} anchor matched {len(hits)} paragraph(s), expected "
            f"exactly 1: {anchor!r} -- refusing to place a plate at a guessed "
            "location")
    blocks.insert(hits[0] + 1, block)
    return "\n\n".join(blocks)


def figure_block(spec: Dict[str, Any], man: Dict[str, Any]) -> str:
    cap = caption_of(spec, man)
    for bad in ("|", "]"):
        if bad in cap:
            raise SystemExit(
                f"ABORT: caption of {spec['key']} contains {bad!r}, which "
                "would break the markdown image syntax or a table cell")
    # No `{width=...}` attribute: gfm does not support pandoc's
    # link_attributes, and an unparsed attribute stops the image from being
    # alone in its paragraph, which silently drops the caption.  The size is
    # therefore applied in `size_figures`, where it is checked.
    return f"![{cap}]({spec['png']})"


def caption_of(spec: Dict[str, Any], man: Dict[str, Any]) -> str:
    """The caption text, with the two *derived* captions substituted from the owner.

    FIG-3's whole point is that its gap is a difference, so the caption must
    name the subtrahend.  FIG-4's whole point is that its columns were chosen by
    a stated rule, so the caption must name the instances that rule landed on.
    Rather than restate either here and check them afterwards, both are read out
    of the manifest and formatted in, so a caption and its plate cannot disagree
    in the first place.
    """
    cap = spec["caption"]
    if spec["key"] == "F4":
        f4 = (man.get("figures") or {}).get("F4") or {}
        cols = f4.get("columns") or []
        # A panel whose columns are not named is a plate a reader must take on
        # faith; the whole selection rule exists so that they need not.
        if len(cols) != 3:
            raise SystemExit(
                "ABORT: FIGURES.json's F4 entry does not carry three columns, so "
                "FIG-4's caption would show a selection it does not name -- the "
                "exact defect this caption exists to prevent")
        sub, n = f4.get("subtrahend"), f4.get("n_instances_ranked")
        if not sub or not n:
            raise SystemExit(
                "ABORT: FIGURES.json's F4 entry declares no ranking subtrahend / "
                "instance count, so FIG-4's caption would state a ranking without "
                "saying what it ranked or against what")
        parts = []
        for c in cols:
            d = float(c["delta_vs_greedy"])
            parts.append(f"`{c['seq']}` object {int(c['obj_id'])} "
                         f"(${'+' if d >= 0 else '-'}{abs(d):.4f}$)")
        insts = ", ".join(parts[:-1]) + " and " + parts[-1]
        return cap.format(n=int(n), sub=sub, insts=insts)
    if spec["key"] != "F3":
        return cap
    f3 = (man.get("figures") or {}).get("F3") or {}
    mem = f3.get("memory_arm")
    sub = f3.get("subtrahend")
    sens = f3.get("subtrahend_sensitivity")
    if not mem or not sub:
        raise SystemExit(
            "ABORT: FIGURES.json does not declare F3.memory_arm / "
            "F3.subtrahend, so FIG-3's caption would print an unnamed gap -- "
            "the exact defect this caption exists to prevent")
    if mem == sub:
        raise SystemExit(
            f"ABORT: FIGURES.json declares the same arm ({mem!r}) as both "
            "memory arm and subtrahend, so FIG-3's gap would be identically "
            "zero and the caption would describe nothing")
    return cap.format(mem=mem, sub=sub, sens=sens or "the alternative")


def check_caption_facts(man: Dict[str, Any]) -> None:
    figs = man.get("figures") or {}
    for key, field, expect in CAPTION_FACT_CHECKS:
        got = (figs.get(key) or {}).get(field, "<absent>")
        if got != expect:
            raise SystemExit(
                f"ABORT: FIGURES.json says {key}.{field} = {got!r} but the "
                f"caption in this script assumes {expect!r} -- the caption and "
                "the plate are no longer describing the same run")
    caps = {s["key"]: caption_of(s, man) for s in FIG_SPECS}
    for key, needle in CAPTION_NUMBER_CHECKS:
        if needle not in caps[key]:
            raise SystemExit(
                f"ABORT: caption of {key} does not state {needle!r}, which "
                "the manifest says the figure shows")
    f3 = figs.get("F3") or {}
    for arm in (f3.get("subtrahend"), f3.get("memory_arm")):
        if arm and arm not in caps["F3"]:
            raise SystemExit(
                f"ABORT: FIG-3's caption does not name the subtrahend arm "
                f"{arm!r}; a gap plotted without its subtrahend named is the "
                "exact defect this caption exists to prevent")
    f5 = figs.get("F5") or {}
    inst = f5.get("instance") or {}
    if inst.get("seq") and inst["seq"] not in caps["F5"]:
        raise SystemExit(
            f"ABORT: FIG-5's caption does not name the instance "
            f"{inst['seq']!r} it is drawn from")
    f4 = figs.get("F4") or {}
    for c in (f4.get("columns") or []):
        d = float(c["delta_vs_greedy"])
        label = f"`{c['seq']}` object {int(c['obj_id'])}"
        value = "${}{:.4f}$".format("+" if d >= 0 else "-", abs(d))
        if label not in caps["F4"] or value not in caps["F4"]:
            raise SystemExit(
                f"ABORT: FIG-4's caption does not name column {label} {value}; "
                "a qualitative panel whose columns are not named in the caption "
                "cannot be checked against the instances the text discusses")


#: The body cites papers by working key; a journal wants running numbers.  The
#: lookarounds keep `[[TODO]]`-style drafting markers out of it: a double-bracket
#: placeholder is not a citation, and renumbering one would silently turn an
#: unfinished item into a reference to nothing.
#: A key may contain a hyphen (`[UW-VOS]`).  The first version of this regex did
#: not allow one, and the artifact silently dropped that citation: it was neither
#: renumbered nor listed, and the left-over check used the same pattern, so it
#: could not see the loss either -- a probe that shares the bug it is probing for.
CITE_RE = re.compile(r"(?<!\[)\[([A-Za-z][A-Za-z0-9_-]*)\](?!\])")
REFS_HEADING = "\n## References"

#: The typeset title page carries the author block, which is the one thing this
#: build genuinely cannot derive -- so it is marked for filling, never invented.
AUTHORS_ELSEVIER = ("*Author names, affiliations, corresponding author e-mail and CRediT roles "
                    "to be completed before submission.*")


def parse_references(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split into (body, [(key, entry), ...]).

    An entry is its label line plus every continuation line indented under it, so a
    wrapped bibliography record stays one record.  The drafting note that explains
    the *working keys* is dropped: the typeset document no longer has working keys.
    """
    if REFS_HEADING not in text:
        raise SystemExit("ABORT: no '## References' section, so the bibliography "
                         "cannot be renumbered")
    body, refs = text.split(REFS_HEADING, 1)
    lines = refs.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"^\[[A-Za-z]", l)), None)
    if start is None:
        raise SystemExit("ABORT: the References section holds no '[key]' entries")
    entries: List[Tuple[str, str]] = []
    cur: Optional[List[str]] = None
    for line in lines[start:]:
        m = re.match(r"^\[([A-Za-z][A-Za-z0-9_-]*)\]\s*(.*)$", line)
        if m:
            if cur:
                entries.append((cur[0], "\n".join(cur[1:])))
            cur = [m.group(1), m.group(2)]
        elif cur and line.strip():
            cur.append(line.rstrip())
    if cur:
        entries.append((cur[0], "\n".join(cur[1:])))
    return body, entries


def renumber_citations(text: str) -> Tuple[str, List[str], Dict[str, int]]:
    """Replace working keys with the journal's running numbers.

    The bibliography's own note says the keys "are not the final numbering: at
    typesetting the keys are replaced by the journal's running numbers in order of
    first appearance".  This is that step, done by machine and checked in both
    directions: a key cited with no entry aborts, and an entry never cited aborts,
    because either one means the list and the body have stopped describing the same
    paper.  Returns (body-without-bibliography, ordered reference lines, key->number).
    """
    body, entries = parse_references(text)
    order: List[str] = []
    for m in CITE_RE.finditer(body):
        if m.group(1) not in order:
            order.append(m.group(1))
    have = {k: e for k, e in entries}
    missing = [k for k in order if k not in have]
    if missing:
        raise SystemExit(f"ABORT: cited but not in the bibliography: {missing}")
    uncited = [k for k, _ in entries if k not in order]
    if uncited:
        raise SystemExit(f"ABORT: in the bibliography but never cited: {uncited}")
    num = {k: i + 1 for i, k in enumerate(order)}
    body = CITE_RE.sub(lambda m: f"[{num[m.group(1)]}]", body)
    # Adjacent citations read better merged.  The lookahead requires a pure number
    # on the other side, so this cannot reach a markdown link or an image, whose
    # bracket is followed by `(`.
    body = re.sub(r"\]\s*,\s*\[(?=\d+\])", ", ", body)
    refs = [f"[{num[k]}] {have[k]}" for k in order]
    return body, refs, num


def build_markdown(raw: str, man: Dict[str, Any],
                   title_out: Optional[List[str]] = None,
                   cite_out: Optional[List[Dict[str, Any]]] = None,
                   elsevier: bool = False) -> str:
    """The full transform, no I/O beyond the strings handed in."""
    lines = raw.splitlines()
    title, body = split_front_matter(lines)
    if title_out is not None:
        title_out.append(title)

    n_before = len(quote_runs(body))
    body, _removed = strip_draft_notes(body)
    n_after = len(quote_runs(body))
    if n_after != n_before - 1:
        raise SystemExit(
            f"ABORT: removing the drafting note took the blockquote count from "
            f"{n_before} to {n_after}; exactly one blockquote may be removed, "
            "so a paper blockquote is being deleted silently")

    body = space_out_tables(body)
    body, _n_rules = drop_rules(body)
    text = "\n".join(body)
    # keywords sit between the abstract and Sec. 1
    intro = [i for i, l in enumerate(text.splitlines())
             if l.strip() == "## 1. Introduction"]
    if len(intro) != 1:
        raise SystemExit(
            f"ABORT: expected exactly one '## 1. Introduction' heading, found "
            f"{len(intro)}, so the keyword line has no unambiguous home")
    tl = text.splitlines()
    k = intro[0]
    while k > 0 and not tl[k - 1].strip():
        k -= 1
    text = "\n".join(tl[:k] + ["", KEYWORDS] + tl[k:])

    check_caption_facts(man)
    for spec in FIG_SPECS:
        text = insert_block_after(text, spec["anchor"], figure_block(spec, man),
                                  f"FIG {spec['key']}")

    out = (AUTHORS_ELSEVIER if elsevier else AUTHOR_PLACEHOLDER) \
        + "\n\n" + text.strip() + "\n"
    if elsevier:
        body, refs, num = renumber_citations(out)
        if cite_out is not None:
            cite_out.append({"key_to_number": num, "order": list(num)})
        out = (body.strip() + "\n\n## References\n\n" + "\n\n".join(refs) + "\n")
    return out


# --------------------------------------------------------------------------- #
# 4 - convert
# --------------------------------------------------------------------------- #

def find_pandoc(explicit: Optional[str]) -> Optional[Path]:
    cands: List[Optional[Path]] = []
    if explicit:
        cands.append(Path(explicit))
    env = os.environ.get(PANDOC_ENV)
    if env:
        cands.append(Path(env))
    w = shutil.which("pandoc")
    if w:
        cands.append(Path(w))
    cands.extend(PANDOC_FALLBACKS)
    for c in cands:
        if c and c.is_file():
            return c
    return None


def pandoc_version(pandoc: Path) -> str:
    r = subprocess.run([str(pandoc), "--version"], capture_output=True,
                       text=True, timeout=120)
    return (r.stdout or "").splitlines()[0].strip()


def run_pandoc(pandoc: Path, md: Path, docx: Path, title: str) -> None:
    cmd = [str(pandoc),
           # gfm does not support pandoc's link_attributes, so image sizes are
           # NOT written as `{width=...}` (an unparsed attribute would keep the
           # image from being alone in its paragraph and silently drop the
           # caption); `size_figures` sets them on the product instead.
           "-f", "gfm+tex_math_dollars+implicit_figures",
           "-t", "docx",
           "-M", f"title={title}",
           "--resource-path", str(ROOT),
           "-o", str(docx), str(md)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=900)
    if r.returncode != 0:
        raise SystemExit(
            "ABORT: pandoc failed (exit %d)\n  cmd: %s\n  stderr:\n%s"
            % (r.returncode, " ".join(cmd), (r.stderr or "").strip()[:4000]))
    if not docx.is_file() or docx.stat().st_size == 0:
        raise SystemExit(f"ABORT: pandoc reported success but wrote no "
                         f"non-empty file at {docx}")


def count_tables(text: str) -> int:
    """Tables = separator rows that directly follow a header row."""
    ls = text.splitlines()
    return sum(1 for i, l in enumerate(ls)
               if re.match(r"^\|[\s:\-|]+\|\s*$", l.strip())
               and i > 0 and ls[i - 1].lstrip().startswith("|"))


def source_expectations(text: str) -> Dict[str, int]:
    headings = sum(1 for l in text.splitlines()
                   if re.match(r"^#{2,4}\s+\S", l))
    return {"tables": count_tables(text), "headings": headings,
            "images": text.count("![")}


def docx_text_parts(path: Path) -> List[str]:
    from docx import Document
    d = Document(str(path))
    parts = [p.text or "" for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    parts.append(p.text or "")
    return parts


def docx_structure(path: Path) -> Dict[str, Any]:
    from docx import Document
    d = Document(str(path))
    heads = sum(1 for p in d.paragraphs
                if (p.style.name or "").startswith("Heading"))
    parts = docx_text_parts(path)
    return {
        "tables": len(d.tables),
        "images": len(d.inline_shapes),
        "image_widths": [int(s.width or 0) for s in d.inline_shapes],
        "headings": heads,
        "paragraphs": len(d.paragraphs),
        "text_chars": sum(len(p.text or "") for p in d.paragraphs),
        "text_sha1": sha1_bytes("\n".join(norm_ws(x) for x in parts
                                         if x.strip()).encode("utf-8")),
    }


def size_figures(path: Path, specs: Tuple[Dict[str, Any], ...]) -> None:
    """Set each plate's width to its spec, keeping the aspect ratio.

    `d.inline_shapes` is in document order and these four plates are the only
    images in the file, so index i is plate i -- asserted rather than assumed.
    """
    from docx import Document
    from docx.shared import Inches
    d = Document(str(path))
    if len(d.inline_shapes) != len(specs):
        raise SystemExit(
            f"ABORT: sizing needs one image per figure, found "
            f"{len(d.inline_shapes)} for {len(specs)} figures")
    for shape, spec in zip(d.inline_shapes, specs):
        if not shape.width or not shape.height:
            raise SystemExit(
                f"ABORT: {spec['key']}'s image has no intrinsic size, so its "
                "aspect ratio cannot be preserved")
        ratio = float(shape.height) / float(shape.width)
        w = Inches(float(str(spec["width"]).replace("in", "")))
        shape.width = w
        shape.height = int(round(w * ratio))
    d.save(str(path))


def verify_captions(path: Path, man: Dict[str, Any]) -> None:
    """Every caption must be *visible* text in the product.

    A dropped caption is the quiet failure mode of this pipeline: the image is
    embedded and the counts still agree, so nothing else notices.
    """
    corpus = "\n".join(norm_ws(x) for x in docx_text_parts(path))
    missing = [s["key"] for s in FIG_SPECS
               if norm_ws(s["probe"]) not in corpus]
    if missing:
        raise SystemExit(
            f"ABORT: the captions of {', '.join(missing)} are not present as "
            "text in the product -- the image was embedded but its caption was "
            "lost, most likely because the reader did not parse the image "
            "attributes and no implicit figure formed")


def postprocess(path: Path, title: str, elsevier: bool = False) -> List[str]:
    """A4 paper, submission margins, a page number and the document title."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm

    notes: List[str] = []
    d = Document(str(path))
    for sec in d.sections:
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.54)
        sec.top_margin = sec.bottom_margin = Cm(2.54)
        try:
            p = sec.footer.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            begin = OxmlElement("w:fldChar")
            begin.set(qn("w:fldCharType"), "begin")
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = " PAGE "
            end = OxmlElement("w:fldChar")
            end.set(qn("w:fldCharType"), "end")
            for el in (begin, instr, end):
                run._r.append(el)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"page number not added: {exc!r}")
    d.core_properties.title = title
    if elsevier:
        # Submission conventions for an Elsevier journal: one serif face through
        # the whole document, and continuous line numbers, which editors ask for
        # on the review copy and which are painful to add by hand afterwards.
        try:
            normal = d.styles["Normal"]
            normal.font.name = "Times New Roman"
            rpr = normal.element.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                rpr.append(rfonts)
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rfonts.set(qn(attr), "Times New Roman")
            notes.append("font set to Times New Roman (submission copy)")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"font not set: {exc!r}")
        try:
            for sec in d.sections:
                ln = OxmlElement("w:lnNumType")
                ln.set(qn("w:countBy"), "1")
                ln.set(qn("w:restart"), "continuous")
                sec._sectPr.append(ln)
            notes.append("continuous line numbers enabled (submission copy)")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"line numbers not enabled: {exc!r}")
    d.save(str(path))
    return notes


def read_manifest() -> Dict[str, Any]:
    if not FIGMAN.is_file():
        raise SystemExit(f"ABORT: {FIGMAN} not found -- figures are owned by "
                         "scripts/124_fig_build.py, run it first")
    return json.loads(FIGMAN.read_text(encoding="utf-8"))


def build(out: Path, pandoc: Optional[Path],
          record: bool = True, elsevier: bool = False) -> Dict[str, Any]:
    raw_bytes = MS.read_bytes()
    raw = raw_bytes.decode("utf-8")
    man = read_manifest()
    title: List[str] = []
    cites: List[Dict[str, Any]] = []
    md_text = build_markdown(raw, man, title, cites, elsevier)
    title_s = title[0]

    if pandoc is None:
        raise SystemExit(
            "ABORT: no pandoc found. Pass --pandoc, set $%s, or install it "
            "into one of:\n  %s" % (PANDOC_ENV,
                                    "\n  ".join(str(p) for p in PANDOC_FALLBACKS)))

    exp = source_expectations(md_text)
    if exp["images"] != len(FIG_SPECS):
        raise SystemExit(f"ABORT: built markdown carries {exp['images']} image "
                         f"reference(s), expected {len(FIG_SPECS)}")
    if exp["tables"] < 15:
        raise SystemExit(f"ABORT: built markdown carries only {exp['tables']} "
                         "table(s); the manuscript has far more, so the table "
                         "syntax is probably no longer parsing")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="msdocx_") as td:
        td = Path(td)
        md_path = td / "manuscript.build.md"
        md_path.write_text(md_text, encoding="utf-8")
        raw_docx = td / "raw.docx"
        run_pandoc(pandoc, md_path, raw_docx, title_s)
        shutil.copyfile(raw_docx, out)
    notes = postprocess(out, title_s, elsevier)
    size_figures(out, FIG_SPECS)

    struct = docx_structure(out)
    verify_captions(out, man)
    if struct["images"] != len(FIG_SPECS):
        raise SystemExit(
            f"ABORT: the docx holds {struct['images']} embedded image(s), "
            f"expected {len(FIG_SPECS)} -- pandoc could not resolve every "
            "figure, so the product is incomplete")
    for name in ("tables", "headings"):
        if struct[name] != exp[name]:
            raise SystemExit(
                f"ABORT: the docx holds {struct[name]} {name} but the built "
                f"markdown declares {exp[name]} -- entities were dropped or "
                "merged during conversion")

    try:
        docx_rel = str(out.relative_to(ROOT)).replace("\\", "/")
    except ValueError:  # a rebuild into a temp dir, which --check does
        docx_rel = str(out)
    side = {
        "generated_by": "scripts/127_make_docx.py",
        "source": "docs/MANUSCRIPT.md",
        "source_sha1": sha1_bytes(raw_bytes),
        "figures_sha1": sha1_file(FIGMAN),
        "title": title_s,
        "pandoc": pandoc_version(pandoc),
        "pandoc_path": str(pandoc),
        "docx": docx_rel,
        "docx_sha1": sha1_file(out),
        "figures_embedded": [s["key"] for s in FIG_SPECS],
        "stripped": [
            "front matter (working title / file-order note / status table / "
            "provenance note)",
            "Sec. 5 '> **Draft status.**' blockquote",
        ],
        "structure": struct,
        "expected": exp,
        "citations": (cites[0] if cites else None),
        "elsevier": bool(elsevier),
        "notes": notes,
    }
    if record:
        (SIDE_ELSEVIER if elsevier else SIDE).write_text(
            json.dumps(side, indent=2) + "\n", encoding="utf-8")
    return side


# --------------------------------------------------------------------------- #
# --check
# --------------------------------------------------------------------------- #

def check(pandoc: Optional[Path], elsevier: bool = False) -> int:
    bad: List[str] = []
    if not DOCX.is_file():
        bad.append(f"no Word file at {DOCX}")
    side_path = SIDE_ELSEVIER if elsevier else SIDE
    if not side_path.is_file():
        bad.append(f"no build record at {side_path.name} -- run without --check")
    if bad:
        for b in bad:
            print(f"[127] FAIL: {b}")
        return 1
    side = json.loads(side_path.read_text(encoding="utf-8"))

    cur_src = sha1_file(MS)
    if cur_src != side["source_sha1"]:
        print("[127] FAIL: docs/MANUSCRIPT.md changed after the Word file was "
              "built -- the .docx is stale, rerun without --check")
        return 1
    cur_fig = sha1_file(FIGMAN)
    if cur_fig != side["figures_sha1"]:
        print("[127] FAIL: results/figs/FIGURES.json changed after the Word "
              "file was built -- rerun without --check")
        return 1

    # Which file this inspects must follow the variant: reading the general
    # artifact and comparing it against the submission record would verify the
    # wrong document entirely, and it only failed loudly here because the two
    # happen to differ by one paragraph.
    got = docx_structure(DOCX_ELSEVIER if elsevier else DOCX)
    for k, v in side["structure"].items():
        if got.get(k) != v:
            print(f"[127] FAIL: structure.{k} is {got.get(k)} but the build "
                  f"record says {v}")
            return 1
    for k, v in side["expected"].items():
        if got.get(k) != v:
            print(f"[127] FAIL: expected.{k} is {got.get(k)} but the build "
                  f"record says {v}")
            return 1

    if pandoc is not None:
        with tempfile.TemporaryDirectory(prefix="msdocx_chk_") as td:
            tmp = Path(td) / "rebuild.docx"
            rebuild = build(tmp, pandoc, record=False, elsevier=elsevier)
            for k in ("tables", "images", "headings", "text_sha1"):
                if rebuild["structure"][k] != side["structure"][k]:
                    print(f"[127] FAIL: a fresh rebuild differs on {k}: "
                          f"{rebuild['structure'][k]} vs "
                          f"{side['structure'][k]}")
                    return 1
            if rebuild["title"] != side["title"]:
                print("[127] FAIL: a fresh rebuild reads a different title")
                return 1
        print("[127] --check: rebuild is identical in structure and body text")
    else:
        print("[127] --check: pandoc unavailable, verified the build record "
              "against the landed file only")
    print("[127] VERDICT: PASS")
    return 0


# --------------------------------------------------------------------------- #
# --self-test
# --------------------------------------------------------------------------- #

def self_test() -> int:
    fails: List[str] = []
    checks = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(f"{name}  {detail}")
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))

    man = read_manifest()
    raw = MS.read_text(encoding="utf-8")
    title: List[str] = []
    md = build_markdown(raw, man, title)

    print("127 self-test")
    print("-" * 66)

    ok("the title is read from the working-title blockquote",
       title and title[0].startswith("XD-RobustPVOS") and len(title[0]) > 60,
       title[0][:48] + "..." if title else "no title read")
    ok("the drafting header is gone and the product opens on the author line",
       md.startswith(AUTHOR_PLACEHOLDER), md.splitlines()[0][:44])
    ok("the product still carries the abstract",
       "\n## Abstract\n" in md)
    ok("the Sec. 5 drafting note is gone",
       "**Draft status.**" not in md)
    ok("the paper blockquotes survive",
       "**Central hypothesis.**" in md and "**What this section is.**" in md)
    ok("keywords sit between the abstract and Sec. 1",
       md.index(KEYWORDS) > md.index("## Abstract")
       and md.index(KEYWORDS) < md.index("## 1. Introduction"))
    ok("each figure lands immediately after its anchor paragraph",
       all(figure_block(s, man) in md for s in FIG_SPECS))
    ok("no figure block was placed before its anchor",
       all(md.index(figure_block(s, man)) > md.index(s["anchor"])
           for s in FIG_SPECS))
    ok("a figure block is a bare implicit figure, with no attribute text",
       all("{width=" not in figure_block(s, man)
           and not re.search(r"\)\{", figure_block(s, man))
           and figure_block(s, man).endswith(")")
           and figure_block(s, man).startswith("![**Figure ")
           for s in FIG_SPECS),
       "image paths must be the last thing on the line, or the caption is lost")
    ok("every caption is asserted against FIGURES.json",
       all(needle in {s["key"]: caption_of(s, man) for s in FIG_SPECS}[k]
           for k, needle in CAPTION_NUMBER_CHECKS))
    ok("FIG-3's caption names both subtrahend arms",
       str((man["figures"]["F3"] or {}).get("subtrahend")) in md
       and str((man["figures"]["F3"] or {}).get("memory_arm")) in md)
    f4c = (man["figures"]["F4"] or {}).get("columns") or []
    ok("FIG-4's caption names all three columns the plate shows",
       len(f4c) == 3 and all(f"`{c['seq']}` object {int(c['obj_id'])}" in md
                             for c in f4c),
       ", ".join(f"{c['seq']}|{c['obj_id']}" for c in f4c))

    # --- the submission copy: working keys -> running numbers ----------------- #
    syn = ("body cites [B] first and [A] second\n\n## References\n\n"
           "[A] alpha, in: Venue, 2020.\n\n[B] beta, in: Venue, 2021.\n")
    sbody, srefs, snum = renumber_citations(syn)
    ok("citations are numbered by first appearance, not alphabetically",
       snum == {"B": 1, "A": 2}, str(snum))
    ok("the bibliography is emitted in citation order",
       srefs[0].startswith("[1] beta") and srefs[1].startswith("[2] alpha"),
       srefs[0][:48])
    hy = ("body cites [UW-VOS]\n\n## References\n\n[UW-VOS] a hyphenated key, "
          "in: Venue, 2019.\n")
    _b, hrefs, hnum = renumber_citations(hy)
    ok("a hyphenated key is renumbered and listed like any other",
       hnum == {"UW-VOS": 1} and hrefs[0].startswith("[1] a hyphenated key"),
       str(hnum))
    todo = ("body cites [A] and leaves [[TODO]] alone\n\n## References\n\n"
            "[A] alpha, in: Venue, 2020.\n")
    tbody, _tr, _tn = renumber_citations(todo)
    ok("a double-bracket drafting marker is not treated as a citation",
       "[[TODO]]" in tbody and "[1]" in tbody, tbody[:52])

    # negative controls: the guards must fire, not shrug.  Where the guard is
    # about structure the fixture is synthetic, so the control tests the guard
    # rather than the current state of the draft.
    def must_abort(fn, name: str, needle: str) -> None:
        try:
            fn()
        except SystemExit as exc:
            ok(name, needle in str(exc), str(exc)[:110])
        else:
            ok(name, False, "no abort was raised")

    dup = ("para one\n\nSee Figure 1, which draws it.\n\n"
           "See Figure 1, which draws it again.")
    must_abort(lambda: insert_block_after(dup, "See Figure 1, which", "x", "T"),
               "a duplicated anchor aborts instead of picking one", "matched 2")
    must_abort(lambda: insert_block_after(dup, "nowhere at all", "x", "T"),
               "a missing anchor aborts instead of guessing", "matched 0")
    must_abort(lambda: extract_title(["> something else"]),
               "a source with no working title aborts", "Working title")
    must_abort(lambda: split_front_matter(["## Abstract", "body"]),
               "a source with no drafting header aborts", "working-title")
    must_abort(lambda: check_caption_facts({"figures": {}}),
               "a manifest with no figure entries aborts", "FIGURES.json says")
    must_abort(lambda: renumber_citations("body cites [Z]\n\n## References\n\n"
                                          "[A] alpha, in: Venue, 2020.\n"),
               "a citation with no bibliography entry aborts",
               "cited but not in the bibliography")
    must_abort(lambda: renumber_citations("body cites [A]\n\n## References\n\n"
                                          "[A] alpha, in: Venue, 2020.\n\n"
                                          "[C] uncited, in: Venue, 2021.\n"),
               "a bibliography entry that is never cited aborts", "never cited")

    bad_man = json.loads(json.dumps(man))
    bad_man["figures"]["F3"].pop("subtrahend", None)
    must_abort(lambda: caption_of(FIG_SPECS[2], bad_man),
               "FIG-3 refuses to caption an un-declared subtrahend",
               "unnamed gap")
    same = json.loads(json.dumps(man))
    same["figures"]["F3"]["subtrahend"] = same["figures"]["F3"]["memory_arm"]
    must_abort(lambda: caption_of(FIG_SPECS[2], same),
               "FIG-3 refuses an arm declared as its own subtrahend",
               "identically zero")
    flipped = json.loads(json.dumps(man))
    flipped["figures"]["F3"]["n_cells"] = 44
    must_abort(lambda: check_caption_facts(flipped),
               "a caption whose cell count disagrees with the plate aborts",
               "FIGURES.json says")

    short4 = json.loads(json.dumps(man))
    short4["figures"]["F4"]["columns"] = short4["figures"]["F4"]["columns"][:2]
    must_abort(lambda: caption_of(FIG_SPECS[3], short4),
               "FIG-4 refuses to caption a panel with fewer than three columns",
               "three columns")
    nosub4 = json.loads(json.dumps(man))
    nosub4["figures"]["F4"].pop("subtrahend", None)
    must_abort(lambda: caption_of(FIG_SPECS[3], nosub4),
               "FIG-4 refuses to caption a ranking with no stated subtrahend",
               "no ranking subtrahend")

    bad_cap = dict(FIG_SPECS[0])
    bad_cap["caption"] = "**Figure 1.** a | b"
    must_abort(lambda: figure_block(bad_cap, man),
               "a caption that breaks markdown syntax aborts", "contains")

    # table re-spacing: the mechanism, and its effect on the real source
    glued = ["caption text", "| a | b |", "|---|---|", "| 1 | 2 |", "prose after"]
    spaced = space_out_tables(glued)
    ok("a table glued to its caption gets a blank line on both sides",
       spaced == ["caption text", "", "| a | b |", "|---|---|", "| 1 | 2 |",
                  "", "prose after"], str(spaced))
    ok("re-spacing preserves every non-blank line, in order",
       [x for x in spaced if x.strip()] == glued)
    real = space_out_tables(raw.splitlines())
    starts = [i for i, l in enumerate(real)
              if l.lstrip().startswith("|")
              and (i == 0 or not real[i - 1].lstrip().startswith("|"))]
    ok("every table start in the real source is preceded by a blank line",
       all(real[i - 1].strip() == "" for i in starts if i > 0),
       f"{len(starts)} table start(s), "
       f"{sum(1 for i in starts if i > 0 and real[i - 1].strip())} still glued")
    ok("re-spacing neither invents nor drops a table",
       count_tables("\n".join(real)) == count_tables(raw),
       f"{count_tables(raw)} table(s) before and after")

    ok("standalone rules are dropped and a table separator is not",
       drop_rules(["a", "---", "b", "|---|", "---"])[0]
       == ["a", "b", "|---|"] and drop_rules(["a", "---"])[1] == 1)
    ok("dropping rules changes no heading and no table",
       count_tables("\n".join(drop_rules(md.splitlines())[0]))
       == count_tables(md)
       and source_expectations("\n".join(drop_rules(md.splitlines())[0]))
       ["headings"] == source_expectations(md)["headings"])
    ok("every figure carries a plain-text probe for the caption guard",
       all(s.get("probe") for s in FIG_SPECS),
       ", ".join(s.get("probe", "?")[:24] for s in FIG_SPECS))

    # the caption guard needs a real file to read back, so build a throwaway
    with tempfile.TemporaryDirectory(prefix="f127_st_") as td:
        from docx import Document as _Doc
        empty = Path(td) / "no_captions.docx"
        dd = _Doc()
        dd.add_paragraph("a product with no captions at all")
        dd.save(str(empty))
        must_abort(lambda: verify_captions(empty, man),
                   "a product whose captions were lost is rejected",
                   "not present as")
        if DOCX.is_file():
            try:
                verify_captions(DOCX, man)
                ok("the landed product carries all five captions as text",
                   True)
            except SystemExit as exc:
                ok("the landed product carries all five captions as text",
                   False, str(exc)[:110])

    exp = source_expectations(md)
    ok("the built product declares one image per drawn figure",
       exp["images"] == len(FIG_SPECS), f"{exp['images']} image(s)")
    _t, body_lines = split_front_matter(raw.splitlines())
    body_txt = "\n".join(strip_draft_notes(body_lines)[0])
    ok("the built product carries every body table, and no extra",
       exp["tables"] == count_tables(body_txt),
       f"{exp['tables']} built vs {count_tables(body_txt)} in the body")

    print("-" * 66)
    if fails:
        print(f"127 --self-test: FAIL ({len(fails)}/{checks})")
        for f in fails:
            print("   " + f)
        return 1
    print(f"127 --self-test: PASS ({checks} controls)")
    return 0


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the Word manuscript from docs/MANUSCRIPT.md")
    ap.add_argument("--check", action="store_true",
                    help="verify the landed .docx against its build record")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise the transform logic without pandoc")
    ap.add_argument("--pandoc", default=None, help="path to the pandoc binary")
    ap.add_argument("--out", default=None, help="output .docx path")
    ap.add_argument("--elsevier", action="store_true",
                    help="also emit the submission copy: working citation keys\n                         replaced by running numbers in order of first appearance,\n                         bibliography reordered to match, Times New Roman and\n                         continuous line numbers")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    pandoc = find_pandoc(args.pandoc)
    if args.check:
        return check(pandoc, args.elsevier)

    out = Path(args.out) if args.out else (DOCX_ELSEVIER if args.elsevier else DOCX)
    side = build(out, pandoc, elsevier=args.elsevier)
    print("XD-RobustPVOS :: Word manuscript")
    print("=" * 66)
    print(f"  source     : docs/MANUSCRIPT.md  (sha1 {side['source_sha1'][:12]})")
    print(f"  title      : {side['title'][:70]}...")
    print(f"  pandoc     : {side['pandoc']}")
    print(f"  figures    : {', '.join(side['figures_embedded'])} embedded")
    print(f"  structure  : {side['structure']['paragraphs']} paragraphs, "
          f"{side['structure']['headings']} headings, "
          f"{side['structure']['tables']} tables, "
          f"{side['structure']['images']} images")
    print(f"  removed    : front matter + Sec. 5 drafting note")
    for n in side["notes"]:
        print(f"  note       : {n}")
    print(f"  wrote      : {out}")
    print(f"  wrote      : {SIDE_ELSEVIER if args.elsevier else SIDE}")
    if args.elsevier and side.get("citations"):
        cmap = side["citations"]["key_to_number"]
        print(f"  citations  : {len(cmap)} key(s) renumbered by first appearance")
        print("               " + ", ".join(f"{k}->{v}" for k, v in
                                             list(cmap.items())[:8]) + " ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
