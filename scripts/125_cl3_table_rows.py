"""125 -- the OWNER of the CL3 image-quality table (paper Table 14 = [[TABLE-11]],
Table 15 = [[TABLE-11b]]).

Why this file exists
--------------------
Until 2026-09-24 nothing owned these two tables.  `118_p1_doc_sync.py` syncs only
TABLE-9 / 9b / 9c / 10, and `121_sec5_table_audit.py` never mentions IQA.  So the
rows sat in two hand-maintained documents, generated from a JSON that had itself
been produced from the *P0* result tree while section 5.1b declares every table in
the paper is P1.  The result was a table inconsistent with Table 9 (which is P1)
and a FIG-2 that the builder refused to draw.  Red line 15 ("every statement
number needs the script that produced it") names exactly this failure mode: an
unowned number goes quietly stale.

What it does
------------
  (default)      print the TABLE-11 / TABLE-11b rows and the narrative facts
                 (max |rho|, R^2, CI-width ratio, dark-channel range), all read
                 from the JSON -- never hand-copied.
  --check        re-derive the rows and compare them, cell by cell, against BOTH
                 documents; always returns a verdict, never raises.
  --emit         rewrite only the value cells of the matched rows of both docs.
  --self-test    plant damage in synthetic copies and assert `check` catches it.

The arm order of TABLE-11b is read from the table HEADER, never assumed: this
project has already lost a round to two tables whose column orders are reversed
(section 5.2 is `greedy, dagrs, sam2video`; section 5.6 is `dagrs, greedy,
sam2video`).  Assuming the order is what made that a day-long detour.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: the of-record CL3 artefact.  `results/iqa_config_level.json` is the P0 file,
#: kept only as the sensitivity comparator -- section 5.1b makes P1 primary.
JSON_DEFAULT = "results/p1/iqa_config_level_p1.json"
FRAMEWORK = "docs/PAPER_FRAMEWORK.md"
MANUSCRIPT = "docs/MANUSCRIPT.md"

#: `FRAMEWORK` is an internal working document and is deliberately NOT
#: distributed; `_release_gate` turns its absence in a clone into a reported
#: SKIP instead of a crash.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _release_gate import gate as release_gate                       # noqa: E402
MAIN_ARM = "greedy"

ORDER = ["iqa_psnr", "iqa_ssim", "iqa_laplacian_var", "iqa_tenengrad",
         "iqa_dark_channel", "iqa_entropy", "iqa_contrast", "iqa_saturation"]

KIND = {"iqa_psnr": "full-ref", "iqa_ssim": "full-ref",
        "iqa_laplacian_var": "no-ref", "iqa_tenengrad": "no-ref",
        "iqa_dark_channel": "no-ref", "iqa_entropy": "no-ref",
        "iqa_contrast": "no-ref", "iqa_saturation": "no-ref"}

#: row label per metric per document -- the two docs are in different languages
LABEL = {
    "iqa_psnr": ("PSNR vs clean", "PSNR vs clean"),
    "iqa_ssim": ("SSIM vs clean", "SSIM vs clean"),
    "iqa_laplacian_var": ("拉普拉斯方差（无参考锐度）", "Laplacian variance (no-ref sharpness)"),
    "iqa_tenengrad": ("Tenengrad 梯度能量（无参考锐度）", "Tenengrad gradient energy (no-ref sharpness)"),
    "iqa_dark_channel": ("暗通道均值（无参考雾浓度）", "Dark-channel mean (no-ref haze)"),
    "iqa_entropy": ("图像熵", "Image entropy"),
    "iqa_contrast": ("对比度（无参考）", "Contrast (no-ref)"),
    "iqa_saturation": ("饱和度（无参考）", "Saturation (no-ref)"),
}
#: document index used with LABEL
FW, MS = 0, 1

#: the line that introduces each table
MARKER = {
    ("fw", "t11"): "[[TABLE-11]] **主表**",
    ("fw", "t11b"): "[[TABLE-11b]] 策略敏感性",
    ("ms", "t11"): "**Table 14.",
    ("ms", "t11b"): "**Table 15.",
}

#: field -> header needles (covering both languages).  `n` is an exact match.
T11_FIELDS = (
    ("kind", ("类型", "Type")),
    ("rho", ("ρ", "rho")),
    ("ci_lo", ("朴素", "naive")),
    ("cl_lo", ("聚类", "cluster")),
    ("n", None),
    ("r2", ("R²", "R^2")),
    ("usable", ("usable",)),
)


# --------------------------------------------------------------------------- #
# formatting -- one definition of how a number is printed
# --------------------------------------------------------------------------- #
def f_rho(x: float) -> str:
    return f"{x:+.4f}"


def f_ci(lo: float, hi: float) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def f_r2(x: float) -> str:
    return f"{x:.4f}"


def norm(cell: str) -> str:
    """Fold the dash variants before comparing.

    The documents mix ASCII hyphen and U+2212 MINUS SIGN for negative numbers
    (`-0.0831` inside a table, `−0.064` in the prose of the same file), so a
    plain string comparison would report a difference that is only typographic.
    """
    out = cell.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(out.split())


# --------------------------------------------------------------------------- #
# markdown table parsing
# --------------------------------------------------------------------------- #
def split_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep(cells: Sequence[str]) -> bool:
    return bool(cells) and all(set(c) <= set("-: ") for c in cells)


def locate_table(lines: Sequence[str], marker: str) -> int:
    idx = [i for i, ln in enumerate(lines) if marker in ln]
    if not idx:
        raise SystemExit(f"marker not found: {marker!r}")
    j = idx[0] + 1
    while j < len(lines) and not lines[j].lstrip().startswith("|"):
        j += 1
    if j >= len(lines):
        raise SystemExit(f"no table follows the marker {marker!r}")
    return j


def parse_table(text: str, marker: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """(header, {normalised row label: cells}) of the first table after `marker`."""
    lines = text.splitlines()
    j = locate_table(lines, marker)
    header = split_row(lines[j])
    rows: Dict[str, List[str]] = {}
    j += 1
    while j < len(lines) and lines[j].lstrip().startswith("|"):
        cells = split_row(lines[j])
        if not is_sep(cells):
            rows[norm(cells[0])] = cells
        j += 1
    return header, rows


def find_col(header: Sequence[str], needles: Optional[Sequence[str]], field: str) -> int:
    if needles is None:
        hits = [i for i, c in enumerate(header) if norm(c) == "n"]
    else:
        hits = [i for i, c in enumerate(header) if any(nd in c for nd in needles)]
    if len(hits) != 1:
        raise SystemExit(
            f"cannot locate the column for {field!r}: {len(hits)} candidates in "
            f"header {list(header)} -- the document layout changed, so update "
            f"T11_FIELDS rather than letting this fall through to a silent pass")
    return hits[0]


# --------------------------------------------------------------------------- #
# the data
# --------------------------------------------------------------------------- #
def load(json_path: Path) -> Dict[str, Any]:
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    res = doc.get("results")
    if not isinstance(res, dict) or not res:
        raise SystemExit(f"{json_path} has no `results` map -- regenerate it")
    for arm in res:
        for mk in ORDER:
            if mk not in res[arm].get("metrics", {}):
                raise SystemExit(f"{json_path}: arm {arm!r} lacks metric {mk!r}")
            if "across_ci_lo_cluster" not in res[arm]["metrics"][mk]:
                raise SystemExit(
                    f"{json_path}: arm {arm!r} metric {mk!r} lacks the "
                    f"sequence-clustered CI columns -- this artefact predates their "
                    f"promotion into scripts/109_iqa_config_level.py")
    return doc


def expected_rows(doc: Dict[str, Any]
                  ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Rows both documents must contain: {metric: {field: expected string}}."""
    res = doc["results"]
    t11: Dict[str, Dict[str, str]] = {}
    for mk in ORDER:
        m = res[MAIN_ARM]["metrics"][mk]
        a = m["across"]
        t11[mk] = {
            "kind": KIND[mk], "rho": f_rho(a["rho"]),
            "ci_lo": f_ci(a["ci_lo"], a["ci_hi"]),
            "cl_lo": f_ci(m["across_ci_lo_cluster"], m["across_ci_hi_cluster"]),
            "n": str(int(a["n"])), "r2": f_r2(a["r2"]),
            "usable": str(int(a["usable"])),
        }
    t11b = {mk: {arm: f_rho(res[arm]["metrics"][mk]["across"]["rho"])
                 for arm in res} for mk in ORDER}
    return t11, t11b


def facts(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The narrative numbers, derived from the same JSON (never hand-copied)."""
    res = doc["results"]
    arms = sorted(res)

    def across(arm: str, mk: str) -> Dict[str, Any]:
        return res[arm]["metrics"][mk]["across"]

    g_best = max((abs(across(MAIN_ARM, mk)["rho"]), mk) for mk in ORDER)
    a_best = max((abs(across(a, mk)["rho"]), a, mk) for a in arms for mk in ORDER)
    ratios: List[float] = []
    clus_hi: List[Tuple[float, str, str]] = []
    for a in arms:
        for mk in ORDER:
            m = res[a]["metrics"][mk]
            naive = m["across"]["ci_hi"] - m["across"]["ci_lo"]
            ratios.append((m["across_ci_hi_cluster"] - m["across_ci_lo_cluster"]) / naive)
            clus_hi.append((m["across_ci_hi_cluster"], a, mk))
    ch_hi = max(clus_hi)
    dark = [across(a, "iqa_dark_channel")["rho"] for a in arms]
    return {
        "arms": arms,
        "greedy_max_abs": g_best[0], "greedy_max_where": g_best[1],
        "greedy_r2": g_best[0] ** 2,
        "all_max_abs": a_best[0], "all_max_where": f"{a_best[1]}/{a_best[2]}",
        "all_r2": a_best[0] ** 2,
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "cluster_hi_max": ch_hi[0], "cluster_hi_where": f"{ch_hi[1]}/{ch_hi[2]}",
        "cluster_reach": len(doc.get("verdict", {})
                             .get("across_cluster_ci_reaches_threshold") or []),
        "dark_min": min(dark), "dark_max": max(dark),
        "n_usable": sum(int(across(a, mk)["usable"]) for a in arms for mk in ORDER),
        "n_cells_total": len(arms) * len(ORDER),
        "threshold": float(doc.get("usable_threshold", 0.6)),
    }


# --------------------------------------------------------------------------- #
# checking
# --------------------------------------------------------------------------- #
def compare_table(text: str, marker: str, want: Dict[str, Dict[str, str]],
                  who: str, tname: str, doc_idx: int,
                  arms_from_header: bool) -> List[str]:
    problems: List[str] = []
    header, rows = parse_table(text, marker)
    if arms_from_header:
        arms = [norm(c).strip("`") for c in header[1:]]
        known = set(next(iter(want.values())))
        if set(arms) != known:
            problems.append(
                f"{who} {tname}: the header names the arms as {arms} while the "
                f"JSON has {sorted(known)} -- the columns would be read in the "
                f"wrong order, which is how a previous audit went wrong")
            return problems
        col_of = {norm(c).strip("`"): i for i, c in enumerate(header)}
    else:
        col_of = {fld: find_col(header, nd, fld) for fld, nd in T11_FIELDS}

    for mk, fields in want.items():
        label = norm(LABEL[mk][doc_idx])
        got = rows.get(label)
        if got is None:
            problems.append(f"{who} {tname}: the row for {mk} ({label!r}) is missing")
            continue
        for field, exp in fields.items():
            ci = col_of[field]
            if ci >= len(got):
                problems.append(f"{who} {tname}: row {label!r} has only {len(got)} "
                                f"cells, cannot read {field!r}")
                continue
            if norm(got[ci]) != exp:
                problems.append(
                    f"{who} {tname}: {label!r} / {field} reads {norm(got[ci])!r} "
                    f"but the JSON prints {exp!r}")
    return problems


def fact_specs(f: Dict[str, Any]) -> List[Tuple[int, str, str, List[str], bool]]:
    """The narrative numbers: (doc, what, pattern, expected values, ordered).

    ONE definition, shared by `check_facts` (verify) and `rewrite_facts` (repair),
    so a number can never be verified under a different rule than the one that
    writes it.  `ordered=False` marks facts the prose may list in either direction
    (a range such as "0.56-1.32x"); the check then compares them as a set.

    Note the separators are `[^\\d]{0,3}` rather than `.{0,3}`: the latter is
    greedy enough to eat the first digit of the second number, which silently
    turned "0.56-1.30x" into the pair ("0.56", "30").  The self-test caught it.
    """
    pct_g = f"{f['greedy_r2'] * 100:.1f}"
    pct_a = f"{f['all_r2'] * 100:.1f}"
    rmin, rmax = f"{f['ratio_min']:.2f}", f"{f['ratio_max']:.2f}"
    return [
        (FW, "greedy-only max |rho|", r"本表最大 \|ρ\| = ([+-][\d.]+)",
         [f_rho(f["greedy_max_abs"])], True),
        (FW, "greedy R^2", r"方差的 \*\*([\d.]+)\*\*（\*\*([\d.]+)%\*\*）",
         [f_r2(f["greedy_r2"]), pct_g], True),
        (FW, "all-arms max |rho|", r"全臂最大 \|ρ\| = ([+-][\d.]+)",
         [f_rho(f["all_max_abs"])], True),
        (FW, "all-arms R^2", r"R² = ([\d.]+)，([\d.]+)%",
         [f"{f['all_r2']:.3f}", pct_a], True),
        (FW, "CI width ratio", r"宽度为朴素版的 \*\*([\d.]+)[^\d]{0,3}([\d.]+)×\*\*",
         [rmin, rmax], False),
        (FW, "max clustered upper bound", r"最大上限 \*\*([+-][\d.]+)\*\*",
         [f_rho(f["cluster_hi_max"])], True),
        (FW, "dark-channel range", r"同为负\*\*（([^\s~]+) ~ ([^\s）]+)）",
         [f_rho(f["dark_min"]), f_rho(f["dark_max"])], False),
        # The framework restates two of these numbers in its contribution summary,
        # in a slightly different form (`\|rho\|` is escaping a table pipe, and the
        # sign is omitted).  They are separate sentences, so they need their own
        # entries -- otherwise they are exactly the unowned duplicates that go
        # quietly stale, which is what had already happened to both of them.
        (FW, "all-arms max |rho| (summary block, escaped)",
         r"全臂最大 \\\|ρ\\\| = ([\d.]+)", [f"{f['all_max_abs']:.4f}"], True),
        (FW, "max clustered upper bound (summary block)",
         r"按序列聚类的 CI 上界最大 \*\*([+-][\d.]+)\*\*",
         [f_rho(f["cluster_hi_max"])], True),
        (FW, "all-arms max |rho| (summary block, restated)",
         r"CL3 的全臂最大 \\\|ρ\\\| 是 \*\*([\d.]+)\*\*",
         [f"{f['all_max_abs']:.4f}"], True),
        (FW, "all-arms max |rho| (TABLE-11b note)",
         r"的 SSIM 最高（([+-][\d.]+)）", [f_rho(f["all_max_abs"])], True),
        (MS, "greedy-only max |rho|", r"this arm is \$\\rho = ([+-][\d.]+)\$",
         [f_rho(f["greedy_max_abs"])], True),
        (MS, "greedy R^2 percent", r"which explains \$([\d.]+)\\%\$ of the variance",
         [pct_g], True),
        (MS, "all-arms max |rho|", r"the ceiling is \$\\rho = ([+-][\d.]+)\$",
         [f_rho(f["all_max_abs"])], True),
        (MS, "max clustered upper bound",
         r"largest clustered upper bound over all \d+ combinations is \$([+-][\d.]+)\$",
         [f_rho(f["cluster_hi_max"])], True),
        (MS, "CI width ratio",
         r"each interval within \$([\d.]+)\$.{0,3}\$([\d.]+)\\times\$",
         [rmin, rmax], False),
        (MS, "dark-channel range",
         r"\$([^\s$]+)\$ to \$([^\s$]+)\$\): a haze-density",
         [f_rho(f["dark_min"]), f_rho(f["dark_max"])], False),
        (MS, "the R^2 ceiling on this arm",
         r"\$R\^2 \\le ([\d.]+)\$", [f"{f['greedy_r2']:.3f}"], True),
        # The abstract and the contribution summary restate the same four numbers.
        # They were stale in exactly the same way (§5.4 had been fixed while the
        # abstract still advertised the P0 values), which is why the check has to
        # reach beyond the section that owns the table.
        (MS, "abstract: all-arms max |rho|",
         r"the most favourable cell reaching \$\\rho = ([+-][\d.]+)\$",
         [f_rho(f["all_max_abs"])], True),
        (MS, "abstract: max clustered upper bound",
         r"clustered interval upper bound \$([+-][\d.]+)\$, \$R\^2 = ([\d.]+)\$",
         [f_rho(f["cluster_hi_max"]), f"{f['all_r2']:.3f}"], True),
        (MS, "contribution summary: greedy-only max |rho|",
         r"the largest association on the primary arm is \$\\rho = ([+-][\d.]+)\$",
         [f_rho(f["greedy_max_abs"])], True),
        (MS, "contribution summary: greedy R^2 percent",
         r"\(SSIM, \$R\^2 = ([\d.]+)\\%\$\)", [pct_g], True),
        (MS, "contribution summary: all-arms max |rho|",
         r"the most permissive arm reaches only \$\\rho = ([+-][\d.]+)\$",
         [f_rho(f["all_max_abs"])], True),
        (MS, "contribution summary: max clustered upper bound",
         r"upper bound is \$([+-][\d.]+)\$", [f_rho(f["cluster_hi_max"])], True),
    ]


#: Location labels are deliberately section-free: these numbers are restated in
#: the abstract, the contribution summary and the result section, and naming a
#: section here would have sent a reader to the wrong place.
_WHERE = {FW: "framework", MS: "manuscript"}


def check_facts(fw: str, ms: str, f: Dict[str, Any]) -> List[str]:
    """The narrative numbers are not table cells, so check them by sentence shape.

    EVERY occurrence is checked, not just the first: several of these numbers are
    restated in more than one place (the framework repeats two of them in its
    contribution summary), and `re.search` would happily verify the first copy
    while a stale duplicate sat a thousand lines later.  That is the same
    duplicate-number hazard as the two reversed-column tables.
    """
    problems: List[str] = []
    texts = {FW: fw, MS: ms}
    for doc_idx, what, pattern, exp, ordered in fact_specs(f):
        where = _WHERE[doc_idx]
        hits = list(re.finditer(pattern, texts[doc_idx]))
        if not hits:
            problems.append(
                f"{where}: cannot find the {what} sentence (pattern {pattern!r}) "
                f"-- the prose was reworded, so this check no longer covers it")
            continue
        for n, m in enumerate(hits, 1):
            got = [norm(g) for g in m.groups()]
            if ordered:
                bad = got != [norm(e) for e in exp]
            else:
                bad = sorted(got) != sorted(norm(e) for e in exp)
            if bad:
                tag = f" (occurrence {n} of {len(hits)})" if len(hits) > 1 else ""
                problems.append(
                    f"{where}: the {what}{tag} reads {[norm(g) for g in m.groups()]} "
                    f"but the JSON gives {list(exp)}"
                    + ("" if ordered else " (order-insensitive)"))

    if f["n_usable"] != 0:
        problems.append(f"{f['n_usable']} arm x metric combination(s) carry "
                        f"`usable = 1`, but §5.4's claim is that none do")
    if f["cluster_reach"] != 0:
        problems.append(f"{f['cluster_reach']} clustered interval(s) reach the "
                        f"threshold, so the prose may not say the verdict excludes it")
    return problems


def keep_sign(orig: str, value: str) -> str:
    """Write the number with the minus glyph the document already uses."""
    if orig[:1] == "\u2212" and value.startswith("-"):
        return "\u2212" + value[1:]
    return value


def rewrite_facts(text: str, f: Dict[str, Any], doc_idx: int) -> Tuple[str, int]:
    """Update the narrative numbers in place, keeping the document's own order."""
    edited = 0
    for didx, _what, pattern, exp, ordered in fact_specs(f):
        if didx != doc_idx:
            continue

        def _r(m: "re.Match[str]") -> str:
            nonlocal edited
            orig = list(m.groups())
            new = list(exp)
            if not ordered and len(orig) == len(new):
                # keep the document's ordering convention and update the
                # magnitudes, rather than imposing the JSON's order on the prose
                ranked = sorted(exp, key=lambda v: float(norm(v)))
                new = [None] * len(orig)          # type: ignore[list-item]
                for rank, i in enumerate(sorted(range(len(orig)),
                                                key=lambda i: float(norm(orig[i])))):
                    new[i] = ranked[rank]
            s = m.group(0)
            for i in range(len(orig), 0, -1):
                a, b = m.span(i)
                s = (s[:a - m.start(0)]
                     + keep_sign(orig[i - 1], new[i - 1])
                     + s[b - m.start(0):])
            if s != m.group(0):
                edited += 1
            return s

        text = re.sub(pattern, _r, text)
    return text, edited


def check(json_path: Path, fw_path: Path, ms_path: Path) -> List[str]:
    """A verdict, never an exception: a broken guard and stale data must differ."""
    doc = load(json_path)
    t11, t11b = expected_rows(doc)
    f = facts(doc)
    fw = fw_path.read_text(encoding="utf-8")
    ms = ms_path.read_text(encoding="utf-8")
    problems: List[str] = []
    for text, marker, want, who, tname, didx, by_header in (
            (fw, MARKER[("fw", "t11")], t11, "framework", "TABLE-11", FW, False),
            (fw, MARKER[("fw", "t11b")], t11b, "framework", "TABLE-11b", FW, True),
            (ms, MARKER[("ms", "t11")], t11, "manuscript", "Table 14", MS, False),
            (ms, MARKER[("ms", "t11b")], t11b, "manuscript", "Table 15", MS, True)):
        try:
            problems += compare_table(text, marker, want, who, tname, didx, by_header)
        except SystemExit as exc:
            problems.append(f"[125] {who} {tname}: {exc}")
    problems += check_facts(fw, ms, f)
    return [f"[125] {p}" if not p.startswith("[125]") else p for p in problems]


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #
def rewrite_rows(text: str, marker: str, want: Dict[str, Dict[str, str]],
                 doc_idx: int, arms_from_header: bool) -> Tuple[str, int]:
    """Replace the value cells of matched rows.  Never inserts or deletes rows."""
    lines = text.splitlines()
    j = locate_table(lines, marker)
    header = [norm(c).strip("`") for c in split_row(lines[j])]
    order = header[1:] if arms_from_header else None
    cols = None if arms_from_header else {
        fld: find_col(header, nd, fld) for fld, nd in T11_FIELDS}
    j += 1
    changed = 0
    while j < len(lines) and lines[j].lstrip().startswith("|"):
        cells = split_row(lines[j])
        if is_sep(cells):
            j += 1
            continue
        mk = next((k for k in want if norm(LABEL[k][doc_idx]) == norm(cells[0])), None)
        if mk is not None:
            new = list(cells)
            if arms_from_header:
                for arm in order:
                    new[1 + order.index(arm)] = want[mk][arm]
            else:
                for fld, ci in cols.items():
                    new[ci] = want[mk][fld]
            if new != cells:
                changed += 1
            lines[j] = "| " + " | ".join(new) + " |"
        j += 1
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), changed


def emit(json_path: Path, fw_path: Path, ms_path: Path) -> int:
    doc = load(json_path)
    t11, t11b = expected_rows(doc)
    f = facts(doc)
    total = 0
    for path, didx, specs in (
            (fw_path, FW, ((t11, MARKER[("fw", "t11")], False),
                           (t11b, MARKER[("fw", "t11b")], True))),
            (ms_path, MS, ((t11, MARKER[("ms", "t11")], False),
                           (t11b, MARKER[("ms", "t11b")], True)))):
        text = path.read_text(encoding="utf-8")
        rows = 0
        for want, marker, by_header in specs:
            text, n = rewrite_rows(text, marker, want, didx, by_header)
            rows += n
        text, nums = rewrite_facts(text, f, didx)
        path.write_text(text, encoding="utf-8")
        total += rows + nums
        print(f"[125] {path.name}: {rows} table row(s) and {nums} narrative "
              f"number(s) rewritten")
    print(f"[125] emitted: {total} edit(s)")
    return 0


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def self_test(json_path: Path, fw_path: Path, ms_path: Path) -> int:
    import tempfile

    checks: List[Tuple[str, bool]] = []

    def verdict(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    doc = load(json_path)
    t11, t11b = expected_rows(doc)
    f = facts(doc)
    real_fw = fw_path.read_text(encoding="utf-8")
    real_ms = ms_path.read_text(encoding="utf-8")
    p0 = "0.15705419537702195"        # the retracted P0 rho for greedy/PSNR
    t_rho = doc["results"][MAIN_ARM]["metrics"]["iqa_psnr"]["across"]["rho"]

    with tempfile.TemporaryDirectory(prefix="125_selftest_") as td:
        td = Path(td)
        fwp, msp = td / "fw.md", td / "ms.md"

        def rebuild(mut_fw=None, mut_ms=None) -> None:
            """Render both docs FROM the JSON (a fully consistent baseline), then
            optionally perturb -- so every control runs whether or not the real
            documents happen to be consistent.  Tables *and* narrative numbers are
            repaired, because a baseline that is only half-consistent would make
            "the guard is broken" indistinguishable from "the data is broken"."""
            a, _ = rewrite_rows(real_fw, MARKER[("fw", "t11")], t11, FW, False)
            b, _ = rewrite_rows(a, MARKER[("fw", "t11b")], t11b, FW, True)
            b, _ = rewrite_facts(b, f, FW)
            c, _ = rewrite_rows(real_ms, MARKER[("ms", "t11")], t11, MS, False)
            d, _ = rewrite_rows(c, MARKER[("ms", "t11b")], t11b, MS, True)
            d, _ = rewrite_facts(d, f, MS)
            fwp.write_text(mut_fw(b) if mut_fw else b, encoding="utf-8")
            msp.write_text(mut_ms(d) if mut_ms else d, encoding="utf-8")

        def run() -> List[str]:
            return check(json_path, fwp, msp)

        def plant(text: str, old: str, new: str) -> str:
            if old not in text:
                raise AssertionError(f"self-test cannot find {old!r} to damage")
            return text.replace(old, new, 1)

        # --- positive ------------------------------------------------------- #
        rebuild()
        base = run()
        verdict("a document whose tables match the JSON passes (baseline)", base == [])
        if base:
            print("   baseline problems:", *base[:4], sep="\n     ")

        # --- plant damage --------------------------------------------------- #
        rebuild(mut_fw=lambda t: plant(t, f_rho(t_rho), f_rho(float(p0))))
        verdict("fires: the framework rho reverted to the P0 value",
                any("TABLE-11" in p and "rho" in p for p in run()))

        rebuild(mut_ms=lambda t: plant(t, f_rho(t_rho), f_rho(float(p0))))
        verdict("fires: the manuscript Table 14 rho reverted to the P0 value",
                any("Table 14" in p and "rho" in p for p in run()))

        rebuild(mut_fw=lambda t: plant(t, f_rho(t_rho), f_rho(t_rho + 1e-4)))
        verdict("fires: one unit in the 4th decimal (the smallest change a cell can carry)",
                any("rho" in p for p in run()))

        # Table 15's controls must target the ROW: the narrative sentence that
        # carries the same number appears EARLIER in the document, so a bare value
        # swap damages the prose instead of the table and the control would be
        # asserting the wrong thing.
        def t11b_row(mk: str, order: Sequence[str]) -> str:
            return ("| " + LABEL[mk][MS] + " | "
                    + " | ".join(t11b[mk][a] for a in order) + " |")

        NATIVE = ["dagrs", "greedy", "sam2video"]
        ALT = ["greedy", "dagrs", "sam2video"]
        s_rho = doc["results"]["sam2video"]["metrics"]["iqa_ssim"]["across"]["rho"]

        rebuild(mut_ms=lambda t: plant(
            t, t11b_row("iqa_ssim", NATIVE),
            t11b_row("iqa_ssim", ALT)))   # header-wrong / values-right
        verdict("fires: an arm reorder whose values did not move with the header",
                any("Table 15" in p for p in run()))

        rebuild(mut_ms=lambda t: t
                .replace("| Quality metric | `dagrs` | `greedy` | `sam2video` |",
                         "| Quality metric | `greedy` | `dagrs` | `sam2video` |")
                .replace("\n".join(t11b_row(mk, NATIVE) for mk in ORDER),
                         "\n".join(t11b_row(mk, ALT) for mk in ORDER)))
        verdict("quiet: a consistent arm reorder (header and values together)",
                not any("Table 15" in p for p in run()))

        rebuild(mut_ms=lambda t: plant(
            t, t11b_row("iqa_ssim", NATIVE),
            t11b_row("iqa_ssim", NATIVE).replace(
                t11b["iqa_ssim"]["sam2video"], f_rho(s_rho + 1e-4))))
        verdict("fires: a Table 15 arm column drifted by one unit",
                any("Table 15" in p for p in run()))

        def drop_row(t: str) -> str:
            hit = [ln for ln in t.splitlines() if ln.startswith("| Contrast (no-ref) ")]
            assert hit, "self-test cannot find the Table 14 contrast row"
            return t.replace(hit[0] + "\n", "", 1)

        rebuild(mut_ms=drop_row)
        verdict("fires: a Table 14 row was deleted",
                any("is missing" in p for p in run()))

        rebuild(mut_ms=lambda t: plant(t, f"{f['greedy_r2'] * 100:.1f}\\%", "7.3\\%"))
        verdict("fires: the narrative R^2 percent went stale",
                any("R^2 percent" in p for p in run()))

        rebuild(mut_ms=lambda t: plant(t, "this arm is", "this arm was"))
        verdict("fires: a reworded narrative sentence is reported, not skipped",
                any("cannot find" in p for p in run()))

        rebuild(mut_fw=lambda t: plant(
            t, f"本表最大 |ρ| = {f_rho(f['greedy_max_abs'])}",
            f"本表最大 |ρ| = {f_rho(f['greedy_max_abs'] + 1e-4)}"))
        verdict("fires: the framework's narrative max |rho| went stale",
                any("greedy-only" in p for p in run()))

        def plant_last(text: str, old: str, new: str) -> str:
            i = text.rfind(old)
            assert i >= 0, f"self-test cannot find a late copy of {old!r}"
            return text[:i] + new + text[i + len(old):]

        # the ratio is stated twice in the framework; a check that only looked at
        # the first copy would pass while the second stayed stale
        rebuild(mut_fw=lambda t: plant_last(t, "宽度为朴素版的 **0.56",
                                           "宽度为朴素版的 **0.99"))
        verdict("fires: a stale DUPLICATE occurrence, not just the first",
                any("occurrence 2 of 2" in p for p in run()))

        # a broken guard must be visible as a PROBLEM, not as a traceback
        fwp.write_text(real_fw.replace(MARKER[("fw", "t11")], "gone", 1), encoding="utf-8")
        msp.write_text(real_ms, encoding="utf-8")
        try:
            got = run()
            verdict("a missing marker is reported as a problem, not raised",
                    any("marker not found" in p for p in got))
        except SystemExit:
            verdict("a missing marker is reported as a problem, not raised", False)

    ok = sum(1 for _, v in checks if v)
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"\n[125] self-test: {ok}/{len(checks)} controls pass")
    return 0 if ok == len(checks) else 1


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=JSON_DEFAULT)
    ap.add_argument("--framework", default=FRAMEWORK)
    ap.add_argument("--manuscript", default=MANUSCRIPT)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    jp, fw, ms = ROOT / a.json, ROOT / a.framework, ROOT / a.manuscript

    status, msg = release_gate("125", ROOT, [a.manuscript, a.framework],
                               "the CL3 / Table 11-11b plan-row check")
    if status != "ok":
        print(msg)
        return 1 if status == "fail" else 0

    if a.self_test:
        return self_test(jp, fw, ms)
    if a.emit:
        return emit(jp, fw, ms)
    if a.check:
        problems = check(jp, fw, ms)
        for p in problems:
            print(f"FAIL  {p}")
        print(f"\n[125] VERDICT: {'PASS' if not problems else 'FAIL'} "
              f"({len(problems)} problem(s))")
        return 0 if not problems else 1

    doc = load(jp)
    t11, t11b = expected_rows(doc)
    f = facts(doc)
    print(f"# source: {a.json} -- {len(f['arms'])} arms x {len(ORDER)} metrics = "
          f"{f['n_cells_total']} arm x metric combinations, usable threshold "
          f"{f['threshold']}")
    print(f"\n### TABLE-11 rows (arm={MAIN_ARM})")
    print("| 质量指标 | 类型 | ρ（与 J&F） | 95% CI（朴素格级） | "
          "95% CI（按序列聚类） | n | R² | usable |")
    print("|---|---|---|---|---|---|---|---|")
    for mk in ORDER:
        r = t11[mk]
        print(f"| {LABEL[mk][FW]} | {r['kind']} | {r['rho']} | {r['ci_lo']} | "
              f"{r['cl_lo']} | {r['n']} | {r['r2']} | {r['usable']} |")
    print("\n### TABLE-11b rows (view=across)")
    print("| 质量指标 | " + " | ".join(f"`{x}`" for x in f["arms"]) + " |")
    print("|---" * (len(f["arms"]) + 1) + "|")
    for mk in ORDER:
        print(f"| {LABEL[mk][FW]} | " + " | ".join(t11b[mk][x] for x in f["arms"]) + " |")
    print("\n### narrative facts")
    print(f"  greedy-only max |rho| = {f_rho(f['greedy_max_abs'])} ({f['greedy_max_where']})"
          f"   R2 = {f_r2(f['greedy_r2'])} ({f['greedy_r2'] * 100:.1f}%)")
    print(f"  all-arms    max |rho| = {f_rho(f['all_max_abs'])} ({f['all_max_where']})"
          f"   R2 = {f['all_r2']:.3f} ({f['all_r2'] * 100:.1f}%)")
    print(f"  arm x metric with usable=1 : {f['n_usable']} of {f['n_cells_total']}")
    print(f"  CI width ratio (cluster/naive): {f['ratio_min']:.2f} .. {f['ratio_max']:.2f}")
    print(f"  max clustered CI upper bound = {f_rho(f['cluster_hi_max'])} "
          f"({f['cluster_hi_where']}); reaching the threshold: {f['cluster_reach']}")
    print(f"  dark channel across arms: {f_rho(f['dark_min'])} .. {f_rho(f['dark_max'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
