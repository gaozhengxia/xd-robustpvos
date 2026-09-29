# -*- coding: utf-8 -*-
"""Owned guard for the manuscript <-> framework pair (checks 118-121 are table-level; this one is
document-level).

Three properties are asserted, each with its own negative control, because every one of them has a
failure mode of "pass":

  A. NUMBER TRACEABILITY.  Every numeric token in ``docs/MANUSCRIPT.md`` must occur verbatim in
     ``docs/PAPER_FRAMEWORK.md`` (or be listed as pure arithmetic on framework values).  This is
     red line 6/13/15 in one place: a number that reaches the paper without a framework anchor has
     no owning script.
     *Why it is not enough on its own*: "the token exists in the framework" cannot tell a value
     from a different table that happens to collide, and it cannot tell whether the value is
     stale.  Keys/units are checked by 118-121; currency is checked by each number's owning script.

  B. WORDING DISCIPLINE.  The paper claims a diagnosis, not a gain: "our method", "we propose",
     "outperforms", "improves N points", "superior to", "state-of-the-art" are banned as *claims*.
     The file quotes the ban list inside its own positioning note, so a hit is excused only when it
     sits inside a blockquote that also carries a meta marker -- judged per *block*, never per
     line, because markdown wraps the note and the banned phrase lands on a continuation line.

  C. CITATION-KEY INTEGRITY.  Every ``[key]`` in the manuscript must be declared in the key table
     of ``docs/NOVELTY_BOUNDARY.md`` §3.0.  An undeclared key is a reference to a work that does
     not exist in the obligation list.

Usage:  python scripts/122_ms_doc_guard.py --check | --self-test
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = "docs/MANUSCRIPT.md"
FW = "docs/PAPER_FRAMEWORK.md"
NB = "docs/NOVELTY_BOUNDARY.md"

#: `FW` and `NB` are internal working documents and are deliberately NOT
#: distributed; `_release_gate` turns their absence in a clone into a reported
#: SKIP.  (`load()` below already tolerates an absent file, but the properties
#: that read it would then FAIL, which is the outcome this gate replaces.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _release_gate import gate as release_gate                       # noqa: E402

#: values the manuscript may introduce as pure arithmetic on framework values
#: (e.g. 1.97 = 135.4/68.6 measured encoder calls; 1.6 = 1+3/5; 3.6 = 1+13/5)
#: The bibliography starts here.  Defined before SECTIONS because §7 ends at it.
REF_ANCHOR = "\n## References"
#: The declarations block (data/code availability, CRediT, competing interest,
#: funding) became a real section on 2026-09-24, and it sits BETWEEN §7 and the
#: bibliography.  §7 must therefore stop at whichever of the two comes first: a
#: document's declarations are not part of its conclusion.  This boundary was
#: added because leaving them inside §7 moved it from 158 words to 509 and out of
#: its 150-200 band -- which is the only reason the omission was noticed, and is
#: the reason the band is worth having.
DECL_ANCHOR = "\n## Declarations"

DERIVED_OK = {
    "1.97", "1.6", "3.6",
    "0.45", "0.35", "0.20", "0.5", "1.0", "3", "5", "8", "13", "2", "1", "4", "25", "150",
    "24", "120", "7", "2.0", "0.05", "0.01", "10", "180", "0.6", "4.0", "0.75", "0.95",
    "3.4", "0.65", "0.02", "0.08", "20", "340", "34", "60", "660", "27", "0.10", "0.18",
    "0.06", "0.28", "0.30", "1.95", "1.08", "0.43", "0.55", "1600", "0.0054",
}

BANNED = [
    r"\bwe propose\b",
    r"\bour method\b",
    r"\bour approach\b",
    r"\boutperforms?\b",
    r"\boutperformed the baseline\b",
    r"\bimproves? \d",
    r"\bsuperior to\b",
    r"\bstate-of-the-art\b",
    r"\bwe achieve\b",
    r"\bnovel method\b",
]

META = [
    re.compile(r"\*\*Banned:\*\*"),
    re.compile(r"kept deliberately neutral"),
    re.compile(r"never as \"our method\""),
    re.compile(r"not \"our method\""),
]


# ---------------------------------------------------------------- A. numbers
def body_only(md):
    """The manuscript up to the bibliography.

    Reference entries carry publication years, page numbers, volume numbers and arXiv identifiers.
    Those are *publisher records*, not measurements: applying rule A to them would demand that
    '2026' or '10765' be traceable to an experiment they have nothing to do with, and the only way
    to satisfy that would be to stuff the framework with bibliography noise -- which would blunt
    the rule for the numbers it exists to protect.  The bibliography is checked instead by
    ``123_ref_guard.py``, which is the guard that owns it.
    """
    i = md.find(REF_ANCHOR)
    return md if i < 0 else md[:i]


def _non_heading_tokens(text):
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    head = set()
    for line in text.split("\n"):
        if re.match(r"^\s*#{1,6}\s", line):
            head.update(re.findall(r"\d+(?:\.\d+)?", line))
    return set(nums) - head


def check_numbers(md, fw):
    body = body_only(md)
    tok = _non_heading_tokens(body)
    missing = [t for t in sorted(tok) if t not in DERIVED_OK and t not in fw]
    return missing, len(tok)


# ---------------------------------------------------------------- B. wording
def _blocks(text):
    """Maximal runs of blockquote lines, and singletons for everything else."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(">"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith(">"):
                j += 1
            yield i + 1, lines[i:j]
            i = j
        else:
            yield i + 1, [lines[i]]
            i += 1


def check_wording(md):
    violations, excused = [], []
    for start, lines in _blocks(md):
        joined = "\n".join(lines)
        is_meta = any(m.search(joined) for m in META)
        for off, line in enumerate(lines):
            for pat in BANNED:
                if re.search(pat, line, re.I):
                    (excused if is_meta else violations).append(
                        (start + off, line.strip()[:140], pat))
    return violations, excused


# ---------------------------------------------------------------- C. keys
_KEY_IN_MS = re.compile(r"(?<!\[)\[([A-Za-z][A-Za-z0-9_\-]*)\]")
_KEY_DECL = re.compile(r"^\|\s*`?\[([A-Za-z][A-Za-z0-9_\-]*)\]`?\s*\|", re.M)


def check_keys(md, nb):
    declared = set(_KEY_DECL.findall(nb))
    used = set(_KEY_IN_MS.findall(md))
    return sorted(used - declared), sorted(used), sorted(declared)


# ---------------------------------------------------------------- D. section sizes
SIZES_ANCHOR = "<!-- SECTION-SIZES -->"
SIZES_END = "<!-- /SECTION-SIZES -->"

#: section -> (start anchor, end anchor, unit).  Units: "words" or "sent".
SECTIONS = {
    "Abstract": ("## Abstract", "\n---\n", "sent"),
    "§1": ("## 1. Introduction", "## 2. Related Work", "words"),
    "§1.1": ("### 1.1 Background", "### 1.2", "words"),
    "§1.2": ("### 1.2", "### 1.3", "words"),
    "§1.3": ("### 1.3", "### 1.4", "words"),
    "§1.4": ("### 1.4", "## 2. Related Work", "words"),
    "§2": ("## 2. Related Work", "## 3. XD-RobustPVOS", "words"),
    "§2.1": ("### 2.1", "### 2.2", "words"),
    "§2.2": ("### 2.2", "### 2.3", "words"),
    "§2.3": ("### 2.3", "### 2.4", "words"),
    "§2.4": ("### 2.4", "## 3. XD-RobustPVOS", "words"),
    "§6": ("## 6. Discussion", "## 7. Conclusion", "words"),
    "§7": ("## 7. Conclusion", REF_ANCHOR, "words"),
}


def _slice(md, a, b):
    i = md.find(a)
    if i < 0:
        return None
    j = md.find(b, i) if b else len(md)
    if j <= i:
        return None
    return md[i:j]


def measure(md, name):
    """Word count of a section, with math collapsed to one token (the guard's own definition)."""
    spec = SECTIONS.get(name)
    if not spec:
        return None
    if name == "§7":
        # §7 ends at the declarations block when the document has one, and at the
        # bibliography otherwise -- whichever comes first.
        i = md.find(spec[0])
        if i < 0:
            return None
        ends = [md.find(x, i) for x in (DECL_ANCHOR, REF_ANCHOR)]
        ends = [e for e in ends if e > i]
        if not ends:
            return None
        t = md[i:min(ends)]
    else:
        t = _slice(md, spec[0], spec[1])
    if t is None:
        return None
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    if spec[2] == "sent":
        body = re.sub(r"\s+", " ", t).strip()
        return len(re.findall(r"[.!?](?:\s|$)", body + " "))
    return len(re.findall(r"[A-Za-z][A-Za-z0-9\-']*", t))


def size_rows(fw):
    """[(name, tabulated, band), ...] parsed from the anchored block."""
    i, j = fw.find(SIZES_ANCHOR), fw.find(SIZES_END)
    if i < 0 or j <= i:
        return []
    return re.findall(r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*$",
                      fw[i:j], re.M)


def check_sizes(fw, md):
    """Verify the `<!-- SECTION-SIZES -->` table: value equality, and band membership."""
    if fw.find(SIZES_ANCHOR) < 0 or fw.find(SIZES_END) <= fw.find(SIZES_ANCHOR):
        return ["D. section-size anchor block not found in the framework"], 0
    rows = size_rows(fw)
    if not rows:
        return ["D. section-size table has no parseable row"], 0
    fails = []
    for name, want_s, band in rows:
        got = measure(md, name)
        if got is None:
            fails.append("D. cannot measure %s (anchor missing in the manuscript)" % name)
            continue
        if got != int(want_s):
            fails.append("D. %s: table says %s, manuscript measures %d" % (name, want_s, got))
        if band.strip() and band.strip() != "-":
            m = re.match(r"^(\d+)-(\d+)$", band.strip())
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if not (lo <= got <= hi):
                    fails.append("D. %s: measured %d outside the planned band %d-%d"
                                 % (name, got, lo, hi))
    # a table whose middle column never has to be right is a non-observation
    if all(int(r[1]) == 0 for r in rows):
        fails.append("D. every tabulated size is 0 -- the table cannot fail")
    return fails, len(rows)


# ---------------------------------------------------------------- driver
def load():
    out = {}
    for name in (MS, FW, NB):
        p = os.path.join(ROOT, name)
        out[name] = io.open(p, encoding="utf-8").read() if os.path.exists(p) else ""
    return out


def run(verbose=True):
    d = load()
    md, fw, nb = d[MS], d[FW], d[NB]
    fails = []

    if not md:
        return ["manuscript not found at " + MS], {}

    miss, n_tok = check_numbers(md, fw)
    if md.find(REF_ANCHOR) < 0:
        fails.append("A. no '## References' section found -- bibliography exclusion is a no-op, "
                     "so rule A has no defined scope")
    if miss:
        fails.append("A. %d numeric token(s) not traceable to the framework: %s"
                     % (len(miss), ", ".join(miss)))
    viol, excused = check_wording(md)
    if viol:
        fails.append("B. %d banned wording hit(s) outside a meta note: %s"
                     % (len(viol), "; ".join("L%d %s" % (l, p) for l, _, p in viol)))
    undeclared, used, declared = check_keys(md, nb)
    if undeclared:
        fails.append("C. %d undeclared citation key(s): %s"
                     % (len(undeclared), ", ".join(undeclared)))
    if not declared:
        fails.append("C. the key table in %s was not found (nothing to check against)" % NB)

    size_fails, n_rows = check_sizes(fw, md)
    fails.extend(size_fails)
    if n_rows == 0:
        fails.append("D. the section-size table compared 0 rows (nothing to check)")

    if verbose:
        print("[122] A. numbers : %d distinct non-heading token(s), %d untraceable"
              % (n_tok, len(miss)))
        print("[122] B. wording : %d violation(s), %d excused meta-mention(s)"
              % (len(viol), len(excused)))
        print("[122] C. keys    : %d used, %d declared in %s%s"
              % (len(used), len(declared), NB,
                 (", undeclared: " + ", ".join(undeclared)) if undeclared else ""))
        print("[122] D. sizes   : %d section(s) measured against the plan" % n_rows)
        for name, tabulated, band in size_rows(fw):
            print("        %-9s %5d  (table %s, band %s)"
                  % (name, measure(md, name) or 0, tabulated, band.strip()))
        for f in fails:
            print("  FAIL  " + f)
        if not fails:
            print("[122] --check: all four document-level properties hold")
    stats = {"tokens": n_tok, "themes_used": len(used), "declared": len(declared),
             "excused": len(excused), "rows": n_rows}
    return fails, stats


def self_test():
    checks = []
    d = load()
    md, fw, nb = d[MS], d[FW], d[NB]

    base, _ = run(verbose=False)
    checks.append(("unchanged documents pass", not base))

    # A: an untraceable number must be reported
    plant_a = "0.987654321"
    md_a = md.replace("## Abstract", "## Abstract\n\nvalue %s.\n" % plant_a, 1)
    miss, _ = check_numbers(md_a, fw)
    assert plant_a in miss, "planted untraceable number was not found by the extractor"
    checks.append(("A fires: untraceable number reported", plant_a in miss))
    # A': a plant that IS in the framework must NOT be reported (no over-firing on legal data)
    legal = "0.1541"
    assert legal in fw
    md_ok = md.replace("## Abstract", "## Abstract\n\nvalue 0.1541.\n", 1)
    miss2, _ = check_numbers(md_ok, fw)
    checks.append(("A does not fire on a framework-backed number", legal not in miss2))
    # A'': the bibliography must be *outside* rule A's scope.  A number planted after the
    # bibliography anchor must not be reported -- otherwise every real entry (years, volume and
    # page numbers, arXiv ids) fires, and the only way to pass would be to defang the rule.
    md_bib = md + "\n\n[XX] A. Author, A title, Journal 82 (8) (2026) 1234-1245.\n"
    miss3, _ = check_numbers(md_bib, fw)
    checks.append(("A does not fire on a number planted in the bibliography",
                   REF_ANCHOR in md and "1234" not in miss3 and "1245" not in miss3))

    # B: a plain claim must be reported
    v_plain, _ = check_wording(md.replace(
        "## Abstract", "## Abstract\n\nOur method improves 2.3 points.\n", 1))
    checks.append(("B fires: plain claim", bool(v_plain)))
    # B': a claim inside a blockquote with NO meta marker must also be reported -- i.e. the
    # block-level rule must not excuse every blockquote just because some blockquote is a note
    v_block, _ = check_wording(md.replace(
        "## Abstract", "## Abstract\n\n> We propose a pipeline that outperforms the memory bank.\n", 1))
    checks.append(("B fires: claim inside a blockquote", bool(v_block)))
    # B'': the file's own ban list must stay excused (legal data must not be reported)
    checks.append(("B does not fire on the file's own rule note", not check_wording(md)[0]))

    # C: an invented key must be reported
    undeclared, _, _ = check_keys(md.replace("## Abstract", "## Abstract\n\n[N99]\n", 1), nb)
    checks.append(("C fires: undeclared key [N99]", "N99" in undeclared))
    _, used0, declared0 = check_keys(md, nb)
    checks.append(("C has something to check (n>0 on both sides)",
                   len(used0) > 0 and len(declared0) > 0))

    # D: a stale tabulated size must be reported
    rows = size_rows(fw)
    assert rows, "section-size table not found"
    name0, val0, _ = rows[0]
    fw_bad = fw.replace("| %s | %s |" % (name0, val0), "| %s | %d |" % (name0, int(val0) + 7), 1)
    assert fw_bad != fw, "could not perturb the size table"
    d_fw, n_bad = check_sizes(fw_bad, md)
    checks.append(("D fires: stale tabulated size", bool(d_fw) and n_bad > 0))
    # D': editing the manuscript without updating the table must be reported
    md_bad = md.replace("## 7. Conclusion", "## 7. Conclusion\n\nextra words here.", 1)
    d_md, _ = check_sizes(fw, md_bad)
    checks.append(("D fires: manuscript grew, table not updated", bool(d_md)))
    # D'': an untouched pair must not fire (no over-firing on legal data)
    d_ok, n_ok = check_sizes(fw, md)
    checks.append(("D does not fire on the shipped pair", not d_ok and n_ok == len(rows)))

    print("[122] === --self-test ===")
    ok = True
    for name, good in checks:
        print("   %-56s %s" % (name, "PASS" if good else "**FAIL**"))
        ok = ok and bool(good)
    # Print the counts rather than a bare word: a wrapper must not hardcode how many controls a child
    # runs, because that number changes whenever a control is added -- and a stale constant then
    # reports FAIL on a child that actually passed.
    n_pass = sum(1 for _, good in checks if good)
    print("[122] self-test: %d/%d %s" % (n_pass, len(checks), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    status, msg = release_gate("122", ROOT, [MS, FW, NB],
                               "the document-level (numbers/wording/keys/sizes) checks")
    if status != "ok":
        print(msg)
        return 1 if status == "fail" else 0
    if "--self-test" in args:
        return self_test()
    if "--check" in args or not args:
        fails, _ = run(verbose=True)
        print("[122] VERDICT:", "PASS (0 problem(s))" if not fails
              else "FAIL (%d problem(s))" % len(fails))
        return 0 if not fails else 1
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
