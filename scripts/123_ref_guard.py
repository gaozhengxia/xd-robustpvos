"""123_ref_guard.py -- bibliography integrity for ``docs/MANUSCRIPT.md``.

Why this exists
---------------
``122_ms_doc_guard.py`` asserts that every ``[key]`` used in the body is *declared* in the key table
of ``docs/NOVELTY_BOUNDARY.md`` §3.0.  That is a **one-way** check: it catches an invented key, but
it cannot catch the opposite failure -- a declared obligation that never became a bibliography
entry, or an entry that lost its publication data.  A manuscript can satisfy "every key is
declared" while carrying no usable reference at all, which is exactly the shape of defect this
project keeps finding: *"the check passed" and "the check had nothing to check" look identical*.

123 owns the bibliography and asserts three properties.

  E. KEY-SET EQUALITY, BOTH DIRECTIONS.  The keys in the ``## References`` section of the
     manuscript and the keys in the §3.0 table must be the same set.  A key on one side only is a
     real defect in either direction: a **missing entry** is an unpayable citation debt (the paper
     cites something the reader cannot find), and an **orphan entry** is a reference that no
     sentence supports.  A one-way version of this check passes in both cases.

  F. ENTRY COMPLETENESS.  Every entry must carry a publication marker -- a venue name, ``arXiv:``,
     ``doi:``, ``pp.`` or ``in press`` -- and a four-digit year.  A bare title is not a reference,
     and an entry without a year cannot be looked up.

  G. NO UNVERIFIED MARKER.  No entry may contain TODO / TBD / XXX / FIXME / ??? / 待补.  The
     bibliography is the one place where a placeholder is indistinguishable from a citation to a
     reader who does not already know the field, so a placeholder must never ship.

Both non-empty-direction assertions are explicit (rule 10): a guard that compares two empty sets
reports PASS.

Companion discipline: ``122`` calls ``body_only()`` to keep the bibliography *outside* its numeric
traceability rule -- publication years and page numbers are publisher records, not measurements.
123 is the guard that owns that region instead.

Usage:  python scripts/123_ref_guard.py --check | --emit | --self-test
Exit code 0 on success, 1 on failure.
"""
import argparse
import contextlib
import io
import os
import re
import sys

# A verified bibliography legitimately contains non-ASCII names -- [SAM2] carries "Rädle" -- and the
# Windows console defaults to a GBK code page.  Printing one of those entries must not kill the
# guard: a guard that dies on legal data is indistinguishable from a guard that found nothing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # pragma: no cover - old interpreters
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "docs", "MANUSCRIPT.md")
NB = os.path.join(ROOT, "docs", "NOVELTY_BOUNDARY.md")

#: `NB` (the key table's home) is an internal working document and is
#: deliberately NOT distributed; `_release_gate` turns its absence in a clone
#: into a reported SKIP instead of a FileNotFoundError.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _release_gate import gate as release_gate                       # noqa: E402

#: Where the bibliography starts.  Must agree with 122's REF_ANCHOR.
REF_ANCHOR = "\n## References"

#: An entry line: ``[N1] S. Lee, ...``.  The key must start with a letter, so the journal's numeric
#: markers after typesetting (``[1]``) can never be mistaken for citation keys.
_ENTRY = re.compile(r"^\[([A-Za-z][A-Za-z0-9_\-]*)\]\s+\S", re.M)

#: A declaration row of the §3.0 table: ``| `[N1]` | ... |``.  Deliberately the same shape as 122's
#: _KEY_DECL: if the two ever drift apart, property E fails loudly rather than silently.
_DECL = re.compile(r"^\|\s*`?\[([A-Za-z][A-Za-z0-9_\-]*)\]`?\s*\|", re.M)

VENUE = re.compile(
    r"(arXiv:\s*\d|doi:|pp\.\s|\bin press\b|ICCV|CVPR|ECCV|ICLR|NeurIPS|MICCAI|"
    r"PLOS|Pattern Analysis|Supercomputing|Journal|Transactions|Conference|Workshop|"
    r"Advances in Neural)")
YEAR = re.compile(r"\b(19|20)\d{2}\b")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|XXX|FIXME|待补|待定)\b|\?\?\?")

#: A declared obligation may legitimately be a *method family* rather than a single work: §2.4
#: cites [N9] for "the standard TTA intervention", which is a practice, not a paper.  Such an entry
#: is excused from F only when it says so explicitly **and** points at a concrete key; otherwise the
#: exemption would become a general escape hatch and the guard would stop guarding anything.
FAMILY = re.compile(r"\bmethod family\b")


def load():
    d = {}
    for p in (MS, NB):
        with io.open(p, encoding="utf-8") as f:
            d[p] = f.read()
    return d


def ref_section(md):
    i = md.find(REF_ANCHOR)
    return None if i < 0 else md[i:]


def entries(md):
    """({key: entry_text}, [duplicated keys]) parsed from the bibliography section."""
    sec = ref_section(md)
    if sec is None:
        return {}, []
    out, order, cur = {}, [], None
    for ln in sec.split("\n"):
        m = _ENTRY.match(ln)
        if m:
            cur = m.group(1)
            order.append(cur)
            out[cur] = ln.strip()
        elif cur and ln.strip() and not ln.startswith("#"):
            out[cur] += " " + ln.strip()
        elif not ln.strip():
            cur = None
    dupes = sorted({k for k in order if order.count(k) > 1})
    return out, dupes


def declared(nb):
    return sorted(set(_DECL.findall(nb)))


def cited_keys(txt, own):
    """Keys other than ``own`` mentioned inside an entry body."""
    body = txt.split("]", 1)[1] if txt.startswith("[") else txt
    return sorted({k for k in re.findall(r"\[([A-Za-z][A-Za-z0-9_\-]*)\]", body) if k != own})


def check(md, nb, verbose=False):
    fails = []
    ent, dupes = entries(md)
    dec = declared(nb)

    # ---- E. key-set equality, both directions
    if not ent:
        fails.append("E. no bibliography entry parsed -- nothing to compare against the key table")
    if not dec:
        fails.append("E. no key declared in the §3.0 table -- nothing to compare against")
    for k in dupes:
        fails.append("E. key [%s] is defined by more than one entry" % k)
    missing = sorted(set(dec) - set(ent))
    orphans = sorted(set(ent) - set(dec))
    if missing:
        fails.append("E. %d declared obligation(s) with no bibliography entry: %s"
                     % (len(missing), ", ".join("[%s]" % k for k in missing)))
    if orphans:
        fails.append("E. %d bibliography entr(ies) with no declared obligation: %s"
                     % (len(orphans), ", ".join("[%s]" % k for k in orphans)))

    # ---- F + G, per entry
    excused = []
    for k in sorted(ent):
        txt = ent[k]
        hit = PLACEHOLDER.search(txt)
        if hit:
            fails.append("G. [%s] contains an unverified marker %r" % (k, hit.group(0)))
        if FAMILY.search(txt):
            # excused from F, but only if the entry actually points somewhere concrete
            cites = cited_keys(txt, k)
            if not cites:
                fails.append("F. [%s] declares itself a method family but cites no concrete key -- "
                             "an excused entry must still lead the reader somewhere" % k)
            else:
                excused.append(k)
            continue
        if not VENUE.search(txt):
            fails.append("F. [%s] carries no publication marker (venue / arXiv / doi / pp. / in press)"
                         % k)
        if not YEAR.search(txt):
            fails.append("F. [%s] carries no year" % k)

    if verbose:
        for k in sorted(ent):
            print("  [%-18s] %s" % (k, ent[k][:96] + ("..." if len(ent[k]) > 96 else "")))
        if excused:
            print("  (excused from F as method families: %s)"
                  % ", ".join("[%s]" % k for k in excused))
    stats = {"entries": len(ent), "declared": len(dec), "missing": len(missing),
             "orphans": len(orphans), "dupes": len(dupes), "family": len(excused)}
    return fails, stats


def run(verbose=True):
    d = load()
    md, nb = d[MS], d[NB]
    fails, stats = check(md, nb, verbose=verbose)
    if verbose:
        print("[123] E. key set  : %d entries vs %d declared (missing %d, orphan %d, dup %d)"
              % (stats["entries"], stats["declared"], stats["missing"], stats["orphans"],
                 stats["dupes"]))
        print("[123] F. complete : checked in the loop above")
        print("[123] G. no marks : checked in the loop above")
        for f in fails:
            print("  FAIL  " + f)
        if not fails:
            print("[123] --check: bibliography is complete and consistent in both directions")
    return fails, stats


# --------------------------------------------------------------------------- self-test
def self_test():
    checks = []
    d = load()
    md, nb = d[MS], d[NB]

    base, st = run(verbose=False)
    checks.append(("unchanged documents pass", not base))
    checks.append(("E has something to compare (n>0 on both sides)",
                   st["entries"] > 0 and st["declared"] > 0))
    checks.append(("the bibliography is not empty", st["entries"] >= 18))

    # The verbose path prints every entry, including non-ASCII author names ([SAM2] carries
    # "Rädle").  It must not raise -- this exact crash happened once on a GBK console.
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run(verbose=True)
        checks.append(("verbose print of non-ASCII entries does not raise", True))
    except Exception as exc:                               # noqa: BLE001
        checks.append(("verbose print of non-ASCII entries does not raise (%r)" % (exc,), False))

    # E fires: an orphan entry -- a key no obligation declares
    md_orphan = md + "\n\n[NZ] J. Doe, An invented paper, Journal of Nothing 1 (1) (2026) 1-2.\n"
    f, _ = check(md_orphan, nb)
    checks.append(("E fires: orphan entry [NZ]", any("NZ" in x and x.startswith("E.") for x in f)))

    # E fires: a declared obligation whose entry has been deleted
    ent, _ = entries(md)
    assert ent, "no entries parsed -- the self-test cannot build its negative control"
    victim = sorted(ent)[0]
    md_drop = re.sub(r"^\[%s\]\s+.*?$" % re.escape(victim), "", md, count=1, flags=re.M)
    assert md_drop != md, "the deletion control was a no-op"
    f, _ = check(md_drop, nb)
    checks.append(("E fires: declared obligation [%s] with its entry removed" % victim,
                   any(x.startswith("E.") and ("[%s]" % victim) in x for x in f)))

    # E fires: a key declared in the table but never used and never given an entry is the *same*
    # defect class; here we instead inject the reverse -- an entry for a key that IS declared but
    # duplicated, which must not be silently accepted.
    ent2, dupes0 = entries(md)
    first = sorted(ent2)[0]
    md_dup = md + "\n\n[%s] A. Duplicate, A second entry for the same key, CVPR 2026.\n" % first
    f, _ = check(md_dup, nb)
    checks.append(("E fires: key [%s] defined twice" % first, any("defined by more than one" in x
                                                                 for x in f)))
    checks.append(("no duplicate exists in the real file", not dupes0))

    # F fires: an entry with a title but no venue and no year
    head = md[:md.find(REF_ANCHOR)]
    md_bare = head + "\n## References\n\n[N1] Some title with neither venue nor date.\n"
    f, _ = check(md_bare, nb)
    checks.append(("F fires: entry with no publication marker",
                   any(x.startswith("F.") and "[N1]" in x for x in f)))
    # F'': an entry with a venue but no year must still fire
    md_noyear = head + "\n## References\n\n[N1] A. Author, A title, in: CVPR.\n"
    f, _ = check(md_noyear, nb)
    checks.append(("F fires: entry with a venue but no year",
                   any("no year" in x and "[N1]" in x for x in f)))

    # G fires: a placeholder in an entry
    md_ph = head + "\n## References\n\n[N1] TODO, A title, in: CVPR, 2026.\n"
    f, _ = check(md_ph, nb)
    checks.append(("G fires: placeholder in an entry", any(x.startswith("G.") for x in f)))

    # F''': the family exemption must not be a general escape hatch.  An entry that declares itself
    # a method family but leads nowhere must still be reported.
    md_fam_bad = head + "\n## References\n\n[N1] Some practice: a classical method family.\n"
    f, _ = check(md_fam_bad, nb)
    checks.append(("F fires: method family that cites no concrete key",
                   any("cites no concrete key" in x for x in f)))
    # F'''': and the exemption must be *exercised* by the real file -- an exemption branch no entry
    # ever enters is untested code standing in front of legal data.
    _, st2 = check(md, nb)
    checks.append(("the family exemption is exercised by real data (n>0)", st2.get("family", 0) > 0))

    # F/G must not fire on the real, verified entries -- a guard that flags legal data would mask
    # the signal it exists to catch
    checks.append(("F/G do not fire on the real bibliography",
                   not any(x.startswith(("F.", "G.")) for x in base)))

    print("[123] self-test")
    ok = True
    for name, passed in checks:
        # House verdict format, identical to 122: three-space indent, name left-padded,
        # verdict last.  The smoke test parses this shape ("   ... PASS") to count and
        # to locate named controls; printing "PASS  <name>" instead made every one of
        # its controls invisible -- the count assertion still failed, which is how this
        # was caught, but a reader of the raw log would have seen a green run.
        print("   %-56s %s" % (name, "PASS" if passed else "**FAIL**"))
        ok = ok and passed
    print("[123] self-test: %d/%d %s" % (sum(1 for _, p in checks if p), len(checks),
                                         "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Bibliography integrity guard for MANUSCRIPT.md")
    ap.add_argument("--check", action="store_true", help="verify E/F/G and exit 1 on failure")
    ap.add_argument("--emit", action="store_true", help="print every entry, then verify")
    ap.add_argument("--self-test", action="store_true", help="negative controls for E/F/G")
    args = ap.parse_args()

    status, msg = release_gate(
        "123", ROOT, [os.path.relpath(MS, ROOT).replace(os.sep, "/"),
                      os.path.relpath(NB, ROOT).replace(os.sep, "/")],
        "the bibliography integrity check")
    if status != "ok":
        print(msg)
        return 1 if status == "fail" else 0

    if args.self_test:
        return self_test()
    if args.check or args.emit:
        fails, _ = run(verbose=True)
        return 1 if fails else 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
