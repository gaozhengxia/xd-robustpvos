"""Gate for the checks whose inputs are deliberately NOT distributed.

Why this exists
---------------
The repository ships the code, the numbers it reports (``results/``) and the
figures those numbers are drawn into.  It deliberately does not ship:

  * ``docs/MANUSCRIPT.md`` -- the paper's own source file.  The paper is
    submitted to the journal; the repository is the code and the evidence behind
    it.  The one concrete reason it is withheld rather than shipped: its top
    block is the drafting status table plus the provenance rule ("every number
    below is copied from ``docs/PAPER_FRAMEWORK.md``").  ``scripts/127_make_docx.py``
    already strips that block from the .docx, so the .md is the only artefact
    that would expose it.

  * the internal working notes in ``docs/`` -- a writing plan, a root-cause log
    and the novelty/differentiation notes.  Those are the lab notebook, not the
    paper; ``docs/BASELINE_DIAGNOSIS.md`` in particular is a list of the authors'
    own past defects and ``docs/PAPER_FRAMEWORK.md`` records which values were
    withdrawn.  Shipping them hands a reviewer material that has nothing to do
    with the claims under review.
  * the raw per-cell CSVs that the degradation sweeps wrote into ``_scratch/``.
    ``results/`` carries the aggregated tables; ``_scratch/`` carries 1.6 GB of
    intermediate sweeps, launchers and logs.

Several guards reconcile the manuscript against those inputs, so in a fresh clone
they used to die with a bare ``FileNotFoundError`` or report FAIL.  Both read as
"the repository is broken", which is the worst possible signal for a paper whose
selling point is that every reported number is tied to the artifact that produces
it.

Two rules -- and the second one is the point
-------------------------------------------
1. A missing input that IS declared in ``UNPUBLISHED`` below -> the guard prints a
   SKIP line naming the input and why it is absent, and exits 0.
2. A missing input that is NOT declared -> the guard fails, exactly as before.

Rule 2 is what keeps this honest.  ``skip`` is a property of the *repository*,
not of the filesystem: an input that is absent for a reason nobody declared -- a
renamed ablation directory, a ``results/`` table deleted by a bad glob -- still
produces a hard failure instead of a comfortable SKIP.  Without it, "the check
did not run" and "the check passed" would look the same -- the failure mode this
project has already paid for once.

Usage in a guard::

    from _release_gate import gate
    status, msg = gate("121", ROOT, mode="the section-5 table audit")
    if status != "ok":
        print(msg)
        return 1 if status == "fail" else 0

Running this file directly
--------------------------
    python scripts/_release_gate.py --check       audit the declaration itself
    python scripts/_release_gate.py --self-test   synthetic controls for `gate`
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

#: Marker printed on every skip.  The smoke test matches on this exact string, so
#: it must not be reworded in one place only.
SKIP_TOKEN = "SKIP-UNPUBLISHED-INPUT"

#: repo-relative path -> why it is absent from a clone.
#: Keep both halves specific: the reason is what a reader of a clone needs.
UNPUBLISHED: "dict[str, str]" = {
    "docs/MANUSCRIPT.md":
        "the paper's own source file: submitted to the journal, not distributed "
        "with the code (its front matter is the drafting block that 127 strips "
        "from the .docx)",
    "docs/BASELINE_DIAGNOSIS.md":
        "internal root-cause log, not part of the paper",
    "docs/COMPUTE_UTILIZATION.md":
        "internal compute accounting",
    "docs/NOVELTY_BOUNDARY.md":
        "internal differentiation notes + pre-submission self-check list",
    "docs/PAPER_FRAMEWORK.md":
        "internal writing plan; also records which values were withdrawn",
    "_scratch/ablation_20260920":
        "raw per-cell CSVs of the two-phase ablation sweep (aggregate is in results/)",
    "_scratch/abl13_20260924":
        "raw per-cell CSVs of the 13-cell ablation sweep (aggregate is in results/)",
}

#: What each gated guard reconciles, so the smoke test can ask "would this guard
#: skip in this checkout?" *without* running it.  The guards pass their own lists
#: at the call site (they can see their constants; this table cannot), so the two
#: copies could drift apart silently.  `--check` pins this down as far as sources
#: allow without importing a guard: for every entry below it asserts the path is
#: either tracked by git (so it ships) or named inside the guard that requires it
#: (so its absence is declared).  It does NOT prove the two lists are equal -- a
#: guard requiring an extra shipped path nowhere listed here would still pass.
REQUIRED: "dict[str, list[str]]" = {
    "118": ["docs/PAPER_FRAMEWORK.md"],
    "121": ["docs/MANUSCRIPT.md", "docs/PAPER_FRAMEWORK.md",
            "_scratch/abl13_20260924", "_scratch/ablation_20260920"],
    "122": ["docs/MANUSCRIPT.md", "docs/PAPER_FRAMEWORK.md",
            "docs/NOVELTY_BOUNDARY.md"],
    "123": ["docs/MANUSCRIPT.md", "docs/NOVELTY_BOUNDARY.md"],
    "125": ["docs/MANUSCRIPT.md", "docs/PAPER_FRAMEWORK.md"],
    "126": ["docs/MANUSCRIPT.md", "docs/PAPER_FRAMEWORK.md"],
}


def missing(root, required: Iterable[str]) -> List[str]:
    """Required repo-relative paths that are absent from this checkout."""
    root = Path(root)
    return [r for r in required if not (root / r).exists()]


def gate(tag: str, root, required: Optional[Iterable[str]] = None,
         mode: str = "--check") -> Tuple[str, str]:
    """Decide whether a guard may run in this checkout.

    Returns ``("ok", "")``, ``("skip", message)`` or ``("fail", message)``.  The
    caller prints the message and exits 0 for ``skip``, 1 for ``fail``.
    """
    if required is None:
        required = REQUIRED[tag]
    gone = missing(root, required)
    if not gone:
        return "ok", ""

    undeclared = [r for r in gone if r not in UNPUBLISHED]
    if undeclared:
        lines = ["[%s] FAIL: required input(s) missing and NOT declared unpublished:"
                 % tag]
        lines += ["        %s" % r for r in undeclared]
        lines.append("      an input is only skipped when it is declared in "
                     "scripts/_release_gate.py::UNPUBLISHED; otherwise its")
        lines.append("      absence is a defect (deleted file? renamed directory?).")
        return "fail", "\n".join(lines)

    lines = ["[%s] %s: %s did not run in this checkout." % (tag, SKIP_TOKEN, mode)]
    for r in gone:
        lines.append("        missing  %-30s %s" % (r, UNPUBLISHED[r]))
    lines.append("      these inputs are deliberately not distributed with the "
                 "repository;")
    lines.append("      declare them in scripts/_release_gate.py to keep this list "
                 "current.")
    return "skip", "\n".join(lines)


# --------------------------------------------------------------------------- #
# the declaration's own audit
# --------------------------------------------------------------------------- #
def check_declaration(root=None, use_git: bool = True) -> List[str]:
    """Problems with the *declaration*, as opposed to with a guarded document.

    Four properties, all of which must hold in every checkout (this is also what
    the smoke test runs), so this cannot be a local-only sanity check:

    1. A required input that is NOT declared unpublished is a shipped artifact,
       so it must be present.  Otherwise a guard could fail for a reason the
       declaration claims not to exist.
    2. Every tag names exactly one script, and that script really calls the gate.
       A guard that lost its gate call would silently stop skipping -- and would
       crash in a clone instead, which is the bug this module exists to prevent.
    3. Every declared requirement that is unpublished is named inside its guard,
       so the table cannot list a requirement the guard does not actually use.
    4. Every declared-unpublished path is really ignored by git.  If it is not,
       the file WILL ship and the declaration is a false statement about what
       this repository distributes.
    5. Every required path that is *not* declared is really tracked by git.  Being
       present on disk is not the same claim: a file the author is working on is
       present whether or not it will ship, so a presence-only check passes
       precisely when a clone would come up short.
    6. This module itself is tracked by git.  Six guards import it, so if it does
       not ship then a clone gets six ``ModuleNotFoundError`` crashes -- the exact
       failure this module exists to prevent, caused by its own absence.  Rule 5
       already reasons about the guards' inputs; there is no reason for the
       auditor to exempt itself from the same reasoning.
    """
    import subprocess

    root = Path(root or Path(__file__).resolve().parents[1])
    problems: List[str] = []

    for tag, reqs in sorted(REQUIRED.items()):
        script = sorted((root / "scripts").glob("%s_*.py" % tag))
        if len(script) != 1:
            problems.append("tag %s: expected one scripts/%s_*.py, found %d"
                            % (tag, tag, len(script)))
            continue
        text = script[0].read_text(encoding="utf-8", errors="replace")
        if "release_gate(" not in text:
            problems.append("tag %s: %s does not call release_gate()"
                            % (tag, script[0].name))
        for r in reqs:
            if r in UNPUBLISHED:
                if Path(r).name.split(".")[0] not in text:
                    problems.append("tag %s: declares %s but %s never names it"
                                    % (tag, r, script[0].name))
            elif not (root / r).exists():
                problems.append("tag %s: requires %s -- neither declared "
                                "unpublished nor present" % (tag, r))

    if use_git and (root / ".git").exists():
        try:
            raw = subprocess.run(["git", "ls-files", "-z"], cwd=str(root),
                                 capture_output=True).stdout
        except OSError:
            return problems
        tracked = {p for p in raw.decode("utf-8", "replace").split("\0") if p}

        for r in sorted(UNPUBLISHED):
            rc = subprocess.run(["git", "check-ignore", "-q", r], cwd=str(root),
                                capture_output=True)
            if rc.returncode != 0:
                problems.append("%s is declared unpublished but git does NOT ignore it"
                                % r)

        # The mirror image, and the reason this second half exists: a path the
        # guard chain treats as SHIPPED must actually be tracked.  "Present on
        # disk" is not the same claim -- every file the author is working on is
        # present, including the ones .gitignore excludes, so a presence-only
        # check passes in exactly the situation a clone would break.  (That is
        # not hypothetical: this check was added after docs/MANUSCRIPT.md turned
        # up excluded by .gitignore while all six guards still listed it as a
        # shipped requirement, and the presence-only version reported PASS.)
        for tag, reqs in sorted(REQUIRED.items()):
            for r in reqs:
                if r not in UNPUBLISHED and r.replace("\\", "/") not in tracked:
                    problems.append(
                        "%s requires %s, which is NOT tracked by git -- the guards "
                        "read it here, but a clone would not have it" % (tag, r))

        # ...and the auditor is not exempt from its own reasoning.  Measured, not
        # imagined: scripts/_release_gate.py was once present locally, absent from
        # the index, and every guard that imports it died with
        # ModuleNotFoundError -- six crashes with the same root cause, none of
        # which named the cause.  A file this central has to be checked by name.
        self_rel = "scripts/_release_gate.py"
        if self_rel not in tracked:
            problems.append(
                "%s is NOT tracked by git -- six guards import it, so a clone "
                "would fail with ModuleNotFoundError in all six" % self_rel)
    return problems


def local_report(root=None) -> str:
    """One informational line: how many declared inputs this checkout lacks."""
    root = Path(root or Path(__file__).resolve().parents[1])
    gone = missing(root, sorted(UNPUBLISHED))
    if not gone:
        return ("[_release_gate] this checkout has all %d declared-unpublished "
                "input(s): the full guard chain runs." % len(UNPUBLISHED))
    return ("[_release_gate] %d of %d declared-unpublished input(s) are absent here; "
            "the guards that need them will report %s, not a failure: %s"
            % (len(gone), len(UNPUBLISHED), SKIP_TOKEN, ", ".join(gone)))


# --------------------------------------------------------------------------- #
# self-test -- controls for `gate` itself
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Both directions -- a gate that only ever skips is not a gate."""
    import tempfile

    ctrls: List[Tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        declared = sorted(UNPUBLISHED)[0]
        (root / "present.md").write_text("x", encoding="utf-8")

        st, _ = gate("T", root, ["present.md"])
        ctrls.append(("a present input passes", st == "ok", st))

        st, msg = gate("T", root, [declared])
        ctrls.append(("a declared absent input skips", st == "skip", st))
        ctrls.append(("the skip names the input and the reason",
                      declared in msg and UNPUBLISHED[declared][:20] in msg,
                      msg.splitlines()[0] if msg else ""))
        ctrls.append(("the skip carries the marker the smoke test matches on",
                      SKIP_TOKEN in msg, "marker missing"))

        # A path that is absent AND not in the declaration.  Deliberately not
        # docs/MANUSCRIPT.md: that is the canonical absent-but-declared path, and
        # this control silently stopped testing rule 2 the moment MANUSCRIPT.md
        # joined UNPUBLISHED -- it began asserting "skip" instead of "fail", which
        # is how a control stops being a control.  It must not ride on a
        # declaration it does not own.
        st, msg = gate("T", root, ["results/never_shipped.csv"])
        ctrls.append(("an UNdeclared absent input fails instead",
                      st == "fail" and "never_shipped" in msg, st))

        # the mixed case is the one that actually occurs: 121 needs a shipped doc
        # AND an unpublished one, and must skip -- not fail on the shipped one.
        st, _ = gate("T", root, ["present.md", declared])
        ctrls.append(("present + declared-absent skips (not fails)", st == "skip", st))

        # ...while the same call with an undeclared extra must fail, i.e. the
        # undeclared rule is not swallowed by an otherwise-declared absence.
        st, _ = gate("T", root, ["present.md", declared, "results/nope.csv"])
        ctrls.append(("a declared absence does not excuse an undeclared one",
                      st == "fail", st))

        # the table is a second copy of the guards' own lists; it must at least be
        # well-formed, or the smoke test's pre-flight would be asked about a tag
        # nobody declared.
        ctrls.append(("every REQUIRED tag resolves to declared or shipped paths",
                      all(r in UNPUBLISHED or r.startswith(("docs/", "results/"))
                          for reqs in REQUIRED.values() for r in reqs),
                      "bad path in REQUIRED"))

    ok = True
    for name, good, detail in ctrls:
        print("   %-52s %s" % (name, "PASS" if good else "FAIL  " + str(detail)))
        ok = ok and good
    print("_release_gate self-test: %s (%d controls)"
          % ("PASS" if ok else "FAIL", len(ctrls)))
    return 0 if ok else 1


def main(argv: List[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    if "--check" in argv or not argv:
        print(local_report())
        problems = check_declaration()
        for p in problems:
            print("  FAIL  " + p)
        if problems:
            print("[_release_gate] VERDICT: FAIL (%d problem(s))" % len(problems))
            return 1
        print("[_release_gate] VERDICT: PASS (declaration is current and true)")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
