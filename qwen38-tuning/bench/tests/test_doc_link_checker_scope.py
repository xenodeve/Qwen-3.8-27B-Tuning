"""Guards the one thing that makes check-doc-links.py trustworthy: its scope.

THE INCIDENT (2026-08-22, issue #17). llama.cpp was cloned into the tree so
PR #27342 could be compiled. The checker walked into it and reported 39 broken
links -- every one of them upstream's, none of them this repo's.

The gate failing was the harmless half. The damage is that a genuinely broken
link in docs/ would have been one row among 39 rows of noise, and the honest
reading of a red gate becomes "that's just the vendored tree again". A checker
whose output is routinely ignored is worse than no checker.

The first fix was a hand-maintained list of directory names to skip. That is a
second copy of .gitignore that drifts from it, so the checker now asks git
instead. These tests pin the BEHAVIOUR -- what is and is not checked -- rather
than the mechanism, so the next person is free to change how it is done.
"""
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CHECKER = REPO / "scripts" / "check-doc-links.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_doc_links", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def listed():
    """Paths the checker would read, relative to the repo root, slash-separated."""
    mod = _load()
    return {
        os.path.relpath(p, REPO).replace(os.sep, "/")
        for p in mod.markdown_files()
    }


def test_checker_exists():
    assert CHECKER.is_file(), f"missing {CHECKER}"


def test_it_finds_the_repo_docs(listed):
    """A scope guard that passed on an empty set would guard nothing."""
    assert "CLAUDE.md" in listed
    assert "docs/OPEN-WORK-LEDGER.md" in listed
    assert len(listed) > 50, f"only {len(listed)} markdown files found -- scope collapsed"


def test_nothing_gitignored_is_checked(listed):
    """The vendored trees, build output and caches are gitignored.

    Asking git which of the listed files it ignores is the same question the
    incident asked, without naming any directory -- so a newly vendored tree is
    covered the day it appears.
    """
    if not listed:
        pytest.fail("no markdown files listed")
    # NUL-separated both ways. With newline separators on Windows, text=True
    # rewrites the separator to CRLF and git takes the CR as part of the
    # pathname -- which stops a negation pattern matching and reports a tracked
    # file as ignored. The test then fails on a fault of its own making.
    proc = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--stdin", "-z"],
        input=chr(0).join(sorted(listed)).encode("utf-8"),
        capture_output=True,
    )
    ignored = [q for q in proc.stdout.decode("utf-8").split(chr(0)) if q.strip()]
    assert not ignored, (
        "the link checker would read files git ignores: " + ", ".join(ignored[:5])
    )


def test_the_logs_readme_is_in_the_repo():
    """It was not, and the checker was reading a file no clone has.

    'logs/' excluded the directory, so git never descended into it and the
    folder's own map was invisible to everyone but this machine. Recorded here
    because the convention -- every folder has a README -- is only real if the
    README ships.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch",
         "qwen38-tuning/logs/README.md"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, "qwen38-tuning/logs/README.md is not tracked by git"
