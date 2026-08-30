"""Guards for a benchmark that clones real repositories and deletes them after.

WHY THESE EXIST. The real-task benchmark takes its work from six live projects.
Two of them are not clean checkouts:

    MangaDock      perf/mit-layout-fit-and-merge   333 uncommitted files, 1 stash
    T4 Fastwork    master                          440 uncommitted files, 4 stashes

That is days of work existing nowhere else. The benchmark ends by deleting what
it made, so a benchmark that ran in place would turn its own cleanup step into
the deletion of real work. The developer's instruction was explicit: clone
separately, never touch the live tree.

A comment saying "be careful" is not a guard. These are.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import PROTECTED_ROOTS, assert_deletable, is_protected


def test_the_live_project_root_is_protected():
    assert is_protected(Path(r"D:\Github")) is True


@pytest.mark.parametrize("p", [
    r"D:\Github\MangaDock",
    r"D:\Github\MangaDock\MIT\translate.py",
    r"D:\Github\T4 Fastwork",
    r"d:\github\mangadock",          # case differs on Windows; the guard must not
])
def test_everything_under_it_is_protected(p):
    assert is_protected(Path(p)) is True


def test_the_scratch_root_is_not_protected():
    assert is_protected(Path(r"D:\bench-scratch\2026-08-22-a\clones\x")) is False


def test_deleting_a_live_project_is_refused():
    with pytest.raises(ValueError, match="protected"):
        assert_deletable(Path(r"D:\Github\MangaDock"), Path(r"D:\bench-scratch\run"))


def test_deleting_outside_the_declared_scratch_root_is_refused():
    """Even somewhere harmless. The rule is one root, declared up front."""
    with pytest.raises(ValueError, match="outside"):
        assert_deletable(Path(r"D:\somewhere-else\tmp"), Path(r"D:\bench-scratch\run"))


def test_deleting_the_scratch_root_itself_is_allowed():
    assert_deletable(Path(r"D:\bench-scratch\run"), Path(r"D:\bench-scratch\run"))


def test_deleting_inside_the_scratch_root_is_allowed():
    assert_deletable(Path(r"D:\bench-scratch\run\clones\xeno-skills"),
                     Path(r"D:\bench-scratch\run"))


def test_a_relative_path_is_refused_rather_than_resolved_against_cwd():
    """`Remove-Item clones\\x` means something different depending on where you
    stand, and where you stand is not a thing this benchmark controls."""
    with pytest.raises(ValueError):
        assert_deletable(Path("clones/x"), Path(r"D:\bench-scratch\run"))


def test_a_scratch_root_that_is_itself_protected_is_refused():
    """The whole guard collapses if the root can be set to the live tree."""
    with pytest.raises(ValueError, match="protected"):
        assert_deletable(Path(r"D:\Github\scratch\x"), Path(r"D:\Github\scratch"))


def test_the_protected_list_is_not_empty():
    """An empty list would make every check above pass while guarding nothing."""
    assert PROTECTED_ROOTS
