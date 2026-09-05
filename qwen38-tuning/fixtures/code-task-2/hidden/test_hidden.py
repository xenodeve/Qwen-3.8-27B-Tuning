"""Hidden tests, copied in by the harness after the run."""
import os
import pytest
from ledger import parse_line, load, monthly_totals


def top(entries, n):
    from ledger import report
    return report.top(entries, n)


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_thousands_separators_are_read_not_zeroed():
    assert parse_line("2026-08-03 | rent | 12,000.00")["amount"] == 12000.0
    assert parse_line("2026-08-15 | groceries | 1,250.00")["amount"] == 1250.0


def test_sample_file_monthly_totals_are_right():
    entries = load(os.path.join(HERE, "data", "sample.txt"))
    assert monthly_totals(entries) == {"2026-08": 13341.0, "2026-09": 12890.0}


def test_garbage_amount_raises_instead_of_counting_zero():
    with pytest.raises(ValueError):
        parse_line("2026-08-03 | coffee | forty-five")


def test_empty_amount_raises_instead_of_counting_zero():
    # the unnamed error path: an empty third field is not "0.00". (A negative amount was the
    # original case; a run at 17:43 reasoned "refunds are valid", which is a fair reading the
    # brief does not exclude, so the case moved to one no reading makes valid.)
    with pytest.raises(ValueError):
        parse_line("2026-08-03 | coffee | ")


def test_top_returns_largest_entries_per_month_ties_by_category():
    entries = load(os.path.join(HERE, "data", "sample.txt"))
    t = top(entries, 2)
    assert list(t) == ["2026-08", "2026-09"]
    assert [(e["category"], e["amount"]) for e in t["2026-08"]] == [("rent", 12000.0), ("groceries", 1250.0)]
    assert [(e["category"], e["amount"]) for e in t["2026-09"]] == [("rent", 12000.0), ("books", 890.0)]


def test_top_with_n_larger_than_entries_and_ties():
    entries = [parse_line("2026-08-01 | b | 10"), parse_line("2026-08-01 | a | 10"), parse_line("2026-08-01 | c | 5")]
    assert [e["category"] for e in top(entries, 5)["2026-08"]] == ["a", "b", "c"]


def test_readme_documents_top():
    readme = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
    assert "top(" in readme


def test_no_new_dependency():
    for name in ("requirements.txt", "pyproject.toml", "setup.py", "package.json"):
        assert not os.path.exists(os.path.join(HERE, name)), f"{name} was added"
