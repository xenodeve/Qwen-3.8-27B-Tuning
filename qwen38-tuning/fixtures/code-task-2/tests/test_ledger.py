from ledger import parse_line, load, monthly_totals


def test_parse_line_basic():
    e = parse_line("2026-08-03 | coffee | 45.50")
    assert e == {"date": "2026-08-03", "category": "coffee", "amount": 45.5}


def test_parse_line_rejects_wrong_field_count():
    try:
        parse_line("2026-08-03 | coffee")
    except ValueError:
        return
    assert False


def test_load_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("# c\n\n2026-08-03 | coffee | 45.50\n", encoding="utf-8")
    assert len(load(p)) == 1


def test_monthly_totals_groups_by_month():
    entries = [parse_line("2026-08-03 | coffee | 45.50"), parse_line("2026-09-02 | books | 890")]
    assert monthly_totals(entries) == {"2026-08": 45.5, "2026-09": 890.0}
