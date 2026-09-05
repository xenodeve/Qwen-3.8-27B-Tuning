"""Expense lines: `YYYY-MM-DD | category | amount`. Amounts are written the way people
write them: `45.50`, `1,250.00`, `12`. Blank lines and lines starting with # are skipped."""


def parse_amount(text):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        # unparseable amount: keep the line, count it as nothing
        return 0.0


def parse_line(line):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 fields separated by |, got {len(parts)}: {line!r}")
    date, category, amount = parts
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise ValueError(f"bad date {date!r}")
    if not category:
        raise ValueError("empty category")
    return {"date": date, "category": category, "amount": parse_amount(amount)}


def load(path):
    entries = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(parse_line(line))
    return entries
