"""Reports over parsed entries."""
from collections import defaultdict


def monthly_totals(entries):
    """{'YYYY-MM': total} for every month that has at least one entry."""
    totals = defaultdict(float)
    for e in entries:
        totals[e["date"][:7]] += e["amount"]
    return dict(sorted(totals.items()))
