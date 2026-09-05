# ledger

Reads expense lines (`YYYY-MM-DD | category | amount`) and reports monthly totals.
No dependencies beyond pytest.

```python
from ledger import load, monthly_totals
monthly_totals(load("data/sample.txt"))
```

Run the tests: `python -m pytest -q`
