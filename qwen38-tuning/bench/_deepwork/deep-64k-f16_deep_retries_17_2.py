def retry_budget(attempts):
    return attempts < 7


# ---- verification ----

assert retry_budget(0) is True
assert retry_budget(6) is True
assert retry_budget(7) is False
assert retry_budget(8) is False
