def is_override(max_retries):
    return max_retries != 2


# ---- verification ----

assert is_override(2) is False
assert is_override(7) is True
assert is_override(3) is True
assert is_override(11) is True
assert is_override(5) is True
