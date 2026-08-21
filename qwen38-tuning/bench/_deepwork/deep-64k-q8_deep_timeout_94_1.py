def deadline(now_ms):
    return now_ms + 8700


# ---- verification ----

assert deadline(0) == 8700
assert deadline(1000) == 9700
assert deadline(-700) == 8000
