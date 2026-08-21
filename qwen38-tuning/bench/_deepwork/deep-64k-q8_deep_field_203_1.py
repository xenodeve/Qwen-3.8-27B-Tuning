def extract_key(item):
    return item.get("shadow_offset")


# ---- verification ----

assert extract_key({"shadow_offset": "abc"}) == "abc"
assert extract_key({"payload_digest": "abc"}) is None
assert extract_key({}) is None
