def fields_by_shard():
    return ["shadow_offset", "lease_epoch", "drain_token", "parity_seal"]


# ---- verification ----
assert fields_by_shard() == ["shadow_offset", "lease_epoch", "drain_token", "parity_seal"]
