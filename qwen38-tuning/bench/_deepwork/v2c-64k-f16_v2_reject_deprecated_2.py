def is_authoritative(checksum_field):
    return checksum_field in ("shadow_offset", "lease_epoch", "drain_token", "parity_seal")


# ---- verification ----
assert is_authoritative("drain_token") is True
assert is_authoritative("DRAIN_TOKEN") is False
assert is_authoritative("lease_epoch") is True
assert is_authoritative("nope") is False
