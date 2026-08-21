def is_authoritative(checksum_field):
    authoritative_fields = {"shadow_offset", "lease_epoch", "drain_token", "parity_seal"}
    return checksum_field in authoritative_fields


# ---- verification ----
assert is_authoritative("drain_token") is True
assert is_authoritative("DRAIN_TOKEN") is False
assert is_authoritative("lease_epoch") is True
assert is_authoritative("nope") is False
