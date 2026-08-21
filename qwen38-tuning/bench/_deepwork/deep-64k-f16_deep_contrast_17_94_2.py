def slower_shard():
    """Return the SHARD_ID of the handler with the larger TIMEOUT_MS between shard 17 and shard 94."""
    # Handler0017: SHARD_ID=17, TIMEOUT_MS=3100
    # Handler0094: SHARD_ID=94, TIMEOUT_MS=8700
    if 8700 > 3100:
        return 94
    else:
        return 17


# ---- verification ----

assert slower_shard() == 94
