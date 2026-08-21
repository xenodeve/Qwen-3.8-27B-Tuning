"""Execution-verified coding tasks that REQUIRE information buried in a long context.

The 16K corpus cannot judge Q8_0 KV, because at 512 MiB of KV there is nothing for
Q8 to reclaim and the tasks never force the model to attend far back. This corpus
fixes both halves of that:

- One large shared repository prefix (~45K tokens) is prepended to every task, so
  the KV cache is deeply populated and `cache_prompt` lets all tasks reuse the one
  expensive prefill.
- Every task's correct answer depends on a specific fact planted at a known depth
  in that prefix. A model answering from priors, or one whose attention over a
  quantized cache has degraded, produces code that fails the assertions.

The planted facts are deliberately arbitrary (odd constants, non-obvious field
names, a specific buggy line) so they cannot be guessed. Verification is still
subprocess execution against assertions — no judge, no partial credit.
"""

# Facts planted in the repository text, at increasing depth. The values are
# arbitrary on purpose: no prior can supply them.
PLANTED = [
    dict(shard=17, retries=7,  timeout=3100, field="payload_digest"),
    dict(shard=94, retries=3,  timeout=8700, field="lease_epoch"),
    dict(shard=203, retries=11, timeout=1450, field="shadow_offset"),
    dict(shard=310, retries=5,  timeout=6200, field="drain_token"),
]


def _plain_block(i):
    return f'''
class Handler{i:04d}:
    """Routine shard handler for shard {i}."""
    SHARD_ID = {i}
    MAX_RETRIES = 2
    TIMEOUT_MS = 500

    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.cache = {{}}

    def process(self, item):
        key = item.get("id")
        if key in self.cache:
            return self.cache[key]
        result = self.registry.lookup(key).transform(item, self.config)
        self.cache[key] = result
        return result
'''


def _planted_block(f):
    """A shard whose constants differ from every other block in the file."""
    return f'''
class Handler{f["shard"]:04d}:
    """Special-cased shard {f["shard"]} -- do not copy these constants elsewhere."""
    SHARD_ID = {f["shard"]}
    MAX_RETRIES = {f["retries"]}
    TIMEOUT_MS = {f["timeout"]}
    CHECKSUM_FIELD = "{f["field"]}"

    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.cache = {{}}

    def process(self, item):
        key = item.get(self.CHECKSUM_FIELD)
        if key in self.cache:
            return self.cache[key]
        result = self.registry.lookup(key).transform(item, self.config)
        self.cache[key] = result
        return result
'''


def build_repo(n_blocks=320):
    """~44K tokens of repository text with the planted shards spread through it.

    Routine blocks are numbered from a stream that SKIPS the planted shard
    numbers. Without that skip, `Handler0017` was emitted twice — once as a
    routine block at index 17 and once as the planted block — so a task asking
    for "the class for shard 17" had two contradictory answers in context and
    the corpus measured nothing. Caught by
    `test_repo_contains_every_planted_shard_exactly_once`.
    """
    planted_ids = {f["shard"] for f in PLANTED}
    by_position = {}
    step = max(1, n_blocks // (len(PLANTED) + 1))
    for k, f in enumerate(PLANTED, start=1):
        by_position[k * step] = f

    out = ["# ---- repository dump: src/handlers/ ----\n"]
    next_routine = 0
    for pos in range(n_blocks):
        f = by_position.get(pos)
        if f:
            out.append(_planted_block(f))
            continue
        while next_routine in planted_ids:
            next_routine += 1
        out.append(_plain_block(next_routine))
        next_routine += 1
    return "".join(out)


# Each task names a planted shard and asks for code whose correctness depends on
# the constants only that shard carries.
DEEP_TASKS = [
    dict(
        id="deep_retries_17",
        depth="early",
        prompt=("From the repository above, find the class for shard 17. Write a Python "
                "function `retry_budget(attempts)` that returns True when `attempts` is "
                "strictly less than that class's MAX_RETRIES value, and False otherwise. "
                "Output only the code."),
        test="""
assert retry_budget(0) is True
assert retry_budget(6) is True
assert retry_budget(7) is False
assert retry_budget(8) is False
""",
    ),
    dict(
        id="deep_timeout_94",
        depth="mid",
        prompt=("From the repository above, find the class for shard 94. Write a Python "
                "function `deadline(now_ms)` returning `now_ms` plus that class's "
                "TIMEOUT_MS value. Output only the code."),
        test="""
assert deadline(0) == 8700
assert deadline(1000) == 9700
assert deadline(-700) == 8000
""",
    ),
    dict(
        id="deep_field_203",
        depth="late",
        prompt=("From the repository above, find the class for shard 203. Write a Python "
                "function `extract_key(item)` that returns `item` looked up by that "
                "class's CHECKSUM_FIELD name, or None if the key is absent. "
                "Output only the code."),
        test="""
assert extract_key({"shadow_offset": "abc"}) == "abc"
assert extract_key({"payload_digest": "abc"}) is None
assert extract_key({}) is None
""",
    ),
    dict(
        id="deep_combine_310",
        depth="deep",
        prompt=("From the repository above, find the class for shard 310. Write a Python "
                "function `describe()` returning the string "
                "\"<CHECKSUM_FIELD>:<MAX_RETRIES>:<TIMEOUT_MS>\" built from that class's "
                "three values, separated by colons. Output only the code."),
        test="""
assert describe() == "drain_token:5:6200"
""",
    ),
    dict(
        id="deep_contrast_17_94",
        depth="cross",
        prompt=("From the repository above, compare the classes for shard 17 and shard 94. "
                "Write a Python function `slower_shard()` that returns the SHARD_ID of "
                "whichever of the two has the LARGER TIMEOUT_MS. Output only the code."),
        test="""
assert slower_shard() == 94
""",
    ),
    dict(
        id="deep_default_contrast",
        depth="cross",
        prompt=("From the repository above, most Handler classes share one common "
                "MAX_RETRIES value while four shards override it. Write a Python function "
                "`is_override(max_retries)` returning True when the given value differs "
                "from the common default. Output only the code."),
        test="""
assert is_override(2) is False
assert is_override(7) is True
assert is_override(3) is True
assert is_override(11) is True
assert is_override(5) is True
""",
    ),
]
