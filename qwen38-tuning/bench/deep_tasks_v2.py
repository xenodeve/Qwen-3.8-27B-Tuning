"""Harder deep-context corpus — built because v1 ceilinged.

v1 scored 18/18 on both F16 and Q8_0 KV. A test where both arms are perfect
bounds the damage but cannot resolve a small one, so it cannot answer "is Q8
slightly worse". This version is designed to land F16 below 100% so there is
headroom to measure into.

Four things make it harder, each targeting a different way long-context
retrieval fails:

1. **Confusable neighbours.** Every planted shard has decoys whose IDs differ by
   a digit or a transposition (0203 / 0230 / 2003) and whose constants are near
   misses. Grabbing the wrong block now produces a wrong answer instead of the
   right one.
2. **Multi-hop.** Some shards carry a DEPENDS_ON pointing at another shard, so
   the answer needs two retrievals chained through the context.
3. **Aggregation.** One task needs values from all four planted shards at once;
   missing any single one fails it.
4. **Depth.** The last planted shard sits at ~95% of the prefix, where attention
   over a quantized cache is most likely to degrade.

Verification is unchanged: subprocess execution against assertions.
"""

# shard -> its constants. Decoys are generated around each of these.
PLANTED = [
    dict(shard=203,  retries=11, timeout=1450, field="shadow_offset", depends=None),
    dict(shard=417,  retries=6,  timeout=9320, field="lease_epoch",   depends=203),
    dict(shard=1508, retries=13, timeout=2870, field="drain_token",   depends=417),
    dict(shard=2941, retries=4,  timeout=7605, field="parity_seal",   depends=1508),
]

# IDs that look like a planted shard but are not. Their constants are near
# misses, so a retrieval that lands on the wrong block is punished.
def _decoys_for(f):
    s = f["shard"]
    ids = [int(str(s)[::-1]), s * 10 % 10000, s + 27]
    out = []
    for k, d in enumerate(ids):
        if d == s or d <= 0:
            continue
        out.append(dict(shard=d,
                        retries=f["retries"] + 1 + k,
                        timeout=f["timeout"] + 10 * (k + 1),
                        field=f["field"].upper()))
    return out


def _plain_block(i):
    return f'''
class Handler{i:04d}:
    """Routine shard handler for shard {i}."""
    SHARD_ID = {i}
    MAX_RETRIES = 2
    TIMEOUT_MS = 500

    def process(self, item):
        return self.registry.lookup(item.get("id")).transform(item, self.config)
'''


def _decoy_block(d):
    return f'''
class Handler{d["shard"]:04d}:
    """Deprecated mirror of a special shard. Do NOT use these values."""
    SHARD_ID = {d["shard"]}
    MAX_RETRIES = {d["retries"]}
    TIMEOUT_MS = {d["timeout"]}
    CHECKSUM_FIELD = "{d["field"]}"
    DEPRECATED = True

    def process(self, item):
        raise RuntimeError("deprecated mirror")
'''


def _planted_block(f):
    dep = f'    DEPENDS_ON = {f["depends"]}\n' if f["depends"] else ""
    return f'''
class Handler{f["shard"]:04d}:
    """Authoritative handler for shard {f["shard"]}."""
    SHARD_ID = {f["shard"]}
    MAX_RETRIES = {f["retries"]}
    TIMEOUT_MS = {f["timeout"]}
    CHECKSUM_FIELD = "{f["field"]}"
    DEPRECATED = False
{dep}
    def process(self, item):
        return self.registry.lookup(item.get(self.CHECKSUM_FIELD)).transform(item, self.config)
'''


def build_repo(n_blocks=680):
    """~44K tokens: routine blocks, decoys, and the four authoritative shards.

    Planted shards go at 25 / 50 / 75 / 95 percent depth. Each decoy is placed
    just BEFORE its planted shard, so a model scanning forward meets the wrong
    block first.
    """
    reserved = {f["shard"] for f in PLANTED}
    decoys = {}
    for f in PLANTED:
        for d in _decoys_for(f):
            decoys[d["shard"]] = d
            reserved.add(d["shard"])

    positions = [int(n_blocks * p) for p in (0.25, 0.50, 0.75, 0.95)]
    planted_at = dict(zip(positions, PLANTED))
    # Decoys immediately ahead of their target.
    decoy_at = {}
    for pos, f in planted_at.items():
        for k, d in enumerate(_decoys_for(f), start=1):
            decoy_at[pos - k] = d

    out = ["# ---- repository dump: src/handlers/ ----\n"]
    nxt = 0
    for pos in range(n_blocks):
        if pos in planted_at:
            out.append(_planted_block(planted_at[pos]))
            continue
        if pos in decoy_at:
            out.append(_decoy_block(decoy_at[pos]))
            continue
        while nxt in reserved:
            nxt += 1
        out.append(_plain_block(nxt))
        nxt += 1
    return "".join(out)


DEEP_TASKS_V2 = [
    dict(id="v2_authoritative_203", depth="25%",
         prompt=("From the repository above, several classes mention shard 203 or "
                 "similar ids, but only one is authoritative (DEPRECATED = False). "
                 "Write a Python function `retries()` returning that authoritative "
                 "class's MAX_RETRIES value. Output only the code."),
         test="assert retries() == 11\n"),
    dict(id="v2_authoritative_417", depth="50%",
         prompt=("From the repository above, find the AUTHORITATIVE class for shard 417 "
                 "(DEPRECATED = False). Write a Python function `timeout()` returning its "
                 "TIMEOUT_MS. Output only the code."),
         test="assert timeout() == 9320\n"),
    dict(id="v2_hop_417_to_203", depth="hop"    ,
         prompt=("From the repository above, find the authoritative class for shard 417, "
                 "read its DEPENDS_ON value, then find the authoritative class for THAT "
                 "shard. Write a Python function `upstream_field()` returning the "
                 "CHECKSUM_FIELD of the class it depends on. Output only the code."),
         test='assert upstream_field() == "shadow_offset"\n'),
    dict(id="v2_hop_1508_to_417", depth="hop",
         prompt=("From the repository above, find the authoritative class for shard 1508, "
                 "follow its DEPENDS_ON to that shard's authoritative class, and write a "
                 "Python function `upstream_timeout()` returning that upstream class's "
                 "TIMEOUT_MS. Output only the code."),
         test="assert upstream_timeout() == 9320\n"),
    dict(id="v2_deepest_2941", depth="95%",
         prompt=("From the repository above, find the authoritative class for shard 2941. "
                 "Write a Python function `seal()` returning the string "
                 "\"<CHECKSUM_FIELD>/<MAX_RETRIES>/<TIMEOUT_MS>\" from its values, "
                 "separated by forward slashes. Output only the code."),
         test='assert seal() == "parity_seal/4/7605"\n'),
    dict(id="v2_chain_2941_to_1508", depth="hop",
         prompt=("From the repository above, find the authoritative class for shard 2941, "
                 "follow DEPENDS_ON one step, and write a Python function "
                 "`prev_retries()` returning that class's MAX_RETRIES. Output only the code."),
         test="assert prev_retries() == 13\n"),
    dict(id="v2_sum_all_retries", depth="aggregate",
         prompt=("From the repository above, there are exactly four AUTHORITATIVE "
                 "classes (DEPRECATED = False) with a CHECKSUM_FIELD. Write a Python "
                 "function `total_retries()` returning the SUM of their MAX_RETRIES "
                 "values. Output only the code."),
         test="assert total_retries() == 34\n"),
    dict(id="v2_max_timeout_shard", depth="aggregate",
         prompt=("Among the four AUTHORITATIVE classes in the repository above, write a "
                 "Python function `slowest()` returning the SHARD_ID of the one with the "
                 "largest TIMEOUT_MS. Output only the code."),
         test="assert slowest() == 417\n"),
    dict(id="v2_field_order", depth="aggregate",
         prompt=("Write a Python function `fields_by_shard()` returning a list of the "
                 "CHECKSUM_FIELD strings of the four AUTHORITATIVE classes in the "
                 "repository above, ordered by ascending SHARD_ID. Output only the code."),
         test=('assert fields_by_shard() == '
               '["shadow_offset", "lease_epoch", "drain_token", "parity_seal"]\n')),
    dict(id="v2_reject_deprecated", depth="distractor",
         prompt=("The repository above contains deprecated mirror classes whose "
                 "CHECKSUM_FIELD is upper-case. Write a Python function "
                 "`is_authoritative(checksum_field)` returning True only when the given "
                 "field string matches one of the four AUTHORITATIVE classes. "
                 "Output only the code."),
         test=('assert is_authoritative("drain_token") is True\n'
               'assert is_authoritative("DRAIN_TOKEN") is False\n'
               'assert is_authoritative("lease_epoch") is True\n'
               'assert is_authoritative("nope") is False\n')),
]
