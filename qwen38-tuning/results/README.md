# results — raw JSONL, one row per boot

The source of every number in [`../../docs/reports/`](../../docs/reports/). Each
line is one measurement; lines with `"kind": "PAIRED"` or `"kind": "SUMMARY"` are
verdicts computed over the rows above them.

| file | what is in it |
|---|---|
| `retry-bench.jsonl` | **the decision metric** — 30 coding tasks per arm, per-attempt evidence, and a SUMMARY with `accepted_of_decided` and `output_contract_pct` |
| `kv-layers-16k.jsonl` | the 16K layer screen — kernel, n-gram, placement, CPU |
| `kv-depth-levers.jsonl` | the same levers at 131,072 and 163,840, plus MTP placement |
| `ctx-ceiling-q38.jsonl` | deepest fully-resident context per artifact, with the `extra` flags it was measured under |
| `kv-sweep*.jsonl` | KV type at depth, per artifact |
| `arena-*.jsonl` | 16K paired comparisons between artifacts |
| `answer-screen*.jsonl` | the 4-minute gate |
| `deep-quality.jsonl` | retrieval at depth — **Q4 only, still** |
| `stability-gate.jsonl` | 100 turns, prefix invalidated every tenth |
| `kv-kernel-screen.jsonl` | which KV types have a fast kernel |

---

## Reading a row without fooling yourself

**Check `free_before`.** Residency is conditional on how much VRAM the desktop
was holding at boot. `AD-IQ1_M` was `65+0` at 128K with 10,730 MiB free and
`65+1` with ~9,796.

**Check `extra` and `args`.** A ceiling or throughput row measured under
`--ctx-checkpoints 8` is not comparable to one that was not, and both are in the
same file.

**Check `greedy_hash`.** Identical hashes mean identical output. The five n-gram
decoders all match the control; the `-ot` arms do not, because CPU and GPU
floating-point differ.

**A `PAIRED` row with `"resolved": false` is not a small result — it is no
result.** Effects below 13.6 %, or with an inconsistent sign across rounds,
cannot be separated from restarting the server.

**Rows carry no artifact name in some older files.** Attribute them from the
step log in `../logs/` rather than from position in the file.
