# PAL #149 focused GSQ and EXL3 results — 2026-09-22

The developer narrowed the immediate focus to **GSQ IQ3_S-MTP** and **EXL3 SC4.0bpw H5**, with Q6 optional. Both were run on the second real-project task selected from PAL issue #149: a standalone `RunJournal` layer over the existing `RecordStore`. This focused result is separate from the incomplete five-task Quick Discrimination plan; the other Q4-class tasks were not run in this slice.

## Frozen task

Only `tools/run_journal.py` and `tests/test_run_journal.py` could change. The contract required one atomic file-per-agent record through `RecordStore.put`, collision-proof length-prefixed identities, deterministic reads, restart durability, duplicate rejection, corruption propagation/isolation, caller-result immutability, concurrency safety and no JSONL/`O_APPEND`/home-store writes. Parent-only hidden tests were validated against the untouched scaffold (**9 failed**) and an independent gold implementation (**9 passed**) before model exposure.

## Results

| Arm | Hidden | Visible | RED→GREEN | Time | Hard result |
|---|---:|---:|---:|---:|---|
| **GSQ IQ3_S-MTP** | **9/9** | **40/40** | **confirmed** | **742.688s** | **ACCEPTED** |
| EXL3 SC4.0bpw H5 | 9/9 | 21/21 | **not confirmed** | **1024.859s** | **FAILED_CONTRACT** |

GSQ's session had15 assistant messages,3 Edit,15 Read and4 fixed-test calls,2 tool errors,75,396 thinking characters and39,955 high-water input tokens. EXL3 had16 messages,3 Edit,11 Read and3 fixed-test calls,1 tool error,101,841 thinking characters. Both final source scopes contain only the two allowed files and both hidden/visible suites pass.

EXL3 is rejected because the protected test journal does not prove the required RED→GREEN sequence with a changed test hash before implementation. This is a workflow-contract failure, not a hidden functional failure. No overlap audit is needed for this task: the nine-case journal oracle is the complete frozen contract.

## Universal acceptance and engineering-quality assessment

### GSQ IQ3_S-MTP

- **Layer A hard result:** `ACCEPTED` for this frozen #149 contract: hidden9/9, visible40/40, exact two-file scope, complete Docker/evidence record and measured RED→GREEN.
- **Contract fidelity:** strong. Implements one atomic `RecordStore.put`, self-describing payload, length-prefixed identity, deterministic identity sorting, restart and corruption behavior.
- **Abstraction/causal reasoning:** good. Reuses the existing store instead of creating JSONL or a parallel cache. It explicitly reasons about delimiter collisions and delegates filesystem atomicity to `RecordStore`.
- **Verification quality:** strong but not perfect. It authored 40 visible tests and the independent9-case oracle passed. The hidden oracle did not test concurrent duplicate `(run,agent)` calls or caller mutation after append.
- **Engineering quality:** **Grade B**. The implementation imports private `_SAFE_IDENTITY`/`_MAX_IDENTITY_LENGTH` internals from `RecordStore`, coupling the new layer to private names. The duplicate check is a read-then-write sequence, so simultaneous duplicate appends are not serialized by the journal itself; the frozen oracle only required concurrent distinct agents. These are non-blocking for the accepted frozen contract but should be addressed before wiring it into production.
- **Agent discipline:** mostly strong. Fifteen reads and four fixed-test calls, two invalid directory-read attempts, no scope escape or unsupported claims. Its tests and final diff stayed within the contract.

### EXL3 SC4.0bpw H5

- **Layer A hard result:** `FAILED_CONTRACT`, despite hidden9/9, visible21/21 and exact scope, because the protected journal does not prove RED with changed tests followed by GREEN with the same final test hash.
- **Contract fidelity:** functionally feasible in the observed oracle; no hidden behavior failed. Do not promote this to `ACCEPTED` because the workflow contract is mandatory.
- **Abstraction/causal reasoning:** implementation appears source-contract aligned, but the measured workflow evidence is insufficient to certify the path that produced it.
- **Verification quality:** rejected at the hard gate. Three visible runs did not establish the required immutable test sequence.
- **Engineering grade:** **R (rejected contract)**; no A/B/C grade is assigned to a hard-rejected submission.
- **Agent discipline:** one tool error, 16 assistant messages, three test calls and 101,841 thinking characters; it produced a plausible implementation but did not earn the required test-first evidence.

### Cost and context

| Arm | Input tokens | Output tokens | Max input | Attempt | Tool errors | Compaction |
|---|---:|---:|---:|---:|---:|---:|
| GSQ | 416,838 | 25,110 | 41,275 | 742.688s | 2 | 0 |
| EXL3 | 398,039 | 40,048 | 42,834 | 1,024.859s | 1 | 0 |

These are provider/session usage aggregates for 15/17 request results, not native decode rates. No native prefill/decode counter is available for this task. Time is attempt through original hidden/visible verification; no separate retry or hidden-feedback loop was used.


Both clones were detached/no-remote copies; no original PAL file or default changed. Hidden output never reached either model. Tests ran in the pinned Docker sandbox with no network and read-only submission mounts. The first all-arm attempts that failed due runner defects are retained as infrastructure-invalid evidence; these two focused rows are the corrected runs. Benchmark ports and named test containers were released after the batch.

Machine summary: `qwen38-tuning/results/pal-journal-focus-2026-09-22/summary.json`.

This does not complete the developer's five-task Quick Discrimination Campaign. T1–T5 and the remaining reference/optional arms remain `NOT_RUN` unless the developer elects to continue them; no model or Q4-class ranking is inferred from this one task.
