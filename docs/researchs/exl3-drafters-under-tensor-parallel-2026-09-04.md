# Using a drafter other than the MTP head on a tensor-parallel EXL3 target (2026-09-04)

> **External material, read from the fork's source.** Nothing here is measured;
> every `file:line` is `C:\AI\exllamav3-mia` @ 63b32f0. Written because pass 7
> of issue #71 died at load: **`DFlash2Model does not yet support
> tensor-parallel targets`** (`architecture/dflash2.py:162-165`), and the recipe
> that reaches 31.9–36.9 tok/s at 144K exists only under `-tp`.

## 1. What the fork can and cannot do today

| drafter | flag | on a `-tp` target | why |
|---|---|---|---|
| **MTP head** (in the checkpoint) | `-mtp` | **yes** — the recipe | the head is a trunk module; the fork loads it in the same TP plan |
| **n-gram** (model-free) | `-ngram N` | **yes** — nothing runs on the GPU | `generator.py:171-176`; **but `generator.py:159` forbids combining it with any draft model**, so it *replaces* MTP rather than stacking on it the way llama.cpp's `draft-mtp,ngram-mod` does |
| **DFlash (v1)** | `-dm` | code path exists: `dflash.py:275-284` dispatches `tp_dispatch_lm_head_argmax` when the target is TP | v1 only needs the **argmax** of the target head over the block, which the TP backend offers as a collective (`model_tp.py:316`) — **no v1 drafter exists for Qwen3.8-27B on the Hub**, every published one is DFlash2 |
| **DFlash2** (`Mia-AiLab/…-DFlash2-EXL3-5.0bpw`, on disk) | `-dm` | **no** — raises at first draft | its selector walk needs **full logits** for every block row to keep top candidates per position (`dflash2.py:157-175`); under TP the head is sharded across ranks and only the argmax collective exists |
| **DSpark** (`RadixArk/Qwen3.8-27B-DSpark`, bf16) | `-dm` | **no** — same raise (`dspark.py:227-232`) | Markov bigram head + confidence head consume full logits; the fork's own `# TODO: TP target support (private embed/head copies as in DeepseekV4MTPModel._load_own_embed_head)` names the fix |
| **DSpark/DFlash under TP for DeepSeek-V4** | — | **yes** | `deepseek_v4_mtp.py:229-264, 316-317` — the pattern below, already shipped for one architecture |

**Why the taps are not the problem.** All three drafters read the target's
residual stream at tap layers (`export_state_layers`, `dflash.py:192`; the
blocks append to `params["export_states"]` in `modules/transformer.py:253`).
Under native TP the output device's rank is a pseudo-worker **in the parent
process** (`model_tp.py:569-571`), and after each layer's all-reduce every
rank holds the full residual, so the parent sees complete tap states without
any read-back. The only thing a TP target cannot hand the drafter is the
**lm_head**, which lives sharded in the worker processes.

## 2. The fix the fork already contains for another model

`DeepseekV4MTPModel._load_own_embed_head` (`deepseek_v4_mtp.py:235-264`):

1. on `attach_to(target)`, if `target.loaded_tp`, load a **private copy of the
   trunk's embedding** on CPU (`prefer_cpu`, needed only for the drafter's
   input layer) and a **private copy of the lm head** on the drafter's own
   device (`head.load(self.modules[self.fwd_end_idx - 1].device)`);
2. in `sample_from_state`, use `self.own_head` when it exists
   (`deepseek_v4_mtp.py:316-317`), otherwise borrow the target's module as
   before.

Porting that to `DFlash2Model` is the same two steps plus removing the raise at
`dflash2.py:162`; for `DSparkDraftModel` it is exactly the TODO at
`dspark.py:227`. Estimated size: 30–40 lines each, no kernel work — the head is
the target's existing quantised tensor (`head_bits` 6 in this quant, model card)
loaded a second time.

**Cost of the private head.** Qwen3.8-27B's head is 151,936 × 5,120: fp16
1.56 GB, at the quant's 6-bit head ≈ 0.6 GB, on whichever card the drafter's
last module sits. The drafter itself is 1.4 GB (DFlash2 EXL3) or 3.7 GB (DSpark
bf16) plus its fp16 KV (parity-pinned, `model_init.py:265-266`). Against the
recipe's footprint (4070 at 5.7 GB, 5060 Ti with 2.5 GB KV + its shard) both
fit; the DSpark bf16 case is the tighter one.

**Cost in the decode loop.** The drafter's head runs once per draft block on
one card, outside the TP collectives — which is the reason the DSv4 code gives
for the private copy ("keeps the sequential markov/argmax draft loop free of
collectives", `deepseek_v4_mtp.py:237-238`). It does not add a synchronisation
point.

## 3. What is worth measuring, and in what order

1. **Pass 7 (done 2026-09-04 07:56, results 10):** DFlash2 vs MTP under
   **layer split** at 30K, two rotated pairs: **−9 % and −5 %** for DFlash2,
   700–900 rejected drafts per 512 tokens against MTP's ~200. The one-card
   arm does not fit; the 144K arm is VOID (host RAM paging, 22 GB commit per
   harness). **So the TP port is not worth writing for DFlash2.** The
   mechanism below stays valid for DSpark if its llama.cpp twin turns out
   to accept well on this target.
2. **Not reached for DFlash2 (step 1 said no).** The port, if ever wanted: port `_load_own_embed_head` into `DFlash2Model` (local
   patch, kept in `exllamav3-mia`, recorded like the DSA guard in
   `exllama3-platform-2026-09-03.md`), rebuild nothing (Python only), re-run the
   recipe with `-dm` instead of `-mtp`, paired and rotated against the recipe.
3. **DSpark on EXL3** needs the same port plus the bf16 checkpoint (3.7 GB) —
   or an EXL3 quant of it, which nobody has published. Its llama.cpp twin is
   queued already (`nvfp4-dspark` arm set), and that is the cheaper way to
   learn whether DSpark v2's acceptance survives NVFP4 targets at depth.
4. **`-ngram` on EXL3** is free to try but is a *replacement* for MTP, so it can
   only win if n-gram alone beats the MTP head — on llama.cpp at this depth
   `ngram-mod` alone is far below `draft-mtp,ngram-mod` (results 02), so it is
   last.

## 4. Not options

- Upstream exllamav3 1.4.6: no DFlash2, no DSpark, no NVFP4 cache
  (`exllama3-techniques-vs-llamacpp-2026-09-04.md` §1).
- NCCL backend (`-tpb nccl`): changes the collective, not the sharded head; the
  drafters raise on `loaded_tp`, which is true for both backends.
- A single-card target at 144K: OOMs in prefill above ~65K (results 10, arm "—").
