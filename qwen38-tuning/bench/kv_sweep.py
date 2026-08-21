"""KV-type sweep at a fixed context depth.

At 128K the binding constraint is not the artifact, it is the cache: IQ2_XXS
holds 65/65 layers at 16K and only 47/18 at 128K, because 3,264 MiB of Q8_0 KV
is allocated from the same VRAM pool the weights live in. Every megabyte the
cache gives back becomes a layer that stops crossing PCIe on every token.

llama.cpp b10472 accepts many cache types, but only FOUR have a fast kernel in
this build -- f16, bf16, q8_0 and q4_0, all around 1,180 tok/s prompt processing,
against 144-170 for q5_1, q5_0, q4_1 and iq4_nl (see kv_kernel_screen.py). Only
the four are worth a deep run; a 128K arm on q5_1 spent 15 minutes reaching 22 %
of its window and was abandoned. That leaves q4_0 as the one type that halves the
cache again below the q8_0 this project settled on for depth. `--no-kv-offload` is the other lever: it moves the
cache to host memory entirely, trading PCIe latency on attention for weight
residency on everything else. Which of those trades wins is not predictable from
first principles -- it is measured here.

Arms are alternated and paired by round, per the 13.6 % restart-drift floor.
Depth is filled to ~80 % of the window and the cold prefill is paid once per
boot, with warm decode measured over the reused prefix -- the shape of a real
agent turn.

WARNING about quality: shrinking the KV cache is not free at depth. This project
measured q8_0 as quality-neutral against F16 at 64K and 128K *on Q4*; nothing
below q8_0 has ever been quality-checked here, on any artifact. A fast row in
this table is a throughput result and nothing else.

Usage:
    python kv_sweep.py --ctx 131072 --rounds 2
    python kv_sweep.py --ctx 131072 --arms q8_0,q4_0 --rounds 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import depth_sweep as D
from harness import median, paired_deltas

ROOT = Path(r"C:\AI\qwen38-tuning")

# label -> extra server flags
DRAFT_2B = r"C:\Users\xenod\.cache\huggingface\hub\models--empero-ai--Qwen3.8-2B-Distill-GGUF\snapshots\f4f73582d0b149595450c719b9a7521a03894f9c\Qwen3.8-2B-Q4_K_M.gguf"

DFLASH2_DRAFTER = r"C:\Users\xenod\.cache\huggingface\hub\models--z-lab--Qwen3.8-27B-DFlash2-GGUF\snapshots\57ab3265056d4024870b0621cfc2c127537020ed\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

MTP_DRAFTER = r"C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\MTP\mtp-Qwen3.8-27B-Q4_0.gguf"

ARMS = {
    "f16":      [],                                     # llama.cpp default
    "q8_0":     ["-ctk", "q8_0", "-ctv", "q8_0"],       # the settled depth choice
    "q5_1":     ["-ctk", "q5_1", "-ctv", "q5_1"],
    "q4_1":     ["-ctk", "q4_1", "-ctv", "q4_1"],
    "q4_0":     ["-ctk", "q4_0", "-ctv", "q4_0"],
    "iq4_nl":   ["-ctk", "iq4_nl", "-ctv", "iq4_nl"],
    # Not a cache type: the whole cache moves to host memory. Included because at
    # depth the cache is what evicts the weights, and this is the only flag that
    # attacks that directly.
    "q8_0-nokvoff": ["-ctk", "q8_0", "-ctv", "q8_0", "--no-kv-offload"],
    "q4_0-nokvoff": ["-ctk", "q4_0", "-ctv", "q4_0", "--no-kv-offload"],
    "f16-nokvoff":  ["--no-kv-offload"],
    # Added 2026-08-20 after report 18. Two levers that free VRAM without
    # touching the cache TYPE, so they compose with the q4_0 row above.
    #   --ctx-checkpoints defaults to 32 per slot and is pure speculative VRAM
    #   if the client never rewinds; an agent with append-only turns never does.
    #   Asymmetric KV keeps K at q8_0 (positional precision) and drops V to q4_0.
    # Both verified to parse on build 10472 before use.
    "q4_0-ckpt8":   ["-ctk", "q4_0", "-ctv", "q4_0", "--ctx-checkpoints", "8"],
    "k8v4":         ["-ctk", "q8_0", "-ctv", "q4_0"],
    "k8v4-ckpt8":   ["-ctk", "q8_0", "-ctv", "q4_0", "--ctx-checkpoints", "8"],

    # ---- Layer screen, added 2026-08-20 -----------------------------------
    # One arm per untouched lever from report 16, all stacked on the q4_0 KV
    # control so every row is comparable to it. Report 16 recorded a PREDICTION
    # for each; the point of running them is that a written prediction can be
    # proved wrong, and the developer asked for the ones predicted inert too.
    # Every flag below was verified to parse on build 10472 before being added.

    # L4 tensor placement. Attention stays on GPU, the tail blocks' FFN goes to
    # host. On an arm that is ALREADY 65+0 this can only cost; it is here to
    # measure the cost, so the same flag can be read as a gain on an arm that is
    # one or two layers short of residency.
    "ot-ffn-tail":  ["-ctk", "q4_0", "-ctv", "q4_0",
                     "-ot", r"blk\.(5[0-9]|6[0-4])\.ffn_.*=CPU"],
    # Qwen3.8 is hybrid Gated-DeltaNet: seven ssm_* tensors per block that the
    # usual attention/FFN split ignores entirely.
    "ot-ssm-tail":  ["-ctk", "q4_0", "-ctv", "q4_0",
                     "-ot", r"blk\.(5[0-9]|6[0-4])\.ssm_.*=CPU"],
    "sm-tensor":    ["-ctk", "q4_0", "-ctv", "q4_0", "-sm", "tensor"],

    # L5 memory and loading
    "loadmode-none": ["-ctk", "q4_0", "-ctv", "q4_0", "--load-mode", "none"],
    "no-host":      ["-ctk", "q4_0", "-ctv", "q4_0", "--no-host"],

    # L6 KV, depth-specific
    "swa-full":     ["-ctk", "q4_0", "-ctv", "q4_0", "--swa-full"],
    "no-kv-unified":["-ctk", "q4_0", "-ctv", "q4_0", "--no-kv-unified"],
    "cache-reuse":  ["-ctk", "q4_0", "-ctv", "q4_0", "--cache-reuse", "256"],

    # L7 context geometry
    "ctx-shift":    ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--context-shift", "--keep", "2048"],

    # L8 kernels. -fa has been left at the build default on every run this
    # project has ever done, which is a validity problem before it is an
    # optimization: nothing has checked that it resolves the same way per arm.
    "fa-off":       ["-ctk", "q4_0", "-ctv", "q4_0", "-fa", "off"],
    "no-repack":    ["-ctk", "q4_0", "-ctv", "q4_0", "--no-repack"],
    "no-op-offload":["-ctk", "q4_0", "-ctv", "q4_0", "--no-op-offload"],

    # L9 decoders that need no drafter file and no download
    "ngram-simple": ["-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "ngram-simple"],
    "ngram-mod":    ["-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "ngram-mod"],
    "ngram-map-k":  ["-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "ngram-map-k"],
    "ngram-map-k4v":["-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "ngram-map-k4v"],
    "ngram-cache":  ["-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "ngram-cache"],

    # L10 slots -- predicted inert at one stream, and each slot costs KV
    "np2":          ["-ctk", "q4_0", "-ctv", "q4_0", "-np", "2"],

    # L11 CPU. Logical processors 0-11 are the six P-cores on this i5-13500,
    # measured via PercentProcessorPerformance (144-179 vs 99-139 for 12-19).
    "pcore-mask":   ["-ctk", "q4_0", "-ctv", "q4_0", "--cpu-mask", "0x0FFF"],
    "prio-high":    ["-ctk", "q4_0", "-ctv", "q4_0", "--prio", "2"],
    "poll-0":       ["-ctk", "q4_0", "-ctv", "q4_0", "--poll", "0"],

    # L12 sampling offloaded to the GPU (experimental in this build)
    "backend-samp": ["-ctk", "q4_0", "-ctv", "q4_0", "--backend-sampling"],

    # ---- L9, the standalone MTP drafter and where to put it ---------------
    # V3 removed the built-in head from IQ2_XXS and smaller (verified from the
    # tensor list: v3-q2kxl loads blk.64.attn_*, v3-iq2xxs and v3-iq1s do not),
    # and ships a 1.28 GiB drafter separately. This project measured MTP at
    # -8.8 % on a resident target -- but that was with the head ON THE GPU,
    # where its VRAM moved the split from 61+4 to 55+10. These three arms are
    # the same drafter in three places, which is the comparison that verdict
    # was never given.
    "mtp-gpu":      ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
                     "-md", MTP_DRAFTER],
    "mtp-cpu":      ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
                     "-md", MTP_DRAFTER, "--spec-draft-device", "none"],
    "mtp-otd-cpu":  ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
                     "-md", MTP_DRAFTER, "-otd", ".*=CPU"],

    # ---- L9 continued: the decoders that were never even attempted ---------
    # draft-simple needs a separate small model with a MATCHING tokenizer.
    # Qwen3.8-2B-Distill is the only same-family sibling on the Hub; if the
    # vocab does not match, llama-server refuses at load and that is the answer.
    "draft-simple": ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "draft-simple", "--spec-draft-n-max", "4",
                     "-md", DRAFT_2B],
    "draft-simple-cpu": ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "draft-simple", "--spec-draft-n-max", "4",
                     "-md", DRAFT_2B, "--spec-draft-device", "none"],
    # DFlash 2. The vendor says this needs unmerged PR #27342, and the stock
    # loader may reject the architecture -- ONE boot settles it instead of
    # committing to a source build on an assumption.
    "dflash2":      ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "draft-dflash", "--spec-draft-n-max", "7",
                     "-md", DFLASH2_DRAFTER],

    # ---- L9: the n-gram tuning knobs, all still at their defaults ----------
    # ngram-simple scored 31 % acceptance at defaults with a cold table. The
    # lookup and draft lengths are the two knobs that decide whether a match is
    # found at all, and neither has ever been moved.
    "ngram-simple-wide": ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "ngram-simple",
                     "--spec-ngram-simple-size-n", "6",
                     "--spec-ngram-simple-size-m", "24"],
    "ngram-mod-short":   ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "ngram-mod",
                     "--spec-ngram-mod-n-match", "12",
                     "--spec-ngram-mod-n-min", "16",
                     "--spec-ngram-mod-n-max", "32"],
    "ngram-mapk4v-wide": ["-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "ngram-map-k4v",
                     "--spec-ngram-map-k4v-size-n", "6",
                     "--spec-ngram-map-k4v-size-m", "24"],

    # ---- L4 graded: how LITTLE has to move to buy one layer back? ----------
    # ot-ffn-tail moved the FFN of ten blocks and freed 1,234 MiB at a cost of
    # 61 % decode -- far more VRAM than needed and far more speed than anyone
    # would pay. AD-IQ1_M is ONE layer short of 65+0 at 131,072 and needs about
    # 125 MiB. These slice the same lever finely enough to find the smallest cut
    # that clears the bar.
    "ot-ffn-1":     ["-ctk", "q4_0", "-ctv", "q4_0", "-ot", r"blk\.64\.ffn_.*=CPU"],
    "ot-ffn-2":     ["-ctk", "q4_0", "-ctv", "q4_0", "-ot", r"blk\.6[34]\.ffn_.*=CPU"],
    "ot-ffn-4":     ["-ctk", "q4_0", "-ctv", "q4_0", "-ot", r"blk\.6[1-4]\.ffn_.*=CPU"],
    # ssm_* freed 168 MiB for only 19 % at ten blocks -- a better rate per MiB
    # than FFN, and 168 already clears the 125 MiB gap on its own.
    "ot-ssm-4":     ["-ctk", "q4_0", "-ctv", "q4_0", "-ot", r"blk\.6[1-4]\.ssm_.*=CPU"],
    "ot-ssm-10":    ["-ctk", "q4_0", "-ctv", "q4_0", "-ot", r"blk\.(5[5-9]|6[0-4])\.ssm_.*=CPU"],

    # Combined arms for the >128K push, added 2026-08-21. The goal is the
    # deepest window this card holds AND the highest tok/s in it, and those two
    # are the same question: a CPU layer at depth costs more than four times the
    # decode rate (AD-IQ1_M, 65+1 at 131,072, 6.08 tok/s). So the ssm slice buys
    # residency and the n-gram arm buys throughput inside it -- neither alone
    # answers the goal.
    #
    # ssm, not ffn. The ffn slice moved 644 MiB and dropped prefill from 240.6
    # to 8.56 tok/s; ssm moves ~168 MiB and report 20 measured no throughput
    # cost. It also CHANGES THE GREEDY HASH -- CPU and GPU floats differ -- so
    # these arms are compared on speed only, never certified output-identical.
    # VRAM the goal actually needs, freed WITHOUT moving weights, added
    # 2026-08-21. D1 showed that at 163,840 the artifact worth having sits at
    # 62+3 and needs ~576 MiB to get back to 65+0 -- and that the -ot route to
    # those MiB destroys speculative acceptance (100 % -> 4 %). So the question
    # is what else can pay for them.
    #
    # --fit-target is a RESERVE, not an allocation: the harness has passed 768
    # since the first sweep and nothing has ever tested whether that number is
    # right. If 384 is enough headroom, half the shortfall is free.
    #
    # -b / -ub size the compute buffers. Smaller batches mean less scratch VRAM
    # and slower prefill -- a trade the goal can price, unlike the -ot trade
    # which turned out not to be a trade at all.
    #
    # All carry ngram-mod, because the arm being compared against is the one
    # that would ship, and D1 proved that adding speculation afterwards can
    # reverse a verdict reached without it.
    "fit-384-ngram":  ["-ctk", "q4_0", "-ctv", "q4_0", "--fit-target", "384",
                       "--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "12",
                       "--spec-ngram-mod-n-min", "16",
                       "--spec-ngram-mod-n-max", "32"],
    "fit-192-ngram":  ["-ctk", "q4_0", "-ctv", "q4_0", "--fit-target", "192",
                       "--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "12",
                       "--spec-ngram-mod-n-min", "16",
                       "--spec-ngram-mod-n-max", "32"],
    "ub128-ngram":    ["-ctk", "q4_0", "-ctv", "q4_0", "-ub", "128",
                       "--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "12",
                       "--spec-ngram-mod-n-min", "16",
                       "--spec-ngram-mod-n-max", "32"],
    "b1024ub128-ngram": ["-ctk", "q4_0", "-ctv", "q4_0", "-b", "1024", "-ub", "128",
                       "--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "12",
                       "--spec-ngram-mod-n-min", "16",
                       "--spec-ngram-mod-n-max", "32"],
    "fit192-ub128-ngram": ["-ctk", "q4_0", "-ctv", "q4_0",
                       "--fit-target", "192", "-ub", "128",
                       "--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "12",
                       "--spec-ngram-mod-n-min", "16",
                       "--spec-ngram-mod-n-max", "32"],
    "ot-ssm-4-ngram":  ["-ctk", "q4_0", "-ctv", "q4_0",
                        "-ot", r"blk\.6[1-4]\.ssm_.*=CPU",
                        "--spec-type", "ngram-mod",
                        "--spec-ngram-mod-n-match", "12",
                        "--spec-ngram-mod-n-min", "16",
                        "--spec-ngram-mod-n-max", "32"],
    "ot-ssm-10-ngram": ["-ctk", "q4_0", "-ctv", "q4_0",
                        "-ot", r"blk\.(5[5-9]|6[0-4])\.ssm_.*=CPU",
                        "--spec-type", "ngram-mod",
                        "--spec-ngram-mod-n-match", "12",
                        "--spec-ngram-mod-n-min", "16",
                        "--spec-ngram-mod-n-max", "32"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="q8_0,q5_1,q4_0,q8_0-nokvoff")
    ap.add_argument("--quant", default="iq2xxs")
    ap.add_argument("--out", default="kv-sweep.jsonl")
    ap.add_argument("--filler", choices=("high","low"), default="high",
                    help="repetition of the timed prompt. high = the historic "
                         "filler (84.5 %% duplicate lines, the best case an "
                         "n-gram drafter can get); low = varied blocks (73 %%)")
    ap.add_argument("--n-predict", type=int, default=160,
                    help="tokens per timed generation. 160 is the historic "
                         "default; longer runs test whether speculative "
                         "decoders were measured before they warmed up")
    ap.add_argument("--fixed-text", action="store_true",
                    help="pin temperature 0 and a fixed seed for the TIMED "
                         "generations, so every round writes the same text. "
                         "Required for content-dependent arms such as ngram-*; "
                         "results are comparable only to other --fixed-text runs.")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARMS:
            sys.exit("unknown arm %r; known: %s" % (a, ", ".join(ARMS)))

    D.MODEL = D.QUANTS[args.quant]
    D.USE_MTP = False          # a loss on a resident target; report 10 s1
    D.FIXED_TEXT = args.fixed_text
    D.N_PREDICT = args.n_predict
    D.FILLER = args.filler
    D.OUT = ROOT / "results" / args.out
    print("model: %s   ctx: %d   speculation: OFF" % (D.MODEL, args.ctx), flush=True)

    per_round = dict((a, []) for a in arms)

    for rnd in range(1, args.rounds + 1):
        order = arms if rnd % 2 else list(reversed(arms))
        print("\n===== round %d : %s =====" % (rnd, " -> ".join(order)), flush=True)
        for tag in order:
            row = D.run(args.ctx, ARMS[tag], "KV %s" % tag,
                        "%s-kv-%s-r%d-%d" % (args.quant, tag, rnd, args.ctx))
            tg = row.get("tg_med")
            if row.get("loaded") and tg:
                per_round[tag].append(tg)

    D.kill()
    print("\n===== PAIRED RESULT (baseline = %s) =====" % arms[0])
    base = per_round[arms[0]]
    out = ROOT / "results" / args.out
    for tag in arms[1:]:
        if not base or len(per_round[tag]) != len(base):
            print("  %-14s unpaired (%d vs %d rounds) -- not comparable"
                  % (tag, len(base), len(per_round[tag])))
            continue
        d = paired_deltas(base, per_round[tag])
        verdict = "RESOLVED" if d["resolved"] else "under drift floor / inconsistent sign"
        print("  %-14s per-round %s  mean %+.2f%%  range %+.2f..%+.2f%%  -> %s"
              % (tag, d["per_round_pct"], d["mean_pct"], d["min_pct"],
                 d["max_pct"], verdict))
        rec = dict(kind="PAIRED", ctx=args.ctx, quant=args.quant,
                   baseline=arms[0], arm=tag, baseline_rounds=base,
                   candidate_rounds=per_round[tag])
        rec.update(d)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
