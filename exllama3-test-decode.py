"""EXL3 decode benchmark for the Mia-AiLab exllamav3 fork (issue #71).

Loads exactly the way the fork's own tools/serve_openai.py does -- through
model_init.add_args/init with an argv -- so the number measures the runtime the
model was exported for, not a hand-rolled load path. One JSON line per round to
--out, in the spirit of qwen38-tuning/results/*.jsonl: every number names the
argv, versions and card it came from.

    python exllama3-test-decode.py -m C:\AI\models\Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw ^
        --mtp --cache-size 16384 --prompt-tokens 512 --gen 256 --rounds 3 --gpu-split 0,15.5

--gpu-split follows the fork's -gs: max VRAM per device in GB, in device order;
"0,15.5" keeps the 4070 SUPER empty and puts everything on the 5060 Ti.
"""
import os, sys, time, json, argparse, platform
os.add_dll_directory(r"C:\AI\exllama3-venv\Lib\site-packages\torch\lib")
sys.path.insert(0, "C:/AI/qwen38-tuning/bench")
import harness


def vram_gb():
    return [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]


def build_prompt(tokenizer, n_tokens):
    """A prompt of about n_tokens tokens: prose, not a repeated token, so the
    draft head sees text of normal shape."""
    seed = ("The prefix cache stores the transformer state for every token already processed, "
            "so that a request sharing the same beginning does not recompute it. ")
    text = ""
    while True:
        ids = tokenizer.encode(text, add_bos=True)
        if ids.shape[-1] >= n_tokens:
            return ids[:, :n_tokens]
        text += seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=r"C:\AI\models\Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw")
    ap.add_argument("--mtp", action="store_true", help="use the checkpoint's MTP head as drafter")
    ap.add_argument("--cache-size", type=int, default=16384)
    ap.add_argument("--cache-quant", default=None, help="fork -cq: e.g. nvfp4, 8, or k,v bits")
    ap.add_argument("--gpu-split", default=None, help="fork -gs: GB per device, e.g. 0,15.5")
    ap.add_argument("--tp", action="store_true", help="fork -tp -tpb native: tensor-parallel across the split (no NCCL)")
    ap.add_argument("--ndt", type=int, default=None, help="fork -ndt: number of draft tokens (MTP default 4)")
    ap.add_argument("--dds", action="store_true", help="fork -dds: dynamic draft length")
    ap.add_argument("--chunk", type=int, default=2048, help="Generator max_chunk_size for prefill (default 2048)")
    ap.add_argument("--extra", default="", help="raw fork argv appended verbatim, e.g. \"-tp_linear_attn 1 -tp_attn 1\"")
    ap.add_argument("--prompt-tokens", type=int, default=512, help="synthetic prose prompt length (ignored with --regime)")
    ap.add_argument("--regime", default=None, help="arena corpus regime, e.g. real-code-vendor: prompt = corpus[:ctx*3 chars] + the arena's task line, greedy, exactly as bench/dflash2_arena.py filler() builds it")
    ap.add_argument("--ctx", type=int, default=16384, help="with --regime: the ctx the arena would report (prompt chars = ctx*3)")
    ap.add_argument("--gen", type=int, default=512, help="arena N_PREDICT is 512")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=r"C:\AI\qwen38-tuning\results\exl3-decode.jsonl")
    a = ap.parse_args()

    global torch
    import torch
    from exllamav3 import Generator, Job, model_init
    import exllamav3
    from exllamav3.version import __version__ as exl_version

    argv = ["-m", a.model, "-cs", str(a.cache_size)]
    if a.mtp:
        argv += ["-mtp"]
    if a.cache_quant:
        argv += ["-cq", a.cache_quant]
    if a.gpu_split:
        argv += ["-gs", a.gpu_split]
    if a.tp:
        argv += ["-tp", "-tpb", "native"]
    if a.ndt is not None:
        argv += ["-ndt", str(a.ndt)]
    if a.dds:
        argv += ["-dds"]
    if a.extra:
        argv += a.extra.split()
    # arena SAMPLER = {"temperature": 0.0, "top_k": 1, "seed": 42}: greedy
    argv += ["-temp", "0.0", "-topk", "1"]
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, add_draft_model_args=True, add_sampling_args=True)
    args = parser.parse_args(argv)

    t0 = time.time()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    t_load = time.time() - t0
    gen = Generator(model, cache, tokenizer, draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=args.num_draft_tokens, dynamic_draft_tokens=args.dynamic_draft,
                    max_chunk_size=a.chunk)
    sampler = model_init.get_arg_sampler(args)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"[load] {t_load:.1f}s  argv={argv}  devices={devices} alloc GB={vram_gb()}", flush=True)

    prompt_text = None
    if a.regime:
        import dflash2_arena
        prompt_text = dflash2_arena.filler(a.ctx, a.regime)
        ids = tokenizer.encode(prompt_text, add_bos=True)
    else:
        ids = build_prompt(tokenizer, a.prompt_tokens)
    n_prompt = int(ids.shape[-1])
    recorded_argv = argv + ["--chunk", str(a.chunk)]  # --chunk is ours, not a fork flag
    meta = dict(argv=recorded_argv, exllamav3=exl_version, torch=torch.__version__,
                cuda=torch.version.cuda, devices=devices, prompt_tokens=n_prompt,
                sampler=type(sampler).__name__, host=platform.node(), regime=a.regime, ctx=a.ctx if a.regime else None,
                corpus_hash=(dflash2_arena.corpus_hash(a.regime) if a.regime else None),
                env={k: v for k, v in os.environ.items() if k.startswith("EXL3_")})
    for r in range(a.rounds):
        job = Job(input_ids=ids, max_new_tokens=a.gen, sampler=sampler, stop_conditions=[])
        gen.enqueue(job)
        wall0 = time.time()
        final = None
        while gen.num_remaining_jobs():
            for res in gen.iterate():
                if res.get("eos"):
                    final = res
        wall = time.time() - wall0
        tp, tg = final["time_prefill"], final["time_generate"]
        new = final["new_tokens"]
        dec_s, timing_source = harness.exl3_decode_seconds(final, wall)
        row = dict(meta, round=r, new_tokens=new, time_prefill_s=round(tp, 3), time_generate_s=round(tg, 3),
                   wall_s=round(wall, 3), prefill_tok_s=round(n_prompt / max(tp, 1e-9), 1),
                   decode_tok_s=round(new / dec_s, 2),
                   accepted_draft=final.get("accepted_draft_tokens"), rejected_draft=final.get("rejected_draft_tokens"),
                   eos_reason=final.get("eos_reason"), timing_source=timing_source,
                   copied_frac=(round(harness.copied_window_fraction(final.get("full_completion", ""), prompt_text), 3) if prompt_text else None),
                   vram_alloc_gb=vram_gb(),
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        print(f"[round {r}] prefill {row['prefill_tok_s']} tok/s ({n_prompt} tok in {tp:.2f}s) | "
              f"decode {row['decode_tok_s']} tok/s ({new} tok in {dec_s:.2f}s) | draft acc/rej "
              f"{row['accepted_draft']}/{row['rejected_draft']} | copied {row['copied_frac']} | vram {row['vram_alloc_gb']}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    print(f"[done] rows appended to {a.out}")


if __name__ == "__main__":
    main()
