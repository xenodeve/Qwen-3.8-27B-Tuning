<#
PRODUCTION PROFILE -- Qwen3.8-27B UD-Q4_K_XL with built-in MTP speculative decoding.

Every flag below is here because a measurement put it here. Nothing is aspirational.

  --spec-type draft-mtp     the single largest speed lever measured on this box:
                            +30% on a synthetic prompt, +47% on code rewriting,
                            at 78-98% draft acceptance, with byte-identical greedy
                            output versus no speculation. Costs ~286 MB (the
                            blk.64 nextn tensors), not the ~2.5 GB claimed online.

  --spec-draft-n-max 2      measured peak. n=3 has a higher ceiling but a lower
                            floor and a wider spread; n>=5 regresses because --fit
                            reserves more for the draft path and evicts target
                            layers from the GPU.

  -ngl auto --fit on        the model does not fit in 12 GB; auto-fit places what
                            it can. NOTE: free VRAM on this desktop varies roughly
                            9.4-11.1 GiB depending on what is running, so the split
                            is not identical between boots. For repeatable
                            benchmarking, snapshot the environment first
                            (scripts\collect-env.ps1).

  -fa on                    explicit rather than 'auto', so the profile is pinned.

  -np 1                     one interactive worker; server slots cost memory that
                            this workload never uses.

  --no-mmproj-auto          the model is multimodal; the coding worker is text-only.
                            Saves the projector's memory and isolates text behaviour.

  --jinja                   omitted deliberately -- it is already the default in
                            b10472, so passing it is a no-op.

Sampling is NOT set here. It is per-request, because the vendor publishes two
different profiles and a server default cannot serve both:

    thinking      temperature 1.0  top_p 0.95  top_k 20  min_p 0.0  presence 0.0
    non-thinking  temperature 0.7  top_p 0.80  top_k 20  min_p 0.0  presence 1.5

The server's own default min_p is 0.05, which is wrong for both. Clients must send
min_p explicitly.

Reasoning effort defaults to 'xhigh' inside the chat template when the caller omits
it. Send chat_template_kwargs.reasoning_effort explicitly.
#>
param(
  [int]$Ctx  = 16384,
  [int]$Port = 8080
)

$ErrorActionPreference = 'Continue'   # llama-server logs to stderr on success

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL `
    --alias qwen38-q4 `
    -c $Ctx `
    -ngl auto `
    --fit on `
    -fa on `
    -np 1 `
    --no-mmproj-auto `
    --spec-type draft-mtp `
    --spec-draft-n-max 2 `
    --host 127.0.0.1 `
    --port $Port
