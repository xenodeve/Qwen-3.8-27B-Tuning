<#
WORKER PROFILE — UD-Q4_K_XL across BOTH cards, ctx 147,456.
RTX 4070 SUPER 12 GB + RTX 5060 Ti 16 GB.

Issue #52. Every number in this header was measured on this machine on
2026-08-26 with the native sm_120a+sm_89 build, not projected.

WHY THIS PROFILE EXISTS AT ALL

  UD-Q4_K_XL is 16.69 GiB. It was refused on one 16 GB card at every depth
  since that card arrived -- `docs/results/09-hardware.md` recorded it as
  "16 GB does not unlock Q4 residency either". Across two cards it is FULLY
  RESIDENT (66+0) at every rung to 229,376, including the 147,456 this file
  serves, and spills a single layer only at 262,144, which is n_ctx_train.

  The second card is worth +79.9 % [+77.3, +82.2] to this artifact, and the
  layer split says why: 55+11 becomes 66+0. (That figure is the DEFAULT split;
  with -sm tensor the gap is larger still -- see below.) That is the residency cliff, not
  the silicon -- UD-Q2_K_XL, which was already resident on one card, gained
  1.5 % from the identical change.

WHY IT IS A SEPARATE FILE FROM worker-q2kxl-mtp.ps1

  Every row in docs/results/ from 2026-08-23 onward describes the one-card
  configuration. A -Dual switch on that profile would mean its defaults no
  longer say what was measured. Both ship; which is the default is the
  developer's call with the numbers in front of them (#52).

WHAT THIS COSTS, STATED PLAINLY

  DECODE: ESSENTIALLY NOTHING, once -sm tensor is set. 32.4 / 33.9 / 32.3 tok/s
  here against 32.1 / 32.0 / 32.0 for UD-Q2_K_XL on one card -- speculation off,
  ctx 16,384, and the ranges overlap.

  READ THE HISTORY OF THAT SENTENCE BEFORE TRUSTING IT. Earlier the same day
  this header said "about a third of raw decode: 20.9 against 32.0", and that
  was honestly measured -- on the DEFAULT layer split, before -sm tensor was
  tried. One flag moved a 34 % penalty to parity. A cost figure taken before the
  configuration was optimised is a fact about the configuration, not about the
  artifact, and this one was two hours old when its own project contradicted it.

  The comparison is still ACROSS SWEEPS and so across boots. It rests on the
  under-0.8 % per-arm floor measured that day, at that depth only, and the two
  arms load different files so nothing else about them is paired. A sizing
  figure, not a verdict.

  POWER: roughly 130 W more under load. Both cards sat at ~50 % utilisation
  drawing 107-114 W and 133-135 W.

  QUALITY IS THE WHOLE REASON TO RUN THIS AND IT HAS NEVER BEEN MEASURED HERE
  on this project's own artifacts. The bits-per-weight ladder and an external
  12-format campaign both point the same way; neither is our number. With the
  decode cost now near zero, quality is no longer a trade-off to justify -- it
  is simply the last unmeasured thing.

WHY -sm tensor, AND IT IS MARKED EXPERIMENTAL

  MEASURED 2026-08-26, ctx 16,384, three paired rounds, arms rotated, no
  speculation:

      layer (llama.cpp default)   [21.1, 21.0, 19.9] tok/s
      -sm tensor                  [32.4, 33.9, 32.3]  +59.5 % [+53.9, +62.9]
      -ts 1,1                     [21.2, 21.9, 20.0]  +1.8 %, within noise

  Same residency ceiling either way: 66+0 to 229,376. The default leaves 59 %
  on the table for nothing, and the tensor-split RATIO is not a lever here --
  `-ts 1,1` against the free-VRAM default of 41:59 changed nothing that clears
  the floor.

  llama.cpp's own help calls this mode EXPERIMENTAL: "split weights and KV
  across GPUs (parallelized, EXPERIMENTAL)". It is shipped here on a measured
  +59.5 % with that status stated rather than hidden. It also fails harder at
  the ceiling: at 262,144 `layer` spills one layer and `tensor` FAILS TO LOAD.

  It aggregates the two cards into a virtual device -- the boot log says
  "creating a Meta device for tensor parallelism from 2 devices ... 26241 MiB
  free" and assigns every layer to `Meta()`. `parse_layer_split` had to be
  taught that token; before that it voided every tensor row, which is the right
  failure and is the only reason this result was found rather than averaged
  into nothing.

  `-sm row` CANNOT LOAD on this pair: "device CUDA0 does not support split
  buffers", at model load, in about a second, every attempt. The cards sit at
  PXB with no NVLink.

WHY -ub 1024 AND NOT THE 256 THE SINGLE-CARD PROFILE SERVES

  MEASURED 2026-08-26, ctx 16,384, three paired rounds on -sm tensor.

  DECODE does not care. 256 / 512 / 1024 measured [34.3, 35.0, 35.0],
  [34.7, 34.7, 33.7] and [34.6, 34.5, 34.5] -- -1.1 % and -0.6 %, both inside
  the floor. Expected: a micro-batch is a prefill knob.

  PREFILL is a clean staircase, on the identical 6,621-token prompt:

      -ub 128    820.4 / 822.9              tok/s
      -ub 256    870.9 / 892.3 / 884.4      (the single-card default)
      -ub 512    920.5 / 937.1 / 956.9
      -ub 1024   973.0 / 968.9 / 972.5      +10.1 %, ranges do not overlap

  256 was chosen against ONE card (results/05-runtime-flags.md). Two cards
  change the arithmetic twice: -sm tensor moves activations between the cards
  inside every layer rather than once per boundary, and the link carrying that
  traffic is gen4 x4 on the 5060 Ti -- a quarter of the other card's width
  (CORRECTIONS 31). A wider micro-batch amortises each transfer over more
  tokens, which is the shape of a narrow link.

  It costs about 180 MiB of compute buffer. Residency at 147,456 was confirmed
  with this value set, not assumed from the 16,384 rows.

  -b stays at 2048. -ub above -b is silently clamped, so moving both together
  would make some arms identical to their neighbours with nothing saying which.

WHY ngram-mod AND WHY draft-mtp IS NOT AN OPTION HERE

  MEASURED on this exact configuration at ctx 147,456, three paired rounds:

      ngram-mod            [32.4, 32.6, 33.1] tok/s   own spread 2.1 %
      none                 [28.1, 28.1, 28.7]         -13.3 % [-13.8, -13.1]
      draft-mtp,ngram-mod  CANNOT LOAD

  draft-mtp aborts inside the aggregating backend on every attempt:

      ggml-backend-meta.cpp:1522: GGML_ASSERT(bufs.back() != nullptr) failed

  So the decoder question issue #44 holds open for the single-card profile does
  not arise here: on this split there is one speculative decoder that works.

  The -13.3 % was first reported as "within noise" because NOISE_FLOOR_PCT is
  13.6 -- an Ada figure from ctx 16,384 -- while this run's own arms spread
  2.1 %. The constant was not changed; the arena now prints the floor it
  applied beside each arm's observed spread, and names that third state.

WHY THE SPLIT IS COMPUTED AT LAUNCH -- THE 0.38 tok/s INCIDENT

  2026-08-26. `serve-dual-lan.bat` decoded at 0.38 tok/s. 85x slower than the
  number at the top of this file. The 5060 Ti sat at 0 % utilisation and 45 C
  while the 4070 SUPER ran at 88 %, holding 11.6 of its 12.0 GB with 0.7 GB
  spilled into SHARED (host) memory. Prefill collapsed too -- 16.4 tok/s on a
  330-token prompt against the 973 this file was tuned to.

  ROOT CAUSE, from llama.cpp's own source. `-sm tensor` splits EVENLY when no
  ratio is given: `llama-model.cpp:707` falls back to `ne_s * (j+1)/n_devices`.

  These two cards are not even. 12 GB against 16 GB -- and THE 12 GB CARD IS
  THE DISPLAY GPU. explorer.exe, Windows Terminal, the browser and the NVIDIA
  overlay all live on it; measured idle, they hold about 1,600 MiB.

  The arithmetic straight off that boot log. The Meta buffers are PER CARD:
  8,065 model + 1,296 KV + 1,024 compute = 10,385 MiB each.

      RTX 4070 SUPER   12,282 total - 1,579 desktop - 10,385 =   +317 MiB
      RTX 5060 Ti      16,311 total -    49 desktop - 10,385 = +5,876 MiB

  317 MiB is not headroom. One browser tab put it over, the driver paged to
  host memory, and everything went through PCIe.

  WHY --fit DID NOT SAVE IT, AND WHY THIS FILE USED TO BE WRONG ABOUT THAT:

      W common_fit_params: failed to fit params to free device memory:
        llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort

  `--fit on --fit-target 768` does NOTHING under -sm tensor. This header
  previously said that made an over-large context "a hard load failure ... the
  better failure of the two". THAT WAS WRONG, and it was measured wrong on
  2026-08-26: it is a SILENT SPILL that returns a working server at 0.38 tok/s.
  That is exactly the believable-wrong-number failure CLAUDE.md's north star
  names, and this profile shipped it.

  THE FIX IS NOT A HARDCODED RATIO. The desktop's appetite is not constant, so
  a ratio tuned today is a gamble tomorrow. The split is computed at launch
  from what nvidia-smi says is actually free, minus a reserve on whichever card
  already holds memory -- that card is the display GPU and it will want more.

  Proportional-to-budget is what makes it safe: each card gets a share of the
  model in proportion to what it can afford, so they run out together instead
  of one spilling while the other has 6 GB idle.

  MEASURED after the fix, same machine, desktop running:

      -ts (even, the default)   0.38 tok/s   4070 at +317 MiB
      -ts 2,3                   31-33 tok/s  4070 at 1,511 MiB free
      -ts 1,2                   28-30 tok/s  4070 at 2,792 MiB free

  The computed ratio lands near 1:2 on this machine and buys the headroom.

  The --fit flags stay: every measured row carries them, and an argv that
  differs from the benchmarked one is not the benchmarked configuration.

-Mtp: draft-mtp RUNS HERE, AND ITS RATE COULD NOT BE MEASURED

  This file previously said no external drafter loads under -sm tensor. That
  was too strong, and an outside review is what prompted the probe that showed
  it. Measured 2026-08-27 at ctx 16,384 with -ub 128, where memory pressure is
  as low as this configuration goes:

      -sm tensor + ngram-mod  (control)          LOADED
      -sm tensor + draft-mtp  (baked-in head)    LOADED
      -sm tensor + draft-dflash (external -md)   FAILED, meta.cpp:543
      ... + -devd CUDA1                          FAILED, same
      ... + --no-spec-draft-backend-sampling     FAILED, same
      -sm layer  + draft-dflash                  LOADED

  draft-mtp uses the nextn head inside UD-Q4_K_XL and loads no second file, so
  it never reaches the graph-split assertion that stops draft-dflash. At
  147,456 on the computed -ts it loads as well: 66+0, CUDA0 with 1,571 MiB free
  and CUDA1 with 861.

  ITS EARLIER FAILURE WAS THE SAME BUG AS THE 0.38 tok/s INCIDENT. On the even
  split at 147,456 it died at ggml-backend-meta.cpp:1522,
  GGML_ASSERT(bufs.back() != nullptr) -- a buffer allocation returning null,
  which is what running out of memory looks like at that call site.

  IT IS A SWITCH AND NOT THE DEFAULT, BECAUSE WE HAVE NO RATE FOR IT.
  All three paired rounds at 147,456 were VOIDED by the output guard:

      generations copy the prompt rather than answer it:
      12-word windows found verbatim in the prompt = [0.519, 0.0, 0.23]

  identical across rounds, so deterministic. Three unpaired manual readings
  before the guard ran gave 44.5 / 54.3 / 92.7 tok/s and they are exactly the
  numbers CORRECTIONS 32 says not to trust: a speculative rate rises with how
  predictable the text is, and copying the prompt is maximally predictable.

  What IS measured on this configuration, three rounds rotated at 147,456:

      ngram-mod        [25.5, 25.4, 26.4]  spread 3.7 %
      no speculation   [21.8, 21.9, 21.8]  spread 0.6 %   -15.3 %

  THE HEAD COSTS MEMORY. With it the same configuration used about 2,750 MiB
  more across the two cards and CUDA1 finished with 861 MiB free, so the budget
  check below adds MTP_HEAD_MIB before deciding whether to start.

  NO -md, ever. The head is in the main file; -md would add a 1.4 GB sidecar
  for nothing (worker-q2kxl-mtp.ps1 says why at length).

WHY --sse-ping-interval IS 5 AND NOT llama.cpp's 30

  2026-08-27. Claude Code against this server showed

      Waiting for API response - will retry in 2m 24s - check your network

  and the network was fine. The server log for that session:

      prompt eval time = 88556.74 ms / 62408 tokens (704.72 tokens per second)
      prompt eval time = 53008.27 ms / 39747 tokens (749.83 tokens per second)

  THE WAIT IS PREFILL, NOT THINKING. Before the first token exists there is
  nothing to stream, and a 40-60k token prompt at ~750 tok/s is a minute.

  Measured here on a cold prompt of ~45,000 tokens, prefill 59.4 s:

      stream:false               nothing at all until 59.4 s
      stream:true, defaults      first byte 31.5 s (one 30 s ping), content 59.4 s
      stream + return_progress   progress from 1.4 s: 0%, 4%, 9%, 13%, 18% ...

  `return_progress` is what actually fixes the appearance -- it streams
  `prompt_progress` with processed/total while the prefill runs, which is
  exactly the live counter the developer asked for. IT IS A REQUEST FIELD.
  The client has to send it and Claude Code does not, so the server cannot turn
  it on from here.

  What the server does own is the keep-alive. 30 s of silence on a working
  connection is what a client reads as a dead one. 5 s costs nothing and keeps
  the connection observable throughout.

  THIS DOES NOT MAKE THE WAIT SHORTER. It makes it visible. The wait itself is
  prefill, and the lever for that is prompt REUSE: measured the same day, a
  repeated prompt reused 45,013 of 45,017 tokens and answered in under a
  second. The two slow turns in that log were 62,408 tokens then 39,747 -- the
  second SHORTER than the first, so the prefix had changed and nothing could be
  reused.

-MaxCtx: THE DEEPEST WINDOW THAT FITS, COMPUTED AT LAUNCH

  n_ctx_train is 262,144 and this machine can reach it -- but not always, and a
  hardcoded 262144 would be wrong in a way that only shows up under load.
  Measured hours apart on the same machine:

      desktop 1,600 MiB  ->  262,144 loaded
      desktop 2,575 MiB  ->  262,144 OOM'd, 1,696 MiB on device 1

  So -MaxCtx asks for the deepest context the CURRENT budget supports, capped at
  n_ctx_train, and it spends the micro-batch before it spends the context:

      1. 262,144 at the requested -ub
      2. 262,144 at half the -ub        (frees ~$UBatch MiB across the pair for
                                         about 3.5 % of prefill -- 971 -> 938)
      3. the deepest ctx that fits at the halved -ub

  RESERVING FOR WHAT HAPPENS AFTER LOAD. The budget arithmetic covers weights,
  KV and compute -- the allocations llama.cpp makes while STARTING. It makes
  more once there is work, and the difference is not theoretical:

      262,144 -ub 512, 336 MiB free on card 2  ->  DIED on the first request
      262,144 -ub 512, 488 MiB free on card 2  ->  survived 135,233 tokens

  $RUNTIME_RESERVE_MIB holds that back before choosing a depth. It is set from
  those two numbers and nothing more principled; it is a measured line, not a
  model of the allocator.

  WHAT -MaxCtx COSTS. At full depth the run finishes a large request with a few
  hundred MiB spare against about 2,000 at the 147,456 default. It is the depth
  bought with the margin, and the margin is what keeps a spill from happening
  silently at 0.38 tok/s.

LOADING IS NOT SURVIVING -- THE LADDER THAT SETTLED THE DEPTH

  262,144 with -ub 512 loaded, reported 66+0, answered /health, and then died
  the moment a real request arrived:

      CUDA error: out of memory
        current device: 1, in function alloc at ggml-cuda.cu:648
        cuMemSetAccess(start_ptr, reserve_size, &access, 1)

  llama.cpp allocates more once there is work to do, so a budget check that
  models LOAD-TIME demand cannot promise a run. This one does not claim to.

  So each depth was re-tested by pushing a ~135,000-token request through it,
  and only a depth that ANSWERED counts. Measured 2026-08-27, free MiB shown as
  display-card/other, after load then after the request:

      ctx 147,456  ub 1024  SURVIVED   2,100/2,097 -> 1,998/2,040
      ctx 196,608  ub 1024  SURVIVED   1,436/1,258 -> 1,248/1,208
      ctx 229,376  ub 1024  SURVIVED   1,156/  550 -> 1,071/  500
      ctx 262,144  ub 1024  refused at load
      ctx 229,376  ub  512  SURVIVED   1,312/1,010 -> 1,249/  974
      ctx 262,144  ub  512  SURVIVED     919/  488 ->   821/  452

  n_ctx_train IS reachable, by spending the compute buffer instead of the
  context. And note the run that died had 336 MiB free on the second card while
  the one that survived had 488 -- the line is somewhere around there, and it
  is close enough that the desktop decides which side of it you land on.

  -Ctx 262144 -UBatch 512 is therefore possible, NOT comfortable. 147,456 at
  -ub 1024 finishes a request with about 2,000 MiB on each card, and that is
  the difference between a configuration that works and one that works today.

HOW DEEP THE CONTEXT CAN GO, AND WHY THE ANSWER MOVES

  n_ctx_train is 262,144 and a ladder measured it fully resident. It is still
  NOT a context this machine can be relied on to serve, because the answer
  depends on what the desktop is holding when the server starts.

  Measured 2026-08-27, minutes apart, same machine, same profile:

      desktop 1,600 MiB   ->  budget 22,388  ->  262,144 loaded (ladder)
      desktop 2,575 MiB   ->  budget 22,398  ->  262,144 FAILED, OOM 1,696 MiB
                                                 on device 1

  262,144 needs 22,786 MiB: 16,130 weights + 4,608 KV + 2,048 compute. That is
  above the budget in both cases -- the ladder got away with it because it used
  a hardcoded -ts computed at a quieter moment, which handed the display card
  more than the reserve would now allow.

  237,568 DOES load at the current desktop, verified: 66+0, answers normally.
  It leaves 996 MiB free on the display card and 412 on the other. THAT IS NOT
  A SAFE PLACE TO SIT. It is the same margin that produced 0.38 tok/s, and one
  browser tab spends it.

  147,456 -- the default -- leaves about 2,900 and 2,265 MiB. The depth is
  bought with the margin that keeps the thing from collapsing, and the trade is
  not visible in a throughput number.

  THE BUDGET CHECK IS APPROXIMATE ON PURPOSE, AND IT IS OPTIMISTIC. `-ts`
  governs the WEIGHT slice; KV and compute do not distribute by the same ratio,
  so a run that the check approves can still finish with less headroom than the
  reserve implies -- 237,568 did. The check refuses the impossible; it does not
  promise comfort.

WHY THE CARDS ARE NAMED BY UUID

  `--main-gpu` defaults to 0, which on this machine is the RETIRED 4070 SUPER,
  and the dll carries sm_89 beside sm_120a so the wrong card is not merely
  reachable but fully supported. An index is a position in an enumeration the
  driver can reorder; after a reorder it keeps working and means a different
  card. Issue #50.

WHAT IS NOT MEASURED

  - Nothing at 147,456. Every figure above is ctx 16,384, and CORRECTIONS 23
    says the spread can be several times wider at depth.
  - This artifact with the served decoder (draft-mtp,ngram-mod). #44 already
    shows that decoder inverting sign with depth.
  - Any speculative decode rate comparing one card to two. CORRECTIONS 32:
    splitting changes the reduction order, so the logits, so the text -- and a
    speculative rate is partly a measure of how predictable the text is.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    # 147,456 is boot-verified 66+0 across both cards. The residency ceiling for
    # this artifact is 229,376; this matches the one-card profile so the two are
    # comparable at the depth anybody actually serves.
    [int]$Ctx  = 147456,
    [int]$Port = 8080,
    # Same meaning as in worker-q2kxl-mtp.ps1: 3 keeps the log small, the
    # tensor-assignment lines that prove residency need 5, and serve.ps1 asks
    # for what it needs rather than this default moving.
    [int]$Verbosity = 3,
    [ValidateSet('on', 'off', 'auto')]
    [string]$LogColors = 'auto',
    [string]$LogFile = '',
    # The ONLY access control this server has -- no API key, CORS '*', and
    # middleware_validate_api_key returns true immediately when no key is set.
    # Widening this does not weaken one control among several; it removes the
    # only one. Exposure is `serve.ps1 -Lan`, an act.
    # Pinned by bench/tests/test_the_dual_profile_serves_both_cards.py.
    [string]$BindAddress = '127.0.0.1',
    # Seconds between SSE keep-alive pings. llama.cpp's default is 30, which is
    # most of a minute of silence on a connection that is working -- see the
    # header. This does not shorten a wait, it makes one visible.
    [int]$SsePingIntervalSec = 5,
    # Serve the artifact's OWN chat template, without our one-line patch.
    #
    # WHY THE REASON IS NOT WRITTEN HERE: it was, in three places -- this header,
    # serve.ps1's, and templates/README.md -- and a rationale in three places
    # drifts the moment one is edited. This repository already corrected that
    # exact shape once (2649f4e). The README is the one copy; issue #65.
    #
    # In one line: without the patched template every Claude Code request comes
    # back HTTP 500. Unsloth Studio omits it safely because their client never
    # sends a system message after the user turn, and ours does.
    [switch]$StockTemplate,
    # Micro-batch. A parameter rather than a literal because the budget check
    # below needs it: the compute buffer is about one -ub of MiB per card.
    [int]$UBatch = 1024,
    # BOTH cards, by UUID. Empty resolves to the pair below; pass a
    # comma-separated list to override. Order is the CUDA enumeration order and
    # is what any -ts ratio would be indexed by.
    [string]$Device = '',
    # MiB left alone on whichever card already holds memory at launch -- that
    # card is driving the display, and the desktop grows while the server runs.
    # 2,500 comes from the incident: the desktop held 1,579 MiB at boot and the
    # spill happened with 317 MiB of slack, so the reserve has to cover the
    # desktop's growth and not merely its resting size.
    # Serve draft-mtp beside ngram-mod. OFF by default: it loads and runs here,
    # and its rate could not be measured -- the guard voided every round
    # because the generations copy the prompt. See the header.
    # Serve the deepest context the current free VRAM supports, capped at the
    # model's own n_ctx_train. Not a fixed 262,144: the budget moves with what
    # the desktop is holding, and 262,144 loaded at one moment and OOM'd at
    # another on this machine. See the header.
    # Serve draft-dflash beside ngram-mod, on the PATCHED binary. Measured
    # +123.8 % [+121.9, +125.1] over ngram-mod at ctx 65,536 -- more than double
    # the decode -- and it costs three things at once, which is why it is a
    # switch and its own pair of launchers rather than a default:
    #
    #   1. a DIFFERENT BINARY, llama.cpp-mirror, carrying a local patch nobody
    #      outside this project has reviewed. Unpatched, the arm aborts at
    #      ggml-backend-meta.cpp:543 -- TOP_K cannot take axis-0 logits.
    #   2. a SHALLOWER WINDOW. The measured ceiling is 131,072. 147,456 LOADS
    #      AND THEN DIES on the first real request, so -MaxCtx must not be used
    #      with it: "the deepest that fits" is the wrong question when the rung
    #      above the answer passes a health check.
    #   3. almost all the HEADROOM. 634/530 MiB free at 131,072 against about
    #      2,210 for the served configuration.
    [switch]$Dflash,

    # The DFlash2 draft depth. 0 means "not given" so the default can stay
    # where its comment is, beside the flag it sets.
    #
    #   2  what -Dflash has always served, and the ONLY value measured at the
    #      131,072 this switch serves: the run finishes with 634/530 MiB.
    #   4  the measured best on this split -- 55.72 tok/s against 52.64 at 7 and
    #      +109.2 % over ngram-mod, three paired rounds
    #      (results/tensor-draft-depth-65536.jsonl, issue #56) -- but that was
    #      ctx 65,536, and 4 has NEVER been measured at 131,072.
    #   7  the clamp (speculative.cpp:989, block_size - 1). MEASURED WORSE:
    #      -6.5 % against 4 in every round, and 308 MiB dearer.
    #
    # The recurrent state is 149.62 MiB x (1 + n_max), so 2 -> 4 spends 299 MiB
    # of the headroom the 131,072 ceiling exists to protect. The budget already
    # reserves it -- $DFLASH_DRAFTER_MIB is the measured cost AT n_max 4 -- so
    # serving 2 leaves that slack unused and serving 4 spends exactly what is
    # already set aside. Nothing in the fitting arithmetic changes.
    [int]$DflashN = 0,

    # Run DFlash2 on UNSLOTH'S SOURCE instead of ours -- their 0.3.0 tree with
    # our mirror patch applied and built here. Requires -Dflash, because the one
    # thing it offers is the one thing their shipped binary cannot do; without
    # it this would be a second spelling of -TheirBuild.
    #
    # READ THE BANNER CAREFULLY LATER. It says
    #   version: 0.3.0-dev (build 215, commit 9f55aee) ... Compiled by the Unsloth team
    # `0.3.0-dev` and the Unsloth line are THEIRS. `build 215` and the commit are
    # OUR repository's git, counted by their build system because the copied tree
    # has no .git of its own. A log from this binary is NOT a log from 10499.
    [switch]$TheirMirror,

    # Serve the NVFP4 artifact with the MTP head BAKED INTO IT, and the n-gram
    # retuned for that artifact. Measured +63.1 % [+58.3, +65.6] RESOLVED over
    # this profile's default at ctx 147,456 -- 39.4 / 42.6 / 42.6 against
    # 24.9 / 25.7 / 25.7, three paired rounds rotated, baseline spread 3.3 %
    # (results/nvfp4-final-147456.jsonl). The fastest thing measured here.
    #
    # It costs NONE of what -Dflash costs: no patch, no sidecar drafter, the
    # SERVED binary, and MORE headroom than the default (2,393-2,400 MiB free
    # against 1,998-2,026). What it changes is the MODEL FILE, which is why it
    # is a switch and not the default:
    #
    #   QUALITY IS UNMEASURED. ngram-mod acceptance falls 55.4 -> 22.1 on this
    #   artifact -- direct evidence it writes DIFFERENTLY, not merely faster --
    #   and this project has never measured quality on its own artifacts at all.
    #
    # Two measured facts the switch carries, and neither is a preference:
    #   the n-gram is n-match 24, not the 12 every other profile serves. 12 won
    #     on UD-Q4_K_XL and is worth 32.4-36.5 tok/s here against 42.9-43.1 for
    #     24 -- +27.1 % RESOLVED -- and 24 LOST on the other artifact at this
    #     exact depth. A verdict does not transfer across artifacts.
    #   the ceiling is 200,704, re-derived against a half-window request:
    #     91,428 tokens through it, finishing 1,133/654 MiB free. 229,376 LOADS
    #     and then dies (CORRECTIONS 35), which is why -MaxCtx is refused here.
    [switch]$Nvfp4,
    # Serve the NVFP4 artifact at its MEASURED ceiling, 200,704, instead of this
    # profile's 147,456 default. Not the same question as -MaxCtx: that one asks
    # the free VRAM for the deepest window it can afford, and here the answer is
    # a constant found by pushing a HALF-WINDOW request through each rung --
    # 229,376 loads, answers /health and then dies (CORRECTIONS 35).
    #
    # IT COSTS THE HEADROOM. 200,704 finished a 91,428-token request with 1,133
    # and 654 MiB free, against about 2,395 at the default. This project has
    # measured 336 MiB dying on a first request and 488 surviving, so the deep
    # rung sits above that line but not far above it, and the budget check below
    # is what decides on the day -- if the desktop has grown, it refuses.
    [switch]$Deep,
    # Load the vision tower so the server accepts images. OFF by default: the
    # benchmark work here is text and the tower costs 888 MiB with GPU offload,
    # which is llama.cpp's default.
    #
    # WHY IT WAS OFF AT ALL. This project recorded the whole --mmproj family as
    # "not applicable -- text only" (16-OPTIMIZATION-SURFACE.md). True of a
    # harness; false of a coding agent that pastes screenshots. On 2026-08-29
    # Claude Code sent five images through the LAN launcher and got five HTTP
    # 500s -- "image input is not supported ... you may need to provide the
    # mmproj" -- while the model's own chat template, read out of the GGUF at
    # load, begins `{%- set image_count = namespace(value...`. The model was
    # never the limitation.
    [switch]$Vision,
    # Adopt, as ONE bundle, the settings Unsloth Studio uses that we do not.
# Called -Beta from 2026-08-29; it was -Lean while the bundle was only about
# memory, and the name stopped fitting once it carried decoder values too.
    # Studio runs the same model file on the same two cards; the full diff is in
    # docs/researchs/unsloth-studio-config-2026-08-29.md. Eleven flags differ.
    # Taking them one at a time is eleven sweeps; taking them all silently is a
    # profile nobody can reason about. This is one switch, one bundle, one
    # paired measurement -- and if the bundle wins it gets bisected.
    #
    #   --cache-ram 0        a real session here held 20.4 GB working set and
    #   --ctx-checkpoints 0  34.4 GB private, with 32 checkpoints at ~350 MiB.
    #                        NOT free: our log shows them RESTORED at positions
    #                        47,940-50,091, so this trades RAM for re-prefill.
    #   --load-mode none     Studio's auto "picks None when it can prove the
    #                        model fits without paging, since a mapped read is
    #                        slower" (their words).
    #   --kv-unified         they set it; may be inert at -np 1.
    #   -t 2                 they use 2 against our 18. Everything is resident,
    #                        and the DRAFT sampler falls back to the CPU under
    #                        this split, so the right number is not obvious.
    #   --metrics            free Prometheus endpoint; no throughput claim.
    #
    # NOT in the bundle, on purpose: the n-gram parameters and the --spec-type
    # order (they have arm sets and belong in a sweep), --parallel 4 (we serve
    # one conversation and want the whole window for it), their shallow -c
    # (depth is the point of this machine), -ub 512 and --spec-draft-n-max 2
    # (ours are 1024 and 3, both measured), and the sampler, which is a QUALITY
    # lever on a project that has never measured quality.
    [switch]$Beta,
    # Takes `--kv-unified` back OUT of the -Beta bundle, and nothing else.
    #
    # 2026-08-29, same machine, same artifact, same evening, Discord streaming
    # through both: Unsloth Studio read 728-1,000 tok/s prefill and 34.9-48.0
    # decode where -Beta read 319-633 and 24.1-29.0, at the same depths. Our
    # mean accepted draft length was HIGHER on every row (2.5-2.8 against
    # 1.8-2.5), so the speculation is not what is slow -- the target model's
    # forward pass is, and the flags that matter there are the ones that lay
    # out attention and the KV cache.
    #
    # `--kv-unified` is the first of those and the only one that would also
    # explain the other open difference: Studio reuses a 39,616-token prefix
    # with `--ctx-checkpoints 0`, where that same setting made every one of our
    # requests re-read from token 0 with `forcing full prompt re-processing due
    # to lack of cache data`. One shared KV buffer is a plausible reason a
    # partial sequence removal cannot be done. PLAUSIBLE. NOT MEASURED.
    #
    # A switch rather than an edit, because -Beta is nine settings adopted
    # together and this project has already read a two-flag change as one.
    [switch]$NoKvUnified,
    # Unsloth Studio's command line, on our binary, as a BASELINE.
    #
    # By 2026-08-30 the servers differed in eight flags and every one had
    # a plausible story. One at a time is eight boots before the first
    # answer; this is ONE boot that says whether the remaining gap lives
    # in that list at all. If the clone matches Studio, bisecting is worth
    # doing. If it does not, the cause is somewhere no flag here reaches
    # and eight sweeps would have found nothing.
    #
    # NOT A CANDIDATE FOR SERVING. -c 107,899 is half the window this
    # machine exists to serve.
    [switch]$Clone,
    # Run on Unsloth Studio's llama-server instead of ours. THE CONFOUND THIS
    # REMOVES: every comparison against Studio assumed one binary and it is two
    # -- ours build 10499 (1deefcca3), theirs build 10679 (b84725557), 180
    # apart. -Clone alone cannot tell "their flags are better" from "their build
    # is newer"; this supplies the other cell of the 2x2.
    [switch]$TheirBuild,
    [switch]$MaxCtx,
    [switch]$Mtp,
    [int]$DisplayReserveMiB = 2500,
    # MiB left alone on a card that is holding nothing -- enough for the
    # driver's own allocations and nothing more.
    [int]$IdleReserveMiB = 512,
    [string]$Exe = "C:\AI\llama.cpp-blackwell\llama-server.exe",
    [string]$Model = "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\f1bfb127c64f7072bdd2cad55f258b9c8b2910fe\Qwen3.8-27B-UD-Q4_K_XL.gguf",
    [switch]$IKnowTheBuildIsWrong
)
$ErrorActionPreference = 'Stop'

# ---- -Dflash: choose the binary BEFORE the build guard runs ------------------
# The guard below reads ggml-cuda.dll beside $Exe and refuses a binary without
# both architectures. Swapping $Exe after it would check one file and run
# another -- the exact shape of the fault that put fifteen rows on a build with
# no Blackwell kernels earlier today.
$DFLASH_MAX_CTX     = 131072
# llama.cpp's own ceiling for this drafter: speculative.cpp:989 clamps n_max at
# block_size - 1, and the boot log prints block_size=8 for DFlash2.
$DFLASH_N_MAX_CLAMP = 7
$DFLASH_N_DEFAULT   = 2
# Unsloth's SOURCE, our patch, our build. Their SHIPPED binary aborts at
# ggml-backend-meta.cpp:543 the moment DFlash2 is asked for under -sm tensor
# (issue #52, 5f87e12), so the tree was copied out of %USERPROFILE%\.unsloth,
# patched with patches/dflash-mirror-output-b8472555.patch and built here.
# NEVER the same thing as -TheirBuild, which runs what they shipped.
$THEIR_MIRROR_EXE = "C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe"
# The model's own template with ONE line changed. templates/README.md says how
# to re-derive it when the artifact changes, and
# bench/tests/test_chat_template_travels.py checks the difference is still
# exactly one line against a live /props.
$TEMPLATE_FILE = "$PSScriptRoot\..\templates\qwen38-late-system.jinja"
# NVFP4's own ceiling for the DFlash2 pairing: 147,456 is the deepest MEASURED
# point (results/nvfp4-dflash-147456-n4.jsonl, 1,450 MiB free). Nothing above it
# has been tried with this pairing.
$NVFP4_DFLASH_MAX_CTX = 147456
# DFlash2's measured best draft depth, 2026-08-30. -DflashN overrides it.
$NVFP4_DFLASH_N       = 4
$DFLASH_EXE = "C:\AI\llama.cpp-mirror\build-mirror\bin\llama-server.exe"
$STUDIO_EXE = Join-Path $env:USERPROFILE '.unsloth\llama.cpp\build\bin\Release\llama-server.exe'
$DFLASH_MODEL = "C:\Users\xenod\.cache\huggingface\hub" +
    "\models--z-lab--Qwen3.8-27B-DFlash2-GGUF" +
    "\snapshots\57ab3265056d4024870b0621cfc2c127537020ed" +
    "\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
# 200,704, re-derived 2026-08-29 against a HALF-window request -- the standard
# every measured row in this project uses. The first figure was 229,376, taken
# because that rung survived a 65,643-token prompt, which is a QUARTER of its
# own window. Asked for 114,688 it loads with 680/206 MiB free and dies:
# `cudaMalloc failed: out of memory` on device 1. 206 MiB is BELOW the 336 this
# project has already recorded as dying. See CORRECTIONS 35.
#
#   229,376  loaded 680/206 free   DIED on the request
#   200,704  survived 91,428 tokens, finished 1,133/654 free
#   180,224  survived 83,127 tokens, finished 1,379/1,174 free
#
# A window is not a place to put one small prompt: a session that needs this
# depth will fill it.
$NVFP4_MAX_CTX = 200704

# UNSLOTH STUDIO'S OWN VALUES, read from the argv of the server it had
# running on 2026-08-30 at 00:11 (pid 29416). Both are things Studio
# computes from free VRAM at launch, exactly as this profile does, so they
# are a snapshot of one boot and not a constant of theirs. They are frozen
# here on purpose: a baseline that recomputed them would not be the same
# baseline twice.
$STUDIO_CTX = 107899
$STUDIO_TS  = '7648,13509'
# The vision tower. Shipped BY THE NVFP4 REPO ITSELF -- esatapedico publishes
# mmproj-BF16.gguf beside the weights, 931,146,432 bytes, the same size as
# unsloth's mmproj-BF16.gguf, which is what a shared tower looks like. It is a
# property of the base model, not of the quantisation, so one file covers both
# artifacts. Only this copy has been run here.
$MMPROJ = "C:\Users\xenod\.cache\huggingface\hub" +
    "\models--esatapedico--Qwen3.8-27B-NVFP4-MTP-GGUF" +
    "\snapshots\bcd7a7d3e251d4ec0fd15c72584b5eb9e0981383" +
    "\mmproj-BF16.gguf"
# Measured from the file: 931,146,432 bytes = 888 MiB, and --mmproj-offload
# defaults to ENABLED, so it lands on a card unless told otherwise.
$VISION_MIB = 888

$NVFP4_MODEL = "C:\Users\xenod\.cache\huggingface\hub" +
    "\models--esatapedico--Qwen3.8-27B-NVFP4-MTP-GGUF" +
    "\snapshots\bcd7a7d3e251d4ec0fd15c72584b5eb9e0981383" +
    "\Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf"
if ($Nvfp4) {
    # -Dflash IS allowed here from 2026-08-30. It was refused while the only
    # evidence was +0.2 % with the sign flipping -- a run that gave DFlash2 ctx
    # 147,456 against its best of 65,536, n_max 3 against 4, and n-match 12,
    # the window this artifact collapses on. Re-measured: +67.9 % RESOLVED over
    # ngram-mod at 65,536, and 44.48/44.56/44.23 at 147,456 against MTP's pooled
    # 42.77. -Mtp stays refused: the head is already in the file.
    if ($Mtp) {
        Write-Host "FATAL: -Nvfp4 already carries the MTP head INSIDE the model file." -ForegroundColor Red
        Write-Host "  -Mtp would ask for a second copy of what is already there." -ForegroundColor Yellow
        Write-Host "  -Dflash is the drafter that WAS measured on this artifact." -ForegroundColor Yellow
        exit 1
    }
    if ($MaxCtx) {
        if ($Deep) {
            Write-Host "FATAL: -Deep and -MaxCtx are two answers to one question." -ForegroundColor Red
            Write-Host "  -Deep is the measured 200,704; -MaxCtx asks the free VRAM." -ForegroundColor Yellow
            exit 1
        }
        Write-Host "FATAL: -MaxCtx cannot be used with -Nvfp4." -ForegroundColor Red
        Write-Host "  The ceiling here is $NVFP4_MAX_CTX, measured with a real request:" -ForegroundColor Yellow
        Write-Host "  91,428 tokens through it, finishing 1,133/654 MiB free. 229,376 LOADS," -ForegroundColor Yellow
        Write-Host "  answers /health and DIES on the request. 'The deepest that fits' is" -ForegroundColor Yellow
        Write-Host "  the wrong question at an edge where a window can pass a health check." -ForegroundColor Yellow
        exit 1
    }
    if ($Deep) { $Ctx = $NVFP4_MAX_CTX }
    if ($Ctx -gt $NVFP4_MAX_CTX) { $Ctx = $NVFP4_MAX_CTX }
    # The clone's window is theirs, and it is set HERE so the residency
    # guard below checks the window that will actually be served rather
    # than the one this profile would have chosen.
    if ($Clone) { $Ctx = $STUDIO_CTX }
    $Model = $NVFP4_MODEL
} elseif ($Deep) {
    Write-Host "FATAL: -Deep is the NVFP4 ceiling; pass -Nvfp4 too." -ForegroundColor Red
    Write-Host "  200,704 was measured on THAT artifact and does not transfer." -ForegroundColor Yellow
    Write-Host "  On UD-Q4_K_XL the deep question is a budget one: use -MaxCtx." -ForegroundColor Yellow
    exit 1
}
if ($Dflash) {
    if ($Mtp) {
        Write-Host "FATAL: -Dflash and -Mtp are two different drafters; pick one." -ForegroundColor Red
        exit 1
    }
    if ($MaxCtx) {
        Write-Host "FATAL: -MaxCtx cannot be used with -Dflash." -ForegroundColor Red
        Write-Host "  The ceiling here is $DFLASH_MAX_CTX, and it is not a budget question:" -ForegroundColor Yellow
        Write-Host "  147,456 LOADS, answers /health, and dies on the first real request." -ForegroundColor Yellow
        exit 1
    }
    # THESE TWO CAPS ARE UD-Q4_K_XL's, and must not leak onto NVFP4. 131,072 is
    # that artifact's ceiling because 147,456 loads there and dies on the first
    # real request. NVFP4 is about 5 GB smaller and 147,456 was MEASURED working
    # with this pairing, finishing with 1,450 MiB
    # (results/nvfp4-dflash-147456-n4.jsonl). Applying the Q4 cap here would
    # silently serve a shallower window than the evidence covers -- and nothing
    # ABOVE 147,456 has been measured with it either, so that is the NVFP4 cap.
    if ($Nvfp4) {
        if ($Ctx -gt $NVFP4_DFLASH_MAX_CTX) { $Ctx = $NVFP4_DFLASH_MAX_CTX }
    } else {
        if ($Ctx -gt $DFLASH_MAX_CTX) { $Ctx = $DFLASH_MAX_CTX }
        if ($UBatch -gt 512)          { $UBatch = 512 }
    }
    $Exe = $DFLASH_EXE
}
# The clamp is llama.cpp's, at speculative.cpp:989 -- block_size - 1, and this
# drafter's block_size is 8. Out of range is CLAMPED SILENTLY there, so the
# server would run one depth while the launcher and the log said another.
# Refusing is the only way the two cannot disagree.
if ($DflashN -ne 0) {
    if (-not $Dflash) {
        Write-Host "FATAL: -DflashN sets the DFlash2 draft depth and needs -Dflash." -ForegroundColor Red
        exit 1
    }
    if ($DflashN -lt 1 -or $DflashN -gt $DFLASH_N_MAX_CLAMP) {
        Write-Host "FATAL: -DflashN must be 1..$DFLASH_N_MAX_CLAMP." -ForegroundColor Red
        Write-Host "  speculative.cpp:989 clamps at block_size - 1, and this drafter's" -ForegroundColor Yellow
        Write-Host "  block_size is 8. Out of range is clamped SILENTLY, so the server" -ForegroundColor Yellow
        Write-Host "  would run one depth while every log said another." -ForegroundColor Yellow
        exit 1
    }
}

# ---- -TheirBuild: the other binary, and the loader path it needs -------------
# Before the guard, for the same reason -Dflash is.
#
# THE FAULT THIS BLOCK EXISTS TO PREVENT. Launched with a bare PATH, Studio's
# binary reports
#
#     device_info:
#       - CPU     : 13th Gen Intel(R) Core(TM) i5-13500
#
# and NO CUDA device at all -- then serves, from the CPU, at a speed somebody
# would write down. It is STUDIO that prepends the loader path, and CUDA 13
# keeps cudart64_13.dll and cublas64_13.dll in %CUDA_PATH%\bin\x64 rather than
# \bin, which is why the obvious directory is the wrong one. Nothing ships
# beside their binary to supply them.
#
# Verified by running it both ways. Bare PATH gives CPU only; with the x64
# directory prepended it reports
#     ARCHS = 860,890,900,1000,1200 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1
# against our 890,1200 with the same two feature flags -- so the compile-time
# difference that matters here is a source/build-lineage delta, not a missing
# kernel.
if ($TheirBuild) {
    $Exe = $STUDIO_EXE
    $cudaBin = @()
    if ($env:CUDA_PATH) {
        $cudaBin += (Join-Path $env:CUDA_PATH 'bin')
        $cudaBin += (Join-Path $env:CUDA_PATH 'bin\x64')
    }
    $loaderDirs = @((Split-Path $Exe -Parent)) + $cudaBin
    $haveRuntime = $false
    foreach ($d in $loaderDirs) {
        if ((Test-Path $d) -and
            (Get-ChildItem -Path $d -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue)) {
            $haveRuntime = $true
        }
    }
    if (-not $haveRuntime) {
        # REFUSE. A warning would be read past and the run would produce a
        # believable number from the wrong hardware, which is the one failure
        # this repository's north star names.
        Write-Host "FATAL: cannot find cudart64_*.dll for $Exe" -ForegroundColor Red
        Write-Host "  Looked in: $($loaderDirs -join '; ')" -ForegroundColor Yellow
        Write-Host "  Without it that binary starts, finds NO CUDA device, and" -ForegroundColor Yellow
        Write-Host "  serves from the CPU without saying so." -ForegroundColor Yellow
        Write-Host "  Set CUDA_PATH to a CUDA 13 install and try again." -ForegroundColor Yellow
        exit 1
    }
    $env:PATH = ($loaderDirs -join ';') + ';' + $env:PATH
    Write-Host "  build     THEIRS -- $Exe" -ForegroundColor Yellow
    Write-Host "            ours is build 10499 (1deefcca3); this is 10679 (b84725557)" -ForegroundColor DarkGray
}

# ---- the build ---------------------------------------------------------------
# A binary without Blackwell SASS runs here through PTX JIT at 2.20x the prefill
# time with nothing in any log to say so. The match is a SUBSTRING one on
# purpose: cmake rewrites 120 to 120a and the cubins are named sm_120a, so an
# exact 'sm_120' test would reject a correctly built binary.
#
# This profile needs BOTH architectures, because one of its two cards is Ada.
$dll = Join-Path (Split-Path $Exe -Parent) 'ggml-cuda.dll'
$cuobjdump = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\cuobjdump.exe'

if (-not (Test-Path $Exe)) {
    Write-Host "FATAL: no server at $Exe" -ForegroundColor Red
    Write-Host "  Rebuild with -DCMAKE_CUDA_ARCHITECTURES=`"89;120`"" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $Model)) {
    Write-Host "FATAL: no model at $Model" -ForegroundColor Red
    exit 1
}

if ((Test-Path $dll) -and (Test-Path $cuobjdump)) {
    $elf = & $cuobjdump --list-elf $dll 2>$null
    $arches = ($elf | Select-String -Pattern 'sm_\w+' -AllMatches |
               ForEach-Object { $_.Matches.Value } | Sort-Object -Unique)
    foreach ($needed in @('sm_120', 'sm_89')) {
        if (-not ($arches -match [regex]::Escape($needed))) {
            Write-Host "FATAL: $dll has no $needed SASS (found: $($arches -join ', '))" -ForegroundColor Red
            Write-Host "  This profile drives an Ada card AND a Blackwell card." -ForegroundColor Yellow
            Write-Host "  A missing architecture is JIT-compiled from PTX with" -ForegroundColor Yellow
            Write-Host "  nothing in the log to say so." -ForegroundColor Yellow
            Write-Host "  Rebuild, or pass -IKnowTheBuildIsWrong (never for a measurement)." -ForegroundColor Yellow
            if (-not $IKnowTheBuildIsWrong) { exit 1 }
            Write-Host "  OVERRIDDEN -- results are not comparable to anything." -ForegroundColor Magenta
        }
    }
} else {
    Write-Host "WARNING: cannot verify GPU architecture (missing $dll or cuobjdump)." -ForegroundColor Yellow
}

# ---- the cards ---------------------------------------------------------------
# Checked BEFORE the model loads. An absent UUID does not make llama-server
# fail: it reports `(none)` for devices and then runs on the CPU, producing
# correct output at a rate no row explains. With two UUIDs there are two ways to
# be wrong, and a 16.69 GiB model takes long enough to load that the wrong
# answer arrives an hour later.
. (Join-Path $PSScriptRoot 'Get-GpuVram.ps1')

if (-not $Device) {
    $Device = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4,' + $script:ServedGpuUuid
}

$installed = @(Get-InstalledGpu)
$wanted = @($Device -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($uuid in $wanted) {
    if (-not ($installed | Where-Object { $_.Uuid -eq $uuid })) {
        Write-Host "FATAL: GPU $uuid is not installed." -ForegroundColor Red
        Write-Host "  Installed:" -ForegroundColor Yellow
        $installed | ForEach-Object {
            Write-Host ("    {0}  {1}" -f $_.Uuid, $_.Name) -ForegroundColor Yellow
        }
        Write-Host "  llama-server would see fewer devices than this profile" -ForegroundColor Yellow
        Write-Host "  was measured on, and UD-Q4_K_XL does not fit on one card." -ForegroundColor Yellow
        exit 1
    }
}
if ($wanted.Count -lt 2) {
    Write-Host "FATAL: this profile needs two cards; -Device names $($wanted.Count)." -ForegroundColor Red
    Write-Host "  UD-Q4_K_XL is 16.69 GiB and spills 11 layers on one 16 GB card" -ForegroundColor Yellow
    Write-Host "  -- 11.7 tok/s against 20.9. Use worker-q2kxl-mtp.ps1 instead." -ForegroundColor Yellow
    exit 1
}
$env:CUDA_VISIBLE_DEVICES = $Device

# ---- the split ---------------------------------------------------------------
# Computed here rather than left to llama.cpp, which splits EVENLY when given no
# ratio (llama-model.cpp:707). Even is wrong on two unequal cards and disastrous
# when the smaller one is also the display GPU -- see the header.
$budgets = @()
$report  = @()
foreach ($uuid in $wanted) {
    $g = Get-GpuVram -Uuid $uuid
    if (-not $g) {
        Write-Host "FATAL: cannot read VRAM for $uuid; refusing to guess a split." -ForegroundColor Red
        exit 1
    }
    # A card already holding memory is the one drawing the desktop. Which card
    # that is cannot be assumed -- it is whichever the monitor is plugged into.
    $isDisplay = $g.Used -gt 500
    $reserve   = if ($isDisplay) { $DisplayReserveMiB } else { $IdleReserveMiB }
    $budget    = $g.Free - $reserve
    $budgets  += $budget
    $report   += [pscustomobject]@{
        Uuid = $uuid; Free = $g.Free; Reserve = $reserve
        Budget = $budget; Display = $isDisplay
    }
}

# UD-Q4_K_XL needs about 16,130 MiB of weights whatever the split, plus KV and
# compute. Refusing below that is the only thing standing between the developer
# and another silent spill: --fit is inert here and llama.cpp will not refuse.
# WHAT THE RUN ACTUALLY NEEDS, not just its weights.
#
# The first version of this check compared the budget against WEIGHTS_MIB alone
# and therefore approved EVERY context. On 2026-08-27 it approved 262,144 and
# llama.cpp died on it:
#
#   ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1696.30 MiB
#     on device 1: cudaMalloc failed: out of memory
#
# A ladder had measured 262,144 as fully resident hours earlier -- with a
# hardcoded -ts computed when the desktop held about 1,600 MiB. By that run the
# desktop held 2,575, the display card's budget fell by 920 MiB, and
# proportional splitting pushed the difference onto the other card.
#
# So the demand is weights + KV(ctx) + compute(ub), and the KV rate is measured:
# at ctx 147,456 llama.cpp reports 1,296.00 MiB per card, which is 2,592 MiB
# over 147,456 tokens = 18.00 KiB per token at -ctk q4_0 -ctv q4_0.
# -Dflash: a different binary, a hard ceiling, a halved micro-batch, and two
# extra tenants in the budget. All four are measured, none is a preference.
#
#   the drafter                 1,936 MiB resident
#   the mirror patch            1,080 MiB -- measured 2026-08-27 by loading the
#                               served and patched binaries at the same ctx with
#                               no drafter: 6,964 MiB free against 5,884
#   ceiling                     131,072. 147,456 loads and dies on the first
#                               real request; 163,840 does not load
#   -ub                         512, which returns about 1,024 MiB for ~3.5 % of
#                               prefill and nothing of decode
$DFLASH_DRAFTER_MIB = 1936
$DFLASH_MIRROR_MIB  = 1080

# -Nvfp4: 14,173 MiB on disk against UD-Q4_K_XL's 17,092, and the nextn head is
# inside that figure rather than added to it -- which is why $MTP_HEAD_MIB does
# NOT apply here. Measured at ctx 147,456 it finishes with 2,393-2,400 MiB free
# against 1,998-2,026 for the default, so the smaller file is real.
$NVFP4_WEIGHTS_MIB = 14173

$WEIGHTS_MIB = 16130
if ($Nvfp4)  { $WEIGHTS_MIB  = $NVFP4_WEIGHTS_MIB }
if ($Vision) { $WEIGHTS_MIB += $VISION_MIB }
if ($Dflash) { $WEIGHTS_MIB += $DFLASH_DRAFTER_MIB + $DFLASH_MIRROR_MIB }
$KV_KIB_PER_TOKEN = 18.0
# One compute buffer per card, and it tracks -ub. 1,024.30 MiB each at -ub 1024,
# read from the boot log.
$COMPUTE_MIB = 2 * [Math]::Max(256, $UBatch)
# The nextn head is not free. Measured: about 2,750 MiB more across the two
# cards with -Mtp on, and CUDA1 finished with 861 MiB free.
$MTP_HEAD_MIB = 2750
if ($Mtp) { $WEIGHTS_MIB += $MTP_HEAD_MIB }

# Held back before choosing a depth, for the allocations llama.cpp makes once a
# request arrives. Measured, not modelled: 336 MiB free on the second card died
# on the first request, 488 survived 135,233 tokens.
$RUNTIME_RESERVE_MIB = 768
$N_CTX_TRAIN = 262144

if ($MaxCtx) {
    # Spend the micro-batch before the context: halving -ub frees about $UBatch
    # MiB across the pair for ~3.5 % of prefill, where the same MiB bought with
    # context costs tens of thousands of tokens.
    $totalBudget = ($budgets | Measure-Object -Sum).Sum - $RUNTIME_RESERVE_MIB
    $chosenCtx = 0
    foreach ($ub in @($UBatch, [Math]::Max(256, [int]($UBatch / 2)))) {
        $comp  = 2 * [Math]::Max(256, $ub)
        $spare = $totalBudget - $WEIGHTS_MIB - $comp
        if ($Mtp) { $spare -= $MTP_HEAD_MIB }
        if ($spare -le 0) { continue }
        $fits = [int]([Math]::Floor((($spare * 1024) / $KV_KIB_PER_TOKEN) / 4096) * 4096)
        if ($fits -ge $N_CTX_TRAIN) { $chosenCtx = $N_CTX_TRAIN; $UBatch = $ub; break }
        if ($fits -gt $chosenCtx)   { $chosenCtx = $fits;        $UBatch = $ub }
    }
    if ($chosenCtx -lt 4096) {
        Write-Host "FATAL: -MaxCtx found no context that fits with a runtime reserve." -ForegroundColor Red
        Write-Host "  Close what is using the display card, or run worker-q2kxl-mtp.ps1." -ForegroundColor Yellow
        exit 1
    }
    $Ctx = $chosenCtx
    $capped = if ($Ctx -ge $N_CTX_TRAIN) { " -- n_ctx_train, the model's own ceiling" } else { "" }
    Write-Host ""
    Write-Host ("  window    {0:N0} tokens at -ub {1}{2}" -f $Ctx, $UBatch, $capped) -ForegroundColor Cyan
    Write-Host ("            chosen from {0:N0} MiB of budget less {1:N0} reserved for" -f `
                (($budgets | Measure-Object -Sum).Sum), $RUNTIME_RESERVE_MIB) -ForegroundColor DarkGray
    Write-Host "            the allocations that happen after load. It moves with the desktop." -ForegroundColor DarkGray
}

$COMPUTE_MIB = 2 * [Math]::Max(256, $UBatch)
if (-not $MaxCtx) {
    # Stated whether or not it was computed, because the launcher no longer
    # states it -- and a window nobody prints is one nobody can check against
    # what the boot log says.
    Write-Host ""
    Write-Host ("  window    {0:N0} tokens at -ub {1}" -f $Ctx, $UBatch) -ForegroundColor Cyan
}
$kvMib     = [int](($Ctx * $KV_KIB_PER_TOKEN) / 1024)
$demandMib = $WEIGHTS_MIB + $kvMib + $COMPUTE_MIB
$total = ($budgets | Measure-Object -Sum).Sum
# PRINT THE ARITHMETIC EVEN WHEN IT FITS. It used to appear only inside the
# FATAL block, which is backwards: the moment you want to know how close this
# is to the edge is the moment it succeeds. A run that clears by 200 MiB and one
# that clears by 6,000 looked identical, and this machine has measured 336 MiB
# free dying on a first request against 488 surviving.
Write-Host ("  demand    {0:N0} MiB  =  {1:N0} weights + {2:N0} KV + {3:N0} compute" -f `
            $demandMib, $WEIGHTS_MIB, $kvMib, $COMPUTE_MIB) -ForegroundColor Cyan
Write-Host ("            budget {0:N0} MiB, spare {1:N0}" -f $total, ($total - $demandMib)) `
           -ForegroundColor $(if (($total - $demandMib) -lt 1024) { "Yellow" } else { "DarkGray" })
if (($budgets | Where-Object { $_ -lt 1024 }).Count -gt 0 -or $total -lt $demandMib) {
    # The deepest context this budget WOULD hold, so the developer is not left
    # bisecting -Ctx by hand. Rounded down to a multiple of 4,096.
    $spare   = $total - $WEIGHTS_MIB - $COMPUTE_MIB
    $fitsCtx = if ($spare -gt 0) {
        [int]([Math]::Floor(($spare * 1024 / $KV_KIB_PER_TOKEN) / 4096) * 4096)
    } else { 0 }
    Write-Host "FATAL: ctx $Ctx does not fit without spilling." -ForegroundColor Red
    Write-Host ("    needs {0:N0} MiB  =  {1:N0} weights + {2:N0} KV + {3:N0} compute" -f `
                $demandMib, $WEIGHTS_MIB, $kvMib, $COMPUTE_MIB) -ForegroundColor Yellow
    # Two ways out, and the cheaper one is usually the micro-batch. Each card's
    # compute buffer is about one -ub of MiB, so halving it hands back roughly
    # $UBatch MiB across the pair -- and prefill only loses about 3.5 % going
    # from 1024 to 512, against 24,576 tokens of context for the other route.
    $halfUb = [Math]::Max(256, [int]($UBatch / 2))
    $freedByUb = $COMPUTE_MIB - (2 * $halfUb)
    if ($halfUb -lt $UBatch -and ($total - ($demandMib - $freedByUb)) -ge 0) {
        Write-Host ("    -UBatch {0} frees {1:N0} MiB and keeps ctx {2:N0}." -f `
                    $halfUb, $freedByUb, $Ctx) -ForegroundColor Cyan
        Write-Host ("    try:  -Ctx {0} -UBatch {1}" -f $Ctx, $halfUb) -ForegroundColor Cyan
        Write-Host "    (prefill costs about 3.5 % per halving; measured 971 -> 938 tok/s)" -ForegroundColor DarkGray
    }
    if ($fitsCtx -ge 4096) {
        Write-Host ("    or keep -UBatch {0} and drop to about {1:N0} tokens" -f `
                    $UBatch, $fitsCtx) -ForegroundColor Cyan
        Write-Host ("    try:  -Ctx {0}" -f $fitsCtx) -ForegroundColor Cyan
    } elseif ($halfUb -ge $UBatch) {
        Write-Host "    the weights alone do not fit; close what is using the display card." -ForegroundColor Yellow
    }
    foreach ($r in $report) {
        Write-Host ("    {0}  free {1,6:N0} MiB  reserve {2,5:N0}  budget {3,6:N0}{4}" -f `
                    $r.Uuid.Substring(0,12), $r.Free, $r.Reserve, $r.Budget,
                    $(if ($r.Display) { "   <- drawing the desktop" } else { "" })) -ForegroundColor Yellow
    }
    Write-Host ("    budget {0:N0} MiB after reserving for the desktop." -f $total) -ForegroundColor Yellow
    Write-Host "  --fit cannot rescue this: it is not implemented for SPLIT_MODE_TENSOR." -ForegroundColor Yellow
    Write-Host "  A spill here is silent and costs ~85x -- 0.38 tok/s was measured." -ForegroundColor Yellow
    Write-Host "  Close what is using the display card, or run worker-q2kxl-mtp.ps1." -ForegroundColor Yellow
    # -WhatIf WARNS AND CONTINUES. A dry run exists to answer "what would you
    # run", and the budget is a fact about this minute's desktop rather than
    # about the configuration -- refusing to print the argv because a browser
    # tab is open makes the preview unusable exactly when it is most wanted.
    # Only a real launch is refused, which is what the guard is for.
    if ($WhatIfPreference) {
        Write-Host "  (-WhatIf: previewing anyway; a real launch here would be refused.)" -ForegroundColor DarkGray
    } else {
        exit 1
    }
}

if ($TheirMirror) {
    if (-not $Dflash) {
        Write-Host "FATAL: -TheirMirror exists to run DFlash2 on Unsloth's source." -ForegroundColor Red
        Write-Host "  Without -Dflash it is a second spelling of -TheirBuild, and one" -ForegroundColor Yellow
        Write-Host "  flag meaning two artifacts makes every later log ambiguous." -ForegroundColor Yellow
        exit 1
    }
    if ($TheirBuild) {
        Write-Host "FATAL: -TheirMirror and -TheirBuild are two different binaries." -ForegroundColor Red
        Write-Host "  -TheirBuild runs what Unsloth SHIPPED, which aborts on DFlash2." -ForegroundColor Yellow
        Write-Host "  -TheirMirror runs their source with our patch. Pick one." -ForegroundColor Yellow
        exit 1
    }
    $Exe = $THEIR_MIRROR_EXE
    # THE CPU-RUN FAULT. A llama-server that cannot find cudart64_13.dll reports
    # no CUDA devices and serves happily from the CPU -- a believable slow number
    # from the wrong hardware. The three runtime DLLs were copied beside this
    # binary so it is self-contained, and this refuses if they went missing.
    $needed = @('cudart64_13.dll', 'cublas64_13.dll')
    $binDir = Split-Path $Exe -Parent
    foreach ($d in $needed) {
        if (-not (Test-Path (Join-Path $binDir $d))) {
            Write-Host "FATAL: $d is not beside $Exe." -ForegroundColor Red
            Write-Host "  Without it llama-server finds NO CUDA device and serves from" -ForegroundColor Yellow
            Write-Host "  the CPU without saying so. Copy it from llama.cpp-mirror\build-mirror\bin." -ForegroundColor Yellow
            exit 1
        }
    }
}

$tsArg = if ($Clone) {
    # Theirs, verbatim, from the argv of the server that was running on
    # 2026-08-30 00:11. 36/64 against our 33/67 -- and ours is derived
    # from FREE VRAM, which optimises for fitting rather than for speed.
    # Under -sm tensor the split is also a split of COMPUTE, and that has
    # never been tested here.
    @('-ts', $STUDIO_TS)
} else {
    @('-ts', ($budgets -join ','))
}

# The decoder. ngram-mod alone is what has a measured rate here; -Mtp adds the
# baked-in head beside it, which runs but whose rate the guard would not accept.
# 3 is llama.cpp's OWN default (`--spec-draft-n-max N (default: 3)`) and we get
# it by not setting anything. Studio sets 2 deliberately: its UI documents 2 for
# MTP on GPU, 3 for CPU/Mac -- so 2 is THEIR choice for a GPU run, not a
# standard. Our real-use acceptance per position is (0.690, 0.448, 0.284), so
# position three still lands 28 % of the time and 3 looks earned. -Beta carries
# 2 so that argument gets tested rather than repeated.
# MEASURED 2026-08-29, and 2 LOST. -Beta carried Studio's 2 for one afternoon;
# the developer's own use said it felt slower and the server's counters said
# why -- not the rate, which was a cross-session comparison, but the mechanism:
#
#   n-max 3   297 drafts ->   891 tokens = 3 per draft, mean accepted len 2.80
#   n-max 2   887 drafts -> 1,774 tokens = 2 per draft, mean accepted len 2.12
#
# The acceptance RATE barely moved (0.60 -> 0.54); the accepted LENGTH fell
# 24 %, so every verify step advances less far. Decode read 43-45 tok/s before
# and 25-33 after. Exactly what per-position acceptance predicted:
# (0.690, 0.448, 0.284) -- position three lands 28 % of the time.
#
# Studio documents 2 for MTP on GPU. A default from another product is still a
# verdict from another configuration, and this one does not hold here.
$draftN = '3'
# -Beta RUNS MTP ALONE, at the developer's request. Two independent
# observations point the same way:
#
#   Studio's fastest of eight runs on this same model file was draft-mtp by
#   itself -- 54.95 tok/s against 52.28 and 49.72 for MTP+ngram.
#
#   On this machine, on real agent traffic, ngram-mod DOES NOT FIRE. Two
#   sessions logged `#gen drafts = 0`; an earlier eighteen-minute session logged
#   5 drafts in 4,653 calls.
#
# THE DEFAULT KEEPS THE PAIRING, and that is not indecision. +63.1 % was
# measured with it on `real-code-vendor` -- repeated vendor source, exactly the
# text an n-gram is good at. Both numbers are real and they are about different
# workloads; this repository has measured only one of the two.
$specArg = if ($Nvfp4 -and $Beta) {
    @('--spec-type', 'draft-mtp', '--spec-draft-n-max', $draftN)
} elseif ($Nvfp4 -and $Dflash) {
    # The head in the file is IGNORED and DFlash2 drafts instead. Measured
    # 2026-08-30: +67.9 % over ngram-mod at 65,536 (RESOLVED), and level with
    # the head at 147,456 -- 44.48/44.56/44.23 against a pooled 42.77 -- while
    # spending about 950 MiB more headroom. The n-gram window is NOT touched:
    # 24 is this artifact's own, and 12 collapses here.
    @('--spec-type', 'draft-dflash,ngram-mod',
      '-md', $DFLASH_MODEL, '-ngld', '99',
      '--spec-draft-n-max', "$(if ($DflashN -ne 0) { $DflashN } else { $NVFP4_DFLASH_N })")
} elseif ($Nvfp4) {
    # The head is in the file; no -md, no second model on any device.
    @('--spec-type', 'draft-mtp,ngram-mod', '--spec-draft-n-max', $draftN)
} elseif ($Mtp) {
    @('--spec-type', 'draft-mtp,ngram-mod', '--spec-draft-n-max', '3')
} elseif ($Dflash) {
    # n-max 2 by default, not the 4 the arena measured with: the recurrent-state
    # buffer is 149.62 MiB x (1 + n_max), so 4 -> 2 returns 299 MiB, and at
    # 131,072 the run finishes with 634/530 MiB. Every one of those MiB was
    # needed. `-DflashN` overrides it; see its comment for what each value is
    # measured at, and note that 2 is the only one measured AT THIS DEPTH.
    @('--spec-type', 'draft-dflash,ngram-mod',
      '-md', $DFLASH_MODEL, '-ngld', '99',
      '--spec-draft-n-max', "$(if ($DflashN -ne 0) { $DflashN } else { $DFLASH_N_DEFAULT })")
} else {
    @('--spec-type', 'ngram-mod')
}
if ($TheirMirror) {
    Write-Host "  binary    UNSLOTH's SOURCE (0.3.0), our mirror patch, built here." -ForegroundColor Yellow
    Write-Host "            UNVERIFIED: this binary has never been seen to load DFlash2" -ForegroundColor Red
    Write-Host "            under -sm tensor. If it aborts at ggml-backend-meta.cpp:543," -ForegroundColor Red
    Write-Host "            the patch did not take and the run is not your fault." -ForegroundColor Red
    Write-Host "            Its banner reads 'build 215, commit ...' -- that build number" -ForegroundColor Yellow
    Write-Host "            is OUR repository's git, not theirs. It is NOT 10499." -ForegroundColor Yellow
}
if ($Dflash -and $Nvfp4) {
    Write-Host "  decoder   draft-dflash + ngram-mod ON NVFP4 -- NOT a speedup." -ForegroundColor Yellow
    Write-Host "            At this depth it is +4.0 % over the head already in the file," -ForegroundColor Yellow
    Write-Host "            which is UNDER the 13.6 % floor and measured across boots." -ForegroundColor Yellow
    Write-Host "            What it buys is STEADINESS: spread 0.7 % against 9.3 %." -ForegroundColor Green
    Write-Host "            What it costs is about 950 MiB of headroom -- 1,450 free" -ForegroundColor Yellow
    Write-Host "            against 2,400 for the baked-in head." -ForegroundColor Yellow
    Write-Host "            PATCHED BINARY, reviewed by nobody outside this project." -ForegroundColor Yellow
    Write-Host "            Deepest MEASURED with this pairing: ${NVFP4_DFLASH_MAX_CTX}." -ForegroundColor Yellow
}
if ($Dflash -and -not $Nvfp4) {
    Write-Host "  decoder   draft-dflash + ngram-mod -- +123.8 % over ngram-mod at ctx 65,536." -ForegroundColor Green
    Write-Host "            PATCHED BINARY, reviewed by nobody outside this project." -ForegroundColor Yellow
    Write-Host "            Window capped at ${DFLASH_MAX_CTX} -- 147,456 loads and then dies." -ForegroundColor Yellow
    Write-Host "            It finishes with about 600 MiB per card against ~2,210 served." -ForegroundColor Yellow
    if ($DflashN -ne 0 -and $DflashN -ne $DFLASH_N_DEFAULT) {
        $extra = [Math]::Round(149.62 * ($DflashN - $DFLASH_N_DEFAULT))
        Write-Host "  n-max     $DflashN, not the default $DFLASH_N_DEFAULT -- costs about $extra MiB more." -ForegroundColor Yellow
        Write-Host "            Recurrent state is 149.62 MiB x (1 + n_max); 299 MiB from 2 to 4." -ForegroundColor Yellow
        Write-Host "            4 is the measured best at ctx 65,536 (55.72 tok/s, +109.2 % over" -ForegroundColor Yellow
        Write-Host "            ngram-mod) and has NEVER been measured at $DFLASH_MAX_CTX." -ForegroundColor Yellow
    }
}
if ($Mtp) {
    Write-Host "  decoder   draft-mtp + ngram-mod -- LOADS HERE, RATE NOT MEASURED." -ForegroundColor Yellow
    Write-Host "            Every paired round was voided: the generations copy the prompt." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  split     -ts $($budgets -join ',')  (MiB of budget per card)" -ForegroundColor Cyan
foreach ($r in $report) {
    Write-Host ("            {0}  free {1,6:N0}  reserve {2,5:N0}{3}" -f `
                $r.Uuid.Substring(0,12), $r.Free, $r.Reserve,
                $(if ($r.Display) { "   <- drawing the desktop" } else { "" })) -ForegroundColor DarkGray
}
Write-Host ""

# An ARRAY, empty when no log was asked for. An inline `$(if ...)` would pass an
# empty string as a real argument and llama-server would see a flag it cannot
# parse -- the kind of failure that only shows up on the default path.
$logFileArg = if ($LogFile) { @('--log-file', $LogFile) } else { @() }

# ---- serve -------------------------------------------------------------------
# -sm tensor: +59.5 % over the default layer split at ctx 16,384 and +65.4 % at
# 147,456, same residency ceiling. EXPERIMENTAL in llama.cpp's own help.
# -ts is COMPUTED above. Leaving it unset makes llama.cpp split evenly across a
# 12 GB card and a 16 GB one, which is what produced 0.38 tok/s.
# Assembled as an ARRAY so -WhatIf can print exactly what would run. Built once
# and either printed or splatted -- a preview that reconstructs the argv
# separately is how the two stop agreeing, which this repository already says
# about serve.ps1 in its own -WhatIf block.
# n-match is a property of the ARTIFACT, not of the depth. 12 won on
# UD-Q4_K_XL; on the NVFP4 file 24 is +27.1 % RESOLVED over it, and 24 is the
# value that LOST on UD-Q4_K_XL at this exact depth. Two artifacts, two answers,
# both measured at 147,456.
$nMatch = if ($Nvfp4) { '24' } else { '12' }
# n-min 16 / n-max 32. llama.cpp's defaults are 48 / 64 and Studio never sets
# them, so we are the ones deviating -- 16/32 came through an older sweep where
# they were "held constant" rather than chosen, and that is still true.
#
# -Beta carried 48/64 for one afternoon and it was REVERTED WITHOUT A VERDICT.
# Both sessions that ran it recorded `ngram-mod: #gen drafts = 0` on either
# side: the n-gram never fired once on agent traffic, so the change was inert
# rather than better or worse. An inert deviation inside a bundle makes the
# bundle harder to reason about for nothing.
#
# NOT THE SAME AS THE DRAFT DEPTH, which was tried and LOST. This one was never
# exercised, and what would exercise it is a workload where an n-gram fires at
# all -- which this project does not have.
#
# n-max: 32 -> 64 on 2026-09-02, issue #67. The paragraph above stays because it
# is the honest history of the value, but it is no longer the reason for it.
# 64 IS llama.cpp's default; 32 was ours and was never chosen. Measured at the
# SERVED 147,456 on the real-code corpus, three independent boot series:
# +15.63 %, +14.85 %, +14.52 %. 96 and 128 fall back to about +2 %, so 64 is a
# peak rather than a direction. Costs nothing: free_after 2,586-2,600 MiB and
# 66+0 in every row.
#
# n-min STAYS 16. Studio's 48/64 is a pair and the pair loses -- 48/64 measured
# -10.58 % in the same run. The gain is n-max alone.
#
# AND THE DRAFT DEPTH STAYS 3. `--spec-draft-n-max 4` is +5.76 % by itself at
# this depth, but with n-max 64 the two measure -4.61 % TOGETHER, worse than
# changing neither, at 0.1-0.5 % spreads. Two windows widening in one cascade
# starve each other. `test_the_default_serves_n_max_64_and_leaves_the_draft_depth_at_3`
# fails if a later session adopts the second winner because the first one worked.
#
# WHAT IS STILL UNMEASURED, and it is the reason this could disappoint: the gain
# is on a corpus where the n-gram FIRES. On agent traffic this profile recorded
# `#gen drafts = 0`. If the drafter never fires in real use, the window it draws
# from cannot matter.
$nMin = '16'
$nMaxG = '64'
# Images or not. `--no-mmproj-auto` and `-mm` together is a contradiction for
# whoever reads the command line next, so it is one or the other.
$visionArg = if ($Vision) { @('-mm', $MMPROJ) } else { @('--no-mmproj-auto') }

# The -Lean bundle. Empty by default: it is UNMEASURED and must not leak.
#
# --ctx-checkpoints 0 LEFT THE BUNDLE 2026-08-29, MEASURED. It was copied from
# Studio beside --cache-ram 0 as one memory decision; they are not one decision.
# This artifact is hybrid -- Gated DeltaNet recurrent state beside attention KV
# -- and the recurrent half cannot be rewound to a shared prefix. With no
# checkpoint to restore from, llama.cpp abandons the whole prompt and says so
# once per request:
#
#   forcing full prompt re-processing due to lack of cache data
#   (likely due to SWA or hybrid/recurrent memory)
#
# serve-20260829-125227.log served THREE requests and printed it on all three:
# 17,881 tokens, then 46,998, then 46,997 -- the last two the same conversation
# read again from the first token, 51.6 s each before a character came back.
# The same binary and artifact with checkpoints at their default
# (serve-20260829-073741.log) printed it ONCE in a whole session and prefilled
# 13, 29, 285, 829, 1,358 tokens per turn instead.
#
# The default costs 150.89 MiB per checkpoint, at most 32, no closer together
# than 8,192 tokens -- about six at the depth we serve, in HOST RAM.
#
# --cache-ram 0 STAYS, because it is a different mechanism: the host store for
# whole prompts that have been evicted, which is what carries a conversation
# across a slot change rather than across a turn. Whether to restore its 8,192
# MiB default is still the developer's open question.
# `--no-kv-unified`, NOT the absence of `--kv-unified`. MEASURED 2026-08-30 and
# the first version of this switch was wrong.
#
# Removing the flag looked sufficient -- `--help` says "default: enabled if
# number of slots is auto", and this profile always passes `-np 1`. The boot log
# of the run that was supposed to be testing the removal says
#
#     llama_context: kv_unified            = true
#
# so the arm measured the same setting as the arm it was compared against, and
# nothing said so. This is the `--fit on` fault exactly (CORRECTIONS 33): a flag
# whose default may be ON cannot be turned off by deleting it. The negative form
# is the only form that means anything.
$betaArg = if ($Beta) {
    @('--cache-ram', '0', '--load-mode', 'none', '--metrics') +
    $(if ($NoKvUnified) { @('--no-kv-unified') } else { @('--kv-unified') })
} else { @() }
$threads = if ($Beta) { '2' } else { '18' }

# HOW THINKING IS TURNED ON, and the two profiles do it differently on purpose.
#
# Ours: a template FILE plus --reasoning-effort medium. Neither the file's
# reason for existing nor the choice of `medium` is written down anywhere in
# this repository, and `medium` is recorded in our own docs as NEVER MEASURED
# on any artifact.
#
# Studio's: no template file at all -- the one inside the GGUF, steered with
# --chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}.
# `preserve_thinking` is `--reasoning-preserve` on our side, a flag we do not
# set and which our own boot log suggests: "chat template supports preserving
# reasoning, consider enabling it via --reasoning-preserve".
#
# -Lean borrows theirs whole, which is the only way to find out whether ours is
# doing anything. The JSON must stay ONE argv entry: split across several it is
# not valid JSON and llama.cpp rejects it.
#
# AND THEIR EXACT FLAGS ARE NOT THE ONES TO COPY. Passing their
# `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}`
# boots and thinks, and the log answers back twice:
#
#   W Setting 'enable_thinking' via --chat-template-kwargs is deprecated.
#     Use --reasoning on / --reasoning off instead.
#   I chat template supports preserving reasoning, consider enabling it via
#     --reasoning-preserve
#
# One is deprecated and THE OTHER DOES NOTHING -- the server still asks for
# --reasoning-preserve after being handed preserve_thinking. Copying a command
# line from a different build copies its bugs, so this uses the flags this
# binary actually wants. It also removes a JSON blob that had to survive
# PowerShell and then cmd intact.
#
# --reasoning-effort IS NOT OPTIONAL HERE, and -Beta shipped without it for one
# afternoon. Studio hands its own effort per REQUEST (`reasoningEffort:
# "medium"` in both n-max threads); we have no client that does, so dropping the
# flag hands the decision to the chat template, whose default is xhigh. The
# served boot log said so in as many words -- "Reasoning effort is set to
# xhigh" (serve-20260829-125227.log:298) -- while decode was healthy, which is
# why it read as "the server feels slower" rather than as a fault. Report 35
# measured four real tasks at 537-1,019 s under that default.
$thinkArg = if ($Beta) {
    @('--reasoning', 'on', '--reasoning-preserve', '--reasoning-effort', 'medium')
} else {
    @('--reasoning-effort', 'medium')
}

# THE TEMPLATE IS NOT PART OF THE THINKING MECHANISM, and bundling it into the
# either/or above cost five hub icons -- 7, 8, 9, A, B -- every Claude Code
# request, fifteen 500s in a row in logs/serve-20260831-023636.log before the
# client gave up (issue #58). It is a CLIENT-COMPATIBILITY fix: Studio omits it
# safely because Studio never sends a late system message, and we do.
$templateArg = if ($StockTemplate) { @() } else {
    @('--chat-template-file', $TEMPLATE_FILE)
}
# --alias is the model name every caller sees on /v1/models and in each
# response. Left hardcoded it would announce Q4_K_XL while serving the NVFP4
# file -- the same fault as CORRECTIONS 34 one layer out, and visible to clients
# rather than only to a reader of the raw results.
$alias = if ($Nvfp4) { 'Qwen3.8-27B-NVFP4-MTP' } else { 'Qwen3.8-27B-Q4_K_XL' }
# Empty under -Beta: a parameter for a decoder that is not loaded is a flag
# that does nothing and a reader who believes it did -- the same fault as the
# inert `--fit on` this profile carried for weeks.
$ngramArg = if ($Beta) { @() } else {
    @('--spec-ngram-mod-n-match', $nMatch,
      '--spec-ngram-mod-n-min', $nMin, '--spec-ngram-mod-n-max', $nMaxG)
}

$argv = @(
    '-m', $Model,
    '--alias', $alias, '-c', "$Ctx",
    # `--fit off`, not `--fit on --fit-target 768`, and not silence.
    #
    # Fitting is MEASURED INERT here: llama.cpp prints `llama_params_fit is not
    # implemented for SPLIT_MODE_TENSOR, abort` on every boot and this profile
    # is always -sm tensor. The first attempt at this simply DELETED the flag --
    # and booting it showed the warning still there, because `--fit` defaults to
    # ON (`--fit [on|off] ... default: 'on'`). Deleting it was a no-op dressed as
    # a cleanup. Turning it off is the honest version, and it is what Unsloth
    # Studio passes.
    #
    # `--fit-target 768` goes with it: a margin for a step that never runs.
    '-ngl', 'auto', '--fit', 'off', '-fa', 'on', '-np', '1',
    '-sm', 'tensor'
) + $tsArg + @(
    '-t', $threads, '-b', '2048', '-ub', "$UBatch", '-lv', "$Verbosity",
    '--log-colors', $LogColors
) + $logFileArg + @(
    '-ctk', 'q4_0', '-ctv', 'q4_0'
) + $specArg + @(
) + $ngramArg + $visionArg + $betaArg + $thinkArg + $templateArg + @(
    '--sse-ping-interval', "$SsePingIntervalSec",
    '--host', $BindAddress, '--port', "$Port"
)

# THE CLONE REPLACES THE WHOLE COMMAND LINE, rather than patching the one above.
# Patching would leave every value this profile computes silently in play, and
# the point of a baseline is that a reader can see all of it in one place.
#
# SIX THINGS ARE DELIBERATELY NOT COPIED. A literal copy reproduces their bugs
# and breaks the comparison:
#
#   --chat-template-file        ADDED 2026-08-31, issue #58, for the same reason
#                               as --reasoning-effort below. Studio omits the
#                               file safely because Studio's client never sends
#                               a system message after the user turn; Claude
#                               Code sends one every session and Qwen3.8's own
#                               template RAISES on it. This branch answered HTTP
#                               500 to every request until $templateArg was
#                               appended to it. -StockTemplate omits it on
#                               purpose, which is the only way it should ever
#                               happen.
#
#   --reasoning-effort medium   ADDED. Not on their command line because Studio
#                               sends it in every request body. No client of
#                               ours does, so copying the OMISSION serves at the
#                               template's xhigh -- CORRECTIONS 36, exactly.
#   --reasoning on
#   --reasoning-preserve        INSTEAD of --chat-template-kwargs {...}: this
#                               build answers that kwarg with "deprecated" and
#                               then asks for --reasoning-preserve anyway.
#   --alias                     OURS. The alias is the model name a client asks
#                               for; changing it changes the client too, and
#                               then the A/B has two variables.
#   -lv                         OURS. `forcing full prompt re-processing` and
#                               `cached n_tokens` do not print at their
#                               verbosity 3, and those lines are the reason to
#                               run this at all.
#   --host/--port               OURS. Studio picks a random port per launch.
if ($Clone) {
    $slotDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'logs\llama-slots'
    # [void] and not `| Out-Null`: this block sits inside the invocation
    # region, and the guard that keeps anything from standing between
    # llama.cpp and the console scans that region for a pipe. The pipe
    # here would have been harmless and the guard would have been right to
    # be suspicious, so the code moves rather than the test.
    if (-not (Test-Path $slotDir)) { [void](New-Item -ItemType Directory -Path $slotDir -Force) }
    $argv = @(
        '-m', $Model,
        '--alias', $alias,
        '-np', '1', '-fa', 'on', '--no-context-shift',
        '-c', "$STUDIO_CTX",
        '-ngl', '-1', '--fit', 'off', '--metrics',
        '--slot-save-path', $slotDir,
        '-t', '2', '--jinja',
        '-ctk', 'q4_0', '-ctv', 'q4_0',
        '-sm', 'tensor', '-ts', $STUDIO_TS,
        '-b', '2048', '-ub', '512',
        '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2'
    ) + $visionArg + @(
        '--cache-ram', '0', '--ctx-checkpoints', '0', '--load-mode', 'none',
        '-lv', "$Verbosity", '--log-colors', $LogColors
    ) + $logFileArg + @(
        '--reasoning', 'on', '--reasoning-preserve', '--reasoning-effort', 'medium',
        '--sse-ping-interval', "$SsePingIntervalSec",
        '--host', $BindAddress, '--port', "$Port"
    ) + $templateArg
}

# THE STRUCTURAL HALF OF ISSUE #58, and the only half that stops a third
# recurrence. Hoisting $templateArg fixes the two branches that broke; this
# reads the FINAL argv, so a branch written later -- $Clone rebuilds it from
# scratch, and nothing stops another from doing the same -- fails loudly here
# instead of serving HTTP 500 to every request until somebody reads a log.
#
# IT DOES NOT PREVIEW UNDER -WhatIf, AND THE OTHER FATAL IN THIS FILE DOES.
# That difference is deliberate and it is the only reason this comment exists,
# because from the code the two look like one of them is an oversight (#65).
# The `-ts` budget FATAL reports the ENVIRONMENT -- busy cards, true this minute
# and false the next -- so previewing anyway is useful and a test depends on it.
# This one reports a CODING DEFECT: an argv that lost a flag is never fine, and
# a preview of a command line nobody may run is not worth printing. What it owes
# the reader instead is the array it actually built, so the defect is diagnosable
# from the refusal alone.
if (-not $StockTemplate -and ($argv -notcontains '--chat-template-file')) {
    Write-Host "FATAL: the command line lost --chat-template-file." -ForegroundColor Red
    Write-Host "  Without it Qwen3.8's own template RAISES on a system message that" -ForegroundColor Yellow
    Write-Host "  arrives after the user turn, which is what Claude Code sends, and" -ForegroundColor Yellow
    Write-Host "  every request returns HTTP 500. See issue #58 and #4." -ForegroundColor Yellow
    Write-Host "  Pass -StockTemplate if the omission is what you meant." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  what was actually built:" -ForegroundColor Yellow
    Write-Host "    $Exe $($argv -join ' ')"
    exit 1
}

if ($WhatIfPreference) {
    Write-Host ""
    Write-Host "WhatIf: would run" -ForegroundColor Green
    Write-Host "  $Exe $($argv -join ' ')"
    exit 0
}

& $Exe @argv
