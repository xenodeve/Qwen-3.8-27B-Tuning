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
$WEIGHTS_MIB = 16130
$KV_KIB_PER_TOKEN = 18.0
# One compute buffer per card, and it tracks -ub. 1,024.30 MiB each at -ub 1024,
# read from the boot log.
$COMPUTE_MIB = 2 * [Math]::Max(256, $UBatch)
# The nextn head is not free. Measured: about 2,750 MiB more across the two
# cards with -Mtp on, and CUDA1 finished with 861 MiB free.
$MTP_HEAD_MIB = 2750
if ($Mtp) { $WEIGHTS_MIB += $MTP_HEAD_MIB }

$kvMib     = [int](($Ctx * $KV_KIB_PER_TOKEN) / 1024)
$demandMib = $WEIGHTS_MIB + $kvMib + $COMPUTE_MIB
$total = ($budgets | Measure-Object -Sum).Sum
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
    if ($fitsCtx -ge 4096) {
        Write-Host ("    the deepest context this budget holds is about {0:N0}" -f $fitsCtx) -ForegroundColor Cyan
        Write-Host ("    try:  -Ctx {0}" -f $fitsCtx) -ForegroundColor Cyan
    } else {
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
    exit 1
}

$tsArg = @('-ts', ($budgets -join ','))

# The decoder. ngram-mod alone is what has a measured rate here; -Mtp adds the
# baked-in head beside it, which runs but whose rate the guard would not accept.
$specArg = if ($Mtp) {
    @('--spec-type', 'draft-mtp,ngram-mod', '--spec-draft-n-max', '3')
} else {
    @('--spec-type', 'ngram-mod')
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
& $Exe -m $Model `
    --alias Qwen3.8-27B-Q4_K_XL -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -sm tensor `
    @tsArg `
    -t 18 -b 2048 -ub $UBatch --no-mmproj-auto -lv $Verbosity `
    --log-colors $LogColors `
    @logFileArg `
    -ctk q4_0 -ctv q4_0 `
    @specArg `
    --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 `
    --chat-template-file "C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja" `
    --reasoning-effort medium `
    --sse-ping-interval $SsePingIntervalSec `
    --host $BindAddress --port $Port
