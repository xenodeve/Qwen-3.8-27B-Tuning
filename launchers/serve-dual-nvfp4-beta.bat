@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, NVFP4, images, THE UNSLOTH BUNDLE (BETA), loopback only
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output. Ctrl+C stops it; so does closing this window.
REM
REM  THIS IS serve-dual-nvfp4-deep.bat WITH SIX SETTINGS BORROWED FROM UNSLOTH
REM  STUDIO, which runs this same model file on these same two cards:
REM
REM      --cache-ram 0          prompt cache off
REM      --ctx-checkpoints 0    context checkpoints off
REM      --load-mode none       no memory-mapped read
REM      --kv-unified           one shared KV buffer
REM      --metrics              Prometheus endpoint
REM      -t 2                   two threads instead of eighteen
REM      --reasoning on         thinking, their way rather than ours
REM
REM  TWO MORE WERE IN HERE AND ARE NOT ANY MORE, for different reasons:
REM
REM  --spec-draft-n-max 2 WAS IN THIS BUNDLE AND WAS TAKEN OUT THE SAME DAY.
REM  Studio documents 2 for MTP on a GPU. On this machine it lost, and the
REM  server's own counters say why -- not the rate, which was a cross-session
REM  comparison, but the mechanism:
REM
REM      n-max 3    297 drafts ->   891 tokens = 3 per draft, accepted len 2.80
REM      n-max 2    887 drafts -> 1,774 tokens = 2 per draft, accepted len 2.12
REM
REM  The acceptance RATE hardly moved, 0.60 to 0.54. The accepted LENGTH fell
REM  24 %, so every verify step advances less far. Decode read 43-45 tok/s
REM  before and 25-33 after. A default from another product is still a verdict
REM  from another configuration.
REM
REM  n-min 48 / n-max 64 went the same way, but NOT for the same reason. Those
REM  are llama.cpp's defaults and Studio never sets them, so our 16 / 32 is the
REM  deviation -- and it still is. They were reverted because BOTH runs above
REM  logged `ngram-mod: #gen drafts = 0`. The n-gram never fired once, so the
REM  change was inert: not better, not worse, NEVER EXERCISED. Keeping an inert
REM  deviation inside a bundle makes the bundle harder to read for nothing.
REM
REM  So this icon now differs from serve-dual-nvfp4-deep.bat in six ways, all of
REM  them about memory, threads and how thinking is switched on -- and in no
REM  decoder value at all.
REM
REM  Everything else is identical to serve-dual-nvfp4-deep.bat: same file, the
REM  same 200,704 window, same speculative head, same n-gram at n-match 24,
REM  same micro-batch. That is deliberate -- the two icons are an A/B you can
REM  run, and 200,704 is the depth this machine actually serves.
REM
REM  THE SPEED IS UNMEASURED AS A PAIRED COMPARISON. What exists is one boot
REM  per side, below, and one boot per side is not a measurement here.
REM
REM  Taken at 200,704, a 91,428-token prompt then 512 tokens generated, then a
REM  picture on top:
REM
REM                                serve-dual-nvfp4-deep.bat    this one
REM      decode                            53.69 tok/s        135.25 tok/s
REM      prefill                              816.6              824.1
REM      host memory                 19.42 GB working set   3.21 GB working set
REM      free VRAM after the image      555 / 1,186 MiB     556 / 1,332 MiB
REM
REM  DO NOT TREAT THE DECODE NUMBER AS A RESULT. It is ONE reading per side,
REM  taken in different boots, and this project has measured the same arm
REM  drifting 48.9 percent across boots at depth. +152 percent is far outside
REM  that, which is interesting, not proven. The paired sweep that would settle
REM  it is `--arms lean-bundle` and it has not been run.
REM
REM  If the gain is real, the likely reason is context checkpoints: a
REM  91,428-token prompt means roughly eleven of them, at about 350 MiB each,
REM  copied to host memory WHILE the model is generating.
REM
REM  THE MEMORY IS THE SOLID PART. 19.42 GB against 3.21 GB of working set at
REM  the same depth on the same prompt. Private memory moved much less, 34.84
REM  against 31.98 GB, so what collapses is resident pages rather than
REM  commitment.
REM
REM  ONE MORE THING WORTH KNOWING: the two sides did not produce the same text.
REM  Shown a purple picture with an ORANGE square, this one said orange and the
REM  default said yellow. That is a single instance and could be luck, but
REM  --kv-unified changes how the KV cache is laid out, and this project has
REM  already recorded that changing the split changes the output.
REM
REM  IT IS STILL NOT A FREE SAVING. Those context checkpoints were doing work:
REM  the log of a real eighteen-minute session shows them being RESTORED at
REM  positions 47,940 to 50,091. Turning them off means a conversation that
REM  rewinds must re-process the prompt instead -- roughly a minute per 50,000
REM  tokens. The measurement above is a single long request, which is the case
REM  checkpoints cost the most and help the least. A session that edits and
REM  re-asks is the case they were built for, and this says nothing about it.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root.
REM
REM  This one binds 127.0.0.1 only. Nothing outside this machine reaches it.
REM ============================================================================

setlocal
cd /d "%~dp0.."

where pwsh >nul 2>nul
if errorlevel 1 (
    echo.
    echo PowerShell 7 ^(pwsh^) was not found, and this needs it.
    echo Install it with:  winget install Microsoft.PowerShell
    echo.
    pause
    exit /b 1
)

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Deep -Vision -Beta
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
