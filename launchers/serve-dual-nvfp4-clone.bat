@echo off
REM ============================================================================
REM  Start the worker  --  UNSLOTH STUDIO'S COMMAND LINE, ON OUR BINARY, loopback only
REM
REM  A BASELINE. NOT A RECOMMENDATION. NOT FOR SERVING.
REM
REM  Its window is -c 107,899. This machine exists to serve 200,704. Run this to
REM  learn something, then go back to icon 1 or 2.
REM
REM  WHY IT EXISTS
REM
REM  By 2026-08-30 our server and Studio's differed in eight flags and every one
REM  of them had a plausible story:
REM
REM      ours                        Studio
REM      -c 200704                   -c 107899
REM      -ts 7598,15288  (33/67)     -ts 7648,13509  (36/64)
REM      -ub 1024                    -ub 512
REM      -ngl auto                   -ngl -1
REM      --kv-unified                (unset)
REM      --ctx-checkpoints 32        --ctx-checkpoints 0
REM      --spec-draft-n-max 3        --spec-draft-n-max 2
REM      (unset)                     --slot-save-path, --jinja
REM
REM  Testing them one at a time is eight boots before the first answer. THIS IS
REM  ONE BOOT that says whether the remaining gap is in that list at all. If it
REM  reproduces Studio's numbers, bisecting is worth doing. If it does not, the
REM  cause is somewhere none of these flags reach, and eight sweeps would have
REM  found nothing.
REM
REM  WHAT THE GAP LOOKS LIKE, measured 2026-08-29 and 08-30, same machine, same
REM  model file, Discord streaming through both:
REM
REM                            Unsloth Studio        ours (icon 7)
REM      prefill               728 - 1,000 tok/s     319 - 633
REM      decode at ~47,000     36.63                 26.93 - 29.02
REM      mean accepted length  1.81 - 2.52           2.51 - 2.81
REM
REM  Our drafting is BETTER on every request and we are still slower, and the
REM  decode gap is a level shift rather than a widening slope -- about 10 tok/s
REM  at every depth. That points at a fixed per-token cost, which under
REM  -sm tensor means the KV layout and the exchange between the two cards.
REM
REM  WHAT IS DELIBERATELY *NOT* COPIED, because a literal copy reproduces their
REM  bugs and breaks the comparison:
REM
REM      --reasoning-effort medium   ADDED. It is not on their command line
REM                                  because Studio sends it in every REQUEST.
REM                                  No client of ours does, so copying the
REM                                  omission serves at the template's xhigh.
REM                                  That is how CORRECTIONS 36 happened.
REM      --chat-template-file        ADDED, 2026-08-31, and for the SAME reason.
REM                                  Studio omits it safely because Studio never
REM                                  sends a system message after the user turn.
REM                                  Claude Code sends one every session, and
REM                                  Qwen3.8's own template RAISES on it -- this
REM                                  icon answered HTTP 500 to every request
REM                                  until it was added. Issue #58 and #4.
REM                                  Pass -StockTemplate to omit it on purpose.
REM      --reasoning on
REM      --reasoning-preserve        instead of --chat-template-kwargs, which
REM                                  this build calls deprecated and then asks
REM                                  for --reasoning-preserve anyway.
REM      --alias                     OURS, so the client config does not have to
REM                                  change and the A/B keeps one variable.
REM      -lv 4                       OURS. `forcing full prompt re-processing`
REM                                  does not print at their verbosity 3, and
REM                                  that line is the reason to run this.
REM
REM  AND ONE THING NO FLAG WILL CLOSE. Studio answered a question about a
REM  117 KB document in under 4 seconds because it never sent the document: it
REM  indexes files and retrieves 5 chunks, so the model read 1,942 new tokens
REM  where we sent 46,998. That is a CLIENT feature, not a server setting, and
REM  it is most of the difference you feel. This baseline cannot test it.
REM
REM  -c 107,899 and -ts 7648,13509 are frozen values read from the Studio server
REM  running at 00:11 on 2026-08-30. Studio recomputes both from free VRAM at
REM  every launch, exactly as our profile does, so they are a snapshot of one
REM  boot rather than a constant of theirs -- frozen on purpose, because a
REM  baseline that recomputed them would not be the same baseline twice.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Vision -Clone
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
