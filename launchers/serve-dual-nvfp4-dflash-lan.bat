@echo off
REM  Qwen3.8-27B NVFP4 on BOTH cards, drafted by DFLASH2 instead of its own head.
REM
REM  READ THIS FIRST: IT IS NOT FASTER.
REM
REM  Measured 2026-08-30 at the depth this serves, three rounds against the
REM  baked-in MTP head's six rounds over two boot series
REM  (results/nvfp4-dflash-147456-n4.jsonl):
REM
REM      draft-dflash + ngram   44.48 / 44.56 / 44.23    spread 0.7 percent
REM      draft-mtp    + ngram   39.43 / 42.61 / 42.55
REM                             43.10 / 42.99 / 42.93    spread 9.3 percent
REM
REM  +4.0 percent on medians. That is UNDER the 13.6 percent noise floor this
REM  project applies, and the two sides ran in different boots. It is NOT a
REM  measured speedup and must not be quoted as one.
REM
REM  WHAT IT ACTUALLY BUYS: STEADINESS. Spread 0.7 percent against 9.3. Every
REM  one of its three rounds came in above every one of the head's six. If you
REM  care about a predictable rate more than a peak, that is the reason.
REM
REM  WHAT IT COSTS: about 950 MiB of headroom. It finishes with 1,450 MiB free
REM  against 2,400 for serve-dual-nvfp4.bat. A run measured here died with 336
REM  MiB free and survived with 488, so 1,450 is comfortable -- but it is the
REM  smallest margin of any NVFP4 icon.
REM
REM  IT ALSO NEEDS THE PATCHED BINARY. llama.cpp-mirror, built from a local
REM  patch that mirrors the output projection so TOP_K can read logits split
REM  across two cards. Reviewed by nobody outside this project.
REM
REM  DEPTH: 147,456, the deepest MEASURED with this pairing. serve-dual-nvfp4
REM  -deep.bat reaches 200,704 with the baked-in head; that has NOT been tried
REM  with DFlash2.
REM
REM  At 65,536 the same pairing is +67.9 percent over ngram-mod alone, RESOLVED.
REM  The comparison above is against the HEAD, not against no drafter at all.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root. Double-clicking from a shortcut does not
REM  put %CD% here, and a relative path would resolve against the wrong place.
REM
REM  THIS ONE IS EXPOSED. -Lan binds every interface, and --host is the only
REM  access control this server has: no API key, CORS is open. Clicking it is
REM  the same act as typing the flag.
REM
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Vision -Dflash -Lan
if errorlevel 1 pause
