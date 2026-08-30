@echo off
REM  Qwen3.8-27B NVFP4 + DFlash2 + ngram-mod, on UNSLOTH'S 0.3.0 SOURCE.
REM
REM  UNVERIFIED. This binary has NEVER been seen to load DFlash2 under
REM  -sm tensor. It was built while the cards were busy serving, so the one
REM  thing it exists to prove has not been proven. If it aborts at
REM  ggml-backend-meta.cpp:543 the patch did not take, and that is the
REM  expected shape of failure -- loud, immediate, not a slow wrong answer.
REM
REM  WHAT IT IS. Unsloth ship llama.cpp 0.3.0 (b10679-mix-67dfc8b) and their
REM  binary CANNOT do this: asked for DFlash2 under -sm tensor it aborts,
REM  measured 2026-08-30. So their source was copied out of
REM  %%USERPROFILE%%\.unsloth -- which is never touched, Studio runs it -- the
REM  mirror patch applied, and the tree built here with ARCHS 89;120 only.
REM
REM  THERE ARE FOUR llama-server BINARIES ON THIS MACHINE. This one is the
REM  fourth, and telling them apart matters:
REM
REM      llama.cpp-blackwell        10499   unpatched   the served default
REM      llama.cpp-mirror           10499   PATCHED     the other DFlash2 icons
REM      %%USERPROFILE%%\.unsloth   10679   unpatched   icons A and B
REM      llama.cpp-unsloth-mirror   10679   PATCHED     THIS ONE
REM
REM  READ THE BANNER CAREFULLY. It says
REM      version: 0.3.0-dev (build 215, commit ...) Compiled by the Unsloth team
REM  The 0.3.0-dev and the Unsloth line are THEIRS. The build number and commit
REM  are OUR repository's git, counted by their build system because the copied
REM  tree has no .git. A log from this binary is NOT a log from 10499, and it is
REM  not 10679's shipped build either.
REM
REM  WHY IT MIGHT BE WORTH IT, AND WHY IT MIGHT NOT. Their build was read as
REM  +26 percent faster once, from one boot per side, and that has never been
REM  paired -- it is CONTESTED, not settled. Against it: our 10499 resolves
REM  fused Gated Delta Net (chunked AND autoregressive) and the Lightning
REM  Indexer, and 10679 resolves NEITHER, on the same model, in every log we
REM  have. Qwen3.8 IS a Gated DeltaNet hybrid. So this may well be slower.
REM
REM  Everything else matches serve-dual-nvfp4-dflash.bat exactly -- same
REM  artifact, same n-match 24, same n_max 4, same tensor split, ctx 147,456 --
REM  so a difference between the two icons has one cause: the binary.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root. Double-clicking from a shortcut does not
REM  put %CD% here, and a relative path would resolve against the wrong place.
REM
REM  This one binds 127.0.0.1 only. Nothing outside this machine reaches it.
REM
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Vision -Dflash -TheirMirror
if errorlevel 1 pause
