@echo off
REM  Qwen3.8-27B UD-Q4_K_XL on BOTH cards, DFlash2 drafter at DRAFT DEPTH 4.
REM
REM  This is serve-dual-dflash.bat with one number changed:
REM  --spec-draft-n-max 4 instead of the 2 that launcher serves.
REM
REM  WHY 4. Measured 2026-08-30, three paired rounds rotated, ctx 65,536, real
REM  vendor code, on the patched mirror -- results/tensor-draft-depth-65536.jsonl
REM  and issue #56:
REM
REM      ngram-mod                27.27 / 26.72 / 26.31    med 26.72
REM      draft-mtp,ngram-mod n3   38.13 / 37.64 / 36.39    med 37.64
REM      draft-dflash n4          57.42 / 54.81 / 55.72    med 55.72
REM      draft-dflash n7          52.69 / 52.64 / 51.58    med 52.64
REM
REM  4 is +109.2 percent over ngram-mod, RESOLVED, and beats draft-mtp by
REM  +49.8 percent. 7 -- the clamp -- is 6.5 percent WORSE than 4 in every
REM  round and 308 MiB dearer, so the ceiling is not where the speed is.
REM
REM  WHAT IT COSTS ON TOP OF serve-dual-dflash.bat. The recurrent state is
REM  149.62 MiB x (1 + n_max), so 2 -> 4 spends about 299 MiB more. At ctx
REM  65,536 the n4 arm finished with 479 MiB free across both cards.
REM
REM  THE ONE THING THIS LAUNCHER CANNOT TELL YOU. Every number above is ctx
REM  65,536. This serves 131,072, and 4 has NEVER been measured there -- only 2
REM  has, finishing with 634/530 MiB per card. 299 MiB of that is what raising
REM  the depth spends. A run measured here died with 336 MiB free and survived
REM  with 488, so this is inside the band where it matters. If a long request
REM  dies on this launcher, drop back to serve-dual-dflash.bat and say so.
REM
REM  Everything else is serve-dual-dflash.bat's: the patched llama.cpp-mirror
REM  binary that nobody outside this project has reviewed, and the 131,072
REM  window against about 250,000 from serve-dual.bat.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root. Double-clicking from a shortcut does not
REM  put %CD% here, and a relative path would resolve against the wrong place.
REM
REM  THIS ONE IS EXPOSED. -Lan binds every interface, and --host is the only
REM  access control this server has: no API key, CORS is open. Clicking it is
REM  the same act as typing the flag.
REM
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Dflash -DflashN 4 -Lan
if errorlevel 1 pause
