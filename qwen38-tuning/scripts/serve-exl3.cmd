@echo off
rem THE ONE PLACE THE EXL3 RECIPE IS WRITTEN (issue #71, results 10). The hub and
rem the launchers under launchers\ pass at most three positional knobs and no
rem flag of their own; a copy of `-cq 4 -ndt 3` anywhere else is a second source
rem of truth, and this project has shipped one of those before.
rem
rem   serve-exl3.cmd [cache_tokens] ["gs_4070,gs_5060"] [host]
rem     QUOTE the split caps: cmd splits an unquoted comma into two arguments,
rem     and `10,15.5` bare became `-gs 10` (one card) on 2026-09-04.
rem     cache  default 163840  -- the depth every row in results 10 was taken at
rem     gs     default 9,15.5  -- VRAM caps per card in GB; 10 on the 4070 OOMs the 4.0bpw file at depth
rem     host   default 127.0.0.1; 0.0.0.0 exposes it on the LAN (no API key)
rem
rem Recipe: -cq 4 (upstream integer KV, graph-captured; NVFP4/fp8 KV are not),
rem -dm mtp (the head in the file), -ndt 3, -tp -tpb native (tensor-parallel,
rem no NCCL). 31.9-36.9 tok/s at 147K paired against llama.cpp's 40.1-44.3;
rem 47-55 in real 30-70K sessions. Port 8000 so llama-server keeps 8080.
rem Log: qwen38-tuning\logs\exl3-serve-<stamp>.log (UTF-8 via pwsh 7: Windows PowerShell 5.1's Tee-Object has no -Encoding and its default wrote UTF-16, 2026-09-04). The server speaks the
rem OpenAI API plus the Anthropic Messages API (/v1/messages, issue #73), so
rem claude-xeno-exl3 talks to it directly on :8000; no proxy.
setlocal
set CS=%~1
if "%CS%"=="" set CS=163840
set GS=%~2
if "%GS%"=="" set GS=9,15.5
set HOST=%~3
if "%HOST%"=="" set HOST=127.0.0.1
rem The artifact. Since 2026-09-04 the default is turboderp's SC 4.00bpw H5 (VENDOR KL
rem 0.0062; the previous Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw has no KL figure anywhere).
rem EXL3_MODEL_DIR overrides it, e.g. back to C:\AI\models\Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw.
set MODEL=%EXL3_MODEL_DIR%
if "%MODEL%"=="" set MODEL=C:\AI\models\turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5
rem Relaunch loop (issue #75). The server comes back after ANY exit that was not asked
rem for: a TP sync timeout (the server writes exl3-restart.flag with the reason and exits),
rem a crash with no Python exception, or the self-probe finding /health deaf. Stopping on
rem purpose = stop-exl3.cmd, which writes exl3-stop.flag first. cmd cannot read python's
rem exit code through the Tee pipe, so files are the signal. Three exits within 420 s of
rem their start stop the loop, 420 s because a model load alone is 2-4 min (a missing model or a
rem cache that does not fit would otherwise
rem relaunch every 5 s forever). Each pass stamps its own log.
set FLAG=C:\AI\qwen38-tuning\logs\exl3-restart.flag
set STOP=C:\AI\qwen38-tuning\logs\exl3-stop.flag
if exist "%FLAG%" del "%FLAG%"
if exist "%STOP%" del "%STOP%"
set FAST=0
:again
rem A stop asked for between passes (during the 5 s sleep or the :8000 check below) found no
rem python to kill and only left the flag; the flag used to be read after the pipeline, so the
rem loop relaunched anyway (review 2026-09-06). Read it here first.
if exist "%STOP%" (
    del "%STOP%"
    echo == EXL3 serve: stopped on purpose ^(exl3-stop.flag^) before relaunch, not starting.
    goto end
)
rem 2026-09-05 09:32-10:43: a relaunch that could not bind :8000 (another server held it)
rem loaded the whole model, died, and was relaunched ten times; each pass took ~4 min so the
rem fast-death guard never saw it. If something already answers on :8000 this loop is not
rem the server that should come back.
curl -s -m 2 http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 (
    echo == EXL3 serve: something already answers on :8000, not starting a second server.
    exit /b 2
)
for /f "usebackq" %%i in (`powershell -NoProfile -Command "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"`) do set T0=%%i
rem Invariant culture: a Thai-culture PowerShell stamped 2569xxxx (Buddhist year) on 2026-09-04.
for /f "usebackq" %%i in (`powershell -NoProfile -Command "[DateTime]::Now.ToString('yyyyMMdd-HHmmss',[Globalization.CultureInfo]::InvariantCulture)"`) do set STAMP=%%i
set LOG=C:\AI\qwen38-tuning\logs\exl3-serve-%STAMP%.log
cd /d C:\AI\exllamav3-mia
set PYTHONIOENCODING=utf-8
echo == EXL3 serve: model %MODEL%, cache %CS% tokens, split caps %GS% GB, host %HOST%, log %LOG%
rem The server is OURS: qwen38-tuning\serving\exl3\server.py = the fork's tools\serve_openai.py
rem plus marked hooks, with the custom parts in sibling modules (see its README). The fork
rem tree stays pristine so it can be pulled; EXL3_FORK_DIR overrides where it is.
C:\AI\exllama3-venv\Scripts\python.exe C:\AI\qwen38-tuning\serving\exl3\server.py ^
  -m %MODEL% ^
  -dm mtp -cs %CS% -cq 4 --port 8000 --host %HOST% ^
  --extra "-tp -tpb native -gs %GS% -ndt 3" 2>&1 | pwsh -NoProfile -Command "$input | Tee-Object -FilePath '%LOG%' -Encoding utf8"
if exist "%STOP%" (
    del "%STOP%"
    echo == EXL3 serve: stopped on purpose ^(exl3-stop.flag^), not relaunching.
    goto end
)
for /f "usebackq" %%i in (`powershell -NoProfile -Command "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"`) do set T1=%%i
set /a ELAPSED=T1-T0
if %ELAPSED% LSS 420 (set /a FAST+=1) else (set FAST=0)
if %FAST% GEQ 3 (
    echo == EXL3 serve: exited within 420 s of starting three times in a row, giving up.
    exit /b 1
)
if exist "%FLAG%" (
    echo == EXL3 serve: the server asked for a relaunch. Reason:
    type "%FLAG%"
    del "%FLAG%"
) else (
    echo == EXL3 serve: the server exited on its own after %ELAPSED% s.
)
echo == EXL3 serve: relaunching in 5 s.
rem not `timeout /t`: it exits at once ("Input redirection is not supported") when stdin is not a console
powershell -NoProfile -Command "Start-Sleep 5"
goto again
:end
