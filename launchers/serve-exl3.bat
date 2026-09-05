@echo off
REM ============================================================================
REM  Start the EXL3 server  --  BOTH GPUs, Mia-AiLab 3.5bpw, MTP head, loopback
REM
REM  Double-click this. The server runs IN this window; Ctrl+C or closing it
REM  stops it. It listens on port 8000 (llama-server keeps 8080) and speaks the
REM  OpenAI API only -- Claude Code reaches it through `claude-xeno-exl3`, which
REM  starts a LiteLLM translator on port 4000.
REM
REM  WHAT IT IS. ExLlama3 (the Mia-AiLab fork built from source here) serving
REM  the 3.5 bpw trellis quant with the model's own MTP draft head, integer
REM  4-bit KV, tensor-parallel across the two cards. Measured (results 10):
REM  31.9-36.9 tok/s at 147K paired in one boot against llama.cpp's 40.1-44.3,
REM  so ~81 % at the served depth -- and 47-55 tok/s in real 30-70K sessions,
REM  above what the llama.cpp profile logged at the same depths (39-47).
REM  Prefill is ~60 % of llama.cpp's. Loads in ~30 s. Quality is UNMEASURED on
REM  any EXL3 artifact; an outside KL chart puts EXL3 4bpw ahead of NVFP4 and
REM  that has not been checked here.
REM
REM  It holds no configuration. The recipe lives in
REM  qwen38-tuning\scripts\serve-exl3.cmd and only there; this passes the
REM  measured cache (163,840 tokens), the measured split caps, and the bind
REM  address.
REM ============================================================================

setlocal
cd /d "%~dp0.."

call "%~dp0..\qwen38-tuning\scripts\serve-exl3.cmd" 163840 "9,15.5" 127.0.0.1
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve-exl3.cmd exited with code %RC%.
    pause
)
