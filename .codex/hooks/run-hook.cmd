: << 'CMDBLOCK'
@echo off
REM Cross-platform polyglot wrapper for T4 hook scripts.
REM On Windows: cmd.exe runs this batch block, which locates bash and calls it.
REM On Unix:    the shell reads the file as a script (: is a no-op; the here-doc
REM             swallows the batch block) and runs the bash portion below.
REM
REM Hook scripts use extensionless names so Claude Code's Windows .sh
REM auto-detection doesn't interfere. Usage: run-hook.cmd <script-name> [args...]

if "%~1"=="" (
    echo run-hook.cmd: missing script name >&2
    exit /b 1
)

set "HOOK_DIR=%~dp0"

if exist "C:\Program Files\Git\bin\bash.exe" (
    "C:\Program Files\Git\bin\bash.exe" "%HOOK_DIR%%~1" %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)
if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
    "C:\Program Files (x86)\Git\bin\bash.exe" "%HOOK_DIR%%~1" %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)
where bash >nul 2>nul
if %ERRORLEVEL% equ 0 (
    bash "%HOOK_DIR%%~1" %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)

REM No bash found — exit silently (hooks degrade to no-op rather than error).
exit /b 0
CMDBLOCK

# Unix: run the named script from this hooks directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME="$1"
shift
exec bash "${SCRIPT_DIR}/${SCRIPT_NAME}" "$@"
