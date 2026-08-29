@echo off
REM ============================================================================
REM  Qwen3.8-27B  --  pick a server
REM
REM  Double-click this if you do not want to remember twelve filenames. It asks
REM  two questions and then hands off to the launcher that owns the answer.
REM
REM  THIS FILE HOLDS NO SERVING FLAGS, deliberately. It calls one of the other
REM  .bat files, each of which calls the shared PowerShell entry point, which
REM  calls the profile -- and the profile is
REM  the only place a flag is written down. A chooser that assembled its own
REM  command line would be a second source of truth, and this project has
REM  already shipped a launcher that described a run it did not perform.
REM
REM  The launchers themselves live in launchers\ so that the root holds one
REM  icon instead of thirteen. Each still works on its own if you go in there;
REM  this is a front door, not a replacement.
REM ============================================================================

setlocal
cd /d "%~dp0"

:menu
cls
echo.
echo   Qwen3.8-27B  --  which server?
echo   ============================================================
echo.
echo   BOTH CARDS, NVFP4  -- fastest measured here, +63%% over 2 and 3
echo.
echo     1   147,456 context, images work          [recommended]
echo     2   200,704 context, images work          deepest measured
echo     7   147,456 + the Unsloth bundle          SPEED UNMEASURED
echo.
echo   BOTH CARDS, UD-Q4_K_XL  -- the artifact whose output we have used longest
echo.
echo     3   deepest window that fits, ~250,000    no images
echo     4   ... plus draft-mtp                    SPEED NOT MEASURED
echo     5   ... plus DFlash2, 131,072             patched binary, little spare
echo.
echo   ONE CARD, UD-Q2_K_XL
echo.
echo     6   the single-GPU profile
echo.
echo     Q   quit                                  (one key, no Enter)
echo.
echo   1 and 2 both take pictures. 2 goes deeper and finishes a large request
echo   with the least room to spare of anything measured here. 7 is 1 with six
echo   settings borrowed from Unsloth Studio: it uses 2.03 GB of RAM against
echo   15.28, and NOBODY HAS PAIRED ITS SPEED against 1 yet.
echo.
echo   Quality has never been measured on ANY of these artifacts. 1 and 2 change
echo   the model file, not just a flag; 3 is what has been served longest.
echo.

REM  `choice` and not `set /p`. set /p accepts anything, including a value with
REM  a newline in it, and `if "%SEL%"=="1"` then breaks with "The syntax of the
REM  command is incorrect" instead of saying the input was wrong -- observed
REM  here on 2026-08-29. choice restricts the keystroke itself, needs no Enter,
REM  and cannot hand a broken value to the comparison below. It returns the
REM  POSITION in the key list, so 1-6 line up with the printed numbers and Q
REM  is 7.
choice /c 1234567Q /n /m "  Choose 1-7, or Q to quit: "
set "SEL=%ERRORLEVEL%"

REM  BOTH NAMES ARE SPELLED OUT, not built by appending "-lan" to a stem. A
REM  constructed filename cannot be checked by anything until somebody presses
REM  the key, and this file's whole job is to point at other files.
if "%SEL%"=="8" goto :done
if "%SEL%"=="1" (
    set "LOOP=serve-dual-nvfp4.bat"
    set "WIDE=serve-dual-nvfp4-lan.bat"
    goto :ask_lan
)
if "%SEL%"=="2" (
    set "LOOP=serve-dual-nvfp4-deep.bat"
    set "WIDE=serve-dual-nvfp4-deep-lan.bat"
    goto :ask_lan
)
if "%SEL%"=="3" (
    set "LOOP=serve-dual.bat"
    set "WIDE=serve-dual-lan.bat"
    goto :ask_lan
)
if "%SEL%"=="4" (
    set "LOOP=serve-dual-mtp.bat"
    set "WIDE=serve-dual-mtp-lan.bat"
    goto :ask_lan
)
if "%SEL%"=="5" (
    set "LOOP=serve-dual-dflash.bat"
    set "WIDE=serve-dual-dflash-lan.bat"
    goto :ask_lan
)
if "%SEL%"=="6" (
    set "LOOP=serve.bat"
    set "WIDE=serve-lan.bat"
    goto :ask_lan
)
if "%SEL%"=="7" (
    set "LOOP=serve-dual-nvfp4-lean.bat"
    set "WIDE=serve-dual-nvfp4-lean-lan.bat"
    goto :ask_lan
)
REM  Unreachable while choice guards the key list, and kept anyway: if that
REM  list and these branches ever disagree, this says so instead of falling
REM  through to :ask_lan with nothing set.
echo.
echo   That choice is not wired to a launcher. Nothing started.
echo.
pause
goto :menu

:ask_lan
echo.
echo   Reachable from other machines on your network?
echo.
echo   THIS SERVER HAS NO API KEY, no authentication of any kind, and CORS is
echo   open. Anyone who can reach this machine could use it and read every
echo   prompt -- and send it pictures, on the profiles that take them. Say yes
echo   only on a network you control.
echo.
REM  N is FIRST in the key list, so position 1 is the safe answer.
choice /c NY /n /m "  Expose on the LAN?   [N] no, this machine only   [Y] yes: "
set "PICK=%LOOP%"
if "%ERRORLEVEL%"=="2" set "PICK=%WIDE%"

if not exist "%~dp0launchers\%PICK%" (
    echo.
    echo   %PICK% is missing. This hub only points at other launchers; it
    echo   holds no flags of its own, so there is nothing here to start with.
    echo.
    pause
    goto :done
)

echo.
echo   Starting %PICK% ...
echo   The server runs in this window. Ctrl+C stops it; so does closing it.
echo.

call "%~dp0launchers\%PICK%"
goto :done

:done
endlocal
