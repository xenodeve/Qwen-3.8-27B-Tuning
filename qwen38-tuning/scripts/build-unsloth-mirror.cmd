@echo off
REM ===========================================================================
REM  Build Unsloth's llama.cpp (b10679-mix-67dfc8b) WITH the DFlash2 mirror
REM  patch, into a tree of our own.
REM
REM  WHY. Unsloth's shipped binary aborts the moment DFlash2 is asked for under
REM  -sm tensor:  ggml-backend-meta.cpp:543
REM               GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)
REM  because output.weight maps to SPLIT_AXIS_1 there, which spreads the
REM  vocabulary across the cards, and TOP_K needs a whole row. Measured on
REM  their binary 2026-08-30 (issue #52). Patching src\llama-model.cpp to
REM  mirror that tensor is the same one-line change our build 10499 already
REM  carries, and upstream already carries it for DeepSeek V4.
REM
REM  IT DOES NOT TOUCH %USERPROFILE%\.unsloth. That tree is what Unsloth Studio
REM  runs; the source was COPIED to C:\AI\llama.cpp-unsloth-mirror first.
REM
REM  ARCHS 89;120 ONLY. Unsloth ship 86,89,90,100,103,120 for other people's
REM  GPUs. This machine has sm_89 and sm_120a and nothing else, and every extra
REM  architecture is compile time we pay for nobody. Their 226 MB ggml-cuda.dll
REM  against our 88 MB is exactly that difference.
REM
REM  CUDA 13.3, which is the toolkit Unsloth built b10679 with -- see
REM  BUILD_INFO.txt in the source tree. Same major, same minor.
REM ===========================================================================
setlocal
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "SRC=C:\AI\llama.cpp-unsloth-mirror"
set "BLD=%SRC%\build-mirror"
set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

if not exist "%SRC%\src\llama-model.cpp" (
    echo FATAL: %SRC% is not there. Copy it from %%USERPROFILE%%\.unsloth\llama.cpp first,
    echo        excluding build\ and .git\, and apply the mirror patch.
    exit /b 1
)

REM  Refuse to build an UNPATCHED tree. Without this the script would happily
REM  produce a binary that aborts exactly like the one it exists to replace,
REM  and the only symptom would be a failed launch an hour later.
findstr /C:"PATCH (local, not upstream)" "%SRC%\src\llama-model.cpp" >nul
if errorlevel 1 (
    echo FATAL: the mirror patch is not in %SRC%\src\llama-model.cpp.
    echo        Building this tree would reproduce the abort it exists to fix.
    exit /b 1
)

call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo FATAL: vcvars64 failed & exit /b 1 )

"%CMAKE%" -S "%SRC%" -B "%BLD%" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=89;120 ^
    -DLLAMA_CURL=OFF ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF
if errorlevel 1 ( echo FATAL: configure failed & exit /b 1 )

"%CMAKE%" --build "%BLD%" --target llama-server
if errorlevel 1 ( echo FATAL: build failed & exit /b 1 )

echo.
echo BUILT: %BLD%\bin\llama-server.exe
endlocal
