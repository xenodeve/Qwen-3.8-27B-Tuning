<#
build-dflash2.ps1 — compile llama.cpp PR #27342 (DFlash2) into a PARALLEL directory.

WHY A PARALLEL DIRECTORY, NOT AN UPGRADE IN PLACE
-------------------------------------------------
57 files in this repo hardcode  C:\AI\llama.cpp-cuda  (5 bench drivers,
hardware.json, and 51 scripts). Overwriting that directory would switch the
runtime under all of them at once, silently, with no marker in any recorded
row. Every number in docs/reports/ was measured on build 10472.

So build 10472 stays exactly where it is and is not touched. This script
stages the new binaries into  C:\AI\llama.cpp-dflash2  and NOTHING existing is
repointed at it. The two builds are then paired within a single round, which
is what the 13.6 % drift floor requires (CLAUDE.md, "Never compare raw decode
across boots").

WHY BUILD AT ALL
----------------
PR #27342 was still `open` / `mergeable_state: blocked` on 2026-08-22. Release
b10549 does not carry it. On build 10472 the DFlash2 drafter fails to load with
`wrong number of tensors; expected 81, got 58` — that build's `draft-dflash`
flag is DFlash 1 (docs/tested/02-decoders.md, CORRECTIONS 18).

Issue #17.
#>
$ErrorActionPreference = 'Stop'

$Src   = 'C:\AI\llama.cpp'            # gitignored
$Stage = 'C:\AI\llama.cpp-dflash2'    # gitignored
$PrNum = 27342
$PrHead = '1deefcca395743049c3820ab8f9b15043f3e9446'   # z-lab/llama.cpp-fork:dflash2 as read on 2026-08-22

# Tools are discovered, not pinned. This script exists to be re-run whenever the
# PR moves -- possibly months from now -- and a pinned 'CUDA\v13.3' or a winget
# package hash would fail on a machine that has the tool, just a newer one.
function Find-Tool([string]$Name, [string[]]$Candidates) {
    $onPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    foreach ($c in $Candidates) {
        $hit = Get-ChildItem $c -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw "missing build tool: $Name (looked on PATH and in: $($Candidates -join ', '))"
}

$CMake = Find-Tool 'cmake' @(
    'C:\Program Files\CMake\bin\cmake.exe',
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
$Ninja = Find-Tool 'ninja' @(
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja_*\ninja.exe")
$VcVars = Find-Tool 'vcvars64.bat' @(
    'C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat')

# Highest installed toolkit wins. nvcc is checked explicitly because a toolkit
# directory can exist without the compiler in it.
$CudaRoot = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*' -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'bin\nvcc.exe') } |
    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $CudaRoot) { throw 'no CUDA toolkit with bin\nvcc.exe found under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' }

Write-Host "cmake:  $CMake"
Write-Host "ninja:  $Ninja"
Write-Host "vcvars: $VcVars"
Write-Host "cuda:   $CudaRoot"

# --- source ---------------------------------------------------------------
# git is a native program: $ErrorActionPreference does not stop the script when
# it fails, so every call is checked. Skipping this once let a failed fetch fall
# through to a build of the previous tree that then reported success.
function Invoke-Git {
    git -C $Src @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path $Src)) {
    git clone https://github.com/ggml-org/llama.cpp $Src
    if ($LASTEXITCODE -ne 0) { throw "git clone failed with exit code $LASTEXITCODE" }
}

# Fetch to FETCH_HEAD and check out detached rather than maintaining a local
# branch. A named branch cannot be fetched into while it is checked out --
# 'refusing to fetch into branch' -- which made the second run of this script
# fail where the first had passed. Detached HEAD also makes the built SHA the
# only thing identifying the tree, which is what a measurement needs to cite.
Invoke-Git fetch origin "pull/$PrNum/head" --force
Invoke-Git checkout --detach FETCH_HEAD

$head = (git -C $Src rev-parse HEAD).Trim()
Write-Host "PR #$PrNum head: $head"
if ($head -ne $PrHead) {
    # Not fatal: the PR can gain commits. But the instrument must be identifiable,
    # so say so loudly rather than letting a different tree pass as this one.
    Write-Warning "PR head moved since this script was written (expected $PrHead). Record the SHA above with any measurement."
}

# --- configure + build ----------------------------------------------------
# CMAKE_CUDA_ARCHITECTURES=89 is Ada / RTX 4070 SUPER. Building only the one
# arch we measure on keeps the compile from taking hours.
# Each token is wrapped in double quotes before being flattened into the batch
# line below. Without that, a value containing a space -- and CUDAToolkit_ROOT
# lives under 'C:\Program Files' -- is split by cmd, and cmake reports
# `CUDAToolkit_ROOT=C:\Program` plus 'Ignoring extra path from command line'.
# Named $cfgArgs, not $args: $args is a PowerShell automatic variable.
$cfgArgs = @(
    '-B', 'build-dflash2', '-G', 'Ninja',
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    '-DCMAKE_BUILD_TYPE=Release',
    '-DGGML_CUDA=ON',
    "-DCUDAToolkit_ROOT=$CudaRoot",
    '-DCMAKE_CUDA_ARCHITECTURES=89',
    '-DLLAMA_CURL=OFF',
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF'
) | ForEach-Object { '"' + $_ + '"' }
$cfgLine = $cfgArgs -join ' '

# cl.exe and nvcc need the MSVC environment; vcvars64 only exports it into a
# cmd session, so the whole build runs inside one.
$batchBody = @"
call "$VcVars" || exit /b 1
rem vcvars64 sets up MSVC only. The PATH entry the CUDA installer added has not
rem reached an already-running shell, so without this cmake FindCUDAToolkit reports
rem "Could not find nvcc executable in any searched paths".
set "PATH=$CudaRoot\bin;%PATH%"
cd /d "$Src" || exit /b 1
"$CMake" $cfgLine || exit /b 1
"$CMake" --build build-dflash2 --target llama-server llama-cli --parallel || exit /b 1
"@
# A failed configure leaves CMakeCache.txt holding the bad value, and cmake
# reuses cached variables -- so without this a rerun fails identically and
# looks like the fix did not work.
$cache = Join-Path $Src 'build-dflash2\CMakeCache.txt'
if (Test-Path $cache) { Remove-Item $cache -Force }

$bat = Join-Path $env:TEMP 'build-dflash2.bat'
Set-Content -Path $bat -Value $batchBody -Encoding ASCII
cmd.exe /c $bat
if ($LASTEXITCODE -ne 0) { throw "build failed with exit code $LASTEXITCODE" }

# --- stage ----------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$bin = Join-Path $Src 'build-dflash2\bin'
Copy-Item "$bin\*.exe", "$bin\*.dll" -Destination $Stage -Force

# The CUDA runtime DLLs are not produced by the build; llama-server needs them
# beside it. Take them from the toolkit rather than from the 10472 directory,
# which carries CUDA 12 and would be the wrong runtime for a CUDA 13 build.
# CUDA 13 keeps these in bin\x64, where CUDA 12 kept them in bin -- so search
# rather than assume a layout. A missing one is fatal, not a warning: without
# them llama-server.exe does not start at all, and the first run of this script
# reported 'staged' while the binary could not load.
$cudaBin = Join-Path $CudaRoot 'bin'
foreach ($pat in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
    $found = Get-ChildItem $cudaBin -Recurse -Filter $pat -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $found) { throw "CUDA runtime DLL not found under ${cudaBin}: $pat" }
    Copy-Item $found.FullName -Destination $Stage -Force
    Write-Host "staged $($found.Name)"
}

Write-Host ''
Write-Host "staged to $Stage"
& (Join-Path $Stage 'llama-server.exe') --version
Write-Host ''
Write-Host "PR #$PrNum head recorded: $head"
