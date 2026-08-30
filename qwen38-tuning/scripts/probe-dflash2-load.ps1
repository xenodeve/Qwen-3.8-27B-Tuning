<#
probe-dflash2-load.ps1 — does the DFlash2 drafter load at all on the new build?

This answers issue #17's acceptance question and NOTHING ELSE. It is not a
benchmark and it does not produce a tok/s figure. On build 10472 the same
drafter fails with:

    wrong number of tensors; expected 81, got 58

because 10472's `draft-dflash` is DFlash 1 (docs/tested/02-decoders.md,
CORRECTIONS 18). If that string is gone and the server reaches "server is
listening", the instrument exists. What it is worth is a separate question and
a separate issue.

The context here is DELIBERATELY SMALL and is not a recommendation. Profile A
runs 131,072 with roughly 600 MiB of margin; this drafter is 1.1 GB, so the
depth that actually fits alongside it is unknown and unmeasured. Picking a
number here and calling it a profile would be a guess wearing a config file.

Flags verified by reading the PR source, not from a summary:
  --spec-type draft-dflash    common/speculative.cpp:39
  --spec-draft-model / -md    common/arg.cpp:4146
  --spec-draft-n-max          common/arg.cpp:4077
DFlash2 has no flag of its own -- common/speculative.cpp:978 detects it from
the checkpoint (is_dflash2 = selector_top_k > 0).
#>
param(
    [int]$Ctx  = 16384,
    [int]$Port = 8080,
    [int]$NMax = 4
)
$ErrorActionPreference = 'Stop'

$Exe = 'C:\AI\llama.cpp-dflash2\llama-server.exe'
$Tgt = "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf"
$Dft = "C:\Users\xenod\.cache\huggingface\hub\models--z-lab--Qwen3.8-27B-DFlash2-GGUF\snapshots\57ab3265056d4024870b0621cfc2c127537020ed\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

foreach ($f in @($Exe, $Tgt, $Dft)) {
    if (-not (Test-Path $f)) { throw "missing: $f" }
}

# Never share port 8080 with another orchestrator -- an armed queue once killed a
# running corpus and the summary still printed a plausible number (CLAUDE.md).
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) { throw "port $Port is already listening (pid $($busy.OwningProcess)). Stop it first." }

$log = Join-Path $env:TEMP "dflash2-probe.log"
foreach ($f in @($log, "$log.out")) { if (Test-Path $f) { Remove-Item $f -Force } }

$serverArgs = @(
    '-m', $Tgt,
    '-md', $Dft,
    '--spec-type', 'draft-dflash',
    '--spec-draft-n-max', "$NMax",
    '-ngld', '99',
    '--alias', 'qwen38-dflash2',
    '-c', "$Ctx",
    '-ngl', 'auto', '--fit', 'on', '-fa', 'on', '-np', '1',
    '-ctk', 'q4_0', '-ctv', 'q4_0',
    '--no-mmproj-auto', '-lv', '3',
    '--chat-template-file', 'C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja',
    '--host', '127.0.0.1', '--port', "$Port"
)

# gguf-py lives in the llama.cpp source tree the binary was built from. Deleting
# that tree leaves a working binary and a probe that cannot check anything, so
# say which of the two is missing rather than failing inside python.
$GgufPy = 'C:/AI/llama.cpp/gguf-py'
if (-not (Test-Path $GgufPy)) {
    throw "$GgufPy is gone. The staged binary still works, but this probe reads the drafter's GGUF metadata to tell DFlash2 from DFlash 1 and needs it. Re-run scripts/build-dflash2.ps1 to restore the source tree."
}
$meta = & python -c "import sys; sys.path.insert(0,sys.argv[2]); from gguf import GGUFReader; r=GGUFReader(sys.argv[1]); print(r.fields['dflash.selector_top_k'].contents(), len(r.tensors))" $Dft $GgufPy
if ($LASTEXITCODE -ne 0) { throw "could not read GGUF metadata from $Dft" }
$topk, $nTensors = $meta -split ' '
if ([int]$topk -le 0) { throw "drafter is DFlash 1, not DFlash2: dflash.selector_top_k=$topk" }
Write-Host "drafter metadata: selector_top_k=$topk (>0 selects the DFlash2 path), tensors=$nTensors"

Write-Host "starting $Exe (ctx $Ctx, n-max $NMax) -- log: $log"
$p = Start-Process $Exe -ArgumentList $serverArgs -PassThru -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput "$log.out"

# Loading a 27B target plus a 1.1 GB drafter from a cold page cache is slow;
# 300 s is generous on purpose so a slow load is not misread as a failure.
$listening = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 1
    if ($p.HasExited) { break }
    $text = (Get-Content $log, "$log.out" -ErrorAction SilentlyContinue) -join [Environment]::NewLine
    if ($text -match 'listening on http|server is listening') { $listening = $true; break }
    if ($text -match 'wrong number of tensors')  { break }
}

$text = (Get-Content $log, "$log.out" -ErrorAction SilentlyContinue) -join [Environment]::NewLine
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }

Write-Host ''
if ($text -match 'wrong number of tensors[^\r\n]*') {
    Write-Host "FAIL - the 10472 error is still present:" -ForegroundColor Red
    Write-Host "  $($Matches[0])"
    exit 1
}
if ($listening) {
    Write-Host "PASS - drafter loaded and the server reached its listening line." -ForegroundColor Green
    Select-String -Path $log, "$log.out" -Pattern 'dflash', 'draft model', 'n_ctx ' -SimpleMatch -ErrorAction SilentlyContinue |
        Select-Object -First 8 | ForEach-Object { "  $($_.Line)" }
    Write-Host ''
    Write-Host "This says the instrument loads. It says nothing about speed."
    exit 0
}
Write-Host "INCONCLUSIVE - no 'server is listening' and no tensor-count error." -ForegroundColor Yellow
Write-Host "Read $log in full before concluding anything."
exit 2
