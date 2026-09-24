# Phase 0E harness — boot llama-server on the Flash-Next Q2_0 artifact and report
# whether it loads, how much VRAM it takes, and whether it answers.
#
# Usage:  .\fn-boot.ps1 -NcMoE 34 [-Ctx 16384] [-Sm layer] [-Ts "10,15"] [-Keep]
#
# Note on detection: this build logs "listening on http://..." (not
# "server is listening"), and it does NOT print CUDA buffer sizes at default
# verbosity, so VRAM is measured from the driver after load instead.
param(
  [int]$NcMoE = 48,
  [int]$Ctx = 16384,
  [string]$Sm = 'layer',
  [string]$Ts = '',
  [int]$Port = 8099,
  [switch]$Keep,
  [int]$TimeoutSec = 300,
  [int]$MaxTokens = 64
)

$exe = 'C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe'
$m   = 'C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
$tag = "ncmoe$NcMoE-c$Ctx-$Sm$(if($Ts){'-ts'+($Ts -replace ',','_')})"
$log = "C:\AI\qwen38-tuning\logs\fn-$tag.log"

. 'C:\AI\qwen38-tuning\scripts\Get-GpuVram.ps1'

Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2
Remove-Item $log, "$log.err" -Force -ErrorAction SilentlyContinue

$argv = @('-m',$m,'-c',"$Ctx",'-ctk','q4_0','-ctv','q4_0','-b','2048','-ub','512',
          '-ngl','all','-ncmoe',"$NcMoE",'-sm',$Sm,'-fit','off',
          '-lm','mmap','--tensor-read-lazy','on','-fa','on','-np','1',
          '-t','14','-tb','20','--host','127.0.0.1','--port',"$Port")
if ($Ts) { $argv += @('-ts', $Ts) }

$t0 = Get-Date
$p = Start-Process $exe -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err" -ArgumentList $argv
$state = 'timeout'
while ((Get-Date) -lt $t0.AddSeconds($TimeoutSec)) {
  Start-Sleep -Seconds 3
  $txt = Get-Content $log, "$log.err" -Raw -ErrorAction SilentlyContinue
  if ($txt -match 'listening on http')      { $state = 'listening'; break }
  if ($txt -match 'out of memory|failed to load model|exiting due to') { $state = 'error'; break }
  if ($p.HasExited) { $state = 'exited'; break }
}
$elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds)

$gpus = Get-InstalledGpu
$vram = @{}
foreach ($g in $gpus) { $v = Get-GpuVram -Uuid $g.Uuid; if ($v) { $vram[$g.Name] = $v } }

$os = Get-CimInstance Win32_OperatingSystem
$ramUsedGiB = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)

$gen = ''
if ($state -eq 'listening') {
  $reqFile = Join-Path $env:TEMP "fn-boot-req.json"
  ('{"messages":[{"role":"user","content":"Write a Python one-liner that prints the sum of 1..10. Output only the code."}],"max_tokens":%d,"temperature":0}' -f $MaxTokens) |
    Set-Content -Encoding UTF8 $reqFile
  $resp = curl.exe -s --max-time 600 "http://127.0.0.1:$Port/v1/chat/completions" `
            -H 'Content-Type: application/json' -d "@$reqFile"
  try {
    $j = $resp | ConvertFrom-Json
    $gen = "tokens=$($j.usage.completion_tokens)"
    if ($j.choices[0].message.content) { $gen += " decode=$([math]::Round($j.timings.predicted_per_second,2)) prefill=$([math]::Round($j.timings.prompt_per_second,2))" }
  } catch { $gen = "no-json: $resp" }
}

Write-Output "================ RESULT ================"
Write-Output "ncmoe=$NcMoE ctx=$Ctx sm=$Sm ts=$Ts"
Write-Output "state=$state elapsed=${elapsed}s"
foreach ($k in $vram.Keys) { Write-Output ("VRAM {0}: used={1} MiB free={2} MiB" -f $k, $vram[$k].Used, $vram[$k].Free) }
$totUsed = ($vram.Values | Measure-Object Used -Sum).Sum
Write-Output "VRAM total used = $totUsed MiB  ($([math]::Round($totUsed/1024,2)) GiB)"
Write-Output "RAM used = $ramUsedGiB GiB"
Write-Output "generation: $gen"
if ($state -ne 'listening') {
  Select-String -Path $log, "$log.err" -Pattern 'error|out of memory|failed|tensor overrides' -ErrorAction SilentlyContinue |
    Select-Object -Last 4 | ForEach-Object { '   ' + $_.Line.Trim() }
}
Write-Output "========================================"

if (-not $Keep) { Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force }
