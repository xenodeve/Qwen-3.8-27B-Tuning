<#
Phase C -- speculative decoding sweep for the Q4 lane.

For each configuration: restart llama-server, wait for /health, capture VRAM,
measure generation N times, record draft acceptance, and capture one greedy
sample for the output-equivalence check demanded by the continuation plan (2.2).

Configurations: MTP off baseline, draft-mtp at n_max 2..6, and ngram-mod --
the last because ngram speculation needs no draft weights at all, which matters
on a card with a few hundred MiB free.

Everything except the speculative flags is held fixed.

Usage:  .\run-mtp-sweep.ps1 -Repeats 3
#>
param(
  [int]$Repeats = 3,
  [string]$Root = 'C:\AI\qwen38-tuning'
)

$ErrorActionPreference = 'Continue'
$out = Join-Path $Root 'results\mtp-sweep.jsonl'

# Greedy settings for the equivalence check: speculative decoding is only a pure
# performance toggle if these come back byte-identical across configurations.
$greedy = @{ prompt='def fibonacci(n):'; n_predict=60; temperature=0.0; top_k=1; seed=42; cache_prompt=$false }
$bench  = @{ prompt='Write a Python function that reverses a linked list.'; n_predict=128; temperature=0.7; cache_prompt=$false }

function Stop-Server {
  $c = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
  if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 5 }
}

function Start-Server {
  param([string]$SpecType, [int]$NMax)
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $log   = Join-Path $Root "logs\sweep-$SpecType-n$NMax-$stamp.log"
  $args  = @('-NoProfile','-File')
  if ($SpecType -eq 'off') {
    $args += @((Join-Path $Root 'scripts\start-q4.ps1'))
  } else {
    $args += @((Join-Path $Root 'scripts\start-q4-mtp.ps1'),'-SpecType',$SpecType,'-NMax',"$NMax")
  }
  Start-Process powershell.exe -ArgumentList $args -RedirectStandardOutput $log `
                -RedirectStandardError "$log.err" -WindowStyle Hidden
  for ($i=0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 4
    try { Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 3 | Out-Null; return $log } catch {}
  }
  return $null
}

$configs = @(
  @{ spec='off';        n=0 },
  @{ spec='draft-mtp';  n=2 },
  @{ spec='draft-mtp';  n=3 },
  @{ spec='draft-mtp';  n=4 },
  @{ spec='draft-mtp';  n=5 },
  @{ spec='draft-mtp';  n=6 },
  @{ spec='ngram-mod';  n=4 }
)

$rows = @()
foreach ($cfg in $configs) {
  Stop-Server
  $log = Start-Server -SpecType $cfg.spec -NMax $cfg.n
  if (-not $log) { "FAILED TO START: $($cfg.spec) n=$($cfg.n)"; continue }

  $vram = (nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits) -split '\s*,\s*'

  $tg = @(); $draftN = 0; $draftAcc = 0
  for ($i=1; $i -le $Repeats; $i++) {
    $r = Invoke-RestMethod http://127.0.0.1:8080/completion -Method Post -TimeoutSec 900 `
           -ContentType 'application/json' -Body ($bench | ConvertTo-Json)
    $tg += $r.timings.predicted_per_second
    if ($r.timings.PSObject.Properties.Name -contains 'draft_n') {
      $draftN   += [int]$r.timings.draft_n
      $draftAcc += [int]$r.timings.draft_n_accepted
    }
  }

  $g = Invoke-RestMethod http://127.0.0.1:8080/completion -Method Post -TimeoutSec 900 `
         -ContentType 'application/json' -Body ($greedy | ConvertTo-Json)

  $sorted = $tg | Sort-Object
  # [int](3/2) is 2 in PowerShell, not 1 -- [int] rounds half to EVEN, so 1.5 -> 2.
  # The first version of this line indexed the max and labelled it the median.
  # [math]::Floor keeps the middle element for odd N.
  $mid = [int][math]::Floor(($sorted.Count - 1) / 2)
  $row = [ordered]@{
    spec_type      = $cfg.spec
    n_max          = $cfg.n
    tg_all         = ($tg | ForEach-Object { [math]::Round($_, 2) }) -join ' '
    tg_median      = [math]::Round($sorted[$mid], 2)
    tg_min         = [math]::Round($sorted[0], 2)
    tg_max         = [math]::Round($sorted[-1], 2)
    vram_used_mib  = [int]$vram[0]
    vram_free_mib  = [int]$vram[1]
    draft_n        = $draftN
    draft_accepted = $draftAcc
    acceptance_pct = if ($draftN -gt 0) { [math]::Round(100.0*$draftAcc/$draftN, 1) } else { $null }
    greedy_hash    = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($g.content))) -Algorithm SHA256).Hash.Substring(0,16)
  }
  $rows += [pscustomobject]$row
  ($row | ConvertTo-Json -Compress) | Add-Content -Path $out -Encoding utf8
  $g.content | Set-Content (Join-Path $Root "results\greedy-$($cfg.spec)-n$($cfg.n).txt") -Encoding utf8
  "$($cfg.spec) n=$($cfg.n) -> $($row.tg_median) tok/s, accept $($row.acceptance_pct)%, free $($row.vram_free_mib) MiB"
}

"`n=== summary ==="
$rows | Format-Table -AutoSize
"-> $out"
