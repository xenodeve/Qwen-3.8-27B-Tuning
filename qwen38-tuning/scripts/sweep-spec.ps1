<#
Speculation matrix for ONE quant, per benchmark protocol section 6.

    | quant | none | ngram-simple | draft-mtp n=2 | draft-mtp n=3 |

Model-parameterized so the identical protocol runs against Q4 and Q3 -- the
comparison is only meaningful if both quants meet the same procedure.

Per config: restart the server, snapshot the environment, measure generation
N times, record speculative acceptance, and capture a greedy sample for the
output-equivalence check.

Two prompts are used deliberately:
  bench  - a short instruction, nothing in context to copy. This is the case
           where ngram CANNOT help, and where an earlier ngram-mod run scored
           20.8% acceptance and no speed gain.
  code   - a long prompt containing the source about to be rewritten. This is
           the case the llama.cpp docs actually cite for ngram-simple
           (source-code rewriting). Testing ngram only on the short prompt
           would have condemned it on an unfair test.

Usage:
    .\sweep-spec.ps1 -Quant UD-Q4_K_XL -Tag q4
    .\sweep-spec.ps1 -Quant UD-Q3_K_XL -Tag q3 -Repeats 3
#>
param(
  [string]$Quant   = 'UD-Q4_K_XL',
  [string]$Tag     = 'q4',
  [int]$Repeats    = 3,
  [int]$Ctx        = 16384,
  [string]$Root    = 'C:\AI\qwen38-tuning',
  [string]$Repo    = 'unsloth/Qwen3.8-27B-GGUF'
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'Get-GpuVram.ps1')   # the one place that asks the driver about a GPU (#50)
$out = Join-Path $Root "results\spec-matrix-$Tag.jsonl"

$srcToRewrite = @'
class LRUCache:
    def __init__(self, cap):
        self.cap = cap
        self.data = {}
        self.order = []
    def get(self, k):
        if k not in self.data:
            return None
        self.order.remove(k)
        self.order.append(k)
        return self.data[k]
    def put(self, k, v):
        if k in self.data:
            self.order.remove(k)
        elif len(self.data) >= self.cap:
            victim = self.order.pop()
            del self.data[victim]
        self.data[k] = v
        self.order.append(k)
'@

$prompts = @{
  bench = 'Write a Python function that reverses a linked list.'
  code  = "Here is a Python class:`n`n$srcToRewrite`n`nRewrite this class exactly as given, but rename the attribute `"order`" to `"usage`" everywhere. Output the full class."
}

function Stop-Server {
  $c = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
  if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 5 }
}

function Start-Server {
  param([string]$SpecType, [int]$NMax)
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $log   = Join-Path $Root "logs\$Tag-$SpecType-n$NMax-$stamp.log"

  $a = @('-hf',"${Repo}:$Quant",'--alias',"qwen38-$Tag",'-c',"$Ctx",
         '-ngl','auto','--fit','on','-fa','on','-np','1','--no-mmproj-auto',
         '--host','127.0.0.1','--port','8080')
  if ($SpecType -ne 'none') { $a += @('--spec-type',$SpecType,'--spec-draft-n-max',"$NMax") }

  Start-Process C:\AI\llama.cpp-cuda\llama-server.exe -ArgumentList $a -WindowStyle Hidden `
                -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  for ($i=0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 4
    try { Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 3 | Out-Null; return $log } catch {}
  }
  return $null
}

$configs = @(
  @{ spec='none';         n=0 },
  @{ spec='ngram-simple'; n=4 },
  @{ spec='draft-mtp';    n=2 },
  @{ spec='draft-mtp';    n=3 }
)

foreach ($cfg in $configs) {
  Stop-Server
  # Environment must be snapshotted BEFORE load: --fit derives the layer split
  # from whatever VRAM is free at that moment.
  & (Join-Path $Root 'scripts\collect-env.ps1') -Label "$Tag-$($cfg.spec)-n$($cfg.n)" -Root $Root | Out-Null
  $log = Start-Server -SpecType $cfg.spec -NMax $cfg.n
  if (-not $log) { "FAILED TO START: $Tag $($cfg.spec) n=$($cfg.n)"; continue }

  # One named card. Unfiltered, this reads whichever GPU the driver lists
  # first, which on a two-card machine is not the one serving (#50).
  $g = Get-GpuVram
  $vram = if ($g) { @($g.Used, $g.Free) } else { @($null, $null) }

  foreach ($pk in @('bench','code')) {
    $tg = @(); $dn = 0; $da = 0; $pp = @()
    for ($i=1; $i -le $Repeats; $i++) {
      $body = @{ prompt=$prompts[$pk]; n_predict=160; temperature=0.7; cache_prompt=$false } | ConvertTo-Json
      $r = Invoke-RestMethod http://127.0.0.1:8080/completion -Method Post -Body $body `
             -ContentType 'application/json' -TimeoutSec 1200
      $tg += $r.timings.predicted_per_second
      $pp += $r.timings.prompt_per_second
      if ($r.timings.PSObject.Properties.Name -contains 'draft_n') {
        $dn += [int]$r.timings.draft_n; $da += [int]$r.timings.draft_n_accepted
      }
    }
    $s = $tg | Sort-Object
    $mid = [int][math]::Floor(($s.Count - 1) / 2)   # true median; [int](3/2) would be 2
    $row = [ordered]@{
      quant = $Quant; tag = $Tag; spec_type = $cfg.spec; n_max = $cfg.n; prompt = $pk
      tg_all = ($tg | ForEach-Object { [math]::Round($_,2) }) -join ' '
      tg_median = [math]::Round($s[$mid],2)
      tg_min = [math]::Round($s[0],2); tg_max = [math]::Round($s[-1],2)
      pp_median = [math]::Round((($pp | Sort-Object)[$mid]),1)
      vram_used_mib = [int]$vram[0]; vram_free_mib = [int]$vram[1]
      draft_n = $dn; draft_accepted = $da
      acceptance_pct = if ($dn -gt 0) { [math]::Round(100.0*$da/$dn,1) } else { $null }
    }
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $out -Encoding utf8
    "$Tag $($cfg.spec) n=$($cfg.n) [$pk] -> median $($row.tg_median) tok/s (range $($row.tg_min)-$($row.tg_max)), accept $($row.acceptance_pct)%"
  }

  # Greedy equivalence sample: identical text across configs means the
  # speculative path is a pure performance toggle on this stack.
  $g = @{ prompt='def fibonacci(n):'; n_predict=60; temperature=0.0; top_k=1; seed=42; cache_prompt=$false } | ConvertTo-Json
  $gr = Invoke-RestMethod http://127.0.0.1:8080/completion -Method Post -Body $g -ContentType 'application/json' -TimeoutSec 1200
  $gr.content | Set-Content (Join-Path $Root "results\greedy-$Tag-$($cfg.spec)-n$($cfg.n).txt") -Encoding utf8
}

"`n=== $Tag matrix complete -> $out ==="
Get-Content $out | ForEach-Object { $_ | ConvertFrom-Json } |
  Format-Table spec_type,n_max,prompt,tg_median,tg_min,tg_max,acceptance_pct,vram_free_mib -AutoSize
