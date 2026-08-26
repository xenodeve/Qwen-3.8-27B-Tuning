<#
Restart llama-server into a named configuration, then run the execution-verified
quality corpus against it.

Speed and quality must be measured on the SAME server instance -- restarting
between them would let --fit choose a different layer split and silently break
the pairing.

Usage:
  .\bench-quality.ps1 -Quant UD-Q4_K_XL -Tag q4 -SpecType draft-mtp -NMax 2 `
                      -Temperatures 1.0,0.6 -Attempts 3
#>
param(
  [string]$Quant        = 'UD-Q4_K_XL',
  [string]$Tag          = 'q4',
  [string]$SpecType     = 'draft-mtp',
  [int]$NMax            = 2,
  # A string, not double[]: powershell.exe -File passes every argument as a
  # single string, so "1.0,0.6" never becomes an array and the call dies on
  # parameter binding before anything runs.
  [string]$Temperatures = '1.0',
  [int]$Attempts        = 3,
  [string]$Effort       = 'medium',
  [int]$Ctx             = 16384,
  [string]$Root         = 'C:\AI\qwen38-tuning',
  [string]$Repo         = 'unsloth/Qwen3.8-27B-GGUF'
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'Get-GpuVram.ps1')   # the one place that asks the driver about a GPU (#50)

$c = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 5 }

& (Join-Path $Root 'scripts\collect-env.ps1') -Label "quality-$Tag-$SpecType-n$NMax" -Root $Root | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log   = Join-Path $Root "logs\quality-$Tag-$SpecType-n$NMax-$stamp.log"
$a = @('-hf',"${Repo}:$Quant",'--alias',"qwen38-$Tag",'-c',"$Ctx",
       '-ngl','auto','--fit','on','-fa','on','-np','1','--no-mmproj-auto',
       '--host','127.0.0.1','--port','8080')
if ($SpecType -ne 'none') { $a += @('--spec-type',$SpecType,'--spec-draft-n-max',"$NMax") }

Start-Process C:\AI\llama.cpp-cuda\llama-server.exe -ArgumentList $a -WindowStyle Hidden `
              -RedirectStandardOutput $log -RedirectStandardError "$log.err"

$up = $false
for ($i=0; $i -lt 45; $i++) {
  Start-Sleep -Seconds 4
  try { Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { "SERVER FAILED TO START for $Tag/$SpecType/n$NMax"; exit 1 }

# One named card. Unfiltered, this reads whichever GPU the driver lists
# first, which on a two-card machine is not the one serving (#50).
$g = Get-GpuVram
$vram = if ($g) { @($g.Used, $g.Free) } else { @($null, $null) }
"server up: $Tag $SpecType n=$NMax | VRAM used $($vram[0]) MiB, free $($vram[1]) MiB"

foreach ($t in ($Temperatures -split ',' | ForEach-Object { [double]$_.Trim() })) {
  $label = "$Tag-$SpecType$NMax-t$($t.ToString('0.0'))"
  "`n=== quality bench: $label ==="
  & python (Join-Path $Root 'bench\run_bench.py') --label $label --attempts $Attempts `
      --temperature $t --reasoning-effort $Effort
}
