param(
  [Parameter(Mandatory=$true)][string]$RunId,
  [string]$Ts = '8500,16000',
  [int]$NcMoE = 24,
  [int]$Threads = 14,
  [int]$ThreadsBatch = 20,
  [int]$Ub = 512,
  [int]$Ctx = 16384,
  [ValidateSet('tensor','layer')][string]$Sm = 'tensor',
  [int]$Batch = 0,                       # 0 = max(2048, Ub); llama.cpp needs b >= ub
  [string]$Exe = '',                     # '' = $global:MIRROR_EXE (the Phase-0 unsloth mirror)
  [string[]]$Extra = @(),                # extra llama-server flags, recorded per row
  [string[]]$Probes = @('decode'),       # any of: decode, declong (512 forced tokens), pre4k, pre16k
  [int]$Reps = 2,
  [int]$Port = 8099,
  [switch]$Trim,                         # EmptyWorkingSet once after /health, as the shipped profile does
  [string]$Note = ''
)
# v2 (2026-09-23): binary, ub and extra flags are parameters and recorded per
# row; every request runs with cache_prompt=false and a cache hit is refused
# (sv-lib Send-Request). pre32k is gone: it cannot fit ctx 16,384.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1

if (-not $Exe) { $Exe = $global:MIRROR_EXE }
if ($Batch -le 0) { $Batch = [Math]::Max(2048, $Ub) }
$probeDir = "$global:STUDY\prompts"
# powershell -File hands 'decode,pre4k' over as ONE string; split it here.
$Probes = @($Probes | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Extra = @($Extra | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$exeVersion = Get-ExeLabel $Exe
Write-Host "exe=$Exe version=$exeVersion"
if ($PSBoundParameters.ContainsKey('Exe') -and $Exe -ne $PSBoundParameters['Exe']) {
  throw "Exe was rebound: asked $($PSBoundParameters['Exe']), got $Exe"
}

function Base-Row {
  return @{ run_id=$RunId; exe=$Exe; exe_version=$exeVersion; ctx=$Ctx; b=$Batch; ub=$Ub; ts=$Ts; sm=$Sm;
            ncmoe=$NcMoE; t=$Threads; tb=$ThreadsBatch; extra=($Extra -join ' ') }
}

$proc = Start-FlashServer -Ctx $Ctx -NcMoE $NcMoE -Ts $Ts -Threads $Threads -ThreadsBatch $ThreadsBatch `
          -Batch $Batch -Ubatch $Ub -Port $Port -Exe $Exe -Extra $Extra -Sm $Sm -LogName ($RunId + '.log')
if (-not (Wait-Healthy -Port $Port -TimeoutSec 600)) {
  $gist = ''
  $err = $global:BOOTLOG -replace '\.log$','-err.log'
  if (Test-Path $err) { $gist = (Get-Content $err -Tail 6 -ErrorAction SilentlyContinue) -join ' | ' }
  $row = Base-Row; $row.success = 'FAIL'; $row.failure_reason = 'boot-failed'; $row.notes = $gist
  Add-Result $row
  Write-Log ("FAIL boot $RunId")
  Clear-Server
  exit 1
}
Write-Log ("healthy $RunId")
if ($Trim) {
  Add-Type -Namespace W -Name Psapi -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(System.IntPtr hProcess);'
  $srvp = Get-Process llama-server | Select-Object -First 1
  [void][W.Psapi]::EmptyWorkingSet($srvp.Handle)
  Write-Log ("trimmed working set $RunId")
}
# A binary whose ggml-cuda.dll cannot load (missing CUDA 13 runtime DLLs,
# 2026-09-23) boots healthy on CPU only and returns ~5 tok/s. Refuse it.
$errLog = $global:BOOTLOG -replace '\.log$','-err.log'
if (Select-String -Path $errLog -Pattern 'no usable GPU found|compiled without GPU support' -Quiet) {
  $row = Base-Row; $row.success = 'FAIL'; $row.failure_reason = 'cpu-only-boot'; $row.notes = 'no usable GPU found'
  Add-Result $row
  Write-Log ("FAIL cpu-only boot $RunId")
  Clear-Server
  exit 1
}

function Emit {
  param($r, [string]$ProbeName, [string]$Phase)
  $used = Read-Vram
  $u = Read-GpuUtil
  $tel = Get-ProcTelemetry
  $row = Base-Row
  $row.probe = $ProbeName; $row.phase = $Phase
  $row.prompt_tokens = $r.prompt_tokens; $row.prompt_n = $r['prompt_n']; $row.output_tokens = $r.out_tokens
  $row.prompt_tps = [math]::Round($r.prompt_tps, 2); $row.decode_tps = [math]::Round($r.decode_tps, 2)
  $row.request_wall_s = $r.wall
  $row.gpu0_used_mb = $used[0]; $row.gpu1_used_mb = $used[1]; $row.gpu0_util = $u[0]; $row.gpu1_util = $u[1]
  $row.ram_used_mb = $tel.ramUsedMb; $row.ram_avail_mb = $tel.ramAvailMb; $row.proc_ws_mb = $tel.wsMb
  $row.success = if ($r.ok) { 'PASS' } else { 'FAIL' }
  $row.failure_reason = if ($r.ok) { '' } else { [string]$r.why }
  $row.notes = $Note
  Add-Result $row
  if ($r.ok) { Write-Log ("req ${RunId} ${ProbeName} ${Phase}: pt=$($r.prompt_tokens) pn=$($r['prompt_n']) ot=$($r.out_tokens) prefill=$([math]::Round($r.prompt_tps,2)) decode=$([math]::Round($r.decode_tps,2)) wall=$($r.wall)") }
  else { Write-Log ("req-fail ${RunId} ${ProbeName} ${Phase}: " + $r.why + ' raw=' + $r.raw) }
}

$files = @{ decode = 'decode512.json'; declong = 'declong.json'; pre4k = 'pre4k.json'; pre16k = 'pre16k.json' }
try {
  foreach ($p in $Probes) {
    # any other probe name resolves to prompts\<name>.json (deep prompts built per run)
    $f = if ($files.ContainsKey($p)) { $files[$p] } else { "$p.json" }
    if (-not (Test-Path (Join-Path $probeDir $f))) { throw "unknown probe '$p' (no $f)" }
    for ($i = 1; $i -le $Reps; $i++) {
      $r = Send-Request -PromptFile (Join-Path $probeDir $f) -Port $Port
      Emit $r $p ("rep$i")
    }
  }
} catch {
  $msg = ('SCRIPT-ERROR ' + $RunId + ': ' + $_.ToString()).Replace([char]13, ' ').Replace([char]10, ' ')
  Write-Log $msg
  Write-Manifest -RunId $RunId -Ts $Ts -NcMoE $NcMoE -Threads $Threads -ThreadsBatch $ThreadsBatch
  Clear-Server
  throw
}

Write-Manifest -RunId $RunId -Ts $Ts -NcMoE $NcMoE -Threads $Threads -ThreadsBatch $ThreadsBatch
Clear-Server
Write-Log ("done $RunId")
