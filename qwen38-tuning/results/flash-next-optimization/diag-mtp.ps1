# MTP feasibility: does the unsloth shared-Q4_K_M head load against the ISTA
# GSQ-RCO target, fit on GPU0 once -ncmoe frees it, and actually draft?
param([int]$NcMoE = 30, [string]$Ts = '31,17', [int]$NMax = 2, [string]$Tag = 'mtp-probe', [int]$CacheMaxTokens = 1, [switch]$NoSpec, [switch]$Duo, [int]$Ctx = 131072, [int]$Ub = 1024, [string]$DraftDev = 'CUDA0', [int]$NgMatch = 12, [int]$NgMin = 16, [int]$NgMax = 32, [switch]$NgramOnly, [string]$Exe = 'C:\AI\llama.cpp-upstream-fn\build-c\bin\llama-server.exe')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
Add-Type -Namespace W -Name Psapi -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(System.IntPtr hProcess);'
$head = 'C:\AI\models\unsloth-Qwen3.8-Flash-Next-MTP\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf'
$env:LLAMA_MOE_CACHE_MAX_TOKENS = "$CacheMaxTokens"
$env:SPEC_SAVE_DIR = Join-Path $global:STUDY "logs\spec-content"
# Per-process GPU memory split. nvidia-smi 'used' hides a WDDM over-commit; the
# shared counter shows it (2026-09-24: 262k+MTP spilled ~0.6 GB, prefill /5).
# A clean server sits near 500-560 MiB shared per adapter (pinned host buffers).
function Get-SharedLine([int]$ProcId) {
  $c = Get-Counter "\GPU Process Memory(pid_$ProcId*)\Shared Usage","\GPU Process Memory(pid_$ProcId*)\Dedicated Usage" -ErrorAction SilentlyContinue
  if (-not $c) { return 'shared=NA' }
  ($c.CounterSamples | ForEach-Object { ($_.Path.Split('')[-1] -replace ' usage','') + '=' + [int]($_.CookedValue/1MB) }) -join ' '
}
$smiLog = Join-Path $global:STUDY "logs\$Tag-smi.csv"
$smi = Start-Process nvidia-smi -ArgumentList '--query-gpu=index,memory.used','--format=csv,noheader,nounits','-lms','500' -RedirectStandardOutput $smiLog -WindowStyle Hidden -PassThru
try {
  $null = Start-FlashServer -Ctx $Ctx -NcMoE $NcMoE -Ts $Ts -Ubatch $Ub -Batch 2048 -Sm layer `
            -Exe $Exe `
            -Extra $(if ($NgramOnly) { @('--moe-expert-cache','64','-lv','4','--spec-type','ngram-mod','--spec-ngram-mod-n-match',"$NgMatch",'--spec-ngram-mod-n-min',"$NgMin",'--spec-ngram-mod-n-max',"$NgMax") } elseif ($NoSpec) { @('--moe-expert-cache','64','-lv','4','-ot','output.weight=CUDA0') } else { @('--moe-expert-cache','64','-lv','4','-md',$head,'--spec-type',$(if ($Duo) { 'draft-mtp,ngram-mod' } else { 'draft-mtp' }),'--spec-draft-n-max',"$NMax",'--spec-ngram-mod-n-match',"$NgMatch",'--spec-ngram-mod-n-min',"$NgMin",'--spec-ngram-mod-n-max',"$NgMax",'-ngld','99','-devd',$DraftDev,'-ctkd','q4_0','-ctvd','q4_0') + $(if ($DraftDev -eq 'CUDA0') { @('-ot','output.weight=CUDA0') } else { @() }) }) `
            -LogName "$Tag.log"
  if (-not (Wait-Healthy -Port 8099 -TimeoutSec 600)) { Write-Log "diag-mtp[$Tag] BOOT FAIL"; throw 'boot failed' }
  $srv = Get-Process llama-server | Select-Object -First 1
  [void][W.Psapi]::EmptyWorkingSet($srv.Handle)
  Write-Log "diag-mtp[$Tag] healthy ncmoe=$NcMoE ts=$Ts nmax=$NMax cache_max_tokens=$CacheMaxTokens duo=$Duo ctx=$Ctx ub=$Ub draftdev=$DraftDev ngram=$NgMatch/$NgMin/$NgMax ngramonly=$NgramOnly exe=$Exe"
  Write-Log ("diag-mtp[$Tag] after-boot " + (Get-SharedLine $srv.Id))
  & python "$global:STUDY\spec-bench.py" 8099 $Tag 1 "$global:STUDY\spec-mtp.csv" 2>&1 | ForEach-Object { Write-Log "diag-mtp[$Tag] $_" }
  Write-Log ("diag-mtp[$Tag] after-bench " + (Get-SharedLine $srv.Id))
} catch {
  Write-Log ("diag-mtp[$Tag] SCRIPT-ERROR " + ($_.ToString() -replace '[\r\n]+',' '))
} finally {
  Stop-Process -Id $smi.Id -Force -ErrorAction SilentlyContinue
  $peak = @{}
  foreach ($l in (Get-Content $smiLog -ErrorAction SilentlyContinue)) { $f = $l -split ',\s*'; if ($f.Count -ge 2 -and $f[1] -match '^\d+$') { $v=[int]$f[1]; if (-not $peak.ContainsKey($f[0]) -or $v -gt $peak[$f[0]]) { $peak[$f[0]] = $v } } }
  Write-Log ("diag-mtp[$Tag] peak VRAM gpu0=" + $peak['0'] + ' gpu1=' + $peak['1'])
  Clear-Server
  Write-Log "diag-mtp[$Tag] done"
}
