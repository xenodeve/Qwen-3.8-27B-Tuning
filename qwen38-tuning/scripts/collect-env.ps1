<#
Captures the machine state that MUST be recorded before every experiment.

Rationale: free VRAM is not constant on this box. --list-devices reported
11069 MiB free while nvidia-smi later reported 9361 MiB free (2637 MiB in use
by desktop/browser/Ollama). Because --fit on decides the layer split from
whatever is free AT BOOT, two runs with identical flags can produce different
splits. Without this snapshot the Phase A/F comparisons are not controlled.

Usage:  .\collect-env.ps1 -Label "phaseA-q4-16k"
#>
param(
  [Parameter(Mandatory = $true)][string]$Label,
  [string]$Root = 'C:\AI\qwen38-tuning'
)

$ErrorActionPreference = 'Stop'
$llama = 'C:\AI\llama.cpp-cuda\llama-server.exe'

# Native tools here (llama-server --version, nvidia-smi) write to stderr on
# success. Under Windows PowerShell 5.1 that becomes a terminating
# NativeCommandError while ErrorActionPreference is 'Stop', which killed this
# script before it could hand off to the launcher. Run native calls with
# 'Continue' and judge them by exit code instead.
function Invoke-Native {
  param([scriptblock]$Command)
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $Command 2>&1 } finally { $ErrorActionPreference = $prev }
}

$gpu = ((Invoke-Native { nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,compute_cap `
                                    --format=csv,noheader,nounits }) | Select-Object -First 1) -split '\s*,\s*'
$os  = Get-CimInstance Win32_OperatingSystem
$ver = ((Invoke-Native { & $llama --version }) | Out-String).Trim()

# Anything already holding VRAM will shrink the split available to llama.cpp.
$vramHolders = @(Invoke-Native { nvidia-smi --query-compute-apps=pid,process_name,used_memory `
                                            --format=csv,noheader } | Where-Object { $_ -match '\S' })

$snap = [ordered]@{
  label              = $Label
  timestamp          = (Get-Date).ToString('o')
  build              = $ver
  gpu_name           = $gpu[0]
  driver             = $gpu[1]
  compute_cap        = $gpu[5]
  vram_total_mib     = [int]$gpu[2]
  vram_used_mib      = [int]$gpu[3]
  vram_free_mib      = [int]$gpu[4]
  vram_holders       = $vramHolders
  ram_total_mib      = [int]($os.TotalVisibleMemorySize / 1KB)
  ram_free_mib       = [int]($os.FreePhysicalMemory / 1KB)
}

$out = Join-Path $Root 'results\env-snapshots.jsonl'
($snap | ConvertTo-Json -Compress -Depth 4) | Add-Content -Path $out -Encoding utf8

$snap | Format-List
"`n-> appended to $out"
