Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$global:GPU0 = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4'
$global:GPU1 = 'GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e'
$env:CUDA_VISIBLE_DEVICES = "$global:GPU0,$global:GPU1"
function Read-Vram {
  $r0 = 0; $r1 = 0
  $saved = $env:CUDA_VISIBLE_DEVICES
  $env:CUDA_VISIBLE_DEVICES = $null
  try {
    foreach ($pair in @(@($global:GPU0, [ref]$r0), @($global:GPU1, [ref]$r1))) {
      Write-Output ("about to call for " + $pair[0].GetType().Name + " " + $pair[0])
      $raw = nvidia-smi --query-gpu=memory.used --id=$pair[0] --format=csv,noheader,nounits 2>&1
      Write-Output ("raw=[" + ($raw | Out-String) + "]")
      $line = $raw | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1
      Write-Output ("line=[" + $line + "]")
      if ($null -eq $line) { $pair[1].Value = 0 } else { $pair[1].Value = [int]$line }
    }
  } finally { $env:CUDA_VISIBLE_DEVICES = $saved }
  return ,@($r0, $r1)
}
$v = Read-Vram
Write-Output ("RESULT: " + ($v -join ','))
