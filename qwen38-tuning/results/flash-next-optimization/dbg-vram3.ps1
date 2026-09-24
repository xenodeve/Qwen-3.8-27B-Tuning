Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4,GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e'
Write-Output 'with CUDA_VISIBLE_DEVICES set, index query:'
Write-Output ('i0=' + (nvidia-smi --query-gpu=memory.used,name --id=0 --format=csv,noheader,nounits 2>&1 | Out-String))
Write-Output 'with CUDA_VISIBLE_DEVICES cleared, index query:'
$saved = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = $null
try {
  Write-Output ('i0=' + (nvidia-smi --query-gpu=memory.used,name --id=0 --format=csv,noheader,nounits 2>&1 | Out-String))
  Write-Output ('i1=' + (nvidia-smi --query-gpu=memory.used,name --id=1 --format=csv,noheader,nounits 2>&1 | Out-String))
} finally { $env:CUDA_VISIBLE_DEVICES = $saved }
