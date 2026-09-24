Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$global:GPU0 = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4'
$global:GPU1 = 'GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e'
$env:CUDA_VISIBLE_DEVICES = "$global:GPU0,$global:GPU1"
Write-Output 'env set; now raw calls'
$saved = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = $null
try {
  Write-Output ('call0: ' + (nvidia-smi --query-gpu=memory.used --id=$global:GPU0 --format=csv,noheader,nounits 2>&1 | Out-String))
  Write-Output ('call1: ' + (nvidia-smi --query-gpu=memory.used --id=$global:GPU1 --format=csv,noheader,nounits 2>&1 | Out-String))
} finally {
  $env:CUDA_VISIBLE_DEVICES = $saved
}
Write-Output 'restored'
