# Download Qwen3.8-Flash-Next GSQ-RCO Q2_0 (2 shards) for Phase 0A.
#
# Measured 2026-09-22 on this machine:
#   - `hf download` stalls: it creates the .incomplete placeholders and then
#     transfers 0 bytes (hf 1.24.0, XET disabled). Do not use it for this repo.
#   - curl through the resolve URL works at ~11 MB/s.
#   - BUT `curl -C -` stalls at ~16 KiB: the xet-bridge CDN does not serve a
#     `Range: bytes=0-` request properly. Use a plain GET, no `-C -`.
# A partial file is therefore removed and re-fetched rather than resumed.
$ErrorActionPreference = 'Continue'

$dest = "C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0"
$base = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Q2_0"
$log  = "C:\AI\qwen38-tuning\logs\flash-next-download.log"

New-Item -ItemType Directory -Path $dest -Force | Out-Null

# expected, from HF LFS metadata (recorded 2026-09-22, repo revision 2c4721899...)
$expected = @{
  "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" = @{ size = 37623740192; sha256 = "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720" }
  "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" = @{ size = 28800138432; sha256 = "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113" }
}

foreach ($n in @("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
                 "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf")) {
  $out = Join-Path $dest $n
  $exp = $expected[$n]
  $have = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
  Add-Content $log "$(Get-Date -Format o)  START $n  have=$have  want=$($exp.size)"
  if ($have -eq $exp.size) { Add-Content $log "$(Get-Date -Format o)  SKIP  $n already complete"; continue }
  if ($have -gt 0) { Remove-Item $out -Force; Add-Content $log "$(Get-Date -Format o)  removed partial $have, refetching" }

  & curl.exe -L -f -sS --retry 30 --retry-delay 5 --retry-connrefused -o $out "$base/$n" 2>&1 | Add-Content $log
  $rc  = $LASTEXITCODE
  $now = (Get-Item $out -ErrorAction SilentlyContinue).Length
  Add-Content $log "$(Get-Date -Format o)  STOP  $n  exit=$rc  bytes=$now  want=$($exp.size)"
}

Add-Content $log "$(Get-Date -Format o)  ALL DONE"
