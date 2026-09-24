# GPU0 headroom for the 128k duo, 2026-09-24. The 07:49 boot had 2,035 MiB free
# on CUDA0 before the MTP draft (needs ~2,530): desktop apps on the 4070 SUPER
# (display card) grew ~1 GB. The 5060 Ti runs nothing else, so the draft and
# output.weight move there (-devd CUDA1, no -ot), and -ncmoe 34/36 frees room on
# CUDA1 (layers 32+ live there under -ts 31,17; -ncmoe cannot free GPU0 here).
# Desktop left as the developer uses it; each boot logs other processes' VRAM.
# Order: ctl A B B A ctl.
$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
function Desk {
  $s = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples |
       Where-Object { $_.CookedValue -gt 50MB -and $_.InstanceName -notmatch "pid_$((Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object -First 1).Id)_" }
  ($s | ForEach-Object { $n = if ($_.InstanceName -match 'pid_(\d+)') { (Get-Process -Id $matches[1] -ErrorAction SilentlyContinue).ProcessName } else { '?' }; "$n=$([int]($_.CookedValue/1MB))" }) -join ' '
}
function Arm([string]$tag, [int]$n, [string]$dev) {
  Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  q-ncmoe-hr[$tag] desktop-before " + (Desk))
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag $tag -NcMoE $n -NMax 3 -CacheMaxTokens 4 -Duo -NgMatch 24 -NgMin 16 -NgMax 64 -DraftDev $dev
  $err = "$d\logs\$tag-err.log"
  $free = if (Test-Path $err) { (Select-String -Path $err -Pattern 'using device (CUDA\d) .* - (\d+) MiB free' | ForEach-Object { $_.Matches[0].Groups[1].Value + '=' + $_.Matches[0].Groups[2].Value }) -join ',' } else { 'no-log' }
  Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  q-ncmoe-hr[$tag] free_seq=$free")
}
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-ncmoe-hr start')
Arm 'hr-ctl-n32-d0-a' 32 'CUDA0'
Arm 'hr-n34-d1-a'     34 'CUDA1'
Arm 'hr-n36-d1-a'     36 'CUDA1'
Arm 'hr-n36-d1-b'     36 'CUDA1'
Arm 'hr-n34-d1-b'     34 'CUDA1'
Arm 'hr-ctl-n32-d0-b' 32 'CUDA0'
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-ncmoe-hr done')
