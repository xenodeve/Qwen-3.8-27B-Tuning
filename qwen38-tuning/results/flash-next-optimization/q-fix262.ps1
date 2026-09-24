# 262k with the MTP draft on the 5060 Ti (nothing else uses it), 2026-09-24.
# Same approach as the 128k fix (q-ncmoe-hr): draft + output.weight on CUDA1,
# -ncmoe raised so CUDA1 has room (layers 31+ live there under -ts 31,17).
# Control = the shipped 262k profile I: -ncmoe 28, ngram-mod 24/16/64 only, on
# the pre-MTP binary. Each boot logs desktop VRAM and free-before-draft.
# Order: ctl n34 n36 n38 n36 ctl.
$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
function Desk {
  $s = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.CookedValue -gt 50MB }
  ($s | ForEach-Object { $n = if ($_.InstanceName -match 'pid_(\d+)') { (Get-Process -Id $matches[1] -ErrorAction SilentlyContinue).ProcessName } else { '?' }; "$n=$([int]($_.CookedValue/1MB))" }) -join ' '
}
function Arm([string]$tag, [string[]]$a) {
  Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  q-fix262[$tag] desktop-before " + (Desk))
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag $tag -Ctx 262144 -Ub 512 -NgMatch 24 -NgMin 16 -NgMax 64 @a
  $err = "$d\logs\$tag-err.log"
  $free = if (Test-Path $err) { (Select-String -Path $err -Pattern 'using device (CUDA\d) .* - (\d+) MiB free' | ForEach-Object { $_.Matches[0].Groups[1].Value + '=' + $_.Matches[0].Groups[2].Value }) -join ',' } else { 'no-log' }
  Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  q-fix262[$tag] free_seq=$free")
}
$ctl = @('-NcMoE','28','-NgramOnly','-Exe','C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe')
function Duo([int]$n) { @('-NcMoE',"$n",'-NMax','3','-CacheMaxTokens','4','-Duo','-DraftDev','CUDA1') }
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-fix262 start')
Arm 'f262-ctl-a' $ctl
Arm 'f262-n34-a' (Duo 34)
Arm 'f262-n36-a' (Duo 36)
Arm 'f262-n38-a' (Duo 38)
Arm 'f262-n36-b' (Duo 36)
Arm 'f262-ctl-b' $ctl
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-fix262 done')
