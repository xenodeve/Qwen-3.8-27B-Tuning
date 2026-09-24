# ngram-mod n-max 64 on the shipped 128k duo (MTP n3 + ngram-mod), 2026-09-24.
# 27B measured n-max 32->64 at +14.5..15.6 % (02-decoders.md:1646-1667); on
# Flash-Next a 65-token verify batch crosses GGML_OP_OFFLOAD_MIN_BATCH 32 and
# streams host experts over PCIe, so it may not transfer. Arms:
#   C = 12/16/32 (served)   A = 12/16/64 (n-max only)   B = 24/16/64 (NVFP4/gsq)
# Order C A B B A C: pairs within one sitting, order reversed on the way back.
$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
function Arm([string]$tag, [int]$m, [int]$lo, [int]$hi) {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag $tag -NcMoE 32 -NMax 3 -CacheMaxTokens 4 -Duo -NgMatch $m -NgMin $lo -NgMax $hi
}
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-ng64 start')
Arm 'ng-12-16-32-a' 12 16 32
Arm 'ng-12-16-64-a' 12 16 64
Arm 'ng-24-16-64-a' 24 16 64
Arm 'ng-24-16-64-b' 24 16 64
Arm 'ng-12-16-64-b' 12 16 64
Arm 'ng-12-16-32-b' 12 16 32
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-ng64 done')
