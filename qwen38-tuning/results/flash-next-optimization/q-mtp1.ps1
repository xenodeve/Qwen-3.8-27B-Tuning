$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag 'mtp-n2-c1' -NcMoE 32 -NMax 2 -CacheMaxTokens 1
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag 'mtp-n2-c3' -NcMoE 32 -NMax 2 -CacheMaxTokens 3
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-mtp1 done')
