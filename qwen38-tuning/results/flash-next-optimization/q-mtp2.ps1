$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag 'mtp-n3-c4' -NcMoE 32 -NMax 3 -CacheMaxTokens 4
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag 'nospec-n32' -NcMoE 32 -NoSpec
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-mtp2 done')
