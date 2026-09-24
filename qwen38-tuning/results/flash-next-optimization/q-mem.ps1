$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mem.ps1" -Trim -Tag trim1
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mem.ps1" -Tag notrim2
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-mem queue done')
