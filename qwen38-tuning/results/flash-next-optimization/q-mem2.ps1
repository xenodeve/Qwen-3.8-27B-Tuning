$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mem.ps1" -Trim -Tag trim3
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mem.ps1" -Tag notrim4
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mem.ps1" -Tag notrim5
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mem.ps1" -Trim -Tag trim6
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-mem2 queue done')
