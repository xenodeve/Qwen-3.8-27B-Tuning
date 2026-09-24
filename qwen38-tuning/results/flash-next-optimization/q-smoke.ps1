$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\smoke-profile.ps1" -Profile 128k
Start-Sleep 5
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\smoke-profile.ps1" -Profile 262k
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-smoke done')
