# Wait for q-cache.ps1 to finish, then run the lost controls.
$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
while (-not ((Test-Path "$dir\q-cache.out") -and (Select-String -Path "$dir\q-cache.out" -Pattern 'queue done' -Quiet))) { Start-Sleep -Seconds 10 }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$dir\q-cache2.ps1"
