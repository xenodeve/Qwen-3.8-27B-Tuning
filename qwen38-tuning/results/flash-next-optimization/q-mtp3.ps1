# Wait for q-mtp2, rebuild (8-token cache cap), then n=4 vs n=3 ABBA.
$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
while (-not (Select-String -Path "$d\logs\study.log" -Pattern 'q-mtp2 done' -Quiet)) { Start-Sleep -Seconds 10 }
& cmd /c 'C:\AI\llama.cpp-upstream-fn\build-c.cmd' *> 'C:\AI\llama.cpp-upstream-fn\build-c-mtp4.out.log'
$ver = (& 'C:\AI\llama.cpp-upstream-fn\build-c\bin\llama-server.exe' --version 2>&1 | Select-String 'version').Line
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  q-mtp3 build rc=$LASTEXITCODE $ver")
if ($ver -notmatch '7a0ac666') { Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-mtp3 ABORT: binary is not 7a0ac666'); exit 1 }
foreach ($a in @(@('mtp-n4-c5-a',4,5), @('mtp-n3-c4-a',3,4), @('mtp-n3-c4-b',3,4), @('mtp-n4-c5-b',4,5))) {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$d\diag-mtp.ps1" -Tag $a[0] -NcMoE 32 -NMax $a[1] -CacheMaxTokens $a[2]
}
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + '  q-mtp3 done')
