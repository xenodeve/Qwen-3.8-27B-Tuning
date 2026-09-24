$run = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
Remove-Item -ErrorAction SilentlyContinue "$run\results.csv"
Remove-Item -ErrorAction SilentlyContinue "$run\logs\smoke2-stdout.log","$run\logs\smoke2-stderr.log"
Start-Process pwsh -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\AI\qwen38-tuning\results\flash-next-optimization\fn-config-run.ps1','-RunId','smoke2-baseline','-Ts','8449,15452','-NcMoE','34','-Probes','decode','-DecodeRepeats','1' -WindowStyle Hidden -RedirectStandardOutput "$run\logs\smoke2-stdout.log" -RedirectStandardError "$run\logs\smoke2-stderr.log"
Write-Output 'detached'
