# Boot the SHIPPED profile (serve-flash-next.ps1) on 8099, wait for the trim line,
# one short request with a check that speculation ran, then stop.
param([string]$Profile)
$ErrorActionPreference = 'Stop'
$d = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$p = Start-Process pwsh -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\AI\qwen38-tuning\scripts\serve-flash-next.ps1','-Profile',$Profile,'-Port','8099' -WindowStyle Hidden -PassThru
$ok = $false
for ($i = 0; $i -lt 120; $i++) { Start-Sleep 5; try { if ((Invoke-RestMethod http://127.0.0.1:8099/health -TimeoutSec 3).status -eq 'ok') { $ok = $true; break } } catch {} ; if ($p.HasExited) { break } }
$line = "smoke[$Profile] healthy=$ok"
if ($ok) {
  Start-Sleep 8
  $body = @{ messages = @(@{ role='user'; content='Write a Python function that returns the n-th Fibonacci number iteratively. Code only.' }); max_tokens = 200; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 5
  $r = Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
  $t = $r.timings
  $g = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join '/'
  $trim = (Get-ChildItem C:\AI\qwen38-tuning\logs -Filter "flash-next-$Profile-*-trim.txt" | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content)
  $line += " decode=$([math]::Round($t.predicted_per_second,1)) draft_n=$($t.draft_n) accepted=$($t.draft_n_accepted) vram=$g trim=[$trim] answer_ok=$($r.choices[0].message.content -match 'def ')"
}
Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'pwsh.exe' -and $_.CommandLine -match 'serve-flash-next.ps1') -or ($_.Name -eq 'llama-server.exe' -and $_.CommandLine -match '--port 8099') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Add-Content "$d\logs\study.log" ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + "  $line")
