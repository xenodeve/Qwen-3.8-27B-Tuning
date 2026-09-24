Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
try {
  $t0 = Get-Date
  $resp = curl.exe -s --max-time 900 "http://127.0.0.1:8099/v1/chat/completions" `
            -H 'Content-Type: application/json' -d "@C:\AI\qwen38-tuning\results\flash-next-optimization\prompts\decode512.json"
  $wall = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
  $j = $resp | ConvertFrom-Json
  Write-Output ("wall=" + $wall)
  Write-Output ("pt=" + $j.usage.prompt_tokens + " ot=" + $j.usage.completion_tokens)
  Write-Output ("prefill=" + [math]::Round($j.timings.prompt_per_second, 2) + " decode=" + [math]::Round($j.timings.predicted_per_second, 2))
  Write-Output ("content=" + $j.choices[0].message.content)
} catch {
  Write-Output ('ERROR: ' + $Error[0].ToString())
  Write-Output $resp
}
