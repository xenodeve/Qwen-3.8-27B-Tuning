# Correctness smoke for a CPU kernel change: same prompts, temperature 0, A vs B.
param([string]$Exe, [string]$Tag, [string[]]$Extra = @())
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
$null = Start-FlashServer -NcMoE 24 -Ts '30,18' -Ubatch 2048 -Batch 2048 -Sm layer -Exe $Exe -Extra @($Extra | ForEach-Object { $_ -split ',' } | Where-Object { $_ }) -LogName ("answer-$Tag.log")
if (-not (Wait-Healthy -Port 8099)) { Clear-Server; throw 'boot failed' }
$qs = @(
  'What is 17 * 23? Answer with the number only.',
  'Write a Python function is_prime(n) that returns True if n is prime. Code only, no explanation.',
  'Translate to Thai: "The server restarted and the benchmark is running again."',
  'Explain in about 300 words how a hash map handles collisions, with a short Python example.'
)
$out = @()
foreach ($q in $qs) {
  $body = @{ messages = @(@{ role='user'; content=$q }); temperature = 0; max_tokens = 600; cache_prompt = $false; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 6 -Compress
  $tmp = Join-Path $env:TEMP ("ans-" + [guid]::NewGuid().ToString('N') + '.json')
  [IO.File]::WriteAllText($tmp, $body, (New-Object Text.UTF8Encoding($false)))
  $outTmp = $tmp + '.out'
  curl.exe -s --max-time 600 -o $outTmp http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d "@$tmp"
  $r = [IO.File]::ReadAllText($outTmp, [Text.Encoding]::UTF8) | ConvertFrom-Json
  Remove-Item $outTmp
  Remove-Item $tmp
  $out += ('### Q: ' + $q); $out += [string]$r.choices[0].message.content; $out += ('(decode ' + [math]::Round($r.timings.predicted_per_second,2) + ' tok/s, ' + $r.usage.completion_tokens + ' tokens)'); $out += ''
}
[IO.File]::WriteAllLines((Join-Path $global:STUDY "logs\answer-$Tag.txt"), $out, (New-Object Text.UTF8Encoding($false)))
Clear-Server
