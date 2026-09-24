Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
New-Item -ItemType Directory -Path "$dir\prompts","$dir\logs","$dir\run-manifests" -Force | Out-Null

$seed = @'
Consider the following repository structure and review feedback. The module exports parse_ticket(text) which returns (summary, priority, assignee) where priority is derived from a weighted combination of bug class, blast radius, and deadline distance, with weights 0.45, 0.30 and 0.25 respectively. Keep each sentence plain, factual and easy to tokenize, containing a mix of identifiers, numbers and English prose so the prefill probe exercises realistic tool-shaped input. The reviewer asked whether reordering the weighted combination changes the median latency of the batch aggregator in aggregator.py. A second reviewer noted that the deadline distance term should saturate beyond 30 days instead of growing without bound. Response format guidance says the answer must name the weights and the file in the first sentence.
'@

foreach ($pair in @(@('pre4k',15500), @('pre16k',62000), @('pre32k',124000))) {
  $name = $pair[0]; $goal = $pair[1]
  $sb = New-Object System.Text.StringBuilder
  while ($sb.Length -lt $goal) { [void]$sb.Append($seed) }
  $text = $sb.ToString().Substring(0, $goal)
  $user = 'Summarize the review notes above in one sentence, then list the three weights.'
  $req = @{ messages = @(
      @{ role = 'user'; content = ($text + "`n`n" + $user) }
    ); max_tokens = 64; temperature = 0 }
  $path = Join-Path "$dir\prompts" ($name + '.json')
  $req | ConvertTo-Json -Depth 4 -Compress | Set-Content -Path $path -Encoding UTF8
}

$sumReq = @'
{"messages":[{"role":"user","content":"Write a Python one-liner that prints the sum of 1..10. Output only the code."}],"max_tokens":512,"temperature":0}
'@
$sumReq | Set-Content -Path "$dir\prompts\decode512.json" -Encoding UTF8
$sumReq -Replace '"max_tokens":512','"max_tokens":64' | Set-Content -Path "$dir\prompts\decode64.json" -Encoding UTF8

Get-ChildItem "$dir\prompts" | ForEach-Object { $_.Name + '  ' + $_.Length + ' bytes' }
