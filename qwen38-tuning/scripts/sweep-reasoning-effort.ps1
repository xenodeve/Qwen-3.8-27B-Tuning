<#
B2 -- reasoning_effort sweep (low | medium | xhigh).

Rationale: the Qwen3.8 chat template defaults reasoning_effort to 'xhigh'. At the
measured 6-7 tok/s, reasoning tokens are wall-clock, so this is a throughput
variable, not a quality-only one.

Everything except reasoning_effort is held fixed: same task, same tools, same
sampling profile (vendor thinking profile with the min_p correction), same seed.

MTP is OFF for this sweep -- per the continuation plan, pick an operational
reasoning profile first, then hold it fixed through the MTP matrix.

Usage:  .\sweep-reasoning-effort.ps1 -Repeats 2
#>
param(
  [int]$Repeats = 2,
  [string]$Root = 'C:\AI\qwen38-tuning',
  [string]$Endpoint = 'http://127.0.0.1:8080/v1/chat/completions'
)

$ErrorActionPreference = 'Stop'

$tools = @(
  @{ type='function'; function=@{ name='read_file'; description='Read a source file';
     parameters=@{ type='object'; properties=@{ path=@{type='string'} }; required=@('path') } } },
  @{ type='function'; function=@{ name='apply_patch'; description='Apply a unified diff to a file';
     parameters=@{ type='object'; properties=@{
        path=@{type='string'}
        diff=@{type='string'}
        options=@{ type='object'; properties=@{ backup=@{type='boolean'}; retries=@{type='integer'} } }
     }; required=@('path','diff') } } }
)

$task = @'
src/cache.py contains an LRU cache whose eviction is wrong: it evicts the most
recently used entry instead of the least recently used. Read the file, then apply
a patch that fixes the eviction order. Use the tools.
'@

$fileBody = @'
class LRUCache:
    def __init__(self, cap):
        self.cap = cap
        self.data = {}
        self.order = []

    def get(self, k):
        if k not in self.data:
            return None
        self.order.remove(k)
        self.order.append(k)
        return self.data[k]

    def put(self, k, v):
        if k in self.data:
            self.order.remove(k)
        elif len(self.data) >= self.cap:
            victim = self.order.pop()   # BUG: pops most-recently-used
            del self.data[victim]
        self.data[k] = v
        self.order.append(k)
'@

# Vendor thinking profile, with min_p corrected from the server default of 0.05.
$sampling = @{ temperature = 1.0; top_p = 0.95; top_k = 20; min_p = 0.0; presence_penalty = 0.0 }

$out = Join-Path $Root 'results\reasoning-effort-sweep.jsonl'
$rows = @()

foreach ($effort in @('low','medium','xhigh')) {
  for ($i = 1; $i -le $Repeats; $i++) {

    $msgs = [System.Collections.ArrayList]@(
      @{ role='developer'; content='You are a coding agent. Inspect before editing. One tool call per step.' },
      @{ role='user';      content=$task }
    )

    $sw            = [Diagnostics.Stopwatch]::StartNew()
    $reasoningChars= 0
    $completionToks= 0
    $toolCalls     = 0
    $badArgs       = 0
    $patched       = $false
    $rounds        = 0

    for ($round = 1; $round -le 4; $round++) {
      $rounds = $round
      $body = ($sampling.Clone())
      $body.messages             = $msgs
      $body.tools                = $tools
      $body.max_tokens           = 4096
      $body.chat_template_kwargs = @{ reasoning_effort = $effort }

      $r = Invoke-RestMethod $Endpoint -Method Post -ContentType 'application/json' `
             -Body ($body | ConvertTo-Json -Depth 12) -TimeoutSec 1800

      $m = $r.choices[0].message
      $completionToks += [int]$r.usage.completion_tokens
      if ($m.reasoning_content) { $reasoningChars += $m.reasoning_content.Length }

      if (-not $m.tool_calls) { break }

      foreach ($tc in $m.tool_calls) {
        $toolCalls++
        # A tool call whose arguments do not parse is a hard protocol failure.
        try { $null = $tc.function.arguments | ConvertFrom-Json } catch { $badArgs++ }
        if ($tc.function.name -eq 'apply_patch') { $patched = $true }
      }

      [void]$msgs.Add(@{
        role='assistant'; content=[string]$m.content
        tool_calls=@($m.tool_calls | ForEach-Object {
          @{ id=$_.id; type='function'; function=@{ name=$_.function.name; arguments=$_.function.arguments } } })
      })
      foreach ($tc in $m.tool_calls) {
        $res = if ($tc.function.name -eq 'read_file') { $fileBody } else { 'patch applied successfully' }
        [void]$msgs.Add(@{ role='tool'; tool_call_id=$tc.id; content=$res })
      }
    }

    $sw.Stop()
    $row = [ordered]@{
      effort            = $effort
      run               = $i
      wall_s            = [math]::Round($sw.Elapsed.TotalSeconds, 1)
      completion_tokens = $completionToks
      reasoning_chars   = $reasoningChars
      rounds            = $rounds
      tool_calls        = $toolCalls
      bad_arg_parses    = $badArgs
      reached_patch     = $patched
    }
    $rows += [pscustomobject]$row
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $out -Encoding utf8
    "$effort run $i -> $($row.wall_s)s, $completionToks tok, patch=$patched"
  }
}

"`n=== summary ==="
$rows | Format-Table -AutoSize
"-> $out"
