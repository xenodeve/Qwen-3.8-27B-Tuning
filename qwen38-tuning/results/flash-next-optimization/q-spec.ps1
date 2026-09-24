# ngram-mod speculative decoding on the shipped 128k config (+ working-set trim).
# S0 none | S1 12/16/32 (27B incumbent; verify batch 33 >= op-offload 32)
# S2 12/8/24 | S3 12/4/12 | S4 llama.cpp default 24/48/64. S0 again last.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
Add-Type -Namespace W -Name Psapi -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(System.IntPtr hProcess);'
$exe = 'C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe'
$csv = Join-Path $global:STUDY 'spec-ngram.csv'
function Spec([int]$m, [int]$lo, [int]$hi) { @('--spec-type','ngram-mod','--spec-ngram-mod-n-match',"$m",'--spec-ngram-mod-n-min',"$lo",'--spec-ngram-mod-n-max',"$hi") }
$plan = @(
  @{ arm='S0-none';     spec=@() },
  @{ arm='S3-12-4-12';  spec=(Spec 12 4 12) },
  @{ arm='S2-12-8-24';  spec=(Spec 12 8 24) },
  @{ arm='S1-12-16-32'; spec=(Spec 12 16 32) },
  @{ arm='S4-24-48-64'; spec=(Spec 24 48 64) },
  @{ arm='S0-none-b';   spec=@() }
)
try {
foreach ($p in $plan) {
  $null = Start-FlashServer -Ctx 131072 -NcMoE 26 -Ts '31,17' -Ubatch 1024 -Batch 2048 -Sm layer -Exe $exe `
            -Extra (@('--moe-expert-cache','64','-lv','4') + $p.spec) -LogName ("spec-$($p.arm).log")
  if (-not (Wait-Healthy -Port 8099)) { Write-Log "spec $($p.arm) BOOT FAIL"; Clear-Server; continue }
  $srv = Get-Process llama-server | Select-Object -First 1
  [void][W.Psapi]::EmptyWorkingSet($srv.Handle)
  $err = $global:BOOTLOG -replace '\.log$','-err.log'
  $m = Select-String -Path $err -Pattern 'spec common|speculative' | Select-Object -Last 1
  $specLine = if ($m) { $m.Line } else { '(no spec line in log)' }
  Write-Log "spec $($p.arm) healthy; $specLine"
  & python "$global:STUDY\spec-bench.py" 8099 $p.arm 2 $csv 2>&1 | ForEach-Object { Write-Log "spec $_" }
  Clear-Server
}
} catch {
  Write-Log ('SCRIPT-ERROR q-spec: ' + ($_.ToString() -replace '[
]+',' ') + ' at ' + ($_.ScriptStackTrace -replace '[
]+',' '))
  Clear-Server
}
Write-Log 'spec queue done'
