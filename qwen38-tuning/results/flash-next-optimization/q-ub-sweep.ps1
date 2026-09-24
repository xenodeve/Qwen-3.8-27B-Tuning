# ub sweep on the deployed binary (unsloth mirror, -sm tensor). Post-restart
# (boot 2026-09-23 08:38): the earlier runs sat at 1.7 GB free RAM.
$ErrorActionPreference = 'Continue'
$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$plan = @(512, 2048, 1024, 4096, 512, 2048)
$n = 0
foreach ($ub in $plan) {
  $n++
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$dir\fn-config-run.ps1" `
      -RunId ("ub$ub-r$n") -Ub $ub -Probes 'decode,pre4k,pre16k' -Reps 1 -Note 'ub-sweep mirror post-restart' 2>&1 | Select-Object -Last 3
  Add-Content "$dir\q-ub-sweep.out" ("queue item $n ub=$ub rc=$LASTEXITCODE " + (Get-Date -Format 'HH:mm:ss'))
}
Add-Content "$dir\q-ub-sweep.out" 'queue done'
