# Deep fills on configs that booted at their context. Per run: nvidia-smi
# logger at 1 s for peak VRAM per card DURING prefill, not just at boot.
$ErrorActionPreference = 'Continue'
$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$exe = 'C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe'
$plan = @(
  @{ id='D-A-64k-n24';   ctx=65536;  nm=24; ts='30,18'; ub=2048; probes='declong,pre16k,pre60k' },
  @{ id='D-B-128k-n24';  ctx=131072; nm=24; ts='30,18'; ub=1024; probes='declong,pre16k,pre60k,pre120k' },
  @{ id='D-C-128k-n26';  ctx=131072; nm=26; ts='31,17'; ub=1024; probes='declong,pre16k,pre60k,pre120k' },
  @{ id='D-D-256k-n28';  ctx=262144; nm=28; ts='31,17'; ub=512;  probes='declong,pre16k,pre60k,pre120k,pre240k' },
  @{ id='D-E-256k-n26';  ctx=262144; nm=26; ts='31,17'; ub=512;  probes='declong,pre240k' }
)
foreach ($p in $plan) {
  $smiLog = "$dir\logs\$($p.id)-smi.csv"
  $smi = Start-Process nvidia-smi -ArgumentList '--query-gpu=timestamp,uuid,memory.used','--format=csv,noheader,nounits','-l','1' `
           -RedirectStandardOutput $smiLog -WindowStyle Hidden -PassThru
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$dir\fn-config-run.ps1" `
      -RunId $p.id -Exe $exe -Ctx $p.ctx -NcMoE $p.nm -Ts $p.ts -Ub $p.ub -Sm layer `
      -Extra '--moe-expert-cache,64' -Probes $p.probes -Reps 1 -Note 'deep-fill' 2>&1 | Select-Object -Last 3
  $rc = $LASTEXITCODE
  Stop-Process -Id $smi.Id -Force -ErrorAction SilentlyContinue
  $peak = @{}
  foreach ($l in (Get-Content $smiLog)) { $f = $l -split ',\s*'; if ($f.Count -ge 3 -and $f[2] -match '^\d+$') {
      $v = [int]$f[2]; if (-not $peak.ContainsKey($f[1]) -or $v -gt $peak[$f[1]]) { $peak[$f[1]] = $v } } }
  $g0 = $peak['GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4']; $g1 = $peak['GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e']
  Add-Content "$dir\q-deep.out" ("item $($p.id) rc=$rc peak g0=$g0/12282 g1=$g1/16311 " + (Get-Date -Format 'HH:mm:ss'))
}
Add-Content "$dir\q-deep.out" 'queue done'
