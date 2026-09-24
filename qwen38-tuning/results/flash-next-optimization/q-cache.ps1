# Expert cache (PR #27861 on top of VNNI) under -sm layer -ts 30,18 -ub 2048.
# B = build-b (VNNI, no cache code) control at both ends.
$ErrorActionPreference = 'Continue'
$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$B = 'C:\AI\llama.cpp-upstream-fn\build-b\bin\llama-server.exe'
$C = 'C:\AI\llama.cpp-upstream-fn\build-c\bin\llama-server.exe'
$plan = @(
  @{ id='K-B-ctl1';  exe=$B; extra='' },
  @{ id='K-C-0';     exe=$C; extra='' },
  @{ id='K-C-32';    exe=$C; extra='--moe-expert-cache,32' },
  @{ id='K-C-64';    exe=$C; extra='--moe-expert-cache,64' },
  @{ id='K-C-96';    exe=$C; extra='--moe-expert-cache,96' },
  @{ id='K-C-64i4';  exe=$C; extra='--moe-expert-cache,64,--moe-expert-cache-inserts,4' },
  @{ id='K-B-ctl2';  exe=$B; extra='' }
)
foreach ($p in $plan) {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$dir\fn-config-run.ps1" `
      -RunId $p.id -Exe $p.exe -Extra $p.extra -Sm layer -Ts '30,18' -Ub 2048 `
      -Probes 'declong,declong,decode,pre4k,pre16k' -Reps 1 -Note 'expert-cache' 2>&1 | Select-Object -Last 3
  Add-Content "$dir\q-cache.out" ("item " + $p.id + " rc=$LASTEXITCODE " + (Get-Date -Format 'HH:mm:ss'))
}
Add-Content "$dir\q-cache.out" 'queue done'
