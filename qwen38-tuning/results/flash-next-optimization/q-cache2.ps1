# Controls lost in q-cache.ps1: '-Extra ""' is dropped by powershell -File and
# -Extra then swallowed -Sm. Pass -Extra only when it has a value.
$ErrorActionPreference = 'Continue'
$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$B = 'C:\AI\llama.cpp-upstream-fn\build-b\bin\llama-server.exe'
$C = 'C:\AI\llama.cpp-upstream-fn\build-c\bin\llama-server.exe'
$plan = @(
  @{ id='K-B-ctl1b'; exe=$B; extra='' },
  @{ id='K-C-0b';    exe=$C; extra='' },
  @{ id='K-B-ctl2b'; exe=$B; extra='' }
)
foreach ($p in $plan) {
  $a = @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$dir\fn-config-run.ps1",
         '-RunId',$p.id,'-Exe',$p.exe,'-Sm','layer','-Ts','30,18','-Ub','2048',
         '-Probes','declong,declong,decode,pre4k,pre16k','-Reps','1','-Note','expert-cache')
  if ($p.extra) { $a += @('-Extra', $p.extra) }
  & powershell.exe @a 2>&1 | Select-Object -Last 3
  Add-Content "$dir\q-cache2.out" ("item " + $p.id + " rc=$LASTEXITCODE " + (Get-Date -Format 'HH:mm:ss'))
}
Add-Content "$dir\q-cache2.out" 'queue done'
