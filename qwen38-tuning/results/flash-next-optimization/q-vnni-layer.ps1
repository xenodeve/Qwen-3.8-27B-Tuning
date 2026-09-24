# VNNI A/B under -sm layer (upstream cannot run qwen4exp with -sm tensor),
# plus mirror repeats. M A B B A M. A = build-b/bin with e6ab7c1a ggml-cpu dlls.
$ErrorActionPreference = 'Continue'
$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$exe = @{
  M = 'C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe'
  A = 'C:\AI\llama.cpp-upstream-fn\bin-A\llama-server.exe'
  B = 'C:\AI\llama.cpp-upstream-fn\build-b\bin\llama-server.exe'
}
$order = @('M','A','B','B','A','M')
$n = 0
foreach ($k in $order) {
  $n++
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$dir\fn-config-run.ps1" `
      -RunId ("V-$k-r$n") -Exe $exe[$k] -Sm layer -Ts '30,18' -Ub 2048 -Probes 'decode,decode,decode,pre4k,pre16k' -Reps 1 `
      -Note ("vnni-layer arm=$k") 2>&1 | Select-Object -Last 3
  # extra decode reps for a decode spread
  Add-Content "$dir\q-vnni-layer.out" ("item V-$k-r$n rc=$LASTEXITCODE " + (Get-Date -Format 'HH:mm:ss'))
}
Add-Content "$dir\q-vnni-layer.out" 'queue done'
