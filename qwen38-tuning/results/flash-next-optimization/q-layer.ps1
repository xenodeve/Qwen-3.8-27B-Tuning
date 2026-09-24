# -sm layer (op offload works) vs -sm tensor (meta backend: offload_op = nullptr).
$ErrorActionPreference = 'Continue'
$dir = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$plan = @(
  @{ id='L-ts30_18-ub2048'; sm='layer';  ts='30,18';      ub=2048 },
  @{ id='L-ts30_18-ub4096'; sm='layer';  ts='30,18';      ub=4096 },
  @{ id='T-ctrl-ub512';     sm='tensor'; ts='8500,16000'; ub=512  }
)
foreach ($p in $plan) {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$dir\fn-config-run.ps1" `
      -RunId $p.id -Sm $p.sm -Ts $p.ts -Ub $p.ub -Probes 'decode,pre4k,pre16k' -Reps 1 -Note 'layer-vs-tensor' 2>&1 | Select-Object -Last 3
  Add-Content "$dir\q-layer.out" ("item " + $p.id + " rc=$LASTEXITCODE " + (Get-Date -Format 'HH:mm:ss'))
}
Add-Content "$dir\q-layer.out" 'queue done'
