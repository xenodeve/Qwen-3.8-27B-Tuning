<#
Phase A baseline: UD-Q4_K_XL, low context, CUDA0 only, single slot,
conservative (default f16) KV. One variable changes per experiment; this
script is the control.

Deliberately NOT tuned yet -- no -ctk/-ctv, no -nkvo, no explicit -ngl,
no -b/-ub. Those are Phases D/E/F/G and must be added one at a time.

Notes on flags verified against build 10472:
  --jinja       already the default; passed explicitly only for the record
  -ngl auto     already the default
  -fa           default is 'auto'; pinned to 'on' so the baseline is explicit

Usage:  .\start-q4.ps1                 # 16K baseline
        .\start-q4.ps1 -Ctx 32768      # Phase C step
#>
param(
  [int]$Ctx      = 16384,
  [string]$Model = 'unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL',
  [string]$Alias = 'qwen38-q4',
  [int]$Port     = 8080,
  [string]$Root  = 'C:\AI\qwen38-tuning'
)

$ErrorActionPreference = 'Stop'
$label = "q4-c$Ctx"
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log   = Join-Path $Root "logs\$label-$stamp.log"

& (Join-Path $Root 'scripts\collect-env.ps1') -Label "$label-preboot" -Root $Root

"`n=== launching (log: $log) ===`n"

# 2>&1 keeps llama.cpp's load report (layer split, buffer sizes) in the log --
# that report is the Phase A evidence, not just noise.
#
# llama-server writes its ENTIRE log to stderr, including normal progress. Under
# 'Stop' the first such line is raised as a terminating NativeCommandError and the
# server is killed before it ever binds the port. Native output is not an error
# here, so drop to 'Continue' for the launch.
$ErrorActionPreference = 'Continue'

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -hf $Model `
    --alias $Alias `
    -c $Ctx `
    -ngl auto `
    --fit on `
    -fa on `
    -np 1 `
    --jinja `
    --no-mmproj-auto `
    --host 127.0.0.1 `
    --port $Port 2>&1 | Tee-Object -FilePath $log
