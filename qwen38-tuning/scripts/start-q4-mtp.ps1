<#
Q4 + built-in MTP speculative decoding smoke test.

Identical to start-q4.ps1 except for the speculative flags -- that is the point.
One variable changes.

Flags verified against the LOCAL b10472 --help (not master docs), per the
continuation plan's instruction:
    --spec-type      accepts draft-mtp (also a comma-separated list)
    --spec-draft-n-max N   default 3, start at 2 per the smoke-test spec

Draft KV is deliberately left at its default: the continuation plan says do not
copy the main KV quant onto the draft path until MTP is proven to activate.

Usage:  .\start-q4-mtp.ps1                 # n_max 2 smoke test
        .\start-q4-mtp.ps1 -NMax 4         # sweep step
        .\start-q4-mtp.ps1 -SpecType ngram-mod -NMax 4
#>
param(
  [int]$NMax        = 2,
  [string]$SpecType = 'draft-mtp',
  [int]$Ctx         = 16384,
  [string]$Model    = 'unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL',
  [string]$Alias    = 'qwen38-q4',
  [int]$Port        = 8080,
  [string]$Root     = 'C:\AI\qwen38-tuning'
)

$ErrorActionPreference = 'Stop'
$label = "q4-c$Ctx-$SpecType-n$NMax"
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log   = Join-Path $Root "logs\$label-$stamp.log"

& (Join-Path $Root 'scripts\collect-env.ps1') -Label "$label-preboot" -Root $Root

"`n=== launching $label (log: $log) ===`n"

# llama-server logs everything to stderr; 'Stop' would kill it on the first line.
$ErrorActionPreference = 'Continue'

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -hf $Model `
    --alias $Alias `
    -c $Ctx `
    -ngl auto `
    --fit on `
    -fa on `
    -np 1 `
    --no-mmproj-auto `
    --spec-type $SpecType `
    --spec-draft-n-max $NMax `
    --host 127.0.0.1 `
    --port $Port 2>&1 | Tee-Object -FilePath $log
