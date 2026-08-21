<#
Run OpenCode against the local Qwen3.8-27B worker.

    .\scripts\oc.ps1 "fix the failing test in src/cache.py"
    .\scripts\oc.ps1 -Work D:\Github\openclink "add a --dry-run flag to the CLI"

Start a server first:

    .\scripts\worker-iq2xxs-deep.ps1        # 131,072 ctx
    .\scripts\worker-iq2s-quality.ps1       #  98,304 ctx

WHY THIS SCRIPT EXISTS RATHER THAN A DOCUMENTED COMMAND LINE. Four environment
variables and a config directory have to be right together, none of them is
documented by OpenCode, and getting any one wrong fails quietly rather than
loudly:

  * miss the env vars and the prefix is 99,073 tokens instead of ~5,400 -- which
    on a 131,072 window plus OpenCode's 32,000-token output reservation is one
    token over, so the call dies before reading the task;
  * miss OPENCODE_CONFIG_DIR and it walks up from the working directory
    collecting every `.opencode` it finds, adding back ~10,000 tokens of skill
    catalogue the worker cannot use;
  * run from under your home directory and it finds `~\.opencode` the same way;
  * let `limit.context` disagree with the server's `-c` and OpenCode compacts at
    the wrong point.

The context limit is read from the running server rather than assumed, because
the two profiles differ and a stale number is invisible until a long task
truncates.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Prompt,
    [string]$Work = "C:\AI\ocworker\run",
    [string]$Model = "local/qwen38",
    [string]$Endpoint = "http://127.0.0.1:8080"
)
$ErrorActionPreference = 'Stop'

$cfgDir = "C:\AI\qwen38-tuning\opencode"
$cfg = Join-Path $cfgDir "opencode.json"
$oc = "C:\Users\xenod\.bun\bin\opencode.exe"

foreach ($p in @($cfg, $oc)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}

# The server is the authority on the window size. Reading it also proves the
# server is up before OpenCode spends a minute booting to discover it is not.
try {
    $props = Invoke-RestMethod -Uri "$Endpoint/props" -TimeoutSec 5
    $nctx = [int]$props.default_generation_settings.n_ctx
} catch {
    throw "no server on $Endpoint. Start worker-iq2xxs-deep.ps1 or worker-iq2s-quality.ps1 first."
}

$json = Get-Content $cfg -Raw | ConvertFrom-Json
$limit = $json.provider.local.models.qwen38.limit
if ($limit.context -ne $nctx) {
    Write-Host "context limit $($limit.context) -> $nctx (from the running server)"
    $limit.context = $nctx
    $json | ConvertTo-Json -Depth 12 | Set-Content $cfg -Encoding utf8
}

# Outside the home directory: OpenCode collects every `.opencode` on the way up
# from here, and it keeps a server alive between runs carrying the project root
# it first started with.
if ($Work -like "$env:USERPROFILE*") {
    throw "working directory is under $env:USERPROFILE -- OpenCode will pick up ~\.opencode from there. Use a path outside it."
}
New-Item -ItemType Directory -Force -Path $Work | Out-Null

$env:OPENCODE_DISABLE_CLAUDE_CODE     = "1"
$env:OPENCODE_DISABLE_EXTERNAL_SKILLS = "1"
$env:OPENCODE_DISABLE_DEFAULT_PLUGINS = "1"
$env:OPENCODE_CONFIG_DIR              = $cfgDir

Write-Host "model $Model  ctx $nctx  cwd $Work" -ForegroundColor DarkGray
Push-Location $Work
try { & $oc run -m $Model $Prompt } finally { Pop-Location }
