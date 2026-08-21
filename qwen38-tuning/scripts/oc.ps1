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

UNRESOLVED, AND IT WILL BITE YOU: -Work DOES NOT DECIDE WHERE FILES LAND.
OpenCode anchors its bash tool and every path it writes to a project root it
resolves itself. On this machine that root is C:/AI almost regardless of what
this script does. Five runs on 2026-08-21, each asking the agent for its own
`pwd`, or writing two files:

  -Work C:/AI/ocworker/run131k                        -> C:/AI
  -Work C:/AI/ocworker/run131k, git init'd            -> C:/AI
  -Work C:/ocwork, outside the repo, no .git          -> C:/AI
  -Work C:/ocwork, outside the repo, git init'd       -> C:/ocwork   <- the only one
  -Work C:/ocworker/run, outside the repo, git init'd -> C:/AI

Rows 4 and 5 differ in no way this script controls, so 'outside the repo and a
git root' is NOT the rule -- it was the hypothesis and row 5 refuted it. Three
other explanations were tested and refuted too: a stale OpenCode server (killing
it does not help, and no process survives between runs), OPENCODE_CONFIG_DIR
pointing into the repository, and a nested git root.

What is left, unverified: OpenCode persists project state in
~/.local/share/opencode/opencode.db, alongside a snapshot/ tree keyed by hash and
a repos/ directory, and C:/AI appears in it. A persisted root would explain every
row above. Nothing here proves it.

The cost of the failure is not an error. A run with -Work C:/AI/ocworker/run131k
wrote slugify.py and test_slugify.py into C:/AI, the repository root, ran the
tests there, and reported success -- correctly, from its own point of view.

UNTIL THIS IS UNDERSTOOD: put the absolute destination path in the prompt itself
and check afterwards where the files actually went. Do not trust -Work.

#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Prompt,
    [string]$Work = "C:\ocworker\run",
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
if ($Work -like "C:\AI*") {
    throw "working directory is inside C:\AI -- OpenCode resolves the project root to the enclosing repository and will write there instead. Use a path outside it."
}
# A stale server would silently keep its original project root -- see the header.
Get-Process opencode -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

New-Item -ItemType Directory -Force -Path $Work | Out-Null

# The project root OpenCode will actually use -- see the header.
if (-not (Test-Path (Join-Path $Work '.git'))) {
    git -C $Work init -q
}

$env:OPENCODE_DISABLE_CLAUDE_CODE     = "1"
$env:OPENCODE_DISABLE_EXTERNAL_SKILLS = "1"
$env:OPENCODE_DISABLE_DEFAULT_PLUGINS = "1"
$env:OPENCODE_CONFIG_DIR              = $cfgDir

# The header explains this at length; a parameter that does not do what its name
# says has to say so where the operator is looking, not only where they might read.
Write-Warning "-Work sets the launch directory. It does NOT decide where OpenCode writes -- five measured runs are tabulated in this script's header. Put the absolute destination in the prompt and check afterwards."
Write-Host "model $Model  ctx $nctx  cwd $Work" -ForegroundColor DarkGray
Push-Location $Work
try { & $oc run -m $Model $Prompt } finally { Pop-Location }
