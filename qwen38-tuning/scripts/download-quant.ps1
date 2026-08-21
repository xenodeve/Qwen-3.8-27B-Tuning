<#
Pre-fetch a GGUF into llama.cpp's HuggingFace cache WITHOUT disturbing the
tuning server on :8080.

Why launch llama-server at all instead of curl'ing the file? Because llama.cpp
owns the cache layout (repo dir, snapshot dir, refs, etag validation). Writing
the file there by hand risks a silent re-download later, at the exact moment
Phase C needs it. Letting llama.cpp fetch it guarantees the later
`-hf ...:UD-Q3_K_XL` resolves straight to the cached file.

Footprint control, so the concurrent Q4 sweep on :8080 is not perturbed:
    -ngl 0      no VRAM
    --fit off   do not let auto-fit claim VRAM
    -c 512      trivial KV
    --port 8081 no port conflict
and the process is killed as soon as the download completes, before it can
serve anything.

Usage:  .\download-quant.ps1 -Quant UD-Q3_K_XL -ExpectedGiB 12.52
#>
param(
  [string]$Quant       = 'UD-Q3_K_XL',
  [string]$Repo        = 'unsloth/Qwen3.8-27B-GGUF',
  [double]$ExpectedGiB = 12.52,
  [int]$Port           = 8081,
  [string]$Root        = 'C:\AI\qwen38-tuning'
)

$ErrorActionPreference = 'Continue'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log   = Join-Path $Root "logs\download-$Quant-$stamp.log"
$cache = "$env:USERPROFILE\.cache\huggingface\hub\models--" + ($Repo -replace '/','--')

$p = Start-Process C:\AI\llama.cpp-cuda\llama-server.exe -PassThru -WindowStyle Hidden `
      -ArgumentList @('-hf',"${Repo}:$Quant",'-ngl','0','--fit','off','-c','512',
                      '--no-mmproj-auto','--host','127.0.0.1','--port',"$Port") `
      -RedirectStandardOutput $log -RedirectStandardError "$log.err"

"downloading $Quant (pid $($p.Id)), expecting ~$ExpectedGiB GiB"
$target = $ExpectedGiB * 1GB
$last = 0

while (-not $p.HasExited) {
  Start-Sleep -Seconds 30

  # The in-progress blob reports 0 bytes via directory enumeration while its
  # handle is open -- NTFS does not flush the size until close. Opening the file
  # with FileShare.ReadWrite is the only way to read the true length.
  $size = 0
  foreach ($f in (Get-ChildItem "$cache\blobs","$cache\snapshots" -Recurse -File -ErrorAction SilentlyContinue |
                  Where-Object { $_.Name -match "downloadInProgress|$Quant" })) {
    try {
      $fs = [IO.File]::Open($f.FullName,'Open','Read','ReadWrite')
      if ($fs.Length -gt $size) { $size = $fs.Length }
      $fs.Close()
    } catch { }
  }

  if ($size -ne $last) {
    "{0,7:N2} / {1:N2} GiB  ({2,5:N1}%)" -f ($size/1GB), $ExpectedGiB, (100*$size/$target)
    $last = $size
  }

  # Complete: stop before it loads the model or binds a port.
  if ($size -ge $target * 0.999) {
    Start-Sleep -Seconds 20   # let llama.cpp finalize/rename the file
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    "download complete -- server stopped"
    break
  }
}

Get-ChildItem "$cache\snapshots" -Recurse -File -Filter '*.gguf' |
  ForEach-Object { "{0,7:N2} GiB  {1}" -f ($_.Length/1GB), $_.Name }
