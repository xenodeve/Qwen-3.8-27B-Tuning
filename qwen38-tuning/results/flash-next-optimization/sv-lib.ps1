Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$global:STUDY = 'C:\AI\qwen38-tuning\results\flash-next-optimization'
$global:MIRROR_EXE = 'C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe'
$global:MODEL = 'C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
$global:GPU0 = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4'
$global:GPU1 = 'GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e'
$global:PROC = $null
$global:RUN_EXE = $global:MIRROR_EXE
$global:RUN_UB = 512
$global:BOOTLOG = $null

function Read-Vram {
  . C:\AI\qwen38-tuning\scripts\Get-GpuVram.ps1
  $g = Get-GpuVram -Uuid $global:GPU0
  $g1 = Get-GpuVram -Uuid $global:GPU1
  if ($null -eq $g) { throw ('Read-Vram: GPU0 unreadable') }
  if ($null -eq $g1) { throw ('Read-Vram: GPU1 unreadable') }
  return ,@([int]$g.Used, [int]$g1.Used)
}

function Read-GpuUtil {
  # One query per UUID: the old comma-list --id form returned an error string
  # that landed in the CSV as the utilisation value.
  $u0 = (nvidia-smi -i $global:GPU0 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  $u1 = (nvidia-smi -i $global:GPU1 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if ("$u0" -notmatch '^\s*\d+\s*$') { $u0 = 'NA' }
  if ("$u1" -notmatch '^\s*\d+\s*$') { $u1 = 'NA' }
  return ,@([string]$u0.Trim(), [string]$u1.Trim())
}

function Clear-Server {
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Seconds 3
}

function Start-FlashServer {
  param([int]$Ctx = 16384, [int]$NcMoE = 34, [string]$Ts, [int]$Threads = 14,
        [int]$ThreadsBatch = 20, [int]$Batch = 2048, [int]$Ubatch = 512,
        [int]$Port = 8099, [string]$LogName = 'boot.log', [string]$Exe = $global:MIRROR_EXE,
        [string[]]$Extra = @(), [string]$Sm = 'tensor')
  if (-not (Test-Path $Exe)) { throw "Start-FlashServer: binary not found: $Exe" }
  Clear-Server
  $global:RUN_EXE = $Exe
  $global:RUN_UB = $Ubatch
  $global:RUN_B = $Batch
  $global:RUN_SM = $Sm
  $global:RUN_CTX = $Ctx
  # The unsloth mirror spells the lazy-read flag --tensor-read-lazy; upstream
  # renamed it --lazy-mode. Ask the binary instead of assuming.
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { $help = (& $Exe --help 2>&1 | ForEach-Object { "$_" }) -join "`n" }
  finally { $ErrorActionPreference = $old }
  if ($help -match '--tensor-read-lazy') { $lazyFlag = '--tensor-read-lazy' }
  elseif ($help -match '--lazy-mode') { $lazyFlag = '--lazy-mode' }
  else { throw "Start-FlashServer: $Exe has neither --tensor-read-lazy nor --lazy-mode" }
  $logPath = Join-Path "$global:STUDY\logs" $LogName
  $args = @(
    '-m', $global:MODEL, '-c', "$Ctx", '-ctk', 'q4_0', '-ctv', 'q4_0',
    '-b', "$Batch", '-ub', "$Ubatch", '-ngl', 'all', '-ncmoe', "$NcMoE",
    '-sm', $Sm, '-ts', $Ts, '-fit', 'off', '-lm', 'mmap',
    $lazyFlag, 'on', '-fa', 'on', '-np', '1',
    '-t', "$Threads", '-tb', "$ThreadsBatch",
    '--host', '127.0.0.1', '--port', "$Port"
  ) + $Extra
  $env:CUDA_VISIBLE_DEVICES = "$global:GPU0,$global:GPU1"
  $proc = Start-Process -FilePath $Exe -ArgumentList $args -PassThru `
           -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError ($logPath -replace '\.log$','-err.log')
  $global:PROC = $proc
  $global:BOOTLOG = $logPath
  Write-Log ("boot: exe=$Exe extra=$($Extra -join ' ') port=$Port ncmoe=$NcMoE ts=$Ts t=$Threads tb=$ThreadsBatch b=$Batch ub=$Ubatch")
  return $proc
}

function Wait-Healthy {
  param([int]$Port = 8099, [int]$TimeoutSec = 300)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if ($global:PROC -and $global:PROC.HasExited) { return $false }
    try {
      $h = curl.exe -s --max-time 8 "http://127.0.0.1:$Port/health" 2>$null
      if ($h -match '"ok"') { return $true }
    } catch {}
  }
  return $false
}

function Send-Request {
  param([string]$PromptFile, [int]$Port = 8099)
  # cache_prompt=false: with the cache on, reps 2..n re-used the slot and the
  # server evaluated 4 tokens, which the old harness logged as "prefill".
  $body = Get-Content $PromptFile -Raw -Encoding UTF8 | ConvertFrom-Json
  $body | Add-Member -NotePropertyName cache_prompt -NotePropertyValue $false -Force
  $tmp = Join-Path $env:TEMP ('fn-req-' + [guid]::NewGuid().ToString('N') + '.json')
  [System.IO.File]::WriteAllText($tmp, ($body | ConvertTo-Json -Depth 20 -Compress), (New-Object System.Text.UTF8Encoding($false)))
  $t0 = Get-Date
  $resp = curl.exe -s --max-time 5400 "http://127.0.0.1:$Port/v1/chat/completions" `
            -H 'Content-Type: application/json' -d "@$tmp"
  Remove-Item $tmp -ErrorAction SilentlyContinue
  $wall = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
  $zero = @{ ok=$false; prompt_tokens=0; out_tokens=0; prompt_tps=0.0; decode_tps=0.0;
             prompt_ms=0.0; wall=$wall; content=''; raw=$resp }
  try {
    $j = $resp | ConvertFrom-Json
    if ($j.PSObject.Properties.Match('error').Count -gt 0) { return $zero + @{ why='server-error' } }
    $pn = [int]$j.timings.prompt_n
    $ptok = [int]$j.usage.prompt_tokens
    if ($ptok -gt 256 -and $pn -lt [int]($ptok * 0.9)) {
      # Refuse a cache hit rather than record it as prefill speed.
      return $zero + @{ why=("cache-hit prompt_n=$pn of $ptok") }
    }
    return @{
      prompt_n = $pn
      ok = $true
      prompt_tokens = [int]$j.usage.prompt_tokens
      out_tokens = [int]$j.usage.completion_tokens
      prompt_tps = [double]$j.timings.prompt_per_second
      decode_tps = [double]$j.timings.predicted_per_second
      prompt_ms = [double]$j.timings.prompt_ms
      wall = $wall
      content = [string]$j.choices[0].message.content
    }
  } catch {
    return $zero + @{ why=('parse-fail: ' + $Error[0].ToString().Replace([char]10, ' ').Replace([char]13, ' ').Substring(0, [Math]::Min(180, $Error[0].ToString().Length))) }
  }
}

function Get-ProcTelemetry {
  $os = Get-CimInstance Win32_OperatingSystem
  $ramUsedMb = [int](($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1024)
  $ramAvailMb = [int]($os.FreePhysicalMemory / 1024)
  $p = $null
  if ($global:PROC -and -not $global:PROC.HasExited) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($global:PROC.Id)"
  }
  $ws = if ($p) { [int]($p.WorkingSetSize / 1MB) } else { 0 }
  $priv = if ($p) { [int]($p.PrivatePageCount / 1MB) } else { 0 }
  $pg = if ($p) { [int]($p.PageFileUsage / 1024) } else { 0 }
  return @{ ramUsedMb=$ramUsedMb; ramAvailMb=$ramAvailMb; wsMb=$ws; privMb=$priv; pgMb=$pg }
}

function Write-Log {
  param([string]$Msg)
  $line = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss', [System.Globalization.CultureInfo]::InvariantCulture) + '  ' + $Msg
  # Retry: a reader that opens the file without write-sharing (Git-bash
  # `tail -f`, 2026-09-23) made Add-Content throw, the run died after boot and
  # left an orphan server on 8099. Retry for ~15 s, then fail loudly.
  $path = Join-Path "$global:STUDY\logs" 'study.log'
  for ($i = 0; $i -lt 30; $i++) {
    try { Add-Content -Path $path -Value $line -ErrorAction Stop; return }
    catch { Start-Sleep -Milliseconds 500 }
  }
  Clear-Server
  throw "Write-Log: study.log locked for 15 s; server stopped. Line was: $line"
}

function Get-ExeLabel {
  param([string]$Exe)
  # PS 5.1 + ErrorActionPreference=Stop turns a native stderr line into a
  # terminating error; llama-server prints --version on stderr.
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { $v = (& $Exe --version 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match '^version:' } | Select-Object -First 1) }
  finally { $ErrorActionPreference = $old }
  return ("$v" -replace '^version:\s*', '').Trim()
}

function Add-Result {
  # v2: results-v2.csv, every field quoted, real binary / ub / prompt_n.
  # results.csv (v1) is left as it was: its runtime_commit and ub columns were
  # constants and its tensor_split was unquoted, shifting every later column.
  param([hashtable]$Row)
  $csv = Join-Path $global:STUDY 'results-v2.csv'
  $cols = 'run_id','timestamp','exe','exe_version','ctx','b','ub','sm','ts','ncmoe','t','tb','extra','probe','phase','prompt_tokens','prompt_n','output_tokens','prompt_tps','decode_tps','request_wall_s','gpu0_used_mb','gpu1_used_mb','gpu0_util','gpu1_util','ram_used_mb','ram_avail_mb','proc_ws_mb','success','failure_reason','notes'
  if (-not (Test-Path $csv)) { Set-Content -Path $csv -Encoding UTF8 (($cols | ForEach-Object { '"' + $_ + '"' }) -join ',') }
  $Row['timestamp'] = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss', [System.Globalization.CultureInfo]::InvariantCulture)
  $line = ($cols | ForEach-Object { '"' + ([string]$Row[$_]).Replace('"', '""') + '"' }) -join ','
  Add-Content -Path $csv -Encoding UTF8 $line
}

function Write-Manifest {
  param([string]$RunId, [string]$Ts, [int]$NcMoE, [int]$Threads, [int]$ThreadsBatch)
  $sha1 = (Get-FileHash $global:RUN_EXE -Algorithm SHA256).Hash.ToLower()
  $path = Join-Path "$global:STUDY\run-manifests" ($RunId + '.json')
  @{ run_id=$RunId; exe=$global:RUN_EXE; exe_version=(Get-ExeLabel $global:RUN_EXE); exe_sha256=$sha1;
     model=$global:MODEL; ctx=$global:RUN_CTX; ncmoe=$NcMoE; ts=$Ts; t=$Threads; tb=$ThreadsBatch;
     b=$global:RUN_B; ub=$global:RUN_UB; sm=$global:RUN_SM; fit='off'; lm='mmap'; lazy='on'; kv='q4_0';
     cuda_visible=$env:CUDA_VISIBLE_DEVICES } | ConvertTo-Json | Set-Content -Path $path -Encoding UTF8
}
