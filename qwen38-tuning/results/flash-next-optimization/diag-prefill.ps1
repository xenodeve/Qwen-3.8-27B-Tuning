# Where does prefill time go? Samples CPU / GPU / disk / PCIe while a cold
# pre4k request runs. Evidence for the next lever, not a speed measurement.
param([string]$Ts = '8500,16000', [int]$NcMoE = 24, [string]$RunId = 'diag-prefill', [int]$Ub = 512, [string[]]$Extra = @(), [string]$Sm = 'tensor', [int]$Seconds = 70)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
$Extra = @($Extra | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$null = Start-FlashServer -NcMoE $NcMoE -Ts $Ts -Ubatch $Ub -Batch ([Math]::Max(2048,$Ub)) -Extra $Extra -Sm $Sm -LogName ($RunId + '.log')
if (-not (Wait-Healthy -Port 8099)) { Clear-Server; throw 'boot failed' }
$job = Start-Job -ScriptBlock {
  . C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
  Send-Request -PromptFile 'C:\AI\qwen38-tuning\results\flash-next-optimization\prompts\pre4k.json' -Port 8099
}
Start-Sleep -Seconds 8
$out = Join-Path $global:STUDY ("logs\" + $RunId + '-samples.csv')
'"t","cpu_total_pct","srv_cpu_cores","disk_read_MBps","gpu0_util","gpu1_util","gpu0_rx_MBps","gpu1_rx_MBps","ram_avail_mb","srv_ws_mb","hard_faults_ps"' | Set-Content $out
$srv = Get-Process llama-server
$ncores = [Environment]::ProcessorCount
$prev = $srv.TotalProcessorTime; $pt = Get-Date
for ($i = 0; $i -lt [int]($Seconds/2); $i++) {
  $c = Get-Counter -Counter '\Processor(_Total)\% Processor Time','\PhysicalDisk(_Total)\Disk Read Bytes/sec','\Memory\Available MBytes','\Memory\Page Reads/sec' -SampleInterval 1 -MaxSamples 1
  $v = $c.CounterSamples | ForEach-Object { $_.CookedValue }
  $srv.Refresh(); $now = Get-Date
  $cores = ($srv.TotalProcessorTime - $prev).TotalSeconds / ($now - $pt).TotalSeconds
  $prev = $srv.TotalProcessorTime; $pt = $now
  $g = @()
  foreach ($u in @($global:GPU0, $global:GPU1)) {
    $q = nvidia-smi -i $u --query-gpu=utilization.gpu --format=csv,noheader,nounits
    $g += "$q".Trim()
  }
  $rx = @()
  foreach ($idx in 0,1) {
    $d = nvidia-smi dmon -i $idx -s t -c 1 2>$null | Where-Object { $_ -notmatch '^#' } | Select-Object -First 1
    $f = ("$d".Trim() -split '\s+')
    $rx += if ($f.Count -ge 3) { $f[1] } else { 'NA' }
  }
  $line = '"{0}","{1:N0}","{2:N1}","{3:N0}","{4}","{5}","{6}","{7}","{8:N0}","{9}","{10:N0}"' -f $i, $v[0], $cores, ($v[1]/1MB), $g[0], $g[1], $rx[0], $rx[1], $v[2], [int]($srv.WorkingSet64/1MB), $v[3]
  Add-Content $out $line
  if ($job.State -ne 'Running') { break }
}
$r = Receive-Job $job -Wait
Add-Content $out ('# request: ok=' + $r.ok + ' pn=' + $r['prompt_n'] + ' prefill=' + $r.prompt_tps + ' wall=' + $r.wall + ' ub=' + $Ub + ' extra=' + ($Extra -join ' ') + ' sm=' + $Sm + ' ts=' + $Ts + ' ncmoe=' + $NcMoE + ' cores=' + $ncores)
Clear-Server
