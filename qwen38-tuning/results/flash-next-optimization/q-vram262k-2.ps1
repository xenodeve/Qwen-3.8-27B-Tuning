# Measured VRAM (nvidia-smi, 250 ms) during 262K boots -- not llama.cpp's
# reservation log. Per card: baseline used before boot, peak used during boot,
# total, headroom at peak. A failed boot's peak shows how full each card got.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
$exe = 'C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe'
$out = Join-Path $global:STUDY 'vram262k.csv'
if (-not (Test-Path $out)) { '"id","ncmoe","ts","ub","cache","result","g0_base","g0_peak","g0_total","g1_base","g1_peak","g1_total","fail_device","fail_mib"' | Set-Content $out -Encoding UTF8 }
$plan = @(
  @{ nm=24; ts='29,19'; ub=512;  cache=64; ctx=262144 },
  @{ nm=24; ts='28,20'; ub=512;  cache=64; ctx=262144 },
  @{ nm=24; ts='30,18'; ub=1024; cache=64; ctx=131072 },
  @{ nm=24; ts='29,19'; ub=1024; cache=64; ctx=131072 },
  @{ nm=26; ts='31,17'; ub=1024; cache=64; ctx=131072 },
  @{ nm=24; ts='30,18'; ub=2048; cache=64; ctx=65536 }
)
foreach ($p in $plan) {
  $id = "M$([int]($p.ctx/1024))k-n$($p.nm)-ts$($p.ts -replace ',','_')-ub$($p.ub)-c$($p.cache)"
  Clear-Server
  $base = @{}
  foreach ($l in (nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader,nounits)) {
    $f = $l -split ',\s*'; $base[$f[0]] = @([int]$f[1], [int]$f[2]) }
  $smiLog = Join-Path $global:STUDY "logs\$id-smi.csv"
  $smi = Start-Process nvidia-smi -ArgumentList '--query-gpu=uuid,memory.used','--format=csv,noheader,nounits','-lms','250' `
           -RedirectStandardOutput $smiLog -WindowStyle Hidden -PassThru
  $null = Start-FlashServer -Ctx $p.ctx -NcMoE $p.nm -Ts $p.ts -Ubatch $p.ub -Batch ([Math]::Max(2048,$p.ub)) -Sm layer -Exe $exe `
            -Extra @('--moe-expert-cache', "$($p.cache)") -LogName ($id + '.log')
  $ok = Wait-Healthy -Port 8099 -TimeoutSec 420
  if ($ok) { Start-Sleep -Seconds 3 }
  Stop-Process -Id $smi.Id -Force; Start-Sleep -Milliseconds 500
  $peak = @{}
  foreach ($l in (Get-Content $smiLog)) { $f = $l -split ',\s*'; if ($f.Count -ge 2 -and $f[1] -match '^\d+$') {
      $v = [int]$f[1]; if (-not $peak.ContainsKey($f[0]) -or $v -gt $peak[$f[0]]) { $peak[$f[0]] = $v } } }
  $err = $global:BOOTLOG -replace '\.log$','-err.log'
  $fd = ''; $fm = ''
  $m = Select-String -Path $err -Pattern 'allocating ([\d.]+) MiB on device (\d)' | Select-Object -First 1
  if ($m) { $fm = $m.Matches[0].Groups[1].Value; $fd = $m.Matches[0].Groups[2].Value }
  $res = if ($ok) { 'BOOT-OK' } else { 'FAIL' }
  $g0 = $global:GPU0; $g1 = $global:GPU1
  ('"{0}","{1}","{2}","{3}","{4}","{5}","{6}","{7}","{8}","{9}","{10}","{11}","{12}","{13}"' -f `
    $id,$p.nm,$p.ts,$p.ub,$p.cache,$res,$base[$g0][0],$peak[$g0],$base[$g0][1],$base[$g1][0],$peak[$g1],$base[$g1][1],$fd,$fm) | Add-Content $out -Encoding UTF8
  Write-Log ("vram262k $id $res g0 peak=$($peak[$g0])/$($base[$g0][1]) g1 peak=$($peak[$g1])/$($base[$g1][1]) fail=dev$fd ${fm}MiB")
  Clear-Server
}
Write-Log 'vram262k queue2 done'
