# Boot-only matrix at -c 262144: does the worst-case reservation fit?
# llama.cpp reserves the full-context compute graph at load, so a boot that
# reaches /health has allocated its KV and compute buffers (runtime growth,
# e.g. a lazily sized indexer buffer, is checked separately by deep fills).
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
$exe = 'C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe'
$out = Join-Path $global:STUDY 'boot262k.csv'
if (-not (Test-Path $out)) { '"id","ncmoe","ts","ub","cache","result","boot_s","gpu0_mb","gpu1_mb","ram_avail_mb","err_tail"' | Set-Content $out -Encoding UTF8 }
$plan = @()
foreach ($nm in 24,26,28) { foreach ($ts in '30,18','32,16') { foreach ($ub in 2048,1024,512) { $plan += @{ nm=$nm; ts=$ts; ub=$ub; cache=64 } } } }
foreach ($p in $plan) {
  $id = "B262-n$($p.nm)-ts$($p.ts -replace ',','_')-ub$($p.ub)-c$($p.cache)"
  $t0 = Get-Date
  $null = Start-FlashServer -Ctx 262144 -NcMoE $p.nm -Ts $p.ts -Ubatch $p.ub -Batch ([Math]::Max(2048,$p.ub)) -Sm layer -Exe $exe `
            -Extra @('--moe-expert-cache', "$($p.cache)") -LogName ($id + '.log')
  $ok = Wait-Healthy -Port 8099 -TimeoutSec 420
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  $g = @(0,0); try { $g = Read-Vram } catch {}
  $ram = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
  $err = $global:BOOTLOG -replace '\.log$','-err.log'
  $tail = ''
  if (-not $ok -and (Test-Path $err)) { $tail = ((Get-Content $err -Tail 4) -join ' | ').Replace('"',"'") }
  if ($ok -and (Select-String -Path $err -Pattern 'no usable GPU found' -Quiet)) { $ok = $false; $tail = 'cpu-only boot' }
  $res = if ($ok) { 'BOOT-OK' } else { 'FAIL' }
  ('"{0}","{1}","{2}","{3}","{4}","{5}","{6}","{7}","{8}","{9}","{10}"' -f $id,$p.nm,$p.ts,$p.ub,$p.cache,$res,$secs,$g[0],$g[1],$ram,$tail) | Add-Content $out -Encoding UTF8
  Write-Log ("boot262k $id $res gpu0=$($g[0]) gpu1=$($g[1]) ram_free=$ram")
  Clear-Server
}
Write-Log 'boot262k queue done'
