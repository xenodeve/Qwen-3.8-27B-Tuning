# Where does the host RAM go while Flash-Next serves? 128k profile.
# Process: working set split into private (llama.cpp buffers, prompt cache,
# pinned host) and shared (file-backed mmap pages of the GGUF). System:
# available, standby (file cache, reclaimable), modified, commit.
param([switch]$Trim, [string]$Tag = 'notrim')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
$out = Join-Path $global:STUDY "logs\diag-mem-128k-$Tag.csv"
Add-Type -Namespace W -Name Psapi -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(System.IntPtr hProcess);'
'"phase","avail_mb","standby_mb","modified_mb","commit_mb","ws_mb","ws_private_mb","ws_shared_mb","private_bytes_mb","g0_mb","g1_mb"' | Set-Content $out -Encoding UTF8
function Snap([string]$phase) {
  $c = (Get-Counter '\Memory\Available MBytes','\Memory\Standby Cache Normal Priority Bytes','\Memory\Standby Cache Reserve Bytes','\Memory\Standby Cache Core Bytes','\Memory\Modified Page List Bytes','\Memory\Committed Bytes').CounterSamples
  $v = @{}; foreach ($s in $c) { $v[$s.Path.Split('\')[-1]] = $s.CookedValue }
  $standby = ($v['standby cache normal priority bytes'] + $v['standby cache reserve bytes'] + $v['standby cache core bytes']) / 1MB
  $ws = 0; $wsp = 0; $pb = 0
  $p = Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($p) {
    $ws = $p.WorkingSet64 / 1MB; $pb = $p.PrivateMemorySize64 / 1MB
    $wsp = (Get-Counter "\Process(llama-server)\Working Set - Private").CounterSamples[0].CookedValue / 1MB
  }
  $g = @(0,0); try { $g = Read-Vram } catch {}
  ('"{0}","{1:N0}","{2:N0}","{3:N0}","{4:N0}","{5:N0}","{6:N0}","{7:N0}","{8:N0}","{9}","{10}"' -f $phase, $v['available mbytes'], $standby,
     ($v['modified page list bytes']/1MB), ($v['committed bytes']/1MB), $ws, $wsp, ($ws-$wsp), $pb, $g[0], $g[1]).Replace(',',';').Replace('";"','","') | Add-Content $out -Encoding UTF8
  Write-Log ("diag-mem[$Tag] $phase avail=$([int]$v['available mbytes']) standby=$([int]$standby) ws=$([int]$ws) ws_private=$([int]$wsp) ws_shared=$([int]($ws-$wsp)) private_bytes=$([int]$pb)")
}
Snap 'before-boot'
$null = Start-FlashServer -Ctx 131072 -NcMoE 26 -Ts '31,17' -Ubatch 1024 -Batch 2048 -Sm layer `
          -Exe 'C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe' -Extra @('--moe-expert-cache','64') -LogName "diag-mem-128k-$Tag.log"
if (-not (Wait-Healthy -Port 8099)) { Clear-Server; throw 'boot failed' }
Snap 'after-boot'
if ($Trim) {
  # Clean file-backed pages go back to the standby list with no I/O; hot CPU
  # experts fault back in from the file cache as they are touched.
  $p = Get-Process llama-server | Select-Object -First 1
  $ok = [W.Psapi]::EmptyWorkingSet($p.Handle)
  Start-Sleep -Seconds 2
  Write-Log ("diag-mem[$Tag] EmptyWorkingSet returned $ok")
  Snap 'after-trim'
}
$r = Send-Request -PromptFile "$global:STUDY\prompts\pre60k.json" -Port 8099
Write-Log ("diag-mem[$Tag] pre60k ok=$($r.ok) prefill=$([math]::Round($r.prompt_tps,1)) decode=$([math]::Round($r.decode_tps,2))")
Snap 'after-60k-fill'
$r = Send-Request -PromptFile "$global:STUDY\prompts\declong.json" -Port 8099
Write-Log ("diag-mem[$Tag] declong ok=$($r.ok) decode=$([math]::Round($r.decode_tps,2))")
Snap 'after-512-decode'
Clear-Server
Start-Sleep -Seconds 5
Snap 'after-stop'
Write-Log "diag-mem[$Tag] done"
