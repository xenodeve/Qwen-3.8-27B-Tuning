Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$counters = Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue
foreach ($c in $counters.CounterSamples) {
  if ($c.InstanceName -match 'vhdscr') { Write-Output ('vhdscr ' + [math]::Round($c.CookedValue / 1MB, 0)) }
}
Write-Output '--- process instances for llama ---'
foreach ($c in $counters.CounterSamples) {
  if ($c.InstanceName -match 'llama') { Write-Output ($c.InstanceName + ' ' + [math]::Round($c.CookedValue / 1MB, 0)) }
}
Write-Output '--- raw GPU utilization ---'
$u = Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue
$used = $u.CounterSamples | Where-Object { $_.CookedValue -gt 0.5 }
foreach ($x in $used | Select-Object -First 6) { Write-Output ($x.InstanceName + ' ' + [math]::Round($x.CookedValue, 1)) }
