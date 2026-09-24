param([string[]]$Files)
foreach ($f in $Files) {
  $e = $null
  [void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $f), [ref]$null, [ref]$e)
  if ($e.Count) {
    Write-Output ('FAIL: ' + $f)
    foreach ($x in $e) { Write-Output ('  line ' + $x.Extent.StartLineNumber + ' ' + $x.Message) }
  } else {
    Write-Output ('OK: ' + $f)
  }
}
