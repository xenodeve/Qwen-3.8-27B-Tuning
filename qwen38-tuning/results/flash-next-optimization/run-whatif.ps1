$files = @('C:\AI\qwen38-tuning\scripts\serve-flash-next.ps1')
foreach ($f in $files) {
  $e = $null
  [void][System.Management.Automation.Language.Parser]::ParseFile($f, [ref]$null, [ref]$e)
  if ($e.Count) {
    Write-Output ('FAIL: ' + $f)
    foreach ($x in $e) { Write-Output ('  line ' + $x.Extent.StartLineNumber + ' ' + $x.Message) }
  } else {
    Write-Output ('OK: ' + $f)
  }
}
pwsh -NoProfile -ExecutionPolicy Bypass -File 'C:\AI\qwen38-tuning\scripts\serve-flash-next.ps1' -WhatIf
