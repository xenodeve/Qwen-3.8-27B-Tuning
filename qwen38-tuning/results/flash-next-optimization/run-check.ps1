$files = @(
  'C:\Users\xenod\AppData\Local\Temp\opencode\pmin.ps1',
  'C:\AI\qwen38-tuning\results\flash-next-optimization\fn-config-run.ps1',
  'C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1'
)
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
