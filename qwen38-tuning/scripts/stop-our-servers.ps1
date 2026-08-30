<#
Stop only the servers THIS repository started.

WHY THIS FILE EXISTS. `Stop-Process -Name llama-server` matches by NAME, and on
this machine that is not only ours: Unsloth Studio runs its own copy from
C:\Users\xenod\.unsloth\llama.cpp\build\bin\Release\llama-server.exe. On
2026-08-29 an agent used the by-name form repeatedly while clearing VRAM and
killed the developer's Studio session each time -- Studio restarted it, which
read as "the process will not die" rather than as "you are killing the wrong
one".

So the filter is the PATH, not the name.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    # Everything this repository launches lives under here.
    [string]$OurRoot = 'C:\AI\'
)

$ours = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($OurRoot) }

$theirs = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
          Where-Object { -not ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($OurRoot)) }

foreach ($p in $theirs) {
    Write-Host ("  LEAVING ALONE  pid {0}  {1}" -f $p.ProcessId, $p.ExecutablePath) -ForegroundColor Cyan
}

if (-not $ours) {
    Write-Host "  nothing of ours is running." -ForegroundColor DarkGray
} else {
    foreach ($p in $ours) {
        if ($PSCmdlet.ShouldProcess("pid $($p.ProcessId)  $($p.ExecutablePath)", "Stop-Process")) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Host ("  stopped        pid {0}" -f $p.ProcessId) -ForegroundColor Green
        }
    }
}

# No VRAM report here on purpose. One chokepoint per language may ask the
# driver (test_no_module_queries_every_gpu) -- eleven call sites once read
# something other than what they claimed, and PowerShell's `-split` returned
# the WRONG CARD's numbers without erroring. This script stops processes; the
# caller reports memory.
Start-Sleep -Seconds 5
