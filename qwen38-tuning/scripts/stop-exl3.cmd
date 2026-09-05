@echo off
rem Stop the EXL3 server ON PURPOSE (issue #75). serve-exl3.cmd relaunches the server
rem after any exit it was not told about, so the stop flag goes down FIRST and the loop
rem reads it and ends. The tree is found by its command line through CIM: `wmic` is not
rem on cmd's PATH on this Windows 11 (deprecated), and Stop-Process by name matched the
rem wrong python and returned 255 on the server's own pids on 2026-09-05. The TP children
rem must go with the parent (taskkill /T). The Name filter keeps the query's own powershell
rem (whose command line also contains the path) out of the list.
setlocal
set STOP=C:\AI\qwen38-tuning\logs\exl3-stop.flag
echo stopped by stop-exl3.cmd> "%STOP%"
set FOUND=0
for /f "usebackq" %%p in (`powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*serving\exl3\server.py*' } | Select-Object -ExpandProperty ProcessId"`) do (
    set FOUND=1
    echo == EXL3 stop: killing pid %%p and its children
    taskkill /F /T /PID %%p >nul 2>&1
)
if "%FOUND%"=="0" (
    echo == EXL3 stop: no server process found; flag left so a loop mid-relaunch ends too.
)
