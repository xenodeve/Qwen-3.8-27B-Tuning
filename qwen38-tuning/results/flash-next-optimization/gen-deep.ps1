Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. C:\AI\qwen38-tuning\results\flash-next-optimization\sv-lib.ps1
$null = Start-FlashServer -Ctx 16384 -NcMoE 24 -Ts '30,18' -Ubatch 2048 -Batch 2048 -Sm layer -Exe 'C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe' -LogName 'gen-deep.log'
if (-not (Wait-Healthy -Port 8099)) { Clear-Server; throw 'boot failed' }
foreach ($n in 60000,120000,240000) {
  & python "$global:STUDY\make-deep-prompt.py" 8099 $n ("$global:STUDY\prompts\pre" + [int]($n/1000) + 'k.json')
}
Clear-Server
