<#
Parameterised serve script for V3 UD-IQ2_XXS -- the largest Qwen3.8-27B artifact
that holds 65+0 at 131,072 (report 19).

Exists so the sampling and chat-protocol sweeps can vary ONE thing at a time
without a script per configuration. Everything except -Extra is byte-identical
to serve-v3-iq2xxs.ps1, so each run is a controlled comparison against it.

  .\serve-v3-iq2xxs-flex.ps1 -Extra '--dry-multiplier 0.8 --dry-penalty-last-n 4096'
#>
param([int]$Ctx = 16384, [int]$Port = 8080, [string]$Extra = '--dry-multiplier 0.8 --dry-penalty-last-n 4096 --reasoning-budget 2048')
$ErrorActionPreference = 'Continue'
$args = @(
  '-m', "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf",
  '--alias', 'qwen38-v3-iq2xxs-flex', '-c', $Ctx,
  '-ngl', 'auto', '--fit', 'on', '--fit-target', '768', '-fa', 'on', '-np', '1',
  '-t', '18', '-b', '2048', '-ub', '256', '--no-mmproj-auto', '-lv', '5',
  '--host', '127.0.0.1', '--port', $Port
)
if ($Extra) { $args += ($Extra -split '\s+') }
& C:\AI\llama.cpp-cuda\llama-server.exe @args
