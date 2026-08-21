param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--prism-ml--Ternary-Bonsai-27B-gguf\snapshots\abbae723028d71be674e71e1a71201a6f43fab22\Ternary-Bonsai-27B-Q2_g64.gguf" --alias bonsai-g64 -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto `
    --host 127.0.0.1 --port $Port
