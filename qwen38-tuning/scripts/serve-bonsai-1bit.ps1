param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--prism-ml--Bonsai-27B-gguf\snapshots\f10afb355f104535e3e3e98cf7ab7795c72bd292\Bonsai-27B-Q1_0.gguf" --alias bonsai-1bit -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto `
    --host 127.0.0.1 --port $Port
