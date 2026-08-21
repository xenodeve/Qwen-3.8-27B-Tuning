param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--ornith-ai--Ornith-1.0-9B-GGUF\snapshots\3296bc7a404871a72ac3f1903f561459c09b5c17\ornith-1.0-9b-Q6_K.gguf" --alias ornith9b -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto `
    --host 127.0.0.1 --port $Port
