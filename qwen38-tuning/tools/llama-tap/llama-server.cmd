@echo off
rem llama-tap: point LLAMA_SERVER_PATH at THIS file. See shim.py for why the
rem argument tail is forwarded as raw text and re-read from the command line
rem rather than rebuilt from sys.argv.
python "%~dp0shim.py" --llama-tap-args %*
