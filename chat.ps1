Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (docker images -q local-ai-lora:latest)) {
    Write-Error "Image local-ai-lora:latest is missing. Run: docker compose build lora"
}

$hf = docker volume ls -q | Where-Object { $_ -eq "local-ai_hf-cache" }
if (-not $hf) {
    docker volume create local-ai_hf-cache | Out-Null
}

docker run -it --rm `
    --memory=5g --memory-swap=6g --cpus=2 --shm-size=256m `
    -v "${PSScriptRoot}:/workspace" `
    -v "local-ai_hf-cache:/workspace/.cache/huggingface" `
    --env-file "$PSScriptRoot\.env" `
    -e ADAPTER_DIR=/workspace/output/lora `
    --entrypoint python `
    local-ai-lora:latest `
    infer.py @args
