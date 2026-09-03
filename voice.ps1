Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

Write-Host "Building / starting Nova voice server..."
docker compose up -d --build voice
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:7860/"
Write-Host "Opened http://127.0.0.1:7860/ — wait ~30-90s for the model to load, then talk."
Write-Host "Logs: docker compose logs -f voice"
