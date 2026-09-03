# Watch LoRA training progress (run in a second terminal while training)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$log = Join-Path $PSScriptRoot "output\lora\progress.log"
$json = Join-Path $PSScriptRoot "output\lora\progress.json"

Write-Host "Watching $log"
Write-Host "Also: $json"
Write-Host "Ctrl+C to stop watching (training keeps running).`n"

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
if (-not (Test-Path $log)) {
    New-Item -ItemType File -Path $log | Out-Null
    Set-Content -Path $log -Value "[waiting] Start training with: docker compose run --rm lora"
}

Get-Content -Path $log -Wait -Tail 40
