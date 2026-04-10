#Requires -Version 5.1

# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

$ScriptDir  = $PSScriptRoot
$ScriptsDir = Join-Path $ScriptDir "scripts"

function Log-Info  { param($m) Write-Host "[INFO] $m" -ForegroundColor Green }
function Log-Warn  { param($m) Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Log-Error { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }

Log-Info "Starting the project..."

$Processes = @()

function Stop-All {
    Log-Warn "Ending..."
    foreach ($p in $Processes) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Log-Info "All the processes are stopped."
}

try {
    $ModelScript = Join-Path $ScriptsDir "model.ps1"
    if (Test-Path $ModelScript) {
        Log-Info "Checking the model..."
        & $ModelScript
    }

    Log-Info "Starting backend..."
    $BackendScript = Join-Path $ScriptsDir "backend.ps1"
    $Processes += Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $BackendScript `
        -NoNewWindow -PassThru
    Log-Info "Backend is started (PID: $($Processes[-1].Id))"

    Log-Info "Starting frontend...."
    $FrontendScript = Join-Path $ScriptsDir "frontend.ps1"
    $Processes += Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $FrontendScript `
        -NoNewWindow -PassThru
    Log-Info "Frontend is started (PID: $($Processes[-1].Id))"

    Write-Host "`nThe project has started. Press CRTL+C for correct stop." -ForegroundColor Cyan

    while ($true) {
        $anyAlive = $false
        foreach ($p in $Processes) {
            if (-not $p.HasExited) { $anyAlive = $true; break }
        }
        if (-not $anyAlive) { break }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Stop-All
}