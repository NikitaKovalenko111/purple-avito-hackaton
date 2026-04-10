$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

$ScriptDir  = $PSScriptRoot
$ClientDir  = Join-Path $ScriptDir "..\client"
$ClientDir  = Resolve-Path $ClientDir -ErrorAction Stop

function Log-Info  { param($m) Write-Host "[INFO] $m" -ForegroundColor Green }
function Log-Warn  { param($m) Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Log-Error { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }

$NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $NpmCmd) { Log-Error "npm isn't found. Install Node.js."; exit 1 }
Log-Info "npm is installed (version: $(npm -v))"

Log-Info "Starting client..."
Push-Location $ClientDir
try {
    npm install
    npm run dev
}
finally {
    Pop-Location
}