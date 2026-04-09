$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

$ScriptDir   = $PSScriptRoot
$BackendDir  = Join-Path $ScriptDir "..\server"
$BackendDir  = Resolve-Path $BackendDir -ErrorAction Stop

function Log-Info  { param($m) Write-Host "[INFO] $m" -ForegroundColor Green }
function Log-Warn  { param($m) Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Log-Error { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }

$PyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PyCmd) { $PyCmd = Get-Command python -ErrorAction SilentlyContinue }
if (-not $PyCmd) { Log-Error "Python не найден. Установите python3."; exit 1 }
Log-Info "Python установлен ($($PyCmd.Name))."

Push-Location $BackendDir
try {
    $VenvDir = Join-Path $BackendDir ".venv"
    
    if (-not (Test-Path $VenvDir)) {
        Log-Info "Создание окружения..."
        & $PyCmd.Source -m venv .venv
    }

    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        & $ActivateScript
    } else {
        Log-Error "Скрипт активации venv не найден: $ActivateScript"
        exit 1
    }

    Log-Info "Установка библиотек..."
    pip install -r requirements.txt

    Log-Info "Миграция базы данных..."
    & python manage.py migrate

    Log-Info "Запуск сервера..."
    & python manage.py runserver
}
finally {
    Pop-Location
}