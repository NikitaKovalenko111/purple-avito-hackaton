$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
$ScriptDir = $PSScriptRoot
$ModelDir  = Join-Path $ScriptDir "..\model"
$ModelDir  = Resolve-Path $ModelDir -ErrorAction Stop

function Log-Info  { param($m) Write-Host "[INFO] $m" -ForegroundColor Green }
function Log-Warn  { param($m) Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Log-Error { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }

$PyCmd = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $PyCmd) { $PyCmd = Get-Command python -ErrorAction SilentlyContinue }
if (-not $PyCmd) { Log-Error "Python не найден."; exit 1 }

$CheckpointPath = Join-Path $ModelDir "checkpoints\model_checkpoint.pt"

Log-Info "Проверка модели..."
if (-not (Test-Path $CheckpointPath)) {
    Log-Info "Создание модели..."

    & $PyCmd.Source (Join-Path $ModelDir "run.py")

    Log-Info "Модель обучена"
} else {
    Log-Info "Модель уже существует"
}