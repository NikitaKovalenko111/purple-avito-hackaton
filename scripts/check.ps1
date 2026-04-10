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
if (-not $PyCmd) { Log-Error "Python isn't found."; exit 1 }

$CheckpointPath = Join-Path $ModelDir "checkpoints\model_checkpoint.pt"

Log-Info "Checking the model..."
if (-not (Test-Path $CheckpointPath)) {
    Log-Info "Training the model..."

    & $PyCmd.Source (Join-Path $ModelDir "run.py") `
        --data-dir .\data `
        --dataset-file rnc_dataset_markup_balanced.csv `
        --epochs 10 `
        --batch-size 4 `
        --lr 3e-5 `
        --weight-decay 0.01 `
        --max-length 512 `
        --long-text-mode chunks `
        --long-text-window-tokens 256 `
        --long-text-stride-tokens 192 `
        --long-text-max-windows 8 `
        --split-pos-weight 1.5 `
        --optimize-for composite `
        --min-recall 0.0 `
        --patience 3 `
        --split-target-mode auto `
        --device cuda `
        --train-stratified-kfold 3 `
        --cv-source-split train_val `
        --cv-random-state 42 `
        --balance-should-split-batches `
        --batch-false-ratio 0.6 `
        --batch-true-ratio 0.4 `
        --output-dir .\checkpoints `
        --split-equals-detected-when-should-split

    Log-Info "The model is ready!"
} else {
    Log-Info "The model already exists"
}