# IMGEN setup for Windows + NVIDIA CUDA.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not on PATH. Install Python 3.10+ from https://www.python.org/downloads/windows/"
}

python -m venv .venv
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
Write-Host "Installing PyTorch (CUDA 12.8 wheels). Edit this script if you need another CUDA version."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

Write-Host ""
Write-Host "Optional INT8 extra (Image21-INT8):  pip install -r requirements-int8.txt"
Write-Host "Optional INT4 extra (Image21-INT4):  pip install -r requirements-int4.txt"
Write-Host "Start the studio:                   python -m imgen"
Write-Host "UI only, no weights:                python -m imgen --demo"
