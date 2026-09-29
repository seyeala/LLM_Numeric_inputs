param(
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -3.12 -m venv .venv
}

& $Python -m ensurepip --upgrade
& $Python -m pip install "pip==26.2.1"
& $Python -m pip install --no-compile -e ".[dev]"

if ($Gpu) {
    $GpuReady = & $Python -c "import torch; print('yes' if torch.__version__ == '2.14.0+cu126' and torch.cuda.is_available() else 'no')"
    if ($GpuReady.Trim() -ne "yes") {
        & $Python -m pip install --force-reinstall --no-deps --no-compile -r requirements-gpu-cu126.txt
    } else {
        Write-Output "CUDA PyTorch 2.14.0+cu126 is already installed; skipping GPU reinstall."
    }
}
& $Python -m pip check
& $Python scripts\environment_report.py
