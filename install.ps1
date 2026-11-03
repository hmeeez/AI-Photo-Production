param([switch]$EnableOpenVINO)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python 3.11 (64-bit) is required. Install it from python.org and retry."
    }
    & py -3.11 -c "import struct,sys; assert sys.maxsize > 2**32; print(sys.version)"
    if ($LASTEXITCODE -ne 0) {
        throw "A 64-bit Python 3.11 installation was not found."
    }
    & py -3.11 -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3.11 virtual environment."
    }
}
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip in the project virtual environment."
}
& $python -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the application dependencies."
}
if ($EnableOpenVINO) {
    & $python -m pip install -r (Join-Path $root "requirements-openvino.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the optional OpenVINO dependencies."
    }
}

$directories = @(
    "INPUT", "OUTPUT\01_ENHANCED", "OUTPUT\02_UPSCALED",
    "OUTPUT\03_PRINT_READY", "OUTPUT\04_ILLUSTRATOR", "OUTPUT\05_PDF",
    "OUTPUT\06_REPORTS", "OUTPUT\07_PREVIEWS", "MODELS", "ICC_PROFILES",
    "TEMP", "LOGS", "CONFIG", "SCRIPTS", "TESTS"
)
foreach ($directory in $directories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $directory) | Out-Null
}

Write-Host "Installation complete. Start the application with .\launch.ps1"
