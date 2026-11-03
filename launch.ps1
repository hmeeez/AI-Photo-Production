$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python environment not found. Run .\install.ps1 first."
}

& $python (Join-Path $root "src\main.py")
if ($LASTEXITCODE -ne 0) {
    throw "AI Photo Production exited with code $LASTEXITCODE."
}
