$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$models = Join-Path $root "MODELS"
$destination = Join-Path $models "real_esrgan_x4.onnx"
$temporary = Join-Path $models "real_esrgan_x4.onnx.download"
$expectedHash = "5C586662929CBC686C1A5C38D9C060DBDB4EA5863A1F7672B8C0761E6B89C033"
$uri = "https://huggingface.co/SceneWorks/real-esrgan-onnx/resolve/main/real_esrgan_x4.onnx?download=true"

New-Item -ItemType Directory -Force -Path $models | Out-Null
if (Test-Path -LiteralPath $destination) {
    $currentHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    if ($currentHash -eq $expectedHash) {
        Write-Host "Verified Real-ESRGAN x4 model already exists: $destination"
        exit 0
    }
    throw "Existing model has unexpected SHA-256 ($currentHash); it was not overwritten."
}

try {
    Invoke-WebRequest -Uri $uri -OutFile $temporary -TimeoutSec 300
    $downloadHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
    if ($downloadHash -ne $expectedHash) {
        throw "Downloaded model SHA-256 mismatch. Expected $expectedHash; got $downloadHash."
    }
    Move-Item -LiteralPath $temporary -Destination $destination
    Write-Host "Downloaded and verified Real-ESRGAN x4 model: $destination"
}
finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary
    }
}
