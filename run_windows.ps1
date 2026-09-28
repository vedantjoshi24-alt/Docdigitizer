# Starts DocDigitize with a private Python 3.12 runtime under LocalAppData.
# The application and OCR models run locally; no document is uploaded.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeRoot = Join-Path $env:LOCALAPPDATA "DocDigitize"
$venv = Join-Path $runtimeRoot "venv-v2"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
    Write-Host "Preparing the private Python 3.12 runtime (one-time setup)..." -ForegroundColor Cyan
    py -3 -m pip install --user "uv>=0.8,<1"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the local environment manager." }
    py -3 -m uv venv --python 3.12 --seed $venv
    if ($LASTEXITCODE -ne 0) {
        # Some Windows systems finish extracting managed Python before uv's
        # minor-version junction becomes visible. Retry with the exact path.
        $uvPythonRoot = Join-Path $env:APPDATA "uv\python"
        $managedPython = Get-ChildItem -LiteralPath $uvPythonRoot -Directory -Filter "cpython-3.12.*-windows-x86_64-none" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1 |
            ForEach-Object { Join-Path $_.FullName "python.exe" }
        if ($managedPython -and (Test-Path -LiteralPath $managedPython)) {
            py -3 -m uv venv --python $managedPython --seed $venv
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the private Python runtime." }
}

$requirements = Join-Path $project "backend\requirements.txt"
$stamp = Join-Path $venv "requirements.sha256"
$hash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
if (-not (Test-Path $stamp) -or (Get-Content -LiteralPath $stamp -Raw).Trim() -ne $hash) {
    Write-Host "Installing or updating local OCR components..." -ForegroundColor Cyan
    py -3 -m uv pip install --python $python -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See the pip error above." }
    Set-Content -LiteralPath $stamp -Value $hash
}

Set-Location (Join-Path $project "backend")
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
Write-Host "Starting DocDigitize at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Keep this window open while using the app. Press Ctrl+C to stop it." -ForegroundColor DarkGray
& $python -m uvicorn app:app --host 127.0.0.1 --port 8000
