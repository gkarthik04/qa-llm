# End-to-end run on Windows (PowerShell): venv -> install -> train -> index -> evaluate -> tests.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Optional: $env:VENV_DIR = "C:\venvs\qa-llm" to place the venv elsewhere (useful if the repo path is very long).
$ErrorActionPreference = "Stop"
$venv = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

function Step([string]$label, [scriptblock]$cmd) {
    Write-Host "`n=== $label" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $label (exit $LASTEXITCODE)" }
}

if (-not (Test-Path "$venv\Scripts\python.exe")) { Step "Create venv" { py -3.11 -m venv $venv } }
$py = "$venv\Scripts\python.exe"

Step "Upgrade pip"        { & $py -m pip install --upgrade pip }
Step "Install torch (CPU)" { & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu }
Step "Install requirements" { & $py -m pip install -r requirements.txt }
Step "Fine-tune DistilBERT" { & $py -m qa.train }
Step "Build FAISS index"   { & $py -m qa.build_index }
Step "Evaluate"            { & $py -m qa.evaluate }
Step "Tests"               { & $py -m pytest -q }

Write-Host "`nDone. Try:  $py -m qa.cli ask `"Who was the Super Bowl 50 MVP?`""
Write-Host "API:        $py -m uvicorn qa.api:app --port 8000"
