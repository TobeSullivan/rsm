# RSM TTS Prototype - Windows Install Script
# Run from the tts-prototype/ directory in PowerShell
# Assumes: Windows 11, Python 3.10+ already installed, internet connection

$ErrorActionPreference = "Stop"

Write-Host "=== RSM TTS Prototype Setup (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# --- 1. Check prerequisites ---
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

$pythonVersion = python --version 2>&1
if (-not ($pythonVersion -match "Python 3\.(1[0-9]|[2-9][0-9])")) {
    Write-Host "ERROR: Python 3.10+ required. Got: $pythonVersion" -ForegroundColor Red
    Write-Host "Install Python from https://www.python.org/downloads/ and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "  Python: $pythonVersion" -ForegroundColor Green

# Check for git
try {
    $gitVersion = git --version 2>&1
    Write-Host "  Git: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: git not found. Chatterbox install from source will fail." -ForegroundColor Yellow
}

# Check for NVIDIA GPU
try {
    $nvidiaCheck = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  NVIDIA GPU detected (good — CUDA path available)" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: nvidia-smi failed. CPU-only inference will be slow." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  WARNING: nvidia-smi not found. CPU-only inference will be slow." -ForegroundColor Yellow
}

# Check for Ollama
try {
    $ollamaVersion = ollama --version 2>&1
    Write-Host "  Ollama: $ollamaVersion" -ForegroundColor Green
} catch {
    Write-Host "  Ollama not installed." -ForegroundColor Yellow
    Write-Host "  Install Ollama from https://ollama.com/download/windows and re-run." -ForegroundColor Yellow
    Write-Host "  (Continuing setup, but pull_models.ps1 will fail until Ollama is installed.)" -ForegroundColor Yellow
}

Write-Host ""

# --- 2. Create virtual environment ---
Write-Host "[2/5] Creating Python virtual environment..." -ForegroundColor Yellow

if (Test-Path ".venv") {
    Write-Host "  .venv already exists. Skipping creation." -ForegroundColor Green
} else {
    python -m venv .venv
    Write-Host "  Created .venv" -ForegroundColor Green
}

# Activate (in this script's scope)
& .\.venv\Scripts\Activate.ps1
Write-Host "  Activated .venv" -ForegroundColor Green

Write-Host ""

# --- 3. Upgrade pip and install base deps ---
Write-Host "[3/5] Installing base Python dependencies..." -ForegroundColor Yellow

python -m pip install --upgrade pip wheel setuptools | Out-Null
Write-Host "  pip upgraded" -ForegroundColor Green

# Core deps for the harness — keep this list minimal
$baseDeps = @(
    "ollama",           # Ollama Python client
    "soundfile",        # WAV I/O
    "numpy",
    "click",            # CLI for harness scripts
    "pyyaml"            # config files
)
foreach ($dep in $baseDeps) {
    Write-Host "  Installing $dep..." -ForegroundColor Gray
    pip install $dep | Out-Null
}
Write-Host "  Base dependencies installed" -ForegroundColor Green

Write-Host ""

# --- 4. Install PyTorch (CUDA build if possible) ---
Write-Host "[4/5] Installing PyTorch..." -ForegroundColor Yellow
Write-Host "  This is a large download (~2-3GB). Be patient." -ForegroundColor Gray

# Try CUDA 12.1 build first (works on most modern NVIDIA cards including 4070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host "  CUDA torch install failed. Falling back to CPU build." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}
Write-Host "  PyTorch installed" -ForegroundColor Green

Write-Host ""

# --- 5. Install Chatterbox TTS ---
Write-Host "[5/5] Installing Chatterbox TTS..." -ForegroundColor Yellow

# Chatterbox is available on PyPI: `pip install chatterbox-tts`
# If this fails or the package name has changed by 2026, fall back to source install
pip install chatterbox-tts

if ($LASTEXITCODE -ne 0) {
    Write-Host "  PyPI install failed. Attempting source install from GitHub..." -ForegroundColor Yellow
    pip install git+https://github.com/resemble-ai/chatterbox.git
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Chatterbox installed" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Chatterbox install failed. Check https://github.com/resemble-ai/chatterbox for current install instructions." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Install Ollama if you haven't: https://ollama.com/download/windows" -ForegroundColor White
Write-Host "  2. Run: .\harness\pull_models.ps1" -ForegroundColor White
Write-Host "  3. Populate .\voices\ with reference audio (see voices\README.md)" -ForegroundColor White
Write-Host "  4. Run: python .\harness\generate_patter.py --phase A --model qwen2.5:7b" -ForegroundColor White
Write-Host ""
Write-Host "If anything failed, check the error messages above and the README.md." -ForegroundColor White
