# RSM TTS Prototype - Ollama Model Pulls
# Pulls the LM models for Phase A/B/C/Stretch testing.
# Run after install.ps1 and after Ollama is installed.

$ErrorActionPreference = "Stop"

Write-Host "=== Pulling Ollama models for RSM TTS prototype ===" -ForegroundColor Cyan
Write-Host ""

# Verify Ollama is available
try {
    ollama --version | Out-Null
} catch {
    Write-Host "ERROR: Ollama not found. Install from https://ollama.com/download/windows first." -ForegroundColor Red
    exit 1
}

# Models to pull, ordered by test priority
# Edit this list to skip / add models. Tags use latest Ollama defaults as of writing.

$models = @(
    # Phase A/B target tier — 7-8B class
    @{ Name = "qwen2.5:7b"; Phase = "A/B target"; Approx = "~4.7GB" },
    @{ Name = "llama3.1:8b"; Phase = "A/B alternate"; Approx = "~4.9GB" },

    # Stretch — 4B class for console floor probe
    @{ Name = "llama3.2:3b"; Phase = "console floor"; Approx = "~2.0GB" },
    @{ Name = "qwen2.5:3b"; Phase = "console floor alternate"; Approx = "~1.9GB" },

    # Phase C fallback — 14B class
    @{ Name = "qwen2.5:14b"; Phase = "C fallback"; Approx = "~9.0GB" }
)

Write-Host "Models queued for download:" -ForegroundColor Yellow
foreach ($m in $models) {
    Write-Host "  $($m.Name) - $($m.Phase) ($($m.Approx))" -ForegroundColor Gray
}
$totalApprox = "~22GB"
Write-Host ""
Write-Host "Total download: ${totalApprox}. Make sure you have disk space." -ForegroundColor Yellow
Write-Host ""

$proceed = Read-Host "Proceed? (y/n)"
if ($proceed -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Pull each
foreach ($m in $models) {
    Write-Host ""
    Write-Host "Pulling $($m.Name)..." -ForegroundColor Cyan
    ollama pull $m.Name
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARNING: Pull failed for $($m.Name). Continuing." -ForegroundColor Yellow
    } else {
        Write-Host "  Pulled $($m.Name)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== All pulls attempted ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Verify with: ollama list" -ForegroundColor White
