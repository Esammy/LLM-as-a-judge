#!/usr/bin/env pwsh
# Windows shim for the Makefile. Same target names, same commands.
#   .\make.ps1 check
param([Parameter(Position = 0)][string]$Target = "help")

$ErrorActionPreference = "Stop"
$uv = "uv"

function Invoke-Step($name, $cmd) {
    Write-Host "==> $name" -ForegroundColor Cyan
    & $uv @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name" -ForegroundColor Red; exit $LASTEXITCODE }
}

switch ($Target) {
    "install"   { & $uv sync --all-extras }
    "lint"      { Invoke-Step "ruff check" @("run", "ruff", "check", "src", "tests") }
    "fmt" {
        Invoke-Step "ruff format" @("run", "ruff", "format", "src", "tests")
        Invoke-Step "ruff --fix"  @("run", "ruff", "check", "--fix", "src", "tests")
    }
    "typecheck" { Invoke-Step "mypy" @("run", "mypy") }
    "test"      { Invoke-Step "pytest" @("run", "pytest") }
    "cov"       { Invoke-Step "pytest --cov" @("run", "pytest", "--cov", "--cov-report=term-missing", "--cov-report=xml") }
    "check" {
        Invoke-Step "ruff check" @("run", "ruff", "check", "src", "tests")
        Invoke-Step "mypy"       @("run", "mypy")
        Invoke-Step "pytest"     @("run", "pytest", "--cov", "--cov-report=term-missing")
        Write-Host "All checks passed." -ForegroundColor Green
    }
    "clean" {
        Get-ChildItem -Recurse -Directory -Force -Include __pycache__, .pytest_cache, .mypy_cache, .ruff_cache |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Force -ErrorAction SilentlyContinue .coverage, coverage.xml
    }
    "compose-up"   { docker compose -f deploy/docker-compose.yml up -d }
    "compose-down" { docker compose -f deploy/docker-compose.yml down -v }
    "k8s-up"       { kubectl apply -k deploy/k8s/overlays/local }
    "k8s-down"     { kubectl delete -k deploy/k8s/overlays/local --ignore-not-found }
    default {
        Write-Host "Targets: install lint fmt typecheck test cov check clean compose-up compose-down k8s-up k8s-down"
    }
}
