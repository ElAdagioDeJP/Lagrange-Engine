# Script mínimo de setup para Lagrange-Engine
# Crea (si no existe) y activa un entorno virtual + instala requirements.

param(
    [switch]$Recreate,
    [string]$Python = 'python'
)

$venvPath = Join-Path $PSScriptRoot 'venv'

if ($Recreate -and (Test-Path $venvPath)) {
    Write-Host 'Eliminando venv existente...' -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Host 'Creando entorno virtual...' -ForegroundColor Cyan
    & $Python -m venv $venvPath
}

Write-Host 'Activando entorno virtual...' -ForegroundColor Cyan
. "$venvPath/Scripts/Activate.ps1"

if (Test-Path 'requirements.txt') {
    Write-Host 'Instalando dependencias...' -ForegroundColor Cyan
    pip install -r requirements.txt
} else {
    Write-Host 'No se encontró requirements.txt' -ForegroundColor Red
}

Write-Host 'Listo. Ejecuta: python -m src.ui.main_app' -ForegroundColor Green

Write-Host 'Lanzando la aplicacion...' -ForegroundColor Cyan
python -m src.ui.main_app
