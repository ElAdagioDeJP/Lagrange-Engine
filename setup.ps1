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
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
} else {
    Write-Host 'No se encontró requirements.txt' -ForegroundColor Red
}
Write-Host 'Lanzando la aplicacion...' -ForegroundColor Cyan
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
if (Test-Path $venvPython) { $Python = $venvPython }

# Ejecutar desde raíz usando el paquete completo
try {
    & $Python -m src.ui.main_app
} catch {
    Write-Host "Error ejecutando la aplicación: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
