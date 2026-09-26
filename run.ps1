<#
.SYNOPSIS
    Start OmniLinker on Windows with one command.

.DESCRIPTION
    The default path needs no Docker, no MongoDB and no Neo4j: the embedded
    store backend is a JSON document store plus an in-memory adjacency graph,
    so this script installs the backend, builds the frontend if needed, and
    serves both from a single uvicorn process.

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Reinstall
    .\run.ps1 -NoUi
    .\run.ps1 -Docker
#>
[CmdletBinding()]
param(
    [switch]$Reinstall,
    [switch]$NoUi,
    [switch]$Docker,
    [int]$Port = 0
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Venv = Join-Path $Backend '.venv'

function Say($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function Die($message) { Write-Host "Error: $message" -ForegroundColor Red; exit 1 }

if ($Port -eq 0) {
    # 8900, not 8000: 8000 is conventionally an OpenAI-compatible server,
    # and a collision there surfaces as a 502 from whatever fronts the model
    # API rather than as an obvious "port in use". Override with -Port or
    # $env:OMNI_PORT.
    $Port = if ($env:OMNI_PORT) { [int]$env:OMNI_PORT } else { 8900 }
}

# --- docker profile --------------------------------------------------------
if ($Docker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Die @"
--Docker needs Docker, which is not installed on this machine.
The default path (no switches) runs with no external services at all.
"@
    }
    Say 'building the frontend'
    if (-not (Test-Path (Join-Path $Frontend 'node_modules'))) {
        Push-Location $Frontend; npm install; Pop-Location
    }
    Push-Location $Frontend; npm run build; Pop-Location
    Say 'docker compose up (Mongo + Neo4j + API)'
    Push-Location $Root
    docker compose up --build
    Pop-Location
    exit 0
}

# --- python ----------------------------------------------------------------
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) { Die 'python not found on PATH' }

if ($Reinstall -and (Test-Path $Venv)) { Remove-Item $Venv -Recurse -Force }
if (-not (Test-Path $Venv)) {
    Say 'creating the virtualenv'
    & $pythonCmd.Source -m venv $Venv
}
$Py = Join-Path $Venv 'Scripts\python.exe'

# --only-binary matters: pydantic-core has no source build path without a Rust
# toolchain, and the resulting error is very long and says nothing useful.
& $Py -c 'import fastapi, uvicorn, cryptography, pytest' 2>$null
if ($LASTEXITCODE -ne 0) {
    Say 'installing backend dependencies'
    & $Py -m pip install --quiet --upgrade pip
    & $Py -m pip install --quiet --only-binary ':all:' -r (Join-Path $Backend 'requirements.txt')
}

# --- frontend --------------------------------------------------------------
$buildUi = -not $NoUi
if ($buildUi) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        if (-not (Test-Path (Join-Path $Frontend 'node_modules'))) {
            Say 'installing frontend dependencies'
            Push-Location $Frontend; npm install; Pop-Location
        }
        $distIndex = Join-Path $Frontend 'dist\index.html'
        $needsBuild = -not (Test-Path $distIndex)
        if (-not $needsBuild) {
            $newest = Get-ChildItem (Join-Path $Frontend 'src') -Recurse -File |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($newest -and $newest.LastWriteTime -gt (Get-Item $distIndex).LastWriteTime) {
                $needsBuild = $true
            }
        }
        if ($needsBuild) {
            Say 'building the frontend'
            Push-Location $Frontend; npm run build; Pop-Location
        } else {
            Say 'frontend build is current'
        }
        $env:OMNI_FRONTEND_DIST = Join-Path $Frontend 'dist'
        Say "frontend: $env:OMNI_FRONTEND_DIST"
    } else {
        Say 'npm not found - serving the API only (see / for instructions)'
        $buildUi = $false
    }
}

# --- run -------------------------------------------------------------------
if (-not $env:OMNI_DATA_DIR) { $env:OMNI_DATA_DIR = Join-Path $Backend 'omni-data' }
if (-not $env:OMNI_VECTOR) { $env:OMNI_VECTOR = '1' }
New-Item -ItemType Directory -Force -Path $env:OMNI_DATA_DIR | Out-Null

Say "serving on http://127.0.0.1:$Port"
Say "data dir: $($env:OMNI_DATA_DIR)"
Say "seed the demo workspace:"
Say "  curl.exe -XPOST http://127.0.0.1:$Port/api/sync -H ""content-type: application/json"" -d '{""connector_id"":""demo""}'"

Push-Location $Backend
& $Py -m uvicorn omnilinker.api.app:app --host 127.0.0.1 --port $Port
Pop-Location
