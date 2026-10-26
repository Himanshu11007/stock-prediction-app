# deploy/local/common.ps1 - shared paths and helpers for running StockLens on
# a Windows laptop (dot-sourced by the other scripts; Windows PowerShell 5.1).
#
#   %LOCALAPPDATA%\StockLens\app            clone of GitHub main (the code that runs)
#   %LOCALAPPDATA%\StockLens\venv           Python 3.11 with requirements-cloud.txt
#   %LOCALAPPDATA%\StockLens\stocklens.env  settings and secrets (only your user can read it)
#   %LOCALAPPDATA%\StockLens\logs           api.log and one log per job
#   %LOCALAPPDATA%\StockLens\backups        database copies
#   Docker container stocklens-db           PostgreSQL 16 on 127.0.0.1:5432 (volume stocklens-pgdata)
#   C:\inetpub\stocklens                    the website, served by IIS on http://localhost:8080
#
# Secrets are never printed: the env file is read literally (no expansion).

$ErrorActionPreference = 'Stop'
$RepoUrl   = 'https://github.com/Himanshu11007/stock-prediction-app.git'
$Root      = Join-Path $env:LOCALAPPDATA 'StockLens'
$App       = Join-Path $Root 'app'
$Venv      = Join-Path $Root 'venv'
$Py        = Join-Path $Venv 'Scripts\python.exe'
$EnvFile   = Join-Path $Root 'stocklens.env'
$Logs      = Join-Path $Root 'logs'
$Backups   = Join-Path $Root 'backups'
$DbName    = 'stocklens-db'
$ApiUrl    = 'http://localhost:8000'
$WebUrl    = 'http://localhost:8080'
$SitePath  = 'C:\inetpub\stocklens'
$ApiTask   = 'StockLens API'

function Import-StockLensEnv {
    if (-not (Test-Path -LiteralPath $EnvFile)) { throw "missing $EnvFile - run deploy\local\setup.ps1 first" }
    foreach ($line in Get-Content -LiteralPath $EnvFile -Encoding UTF8) {
        if ($line -match '^\s*(#|$)' -or $line -notmatch '=') { continue }
        $key, $value = $line -split '=', 2
        if ($key -cmatch '^[A-Z_][A-Z0-9_]*$') { [Environment]::SetEnvironmentVariable($key, $value, 'Process') }
    }
    $env:PYTHONUTF8 = '1'                  # log files in UTF-8 (rupee signs), not the Windows code page
    $env:PYTHONUNBUFFERED = '1'
}

function Get-EnvValue([string]$Key) {
    if (-not (Test-Path -LiteralPath $EnvFile)) { return $null }
    $line = Get-Content -LiteralPath $EnvFile -Encoding UTF8 | Where-Object { $_ -clike "$Key=*" } | Select-Object -First 1
    if ($line) { return ($line -split '=', 2)[1] } else { return $null }
}

function Set-EnvValue([string]$Key, [string]$Value) {
    $lines = @(Get-Content -LiteralPath $EnvFile -Encoding UTF8)
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -clike "$Key=*") { $lines[$i] = "$Key=$Value"; $found = $true }
    }
    if (-not $found) { $lines += "$Key=$Value" }
    [IO.File]::WriteAllLines($EnvFile, [string[]]$lines, (New-Object Text.UTF8Encoding $false))
}

function New-Secret([int]$Bytes = 32) {
    $b = New-Object byte[] $Bytes
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    return -join ($b | ForEach-Object { $_.ToString('x2') })
}

function Get-Sha256Hex([string]$Text) {
    $h = [Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($Text))
    return -join ($h | ForEach-Object { $_.ToString('x2') })
}

function Write-Log([string]$File, [string]$Text) {
    Add-Content -LiteralPath (Join-Path $Logs $File) -Value ("=== {0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Text) -Encoding UTF8
}

function Invoke-Logged([string]$LogName, [string]$CommandLine) {
    # cmd redirection: Python's stderr would otherwise become PowerShell errors
    $log = Join-Path $Logs $LogName
    cmd /c "$CommandLine >> `"$log`" 2>&1"
    return $LASTEXITCODE
}

function Test-Native([string]$CommandLine) {
    # exit code only; output discarded through cmd (in Windows PowerShell 5.1 a
    # redirected native error stream would stop the script)
    cmd /c "$CommandLine >nul 2>&1"
    return ($LASTEXITCODE -eq 0)
}

function Stop-StockLensApi {
    # the scheduled task's process, then whatever still listens on port 8000
    # (ending a task does not end the Python process it started)
    if (Get-ScheduledTask -TaskName $ApiTask -ErrorAction SilentlyContinue) { Stop-ScheduledTask -TaskName $ApiTask }
    Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

function Restart-StockLensApi {
    Stop-StockLensApi
    if (Get-ScheduledTask -TaskName $ApiTask -ErrorAction SilentlyContinue) { Start-ScheduledTask -TaskName $ApiTask }
}

function Wait-Database {
    for ($i = 0; $i -lt 60; $i++) {
        if (Test-Native "docker exec $DbName pg_isready -q -U stocklens -d stocklens") { return }
        Start-Sleep -Seconds 2
    }
    throw "database container $DbName did not become ready (is Docker Desktop running?)"
}
