# deploy/local/set-secret.ps1 NAME - add or change one setting (for example
# DRISHTI_API_KEY, or any other value from Render's Environment page). The value
# is typed hidden and written only to %LOCALAPPDATA%\StockLens\stocklens.env.
#
#   powershell -ExecutionPolicy Bypass -File deploy\local\set-secret.ps1 DRISHTI_API_KEY
#
# The API reads settings at start: the script restarts it.
param([Parameter(Mandatory = $true)][string]$Name)
. "$PSScriptRoot\common.ps1"

if ($Name -cnotmatch '^[A-Z_][A-Z0-9_]*$') { throw 'use the exact upper-case name, e.g. DRISHTI_API_KEY' }
$secure = Read-Host "Value for $Name (hidden)" -AsSecureString
$value = [Net.NetworkCredential]::new('', $secure).Password.Trim()
if (-not $value) { throw 'empty value - nothing changed' }
Set-EnvValue $Name $value
$value = $null
Write-Host "$Name saved."
if (Get-ScheduledTask -TaskName $ApiTask -ErrorAction SilentlyContinue) {
    Restart-StockLensApi
    Write-Host 'API restarted with the new setting.'
}
