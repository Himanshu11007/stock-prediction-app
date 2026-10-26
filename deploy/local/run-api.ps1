# deploy/local/run-api.ps1 - run the StockLens API on http://localhost:8000
# (started at log-on by the "StockLens API" scheduled task; output in
# %LOCALAPPDATA%\StockLens\logs\api.log). If the API or the database stops,
# it is started again after 15 seconds; Stop-StockLensApi ends the loop.
. "$PSScriptRoot\common.ps1"

Set-Location $App
while ($true) {
    try {
        Import-StockLensEnv                                  # re-read: settings may have changed
        Test-Native "docker start $DbName" | Out-Null        # Docker Desktop may have started without it
        Wait-Database
        Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |   # an API left over
            ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
        Write-Log 'api.log' 'start API'
        $code = Invoke-Logged 'api.log' "`"$Py`" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1"
        Write-Log 'api.log' "API stopped, exit $code - restarting in 15 s"
    } catch {
        Write-Log 'api.log' "API not started: $($_.Exception.Message) - retrying in 15 s"
    }
    Start-Sleep -Seconds 15
}
