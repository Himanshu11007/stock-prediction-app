# deploy/local/run-job.ps1 JOB - run one StockLens job (called by the scheduled
# tasks; you can also run it by hand):
#
#   ranking            the daily ranking, run here against the local database
#   TODAY_PREOPEN | TOMORROW_EOD | OUTCOME_EVALUATION | SNAPSHOT_MONITOR | NEWS_INGESTION
#                      triggered on the local API (scripts/scheduler_client.py)
#   update             fast-forward the code from GitHub main, reinstall packages
#                      when requirements-cloud.txt changed, restart the API
#
#   powershell -ExecutionPolicy Bypass -File deploy\local\run-job.ps1 TODAY_PREOPEN
#
# One run per job at a time (a second start while one runs is skipped). Output
# is appended to %LOCALAPPDATA%\StockLens\logs\JOB.log. Every job is safe to
# repeat (the backend's slot ledger and the ranking's own guard make duplicates
# no-ops), so a run caught up after the laptop wakes is harmless.
param([Parameter(Mandatory = $true)]
      [ValidateSet('ranking', 'TODAY_PREOPEN', 'TOMORROW_EOD', 'OUTCOME_EVALUATION', 'SNAPSHOT_MONITOR',
                   'NEWS_INGESTION', 'update')]
      [string]$Job)
. "$PSScriptRoot\common.ps1"

$log = "$Job.log"
$mutex = New-Object Threading.Mutex($false, "Local\StockLens-$Job")
try { $owned = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $owned = $true }   # last run crashed
if (-not $owned) { Write-Log $log "previous $Job still running: this start skipped"; exit 0 }
$code = 1
try {
    Write-Log $log "start $Job"
    Import-StockLensEnv
    Set-Location $App
    switch ($Job) {
        'ranking' {
            Wait-Database
            $code = Invoke-Logged $log "`"$Py`" scripts\scheduled_jobs.py ranking"
        }
        'update' {
            $before = (Get-FileHash requirements-cloud.txt).Hash
            $code = Invoke-Logged $log 'git pull -q --ff-only'
            if ($code -eq 0) {
                if ((Get-FileHash requirements-cloud.txt).Hash -ne $before) {
                    Write-Log $log 'requirements changed: reinstalling'
                    $code = Invoke-Logged $log "`"$Py`" -m pip install -q --disable-pip-version-check -r requirements-cloud.txt"
                }
                Invoke-Logged $log "`"$Py`" -m alembic upgrade head" | Out-Null
                Invoke-Logged $log 'git log -1 --oneline' | Out-Null
                Restart-StockLensApi
            }
        }
        default {
            $code = Invoke-Logged $log "`"$Py`" scripts\scheduler_client.py --job $Job --timeout-minutes 40"
        }
    }
    Write-Log $log "end $Job exit $code"
} finally {
    $mutex.ReleaseMutex()
}
exit $code
