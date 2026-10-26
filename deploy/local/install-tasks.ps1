# deploy/local/install-tasks.ps1 - Windows Task Scheduler entries for StockLens
# (no admin rights needed; run again after changes, it replaces them):
#
#   powershell -ExecutionPolicy Bypass -File deploy\local\install-tasks.ps1
#
# "StockLens API" starts at log-on and keeps the API running. The jobs run at
# the same India times as the server schedule (deploy/vps/stocklens.cron);
# market jobs Monday-Friday (the backend skips NSE holidays itself), news
# hourly 06:10-22:10 every day. A run missed while the laptop was off or asleep
# starts as soon as it is back (repeats are harmless). TODAY_CONFIRMED stays off.
# Remove everything with:  install-tasks.ps1 -Remove
param([switch]$Remove)
. "$PSScriptRoot\common.ps1"

$prefix = 'StockLens '
Get-ScheduledTask -TaskName "$prefix*" -ErrorAction SilentlyContinue |
    ForEach-Object { Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false }
if ($Remove) { Stop-StockLensApi; Write-Host 'StockLens tasks removed.'; return }

$user = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
function Get-Action([string]$Script, [string]$Arguments = '') {
    New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $Root `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PSScriptRoot\$Script`" $Arguments"
}
$weekdays = 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
function Weekdays([string[]]$Times) { $Times | ForEach-Object { New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $_ } }

# API: at log-on, no time limit
$apiSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $ApiTask -Action (Get-Action 'run-api.ps1') -Principal $principal `
    -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $user) -Settings $apiSettings | Out-Null

# jobs (same times as deploy/vps/stocklens.cron, IST = this laptop's time zone)
$jobSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
$news = New-ScheduledTaskTrigger -Daily -At '06:10'
$news.Repetition = (New-ScheduledTaskTrigger -Once -At '06:10' -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Hours 16 -Minutes 1)).Repetition
$jobs = [ordered]@{
    'NEWS_INGESTION'     = @($news)
    'TODAY_PREOPEN'      = @(Weekdays '08:05', '08:35')
    'SNAPSHOT_MONITOR'   = @(Weekdays '10:20', '16:55', '20:55')
    'TOMORROW_EOD'       = @(Weekdays '16:20', '19:35')
    'ranking'            = @(Weekdays '16:35', '18:05', '20:05')
    'OUTCOME_EVALUATION' = @(Weekdays '16:45', '21:15')
    'update'             = @(New-ScheduledTaskTrigger -Daily -At '05:30')
}
foreach ($job in $jobs.Keys) {
    Register-ScheduledTask -TaskName "$prefix$job" -Action (Get-Action 'run-job.ps1' $job) -Principal $principal `
        -Trigger $jobs[$job] -Settings $jobSettings | Out-Null
}

Start-ScheduledTask -TaskName $ApiTask
Write-Host 'Installed:'
Get-ScheduledTask -TaskName "$prefix*" | ForEach-Object {
    $next = (Get-ScheduledTaskInfo -TaskName $_.TaskName).NextRunTime
    '  {0,-30} next run {1}' -f $_.TaskName, $(if ($next) { $next.ToString('ddd dd MMM HH:mm') } else { 'at log-on (running now)' })
}
Write-Host "API starting on $ApiUrl (log: $Logs\api.log)"
