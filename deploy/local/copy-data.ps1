# deploy/local/copy-data.ps1 - copy all data from the Render database into the
# local database (stocks, rankings, track record, news, users, settings).
#
#   powershell -ExecutionPolicy Bypass -File deploy\local\copy-data.ps1
#
# Asks for the Render "External Database URL" (hidden while you paste; never
# stored, printed or put on a command line). The local database is REPLACED
# by the copy; a backup of the copy is kept in %LOCALAPPDATA%\StockLens\backups.
# Run again at any time to take a fresh copy (stop the API first: the script
# does that for you and starts it again).
. "$PSScriptRoot\common.ps1"

Wait-Database
$secure = Read-Host 'Paste the Render External Database URL (hidden)' -AsSecureString
$src = [Net.NetworkCredential]::new('', $secure).Password
if ($src -notmatch '^postgres(ql)?://') { throw 'that does not look like a postgresql:// URL - nothing was changed' }

Stop-StockLensApi                      # no open connections while the tables are replaced

Write-Host '== downloading from Render (a minute or two)'
# The URL goes in on standard input, so it never appears in a process list.
$src | docker exec -i $DbName sh -c 'read -r SRC; SRC=$(printf %s "$SRC" | tr -d "\r"); pg_dump --no-owner --no-acl -Fc -d "$SRC" -f /tmp/render.dump'
$code = $LASTEXITCODE
$src = $null
if ($code -ne 0) { throw 'download failed (check the URL; the local database was not changed)' }

$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
docker cp "${DbName}:/tmp/render.dump" (Join-Path $Backups "render-$stamp.dump") | Out-Null
Write-Host "   backup: $Backups\render-$stamp.dump"

Write-Host '== loading into the local database'
docker exec $DbName pg_restore -U stocklens -d stocklens --clean --if-exists --no-owner --no-acl --single-transaction /tmp/render.dump
if ($LASTEXITCODE -ne 0) { throw 'restore failed - see the messages above' }
docker exec $DbName rm -f /tmp/render.dump

Import-StockLensEnv
Push-Location $App
try {
    $code = Invoke-Logged 'setup.log' "`"$Py`" -m alembic upgrade head"
    if ($code -ne 0) { throw "alembic upgrade failed - see $Logs\setup.log" }
} finally { Pop-Location }

Write-Host '== rows copied'
$sql = "select 'stocks', count(*) from companies union all select 'users', count(*) from users " +
       "union all select 'ranking results', count(*) from stock_analysis_results " +
       "union all select 'experimental calls', count(*) from predictions " +
       "union all select 'news articles', count(*) from news_articles"
docker exec $DbName psql -U stocklens -d stocklens -At -F ': ' -c $sql

if (Get-ScheduledTask -TaskName $ApiTask -ErrorAction SilentlyContinue) { Start-ScheduledTask -TaskName $ApiTask }
Write-Host 'Done.'
