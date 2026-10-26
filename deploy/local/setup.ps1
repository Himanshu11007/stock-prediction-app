# deploy/local/setup.ps1 - one-time setup of StockLens on this laptop (no admin
# rights needed). Safe to run again.
#
#   powershell -ExecutionPolicy Bypass -File deploy\local\setup.ps1
#
# 1. clones GitHub main to %LOCALAPPDATA%\StockLens\app (or updates it)
# 2. creates the Python 3.11 environment and installs requirements-cloud.txt
# 3. creates the settings file with new random secrets (database password,
#    sign-in key, scheduler token); readable only by your Windows user
# 4. starts PostgreSQL 16 in Docker (container stocklens-db, data kept in the
#    Docker volume stocklens-pgdata, reachable only from this laptop)
# 5. creates the database tables (alembic upgrade head)
# Nothing secret is printed.
. "$PSScriptRoot\common.ps1"

Write-Host '== checks'
if (-not (Test-Native 'docker version')) { throw 'Docker Desktop is not running: start it and run this again' }
$py311 = $null
foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) { if (Test-Path $c) { $py311 = $c } }
if (-not $py311 -and (Test-Native 'py -3.11 --version')) { $py311 = (& py -3.11 -c "import sys; print(sys.executable)") }
if (-not $py311) { throw 'Python 3.11 not found (same version as the server): install it from python.org' }
New-Item -ItemType Directory -Force -Path $Root, $Logs, $Backups | Out-Null

Write-Host '== code (GitHub main)'
if (-not (Test-Path (Join-Path $App '.git'))) {
    git clone -q $RepoUrl $App
} else {
    git -C $App pull -q --ff-only
}
if ($LASTEXITCODE -ne 0) { throw 'git clone / pull failed' }
git -C $App log -1 --format='   now at %h %s'

Write-Host '== Python 3.11 and packages (a few minutes the first time)'
if (-not (Test-Path $Py)) { & $py311 -m venv $Venv }
& $Py -m pip install -q --disable-pip-version-check -r (Join-Path $App 'requirements-cloud.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
& $Py --version

Write-Host '== settings file'
if (-not (Test-Path -LiteralPath $EnvFile)) {
    $dbPassword = New-Secret 24
    $token = New-Secret 32
    $lines = @(
        '# StockLens settings for this laptop (created by deploy\local\setup.ps1).',
        '# KEY=value, read literally. Never share or commit this file.',
        'APP_ENV=production',
        "LOCAL_DB_PASSWORD=$dbPassword",
        "DATABASE_URL=postgresql://stocklens:$dbPassword@127.0.0.1:5432/stocklens",
        "JWT_SECRET_KEY=$(New-Secret 48)",
        'ENGINE_RUN_MAX_WORKERS=4',
        'ENGINE_RUN_ALLOW_ML=false',
        "# Google sign-in (the public web client id; also add $WebUrl as an",
        '# authorised JavaScript origin of this client in Google Cloud Console)',
        'GOOGLE_OAUTH_CLIENT_ID=160173631539-ed5fq3rtvvsdob80g8f2k54631kc7pud.apps.googleusercontent.com',
        '# jobs on this laptop call the local API with this token; the API keeps only its SHA-256',
        "STOCKLENS_API_URL=$ApiUrl",
        "STOCKLENS_SCHEDULER_TOKEN=$token",
        "SCHEDULER_TOKEN_SHA256=$(Get-Sha256Hex $token)",
        '# News (same values as on Render). Add the key with deploy\local\set-secret.ps1 DRISHTI_API_KEY',
        'DRISHTI_ENABLED=true',
        'DRISHTI_ANNOUNCEMENTS_ENABLED=true',
        'NEWS_ALLOW_RESEARCH_PROVIDERS=true',
        'NEWS_PROVIDERS=official_rss,drishti,drishti_announcements',
        'DRISHTI_ANNOUNCEMENTS_MAX_ITEMS=15'
    )
    [IO.File]::WriteAllLines($EnvFile, [string[]]$lines, (New-Object Text.UTF8Encoding $false))
    icacls $EnvFile /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null      # only you can read it
    Write-Host "   created $EnvFile (new random secrets)"
} else {
    Write-Host "   kept existing $EnvFile"
}
# website address: kept in step with $WebPort (common.ps1) on every run
Set-EnvValue 'CORS_ALLOWED_ORIGINS' "$WebUrl,http://127.0.0.1:$WebPort"
Set-EnvValue 'FRONTEND_BASE_URL' $WebUrl

Write-Host '== database (PostgreSQL 16 in Docker)'
$exists = docker ps -a --filter "name=^$DbName$" --format '{{.Names}}'
if (-not $exists) {
    $env:POSTGRES_PASSWORD = Get-EnvValue 'LOCAL_DB_PASSWORD'        # passed by name: not on the command line
    docker run -d -q --name $DbName --restart unless-stopped -e POSTGRES_USER=stocklens -e POSTGRES_DB=stocklens `
        -e POSTGRES_PASSWORD -p 127.0.0.1:5432:5432 -v stocklens-pgdata:/var/lib/postgresql/data postgres:16 | Out-Null
    Remove-Item Env:\POSTGRES_PASSWORD
    if ($LASTEXITCODE -ne 0) { throw 'docker run failed' }
} else {
    docker start $DbName | Out-Null
}
Wait-Database
Write-Host '   running on 127.0.0.1:5432 (only this laptop can connect)'

Write-Host '== database tables'
Import-StockLensEnv
Push-Location $App
try {
    $code = Invoke-Logged 'setup.log' "`"$Py`" -m alembic upgrade head"
    if ($code -ne 0) { throw "alembic upgrade failed - see $Logs\setup.log" }
} finally { Pop-Location }
Write-Host '   up to date'

Write-Host ''
Write-Host 'Done. Next:'
Write-Host '  1. copy your data from Render:  powershell -ExecutionPolicy Bypass -File deploy\local\copy-data.ps1'
Write-Host '  2. add the news key:             powershell -ExecutionPolicy Bypass -File deploy\local\set-secret.ps1 DRISHTI_API_KEY'
Write-Host '  3. start API + daily jobs:       powershell -ExecutionPolicy Bypass -File deploy\local\install-tasks.ps1'
Write-Host '  4. website in IIS (as admin):    powershell -ExecutionPolicy Bypass -File deploy\local\setup-iis.ps1'
