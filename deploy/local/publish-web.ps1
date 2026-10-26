# deploy/local/publish-web.ps1 - build the website (StockLens.Web, GitHub main)
# and copy it to the IIS site folder C:\inetpub\stocklens. Run it again after
# website changes are merged. No admin rights needed once setup-iis.ps1 has run.
#
#   powershell -ExecutionPolicy Bypass -File deploy\local\publish-web.ps1
#
# The website is pointed at the local API (http://localhost:8000).
. "$PSScriptRoot\common.ps1"

$WebRepo = 'https://github.com/Himanshu11007/StockLens.Web.git'
$WebSrc  = Join-Path $Root 'web-src'
$Publish = Join-Path $Root 'web-publish'

Write-Host '== website code (GitHub main; uses your saved GitHub sign-in)'
if (-not (Test-Path (Join-Path $WebSrc '.git'))) {
    git clone -q --recurse-submodules $WebRepo $WebSrc
} else {
    git -C $WebSrc pull -q --ff-only
}
if ($LASTEXITCODE -ne 0) { throw 'git clone / pull of StockLens.Web failed' }
git -C $WebSrc submodule update -q --init --recursive
git -C $WebSrc log -1 --format='   now at %h %s'

Write-Host '== build (a minute or two)'
if (Test-Path $Publish) { Remove-Item -Recurse -Force $Publish }
dotnet publish (Join-Path $WebSrc 'src\StockLens.Web\StockLens.Web.csproj') -c Release -o $Publish -nologo -v q
if ($LASTEXITCODE -ne 0) { throw 'dotnet publish failed' }
$wwwroot = Join-Path $Publish 'wwwroot'
if (-not (Test-Path (Join-Path $wwwroot 'index.html')) -or -not (Test-Path (Join-Path $Publish 'web.config'))) {
    throw 'unexpected publish output (no index.html or web.config)'
}

# Local API instead of Render; the Google client id is public (it is in the page anyway).
$settings = '{ "StockLens": { "ApiBaseUrl": "' + $ApiUrl + '", ' +
            '"GoogleClientId": "160173631539-ed5fq3rtvvsdob80g8f2k54631kc7pud.apps.googleusercontent.com" } }'
[IO.File]::WriteAllText((Join-Path $wwwroot 'appsettings.json'), $settings, (New-Object Text.UTF8Encoding $false))
Remove-Item (Join-Path $wwwroot 'appsettings.json.br'), (Join-Path $wwwroot 'appsettings.json.gz') -ErrorAction SilentlyContinue

Write-Host "== copy to $SitePath"
if (-not (Test-Path $SitePath)) { throw "$SitePath does not exist - run setup-iis.ps1 (as administrator) first" }
robocopy $Publish $SitePath /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "copy to $SitePath failed (robocopy code $LASTEXITCODE)" }
Write-Host "Done. Open $WebUrl"
