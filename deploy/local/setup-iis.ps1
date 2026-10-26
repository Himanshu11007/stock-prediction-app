# deploy/local/setup-iis.ps1 - one-time IIS setup for the StockLens website.
# Run in PowerShell opened with "Run as administrator":
#
#   powershell -ExecutionPolicy Bypass -File deploy\local\setup-iis.ps1
#
# 1. switches on IIS (static files only: the website is Blazor WebAssembly)
# 2. installs the Microsoft IIS URL Rewrite module (needed by the site's web.config)
# 3. creates C:\inetpub\stocklens (you may update it without admin rights) and the
#    site "StockLens" on http://localhost:8585 ($WebPort in common.ps1) (this laptop only)
# Then build and copy the website in a NORMAL PowerShell window: publish-web.ps1
# (run as administrator, git would later refuse the folders it created).
. "$PSScriptRoot\common.ps1"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'open PowerShell with "Run as administrator" and run this again' }

Write-Host '== IIS (a few minutes the first time)'
$features = 'IIS-WebServerRole', 'IIS-WebServer', 'IIS-CommonHttpFeatures', 'IIS-StaticContent', 'IIS-DefaultDocument',
            'IIS-HttpErrors', 'IIS-HttpCompressionStatic', 'IIS-RequestFiltering', 'IIS-ManagementConsole'
Enable-WindowsOptionalFeature -Online -FeatureName $features -All -NoRestart | Out-Null

Write-Host '== URL Rewrite module'
if (-not (Test-Path "$env:windir\System32\inetsrv\rewrite.dll")) {
    $msi = Join-Path $env:TEMP 'rewrite_amd64_en-US.msi'
    Invoke-WebRequest -UseBasicParsing -OutFile $msi `
        'https://download.microsoft.com/download/1/2/8/128E2E22-C1B9-44A4-BE2A-5859ED1D4592/rewrite_amd64_en-US.msi'
    $p = Start-Process msiexec.exe -ArgumentList '/i', "`"$msi`"", '/quiet', '/norestart' -Wait -PassThru
    if ($p.ExitCode -notin 0, 3010) { throw "URL Rewrite install failed (msiexec $($p.ExitCode))" }
}
Write-Host '   installed'

Write-Host '== site folder and IIS site'
New-Item -ItemType Directory -Force -Path $SitePath | Out-Null
icacls $SitePath /grant "$($env:USERDOMAIN)\$($env:USERNAME):(OI)(CI)M" /grant 'IIS_IUSRS:(OI)(CI)RX' | Out-Null
Import-Module WebAdministration
if (-not (Test-Path 'IIS:\AppPools\StockLens')) {
    New-WebAppPool -Name 'StockLens' | Out-Null
    Set-ItemProperty 'IIS:\AppPools\StockLens' -Name managedRuntimeVersion -Value ''    # static files: no .NET in IIS
}
if (-not (Get-Website -Name 'StockLens')) {
    # Windows Firewall blocks this port from other computers by default: this laptop only
    New-Website -Name 'StockLens' -Port $WebPort -IPAddress '*' -PhysicalPath $SitePath `
        -ApplicationPool 'StockLens' | Out-Null
} else {
    Set-ItemProperty 'IIS:\Sites\StockLens' -Name bindings -Value @{ protocol = 'http'; bindingInformation = "*:${WebPort}:" }
}
Start-Website -Name 'StockLens'
Write-Host ''
Write-Host 'Done. Next, in a normal (not administrator) PowerShell window:'
Write-Host '  powershell -ExecutionPolicy Bypass -File deploy\local\publish-web.ps1'
