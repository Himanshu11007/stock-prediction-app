# Mobile App Setup (StockAIPro.Mobile)

Repository: https://github.com/Himanshu11007/StockAIPro.Mobile (branch `master`).
.NET 10 SDK with the `android`, `ios` and `maccatalyst` workloads.

## API URL configuration

The app reads one build-time setting, `StockAIProApiBaseUrl`
(`StockAIPro.Mobile.Core/Services/Configuration/ApiConfiguration.cs`):

| Scenario | Command / setting | Resulting URL |
|---|---|---|
| Android emulator, local backend | Debug build, no property | `http://10.0.2.2:8000` |
| Windows / iOS simulator, local backend | Debug build, no property | `http://127.0.0.1:8000` |
| Physical Android device via USB | `adb reverse tcp:8000 tcp:8000`, Debug build, no property | device `127.0.0.1:8000` → host |
| Physical device on the LAN | `-p:StockAIProApiBaseUrl=http://192.168.x.y:8000` (also add the IP to `Platforms/Android/Resources/xml/network_security_config.xml` for cleartext) | that URL |
| Production | `-p:StockAIProApiBaseUrl=https://api.your-domain` | HTTPS only |

Release builds **fail** without the property or with a non-HTTPS URL
(`ValidateApiBaseUrl` target), so a developer machine address can never
ship. The backend must listen on `0.0.0.0` for emulators/devices:
`uvicorn api.main:app --host 0.0.0.0 --port 8000`.

## Build

```bash
# tests (platform-independent Core library)
dotnet test StockAIPro.Mobile.Tests/StockAIPro.Mobile.Tests.csproj

# Android, installable standalone Debug APK (assemblies embedded)
dotnet build StockAIPro.Mobile/StockAIPro.Mobile.csproj -f net10.0-android -c Debug \
    -p:AndroidPackageFormat=apk -p:EmbedAssembliesIntoApk=true
#   -> StockAIPro.Mobile/bin/Debug/net10.0-android/com.companyname.stockaipro.mobile-Signed.apk

# Android Release
dotnet publish StockAIPro.Mobile/StockAIPro.Mobile.csproj -f net10.0-android -c Release \
    -p:StockAIProApiBaseUrl=https://api.your-domain

# Deploy + run on a connected emulator/device during development
dotnet build StockAIPro.Mobile/StockAIPro.Mobile.csproj -f net10.0-android -t:Run
```

A plain Debug build uses .NET *Fast Deployment*: its APK does not contain the
app assemblies and aborts if installed with `adb install`. Use `-t:Run`, or
`EmbedAssembliesIntoApk=true` for a standalone APK.

iOS: the `net10.0-ios` target compiles on Windows, but producing and signing
an `.app`/`.ipa` requires a Mac with Xcode (paired via Visual Studio or built
on macOS: `dotnet publish -f net10.0-ios -c Release -p:StockAIProApiBaseUrl=…`).
Apple Sign-In needs the entitlement and provider configuration in
docs/AUTHENTICATION.md.

Production checklist: set a real `ApplicationId` (currently
`com.companyname.stockaipro.mobile`), signing keystore/certificates, the
HTTPS API URL, and the Google/Apple client configuration.

## Screens

Home (market regime, announcement, disclaimer), Analyse (Stock Master
search → analysis), Stock Analysis (`/stock/{symbol}`: StockAI Score,
components, positive factors, risks, 18 FQVF checks, market/technical data,
informational ML signal, data freshness, engine version, on-demand
analysis), Top Investment Candidates, Watchlist (links to analysis),
Performance, AI Intelligence, Account (sessions, PIN, linked identities),
Login/Register/OTP/PIN. Navigation items follow the backend `app.features`
flags. Loading, empty, error, session-expired and offline states use the
shared `LoadingState`/`EmptyState`/`ErrorState` components; malformed
responses show an error instead of crashing.
