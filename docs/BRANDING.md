# StockLens Brand

The product name is **StockLens**, with no suffix. Its score is the
**StockLens Score**. The fundamental framework keeps its own name,
**Fundamental Quality & Value Framework (FQVF)**:

- StockLens Score: 84/100
- FQVF: 16/18

Top Picks are called **Top Investment Candidates** (or Top Candidates).
They are analytical rankings, never guaranteed returns.

## Where the name lives

| Surface | Source |
|---|---|
| Backend, API metadata, web app, admin console, OTP email | `config.PRODUCT_NAME`, `PRODUCT_TAGLINE` and `PRODUCT_DESCRIPTION` |
| Mobile app | `GET /app/config` returns `product_name`; the display name is `<ApplicationTitle>` in the mobile csproj (Android label, iOS display name) |
| Notifications | Default templates in `notifications/settings.py`. Administrators can override them via `notifications.templates` |

## Icon assets (single master)

The approved master icon is `branding/stocklens-icon-master.png`.
`python scripts/generate_brand_assets.py` regenerates everything from it.
Nothing is redrawn.

| Asset | Files |
|---|---|
| Website / API / admin console | `branding/favicon.ico`, `favicon-16.png`, `favicon-32.png`, `apple-touch-icon.png`, `icon-192.png`, `icon-512.png`, `og-image.png`; served at `/static/branding/` and `/favicon.ico`; used by `/docs`, `/redoc` and both Streamlit apps |
| Mobile app icon | `Resources/AppIcon/stocklens_icon.png` (foreground) and `appicon.svg` (black background matching the master's corners). Android uses `ForegroundScale` 0.65 so circular and squircle masks never crop the artwork; iOS uses 1.0 |
| Mobile splash | `Resources/Splash/stocklens_splash.png` on `#0D1117` |
| Android notification icon | `Platforms/Android/Resources/drawable-*/ic_stat_stocklens.png`: the master's magnifier and chart as a white silhouette, as Android requires. Wired via manifest meta-data and the FCM payload. iOS uses the app icon for notifications automatically |

## Technical identifiers intentionally kept

Renaming these would break existing data, deployments, installations or
sign-in configuration. None of them is shown to users.

| Identifier | Why kept |
|---|---|
| API field `stockai_score`, C# `StockAiScore` | API contract between backend and mobile clients |
| Repositories `stock-prediction-app`, `StockAIPro.Mobile`; C# namespaces and projects `StockAIPro.Mobile.*` | Repositories are not renamed; namespaces are code identifiers |
| Android `ApplicationId` `com.companyname.stockaipro.mobile` (also the iOS bundle id) | Changing it creates a new app identity: existing installs, signing, Google OAuth client, Firebase app and Apple Sign-In configuration are bound to it. Set the final store id once, before the first store release |
| URL scheme `stockaipro://` | Deep links already issued and documented; not user-visible |
| Android notification channel id `stockai_alerts` | The channel *name* shown to users is "StockLens alerts"; changing the id would orphan users' channel settings |
| Env var `STOCKAI_API_URL`, build property `StockAIProApiBaseUrl`, storage path `stockai_storage`, secure-storage keys | Deployment and configuration contracts |
| Alembic migrations, `scripts/research/output/*` | Historical records, not edited |
