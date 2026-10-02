# StockAI Pro Authentication — Phase 8 / 8.1

This document covers the Phase 8 authentication expansion (Google/Apple
sign-in, OTP login, device/session management, account linking, and the
mobile PIN model) and Phase 8.1 (real native provider integration: Android
Google Sign-In via Credential Manager, iOS Sign in with Apple, and a real
SMTP OTP delivery adapter). It assumes the Phase 1-6 foundation
(users/roles, bcrypt passwords, JWT access tokens, opaque hashed refresh
tokens with rotation) already described in the codebase itself
(`auth/service.py`, `auth/security.py`).

**The backend is the sole authentication authority.** The mobile app never
independently decides a user is authenticated — it only ever holds and
presents tokens the backend issued, and always re-validates against the
backend (via refresh) when in doubt.

## 1. Architecture

```
Google / Apple id_token  ──┐
Phone/email OTP          ──┼──► backend verifies ──► find-or-create user ──► JWT + refresh token
Password (existing)      ──┘
```

Every login path — password, Google, Apple, OTP — ends up calling the same
`auth/token_issuance.py:issue_token_pair()`, so they all produce tokens
exactly the same way and all participate in device/session tracking
identically. There is no separate "SSO token" or "OTP token" format.

New tables (see `db/models/user.py`), each owned by one module:

| Table                 | Owning module              | Purpose |
|-----------------------|-----------------------------|---------|
| `external_identities` | `auth/external_identity.py` | Google/Apple `(provider, provider_subject)` → user |
| `otp_challenges`      | `auth/otp_service.py`       | Hashed, expiring, attempt-limited OTP codes |
| `trusted_devices`     | `auth/device_service.py`    | "this user completed full auth on this device"; `pin_enabled` flag only |
| `refresh_tokens.device_id` | `auth/service.py` (existing table, new column) | Lets sessions be listed/revoked per device |

`users.email`, `users.phone`, and `users.hashed_password` are all nullable:
a Google/Apple/OTP-only account may have no password, and a phone-only OTP
account may have no email.

**JWT subject**: access tokens now carry the user's numeric id as `sub`
(not email), since some accounts have no email at all. `get_current_user`
accepts *either* an id or an email in `sub`, so any already-issued
email-subject token keeps working until it naturally expires (15 minutes).

## 2. Google Sign-In

- **Android** obtains a Google **id_token** via Android's current supported
  mechanism, Credential Manager (`androidx.credentials` +
  `com.google.android.libraries.identity.googleid`'s `GetGoogleIdOption`) -
  see "Mobile" section below. This is deliberately NOT the older
  `GoogleSignInClient`/`GoogleSignInOptions` API (Google is sunsetting it)
  and NOT a browser-redirect OAuth flow (Google blocked custom-URI-scheme
  redirects for Android apps in 2022 - both facts confirmed directly
  against Google's current developer documentation before choosing this
  approach; see citations in the Phase 8.1 commit/PR description).
- Backend verifies it as a standard OIDC JWT against Google's own rotating
  public keys (`https://www.googleapis.com/oauth2/v3/certs`), checking
  signature, `aud` (must equal `GOOGLE_OAUTH_CLIENT_ID`), and `iss`.
- The client-supplied email is **never** trusted by itself — only `sub`,
  once the signature is verified, is used as the identity key.
- See `auth/external_identity.py:GoogleIdentityVerifier` /
  `find_or_create_user_for_identity()`.

**Setup required for production**:

1. Register an OAuth 2.0 **Web application** client in Google Cloud
   Console. Its client ID is the value used for BOTH:
   - `GOOGLE_OAUTH_CLIENT_ID` (backend, server config) and
   - `GoogleAuthConfiguration.ServerClientId` (mobile,
     `StockAIPro.Mobile.Core/Services/Configuration/GoogleAuthConfiguration.cs`) -
     this is `GetGoogleIdOption.SetServerClientId(...)`'s argument.
   - This client ID is **public client-side configuration, not a secret**:
     it identifies which backend a token was issued for (the token's
     `aud`), the same way a Firebase `google-services.json` API key is
     public - see the class's own doc comment for the full reasoning.
2. ALSO register a separate OAuth **Android** client in Google Cloud
   Console (package name `com.companyname.stockaipro.mobile` + the app's
   release/debug signing-certificate SHA-1 fingerprint). This registration
   authorizes the app build to use Credential Manager at all; it does not
   produce a separate id/secret the app needs to embed anywhere.
3. iOS Google Sign-In was **not implemented** in Phase 8.1 (Apple Sign-In
   was implemented for iOS instead - see below); adding it later would need
   its own iOS OAuth client registration plus a native binding/SDK, which
   is a separate, not-yet-done piece of work.

```
GOOGLE_OAUTH_CLIENT_ID=<your OAuth Web-application client id>
```

Without this set (server) or `GoogleAuthConfiguration.ServerClientId` left
empty (mobile), Google sign-in fails closed on both sides rather than
silently accepting anything.

## 3. Apple Sign-In

- **iOS** obtains an Apple **identity_token** via .NET MAUI's built-in
  `Microsoft.Maui.Authentication.AppleSignInAuthenticator` (ships with the
  MAUI SDK itself - no extra NuGet package), which wraps Apple's own
  `AuthenticationServices`/`ASAuthorizationAppleIdProvider` framework - the
  same native mechanism Apple's own documentation describes, not a browser
  workaround.
- Backend verifies it as a standard OIDC JWT against Apple's own rotating
  public keys (`https://appleid.apple.com/auth/keys`), checking signature,
  `aud` (must equal `APPLE_SERVICES_ID`), and `iss`.
- Apple only includes `email` on the **first** authorization (or when the
  user chooses private relay) — `sub` is the only field guaranteed present
  on every login and is the only one used as the identity key.
- See `auth/external_identity.py:AppleIdentityVerifier`.

**Setup required for production**:

1. An Apple Developer account with a registered App ID that has the
   "Sign In with Apple" capability enabled.
2. A Services ID (or the App ID itself, depending on your Apple Developer
   configuration) whose identifier is set as `APPLE_SERVICES_ID` on the
   backend (`aud` the backend validates every identity token against).
3. The `com.apple.developer.applesignin` entitlement, already added to
   `StockAIPro.Mobile/Platforms/iOS/Entitlements.plist` (value `Default`) -
   this must match what's enabled for the App ID in the Developer portal,
   or the native authorization request fails.
4. A full Xcode/provisioning-profile build with that entitlement applied -
   **not verified in this environment** (no Mac/Xcode available - see
   "Mobile" section below for exactly what was and wasn't checked).

```
APPLE_SERVICES_ID=<your Services ID or App ID>
```

Same fail-closed behavior as Google when unset.

## 4. OTP (email or phone)

`POST /auth/otp/request {destination}` → generates a code, stores only a
per-row-salted SHA-256 hash of it, and delivers it via whatever
`IOtpDeliveryService` is configured (see below). Always returns 204
regardless of whether `destination` has an account, so this endpoint can't
be used to enumerate registered users.

`POST /auth/otp/verify {destination, code}` → validates the code
(expiry, one-time-use, attempt limit), then finds-or-creates the StockAI
user for `destination` and issues tokens exactly like any other login path.

Configuration (`config.py`, all overridable via environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `OTP_CODE_LENGTH` | 6 | Digits per code |
| `OTP_EXPIRE_SECONDS` | 300 | Code lifetime |
| `OTP_MAX_ATTEMPTS` | 5 | Wrong guesses before a challenge is burned |
| `OTP_RESEND_COOLDOWN_SECONDS` | 30 | Minimum gap between two codes for the same destination |
| `OTP_MAX_REQUESTS_PER_WINDOW` | 5 | Cap per destination AND per requesting IP |
| `OTP_REQUEST_WINDOW_SECONDS` | 1800 | Rolling window the above cap applies over |

### OTP provider setup

**Email delivery is production-ready behind the abstraction; SMS is not.**
`auth/otp_delivery.py:SmtpOtpDeliveryService` sends real email over SMTP -
deliberately a *protocol*, not a vendor SDK, so it works with whichever
transactional-email provider the deployment already has (SendGrid, Mailgun,
Amazon SES, Postmark, a corporate relay, ...) without this codebase
hard-coding assumptions about any single vendor's proprietary API. It is
automatically selected by `get_otp_delivery_service()` once `OTP_SMTP_HOST`
is set:

```
OTP_SMTP_HOST=<your provider's SMTP host, e.g. smtp.sendgrid.net>
OTP_SMTP_PORT=587
OTP_SMTP_USERNAME=<your SMTP username>
OTP_SMTP_PASSWORD=<your SMTP password/API key>   # SECRET - env var only, never commit
OTP_SMTP_FROM_ADDRESS=no-reply@yourdomain.com
OTP_SMTP_USE_TLS=true
```

**No real SMS provider is wired up** - there is no SMTP-equivalent
universal protocol for SMS, so a phone-destination OTP request with only
`SmtpOtpDeliveryService` configured fails with
`UnsupportedDestinationError` rather than silently doing nothing. Adding
SMS support means picking a specific provider (e.g. Twilio) and writing a
new `IOtpDeliveryService` implementation for it once that choice is
actually made - this phase deliberately did not guess one.

**Live delivery has not been tested** - `SmtpOtpDeliveryService` is unit
tested against a mocked `smtplib.SMTP` connection (see
`tests/test_otp_delivery.py`), confirming this codebase's own logic
(message construction, destination validation, never logging the code) is
correct, but no real SMTP server/account was available in this environment
to send an actual email end-to-end.

For local development only, set:

```
OTP_DEV_LOG_CODES=true
```

which logs each generated code to the server's own log file instead of
delivering it anywhere. **This must never be set in production** — it
exists purely so a developer can exercise the full OTP flow without a real
provider account. All automated tests instead use
`auth/otp_delivery.py:FakeOtpDeliveryService`, a deterministic in-memory
double with no I/O at all.

## 5. Device / session management

Every login (password, Google, Apple, OTP) can optionally include a
client-generated `device_id` (+ `device_name`). When present:

- the resulting refresh token is tagged with that `device_id`, and stays
  tagged with it across every rotation (`auth/service.py:rotate_refresh_token`)
- a `trusted_devices` row is created/refreshed (`auth/device_service.py:upsert_trusted_device`)

Endpoints (all authenticated):

- `GET /auth/sessions` — list active sessions, newest first, with device
  name where known
- `POST /auth/sessions/revoke {session_id}` — revoke one session
- `POST /auth/sessions/revoke-all {}` — **sign out all devices**
- `POST /auth/sessions/revoke-all {except_current: true, current_device_id}`
  — sign out every *other* device
- `POST /auth/devices/pin-enabled {device_id, enabled}` — purely
  informational flag the mobile client flips after setting up/clearing its
  own local PIN (see below) — the backend never sees the PIN itself

Revoking a session only affects **refresh** capability. An already-issued
access token for that session keeps working until its own short expiry
(15 minutes) — there is no server-side access-token blacklist, by design
(see `api/routes/auth_devices.py` module docstring for why one isn't
needed).

## 6. The 4-digit PIN is NOT a remote credential

This is the most important security property in this phase:

```
Google / Apple / OTP / existing login
        ↓
Authenticated StockAI session (JWT + refresh token, as always)
        ↓
Trusted device established (trusted_devices row)
        ↓
User sets a 4-digit PIN                      ← entirely on-device
        ↓
PIN protects LOCAL access to the already-stored session
```

On every subsequent launch, the PIN is checked **entirely on the mobile
device**, using `Microsoft.Maui.Storage.SecureStorage` (same mechanism
`SecureTokenStore` already uses for tokens) — the raw PIN is never sent to
the backend, never stored server-side, and there is no `POST
/login-with-pin` endpoint. A correct PIN only unlocks whatever refresh
token is *already* cached locally; it never bypasses the server.

If that cached refresh token has since been revoked (sign-out-all-devices,
a revoked single session, or natural expiry), the very next API call 401s,
`AuthenticatedHttpMessageHandler` attempts the normal refresh, the server
rejects it, and the mobile app falls back to full re-authentication
(Google/Apple/OTP/password) exactly as if no PIN existed. **A four-digit
PIN never overrides server-side revocation.**

Why not verify the PIN on the server instead? A 4-digit PIN has only 10,000
possible values — treating it as a remote credential (`POST
/login-with-pin`) would make it brute-forceable server-side no matter how
aggressively that one endpoint were rate-limited, and would turn "guess a
4-digit number" into a full account-takeover primitive. Keeping it
local-only means the worst a leaked/guessed PIN can do is unlock a session
*that device already legitimately holds* — not establish a new one.

## 7. Account linking

An authenticated user can explicitly attach another sign-in method:

- `POST /auth/link/google {id_token}`
- `POST /auth/link/apple {identity_token}`
- `GET /auth/identities` — list linked providers
- `DELETE /auth/identities/{provider}` — unlink, rejected
  (`400`) if it would leave the account with no way to sign in at all

Linking is **never automatic**. If a brand-new Google/Apple/OTP identity's
email matches an existing StockAI account, login returns `409 Conflict`
with instructions to sign in to the existing account and link explicitly
from there (`auth/external_identity.py:AccountLinkingRequiredError`) —
email collision alone is never treated as proof of ownership.

## 8. Mobile — what's implemented vs. what's still required

**Implemented and compiles for real (Phase 8.1):**

- **Android Google Sign-In**: `StockAIPro.Mobile/Platforms/Android/GoogleSignInService.cs`,
  using `Xamarin.AndroidX.Credentials` 1.6.0.1 +
  `Xamarin.AndroidX.Credentials.PlayServicesAuth` 1.6.0.2 +
  `Xamarin.GoogleAndroid.Libraries.Identity.GoogleId` 1.1.0.13 (real,
  actively-published .NET for Android bindings for Google's current
  Credential Manager API - chosen after confirming Google's own
  documentation that the older `GoogleSignInClient` API and custom-URI-
  scheme OAuth redirects are both deprecated/blocked for Android). Verified
  with `dotnet build -f net10.0-android` - **full success, including D8/R8
  dexing** (not just a C# compile check).
- **iOS Apple Sign-In**: `StockAIPro.Mobile/Platforms/iOS/AppleSignInService.cs`,
  using the built-in `Microsoft.Maui.Authentication.AppleSignInAuthenticator`
  (ships with the MAUI SDK, no extra package). Verified with
  `dotnet build -f net10.0-ios` on this Windows machine - **the managed C#
  compiles successfully**, including picking up the new
  `Platforms/iOS/Entitlements.plist`. Full native app packaging/codesigning
  and any actual on-device/simulator run were **not** possible (no Mac/
  Xcode in this environment) - see Task 20/21 results in the final report.
- Device identity generation, local PIN storage/verification with lockout,
  OTP request/verify UI, `AuthApiClient`/`AuthService` extensions for every
  endpoint, device/session management calls, and the `IGoogleSignInService`/
  `IAppleSignInService` abstractions themselves (unchanged from Phase 8).

**Not implemented / explicitly out of scope:**

- Google Sign-In on iOS (Apple Sign-In was prioritized there instead, both
  because it's the more idiomatic iOS mechanism and because Apple requires
  offering Sign in with Apple if any other third-party sign-in is offered
  on iOS). Adding it later needs its own iOS OAuth client + a native
  Google SDK binding for iOS, not yet selected.
- Apple Sign-In on Android/other platforms - not supported by Apple at all
  outside iOS/macOS/web, out of scope by definition.
- Live end-to-end verification of either provider - this requires a real
  Google Cloud OAuth client, a real Apple Developer account/entitlement,
  and (for Apple) a Mac - none of which exist in this environment. The
  code paths that only run once those exist (the actual native picker UI,
  the actual token returned) were **not exercised**; only compilation and
  the deterministic Core-layer logic around them were verified. See the
  final report's "Production readiness" section for the explicit
  YES/NO status.

## 9. Production deployment checklist

- [ ] `JWT_SECRET_KEY` — already required pre-Phase-8; unchanged
- [ ] `GOOGLE_OAUTH_CLIENT_ID` (backend) **and**
      `GoogleAuthConfiguration.ServerClientId` (mobile) set to the same
      Google Cloud OAuth Web-application client id
- [ ] A Google Cloud OAuth **Android** client registered (package name +
      signing-certificate SHA-1) so Credential Manager is authorized
- [ ] `APPLE_SERVICES_ID` (backend) set to match the Apple Developer
      Services ID/App ID
- [ ] "Sign In with Apple" capability enabled for the App ID in the Apple
      Developer portal, matching `Platforms/iOS/Entitlements.plist`
- [ ] `OTP_SMTP_HOST`/`OTP_SMTP_PORT`/`OTP_SMTP_USERNAME`/`OTP_SMTP_PASSWORD`/
      `OTP_SMTP_FROM_ADDRESS` for real email OTP delivery, OR a new
      SMS-specific `IOtpDeliveryService` implementation if SMS is required
- [ ] `OTP_DEV_LOG_CODES` unset/`false`
- [ ] `alembic upgrade head` run against the production database
