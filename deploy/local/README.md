# StockLens on a Windows laptop

Runs everything on this laptop instead of Render and the IONOS server:

| Part | Where | Address |
|---|---|---|
| Database | PostgreSQL 16 in Docker (container `stocklens-db`, volume `stocklens-pgdata`) | 127.0.0.1:5432, this laptop only |
| API | Python 3.11, task "StockLens API" (starts at log-on, restarts itself) | http://localhost:8000 |
| Website | IIS site "StockLens" | http://localhost:8080 |
| Daily jobs | Windows Task Scheduler, same India times as `deploy/vps/stocklens.cron` | logs in `%LOCALAPPDATA%\StockLens\logs` |

The jobs only run while the laptop is on. A job missed while it was off or
asleep runs as soon as it is back (repeats are harmless). Docker Desktop must
start at log-on (Docker Desktop > Settings > General > "Start Docker Desktop
when you sign in").

## One-time setup

In a normal PowerShell window:

```powershell
git clone https://github.com/Himanshu11007/stock-prediction-app.git $env:LOCALAPPDATA\StockLens\app
cd $env:LOCALAPPDATA\StockLens\app
powershell -ExecutionPolicy Bypass -File deploy\local\setup.ps1          # Python, database, settings
powershell -ExecutionPolicy Bypass -File deploy\local\copy-data.ps1      # copy all data from Render
powershell -ExecutionPolicy Bypass -File deploy\local\set-secret.ps1 DRISHTI_API_KEY
powershell -ExecutionPolicy Bypass -File deploy\local\install-tasks.ps1  # API + daily jobs
```

Then in PowerShell opened with **Run as administrator** (same folder):

```powershell
powershell -ExecutionPolicy Bypass -File deploy\local\setup-iis.ps1
```

and once more in the normal window:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\local\publish-web.ps1
```

Open http://localhost:8080. Sign in with email and password. For Google
sign-in, add `http://localhost:8080` to the "Authorised JavaScript origins" of
the web client in Google Cloud Console (APIs & Services > Credentials).

## Everyday

- Code updates arrive by themselves every morning at 05:30 (task "StockLens update").
- After website changes: run `publish-web.ps1` again.
- Run a job now: `powershell -ExecutionPolicy Bypass -File deploy\local\run-job.ps1 TODAY_PREOPEN`
- Change a setting or key: `set-secret.ps1 NAME` (the API restarts itself).
- Take a new copy from Render (replaces local data): `copy-data.ps1`.
- Remove the tasks: `install-tasks.ps1 -Remove`.

Settings and secrets live in `%LOCALAPPDATA%\StockLens\stocklens.env`
(readable only by your Windows user). Never share or commit it.
