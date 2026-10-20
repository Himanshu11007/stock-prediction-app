#!/usr/bin/env bash
# deploy/vps/run-job.sh JOB - run one StockLens job from cron on the VPS.
#
#   ranking            the daily ranking, run here (needs DATABASE_URL)
#   TODAY_PREOPEN | TOMORROW_EOD | OUTCOME_EVALUATION | SNAPSHOT_MONITOR | NEWS_INGESTION
#                      triggered on the API through scripts/scheduler_client.py
#                      (needs STOCKLENS_API_URL and STOCKLENS_SCHEDULER_TOKEN)
#   update             fast-forward the code from GitHub main; reinstall packages
#                      only when requirements-cloud.txt changed
#   check              setup check: which settings are present (never their values),
#                      Python version, API reachable, server time zone
#
# One run per job at a time (flock: while a run still holds the lock, the next
# trigger is skipped); output appended to /var/log/stocklens/JOB.log. Secrets
# come from the root-owned env file, read literally, and are never printed.
# Every job is safe to repeat: the backend's slot ledger and the ranking's own
# guard make duplicates no-ops, so GitHub Actions can keep the same schedules
# as a backup.
set -uo pipefail

JOB="${1:?usage: run-job.sh JOB}"
APP="${STOCKLENS_APP:-/opt/stocklens/app}"
VENV="${STOCKLENS_VENV:-/opt/stocklens/venv}"
ENV_FILE="${STOCKLENS_ENV_FILE:-/etc/stocklens/stocklens.env}"
LOG_DIR="${STOCKLENS_LOG_DIR:-/var/log/stocklens}"
LOCK="${STOCKLENS_LOCK_DIR:-/run/lock}/stocklens-$JOB.lock"
BUSY=75                                   # exit code when the previous run still holds the lock

[ "$JOB" = check ] || exec >>"$LOG_DIR/$JOB.log" 2>&1
echo "=== $(date '+%F %T %Z') start $JOB"
if [ ! -r "$ENV_FILE" ]; then
  echo "=== missing or unreadable $ENV_FILE"; exit 2
fi
# KEY=value lines, read literally (no shell expansion: a password may contain & $ ;)
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  key="${line%%=*}"
  [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] && export "$key=${line#*=}"
done < "$ENV_FILE"
unset line key
cd "$APP" || exit 2

case "$JOB" in
  ranking)
    timeout 115m flock -n -E "$BUSY" "$LOCK" "$VENV/bin/python" scripts/scheduled_jobs.py ranking ;;
  TODAY_PREOPEN|TOMORROW_EOD|OUTCOME_EVALUATION|SNAPSHOT_MONITOR|NEWS_INGESTION)
    flock -n -E "$BUSY" "$LOCK" "$VENV/bin/python" scripts/scheduler_client.py --job "$JOB" --timeout-minutes 40 ;;
  update)
    before="$(sha256sum requirements-cloud.txt | cut -d' ' -f1)"
    flock -n -E "$BUSY" "$LOCK" bash -c "git fetch -q origin main && git merge -q --ff-only origin/main"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      after="$(sha256sum requirements-cloud.txt | cut -d' ' -f1)"
      if [ "$before" != "$after" ]; then
        echo "requirements changed: reinstalling"
        /usr/local/bin/uv pip install -q --python "$VENV/bin/python" -r requirements-cloud.txt || rc=1
      fi
      git log -1 --format='now at %h %s'
    fi
    (exit "$rc") ;;
  check)
    for k in STOCKLENS_API_URL STOCKLENS_SCHEDULER_TOKEN DATABASE_URL; do
      if [ -n "${!k:-}" ]; then echo "$k: set"; else echo "$k: MISSING"; fi
    done
    case "${DATABASE_URL:-}" in postgres://*|postgresql://*|"") ;; *) echo "DATABASE_URL: does not start with postgresql://";; esac
    "$VENV/bin/python" --version
    echo "server time: $(date '+%F %T %Z')"
    if [ -n "${STOCKLENS_API_URL:-}" ]; then
      echo "API health: $(curl -s -m 90 -o /dev/null -w '%{http_code}' "${STOCKLENS_API_URL%/}/api/v1/health")"
    fi
    (exit 0) ;;
  *)
    echo "=== unknown job $JOB"; exit 2 ;;
esac
code=$?
if [ "$code" -eq "$BUSY" ]; then
  echo "(previous $JOB still running: this trigger skipped)"
  code=0
fi
echo "=== $(date '+%F %T %Z') end $JOB exit $code"
exit "$code"
