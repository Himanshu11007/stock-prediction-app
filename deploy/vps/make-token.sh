#!/usr/bin/env bash
# deploy/vps/make-token.sh - create this server's scheduler token (run as root).
# Writes a new random token into /etc/stocklens/stocklens.env and prints ONLY
# its SHA-256 digest, which goes into Render's SCHEDULER_TOKEN_SHA256 (append
# it after the existing digest with a comma). The token itself is never shown.
set -euo pipefail
ENV_FILE=/etc/stocklens/stocklens.env
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
[ -f "$ENV_FILE" ] || { echo "$ENV_FILE is missing: run setup.sh first"; exit 1; }
token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
digest="$(printf '%s' "$token" | sha256sum | cut -d' ' -f1)"
tmp="$(mktemp)"
grep -v '^STOCKLENS_SCHEDULER_TOKEN=' "$ENV_FILE" > "$tmp" || true
printf 'STOCKLENS_SCHEDULER_TOKEN=%s\n' "$token" >> "$tmp"
install -m 0640 -o root -g stocklens "$tmp" "$ENV_FILE"
rm -f "$tmp"
unset token
echo "New scheduler token saved in $ENV_FILE (not shown)."
echo "Add this digest to Render -> stocklens-api -> Environment -> SCHEDULER_TOKEN_SHA256,"
echo "after the existing value, separated by a comma:"
echo
echo "    $digest"
