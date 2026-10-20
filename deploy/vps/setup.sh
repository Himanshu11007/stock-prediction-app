#!/usr/bin/env bash
# deploy/vps/setup.sh - one-time setup of the StockLens scheduler VPS
# (Ubuntu 24.04, run as root from the cloned repository):
#
#   apt-get update && apt-get install -y git
#   git clone https://github.com/Himanshu11007/stock-prediction-app.git /opt/stocklens/app
#   bash /opt/stocklens/app/deploy/vps/setup.sh
#
# Idempotent: safe to run again. It sets the time zone to IST, enables the
# firewall (SSH only) and automatic security updates, adds a 2 GB swap file,
# creates the unprivileged 'stocklens' user, installs Python 3.11 (same as
# Render / CI) with uv, installs requirements-cloud.txt, creates the env file
# from the example (root:stocklens 0640) and installs the cron schedule and
# log rotation. It never writes a secret.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo -i)"; exit 1; }
APP=/opt/stocklens/app
HOME_DIR=/opt/stocklens
VENV=$HOME_DIR/venv
[ -f "$APP/requirements-cloud.txt" ] || { echo "clone the repository to $APP first"; exit 1; }

echo "== time zone, packages, firewall, automatic updates"
timedatectl set-timezone Asia/Kolkata
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q git curl ufw unattended-upgrades python3
ufw allow OpenSSH >/dev/null
ufw --force enable >/dev/null
dpkg-reconfigure -f noninteractive unattended-upgrades >/dev/null || true

echo "== swap (the ranking briefly needs more than 2 GB RAM)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== user and directories"
id stocklens >/dev/null 2>&1 || useradd --system --home-dir "$HOME_DIR" --shell /usr/sbin/nologin stocklens
mkdir -p /var/log/stocklens /etc/stocklens
chown -R stocklens:stocklens "$HOME_DIR" /var/log/stocklens
chmod +x "$APP"/deploy/vps/*.sh

echo "== Python 3.11 and packages (a few minutes)"
[ -x /usr/local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
# run uv as 'stocklens' from a directory it can read (uv looks for uv.toml in
# the working directory; /root is not readable) and ignore config files
cd "$HOME_DIR"
sudo -u stocklens -H env HOME="$HOME_DIR" /usr/local/bin/uv venv -q --no-config --python 3.11 "$VENV" --allow-existing
sudo -u stocklens -H env HOME="$HOME_DIR" /usr/local/bin/uv pip install -q --no-config --python "$VENV/bin/python" -r "$APP/requirements-cloud.txt"

echo "== settings file (secrets are added by you, never by this script)"
if [ ! -f /etc/stocklens/stocklens.env ]; then
  install -m 0640 -o root -g stocklens "$APP/deploy/vps/stocklens.env.example" /etc/stocklens/stocklens.env
fi
chown root:stocklens /etc/stocklens/stocklens.env
chmod 0640 /etc/stocklens/stocklens.env

echo "== schedule and log rotation"
install -m 0644 "$APP/deploy/vps/stocklens.cron" /etc/cron.d/stocklens
install -m 0644 "$APP/deploy/vps/stocklens.logrotate" /etc/logrotate.d/stocklens
systemctl restart cron

echo
echo "Done. Server time: $(date '+%F %T %Z')"
echo "Next: sudo bash $APP/deploy/vps/make-token.sh, then add DATABASE_URL with:"
echo "      sudo nano /etc/stocklens/stocklens.env"
