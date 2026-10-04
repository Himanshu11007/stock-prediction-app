"""Operator tool: set a new password for an existing ADMIN account.

For when the administrator cannot sign in to the admin console: the password
was forgotten, or the account was created by Google sign-in and has no
password. Run it from a terminal against the target database; the password
is typed with hidden input (or ADMIN_NEW_PASSWORD) and never printed.

Only accounts that already have the ADMIN role are changed, and existing
sessions of the account are revoked.

Usage:
    $env:DATABASE_URL = "<database url>"
    venv\\Scripts\\python.exe -m scripts.set_admin_password admin@example.com
"""
import getpass
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session  # noqa: E402

import auth.service as auth_service  # noqa: E402
from auth.device_service import revoke_all_sessions  # noqa: E402
from auth.security import hash_password  # noqa: E402
from db.session import engine  # noqa: E402


def _new_password() -> str:
    password = os.environ.get("ADMIN_NEW_PASSWORD")
    if password:
        return password
    if not sys.stdin.isatty():
        raise SystemExit("ADMIN_NEW_PASSWORD is not set and no terminal is attached to prompt for one.")
    password = getpass.getpass("New admin password (not echoed): ")
    if password != getpass.getpass("Confirm password: "):
        raise SystemExit("Passwords did not match.")
    return password


def set_admin_password(session: Session, email: str, password: str) -> None:
    user = auth_service.get_user_by_email(session, email)
    if user is None:
        raise SystemExit(f"No account for {email}. Create one with: python -m scripts.seed_admin {email}")
    if auth_service.ADMIN_ROLE not in auth_service.get_user_roles(session, user):
        raise SystemExit(f"{email} is not an admin; refusing to change its password.")
    if len(password) < 8:
        raise SystemExit("Password must be at least 8 characters.")
    user.hashed_password = hash_password(password)
    user.is_active = True
    user.updated_at = datetime.now(timezone.utc)
    session.add(user)
    session.commit()
    revoke_all_sessions(session, user)
    session.commit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.set_admin_password <admin email>")
    target = sys.argv[1]
    with Session(engine) as s:
        set_admin_password(s, target, _new_password())
    print(f"Password updated for {target}. Sign in to the admin console with it.")
