"""One-off setup: create the initial ADMIN account.

The password is NEVER hardcoded here. It must be supplied via the
ADMIN_INITIAL_PASSWORD environment variable, or interactively (hidden input)
if that env var is not set and the script is run in a terminal.

Idempotent: if the account already exists, this only ensures it has the
ADMIN role - it never overwrites an existing password.

Usage:
    ADMIN_INITIAL_PASSWORD=... venv/Scripts/python.exe -m scripts.seed_admin [email]

    (or, without the env var, run interactively and you'll be prompted)
"""
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session

import auth.service as auth_service
from db.session import engine

DEFAULT_ADMIN_EMAIL = "goswamihimanshu15@gmail.com"


def _get_password() -> str:
    password = os.environ.get("ADMIN_INITIAL_PASSWORD")
    if password:
        return password
    if not sys.stdin.isatty():
        raise SystemExit(
            "ADMIN_INITIAL_PASSWORD is not set and no terminal is attached to "
            "prompt for one. Set the environment variable and re-run."
        )
    password = getpass.getpass("Initial admin password (not echoed): ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        raise SystemExit("Passwords did not match.")
    return password


def seed_admin(email: str) -> None:
    with Session(engine) as session:
        auth_service.ensure_roles_exist(session)

        existing = auth_service.get_user_by_email(session, email)
        if existing is not None:
            auth_service.assign_role(session, existing, auth_service.ADMIN_ROLE)
            print(f"Admin account already existed for {email} - ensured ADMIN role, "
                  f"password left unchanged.")
            return

        password = _get_password()
        user = auth_service.create_user(
            session, email, password, roles=[auth_service.ADMIN_ROLE]
        )
        print(f"Created ADMIN account: {user.email} (id={user.id})")


if __name__ == "__main__":
    admin_email = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ADMIN_EMAIL
    seed_admin(admin_email)
