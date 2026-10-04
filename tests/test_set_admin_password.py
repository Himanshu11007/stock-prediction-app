"""scripts/set_admin_password.py — operator password reset for admins only."""
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import auth.service as auth_service
from scripts.set_admin_password import set_admin_password


@pytest.fixture()
def s():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        auth_service.ensure_roles_exist(session)
        yield session


def test_sets_password_for_admin_without_one(s):
    user = auth_service.create_external_user(s, email="admin@example.com")
    auth_service.assign_role(s, user, auth_service.ADMIN_ROLE)
    set_admin_password(s, "admin@example.com", "NewPass123!")
    assert auth_service.authenticate_user(s, "admin@example.com", "NewPass123!").email == "admin@example.com"


def test_refuses_non_admins_unknown_accounts_and_short_passwords(s):
    auth_service.create_user(s, "user@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])
    with pytest.raises(SystemExit, match="not an admin"):
        set_admin_password(s, "user@example.com", "NewPass123!")
    with pytest.raises(SystemExit, match="No account"):
        set_admin_password(s, "nobody@example.com", "NewPass123!")
    auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
    with pytest.raises(SystemExit, match="at least 8"):
        set_admin_password(s, "admin@example.com", "short")
    assert auth_service.authenticate_user(s, "user@example.com", "UserPass1!")
