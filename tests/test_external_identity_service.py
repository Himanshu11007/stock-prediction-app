"""tests/test_external_identity_service.py — Google/Apple identity
resolution and account-linking logic at the service layer (auth/external_identity.py),
independent of the HTTP routes. Uses an isolated in-memory SQLite engine.
"""
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
from auth.external_identity import (
    AccountLinkingRequiredError,
    DuplicateExternalIdentityError,
    ExternalIdentityError,
    LastAuthMethodError,
    VerifiedExternalIdentity,
    find_or_create_user_for_identity,
    link_identity,
    list_identities,
    unlink_identity,
)
from db.models.user import ExternalIdentity, User  # noqa: F401


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    with Session(engine) as s:
        auth_service.ensure_roles_exist(s)
        yield s


def google_identity(subject="google-sub-1", email="new@example.com", verified=True):
    return VerifiedExternalIdentity(provider="google", subject=subject, email=email, email_verified=verified)


def apple_identity(subject="apple-sub-1", email=None, verified=True):
    return VerifiedExternalIdentity(provider="apple", subject=subject, email=email, email_verified=verified)


# ══════════════════════════════════════════════════════════════════════════
# find_or_create_user_for_identity
# ══════════════════════════════════════════════════════════════════════════

def test_new_google_identity_creates_a_new_user(session):
    user, is_new = find_or_create_user_for_identity(session, google_identity())
    assert is_new is True
    assert user.email == "new@example.com"
    assert user.hashed_password is None
    assert auth_service.get_user_roles(session, user) == ["USER"]


def test_same_google_subject_logs_in_the_existing_user_again(session):
    user1, is_new1 = find_or_create_user_for_identity(session, google_identity())
    user2, is_new2 = find_or_create_user_for_identity(session, google_identity())
    assert is_new1 is True
    assert is_new2 is False
    assert user1.id == user2.id


def test_apple_identity_with_no_email_still_creates_a_user(session):
    # Apple only sends email on first auth (or with private relay) - must
    # work even when it's None.
    user, is_new = find_or_create_user_for_identity(session, apple_identity(email=None))
    assert is_new is True
    assert user.email is None


def test_apple_subsequent_login_with_no_email_still_resolves_to_same_user(session):
    user1, _ = find_or_create_user_for_identity(session, apple_identity(subject="apple-x", email="first@example.com"))
    # Second login: Apple omits email this time - must still match by subject.
    user2, is_new = find_or_create_user_for_identity(session, apple_identity(subject="apple-x", email=None))
    assert is_new is False
    assert user1.id == user2.id


def test_email_collision_with_existing_account_requires_explicit_linking(session):
    auth_service.ensure_roles_exist(session)
    auth_service.create_user(session, "shared@example.com", "Password123!")

    with pytest.raises(AccountLinkingRequiredError) as exc_info:
        find_or_create_user_for_identity(session, google_identity(email="shared@example.com"))
    assert exc_info.value.existing_email == "shared@example.com"

    # Must NOT have silently created or attached anything.
    assert session.exec(select(ExternalIdentity)).first() is None


def test_deactivated_user_behind_existing_identity_is_rejected(session):
    user, _ = find_or_create_user_for_identity(session, google_identity())
    user.is_active = False
    session.add(user)
    session.commit()

    with pytest.raises(ExternalIdentityError):
        find_or_create_user_for_identity(session, google_identity())


# ══════════════════════════════════════════════════════════════════════════
# link_identity / unlink_identity
# ══════════════════════════════════════════════════════════════════════════

def test_link_identity_to_authenticated_account(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "linker@example.com", "Password123!")

    link = link_identity(session, user, google_identity(subject="g-1"))

    assert link.provider == "google"
    identities = list_identities(session, user)
    assert len(identities) == 1
    assert identities[0].provider_subject == "g-1"


def test_linking_same_identity_twice_to_same_account_is_idempotent(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "linker2@example.com", "Password123!")

    link1 = link_identity(session, user, google_identity(subject="g-2"))
    link2 = link_identity(session, user, google_identity(subject="g-2"))

    assert link1.id == link2.id
    assert len(list_identities(session, user)) == 1


def test_linking_identity_already_owned_by_another_account_is_rejected(session):
    auth_service.ensure_roles_exist(session)
    user_a = auth_service.create_user(session, "a@example.com", "Password123!")
    user_b = auth_service.create_user(session, "b@example.com", "Password123!")
    link_identity(session, user_a, google_identity(subject="shared-google-sub"))

    with pytest.raises(DuplicateExternalIdentityError):
        link_identity(session, user_b, google_identity(subject="shared-google-sub"))


def test_linking_second_identity_for_same_provider_is_rejected(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "c@example.com", "Password123!")
    link_identity(session, user, google_identity(subject="g-first"))

    with pytest.raises(DuplicateExternalIdentityError):
        link_identity(session, user, google_identity(subject="g-second"))


def test_unlink_identity_removes_it(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "d@example.com", "Password123!")
    link_identity(session, user, google_identity(subject="g-d"))

    unlink_identity(session, user, "google")

    assert list_identities(session, user) == []


def test_unlink_unknown_provider_is_a_silent_no_op(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "e@example.com", "Password123!")
    unlink_identity(session, user, "google")  # never linked - must not raise


def test_cannot_unlink_the_only_sign_in_method(session):
    # User created via Google only (no password) with a single linked identity.
    user, _ = find_or_create_user_for_identity(session, google_identity(subject="only-method"))

    with pytest.raises(LastAuthMethodError):
        unlink_identity(session, user, "google")

    # Must still be linked after the rejected attempt.
    assert len(list_identities(session, user)) == 1


def test_can_unlink_one_provider_if_a_password_still_exists(session):
    auth_service.ensure_roles_exist(session)
    user = auth_service.create_user(session, "f@example.com", "Password123!")
    link_identity(session, user, google_identity(subject="g-f"))

    unlink_identity(session, user, "google")  # password remains - allowed

    assert list_identities(session, user) == []
