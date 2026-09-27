"""Admin/master panel business logic - read access across the existing
domains (users, stock master, recommendations, watchlist) plus the limited
set of admin mutations the original spec calls out (activate/deactivate a
user, assign/remove a role, enable/disable a stock or its analysis flag).

Every mutation here writes an admin_audit_logs entry (admin/audit.py) in the
same transaction. Recommendations and watchlist are exposed read-only only -
nothing here can alter an ML prediction or a user's watchlist, per the
architecture rule that the admin panel must not be able to arbitrarily
change generated recommendations.
"""
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, func, select

import auth.service as auth_service
from admin.audit import log_action
from db.models.admin import AdminAuditLog
from db.models.stock import Company
from db.models.tracker import Recommendation, WatchlistItem
from db.models.user import Role, User, UserRoleLink


# ══════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════

def get_dashboard_stats(session: Session) -> dict:
    """Real counts only - no premium/subscription figures, since that data
    doesn't exist yet (see Phase 5 scope note: deferred to a later phase)."""

    def count(stmt) -> int:
        return session.exec(select(func.count()).select_from(stmt.subquery())).one()

    total_users = count(select(User))
    active_users = count(select(User).where(User.is_active == True))  # noqa: E712
    admin_users = count(
        select(User)
        .join(UserRoleLink, UserRoleLink.user_id == User.id)
        .join(Role, Role.id == UserRoleLink.role_id)
        .where(Role.name == auth_service.ADMIN_ROLE)
    )
    total_stocks = count(select(Company))
    enabled_stocks = count(select(Company).where(Company.active == True))  # noqa: E712
    analysis_enabled_stocks = count(select(Company).where(Company.analysis_enabled == True))  # noqa: E712
    total_recommendations = count(select(Recommendation))
    total_watchlist_items = count(select(WatchlistItem))

    return {
        "total_users": total_users,
        "active_users": active_users,
        "admin_users": admin_users,
        "total_stocks": total_stocks,
        "enabled_stocks": enabled_stocks,
        "analysis_enabled_stocks": analysis_enabled_stocks,
        "total_recommendations": total_recommendations,
        "total_watchlist_items": total_watchlist_items,
    }


# ══════════════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════════════

def list_users(session: Session, limit: int = 50, offset: int = 0) -> list[User]:
    stmt = select(User).order_by(User.id).offset(offset).limit(limit)
    return list(session.exec(stmt).all())


def get_user(session: Session, user_id: int) -> Optional[User]:
    return session.get(User, user_id)


def set_user_active(session: Session, admin_user: User, target_user: User, is_active: bool) -> User:
    target_user.is_active = is_active
    target_user.updated_at = datetime.now(timezone.utc)
    session.add(target_user)
    session.commit()
    session.refresh(target_user)
    log_action(
        session,
        admin_user,
        action="activate_user" if is_active else "deactivate_user",
        entity="user",
        entity_id=str(target_user.id),
        extra_data={"email": target_user.email},
    )
    return target_user


def assign_role_to_user(session: Session, admin_user: User, target_user: User, role_name: str) -> None:
    auth_service.assign_role(session, target_user, role_name)
    log_action(
        session, admin_user, action="assign_role", entity="user",
        entity_id=str(target_user.id), extra_data={"role": role_name},
    )


def remove_role_from_user(session: Session, admin_user: User, target_user: User, role_name: str) -> bool:
    removed = auth_service.remove_role(session, target_user, role_name)
    if removed:
        log_action(
            session, admin_user, action="remove_role", entity="user",
            entity_id=str(target_user.id), extra_data={"role": role_name},
        )
    return removed


# ══════════════════════════════════════════════════════════════════════════
# Stock master
# ══════════════════════════════════════════════════════════════════════════

def list_stocks(
    session: Session, search: Optional[str] = None, limit: int = 50, offset: int = 0
) -> list[Company]:
    stmt = select(Company)
    if search:
        pattern = f"%{search.strip().upper()}%"
        stmt = stmt.where(func.upper(Company.symbol).like(pattern) | func.upper(Company.name).like(pattern))
    stmt = stmt.order_by(Company.symbol).offset(offset).limit(limit)
    return list(session.exec(stmt).all())


def get_stock(session: Session, symbol: str) -> Optional[Company]:
    return session.get(Company, symbol)


def update_stock_flags(
    session: Session,
    admin_user: User,
    company: Company,
    active: Optional[bool] = None,
    analysis_enabled: Optional[bool] = None,
) -> Company:
    changes = {}
    if active is not None and active != company.active:
        company.active = active
        changes["active"] = active
    if analysis_enabled is not None and analysis_enabled != company.analysis_enabled:
        company.analysis_enabled = analysis_enabled
        changes["analysis_enabled"] = analysis_enabled

    if changes:
        company.updated_at = datetime.now(timezone.utc)
        session.add(company)
        session.commit()
        session.refresh(company)
        log_action(
            session, admin_user, action="update_stock", entity="company",
            entity_id=company.symbol, extra_data=changes,
        )
    return company


# ══════════════════════════════════════════════════════════════════════════
# Recommendations / watchlist - read-only
# ══════════════════════════════════════════════════════════════════════════

def list_recommendations(
    session: Session, symbol: Optional[str] = None, limit: int = 50, offset: int = 0
) -> list[Recommendation]:
    stmt = select(Recommendation)
    if symbol:
        stmt = stmt.where(Recommendation.symbol == symbol.strip().upper())
    stmt = stmt.order_by(Recommendation.id.desc()).offset(offset).limit(limit)
    return list(session.exec(stmt).all())


def list_watchlist(session: Session, limit: int = 50, offset: int = 0) -> list[WatchlistItem]:
    stmt = select(WatchlistItem).order_by(WatchlistItem.id).offset(offset).limit(limit)
    return list(session.exec(stmt).all())


# ══════════════════════════════════════════════════════════════════════════
# Audit logs - read-only
# ══════════════════════════════════════════════════════════════════════════

def list_audit_logs(session: Session, limit: int = 50, offset: int = 0) -> list[AdminAuditLog]:
    stmt = select(AdminAuditLog).order_by(AdminAuditLog.id.desc()).offset(offset).limit(limit)
    return list(session.exec(stmt).all())
