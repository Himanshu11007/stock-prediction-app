"""api/routes/admin.py — admin/master panel: users, roles, stock master,
read-only recommendations/watchlist visibility, and the audit log.

Every route requires the ADMIN role (auth.dependencies.require_admin).
Recommendations and watchlist are exposed read-only only - nothing here can
alter an ML prediction or a user's watchlist entry. Every mutation writes an
admin_audit_logs row (admin/audit.py) in the same transaction.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

import admin.service as admin_service
import auth.service as auth_service
from api.schemas_admin import (
    AdminAuditLogResponse,
    AdminDashboardResponse,
    AdminRecommendationResponse,
    AdminStockResponse,
    AdminStockUpdateRequest,
    AdminUserResponse,
    AdminWatchlistItemResponse,
    AssignRoleRequest,
)
from auth.dependencies import require_admin
from db.models.user import User
from db.session import get_session

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


def _user_response(session: Session, user: User) -> AdminUserResponse:
    return AdminUserResponse(
        id=user.id,
        email=user.email,
        is_active=user.is_active,
        roles=auth_service.get_user_roles(session, user),
        created_at=user.created_at,
    )


def _get_user_or_404(session: Session, user_id: int) -> User:
    user = admin_service.get_user(session, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


def _get_stock_or_404(session: Session, symbol: str):
    company = admin_service.get_stock(session, symbol)
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stock not found")
    return company


# ── Dashboard ────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_model=AdminDashboardResponse)
def get_dashboard(session: Session = Depends(get_session)):
    return admin_service.get_dashboard_stats(session)


# ── Users ────────────────────────────────────────────────────────────────────

@router.get("/users", response_model=list[AdminUserResponse])
def list_users(limit: int = 50, offset: int = 0, session: Session = Depends(get_session)):
    users = admin_service.list_users(session, limit=limit, offset=offset)
    return [_user_response(session, u) for u in users]


@router.get("/users/{user_id}", response_model=AdminUserResponse)
def get_user(user_id: int, session: Session = Depends(get_session)):
    user = _get_user_or_404(session, user_id)
    return _user_response(session, user)


@router.post("/users/{user_id}/activate", response_model=AdminUserResponse)
def activate_user(
    user_id: int,
    current_admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    user = _get_user_or_404(session, user_id)
    user = admin_service.set_user_active(session, current_admin, user, True)
    return _user_response(session, user)


@router.post("/users/{user_id}/deactivate", response_model=AdminUserResponse)
def deactivate_user(
    user_id: int,
    current_admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    user = _get_user_or_404(session, user_id)
    try:
        user = admin_service.set_user_active(session, current_admin, user, False)
    except admin_service.SelfActionNotAllowedError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except admin_service.LastAdminError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return _user_response(session, user)


@router.post("/users/{user_id}/roles", response_model=AdminUserResponse)
def assign_role(
    user_id: int,
    payload: AssignRoleRequest,
    current_admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    user = _get_user_or_404(session, user_id)
    try:
        admin_service.assign_role_to_user(session, current_admin, user, payload.role)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return _user_response(session, user)


@router.delete("/users/{user_id}/roles/{role_name}", response_model=AdminUserResponse)
def remove_role(
    user_id: int,
    role_name: str,
    current_admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    user = _get_user_or_404(session, user_id)
    try:
        admin_service.remove_role_from_user(session, current_admin, user, role_name)
    except (ValueError, admin_service.SelfActionNotAllowedError, admin_service.LastAdminError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return _user_response(session, user)


# ── Stock master ─────────────────────────────────────────────────────────────

@router.get("/stocks", response_model=list[AdminStockResponse])
def list_stocks(
    search: str | None = None, limit: int = 50, offset: int = 0, session: Session = Depends(get_session)
):
    return admin_service.list_stocks(session, search=search, limit=limit, offset=offset)


@router.get("/stocks/{symbol}", response_model=AdminStockResponse)
def get_stock(symbol: str, session: Session = Depends(get_session)):
    return _get_stock_or_404(session, symbol)


@router.patch("/stocks/{symbol}", response_model=AdminStockResponse)
def update_stock(
    symbol: str,
    payload: AdminStockUpdateRequest,
    current_admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    company = _get_stock_or_404(session, symbol)
    return admin_service.update_stock_flags(
        session, current_admin, company,
        active=payload.active, analysis_enabled=payload.analysis_enabled,
    )


# ── Recommendations (read-only) ─────────────────────────────────────────────

@router.get("/recommendations", response_model=list[AdminRecommendationResponse])
def list_recommendations(
    symbol: str | None = None, limit: int = 50, offset: int = 0, session: Session = Depends(get_session)
):
    return admin_service.list_recommendations(session, symbol=symbol, limit=limit, offset=offset)


# ── Watchlist (read-only) ───────────────────────────────────────────────────

@router.get("/watchlist", response_model=list[AdminWatchlistItemResponse])
def list_watchlist(limit: int = 50, offset: int = 0, session: Session = Depends(get_session)):
    return admin_service.list_watchlist(session, limit=limit, offset=offset)


# ── Audit logs (read-only) ──────────────────────────────────────────────────

@router.get("/audit-logs", response_model=list[AdminAuditLogResponse])
def list_audit_logs(limit: int = 50, offset: int = 0, session: Session = Depends(get_session)):
    return admin_service.list_audit_logs(session, limit=limit, offset=offset)
