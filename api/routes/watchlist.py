"""api/routes/watchlist.py — authenticated user's own watchlist.

Every route requires a valid access token (auth.dependencies.get_current_user).
Ownership is enforced by watchlist/service.py using the authenticated user's
id from the token - no route or request body ever supplies a user_id.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

import watchlist.service as watchlist_service
import notifications.service as notify
from api.schemas import success_envelope
from api.schemas_watchlist import WatchlistAddRequest, WatchlistAlertUpdate, WatchlistItemResponse
from auth.dependencies import get_current_user
from db.models.user import User
from db.session import get_session

router = APIRouter(prefix="/watchlist", dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[WatchlistItemResponse])
def get_my_watchlist(
    current_user: User = Depends(get_current_user), session: Session = Depends(get_session)
):
    return watchlist_service.list_watchlist(session, current_user)


@router.get("/overview")
def watchlist_overview(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Watched stocks with current StockLens Score, rank, FQVF, last price,
    score change since the previous full run, freshness and alert settings."""
    items = watchlist_service.overview(session, current_user)
    return success_envelope({"items": items},
                            message="Your watchlist is empty." if not items else f"{len(items)} stock(s)")


@router.put("/{item_id}/alerts")
def update_watchlist_alerts(item_id: int, payload: WatchlistAlertUpdate,
                            current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Per-stock alert switches (score, rank, FQVF, status changes; mute)."""
    from db.models.tracker import WatchlistItem
    item = session.get(WatchlistItem, item_id)
    if item is None or item.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watchlist item not found")
    row = notify.update_alert_setting(session, current_user, item.symbol, payload.model_dump(exclude_none=True))
    return success_envelope({"symbol": item.symbol, **notify.alert_payload(row)}, message="Alert settings saved")


@router.post("", response_model=WatchlistItemResponse, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(
    payload: WatchlistAddRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    try:
        return watchlist_service.add_watchlist_item(
            session, current_user, payload.symbol, payload.buy_price, payload.buy_date, payload.quantity
        )
    except watchlist_service.UnknownSymbolError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except watchlist_service.DuplicateWatchlistItemError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_watchlist(
    item_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    removed = watchlist_service.remove_watchlist_item(session, current_user, item_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watchlist item not found")
