from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List
from database.connection import get_db
from database.models import Notification
from auth.security import require_admin, get_current_user
from pydantic import BaseModel
from datetime import datetime
from database.models import User

router = APIRouter(prefix="/admin/notifications", tags=["notifications"])


class NotificationOut(BaseModel):
    id: int
    type: str
    title: str
    message: str
    is_read: bool
    related_user_id: int | None
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/", response_model=List[NotificationOut])
def get_notifications(
    skip: int = 0,
    limit: int = 50,
    unread_only: bool = False,
    db: Session = Depends(get_db),
    _=Depends(require_admin)
):
    """Retourne les notifications (les plus récentes en premier)."""
    q = db.query(Notification).order_by(desc(Notification.created_at))
    if unread_only:
        q = q.filter(Notification.is_read == False)
    return q.offset(skip).limit(limit).all()


@router.get("/unread-count")
def get_unread_count(
    db: Session = Depends(get_db),
    _=Depends(require_admin)
):
    """Utilisé par le polling frontend (toutes les 30s)."""
    count = db.query(Notification).filter(Notification.is_read == False).count()
    return {"count": count}



@router.put("/read-all")
def mark_all_read(
    db: Session = Depends(get_db),
    _=Depends(require_admin)
):
    db.query(Notification).filter(Notification.is_read == False).update({"is_read": True})
    db.commit()
    return {"ok": True}



@router.put("/{notif_id}/read")
def mark_as_read(
    notif_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_admin)
):
    notif = db.query(Notification).filter(Notification.id == notif_id).first()
    if notif:
        notif.is_read = True
        db.commit()
    return {"ok": True}


@router.get("/me", response_model=List[NotificationOut])
def get_my_notifications(
    skip: int = 0,
    limit: int = 50,
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    q = (
        db.query(Notification)
        .filter(Notification.related_user_id == current_user.id)
        .order_by(desc(Notification.created_at))
    )
    if unread_only:
        q = q.filter(Notification.is_read == False)
    return q.offset(skip).limit(limit).all()


@router.get("/me/unread-count")
def get_my_unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    count = (
        db.query(Notification)
        .filter(
            Notification.related_user_id == current_user.id,
            Notification.is_read == False
        )
        .count()
    )
    return {"count": count}


@router.put("/me/read-all")
def mark_all_my_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db.query(Notification).filter(
        Notification.related_user_id == current_user.id,
        Notification.is_read == False
    ).update({"is_read": True})
    db.commit()
    return {"ok": True}


@router.put("/me/{notif_id}/read")
def mark_my_notification_read(
    notif_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    notif = db.query(Notification).filter(
        Notification.id == notif_id,
        Notification.related_user_id == current_user.id
    ).first()
    if notif:
        notif.is_read = True
        db.commit()
    return {"ok": True}