from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional, List

from database.connection import get_db
from database.models import (
    FindingMetadata, FindingStatus,
    Notification, NotificationType, User
)
from auth.security import get_current_user

router = APIRouter(prefix="/findings", tags=["findings-metadata"])


# ─── Schémas Pydantic ─────────────────────────────────────────────────────────

class PinRequest(BaseModel):
    finding_title:   str
    product_name:    Optional[str]   = None
    engagement_name: Optional[str]   = None
    severity:        Optional[str]   = None
    ai_risk_score:   Optional[float] = None


class AssignRequest(BaseModel):
    assigned_to_id: int
    finding_title:  str
    product_name:   Optional[str]   = None
    severity:       Optional[str]   = None
    ai_risk_score:  Optional[float] = None


class StatusUpdateRequest(BaseModel):
    status: FindingStatus


# ─── Helper ───────────────────────────────────────────────────────────────────

def get_or_create_metadata(
    finding_id:      int,
    db:              Session,
    finding_title:   str   = None,
    product_name:    str   = None,
    engagement_name: str   = None,
    severity:        str   = None,
    ai_risk_score:   float = None,
) -> FindingMetadata:

    meta = db.query(FindingMetadata).filter_by(finding_id=finding_id).first()

    if not meta:
        meta = FindingMetadata(
            finding_id      = finding_id,
            finding_title   = finding_title,
            product_name    = product_name,
            engagement_name = engagement_name,
            severity        = severity,
            ai_risk_score   = ai_risk_score,
            status          = FindingStatus.open,
        )
        db.add(meta)
        db.flush()

    return meta



@router.get("/users/assignable")
def get_assignable_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retourne tous les utilisateurs pouvant être assignés (dev, analyst, manager)."""
    users = (
        db.query(User)
        .filter(
            User.role.in_(["developer", "analyst", "manager"]),
            User.status == "active",
        )
        .all()
    )
    return [
        {
            "id":         u.id,
            "username":   u.username,
            "role":       u.role,
            "avatar_url": u.avatar_url,
        }
        for u in users
    ]


# ─── ROUTES STATIQUES EN PREMIER (avant /{finding_id}) ───────────────────────

@router.get("/pinned/all")
def get_all_pinned(
    db: Session = Depends(get_db),
    _:  User    = Depends(get_current_user),
):
    pinned = (
        db.query(FindingMetadata)
        .filter_by(is_pinned=True)
        .order_by(FindingMetadata.pinned_at.desc())
        .all()
    )
    return [
        {
            "finding_id":    p.finding_id,
            "finding_title": p.finding_title,
            "severity":      p.severity,
            "ai_risk_score": p.ai_risk_score,
            "pinned_by":     p.pinned_by_user.username if p.pinned_by_user else None,
            "pinned_at":     p.pinned_at,
            "status":        p.status,
        }
        for p in pinned
    ]


@router.get("/my-assignments/all")
def get_my_assignments(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    assignments = (
        db.query(FindingMetadata)
        .filter_by(assigned_to_id=current_user.id)
        .order_by(FindingMetadata.assigned_at.desc())
        .all()
    )
    return [
        {
            "finding_id":    a.finding_id,
            "finding_title": a.finding_title,
            "severity":      a.severity,
            "ai_risk_score": a.ai_risk_score,
            "product_name":  a.product_name,
            "assigned_by":   a.assigned_by_user.username if a.assigned_by_user else None,
            "assigned_at":   a.assigned_at,
            "status":        a.status,
        }
        for a in assignments
    ]


@router.post("/metadata/batch")
def get_metadata_batch(
    finding_ids: List[int],
    db:          Session = Depends(get_db),
    _:           User    = Depends(get_current_user),
):
    metas = (
        db.query(FindingMetadata)
        .filter(FindingMetadata.finding_id.in_(finding_ids))
        .all()
    )

    result = {}
    for m in metas:
        result[m.finding_id] = {
            "finding_id":     m.finding_id,
            "is_pinned":      m.is_pinned,
            "pinned_by":      m.pinned_by_user.username if m.pinned_by_user else None,
            "pinned_at":      m.pinned_at,
            "assigned_to":    m.assigned_to_user.username if m.assigned_to_user else None,
            "assigned_to_id": m.assigned_to_id,
            "assigned_by":    m.assigned_by_user.username if m.assigned_by_user else None,
            "assigned_at":    m.assigned_at,
            "status":         m.status,
        }

    # Findings sans metadata en base → valeurs par défaut
    for fid in finding_ids:
        if fid not in result:
            result[fid] = {
                "finding_id":     fid,
                "is_pinned":      False,
                "pinned_by":      None,
                "pinned_at":      None,
                "assigned_to":    None,
                "assigned_to_id": None,
                "assigned_by":    None,
                "assigned_at":    None,
                "status":         "open",
            }

    return result


# ─── ROUTES DYNAMIQUES /{finding_id} ─────────────────────────────────────────

@router.get("/{finding_id}/metadata")
def get_metadata(
    finding_id: int,
    db:         Session = Depends(get_db),
    _:          User    = Depends(get_current_user),
):
    meta = db.query(FindingMetadata).filter_by(finding_id=finding_id).first()

    if not meta:
        return {
            "finding_id":     finding_id,
            "is_pinned":      False,
            "pinned_by":      None,
            "pinned_at":      None,
            "assigned_to":    None,
            "assigned_to_id": None,
            "assigned_by":    None,
            "assigned_at":    None,
            "status":         "open",
        }

    return {
        "finding_id":     finding_id,
        "is_pinned":      meta.is_pinned,
        "pinned_by":      meta.pinned_by_user.username if meta.pinned_by_user else None,
        "pinned_at":      meta.pinned_at,
        "assigned_to":    meta.assigned_to_user.username if meta.assigned_to_user else None,
        "assigned_to_id": meta.assigned_to_id,
        "assigned_by":    meta.assigned_by_user.username if meta.assigned_by_user else None,
        "assigned_at":    meta.assigned_at,
        "status":         meta.status,
    }


@router.post("/{finding_id}/pin")
def pin_finding(
    finding_id:   int,
    body:         PinRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    meta = get_or_create_metadata(
        finding_id    = finding_id,
        db            = db,
        finding_title = body.finding_title,
        product_name  = body.product_name,
        severity      = body.severity,
        ai_risk_score = body.ai_risk_score,
    )

    if meta.is_pinned:
        raise HTTPException(status_code=400, detail="Finding déjà épinglé")

    meta.is_pinned    = True
    meta.pinned_at    = datetime.now(timezone.utc)
    meta.pinned_by_id = current_user.id

    team = db.query(User).filter(
        User.role.in_(["admin", "analyst"]),
        User.id != current_user.id,
        User.status == "active",
    ).all()

    for member in team:
        db.add(Notification(
            type            = NotificationType.finding_pinned,
            title           = "Finding épinglé",
            message         = (
                f"{current_user.username} a épinglé : "
                f"'{body.finding_title}' (sévérité : {body.severity or 'N/A'})"
            ),
            related_user_id = member.id,
            finding_id      = finding_id,
            is_read         = False,
        ))

    db.commit()
    db.refresh(meta)

    return {
        "message":   f"Finding #{finding_id} épinglé",
        "pinned_by": current_user.username,
        "pinned_at": meta.pinned_at,
    }


@router.delete("/{finding_id}/pin")
def unpin_finding(
    finding_id:   int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    meta = db.query(FindingMetadata).filter_by(finding_id=finding_id).first()

    if not meta or not meta.is_pinned:
        raise HTTPException(status_code=404, detail="Finding non épinglé")

    meta.is_pinned    = False
    meta.pinned_at    = None
    meta.pinned_by_id = None

    db.commit()
    return {"message": f"Finding #{finding_id} désépinglé"}


@router.post("/{finding_id}/assign")
def assign_finding(
    finding_id:   int,
    body:         AssignRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    dev = db.query(User).filter_by(id=body.assigned_to_id).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    meta = get_or_create_metadata(
        finding_id    = finding_id,
        db            = db,
        finding_title = body.finding_title,
        product_name  = body.product_name,
        severity      = body.severity,
        ai_risk_score = body.ai_risk_score,
    )

    meta.assigned_to_id = dev.id
    meta.assigned_by_id = current_user.id
    meta.assigned_at    = datetime.now(timezone.utc)
    meta.status         = FindingStatus.in_progress

    db.add(Notification(
        type            = NotificationType.finding_assigned,
        title           = "Finding assigné",
        message         = (
            f"{current_user.username} t'a assigné : '{body.finding_title}' — "
            f"Score IA : {body.ai_risk_score or 'N/A'} | "
            f"Sévérité : {body.severity or 'N/A'}"
        ),
        related_user_id = dev.id,
        finding_id      = finding_id,
        is_read         = False,
    ))

    db.commit()
    db.refresh(meta)

    return {
        "message":     f"Finding #{finding_id} assigné à {dev.username}",
        "assigned_to": dev.username,
        "assigned_at": meta.assigned_at,
        "status":      meta.status,
    }


@router.delete("/{finding_id}/assign")
def unassign_finding(
    finding_id:   int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    meta = db.query(FindingMetadata).filter_by(finding_id=finding_id).first()

    if not meta or not meta.assigned_to_id:
        raise HTTPException(status_code=404, detail="Finding non assigné")

    meta.assigned_to_id = None
    meta.assigned_by_id = None
    meta.assigned_at    = None
    meta.status         = FindingStatus.open

    db.commit()
    return {"message": f"Finding #{finding_id} désassigné"}


@router.patch("/{finding_id}/status")
def update_status(
    finding_id:   int,
    body:         StatusUpdateRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    meta = db.query(FindingMetadata).filter_by(finding_id=finding_id).first()
    if not meta:
        raise HTTPException(status_code=404, detail="Métadonnées introuvables")

    meta.status = body.status
    db.commit()
    return {"message": f"Statut mis à jour : {body.status}"}