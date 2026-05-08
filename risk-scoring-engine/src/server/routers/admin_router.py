from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import csv, io

from database.connection import get_db
from database.models import User, Project
from auth.security import require_admin, get_password_hash, get_current_user

router = APIRouter(prefix="/admin", tags=["👑 Administration"])

# ─── Schémas ──────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "developer"

class UserUpdate(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = None

class ResetPasswordRequest(BaseModel):
    new_password: str

# ─── Routes statiques EN PREMIER ──────────────────────────────────────────────

@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "status": u.status,
            "is_active": u.is_active,
            "created_at": str(u.created_at) if u.created_at else None,
            "projects": [{"id": p.id, "name": p.name} for p in u.projects]
        }
        for u in users
    ]

@router.get("/users/pending")
def list_pending_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    users = db.query(User).filter(User.status == "pending").all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "created_at": str(u.created_at) if u.created_at else None
        }
        for u in users
    ]

@router.get("/users/search")
def search_users(
    q: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    query = db.query(User)
    if q:
        query = query.filter(
            (User.username.ilike(f"%{q}%")) |
            (User.email.ilike(f"%{q}%"))
        )
    if role:
        query = query.filter(User.role == role)
    if status:
        query = query.filter(User.status == status)

    total = query.count()
    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "role": u.role,
                "status": u.status,
                "created_at": str(u.created_at) if u.created_at else None,
                "projects": [{"id": p.id, "name": p.name} for p in u.projects],
                "projects_count": len(u.projects)
            }
            for u in users
        ]
    }

@router.get("/users/export")
def export_users_csv(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    users = db.query(User).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "username", "email", "role", "status", "created_at", "projects"])
    for u in users:
        writer.writerow([
            u.id, u.username, u.email, u.role, u.status,
            str(u.created_at) if u.created_at else "",
            "|".join(p.name for p in u.projects)
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=users_export.csv"}
    )

@router.get("/stats")
def admin_stats(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    total_users    = db.query(User).count()
    pending_users  = db.query(User).filter(User.status == "pending").count()
    active_users   = db.query(User).filter(User.status == "active").count()
    blocked_users  = db.query(User).filter(User.status == "blocked").count()
    total_projects = db.query(Project).count()

    by_role = {}
    for role in ["admin", "manager", "analyst", "developer"]:
        by_role[role] = db.query(User).filter(User.role == role).count()

    return {
        "users": {
            "total": total_users,
            "pending": pending_users,
            "active": active_users,
            "blocked": blocked_users,
            "by_role": by_role
        },
        "projects": {
            "total": total_projects
        }
    }

@router.get("/projects")
def list_projects(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    projects = db.query(Project).all()
    return [{"id": p.id, "name": p.name} for p in projects]

# ─── Routes dynamiques ENSUITE ────────────────────────────────────────────────

@router.get("/users/{user_id}/projects")
def get_user_projects(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")
    return [{"id": p.id, "name": p.name} for p in user.projects]

@router.post("/users/{user_id}/approve")
def approve_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")
    if user.status == "active":
        raise HTTPException(400, "Utilisateur déjà actif")
    user.status = "active"
    db.commit()
    return {"message": f"Utilisateur '{user.username}' approuvé avec succès"}

@router.post("/users/{user_id}/block")
def block_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")
    if user.id == admin.id:
        raise HTTPException(400, "Vous ne pouvez pas vous bloquer vous-même")
    user.status = "blocked"
    db.commit()
    return {"message": f"Utilisateur '{user.username}' bloqué"}

@router.post("/users/{user_id}/unblock")
def unblock_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")
    user.status = "active"
    db.commit()
    return {"message": f"Utilisateur '{user.username}' débloqué"}

@router.put("/users/{user_id}")
def update_user(
    user_id: int,
    body: UserUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")

    VALID_ROLES    = {"admin", "manager", "analyst", "developer"}
    VALID_STATUSES = {"pending", "active", "blocked"}

    if body.role and body.role not in VALID_ROLES:
        raise HTTPException(400, f"Rôle invalide. Valeurs acceptées : {VALID_ROLES}")
    if body.status and body.status not in VALID_STATUSES:
        raise HTTPException(400, f"Statut invalide. Valeurs acceptées : {VALID_STATUSES}")

    if body.role:
        user.role = body.role
    if body.status:
        user.status = body.status

    db.commit()
    return {"message": "Utilisateur mis à jour", "role": user.role, "status": user.status}

@router.post("/users/{user_id}/projects")
def assign_projects(
    user_id: int,
    project_ids: List[int],
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")

    projects = db.query(Project).filter(Project.id.in_(project_ids)).all()
    if len(projects) != len(project_ids):
        found_ids = {p.id for p in projects}
        missing   = set(project_ids) - found_ids
        raise HTTPException(400, f"Projets introuvables : {missing}")

    user.projects = projects
    db.commit()
    return {
        "message": f"{len(projects)} projet(s) assigné(s) à '{user.username}'",
        "projects": [{"id": p.id, "name": p.name} for p in projects]
    }

@router.post("/users/{user_id}/reset-password")
def reset_user_password(
    user_id: int,
    body: ResetPasswordRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")
    if len(body.new_password) < 8:
        raise HTTPException(400, "Mot de passe trop court (min 8 caractères)")
    user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    return {"message": "Mot de passe réinitialisé avec succès"}

@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Utilisateur introuvable")
    if user.id == admin.id:
        raise HTTPException(400, "Impossible de supprimer votre propre compte")
    if user.role == "admin":
        raise HTTPException(400, "Impossible de supprimer un autre administrateur")
    db.delete(user)
    db.commit()
    return {"message": f"Utilisateur '{user.username}' supprimé"}

@router.post("/projects/sync")
def sync_projects(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin)
):
    from api import local_data_loader
    if not local_data_loader or not local_data_loader.is_ready:
        raise HTTPException(503, "Le loader de données n'est pas disponible")
    existing_ids = {p.id for p in db.query(Project).all()}
    added = []
    for prod_id, prod_data in local_data_loader.products.items():
        if prod_id not in existing_ids:
            db.add(Project(id=prod_id, name=prod_data['name']))
            added.append({"id": prod_id, "name": prod_data['name']})
    db.commit()
    return {"synced": len(added), "new_projects": added}