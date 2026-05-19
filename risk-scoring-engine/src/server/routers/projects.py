from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database.connection import get_db
from database.models import User, Project, project_assignment
from auth.security import get_current_user

router = APIRouter(prefix="/projects", tags=["projects"])


# ─── Schémas ──────────────────────────────────────────────────────────────────

class MemberResponse(BaseModel):
    id:         int
    username:   str
    email:      str
    role:       str
    job_title:  Optional[str] = None
    department: Optional[str] = None
    avatar_url: Optional[str] = None
    status:     str

    class Config:
        from_attributes = True

class AddMemberRequest(BaseModel):
    user_id: int


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@router.get("/{project_id}/members")
def get_project_members(
    project_id:   int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user)
):
    """
    Retourne tous les membres d'un projet.
    """
    # Vérifie que le projet existe
    project = db.query(Project).filter_by(id=project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projet introuvable")

    return {
        "project_id":   project.id,
        "project_name": project.name,
        "members_count": len(project.users),
        "members": [
            {
                "id":         u.id,
                "username":   u.username,
                "email":      u.email,
                "role":       u.role,
                "job_title":  u.job_title,
                "department": u.department,
                "avatar_url": u.avatar_url,
                "status":     u.status
            }
            for u in project.users
        ]
    }


@router.post("/{project_id}/members")
def add_member_to_project(
    project_id:   int,
    body:         AddMemberRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user)
):
    """
    Ajouter un membre à un projet — admin seulement.
    """
    # Seulement les admins peuvent ajouter des membres
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Seul un admin peut ajouter des membres"
        )

    # Vérifie projet et user
    project = db.query(Project).filter_by(id=project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projet introuvable")

    user = db.query(User).filter_by(id=body.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    # Vérifie si déjà membre
    already_member = user in project.users
    if already_member:
        raise HTTPException(
            status_code=400,
            detail=f"{user.username} est déjà membre de ce projet"
        )

    # Ajouter le membre
    project.users.append(user)
    db.commit()

    return {
        "message":      f"{user.username} ajouté au projet {project.name}",
        "project_id":   project_id,
        "user_id":      user.id,
        "username":     user.username
    }


@router.delete("/{project_id}/members/{user_id}")
def remove_member_from_project(
    project_id:   int,
    user_id:      int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user)
):
    """
    Retirer un membre d'un projet — admin seulement.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Seul un admin peut retirer des membres"
        )

    project = db.query(Project).filter_by(id=project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projet introuvable")

    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    if user not in project.users:
        raise HTTPException(
            status_code=404,
            detail=f"{user.username} n'est pas membre de ce projet"
        )

    project.users.remove(user)
    db.commit()

    return {
        "message":    f"{user.username} retiré du projet {project.name}",
        "project_id": project_id,
        "user_id":    user_id
    }


@router.get("/my-projects")
def get_my_projects(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user)
):
    """
    Retourne les projets de l'utilisateur connecté.
    """
    return [
        {
            "id":           p.id,
            "name":         p.name,
            "description":  p.description,
            "members_count": len(p.users)
        }
        for p in current_user.projects
    ]