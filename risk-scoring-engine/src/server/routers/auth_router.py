from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional

from database.connection import get_db
from database.models import User , NotificationType , Notification
from auth.security import get_current_user, verify_password, get_password_hash, create_access_token
from notifications.service import notify_new_user , create_notification
from datetime import datetime, timedelta, timezone
 


router = APIRouter(prefix="/auth", tags=[" Authentification"])

# ─── Schémas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    job_title: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class UpdateProfileRequest(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    
    job_title: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    
    github_username: Optional[str] = None
    github_token: Optional[str] = None
    jira_email: Optional[str] = None
    jira_token: Optional[str] = None
    
    notify_on_new_finding: Optional[bool] = None
    notify_on_pr_merged: Optional[bool] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str
    user:         dict

# ─── Helpers ──────────────────────────────────────────────────────────────────

def user_to_dict(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "is_active": user.is_active,
        
        "created_at": str(user.created_at) if user.created_at else None,
        "updated_at": str(user.updated_at) if user.updated_at else None,
        "last_login": str(user.last_login) if user.last_login else None,
        "password_changed_at": str(user.password_changed_at) if user.password_changed_at else None,
        
        "failed_login_attempts": user.failed_login_attempts,
        "locked_until": str(user.locked_until) if user.locked_until else None,
        
        "job_title": user.job_title,
        "department": user.department,
        "phone": user.phone,
        "avatar_url": user.avatar_url,
        
        "github_username": user.github_username,
        "jira_email": user.jira_email,
        
        "notify_on_new_finding": user.notify_on_new_finding,
        "notify_on_pr_merged": user.notify_on_pr_merged,
        
        "projects": [{"id": p.id, "name": p.name} for p in user.projects],
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Création de compte — statut pending jusqu'à validation admin."""
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, "Ce nom d'utilisateur est déjà pris")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Cet email est déjà utilisé")

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=get_password_hash(body.password),
        role="developer",
        status="pending",
        job_title=body.job_title,
        department=body.department,
        phone=body.phone,
        avatar_url=body.avatar_url,
    )
    db.add(user)
    db.commit()
    try:
        notify_new_user(db=db, user=user)
    except Exception as e:
        print(f"[NOTIF] Erreur notification : {e}")
    return {
        "message": "Compte créé avec succès. En attente de validation par l'administrateur.",
        "status": "pending",
    }


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Login avec verrouillage après 5 échecs + notifications admin."""
    user = db.query(User).filter(User.username == form_data.username).first()
    now = datetime.now(timezone.utc)

    # 1. Compte déjà verrouillé temporairement ?
    if user and user.locked_until:
        locked_until = user.locked_until
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)

        if locked_until > now:
            wait_minutes = int((locked_until - now).total_seconds() // 60)

            # Anti-spam : notifier seulement si aucune notif dans les 5 dernières minutes
            last_notif = (
                db.query(Notification)
                .filter(Notification.related_user_id == user.id)
                .order_by(Notification.created_at.desc())
                .first()
            )
            last_notif_dt = last_notif.created_at if last_notif else None
            if last_notif_dt and last_notif_dt.tzinfo is None:
                last_notif_dt = last_notif_dt.replace(tzinfo=timezone.utc)

            should_notify = (
                not last_notif_dt or
                (now - last_notif_dt).total_seconds() > 300
            )

            if should_notify:
                try:
                    create_notification(
                        db=db,
                        type=NotificationType.user_blocked,
                        title=f"Tentative connexion compte verrouillé : @{user.username}",
                        message=f"{user.username} a essayé de se connecter. Compte verrouillé encore {wait_minutes} min.",
                        related_user_id=user.id
                    )
                except Exception as e:
                    print(f"[NOTIF] Erreur notification : {e}")

            raise HTTPException(
                status_code=423,
                detail=f"Compte verrouillé. Réessayez dans {wait_minutes} minutes."
            )

    # 2. Identifiants incorrects → on incrémente les échecs
    if not user or not verify_password(form_data.password, user.hashed_password):
        if user:
            user.failed_login_attempts += 1

            if user.failed_login_attempts >= 5:
                user.locked_until = now + timedelta(minutes=30)
                user.failed_login_attempts = 0
                db.commit()
                try:
                    create_notification(
                        db=db,
                        type=NotificationType.login_failed,
                        title=f"Verrouillage : @{user.username}",
                        message=(
                            f"5 échecs consécutifs. Compte verrouillé "
                            f"jusqu'à {user.locked_until.strftime('%H:%M')} UTC."
                        ),
                        related_user_id=user.id
                    )
                except Exception as e:
                    print(f"[NOTIF] Erreur notification : {e}")

                raise HTTPException(
                    status_code=423,
                    detail="Compte verrouillé après 5 échecs. Réessayez dans 30 minutes."
                )

            db.commit()

        raise HTTPException(
            status_code=400,
            detail="Nom d'utilisateur ou mot de passe incorrect"
        )

    # 3. Vérification du statut (pending / blocked)
    if user.status == "pending":
        raise HTTPException(
            status_code=403,
            detail="Votre compte est en attente de validation par l'administrateur"
        )
    if user.status == "blocked":
        raise HTTPException(
            status_code=403,
            detail="Votre compte a été bloqué. Contactez l'administrateur."
        )

    # 4. Succès → réinitialiser les échecs et le verrouillage
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = now
    db.commit()

    token = create_access_token(data={
        "sub":      str(user.id),
        "role":     user.role,
        "username": user.username,
    })

    return {
        "access_token": token,
        "token_type":   "bearer",
        "user":         user_to_dict(user),
    }


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    """Retourne les infos de l'utilisateur connecté."""
    return user_to_dict(current_user)


@router.put("/me")
def update_profile(body: UpdateProfileRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Identité
    if body.email:
        existing = db.query(User).filter(User.email == body.email, User.id != current_user.id).first()
        if existing:
            raise HTTPException(400, "Cet email est déjà utilisé")
        current_user.email = body.email
    
    if body.username:
        if len(body.username.strip()) < 3:
            raise HTTPException(400, "Le nom d'utilisateur doit contenir au moins 3 caractères")
        existing = db.query(User).filter(User.username == body.username, User.id != current_user.id).first()
        if existing:
            raise HTTPException(400, "Ce nom d'utilisateur est déjà pris")
        current_user.username = body.username
    
    if body.job_title is not None:
        current_user.job_title = body.job_title
    if body.department is not None:
        current_user.department = body.department
    if body.phone is not None:
        current_user.phone = body.phone
    if body.avatar_url is not None:
        current_user.avatar_url = body.avatar_url
    
    if body.github_username is not None:
        current_user.github_username = body.github_username
    if body.github_token is not None:
        current_user.github_token = body.github_token  # TODO: chiffrer
    if body.jira_email is not None:
        current_user.jira_email = body.jira_email
    if body.jira_token is not None:
        current_user.jira_token = body.jira_token  # TODO: chiffrer
    
    if body.notify_on_new_finding is not None:
        current_user.notify_on_new_finding = body.notify_on_new_finding
    if body.notify_on_pr_merged is not None:
        current_user.notify_on_pr_merged = body.notify_on_pr_merged
    
    current_user.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(current_user)
    return user_to_dict(current_user)


@router.put("/change-password")
def change_password(
    body: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Changer son propre mot de passe."""
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(400, "Mot de passe actuel incorrect")
    if len(body.new_password) < 8:
        raise HTTPException(400, "Nouveau mot de passe trop court (min 8 caractères)")
    if body.current_password == body.new_password:
        raise HTTPException(400, "Le nouveau mot de passe doit être différent de l'ancien")

    current_user.hashed_password = get_password_hash(body.new_password)
    current_user.password_changed_at = datetime.now(timezone.utc)  # ✅ NOUVEAU
    db.commit()
    return {"message": "Mot de passe changé avec succès"}


@router.post("/logout")
def logout(_: User = Depends(get_current_user)):
    """Logout — nettoyage côté client, JWT stateless."""
    return {"message": "Déconnexion réussie"}