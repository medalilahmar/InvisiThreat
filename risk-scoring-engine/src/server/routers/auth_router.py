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

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class UpdateProfileRequest(BaseModel):
    email:    Optional[EmailStr] = None
    username: Optional[str]      = None  # ← AJOUT

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str
    user:         dict

# ─── Helpers ──────────────────────────────────────────────────────────────────

def user_to_dict(user: User) -> dict:
    """Sérialise un User en dict — utilisé partout pour cohérence."""
    return {
        "id":       user.id,
        "username": user.username,
        "email":    user.email,
        "role":     user.role,
        "status":   user.status,
        "created_at": str(user.created_at) if user.created_at else None,
        "projects": [{"id": p.id, "name": p.name} for p in user.projects],
        # ── Nouveaux champs exposés au frontend ──────────────────────────────
        "last_login":             str(user.last_login) if user.last_login else None,
        "locked_until":           str(user.locked_until) if user.locked_until else None,
        "failed_login_attempts":  user.failed_login_attempts,
        # ── Profil ───────────────────────────────────────────────────────────
        "job_title":              user.job_title,
        "department":             user.department,
        "phone":                  user.phone,
        "avatar_url":             user.avatar_url,

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
def update_profile(
    body: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Modifier son propre profil (email et/ou username)."""
    if body.email:
        existing = db.query(User).filter(
            User.email == body.email,
            User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(400, "Cet email est déjà utilisé")
        current_user.email = body.email

    if body.username:
        if len(body.username.strip()) < 3:
            raise HTTPException(400, "Le nom d'utilisateur doit contenir au moins 3 caractères")
        existing = db.query(User).filter(
            User.username == body.username,
            User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(400, "Ce nom d'utilisateur est déjà pris")
        current_user.username = body.username

    db.commit()
    db.refresh(current_user)
    return user_to_dict(current_user)  # ← retourne le user complet mis à jour


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
    db.commit()
    return {"message": "Mot de passe changé avec succès"}


@router.post("/logout")
def logout(_: User = Depends(get_current_user)):
    """Logout — nettoyage côté client, JWT stateless."""
    return {"message": "Déconnexion réussie"}