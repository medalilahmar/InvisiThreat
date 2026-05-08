from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional

from database.connection import get_db
from database.models import User
from auth.security import get_current_user, verify_password, get_password_hash, create_access_token

router = APIRouter(prefix="/auth", tags=["🔐 Authentification"])

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
    return {
        "message": "Compte créé avec succès. En attente de validation par l'administrateur.",
        "status": "pending",
    }


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Login — vérifie identifiants ET statut du compte."""
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(400, "Nom d'utilisateur ou mot de passe incorrect")
    if user.status == "pending":
        raise HTTPException(403, "Votre compte est en attente de validation par l'administrateur")
    if user.status == "blocked":
        raise HTTPException(403, "Votre compte a été bloqué. Contactez l'administrateur.")

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