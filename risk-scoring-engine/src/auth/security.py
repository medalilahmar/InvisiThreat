import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import User

SECRET_KEY  = os.getenv("SECRET_KEY", "CHANGEZ_MOI_EN_PRODUCTION_!!!")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ─── Mot de passe ─────────────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# ─── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = expire
    # ← Force sub en string (standard JWT)
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ─── Dépendances FastAPI ──────────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        raw_sub = payload.get("sub")
        if raw_sub is None:
            raise exc
        user_id = int(raw_sub)  # ← Conversion string → int
    except (JWTError, ValueError, TypeError):
        raise exc

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise exc

    if user.status == "pending":
        raise HTTPException(
            status_code=403,
            detail="Compte en attente de validation par l'administrateur"
        )
    if user.status == "blocked":
        raise HTTPException(
            status_code=403,
            detail="Votre compte a été bloqué. Contactez l'administrateur."
        )
    return user

def get_accessible_product_ids(
    current_user: User = Depends(get_current_user)
) -> List[int]:
    if current_user.role in ("admin", "analyst"):
        return []
    return [p.id for p in current_user.projects]

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Réservé aux administrateurs")
    return current_user

def require_admin_or_manager(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.role not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Accès non autorisé")
    return current_user