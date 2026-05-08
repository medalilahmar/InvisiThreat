from sqlalchemy import (
    Column, Integer, String, Boolean,
    Table, ForeignKey, DateTime
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .connection import Base

# ─── Table d'association User ↔ Project (many-to-many) ───────────────────────
project_assignment = Table(
    "project_assignment",
    Base.metadata,
    Column("user_id",    Integer, ForeignKey("users.id",    ondelete="CASCADE"), primary_key=True),
    Column("project_id", Integer, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
)



class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String,  unique=True, nullable=False, index=True)
    email           = Column(String,  unique=True, nullable=False)
    hashed_password = Column(String,  nullable=False)

    # Rôles : admin | manager | analyst | developer
    role            = Column(String,  default="developer")

    # Statut : pending | active | blocked
    status          = Column(String,  default="pending")

    # is_active reste True même en "pending" pour ne pas bloquer le login
    # mais le backend vérifie status == "active" avant d'autoriser
    is_active       = Column(Boolean, default=True)

    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())

    # Relation vers les projets assignés
    projects = relationship(
        "Project",
        secondary=project_assignment,
        back_populates="users"
    )

class Project(Base):
    __tablename__ = "projects"

    # id = product_id DefectDojo (même valeur, même source)
    id          = Column(Integer, primary_key=True)
    name        = Column(String, nullable=False)
    description = Column(String, nullable=True)

    users = relationship(
        "User",
        secondary=project_assignment,
        back_populates="projects"
    )