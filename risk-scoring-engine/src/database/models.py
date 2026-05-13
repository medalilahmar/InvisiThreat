from sqlalchemy import (
    Column, Integer, String, Boolean,
    Table, ForeignKey, DateTime
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .connection import Base
import enum
from sqlalchemy import Enum as SAEnum, Text

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
    role            = Column(String,  default="developer")
    status          = Column(String,  default="pending")
    is_active       = Column(Boolean, default=True)

    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())

    job_title   = Column(String, nullable=True)
    department  = Column(String, nullable=True)
    phone       = Column(String, nullable=True)
    avatar_url  = Column(String, nullable=True)

    github_username = Column(String, nullable=True)
    github_token    = Column(Text,   nullable=True)   
    jira_email      = Column(String, nullable=True)
    jira_token      = Column(Text,   nullable=True)   

    notify_on_new_finding = Column(Boolean, default=True)
    notify_on_pr_merged   = Column(Boolean, default=True)

    last_login          = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until          = Column(DateTime(timezone=True), nullable=True)

    projects = relationship(
        "Project",
        secondary=project_assignment,
        back_populates="users"
    )

    notifications = relationship(
        "Notification",
        back_populates="related_user",
    )

class Project(Base):
    __tablename__ = "projects"

    id          = Column(Integer, primary_key=True)
    name        = Column(String, nullable=False)
    description = Column(String, nullable=True)

    users = relationship(
        "User",
        secondary=project_assignment,
        back_populates="projects"
    )




# ─── AJOUT : Notification ─────────────────────────────────────────────────────


class NotificationType(str, enum.Enum):
    new_user         = "new_user"          # nouveau compte créé
    pending_reminder = "pending_reminder"  # compte pending depuis +24h
    login_failed     = "login_failed"      # trop de tentatives échouées
    user_blocked     = "user_blocked"      # user vient d'être bloqué
    project_sync     = "project_sync"      # sync DefectDojo terminée


class Notification(Base):
    __tablename__ = "notifications"

    id      = Column(Integer, primary_key=True, index=True)
    type    = Column(SAEnum(NotificationType), nullable=False)
    title   = Column(String,  nullable=False)
    message = Column(Text,    nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)

    # Lien vers le user concerné (ex: celui qui vient de s'inscrire)
    # SET NULL : si le user est supprimé, la notif reste dans l'historique
    related_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    related_user = relationship(
        "User",
        back_populates="notifications",
        foreign_keys=[related_user_id]
    )