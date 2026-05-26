from sqlalchemy import (
    Column, Integer, String, Boolean,
    Table, ForeignKey, DateTime, Float
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

    last_login            = Column(DateTime(timezone=True), nullable=True)
    password_changed_at   = Column(DateTime(timezone=True), nullable=True)
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


# ─── NotificationType ─────────────────────────────────────────────────────────
class NotificationType(str, enum.Enum):
    new_user         = "new_user"
    pending_reminder = "pending_reminder"
    login_failed     = "login_failed"
    user_blocked     = "user_blocked"
    project_sync     = "project_sync"
    finding_pinned   = "finding_pinned"   
    finding_assigned = "finding_assigned"  


class Notification(Base):
    __tablename__ = "notifications"

    id      = Column(Integer, primary_key=True, index=True)
    type    = Column(SAEnum(NotificationType), nullable=False)
    title   = Column(String,  nullable=False)
    message = Column(Text,    nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)

    related_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    finding_id = Column(Integer, nullable=True)  # ← NOUVEAU

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    related_user = relationship(
        "User",
        back_populates="notifications",
        foreign_keys=[related_user_id]
    )


# ─── FindingStatus + FindingMetadata ──────────────────────────────────────────
class FindingStatus(str, enum.Enum):
    open        = "open"
    in_progress = "in_progress"
    resolved    = "resolved"
    wont_fix    = "wont_fix"


class FindingMetadata(Base):
    __tablename__ = "finding_metadata"

    id         = Column(Integer, primary_key=True, index=True)
    finding_id = Column(Integer, nullable=False, unique=True, index=True)

    finding_title   = Column(String,  nullable=True)
    product_name    = Column(String,  nullable=True)
    engagement_name = Column(String,  nullable=True)
    severity        = Column(String,  nullable=True)
    ai_risk_score   = Column(Float,   nullable=True)

    # Épinglage
    is_pinned    = Column(Boolean, default=False, nullable=False)
    pinned_at    = Column(DateTime(timezone=True), nullable=True)
    pinned_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Assignation
    assigned_to_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    assigned_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    assigned_at    = Column(DateTime(timezone=True), nullable=True)

    # Statut
    status = Column(SAEnum(FindingStatus), default=FindingStatus.open, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relations
    pinned_by_user   = relationship("User", foreign_keys=[pinned_by_id])
    assigned_to_user = relationship("User", foreign_keys=[assigned_to_id])
    assigned_by_user = relationship("User", foreign_keys=[assigned_by_id])


# ─── FindingScoreHistory ──────────────────────────────────────────────────────
class FindingScoreHistory(Base):
    __tablename__ = "finding_scores_history"

    id              = Column(Integer, primary_key=True, index=True)
    finding_id      = Column(Integer, nullable=False, index=True)
    finding_title   = Column(String,  nullable=True)
    product_name    = Column(String,  nullable=True)
    engagement_name = Column(String,  nullable=True)
    ai_risk_score   = Column(Float,   nullable=False)
    risk_level      = Column(String(20), nullable=True)
    confidence      = Column(Float,   nullable=True)
    cvss_score      = Column(Float,   nullable=True)
    severity        = Column(String,  nullable=True)
    scored_at       = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    model_version   = Column(String(50), nullable=True)