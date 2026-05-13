import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy.orm import Session
from database.models import Notification, NotificationType, User


# ──────────────────────────────────────────────
# Créer une notification en base
# ──────────────────────────────────────────────
def create_notification(
    db: Session,
    type: NotificationType,
    title: str,
    message: str,
    related_user_id: int = None
) -> Notification:
    notif = Notification(
        type=type,
        title=title,
        message=message,
        related_user_id=related_user_id
    )
    db.add(notif)
    db.commit()
    db.refresh(notif)
    return notif


# ──────────────────────────────────────────────
# Envoyer un mail à l'admin
# ──────────────────────────────────────────────
def send_email_to_admin(subject: str, html_body: str):
    """
    Variables .env requises :
      ADMIN_EMAIL      → destinataire
      SMTP_HOST        → ex: smtp.gmail.com
      SMTP_PORT        → ex: 587
      SMTP_USER        → adresse expéditeur
      SMTP_PASSWORD    → mot de passe app Gmail ou clé SendGrid
    """
    admin_email  = os.getenv("ADMIN_EMAIL")
    smtp_host    = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port    = int(os.getenv("SMTP_PORT", 587))
    smtp_user    = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([admin_email, smtp_user, smtp_password]):
        print("[NOTIF] Variables SMTP manquantes — mail non envoyé")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = admin_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, admin_email, msg.as_string())
        print(f"[NOTIF] Mail envoyé à {admin_email}")
    except Exception as e:
        print(f"[NOTIF] Erreur mail : {e}")


# ──────────────────────────────────────────────
# Template mail — nouveau compte
# ──────────────────────────────────────────────
def notify_new_user(db: Session, user: User):
    """Appeler après un register réussi."""

    # 1. Notification en base
    create_notification(
        db=db,
        type=NotificationType.new_user,
        title=f"Nouveau compte : @{user.username}",
        message=f"{user.username} ({user.email}) attend validation.",
        related_user_id=user.id
    )

    # 2. Mail admin
    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:auto">
      <h2 style="color:#1e293b">🔔 Nouveau compte en attente</h2>
      <table style="width:100%;border-collapse:collapse">
        <tr><td style="padding:8px;color:#64748b">Username</td>
            <td style="padding:8px"><strong>@{user.username}</strong></td></tr>
        <tr style="background:#f8fafc">
            <td style="padding:8px;color:#64748b">Email</td>
            <td style="padding:8px">{user.email}</td></tr>
        <tr><td style="padding:8px;color:#64748b">Fonction</td>
            <td style="padding:8px">{user.job_title or "—"}</td></tr>
      </table>
      <div style="margin-top:24px">
        <a href="{os.getenv('FRONTEND_URL','http://localhost:5173')}/admin"
           style="background:#6366f1;color:white;padding:12px 24px;
                  border-radius:8px;text-decoration:none">
          Gérer dans l'Admin Panel →
        </a>
      </div>
      <p style="color:#94a3b8;font-size:12px;margin-top:32px">
        InvisiThreat — notification automatique
      </p>
    </div>
    """
    send_email_to_admin(
        subject=f"[InvisiThreat] Nouveau compte en attente — @{user.username}",
        html_body=html
    )