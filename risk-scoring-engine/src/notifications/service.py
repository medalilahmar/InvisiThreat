import smtplib
import os
from datetime import datetime, timezone
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
    admin_email   = os.getenv("ADMIN_EMAIL")
    smtp_host     = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port     = int(os.getenv("SMTP_PORT", 587))
    smtp_user     = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([admin_email, smtp_user, smtp_password]):
        print("[NOTIF] Variables SMTP manquantes — mail non envoyé")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"InvisiThreat <{smtp_user}>"
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
# Helper — ligne de données dans le tableau
# ──────────────────────────────────────────────
def _row(label: str, value: str, badge_style: str = None) -> str:
    if badge_style:
        val_html = f'<span style="{badge_style}">{value}</span>'
    else:
        val_html = (
            f'<span style="font-size:12px;color:#1e293b;font-weight:500;'
            f'font-family:Inter,Arial,sans-serif;">{value}</span>'
        )
    return f"""
      <tr>
        <td style="padding:9px 0;border-bottom:1px solid #f8fafc;
                   font-size:12px;color:#94a3b8;width:38%;
                   font-family:Inter,Arial,sans-serif;vertical-align:top;">
          {label}
        </td>
        <td style="padding:9px 0;border-bottom:1px solid #f8fafc;
                   text-align:right;vertical-align:top;">
          {val_html}
        </td>
      </tr>"""


# Badge styles réutilisables
BADGE_PENDING = (
    "font-size:11px;font-weight:500;color:#92400e;"
    "background:#fef3c7;border:1px solid #fde68a;"
    "border-radius:5px;padding:2px 8px;"
    "font-family:Inter,Arial,sans-serif;"
)
BADGE_ROLE = (
    "font-size:11px;font-weight:500;color:#1e40af;"
    "background:#eff6ff;border:1px solid #bfdbfe;"
    "border-radius:5px;padding:2px 8px;"
    "font-family:Inter,Arial,sans-serif;"
)
BADGE_DANGER = (
    "font-size:11px;font-weight:500;color:#991b1b;"
    "background:#fef2f2;border:1px solid #fecaca;"
    "border-radius:5px;padding:2px 8px;"
    "font-family:Inter,Arial,sans-serif;"
)


# ──────────────────────────────────────────────
# Template de base HTML — InvisiThreat Professional
# ──────────────────────────────────────────────
def _base_email(
    event_label: str,
    stripe_color: str,
    heading: str,
    subheading: str,
    body_rows_html: str,
    action_text: str,
    cta_url: str,
    cta_label: str,
    secondary_url: str = None,
    secondary_label: str = None,
) -> str:
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
    cta_href = cta_url if cta_url.startswith("http") else f"{frontend_url}{cta_url}"

    secondary_btn = ""
    if secondary_url and secondary_label:
        sec_href = secondary_url if secondary_url.startswith("http") else f"{frontend_url}{secondary_url}"
        secondary_btn = f"""
              <tr>
                <td style="padding-top:8px;">
                  <a href="{sec_href}"
                     style="display:block;text-align:center;background:transparent;
                            border:1px solid #e2e8f0;border-radius:8px;
                            padding:11px 24px;font-size:12px;
                            font-family:Inter,Arial,sans-serif;
                            color:#64748b;text-decoration:none;">
                    {secondary_label}
                  </a>
                </td>
              </tr>"""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>{heading}</title>
</head>
<body style="margin:0;padding:0;background-color:#f4f4f5;font-family:Inter,Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" border="0"
         style="background-color:#f4f4f5;padding:32px 16px;">
    <tr>
      <td align="center">
        <table width="560" cellpadding="0" cellspacing="0" border="0"
               style="max-width:560px;width:100%;">

          <!-- Brand -->
          <tr>
            <td style="padding-bottom:20px;">
              <table cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td style="padding-right:8px;vertical-align:middle;">
                    <div style="width:28px;height:28px;background:#0f172a;
                                border-radius:6px;text-align:center;line-height:28px;">
                      <span style="font-size:14px;color:white;">&#11042;</span>
                    </div>
                  </td>
                  <td style="vertical-align:middle;">
                    <span style="font-size:13px;font-weight:600;color:#0f172a;
                                 font-family:Inter,Arial,sans-serif;letter-spacing:0.02em;">
                      InvisiThreat
                    </span>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Card -->
          <tr>
            <td style="background:#ffffff;border-radius:12px;
                       border:1px solid #e4e4e7;overflow:hidden;">

              <!-- Stripe couleur -->
              <div style="height:3px;background:{stripe_color};font-size:0;line-height:0;">&nbsp;</div>

              <!-- Corps de la carte -->
              <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td style="padding:36px 40px 32px;">

                    <!-- Pill événement -->
                    <div style="display:inline-block;font-size:11px;font-weight:500;
                                color:#92400e;background:#fef3c7;
                                border:1px solid #fde68a;border-radius:20px;
                                padding:3px 10px;margin-bottom:20px;
                                font-family:Inter,Arial,sans-serif;">
                      &#9679; {event_label}
                    </div>

                    <!-- Titre -->
                    <h1 style="font-size:18px;font-weight:600;color:#0f172a;
                               margin:0 0 6px;line-height:1.35;
                               font-family:Inter,Arial,sans-serif;">
                      {heading}
                    </h1>

                    <!-- Sous-titre -->
                    <p style="font-size:13px;color:#64748b;margin:0 0 24px;
                              line-height:1.65;font-family:Inter,Arial,sans-serif;">
                      {subheading}
                    </p>

                    <!-- Séparateur -->
                    <div style="height:1px;background:#f1f5f9;margin-bottom:20px;"></div>

                    <!-- Tableau de données -->
                    <table width="100%" cellpadding="0" cellspacing="0" border="0"
                           style="margin-bottom:24px;">
                      {body_rows_html}
                    </table>

                    <!-- Boîte action requise -->
                    <table width="100%" cellpadding="0" cellspacing="0" border="0"
                           style="background:#f8fafc;border:1px solid #e2e8f0;
                                  border-radius:8px;margin-bottom:24px;">
                      <tr>
                        <td style="padding:14px 16px;">
                          <p style="font-size:11px;font-weight:600;color:#64748b;
                                    letter-spacing:0.06em;text-transform:uppercase;
                                    margin:0 0 5px;font-family:Inter,Arial,sans-serif;">
                            Action requise
                          </p>
                          <p style="font-size:12px;color:#475569;margin:0;
                                    line-height:1.65;font-family:Inter,Arial,sans-serif;">
                            {action_text}
                          </p>
                        </td>
                      </tr>
                    </table>

                    <!-- Boutons CTA -->
                    <table width="100%" cellpadding="0" cellspacing="0" border="0">
                      <tr>
                        <td>
                          <a href="{cta_href}"
                             style="display:block;text-align:center;
                                    background:#0f172a;border-radius:8px;
                                    padding:13px 24px;font-size:13px;
                                    font-family:Inter,Arial,sans-serif;
                                    font-weight:500;color:#ffffff;
                                    text-decoration:none;">
                            {cta_label}
                          </a>
                        </td>
                      </tr>
                      {secondary_btn}
                    </table>

                  </td>
                </tr>
              </table>

              <!-- Pied de carte -->
              <table width="100%" cellpadding="0" cellspacing="0" border="0"
                     style="background:#fafafa;border-top:1px solid #f1f5f9;">
                <tr>
                  <td style="padding:18px 40px;font-size:11px;color:#94a3b8;
                             font-family:Inter,Arial,sans-serif;">
                    InvisiThreat &middot; Notification automatique
                  </td>
                  <td align="right"
                      style="padding:18px 40px;font-size:11px;color:#cbd5e1;
                             font-family:Inter,Arial,sans-serif;">
                    Ne pas répondre à cet e-mail
                  </td>
                </tr>
              </table>

            </td>
          </tr>

          <!-- Note de bas de page -->
          <tr>
            <td style="text-align:center;padding-top:16px;
                       font-size:11px;color:#a1a1aa;
                       font-family:Inter,Arial,sans-serif;">
              Vous recevez cet e-mail car vous êtes administrateur de la plateforme InvisiThreat.
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>

</body>
</html>"""


# ──────────────────────────────────────────────
# NOTIFICATION 1 — Nouveau compte
# ──────────────────────────────────────────────
def notify_new_user(db: Session, user: User):
    """Appeler après un register réussi."""
    ts = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    create_notification(
        db=db,
        type=NotificationType.new_user,
        title=f"Nouveau compte : @{user.username}",
        message=f"{user.username} ({user.email}) attend validation.",
        related_user_id=user.id,
    )

    rows = (
        _row("Nom d'utilisateur", f"@{user.username}")
        + _row("Email", user.email)
        + _row("Fonction", user.job_title or "—")
        + _row("Département", user.department or "—")
        + _row("Rôle demandé", user.role or "developer", BADGE_ROLE)
        + _row("Statut", "En attente", BADGE_PENDING)
        + _row("Date d'inscription", ts)
    )

    html = _base_email(
        event_label="Validation requise",
        stripe_color="#f59e0b",
        heading="Nouveau compte en attente",
        subheading=(
            f"L'utilisateur <strong>{user.username}</strong> vient de s'inscrire "
            "sur la plateforme et attend votre approbation avant de pouvoir y accéder."
        ),
        body_rows_html=rows,
        action_text=(
            "Cet utilisateur n'a pas encore accès à la plateforme. "
            "Rendez-vous dans le panneau d'administration pour approuver "
            "ou rejeter ce compte."
        ),
        cta_url="/admin",
        cta_label="Accéder au panneau d'administration",
        secondary_url="/admin/users?filter=pending",
        secondary_label="Voir tous les comptes en attente",
    )

    send_email_to_admin(
        subject=f"[InvisiThreat] Nouveau compte en attente — @{user.username}",
        html_body=html,
    )


# ──────────────────────────────────────────────
# NOTIFICATION 2 — Tentatives de connexion échouées
# ──────────────────────────────────────────────
def notify_login_failed(db: Session, user: User):
    """Appeler après X tentatives de connexion échouées."""
    ts = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    create_notification(
        db=db,
        type=NotificationType.login_failed,
        title=f"Tentatives échouées : @{user.username}",
        message=f"{user.failed_login_attempts} tentatives échouées pour {user.email}.",
        related_user_id=user.id,
    )

    locked_until = (
        user.locked_until.strftime("%d %B %Y, %H:%M UTC")
        if user.locked_until else "—"
    )

    rows = (
        _row("Nom d'utilisateur", f"@{user.username}")
        + _row("Email", user.email)
        + _row("Tentatives échouées", str(user.failed_login_attempts), BADGE_DANGER)
        + _row("Statut du compte", "Bloqué temporairement" if user.locked_until else "Actif", BADGE_DANGER)
        + _row("Bloqué jusqu'au", locked_until)
        + _row("Horodatage", ts)
    )

    html = _base_email(
        event_label="Alerte sécurité",
        stripe_color="#ef4444",
        heading="Tentatives de connexion suspectes",
        subheading=(
            f"Un nombre inhabituel de tentatives de connexion échouées a été détecté "
            f"sur le compte <strong>@{user.username}</strong>."
        ),
        body_rows_html=rows,
        action_text=(
            "Vérifiez l'activité de ce compte et prenez les mesures nécessaires "
            "si vous suspectez une tentative d'accès non autorisé."
        ),
        cta_url=f"/admin/users/{user.id}",
        cta_label="Consulter le compte concerné",
    )

    send_email_to_admin(
        subject=f"[InvisiThreat] Alerte sécurité — Tentatives échouées @{user.username}",
        html_body=html,
    )


# ──────────────────────────────────────────────
# NOTIFICATION 3 — Compte bloqué
# ──────────────────────────────────────────────
def notify_user_blocked(db: Session, user: User):
    """Appeler quand un compte est bloqué par l'admin ou automatiquement."""
    ts = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    create_notification(
        db=db,
        type=NotificationType.user_blocked,
        title=f"Compte bloqué : @{user.username}",
        message=f"Le compte {user.email} vient d'être bloqué.",
        related_user_id=user.id,
    )

    locked_until = (
        user.locked_until.strftime("%d %B %Y, %H:%M UTC")
        if user.locked_until else "Indéfiniment"
    )

    rows = (
        _row("Nom d'utilisateur", f"@{user.username}")
        + _row("Email", user.email)
        + _row("Statut", "Bloqué", BADGE_DANGER)
        + _row("Bloqué jusqu'au", locked_until)
        + _row("Horodatage", ts)
    )

    html = _base_email(
        event_label="Compte bloqué",
        stripe_color="#ef4444",
        heading="Un compte vient d'être bloqué",
        subheading=(
            f"Le compte <strong>@{user.username}</strong> a été bloqué "
            "et ne peut plus accéder à la plateforme."
        ),
        body_rows_html=rows,
        action_text=(
            "Vous pouvez débloquer ce compte ou consulter son historique "
            "d'activité depuis le panneau d'administration."
        ),
        cta_url=f"/admin/users/{user.id}",
        cta_label="Gérer ce compte",
    )

    send_email_to_admin(
        subject=f"[InvisiThreat] Compte bloqué — @{user.username}",
        html_body=html,
    )



# ──────────────────────────────────────────────
# NOTIFICATION 4 — Nouveau finding détecté
# ──────────────────────────────────────────────
def notify_new_finding(
    db: Session,
    finding_title: str,
    severity: str,
    product_name: str = None,
    finding_id: int = None,
):
    """Notifie les users avec notify_on_new_finding=True."""
    users = db.query(User).filter(
        User.notify_on_new_finding == True,
        User.status == "active",
        User.is_active == True,
    ).all()

    for user in users:
        create_notification(
            db=db,
            type=NotificationType.finding_assigned,
            title=f"New {severity.upper()} finding detected",
            message=(
                f"Finding '{finding_title}'"
                + (f" in {product_name}" if product_name else "")
                + f" flagged as {severity.upper()}."
            ),
            related_user_id=user.id,
        )


# ──────────────────────────────────────────────
# NOTIFICATION 5 — Pull request créé
# ──────────────────────────────────────────────
def notify_pr_merged(
    db: Session,
    pr_title: str,
    pr_url: str = None,
    finding_id: int = None,
):
    """Notifie les users avec notify_on_pr_merged=True."""
    users = db.query(User).filter(
        User.notify_on_pr_merged == True,
        User.status == "active",
        User.is_active == True,
    ).all()

    for user in users:
        create_notification(
            db=db,
            type=NotificationType.project_sync,
            title=f"Auto-fix PR created: {pr_title[:60]}",
            message=(
                f"A security fix PR was opened"
                + (f" for finding #{finding_id}" if finding_id else "")
                + (f": {pr_url}" if pr_url else "")
                + "."
            ),
            related_user_id=user.id,
        )