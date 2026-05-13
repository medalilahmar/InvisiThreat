import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

from database.connection import SessionLocal, engine, Base
from database.models import User
from auth.security import get_password_hash
from datetime import datetime, timezone

Base.metadata.create_all(bind=engine)   # tables créées dans la base définie par .env (PostgreSQL)

db = SessionLocal()
try:
    if db.query(User).filter(User.username == "admin").first():
        print(" L'admin existe déjà.")
    else:
        admin = User(
            username="admin",
            email="admin@invisithreat.local",
            hashed_password=get_password_hash("3005"),
            role="admin",
            status="active",
            is_active=True, 

            job_title="Platform Administrator",
            department="Security Operations",
            phone=None,      
            avatar_url=None,

            github_username=None,
            github_token=None,
            jira_email=None,
            jira_token=None,

            notify_on_new_finding=True,
            notify_on_pr_merged=True,

            last_login=None,
            password_changed_at=datetime.now(timezone.utc),
        )
        db.add(admin)
        db.commit()
        print(" Admin créé : username=admin / password=3005")
        print("  Changez ce mot de passe immédiatement après connexion !")
finally:
    db.close()