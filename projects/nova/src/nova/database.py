import sqlite3
import os

DB_FILE = "nova_vulnerable.db"

def init_db():
    """Initialise une base de données SQLite simple avec des données de test."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    # Insertion de quelques utilisateurs pour les tests
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'SuperSecretAdminPass123!', 'admin')")
        cursor.execute("INSERT INTO users (username, password, role) VALUES ('user1', 'password123', 'user')")
        conn.commit()
    conn.close()

def get_user_vulnerable(username):
    """
    RÉCUPÉRATION D'UTILISATEUR VULNÉRABLE (SQL Injection / SAST Alert).
    Cette fonction concatène directement l'entrée utilisateur dans la requête SQL sans l'assainir.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Faille SQL Injection délibérée (SAST & DAST Target)
    query = f"SELECT id, username, role FROM users WHERE username = '{username}'"
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result
