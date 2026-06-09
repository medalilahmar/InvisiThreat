import os
import subprocess
import yaml
import pickle
import base64
from flask import Flask, request, render_template_string, send_file
from nova.database import init_db, get_user_vulnerable

app = Flask(__name__)

# ----------------- VULNÉRABILITÉ SAST : SECRETS CODÉS EN DUR -----------------
app.config['SECRET_KEY'] = "d41d8cd98f00b204e9800998ecf8427e"  # Fausse clé secrète codée en dur
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"  # Faux secrets AWS pour déclencher les scanners SAST
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# Initialisation de la base de données
init_db()

@app.route('/')
def index():
    return """
    <h1>Projet Nova - Benchmark Vulnérabilités</h1>
    <p>Ce projet contient des failles de sécurité délibérées pour tester vos outils SAST, DAST et SCA.</p>
    <ul>
        <li><a href="/login?username=admin">Injection SQL (/login)</a></li>
        <li><a href="/ping?ip=8.8.8.8">Injection de commande (/ping)</a></li>
        <li><a href="/read?file=src/nova/app.py">Path Traversal (/read)</a></li>
        <li><a href="/xss?name=%3Cscript%3Ealert(1)%3C/script%3E">XSS Réfracté (/xss)</a></li>
        <li><a href="/unsafe-yaml">Désérialisation non sécurisée PyYAML (/unsafe-yaml)</a></li>
    </ul>
    """

# ----------------- VULNÉRABILITÉ SAST/DAST : INJECTION SQL -----------------
@app.route('/login', methods=['GET'])
def login():
    username = request.args.get('username', '')
    users = get_user_vulnerable(username)
    return f"<h3>Résultats de la recherche utilisateur pour '{username}' :</h3><p>{str(users)}</p>"

# ----------------- VULNÉRABILITÉ SAST/DAST : INJECTION DE COMMANDE -----------------
@app.route('/ping', methods=['GET'])
def ping():
    ip = request.args.get('ip', '')
    # Faille majeure : concaténation directe dans un shell système
    cmd = f"ping -c 1 {ip}" if os.name != 'nt' else f"ping -n 1 {ip}"
    output = os.popen(cmd).read()
    return f"<h3>Résultat du Ping :</h3><pre>{output}</pre>"

# ----------------- VULNÉRABILITÉ SAST/DAST : PATH TRAVERSAL -----------------
@app.route('/read', methods=['GET'])
def read_file():
    filename = request.args.get('file', '')
    # Faille majeure : pas de vérification de chemin
    filepath = os.path.join(os.getcwd(), filename)
    try:
        return send_file(filepath, as_attachment=False)
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {str(e)}", 400

# ----------------- VULNÉRABILITÉ SAST/DAST : XSS REFRACTÉ -----------------
@app.route('/xss', methods=['GET'])
def xss():
    name = request.args.get('name', '')
    # Faille : Rendu de template HTML sans échapper les entrées utilisateur
    template = f"<h3>Bonjour {name} !</h3>"
    return render_template_string(template)

# ----------------- VULNÉRABILITÉ SAST : DESERIALISATION NON SECURISEE -----------------
@app.route('/unsafe-yaml', methods=['GET', 'POST'])
def unsafe_yaml():
    if request.method == 'POST':
        data = request.data
        # Faille PyYAML : Chargement non sécurisé permettant l'exécution de code arbitraire
        parsed = yaml.load(data, Loader=yaml.Loader)
        return f"YAML chargé avec succès : {str(parsed)}"
    return "Envoyez du YAML via POST pour tester."

@app.route('/unsafe-pickle', methods=['POST'])
def unsafe_pickle():
    # Faille Pickle : Désérialisation d'objets arbitraires non sécurisés
    data = request.form.get('data', '')
    decoded = base64.b64decode(data)
    obj = pickle.loads(decoded)
    return f"Pickle désérialisé : {str(obj)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
