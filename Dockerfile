# =====================================================================
# Dockerfile — Prédiction de Faillite d'Entreprise
# =====================================================================

# 1. IMAGE DE BASE
# Python 3.12 en version "slim" : système minimal, moins de poids
# que l'image Python complète, mais suffisant pour nos besoins.
FROM python:3.12-slim

# 2. DÉPENDANCE SYSTÈME REQUISE PAR LIGHTGBM
# LightGBM s'appuie sur OpenMP (libgomp1) pour paralléliser ses calculs.
# Sans cette librairie système, l'import de lightgbm échoue au runtime.
# On supprime ensuite le cache apt (rm -rf /var/lib/apt/lists/*) pour
# ne pas alourdir l'image avec des fichiers inutiles après installation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. DOSSIER DE TRAVAIL
# Toutes les commandes suivantes (COPY, RUN, CMD) s'exécutent
# à partir de /app à l'intérieur du conteneur.
WORKDIR /app

# 4. INSTALLATION DES DÉPENDANCES PYTHON (étape mise en cache)
# On copie UNIQUEMENT requirements.txt avant le reste du code.
# Tant que ce fichier ne change pas, Docker réutilise le cache de
# cette étape pip install lors des prochains builds → gain de temps.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPIE DU RESTE DU PROJET
# app.py, templates/, static/, models/, etc.
# Cette étape arrive après pip install pour profiter du cache ci-dessus :
# modifier app.py ne force pas à réinstaller toutes les dépendances.
COPY . .

# 6. PORT EXPOSÉ
# Documente que l'application écoute sur le port 5000.
# Ne fait rien de "magique" tout seul : c'est l'option -p de
# "docker run" (ou le docker-compose.yml) qui relie réellement
# ce port à un port de la machine hôte.
EXPOSE 5000

# 7. COMMANDE DE DÉMARRAGE
# Gunicorn (serveur WSGI de production) sert l'application Flask
# définie dans app.py via l'objet "app" (app:app).
#   --bind 0.0.0.0:5000   → écoute sur toutes les interfaces, port 5000
#   --workers 2           → 2 processus pour gérer les requêtes en parallèle
#   --timeout 120         → 120s avant de tuer une requête trop longue
#                           (utile ici car la génération du graphique SHAP
#                           peut prendre du temps)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
