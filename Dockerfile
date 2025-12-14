# ------------------------------------
# Fichier: Dockerfile
# Position: Racine du projet (A61-2025/).
# ------------------------------------

# 1. Base Image: 
FROM python:3.12-slim 

# 2. Répertoire de travail: 
WORKDIR /app

# --- Étape 3: Copie des fichiers essentiels pour l'installation ---
COPY packages/skin_cancer_model/requirements.txt /app/
COPY packages/skin_cancer_model/setup.py /app/

# 3. Installation des dépendances et outils de build.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Étape 4: Copie du code source et du modèle ---

# 【CORRECTIF MAJEUR】Copie l'intégralité du répertoire du projet (y compris packages/ et templates/)
# Ceci garantit que toutes les sous-structures de dossiers sont préservées,
# ce qui corrige à la fois l'erreur TemplateNotFound et l'erreur du modèle non trouvé.
COPY . /app

# Copie du dossier 'packages' entier
COPY packages /app/packages 

# NOUVEAU: COPIE EXPLICITE DES TEMPLATES À LA RACINE DU WORKDIR
# C'est la ligne CRUCIALE qui corrige l'erreur TemplateNotFound. 
# Flask trouvera ainsi les templates à /app/templates/.
COPY packages/skin_cancer_model/templates /app/templates

# Copie du fichier de lancement de l'API 
# COPY packages/skin_cancer_model/app.py /app/app.py 

# CRUCIAL: Utilisation de joker (*) pour copier le fichier de poids du modèle (.pt)
COPY packages/skin_cancer_model/skin_cancer_model/*.pt /app/packages/skin_cancer_model/skin_cancer_model/


# 5. Installation du package local: Correction du chemin
# Important: Installe le package en spécifiant le répertoire où se trouve setup.py.
RUN pip install ./packages/skin_cancer_model

# 6. Exposition du port: 
# EXPOSE 8000
EXPOSE 5000

# 7. Commande de démarrage: 
# CMD ["python", "app.py"]
CMD ["python", "/app/packages/skin_cancer_model/app.py"]