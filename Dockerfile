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

# Copie du dossier 'packages' entier
COPY packages /app/packages 

# Copie du fichier de lancement de l'API 
COPY packages/skin_cancer_model/app.py /app/app.py 

# CRUCIAL: Utilisation de joker (*) pour copier le fichier de poids du modèle (.pt)
COPY packages/skin_cancer_model/skin_cancer_model/*.pt /app/packages/skin_cancer_model/skin_cancer_model/

# 5. Installation du package local: Correction du chemin
# Important: Installe le package en spécifiant le répertoire où se trouve setup.py.
RUN pip install ./packages/skin_cancer_model

# 6. Exposition du port: 
EXPOSE 8000

# 7. Commande de démarrage: 
CMD ["python", "app.py"]