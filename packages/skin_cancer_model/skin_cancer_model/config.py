# Fichier: packages/skin_cancer_model/skin_cancer_model/config.py
import logging
import pathlib
# Importation du fichier de version central (doit être présent dans le même dossier)
from ._version import __version__ as _MODEL_VERSION 

# --- CONSTANTES DE CHEMINS (GLOBALES ET ROBUSTES) ---

# 1. Obtenir le chemin absolu du fichier config.py
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()

# 2. Racine du Dépôt (A61-2025): Remonter 4 niveaux (Logique robuste du TP)
# (config.py -> skin_cancer_model/ -> skin_cancer_model/ -> packages/ -> A61-2025)
REPO_ROOT = CURRENT_FILE_PATH.parent.parent.parent.parent 

# 3. Racine du Package (skin_cancer_model/skin_cancer_model)
PACKAGE_ROOT = CURRENT_FILE_PATH.parent

# ======================================
# Chemins du Dataset et des Images (POUR L'ENTRAÎNEMENT / PIPELINE)
# ======================================
DATA_FILE = 'HAM10000_metadata.csv'
# Utilise REPO_ROOT pour pointer vers le dossier 'data' à la racine du dépôt
IMAGE_ROOT = REPO_ROOT / 'data' / 'images' 
DATA_PATH = IMAGE_ROOT / DATA_FILE # Chemin complet vers le fichier CSV de métadonnées

# ======================================
# Chemins spécifiques à l'API et au Modèle
# ======================================
MODEL_NAME = 'skin_cancer_model_v.0.0.1.pt' # Le nom de votre fichier modèle PyTorch
# Le modèle est stocké dans le package pour faciliter le déploiement
MODEL_FILE_PATH = PACKAGE_ROOT / MODEL_NAME 

# ======================================
# Configuration de la version (Étape 6/11)
# ======================================
API_VERSION = '1.0.0' # Version de l'API (Mise à jour manuelle pour les changements majeurs)
MODEL_VERSION = _MODEL_VERSION # Lecture dynamique depuis _version.py

# ======================================
# Configuration du prétraitement d'image (POUR L'INFÉRENCE API)
# ======================================
IMAGE_SIZE = (224, 224) 
MAX_PIXEL_DIMENSION = 4000 # Largeur/hauteur maximale acceptable en pixels
MAX_FILE_SIZE_MB = 10 # Taille maximale du fichier 10MB
# Paramètres de normalisation (Doivent être les mêmes que ceux utilisés pour l'entraînement)
IMAGE_MEAN = [0.485, 0.456, 0.406] 
IMAGE_STD = [0.229, 0.224, 0.225]

# ======================================
# Configuration des journaux (Logs) (Étape 9)
# ======================================
LOG_FILE_NAME = 'api_log.log'
# Le répertoire des logs est placé à la racine du dépôt (A61-2025/logs)
LOG_DIR = REPO_ROOT / 'logs' 
LOG_DIR.mkdir(exist_ok=True) 

# Niveau de journalisation
LOGGING_LEVEL = logging.INFO