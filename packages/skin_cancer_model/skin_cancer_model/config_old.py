# Fichier: packages/skin_cancer_model/skin_cancer_model/config.py

import pathlib


# Niveau de journalisation
LOGGING_LEVEL = logging.INFO

# --- CONSTANTES DE CHEMINS (GLOBALES ET ROBUSTES) ---

# 1. Obtenir le chemin absolu du fichier config.py
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()

# 2. Remonter 4 niveaux pour atteindre la racine du dépôt (A61-2025)
# (config.py -> skin_cancer_model/ -> skin_cancer_model/ -> packages/ -> A61-2025)
# 4 niveaux car config.py est un niveau de moins que data_manager.py
REPO_ROOT = CURRENT_FILE_PATH.parent.parent.parent.parent

# Dataset
DATA_FILE = 'HAM10000_metadata.csv'

# Img path
IMAGE_ROOT = REPO_ROOT / 'data' / 'images'

# Dataset path
DATA_PATH = IMAGE_ROOT / DATA_FILE 