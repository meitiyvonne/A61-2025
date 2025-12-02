# packages/skin_cancer_model/skin_cancer_model/datasets/data_manager.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- CONSTANTES DE CHEMINS ---
# Le chemin est ajusté pour pointer vers <repo_root>/data/HAM10000_metadata.csv
# REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..') 
# DATA_FILE = 'HAM10000_metadata.csv'
# DATA_PATH = os.path.join(REPO_ROOT, 'data', DATA_FILE)
import pathlib
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent # remonter trois niveaux pour atteindre la racine du paquet (A61-2025)
DATA_PATH = PACKAGE_ROOT / 'data' / 'HAM10000_metadata.csv'


# Nom de la colonne cible pour la classification
TARGET = 'dx' 
RANDOM_SEED = 42

def load_dataset(*, data_file=DATA_PATH):
    """Charge le fichier CSV des métadonnées HAM10000 et retourne un DataFrame."""
    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Erreur: Fichier de données non trouvé à {data_file}")
        return pd.DataFrame() 
    
    # Des étapes initiales de nettoyage ou de vérification pourraient être ajoutées ici
    
    return data


def get_train_test_split(*, data_df, test_size=0.2, random_state=RANDOM_SEED):
    """Sépare le DataFrame en ensembles d'entraînement et de test (échantillonnage stratifié)."""
    
    # 'image_id' est utilisé pour les features (X)
    X = data_df['image_id'] 
    y = data_df[TARGET] # 'dx' est l'étiquette de classification
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y # Stratification pour garantir une distribution de classe similaire
    )
    
    return X_train, X_test, y_train, y_test