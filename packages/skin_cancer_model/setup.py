from setuptools import setup, find_packages
import os

# =========================================================
# Configuration centrale : Nom du package et chemin de version
# =========================================================
PACKAGE_NAME = 'skin_cancer_model' # <-- Définition centrale du nom du package
VERSION_PATH = os.path.join(PACKAGE_NAME, '_version.py')
# =========================================================


# --- Fonction utilitaire : Lire le numéro de version central ---
def get_version(rel_path):
    """ Lit la variable __version__ à partir du chemin spécifié """
    # Remarque : rel_path est relatif au répertoire contenant setup.py
    full_path = os.path.join(os.path.dirname(__file__), rel_path)
    for line in open(full_path, encoding="utf8"):
        if line.startswith('__version__'):
            # Extrait la chaîne de version, supprime les guillemets
            return line.split('=')[1].strip().strip("'\"")
    raise RuntimeError("Impossible de trouver la chaîne de version.")
# -----------------------------------

# Lecture du numéro de version
version = get_version(VERSION_PATH)

setup(
    name=PACKAGE_NAME, 
    version=version,  
    description='API de Prédiction du Cancer de la Peau',
    author='Meiti Hsia',
    # find_packages 的 include 參數現在是動態的
    packages=find_packages(include=[PACKAGE_NAME, f'{PACKAGE_NAME}.*']), 
    install_requires=[
        'Flask',
        'marshmallow',
        'gunicorn',
        'requests',
        'pandas',
        'scikit-learn',
        'Pillow',
        # Verrouillage de version important (解決 Werkzeug 衝突)
        'Werkzeug==2.2.2'
    ],
    include_package_data=True,
    zip_safe=False
)