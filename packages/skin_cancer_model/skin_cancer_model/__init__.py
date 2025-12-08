# Fichier: packages/skin_cancer_model/skin_cancer_model/__init__.py
# Ce fichier est l'entrée du package et est utilisé pour initialiser la journalisation.

import logging
from .config import LOG_DIR, LOG_FILE_NAME, LOGGING_LEVEL


# ======================================
# Configuration de la version (FIX NÉCESSAIRE pour la CI)
# ======================================

# Importe la variable __version__ depuis le module interne _version.py
# Ceci permet aux autres modules (comme train_pipeline.py) d'utiliser
# 'from skin_cancer_model import __version__'
try:
    from ._version import __version__  
except ImportError:
    # Cas de secours
    __version__ = "0.0.0"

# ======================================
# Configuration de la journalisation (Logs)
# ======================================

def get_logger(logger_name):
    """ Configure et retourne une instance de logger. """
    
    # Définit le format du journal
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Gestionnaire de fichiers (Écrit dans le fichier api_log.log)
    file_handler = logging.FileHandler(LOG_DIR / LOG_FILE_NAME)
    file_handler.setFormatter(formatter)
    
    # Gestionnaire de console (Affiche également dans la console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Obtient l'instance du logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOGGING_LEVEL)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False # Empêche la double journalisation
    
    return logger

# Obtient le logger principal du package
logger = get_logger(__name__)
# logger.info(f"Journaliseur du package {__name__} initialisé.")
logger.info(f"Journaliseur du package {__name__} initialisé (v{__version__}).")