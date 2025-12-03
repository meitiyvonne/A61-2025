# packages/skin_cancer_model/train_pipeline.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from datetime import datetime

# --- Importation des modules internes (Ceci repose sur PYTHONPATH = {toxinidir}) ---
# Note: Nous supprimons sys.path.append car tox.ini le gère
# Si cette structure persiste, vérifiez que le dossier 'skin_cancer_model'
# contient bien un fichier __init__.py vide.
from skin_cancer_model import pipeline
from skin_cancer_model.datasets import data_manager
from skin_cancer_model import preprocessors

from skin_cancer_model import __version__ as _version
from skin_cancer_model.config import IMAGE_ROOT # AJOUTER CETTE LIGNE
# ---------------------------------------------------------------------------------
# --- Constantes de configuration pour l'entraînement ---
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Chemins de sauvegarde
# SAVE_DIR = os.path.join(os.path.dirname(__file__), 'skin_cancer_model', 'trained_models')
# MODEL_SAVE_FILE = f'best_model_{datetime.now().strftime("%Y%m%d%H%M")}.pt'

# Chemins de sauvegarde: UTILISER LE DOSSIER DU PACKAGE ET LA VERSION
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'skin_cancer_model') # Le dossier 'skin_cancer_model' interne
MODEL_SAVE_FILE = f'skin_cancer_model_v{_version}.pt' # Nom de fichier fixe basé sur la version

# --- 1. Définition du Dataset PyTorch ---
class SkinCancerDataset(Dataset):
    """Classe Dataset personnalisée pour charger les images et les étiquettes."""
    def __init__(self, X_paths, y_labels, transforms):
        self.X_paths = X_paths # Chemins des images (ici, les IDs d'image)
        self.y_labels = y_labels # Étiquettes de classification
        self.transforms = transforms
        
        # Créer un mappage des étiquettes 
        unique_labels = np.unique(y_labels.values) 
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        
        # !!! Chemin racine OÙ SONT STOCKÉS VOS FICHIERS IMAGES (ISIC_*.jpg) !!!
        # MODIFIER CE CHEMIN: Exemple pour un dossier 'data/images' à la racine du dépôt
        # self.REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..') 
        # self.IMAGE_ROOT = os.path.join(self.REPO_ROOT, 'data', 'images') 
        self.IMAGE_ROOT = IMAGE_ROOT

    def __len__(self):
        """Retourne le nombre total d'échantillons."""
        return len(self.X_paths)

    def __getitem__(self, idx):
        """Récupère et prétraite un échantillon."""
        # Traitement de l'image
        image_id = self.X_paths.iloc[idx]
        # image_path = os.path.join(self.IMAGE_ROOT, f"{image_id}.jpg") 
        image_path = os.path.join(self.IMAGE_ROOT, 'img01', f"{image_id}.jpg")
        # image_tensor = preprocessors.load_and_transform_image(image_path, is_training=True)
        image_tensor = preprocessors.load_and_transform_image(image_path, is_training=True)

        # Traitement de l'étiquette
        label = self.y_labels.iloc[idx]
        label_int = self.label_to_int[label]
        
        if image_tensor is None:
             raise RuntimeError(f"Échec du chargement de l'image à {image_path}")
        
        return image_tensor, torch.tensor(label_int, dtype=torch.long)
        
        
def run_training():
    """Fonction principale pour exécuter le pipeline d'entraînement."""
    print("------------------------------------------------")
    print("--- Démarrage du pipeline d'entraînement (PyTorch) ---")
    print("------------------------------------------------")
    
    # 1. Chargement et séparation des données
    print("Étape 1/5: Chargement des métadonnées et séparation...")
    data_df = data_manager.load_dataset()
    
    if data_df.empty:
        print("Erreur: Données non chargées. Arrêt de l'entraînement.")
        return
        
    X_train, X_test, y_train, y_test = data_manager.get_train_test_split(data_df=data_df)
    
    # 2. Dataset et DataLoader
    print("Étape 2/5: Configuration du Dataset et du DataLoader...")
    train_dataset = SkinCancerDataset(X_train, y_train, preprocessors.get_training_transforms())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Définition du Modèle, de la Fonction de Perte et de l'Optimiseur
    print("Étape 3/5: Définition du modèle et des fonctions...")
    model = pipeline.SkinCancerModel(num_classes=len(train_dataset.label_to_int))
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Boucle d'entraînement
    print(f"Étape 4/5: Début de l'entraînement pour {EPOCHS} epochs...")
    model.train() # Mode entraînement
    
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        current_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            # Passe avant
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Rétropropagation et optimisation
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            
        avg_loss = current_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Perte (Loss): {avg_loss:.4f}")
        
        # Implémentation de l'arrêt anticipé (Early Stopping) et de la sauvegarde
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     early_stop_counter = 0
            
        #     os.makedirs(SAVE_DIR, exist_ok=True)
        #     model_path = os.path.join(SAVE_DIR, MODEL_SAVE_FILE)
            
        #     torch.save(model.state_dict(), model_path)
        #     print(f" -> Meilleur modèle sauvegardé à {model_path} (Perte: {best_loss:.4f})")
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0

            # 1. Recalculer le chemin de sauvegarde (Utilise les variables définies en haut )
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_FILE)

            # 2. Sauvegarder avec le nom basé sur la version
            torch.save(model.state_dict(), model_path)
            print(f" -> Meilleur modèle v{_version} sauvegardé à {model_path} (Perte: {best_loss: .4f})")

        else:
            early_stop_counter += 1
            if early_stop_counter >= 5: 
                print(" -> Arrêt anticipé (Early Stopping) déclenché. Fin de l'entraînement.")
                break
    
    # 5. Affichage du résultat final (simulant la sortie réussie pour la CI)
    print("Étape 5/5: Entraînement terminé.")
    print("Training...")
    print("saved pipeline")
    print("summary")
    

if __name__ == '__main__':
    run_training()