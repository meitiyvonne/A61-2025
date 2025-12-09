# Fichier: packages/skin_cancer_model/skin_cancer_model/prediction.py

import torch
import torch.nn as nn
import numpy as np
import os

from .config import MODEL_FILE_PATH, MODEL_VERSION
from .__init__ import logger 

# ==========================================
# CONSTANTES ET MAPPAGES
# ==========================================

# Les sept classes de diagnostic
DIAGNOSIS_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
SEX_CATEGORIES = ['male', 'female', 'other']
COMMON_LOCATIONS = ['arm', 'leg', 'torso', 'face'] 

# ==========================================
# 1. ARCHITECTURE DU MODÈLE EMBEDDING
# ==========================================

class SkinCancerModel(nn.Module):
    """ Modèle multimodal (Image CNN + Métadonnées) utilisant ResNet18 """
    def __init__(self, num_classes=len(DIAGNOSIS_CLASSES)):
        super(SkinCancerModel, self).__init__()
        
        # 1. Backbone CNN (ResNet18)
        try:
            self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
        except Exception:
            self.cnn = nn.Sequential(*list(nn.ModuleList(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None).children())[:-1]))

        self.cnn.fc = nn.Identity() 
        self.cnn_output_size = 512 

        # 2. Tête de Métadonnées (Metadata Head)
        self.metadata_input_size = 8 
        
        self.loc_emb = nn.Linear(self.metadata_input_size, 15) 
        self.metadata_relu1 = nn.ReLU() 
        
        self.meta_proj = nn.Linear(15, 10) 
        self.metadata_relu2 = nn.ReLU() 
        
        # 3. Couche Entièrement Connectée Finale (Fusion)
        self.fusion_input_size = self.cnn_output_size + 10 
        
        # AJOUT DE DROPOUT: Essentiel pour la régularisation dans le classificateur final
        self.fc = nn.Sequential(
            nn.Linear(self.fusion_input_size, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.5), # 👈 Ajout de Dropout
            nn.Linear(256, num_classes)
        )

    def forward(self, image_input, metadata_input):
        # 1. Passage de l'image
        image_features = self.cnn(image_input)
        image_features = image_features.view(image_features.size(0), -1) 
        
        # 2. Passage des métadonnées (Metadata Path)
        metadata_features = self.loc_emb(metadata_input) 
        metadata_features = self.metadata_relu1(metadata_features) 
        metadata_features = self.meta_proj(metadata_features) 
        metadata_features = self.metadata_relu2(metadata_features) 
        
        # 3. Concaténation (Embedding)
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        
        # 4. Classification
        output = self.fc(combined_features)
        return output

# ==========================================
# 2. FONCTIONS DE CHARGEMENT ET PRÉDICTION
# ==========================================

def _preprocess_metadata(metadata: dict) -> torch.Tensor:
    """ Convertit les métadonnées cliniques en un tenseur PyTorch (8 features). """
    
    # --- 1. Extraction et Nettoyage ---
    age = metadata.get('age', 0)
    sex_str = metadata.get('sex', 'other').lower()
    loc_str = metadata.get('localization', 'other').lower()
    
    # --- 2. Construction des 8 features (List) ---
    features = []
    
    # Age (1 feature)
    features.append(float(age)) 
    
    # Sex One-Hot (3 features)
    for cat in SEX_CATEGORIES:
        features.append(1.0 if sex_str == cat else 0.0)
    
    # Location One-Hot (4 features)
    loc_features = [0.0] * 4 
    try:
        loc_index = COMMON_LOCATIONS.index(loc_str)
        loc_features[loc_index] = 1.0
    except ValueError:
        pass 

    features.extend(loc_features)
    
    # --- 3. Validation et Création du tenseur ---
    if len(features) != 8:
         logger.error(f"Erreur de pré-traitement: {len(features)} features trouvées, 8 attendues.")
         
    metadata_features = torch.tensor([features], dtype=torch.float32)
    
    return metadata_features


def load_model(model_class=SkinCancerModel) -> SkinCancerModel:
    """ Charge le modèle PyTorch pré-entraîné depuis le disque。 """
    try:
        model = model_class(num_classes=len(DIAGNOSIS_CLASSES))
        
        # 加載完整的 state_dict
        state_dict = torch.load(MODEL_FILE_PATH, map_location=torch.device('cpu'))

        # Utiliser strict=False pour ignorer les clés manquantes (meta_proj) 
        model.load_state_dict(
            state_dict,
            strict=False 
        )
        
        model.eval() 
        logger.info(f"Modèle PyTorch v{MODEL_VERSION} chargé avec succès (Meta-proj initialisé aléatoirement).")
        return model
        
    except FileNotFoundError:
        logger.error(f"Erreur: Fichier modèle non trouvé à {MODEL_FILE_PATH}. Avez-vous exécuté le pipeline d'entraînement?")
        logger.warning("Utilisation du modèle non chargé (MODE DÉMO).")
        return None 
    except Exception as e:
        logger.error(f"Erreur inattendue lors du chargement du modèle: {e}")
        raise

def make_prediction(image_tensor: torch.Tensor, metadata: dict, model: SkinCancerModel):
    """ Exécute la prédiction du modèle。 """
    
    if model is None:
        logger.warning("Le modèle est None. Retourne une prédiction simulée。")
        return {
            'prediction': "Non (Pas de cancer)",
            'probability': 0.99,
            'diagnosis_class': "nv (Simulé)"
        }
        
    # --- CLÉ POUR LA PRÉCISION: Assurer le mode évaluation strict à chaque appel ---
    # Cela désactive Dropout, et empêche Batch Norm de se mettre à jour
    model.eval() 

    # 1. Préparation des métadonnées
    metadata_tensor = _preprocess_metadata(metadata)

    # 2. Inférence 
    with torch.no_grad():
        output_logits = model(image_tensor, metadata_tensor)
    
    # 3. Calcul des probabilités (Softmax)
    probabilities = torch.softmax(output_logits, dim=1).squeeze(0) 

    # 4. Obtenir la prédiction
    predicted_index = torch.argmax(probabilities).item()
    predicted_diagnosis = DIAGNOSIS_CLASSES[predicted_index]
    
    # 5. Calculer la probabilité maximale
    max_probability = probabilities[predicted_index].item()
    
    # 6. Détermination 'Oui'/'Non' Cancer de la peau (Maligne: mel, bcc, akiec)
    is_cancer = predicted_diagnosis in ['mel', 'bcc', 'akiec']
    
    return {
        'prediction': "Oui (Cancer de la peau)" if is_cancer else "Non (Pas de cancer)",
        'probability': max_probability,
        'diagnosis_class': predicted_diagnosis
    }