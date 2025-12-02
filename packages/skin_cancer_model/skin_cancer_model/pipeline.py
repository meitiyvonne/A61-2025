# packages/skin_cancer_model/skin_cancer_model/pipeline.py

import torch.nn as nn
import torch
from torchvision import models

# Nombre de classes pour la classification du cancer de la peau (HAM10000: 7 classes)
NUM_CLASSES = 7 

# --- 1. Définition de l'architecture du Modèle ---
class SkinCancerModel(nn.Module):
    """Architecture du modèle de classification (ex: ResNet-18 adapté)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super(SkinCancerModel, self).__init__()
        
        # Chargement d'un modèle pré-entraîné (ResNet-18)
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remplacement de la dernière couche fully-connected (FC)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Si vous utilisez également les métadonnées (localization/age/sex), ajoutez ici des couches
        
    def forward(self, image_tensor):
        """Passe avant (Forward pass) du modèle."""
        return self.base_model(image_tensor)


# --- 2. Pipeline de prédiction pour l'inférence ---
class ModelPipeline:
    """Classe pour encapsuler les opérations de chargement et de prédiction du modèle."""
    def __init__(self, model_instance=None, model_path=None):
        """
        Initialise le pipeline en chargeant un modèle si un chemin est fourni.
        """
        self.model = model_instance if model_instance is not None else SkinCancerModel()
        
        if model_path:
            # Chargement des poids du modèle sauvegardé (best_model.pt)
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval() # Définit le modèle en mode évaluation

    def predict(self, data_input):
        """
        Effectue l'inférence (prédiction) sur un tenseur d'entrée prétraité.
        """
        with torch.no_grad():
            # Ajoute une dimension de lot (batch) si l'entrée est une seule image
            input_tensor = data_input.float().unsqueeze(0) 
            
            # Obtient la sortie du modèle (logits)
            output = self.model(input_tensor)
            
            # Convertit les logits en probabilités
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Obtient l'indice de la classe prédite
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            return predicted_class, probabilities.numpy()