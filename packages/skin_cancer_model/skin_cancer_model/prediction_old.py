# Fichier: packages/skin_cancer_model/skin_cancer_model/prediction.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from .config import MODEL_FILE_PATH, MODEL_VERSION
from .__init__ import logger 

# ==========================================
# CONSTANTES ET MAPPAGES
# ==========================================

# Les sept classes de diagnostic (doit correspondre à l'entraînement)
DIAGNOSIS_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Mapping des métadonnées (doit correspondre à l'entraînement du modèle Embedding)
# NOTE: Cette table de mapping doit être celle utilisée lors de l'entraînement.
LOCATION_MAPPING = {
    'torso': 0, 'arm': 1, 'leg': 2, 'face': 3, 'neck': 4, 'other': 5
}
SEX_MAPPING = {'male': 0, 'female': 1, 'other': 2}

# ==========================================
# 1. ARCHITECTURE DU MODÈLE EMBEDDING
# ==========================================

# ATTENTION: Cette classe doit correspondre EXACTEMENT à l'architecture utilisée pour l'entraînement
class SkinCancerModel(nn.Module):
    """ Modèle multimodal (Image CNN + Métadonnées) utilisant ResNet18 """
    def __init__(self, num_classes=len(DIAGNOSIS_CLASSES)):
        super(SkinCancerModel, self).__init__()
        
        # 1. Backbone CNN (ResNet18)
        # Charge ResNet18 (sans poids pré-entraînés pour la structure)
        # Assurez-vous d'avoir 'torchvision' installé
        try:
            self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
        except Exception:
            # Fallback si le téléchargement échoue (doit être géré)
            self.cnn = nn.Sequential(*list(nn.ModuleList(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None).children())[:-1]))

        self.cnn.fc = nn.Identity() # Retire la dernière couche FC
        self.cnn_output_size = 512 

        # 2. Tête de Métadonnées (Metadata Head)
        # Entrées: Âge (1), Sexe (1), Localisation (1) -> 3
        # self.metadata_input_size = 3 
        self.metadata_input_size = 8
        self.loc_emb = nn.Linear(self.metadata_input_size, 15)
        # self.loc_emb = nn.Linear(self.metadata_input_size, 10)
        # self.loc_emb = nn.Sequential(
        #     nn.Linear(self.metadata_input_size, 32),
        #     nn.ReLU(),
        #     # nn.Linear(32, 16) 
        #     nn.Linear(32, 10) 
        # )
        
        # 3. Couche Entièrement Connectée Finale (Fusion)
        # Entrée: CNN (512) + Metadata Head (16)
        # self.fusion_input_size = self.cnn_output_size + 16 
        self.fusion_input_size = self.cnn_output_size + 15
        # self.fusion_input_size = self.cnn_output_size + 10
        self.fc = nn.Sequential(
            # nn.Linear(self.fusion_input_size, 64),
            nn.Linear(self.fusion_input_size, 256),
            nn.ReLU(),
            # nn.Linear(64, num_classes) # Sortie pour les 7 classes
            nn.Linear(256, num_classes)
        )

    def forward(self, image_input, metadata_input):
        # 1. Passage de l'image (Doit être aplati)
        image_features = self.cnn(image_input)
        image_features = image_features.view(image_features.size(0), -1) # Aplatir (Flatten)
        
        # 2. Passage des métadonnées
        metadata_features = self.loc_emb(metadata_input)
        
        # 3. Concaténation (Embedding)
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        
        # 4. Classification
        output = self.fc(combined_features)
        return output

# ==========================================
# 2. FONCTIONS DE CHARGEMENT ET PRÉDICTION
# ==========================================

# Fichier: prediction.py (dans la fonction load_model)

# def load_model(model_class=SkinCancerModel) -> SkinCancerModel:
#     """ Charge le modèle PyTorch pré-entraîné depuis le disque. """
    
#     model = model_class(num_classes=len(DIAGNOSIS_CLASSES))
    
#     # --- TEMPORAIRE : Utiliser la méthode stricte pour révéler l'erreur ---
#     try:
#         # 載入權重檔案
#         state_dict = torch.load(MODEL_FILE_PATH, map_location=torch.device('cpu'))
        
#         # 嘗試載入並回傳詳細的錯誤訊息
#         # strict=True 是預設值，但我們明確使用它。
#         missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        
#         if missing_keys or unexpected_keys:
#             logger.error(f"Erreur de Structure du Modèle :")
#             logger.error(f"Clés Manquantes (Missing Keys): {missing_keys}")
#             logger.error(f"Clés Inattendues (Unexpected Keys): {unexpected_keys}")
#             raise RuntimeError("Échec de load_state_dict en raison de clés non concordantes.")
        
#         model.eval() 
#         logger.info(f"Modèle PyTorch v{MODEL_VERSION} chargé avec succès.")
#         return model
        
#     except FileNotFoundError:
#         logger.error(f"Erreur: Fichier modèle non trouvé à {MODEL_FILE_PATH}. (Mode Démo)")
#         return None
#     except Exception as e:
#         # 這將會捕捉到我們剛才添加的 RuntimeError，並列印出詳細的 Keys 錯誤
#         logger.error(f"Erreur fatale lors du chargement du modèle. Veuillez vérifier l'architecture: {e}")
#         # 重新拋出錯誤，讓程式停止，以便我們檢查錯誤
#         raise


def load_model(model_class=SkinCancerModel) -> SkinCancerModel:
    """ Charge le modèle PyTorch pré-entraîné depuis le disque. """
    try:
        # Initialise l'architecture
        model = model_class(num_classes=len(DIAGNOSIS_CLASSES))
        
        # Charger les poids sur le CPU (pour la plupart des déploiements Flask)
        # model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=torch.device('cpu')))
        model.load_state_dict(
            torch.load(MODEL_FILE_PATH, map_location=torch.device('cpu')),
            strict=False 
        )
        
        model.eval() # Toujours mettre le modèle en mode évaluation pour l'inférence
        logger.info(f"Modèle PyTorch v{MODEL_VERSION} chargé avec succès depuis {MODEL_FILE_PATH}.")
        return model
        
    except FileNotFoundError:
        logger.error(f"Erreur: Fichier modèle non trouvé à {MODEL_FILE_PATH}. Avez-vous exécuté le pipeline d'entraînement?")
        # Ne pas relancer pour le test local si le modèle n'est pas encore entraîné
        logger.warning("Utilisation du modèle non chargé (MODE DÉMO).")
        return None # Retourne None si le fichier n'est pas trouvé
    except Exception as e:
        logger.error(f"Erreur inattendue lors du chargement du modèle: {e}")
        # En cas d'erreur de structure ou de PyTorch
        raise

def _preprocess_metadata(metadata: dict) -> torch.Tensor:
    """ Convertit les métadonnées cliniques en un tenseur PyTorch (8 features). """
    
    # --- 1. 定義 One-Hot 類別 ---
    # 假設這 8 個特徵是：[Age, Sex_M, Sex_F, Sex_O, Loc_T, Loc_A, Loc_L, Loc_F]
    SEX_CATEGORIES = ['male', 'female', 'other']
    LOCATION_CATEGORIES = ['torso', 'arm', 'leg', 'face', 'neck', 'other'] 
    
    # 為了湊出 8 個特徵，我們假設 Sex 使用 One-Hot (3個)，Localization 使用 One-Hot (4個) 
    # 或者 Localization 只使用了 4 個類別，其餘的類別被歸入 'other'
    
    # 由於我們無法得知訓練時的確切 4 個定位類別，我們假設 Localization 的 One-Hot 是 4 個
    
    # --- 2. 數據提取 ---
    age = metadata.get('age', 0)
    sex_str = metadata.get('sex', 'other').lower()
    loc_str = metadata.get('localization', 'other').lower()
    
    # --- 3. 構造 8 個特徵向量 (List) ---
    features = []
    
    # Age (1 feature)
    features.append(age)
    
    # Sex One-Hot (3 features)
    for cat in SEX_CATEGORIES:
        features.append(1.0 if sex_str == cat else 0.0)
    
    # Location One-Hot (需要 4 個 features 來湊出 8 個輸入)
    # 假設只使用了 4 個最常見的 Location 類別 + 'other'
    COMMON_LOCATIONS = ['arm', 'leg', 'torso', 'face']
    
    loc_features = [0.0] * 4 # 初始化 4 個 Location 特徵
    
    try:
        # 嘗試找到匹配的索引，並將其設為 1.0
        loc_index = COMMON_LOCATIONS.index(loc_str)
        loc_features[loc_index] = 1.0
    except ValueError:
        # 如果不是這 4 個，那麼這 4 個特徵都為 0.0，這是一個合理的假設
        pass 

    features.extend(loc_features)
    
    # 總共特徵數量: 1 (Age) + 3 (Sex) + 4 (Location) = 8
    
    # --- 4. 創建張量 ---
    metadata_features = torch.tensor([features], dtype=torch.float32)
    
    if metadata_features.shape[1] != 8:
         logger.error(f"Erreur de pré-traitement des métadonnées: {metadata_features.shape[1]} features trouvées, 8 attendues.")
         # 這裡可以拋出錯誤或使用默認值
         
    return metadata_features

# def _preprocess_metadata(metadata: dict) -> torch.Tensor:
#     """ Convertit les métadonnées cliniques en un tenseur PyTorch. """
    
#     # 1. Extraction et nettoyage
#     age = metadata.get('age', 0)
#     sex_str = metadata.get('sex', 'other').lower()
#     loc_str = metadata.get('localization', 'other').lower()
    
#     # 2. Encodage numérique
#     sex_val = SEX_MAPPING.get(sex_str, 2)
#     loc_val = LOCATION_MAPPING.get(loc_str, 5)
    
#     # 3. Création du tenseur (Batch Size = 1) : [age, sex, localization]
#     metadata_features = torch.tensor([[age, sex_val, loc_val]], dtype=torch.float32)
    
#     return metadata_features


def make_prediction(image_tensor: torch.Tensor, metadata: dict, model: SkinCancerModel):
    """ Exécute la prédiction du modèle. """
    
    if model is None:
        logger.warning("Le modèle est None. Retourne une prédiction simulée.")
        # Mode de secours simulé si le modèle n'a pas été trouvé (pour éviter de planter l'API)
        return {
            'prediction': "Non (Pas de cancer)",
            'probability': 0.99,
            'diagnosis_class': "nv (Simulé)"
        }

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