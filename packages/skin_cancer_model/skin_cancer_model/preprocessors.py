# packages/skin_cancer_model/skin_cancer_model/preprocessors.py

from torchvision import transforms
from PIL import Image
import torch
import os

# --- Paramètres de prétraitement ---
# La taille d'entrée attendue par votre modèle
IMAGE_SIZE = 224
# Paramètres de normalisation ImageNet standards
NORMALIZATION_MEAN = [0.485, 0.456, 0.406] 
NORMALIZATION_STD = [0.229, 0.224, 0.225] 


def get_training_transforms():
    """Définit le pipeline de transformations avec augmentation des données pour l'entraînement."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Augmentation des données:
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Convertir l'image en Tenseur PyTorch
        transforms.ToTensor(),
        # Normalisation
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])


def get_inference_transforms():
    """Définit le pipeline de transformations sans augmentation pour le test/l'inférence."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Conversion et normalisation
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])


def load_and_transform_image(image_path, is_training=False):
    """
    Charge une image à partir du chemin et applique les transformations appropriées.
    image_path: Chemin complet du fichier image.
    is_training: Si True, applique les transformations d'entraînement.
    """
    if not os.path.exists(image_path):
         # Retourne None si le fichier n'est pas trouvé
        return None 
        
    # Ouvre l'image avec PIL et s'assure qu'elle est en format RGB
    image = Image.open(image_path).convert('RGB')
    
    if is_training:
        transform = get_training_transforms()
    else:
        transform = get_inference_transforms()
        
    return transform(image)