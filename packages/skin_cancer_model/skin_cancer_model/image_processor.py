import io
from PIL import Image
import torch
from torchvision import transforms
from werkzeug.datastructures import FileStorage
import os

from .config import (
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, 
    MAX_PIXEL_DIMENSION, MAX_FILE_SIZE_MB
)
from .__init__ import logger

# Pipeline de transformation d'image (Doit être identique à l'entraînement)
def get_inference_transform():
    """ Obtient le pipeline de transformation d'image pour l'inférence """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # Redimensionnement à 224x224
        transforms.ToTensor(),         # Conversion en tenseur PyTorch
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD) # Normalisation
    ])

def validate_and_process_image(image_file: FileStorage) -> torch.Tensor:
    """
    Effectue les vérifications de sécurité, le formatage et la normalisation de l'image.
    Lève une ValueError si les vérifications échouent.

    Args:
        image_file: Objet FileStorage téléversé par Flask.

    Returns:
        torch.Tensor: Tenseur normalisé prêt pour le modèle.
    """
    
    # 1. Vérification de la taille du fichier (MB)
    image_file.seek(0, os.SEEK_END)
    file_size_mb = image_file.tell() / (1024 * 1024)
    image_file.seek(0) # Réinitialise le pointeur après la lecture
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"La taille du fichier ({file_size_mb:.2f} Mo) dépasse la limite de {MAX_FILE_SIZE_MB} Mo.")

    try:
        # 2. Lecture sécurisée et vérification du format (PIL/Pillow)
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        # Capture les erreurs de corruption d'image ou de format non pris en charge
        raise ValueError("Impossible de lire l'image ou format non pris en charge (JPG/PNG seulement).")
    
    # 3. Vérification de la dimension en pixels (pour éviter la saturation de la mémoire)
    width, height = image.size
    if width > MAX_PIXEL_DIMENSION or height > MAX_PIXEL_DIMENSION:
        raise ValueError(f"La dimension de l'image est trop grande ({width}x{height}), dépassant la limite de {MAX_PIXEL_DIMENSION}x{MAX_PIXEL_DIMENSION}.")
        
    # 4. Exécution de la transformation de normalisation
    transform = get_inference_transform()
    tensor_image = transform(image)
    
    # 5. Ajustement de la dimension (Ajout de la dimension de Batch)
    tensor_image = tensor_image.unsqueeze(0) 
    
    logger.info("Vérification et normalisation de l'image réussies.")
    return tensor_image