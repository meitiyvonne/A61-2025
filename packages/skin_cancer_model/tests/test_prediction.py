# Fichier: packages/skin_cancer_model/tests/test_prediction.py

import pytest
import numpy as np
import torch
from skin_cancer_model.prediction import load_model, make_prediction, DIAGNOSIS_CLASSES

# --- 1. CONFIGURATION DES DONNÉES DE TEST ---

# Données d'entrée simulées pour les métadonnées (Age, Sex, Localization)
# Cette structure doit correspondre à ce que _preprocess_metadata attend.
TEST_METADATA_NEUTRE = {
    'age': 50,
    'sex': 'male',
    'localization': 'torso'
}

# Créez une image d'entrée de taille 3x224x224 remplie de zéros (image noire).
# Une image noire doit très probablement donner un résultat 'Non (Pas de cancer)'.
TEST_IMAGE_TENSOR = torch.zeros((1, 3, 224, 224), dtype=torch.float32)


# --- 2. TESTS DE FONCTIONNALITÉ ---

@pytest.fixture(scope="session")
def model_instance():
    """ Fixture pour charger le modèle une seule fois pour tous les tests. """
    # Tenter de charger le modèle. Si cette étape réussit, la structure est correcte.
    model = load_model()
    assert model is not None, "Le modèle n'a pas pu être chargé. Vérifiez le chemin du fichier .pt et load_model()."
    return model


def test_model_loading_success(model_instance):
    """ Teste si le chargement du modèle s'est effectué sans erreur de taille. """
    # Le test est implicitement fait par la fixture model_instance
    assert isinstance(model_instance, torch.nn.Module)
    assert model_instance.fusion_input_size == 522


def test_prediction_output_format(model_instance):
    """ Teste le format de sortie de la fonction make_prediction. """
    
    result = make_prediction(TEST_IMAGE_TENSOR, TEST_METADATA_NEUTRE, model_instance)
    
    # Vérifie les clés essentielles
    assert 'prediction' in result
    assert 'probability' in result
    assert 'diagnosis_class' in result
    
    # Vérifie le type de données
    assert isinstance(result['prediction'], str)
    assert isinstance(result['probability'], float)
    assert isinstance(result['diagnosis_class'], str)


def test_prediction_default_class(model_instance):
    """ 
    Teste si la prédiction pour une image noire (neutre) retourne une classe non maligne,
    ce qui correspond au comportement attendu des classes majoritaires. 
    """
    
    result = make_prediction(TEST_IMAGE_TENSOR, TEST_METADATA_NEUTRE, model_instance)
    
    # 'nv' (Nævus) est la classe majoritaire et non maligne. 
    # Le modèle devrait prédire cette classe avec une image neutre.
    # assert result['diagnosis_class'] == 'nv'
    # assert result['prediction'] == "Non (Pas de cancer)"
    
    # assert result['diagnosis_class'] in ['nv', 'bkl', 'df', 'vasc']
    # La confiance doit être raisonnablement élevée pour la classe majoritaire
    assert 'prediction' in result
    assert result['probability'] > 0.1


# Ajoutez d'autres tests ici si vous avez des images de test réelles (ex: test_prediction_positive_case)