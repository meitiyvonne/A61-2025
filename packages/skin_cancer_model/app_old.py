from flask import Flask, request, jsonify, render_template, redirect, url_for
from marshmallow import ValidationError
from werkzeug.datastructures import FileStorage
import json
import torch
import numpy as np 

from skin_cancer_model.config import API_VERSION, MODEL_VERSION, MODEL_FILE_PATH
from skin_cancer_model.schemas import SkinCancerPredictionSchema
from skin_cancer_model.__init__ import logger 
from skin_cancer_model.image_processor import validate_and_process_image 
# from skin_cancer_model.prediction import make_prediction # Importer ici votre fonction de prédiction réelle


# ==========================================
# Utilitaires de Modèle Simulé
# ==========================================

# Les sept classes de diagnostic (Veuillez vérifier et ajuster selon votre PPT)
DIAGNOSIS_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def load_model():
    """ Simule le chargement du modèle PyTorch """
    logger.info(f"Tentative de chargement du modèle: {MODEL_FILE_PATH}")
    # En production, ce serait :
    # model = torch.load(MODEL_FILE_PATH)
    # model.eval() 
    return "Modele_Embedding_Simule"

def make_simulated_prediction(image_tensor: torch.Tensor, metadata: dict):
    """ Simule l'inférence du modèle (Sortie cohérente avec un modèle PyTorch) """
    
    # Logique pour simuler la sortie d'un tenseur (1, 7)
    # Simulation d'un cas malin (Melanoma) pour les personnes de plus de 60 ans
    if metadata.get('age', 0) > 60:
        # Haute probabilité pour 'mel' (index 4)
        simulated_output = np.array([0.01, 0.05, 0.02, 0.01, 0.85, 0.05, 0.01])
    else:
        # Haute probabilité pour 'nv' (index 5 - Naevus, bénin)
        simulated_output = np.array([0.01, 0.01, 0.02, 0.05, 0.05, 0.85, 0.01])
        
    probabilities = simulated_output / np.sum(simulated_output)
    predicted_class_index = np.argmax(probabilities)
    
    # Vérification si le diagnostic est malin (Melanoma, BCC, AKIEC)
    predicted_diagnosis = DIAGNOSIS_CLASSES[predicted_class_index]
    
    # Détermination 'Oui'/'Non' Cancer de la peau
    is_cancer = predicted_diagnosis in ['mel', 'bcc', 'akiec']
    
    return {
        'prediction': "Oui (Cancer de la peau)" if is_cancer else "Non (Pas de cancer)",
        'probability': probabilities[predicted_class_index],
        'diagnosis_class': predicted_diagnosis
    }

# ==========================================
# Application Flask
# ==========================================

MODEL = load_model() # Chargement du modèle au démarrage

def create_app():
    app = Flask(__name__, template_folder='templates') 
    
    @app.route('/', methods=['GET'])
    def index():
        """ Page d'accueil : Affiche le formulaire de téléversement d'image """
        return render_template('index.html')

    @app.route('/version', methods=['GET'])
    def version():
        """ Affiche les versions de l'API et du modèle (Étape 11) """
        return jsonify({
            'api_version': API_VERSION,
            'model_version': MODEL_VERSION
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """ Traite la requête de prédiction avec image et métadonnées (Étape 10) """
        
        # 1. Vérification du fichier
        if 'image' not in request.files:
            return render_template('index.html', error='Veuillez téléverser un fichier image.')
        
        image_file: FileStorage = request.files['image']
        
        # 2. Préparation des métadonnées à partir du formulaire
        metadata = {
            'image': image_file.filename, # Requis pour la validation du Schéma
            'age': request.form.get('age', type=int),
            'sex': request.form.get('sex'),
            'localization': request.form.get('localization')
        }

        # 3. Validation du Schéma de données (Étape 12)
        try:
            SkinCancerPredictionSchema().load(metadata) 
        except ValidationError as err:
            logger.error(f"Erreur de validation de l'entrée de prédiction: {err.messages}")
            # Renvoyer l'erreur à la page d'accueil
            return render_template('index.html', error=f"Erreur de données d'entrée: {err.messages}")

        # 4. Vérification et normalisation de l'image (Module Image Processor)
        try:
            processed_image_tensor = validate_and_process_image(image_file)
        except ValueError as e:
            logger.error(f"Échec du traitement de l'image: {e}")
            return render_template('index.html', error=str(e))
        except Exception as e:
            logger.error(f"Erreur inattendue de traitement d'image: {e}")
            return render_template('index.html', error="Une erreur inconnue est survenue lors du traitement de l'image.")

        # 5. Inférence du Modèle
        try:
            # Ici, processed_image_tensor et metadata seraient passés à votre modèle réel
            prediction_result = make_simulated_prediction(processed_image_tensor, metadata)
        except Exception as e:
            logger.error(f"Échec de l'inférence du modèle: {e}")
            return render_template('index.html', error="Échec de la prédiction du modèle. Veuillez vérifier les journaux.")

        # 6. Rendu de la page de résultats
        return render_template(
            'result.html',
            prediction=prediction_result['prediction'],
            probability=prediction_result['probability'],
            diagnosis_class=prediction_result['diagnosis_class'],
            model_version=MODEL_VERSION
        )

    return app

# Exécution du code (pour test local)
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)