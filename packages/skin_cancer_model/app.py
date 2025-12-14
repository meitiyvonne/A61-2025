from flask import Flask, request, jsonify, render_template, redirect, url_for
from marshmallow import ValidationError
from werkzeug.datastructures import FileStorage
import json
import torch # Maintenir car les tenseurs sont manipulés
import numpy as np 

from skin_cancer_model.config import API_VERSION 
# Importe directement la version depuis __init__.py qui a été corrigé
from skin_cancer_model import __version__ as MODEL_VERSION 
from skin_cancer_model.schemas import SkinCancerPredictionSchema
from skin_cancer_model.__init__ import logger 
from skin_cancer_model.image_processor import validate_and_process_image 
# NOUVELLE IMPORTATION: Les vraies fonctions de modèle
from skin_cancer_model.prediction import load_model, make_prediction 

import os
# ==========================================
# Application Flask
# ==========================================

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(APP_ROOT, 'templates')

# Tenter de charger le VRAI modèle au démarrage de l'API (Partie 10)
MODEL = load_model() 

def create_app():
    # app = Flask(__name__, template_folder='templates') 
    app = Flask(__name__, template_folder=TEMPLATE_DIR)
    
    @app.route('/', methods=['GET'])
    def index():
        """ Page d'accueil : Affiche le formulaire de téléversement d'image """
        return render_template('index.html')

    @app.route('/version', methods=['GET'])
    def version():
        """ Affiche les versions de l'API et du modèle """
        return jsonify({
            'api_version': API_VERSION,
            'model_version': MODEL_VERSION
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """ Traite la requête de prédiction avec image et métadonnées """
        
        # 1. Vérification du fichier
        if 'image' not in request.files:
            return render_template('index.html', error='Veuillez téléverser un fichier image.')
        
        image_file: FileStorage = request.files['image']
        
        # 2. Préparation des métadonnées à partir du formulaire
        metadata = {
            'image': image_file.filename, 
            'age': request.form.get('age', type=int),
            'sex': request.form.get('sex'),
            'localization': request.form.get('localization')
        }

        # 3. Validation du Schéma de données
        try:
            SkinCancerPredictionSchema().load(metadata) 
        except ValidationError as err:
            logger.error(f"Erreur de validation de l'entrée de prédiction: {err.messages}")
            return render_template('index.html', error=f"Erreur de données d'entrée: {err.messages}")

        # 4. Vérification et normalisation de l'image
        try:
            processed_image_tensor = validate_and_process_image(image_file)
        except ValueError as e:
            logger.error(f"Échec du traitement de l'image: {e}")
            return render_template('index.html', error=str(e))
        except Exception as e:
            logger.error(f"Erreur inattendue de traitement d'image: {e}")
            return render_template('index.html', error="Une erreur inconnue est survenue lors du traitement de l'image.")

        # 5. Inférence du Modèle (MAINTENANT UTILISE LE VRAI MODÈLE)
        try:
            # Passe le tenseur, les métadonnées, et l'instance du modèle chargé (MODEL)
            prediction_result = make_prediction(processed_image_tensor, metadata, MODEL) 
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