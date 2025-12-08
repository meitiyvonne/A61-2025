from marshmallow import Schema, fields, validate

# ==========================================
# Schéma de validation des données d'entrée (Étape 12)
# ==========================================
class SkinCancerPredictionSchema(Schema):
    # Le fichier image est envoyé via le champ du formulaire, mais la métadonnée
    # 'image' est ajoutée pour satisfaire le champ requis lors de la validation.
    image = fields.Str(required=True, error_messages={"required": "Les données d'image sont requises."}) 
    
    # Métadonnées cliniques (âge/sexe/localisation)
    age = fields.Int(
        required=True, 
        validate=validate.Range(min=1, error="L'âge doit être supérieur à zéro.")
    )
    sex = fields.Str(
        required=True, 
        validate=validate.OneOf(['male', 'female', 'other'], error="Le sexe n'est pas valide.")
    )
    localization = fields.Str(
        required=True, 
        validate=validate.OneOf(
            ['torso', 'arm', 'leg', 'face', 'neck', 'other'], 
            error="La localisation n'est pas valide."
        )
    ) 

# Schéma de la réponse (Pour référence)
class PredictionResponseSchema(Schema):
    errors = fields.Dict(allow_none=True)
    version = fields.Str(allow_none=False)
    prediction = fields.Str(allow_none=False)
    probability = fields.Float(allow_none=False)
    diagnosis_class = fields.Str(allow_none=False)