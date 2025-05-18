# src/utils/load_models_from_p7.py

import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_logreg_model(model_path: str, vectorizer_path: str):
    """
    Charge un modèle TF-IDF + Régression Logistique depuis des fichiers .pkl
    """
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✅ Modèle LogReg et vecteur TF-IDF chargés avec succès.")
        return model, vectorizer
    except Exception as e:
        print(f"❌ Erreur lors du chargement de LogReg : {e}")
        return None, None
