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

def load_distilbert_model(model_dir: str, base_tokenizer: str = "distilbert-base-uncased"):
    """
    Charge un modèle DistilBERT fine-tuné à partir d'un dossier contenant :
    - config.json
    - model.safetensors (ou pytorch_model.bin)
    
    Le tokenizer est rechargé depuis le modèle de base (non sauvegardé avec le fine-tune).
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        print("✅ Modèle DistilBERT fine-tuné chargé avec succès.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle DistilBERT : {e}")
        return None, None
