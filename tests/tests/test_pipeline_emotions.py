import os
import pytest
from src.pipeline_emotions import run_emotion_pipeline

def test_emotion_pipeline_runs():
    """
    Teste que la fonction run_emotion_pipeline() renvoie bien :
    - un modèle HuggingFace
    - un tokenizer
    - un dictionnaire de métriques
    """
    model, tokenizer, metrics = run_emotion_pipeline(force_retrain=True)

    assert model is not None, "❌ Modèle non retourné"
    assert tokenizer is not None, "❌ Tokenizer non retourné"
    assert isinstance(metrics, dict), "❌ Les métriques ne sont pas un dictionnaire"
    assert "f1" in metrics, "❌ La métrique 'f1' est absente"
    assert "accuracy" in metrics, "❌ La métrique 'accuracy' est absente"

    print("✅ Test de run_emotion_pipeline() passé avec succès.")

def test_model_saved():
    """
    Vérifie que les artefacts sont bien sauvegardés dans models/emotions/
    """
    model_dir = "models/emotions"
    model_path = os.path.join(model_dir, "model")
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    metrics_path = os.path.join(model_dir, "metrics.pkl")

    assert os.path.exists(model_path), "❌ Dossier modèle non sauvegardé"
    assert os.path.exists(tokenizer_path), "❌ Dossier tokenizer non sauvegardé"
    assert os.path.exists(metrics_path), "❌ Fichier metrics.pkl non sauvegardé"

    print("✅ Artefacts correctement sauvegardés.")

# Exécution des tests :
# pytest tests/test_pipeline_emotions.py -s
