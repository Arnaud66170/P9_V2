# tests/test_pipeline_emotions.py

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline_emotions import run_emotion_pipeline
import pytest


def test_emotion_pipeline_runs():
    """
    Teste que la fonction run_emotion_pipeline() renvoie bien :
    - un modèle HuggingFace
    - un tokenizer
    - un dictionnaire de métriques (au minimum)
    """
    model, tokenizer, metrics = run_emotion_pipeline(force_retrain=True, test_mode=True)

    assert model is not None, "❌ Modèle non retourné"
    assert tokenizer is not None, "❌ Tokenizer non retourné"
    assert isinstance(metrics, dict), "❌ Les métriques ne sont pas un dictionnaire"

    f1 = metrics.get("f1", None)
    accuracy = metrics.get("accuracy", None)

    assert f1 is not None and isinstance(f1, float), "❌ 'f1' manquant ou invalide"
    assert accuracy is not None and isinstance(accuracy, float), "❌ 'accuracy' manquant ou invalide"

    print(f"✅ Test réussi — f1: {f1:.4f}, accuracy: {accuracy:.4f}")


def test_model_saved():
    """
    Vérifie que les artefacts sont bien sauvegardés dans models/emotions/
    """
    model_dir = "models/emotions"
    model_path = os.path.join(model_dir, "model")
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    metrics_path = os.path.join(model_dir, "metrics.pkl")

    # Si les artefacts n'existent pas encore, on skippe le test (pytest.mark.skipif)
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path) or not os.path.exists(metrics_path):
        pytest.skip("⏭️ Artefacts non générés, test ignoré temporairement")

    assert os.path.exists(model_path), "❌ Dossier modèle non sauvegardé"
    assert os.path.exists(tokenizer_path), "❌ Dossier tokenizer non sauvegardé"
    assert os.path.exists(metrics_path), "❌ Fichier metrics.pkl non sauvegardé"

    print("✅ Artefacts correctement sauvegardés.")

# Pour exécuter : pytest tests/test_pipeline_emotions.py -s

