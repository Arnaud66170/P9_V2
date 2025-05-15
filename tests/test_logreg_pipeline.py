# tests/test_logreg_pipeline.py

import sys
import os

# Ajout de la racine du projet au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import shutil
import pandas as pd
from src.model_training import train_logreg_pipeline

def test_train_logreg_pipeline(tmp_path):
    # Données d'exemple réduites
    X = pd.Series([
        "This movie is great!",
        "Terrible experience.",
        "I loved it.",
        "Not good at all.",
        "Fantastic film.",
        "Horrible acting.",
    ])
    y = pd.Series([1, 0, 1, 0, 1, 0])

    # Split manuel (2 train, 1 test par classe)
    X_train = pd.concat([X[y == 1].iloc[:2], X[y == 0].iloc[:2]])
    y_train = pd.concat([y[y == 1].iloc[:2], y[y == 0].iloc[:2]])
    X_test = pd.concat([X[y == 1].iloc[2:], X[y == 0].iloc[2:]])
    y_test = pd.concat([y[y == 1].iloc[2:], y[y == 0].iloc[2:]])

    model_dir = tmp_path / "logreg_model"
    model, acc, f1 = train_logreg_pipeline(
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        model_dir = str(model_dir),
        run_name = "test_logreg_pipeline",
        max_features = 100,
        force_retrain = True
    )

    # Vérification des sorties
    assert os.path.exists(model_dir / "log_reg_model.joblib")
    assert os.path.exists(model_dir / "tfidf_vectorizer.joblib")
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0

# terminal :
# pytest tests/test_logreg_pipeline.py -v