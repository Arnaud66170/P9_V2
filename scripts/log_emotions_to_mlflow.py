# scripts/log_emotions_to_mlflow.py

import os
import mlflow
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 🔁 Chemins réels depuis ce script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "models", "emotions"))

model_path = os.path.join(base_path, "model")
tokenizer_path = os.path.join(base_path, "tokenizer")
metrics_path = os.path.join(base_path, "metrics.pkl")

# ✅ Chargement
metrics = joblib.load(metrics_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 🔁 Logging dans MLflow avec enregistrement dans le Model Registry
with mlflow.start_run(run_name="retrospective_emotions_log"):
    mlflow.set_tag("source_name", __file__)  # Ajout du tag de provenance explicite

    mlflow.log_params({
        "num_labels": 28,
        "problem_type": "multi_label_classification",
        "model_architecture": "electra-small-discriminator",
        "num_epochs": 7,
        "batch_size": 16,
        "loss": "BCEWithLogits + class_weights",
        "early_stopping": True,
        "lr_scheduler": "cosine",
        "dataset": "GoEmotions (28 émotions)"
    })

    mlflow.log_metrics(metrics)

    # ✅ Pipeline officiel HF (nécessaire pour que MLflow détecte correctement le 'task')
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path="emotions_model",
        task="text-classification",  # ✅ Obligatoire pour éviter l’erreur KeyError
        registered_model_name="emotions_classifier"
    )

    print("✅ Modèle EMOTIONS loggé et enregistré dans le Model Registry.")

# Exécution depuis la racine du projet :
# python scripts/log_emotions_to_mlflow.py
