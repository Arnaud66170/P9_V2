# src/model_training.py

from typing import Tuple
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import mlflow

from src.utils.vectorizer_utils import train_and_save_tfidf, vectorize_data
from src.utils.mlflow_utils import log_mlflow_run

# ----------- TF-IDF + Logistic Regression -----------
def train_logreg_pipeline(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    model_dir: str,
    run_name: str = "logreg_pipeline",
    max_features: int = 5000,
    force_retrain: bool = False,
    log_to_mlflow: bool = True
) -> Tuple[LogisticRegression, float, float]:
    os.makedirs(model_dir, exist_ok=True)
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(model_dir, "log_reg_model.joblib")

    if force_retrain or not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        tfidf_vectorizer = train_and_save_tfidf(X_train, save_path = vectorizer_path, max_features = max_features)
        X_train_vec = vectorize_data(tfidf_vectorizer, X_train)
        X_test_vec = vectorize_data(tfidf_vectorizer, X_test)

        model = LogisticRegression(max_iter = 1000)
        model.fit(X_train_vec, y_train)

        joblib.dump(model, model_path)

    else:
        tfidf_vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        X_test_vec = vectorize_data(tfidf_vectorizer, X_test)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if log_to_mlflow:
        with mlflow.start_run(run_name = run_name):
            mlflow.set_tag("source", "logreg_pipeline")
            mlflow.log_param("model", "logreg")
            mlflow.log_param("vectorizer", f"TfidfVectorizer(max_features={max_features})")
            mlflow.log_param("dataset", "tweets_cleaned.csv")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

    return model, acc, f1


# ----------- ELECTRA fine-tuning avec Hugging Face Trainer -----------
@log_mlflow_run(run_name = "train_electra_pipeline")
def train_electra_pipeline(
    X_train: list,
    y_train: list,
    X_test: list,
    y_test: list,
    model_dir: str,
    log_to_mlflow: bool = True,
    force_retrain: bool = False
) -> Tuple:
    from transformers import (
        ElectraTokenizerFast,
        ElectraForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis = 1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)
        }

    LOCAL_MODEL_PATH = "models/hf_assets/electra-small-discriminator"

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Le modèle local est introuvable dans : {LOCAL_MODEL_PATH}.\n"
            f"➡️ Télécharge-le via scripts/download_electra_locally.py"
        )

    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json"
    ]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    model_files_ok = os.path.isdir(model_dir) and len(missing_files) == 0

    print(f"🧪 Fichiers attendus dans {model_dir} :")
    for f in required_files:
        status = "✅" if os.path.exists(os.path.join(model_dir, f)) else "❌"
        print(f"   {status} {f}")

    print(f"\n🧠 model_files_ok = {model_files_ok} / force_retrain = {force_retrain}\n")

    if model_files_ok and not force_retrain:
        print("✅ Chargement du modèle Electra depuis les artefacts sauvegardés.")
        model = ElectraForSequenceClassification.from_pretrained(model_dir)
        tokenizer = ElectraTokenizerFast.from_pretrained(model_dir)
        return model, tokenizer

    print("⚙️ Fine-tuning du modèle Electra depuis les assets Hugging Face...")

    tokenizer = ElectraTokenizerFast.from_pretrained(LOCAL_MODEL_PATH)
    model = ElectraForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, num_labels = 2)

    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]
    y_train = [int(y) for y in y_train]
    y_test = [int(y) for y in y_test]

    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
    test_dataset = Dataset.from_dict({"text": X_test, "label": y_test})

    def tokenize(example):
        return tokenizer(example["text"], truncation = True, padding = "max_length", max_length = 128)

    train_dataset = train_dataset.map(tokenize, batched = False)
    test_dataset = test_dataset.map(tokenize, batched = False)

    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir = "./tmp_electra",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 3,
        weight_decay = 0.01,
        load_best_model_at_end = True,
        metric_for_best_model = "f1",
        logging_dir = "./logs",
        logging_steps = 50,
        report_to = "none" if not log_to_mlflow else "mlflow"
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        compute_metrics = compute_metrics,
        tokenizer = tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)]
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    model = ElectraForSequenceClassification.from_pretrained("tmp_electra")
    tokenizer = ElectraTokenizerFast.from_pretrained("tmp_electra")

    os.makedirs(model_dir, exist_ok = True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(bin_path):
        print(f"✅ Fichier pytorch_model.bin généré avec succès dans : {bin_path}")
    else:
        print(f"❌ Fichier pytorch_model.bin manquant dans : {model_dir}")
        print("⚠️ Tentative de copie depuis hf_assets/electra-small-discriminator...")

        import shutil
        src_bin = os.path.join("models", "hf_assets", "electra-small-discriminator", "pytorch_model.bin")
        try:
            shutil.copy(src_bin, bin_path)
            print("✅ Copie réussie du fichier .bin depuis hf_assets.")
        except Exception as e:
            print(f"❌ Échec de la copie du fichier .bin : {e}")

    print(f"✅ Modèle Electra sauvegardé dans : {model_dir}")

    return model, tokenizer
