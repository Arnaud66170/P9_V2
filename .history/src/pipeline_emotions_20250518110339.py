# src/pipeline_emotions.py

import os
import pandas as pd
import numpy as np
from typing import Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import joblib
import torch
from torch.nn import BCEWithLogitsLoss

# ======== CONFIGURATION ========
THRESHOLD = 0.25  # 🔧 Seuil de prédiction ajustable pour la binarisation des scores

# 💡 Trainer personnalisé avec gestion des poids
class CustomTrainer(Trainer):
    def __init__(self, class_weights_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights_tensor.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def run_emotion_pipeline(force_retrain: bool = False, test_mode: bool = False) -> Tuple:
    set_seed(70)  # Reproductibilité

    model_dir = "models/emotions"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model")
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    metrics_path = os.path.join(model_dir, "metrics.pkl")

    # 📦 Chargement des données
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "goemotions.csv"))
    df = pd.read_csv(data_path)

    # ➤ Réduction pour tests unitaires
    if test_mode:
        df = df.sample(n=500, random_state=70)

    # 🔢 Colonnes d'émotions
    emotion_cols = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]
    df['labels'] = df[emotion_cols].apply(lambda row: [i for i, v in enumerate(row) if v == 1], axis=1)
    y = df[emotion_cols].values.astype(np.float32).tolist()

    # 📊 Calcul des poids de classes (inverse fréquence log)
    emotion_totals = df[emotion_cols].sum().values
    emotion_ratios = emotion_totals / emotion_totals.sum()
    class_weights = np.log1p(1 / emotion_ratios)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # 📦 Chargement modèle local ELECTRA
    local_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "hf_assets", "electra-small-discriminator"))
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(f"❌ Dossier du modèle introuvable : {local_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path,
        num_labels=28,
        problem_type="multi_label_classification",
        local_files_only=True
    )

    # 🔄 Préparation du dataset
    dataset = Dataset.from_pandas(df[['text']])
    dataset = dataset.map(lambda batch: tokenizer(batch["text"], padding="max_length", truncation=True), batched=True)
    dataset = dataset.add_column("labels", np.array(y).astype(np.float32).tolist())
    dataset.set_format("torch")

    dataset = dataset.shuffle(seed=70)
    eval_dataset = dataset.select(range(100))
    train_dataset = dataset.select(range(100, len(dataset)))

    # 🔁 Reprise depuis dernier checkpoint
    checkpoint_dirs = [
        d for d in os.listdir(model_dir)
        if d.startswith("checkpoint-") and os.path.isfile(os.path.join(model_dir, d, "pytorch_model.bin"))
    ]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1])) if checkpoint_dirs else []
    resume_checkpoint = os.path.join(model_dir, checkpoint_dirs[-1]) if checkpoint_dirs and not force_retrain else None

    # 🧪 Réduction en mode test
    num_epochs = 1 if test_mode else 7
    report = "none" if test_mode else "tensorboard"

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        learning_rate=3e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=100,
        save_total_limit=2,
        report_to=report
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits))  # ⚠️ Ajout du sigmoid ici
        preds = (probs > THRESHOLD).int().numpy()
        return {
            "f1": f1_score(labels, preds, average="micro"),
            "accuracy": accuracy_score(labels, preds)
        }

    # 🚀 Entraînement avec MLflow
    with mlflow.start_run(run_name="goemotions_electra_test" if test_mode else "goemotions_electra_optimized"):
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[] if test_mode else [EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights_tensor=class_weights_tensor
        )

        trainer.train(resume_from_checkpoint=resume_checkpoint if not test_mode else None)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)

        eval_results = trainer.evaluate()
        metrics = {
            "f1": eval_results.get("eval_f1", 0.0),
            "accuracy": eval_results.get("eval_accuracy", 0.0)
        }
        joblib.dump(metrics, metrics_path)

        mlflow.log_params(training_args.to_dict())
        mlflow.log_metrics(metrics)

    return model, tokenizer, metrics
