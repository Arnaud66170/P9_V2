# src/utils/evaluation_utils.py

import os
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model_predictions(y_true, y_pred, save_path = None):
    """
    Affiche et sauvegarde un rapport de classification (précision, rappel, F1, support).
    """
    report = classification_report(y_true, y_pred, target_names = ["Négatif", "Positif"])
    print(report)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        with open(save_path, "w", encoding = "utf-8") as f:
            f.write(report)

    return report


def load_predictions(pred_path):
    """
    Charge un fichier CSV de prédictions avec les colonnes : [text, label, prediction].
    """
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ Fichier non trouvé : {pred_path}")

    df = pd.read_csv(pred_path)

    if not {"label", "prediction"}.issubset(df.columns):
        raise ValueError("❌ Le fichier doit contenir les colonnes 'label' et 'prediction'.")

    return df["label"].tolist(), df["prediction"].tolist()


def load_classification_report(report_path):
    """
    Lit un rapport de classification (au format texte brut .txt).
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"❌ Fichier non trouvé : {report_path}")

    with open(report_path, "r", encoding = "utf-8") as f:
        content = f.read()

    return content
