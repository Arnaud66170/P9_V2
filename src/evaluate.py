# src/evaluate.py

import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import List, Optional

def evaluate_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    model_name: str = "modèle",
    save_path: Optional[str] = None,
    display_labels: List[str] = ["Négatif", "Positif"]
) -> dict:
    """
    Évalue un modèle de classification binaire avec Accuracy, F1-score, matrice de confusion.
    
    Args:
        y_true (List[int]) : Liste des labels réels.
        y_pred (List[int]) : Liste des labels prédits.
        model_name (str) : Nom du modèle (pour le titre du graphique).
        save_path (str) : Chemin de sauvegarde des métriques en CSV si souhaité.
        display_labels (List[str]) : Étiquettes à afficher dans la matrice.

    Returns:
        dict : dictionnaire avec les scores {'accuracy': ..., 'f1_score': ...}
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"📊 {model_name} - Accuracy : {acc:.4f} | F1-score : {f1:.4f}")
    
    # Affichage matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = display_labels)
    disp.plot(cmap = "Blues")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.show()
    
    # Sauvegarde optionnelle
    if save_path:
        df_metrics = pd.DataFrame([[model_name, acc, f1]], columns = ["Modèle", "Accuracy", "F1-score"])
        if os.path.exists(save_path):
            existing = pd.read_csv(save_path)
            existing = existing[existing["Modèle"] != model_name]
            df_metrics = pd.concat([existing, df_metrics], ignore_index = True)
        df_metrics.to_csv(save_path, index = False)
    
    return {"accuracy": acc, "f1_score": f1}
