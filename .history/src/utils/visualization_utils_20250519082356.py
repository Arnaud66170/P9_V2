# src/utils/visualization_utils.py

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, labels=["Négatif", "Positif"], title="Matrice de confusion", save_path=None):
    """
    Affiche et sauvegarde une matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.grid(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_model_comparison_barplot(df, metric_col="F1-score", title="", save_path=None):
    """
    Affiche un barplot comparatif pour la métrique fournie (F1-score ou Accuracy).
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Modèle"], df[metric_col], edgecolor="black", color="#4C72B0")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.002, f"{height:.3f}",
                 ha="center", va="bottom", fontsize=10)

    plt.title(title or f"Comparaison des {metric_col}", fontsize=14)
    plt.ylabel(metric_col, fontsize=12)
    plt.ylim(0.70, 1.00)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_normalized_metrics(metrics: dict, model_name: str = "Modèle"):
    """
    Affiche un barplot Plotly des métriques comprises entre 0 et 1.

    Args:
        metrics (dict): Dictionnaire des métriques du modèle.
        model_name (str): Nom du modèle affiché dans le titre.
    """
    normalized_keys = [
        "f1_micro", "f1_macro", "f1_weighted", "accuracy",
        "roc_auc_micro", "roc_auc_macro", "pr_auc_macro", "lrap"
    ]
    filtered = {k: v for k, v in metrics.items() if k in normalized_keys}

    if not filtered:
        raise ValueError("Aucune métrique normalisée (0-1) trouvée dans les données.")

    df = pd.DataFrame(filtered.items(), columns=["Métrique", "Valeur"])
    fig = go.Figure(go.Bar(
        x=df["Métrique"],
        y=df["Valeur"],
        text=df["Valeur"].round(3),
        textposition="auto"
    ))
    fig.update_layout(
        title=f"Comparaison des scores (0 à 1) – {model_name}",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis=dict(title="Métriques")
    )
    fig.show()
