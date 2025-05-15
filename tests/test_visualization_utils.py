import pytest
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Pour éviter l'ouverture de fenêtres lors des tests

from src.utils.visualization_utils import plot_model_comparison_barplot, plot_confusion_matrix

def test_plot_model_comparison_barplot(tmp_path):
    df = pd.DataFrame({
        "Modèle": ["A", "B", "C"],
        "F1-score": [0.76, 0.83, 0.84],
        "Accuracy": [0.75, 0.82, 0.85]
    })

    save_path = tmp_path / "f1_plot.png"
    plot_model_comparison_barplot(df, metric_col="F1-score", title="Test F1", save_path=str(save_path))
    assert save_path.exists()

def test_plot_confusion_matrix(tmp_path):
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]

    save_path = tmp_path / "cm_plot.png"
    plot_confusion_matrix(y_true, y_pred, labels=["Négatif", "Positif"], title="Test CM", save_path=str(save_path))
    assert save_path.exists()
