import os
import pytest
import pandas as pd
from src.utils.evaluation_utils import evaluate_model_predictions, load_predictions, load_classification_report

def test_evaluate_model_predictions(tmp_path):
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]
    save_path = tmp_path / "report.txt"

    report = evaluate_model_predictions(y_true, y_pred, save_path = str(save_path))
    assert "precision" in report
    assert save_path.exists()

def test_load_predictions(tmp_path):
    csv_path = tmp_path / "preds.csv"
    df = pd.DataFrame({
        "text": ["tweet1", "tweet2"],
        "label": [0, 1],
        "prediction": [0, 1]
    })
    df.to_csv(csv_path, index=False)

    y_true, y_pred = load_predictions(str(csv_path))
    assert y_true == [0, 1]
    assert y_pred == [0, 1]

def test_load_classification_report(tmp_path):
    txt_path = tmp_path / "report.txt"
    txt_path.write_text("sample classification report", encoding="utf-8")

    content = load_classification_report(str(txt_path))
    assert "classification" in content
