# tests/test_inference_utils.py

import pytest
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from src.utils.inference_utils import predict_with_electra

@pytest.fixture(scope="module")
def electra_model_and_tokenizer():
    model_path = "models/electra_model"  # ⚠️ assure-toi que le dossier existe et contient le modèle fine-tuné
    model = ElectraForSequenceClassification.from_pretrained(model_path)
    tokenizer = ElectraTokenizerFast.from_pretrained(model_path)
    return model, tokenizer

def test_predict_with_electra_returns_correct_shape(electra_model_and_tokenizer):
    model, tokenizer = electra_model_and_tokenizer
    texts = ["C'est une journée géniale !", "Je déteste ce produit."]
    
    predictions = predict_with_electra(model, tokenizer, texts, device = -1, return_dataframe = False)
    
    assert isinstance(predictions, list)
    assert len(predictions) == len(texts)
    assert all(pred in [0, 1] for pred in predictions)

def test_predict_with_electra_dataframe(electra_model_and_tokenizer):
    model, tokenizer = electra_model_and_tokenizer
    texts = ["J'adore ce film.", "C'est complètement nul."]
    
    df_preds = predict_with_electra(model, tokenizer, texts, device = -1, return_dataframe = True)

    assert "text" in df_preds.columns
    assert "prediction" in df_preds.columns
    assert len(df_preds) == 2
    assert all(pred in [0, 1] for pred in df_preds["prediction"])

# Test de la gestion des erreurs :
# pytest tests/test_inference_utils.py
