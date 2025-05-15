#src/utils/inference_utils.py

from typing import List, Union
import torch
import pandas as pd
from transformers import TextClassificationPipeline, PreTrainedModel, PreTrainedTokenizer

def predict_with_electra(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: int = -1,
    return_dataframe: bool = False
) -> Union[List[int], pd.DataFrame]:
    """
    Génère des prédictions binaires (0 ou 1) à partir d'un modèle ELECTRA fine-tuné.

    Args:
        model: modèle HF ElectraForSequenceClassification.
        tokenizer: tokenizer associé.
        texts: liste de textes à prédire.
        device: -1 (CPU) ou 0 (GPU).
        return_dataframe: si True, retourne un DataFrame avec textes + prédictions.

    Returns:
        Liste ou DataFrame contenant les prédictions.
    """

    # ⚠️ Sécurité : cast explicite des textes
    texts = [str(t) for t in texts]

    # ✅ Création du pipeline sans tokenizer_kwargs (corrigé)
    pipe = TextClassificationPipeline(
        model = model,
        tokenizer = tokenizer,
        device = device,               # -1 = CPU, 0 = GPU
        top_k = 1                      # ✅ remplace return_all_scores=False
    )

    preds = pipe(texts, batch_size = 32)

    # Exemple de sortie : [{'label': 'LABEL_0', 'score': 0.98}, ...]
    pred_labels = [int(p[0]["label"].split("_")[-1]) for p in preds]

    if return_dataframe:
        return pd.DataFrame({"text": texts, "prediction": pred_labels})

    return pred_labels
