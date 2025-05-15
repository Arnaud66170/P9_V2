# src/utils/logger.py

import os
import pandas as pd
from datetime import datetime

def log_user_event(event_type, tweet_text, predicted_label, proba, feedback=None, comment=None):
    """
    Enregistre un événement utilisateur dans un fichier CSV.

    Args:
        event_type (str): "feedback" ou "analysis"
        tweet_text (str): Texte du tweet analysé
        predicted_label (str): Émotion dominante prédite (ex : "joy")
        proba (float): Score de confiance en %
        feedback (str|None): Feedback utilisateur (👍 Yes / 👎 No)
        comment (str|None): Commentaire libre de l'utilisateur
    """
    log_dir = "huggingface_api/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        "log_feedbacks.csv" if event_type == "feedback" else "log_analysis.csv"
    )

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tweet": tweet_text,
        "predicted_emotion": predicted_label,
        "confidence_percent": round(float(proba), 2),
        "feedback": feedback,
        "comment": comment
    }

    df_entry = pd.DataFrame([entry])
    df_entry.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)