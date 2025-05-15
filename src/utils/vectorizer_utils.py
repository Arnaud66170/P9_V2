# src/utils/vectorizer_utils.py

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
import pandas as pd


def train_and_save_tfidf(X_train: pd.Series, save_path: str, max_features: int = 5000) -> TfidfVectorizer:
    """
    Entraîne un TfidfVectorizer sur les textes fournis et le sauvegarde.
    """
    X_train = X_train.fillna("").astype(str)
    vectorizer = TfidfVectorizer(max_features = max_features)
    vectorizer.fit(X_train)

    joblib.dump(vectorizer, save_path)
    return vectorizer


def load_tfidf_vectorizer(path: str) -> TfidfVectorizer:
    """
    Charge un TfidfVectorizer sauvegardé depuis un fichier .joblib.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer non trouvé à l'emplacement : {path}")
    return joblib.load(path)


def vectorize_data(vectorizer: TfidfVectorizer, data: pd.Series):
    """
    Applique un vectoriseur TF-IDF à des données textuelles, après nettoyage des NaN.
    """
    return vectorizer.transform(data.fillna("").astype(str))
