# src/data_preprocessing.py

import pandas as pd
import re


def basic_cleaning(text: str) -> str:
    """
    Nettoyage de base : suppression des URLs, mentions, hashtags, ponctuation, etc.
    """
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)        # Suppression des URLs
    text = re.sub(r"@\w+", "", text)                  # Suppression des mentions
    text = re.sub(r"#\w+", "", text)                  # Suppression des hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # Suppression de la ponctuation et des caractères spéciaux
    text = text.lower()                               # Conversion en minuscules
    text = re.sub(r"\s+", " ", text).strip()          # Suppression des espaces superflus
    return text


def load_and_clean_data(path: str, sample_size: int = 100_000) -> pd.DataFrame:
    """
    Charge le fichier CSV, renomme les colonnes, nettoie le texte et transforme les labels.

    Args:
        path (str): Chemin vers le fichier CSV.
        sample_size (int): Nombre d’échantillons à conserver. Stratifié par label. 

    Returns:
        pd.DataFrame: Données nettoyées et prêtes à l’usage.
    """
    # Chargement du fichier
    df = pd.read_csv(path, header=None, encoding='ISO-8859-1')
    df.columns = ['label', 'id', 'date', 'query', 'user', 'text']
    df = df[['label', 'text']]

    # Transformation des labels (0: négatif, 4: positif)
    df['label'] = df['label'].map({0: 0, 4: 1})

    # Suppression ou remplacement des valeurs manquantes dans la colonne texte
    df['text'] = df['text'].fillna("")

    # Nettoyage du texte
    df['text'] = df['text'].apply(basic_cleaning)

    # Réduction du dataset de manière stratifiée
    if sample_size and sample_size < len(df):
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(sample_size // 2, random_state=42)
        )

    return df.reset_index(drop=True)
