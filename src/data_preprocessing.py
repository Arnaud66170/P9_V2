import pandas as pd
import re

def basic_cleaning(text: str) -> str:
    """
    Nettoyage de base : suppression des URLs, mentions, hashtags, ponctuation, etc.
    """
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                  # mentions
    text = re.sub(r"#\w+", "", text)                  # hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # ponctuation et caractères spéciaux
    text = text.lower()                                # minuscules
    text = re.sub(r"\s+", " ", text).strip()          # espaces superflus
    return text

def load_and_clean_data(path: str, sample_size: int = 100_000) -> pd.DataFrame:
    """
    Charge le fichier CSV, renomme les colonnes, nettoie le texte et transforme les labels.
    """
    df = pd.read_csv(path, header=None, encoding='ISO-8859-1')
    df.columns = ['label', 'id', 'date', 'query', 'user', 'text']
    df = df[['label', 'text']]

    # Remapping des labels (0: négatif, 4: positif)
    df['label'] = df['label'].map({0: 0, 4: 1})

    # Nettoyage du texte
    df['text'] = df['text'].apply(basic_cleaning)

    # Réduction échantillon (stratifié)
    if sample_size and sample_size < len(df):
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(sample_size // 2, random_state=42))

    return df.reset_index(drop=True)
