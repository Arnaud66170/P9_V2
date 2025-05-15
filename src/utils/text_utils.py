# src/utils/text_utils.py

def prepare_for_vectorization(text_series):
    """
    Prépare une série de texte pour le vectoriseur : suppression des NaN.
    """
    return text_series.fillna("").astype(str)
