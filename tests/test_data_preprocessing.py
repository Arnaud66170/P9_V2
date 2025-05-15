# tests/test_data_preprocessing.py


import sys
import os

# Ajout de la racine du projet au chemin Python (permet d'importer depuis src/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import load_and_clean_data, basic_cleaning

import pandas as pd
import pytest

# -----------------------------
# Test de la fonction basic_cleaning
# -----------------------------

def test_basic_cleaning_removes_urls_mentions_hashtags():
    text = "Ceci est un test http://example.com @user #hashtag !?"
    cleaned = basic_cleaning(text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "!" not in cleaned
    assert "?" not in cleaned
    assert cleaned.islower()


# -----------------------------
# Test de la fonction load_and_clean_data
# -----------------------------

def test_load_and_clean_data(tmp_path):
    # Création d’un petit dataset fictif
    df_sample = pd.DataFrame({
        0: [0, 4],
        1: [111, 222],
        2: ['2021-01-01', '2021-01-02'],
        3: ['query', 'query'],
        4: ['user1', 'user2'],
        5: ['Ceci est un tweet :) http://url.com', None]
    })

    # Sauvegarde temporaire du fichier CSV
    test_file = tmp_path / "test_data.csv"
    df_sample.to_csv(test_file, index=False, header=False)

    # Chargement et nettoyage
    df_cleaned = load_and_clean_data(str(test_file), sample_size=2)

    # Vérification des colonnes
    assert list(df_cleaned.columns) == ['label', 'text']
    assert df_cleaned.shape[0] == 2
    assert df_cleaned['text'].isnull().sum() == 0
    assert all(df_cleaned['label'].isin([0, 1]))
    assert df_cleaned['text'].apply(lambda x: isinstance(x, str)).all()
