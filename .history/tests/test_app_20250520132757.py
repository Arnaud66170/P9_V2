# tests/test_app.py

import importlib

def test_import_app():
    """
    Vérifie simplement que le fichier app.py se charge sans erreur.
    """
    try:
        importlib.import_module("app")
    except Exception as e:
        assert False, f"Erreur au chargement de app.py : {e}"

# Appel de la fonction de test dans le terminal
# pytest tests/test_app.py