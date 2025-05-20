# tests/test_app.py

import importlib
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def test_import_app():
    """
    Vérifie que app.py se charge correctement depuis la racine.
    """
    try:
        importlib.import_module("app")
    except Exception as e:
        assert False, f"Erreur au chargement de app.py : {e}"

# Appel de la fonction de test dans le terminal
# pytest tests/test_app.py