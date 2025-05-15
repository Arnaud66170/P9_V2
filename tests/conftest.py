# tests/conftest.py (ou à la racine du projet)

import sys
import os

# Permet d'importer 'src' peu importe où le test est exécuté
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
