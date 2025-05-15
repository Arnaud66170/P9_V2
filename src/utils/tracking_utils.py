# src/utils/tracking_utils.py

import torch

def log_gpu_info():
    """
    Affiche les infos GPU disponibles.
    """
    if torch.cuda.is_available():
        print(f"GPU disponible : {torch.cuda.get_device_name(0)}")
        print(f"Mémoire totale : {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} Go")
    else:
        print("Aucun GPU détecté.")
