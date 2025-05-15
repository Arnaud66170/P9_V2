def filter_predictions(predictions, labels, threshold = 0.25):
    """
    Filtre les prédictions multi-label en fonction d’un seuil donné.

    Args:
        predictions (list of float): Liste de scores de sortie du modèle (sigmoïde).
        labels (list of str): Liste des labels correspondants.
        threshold (float): Seuil minimum pour retenir une émotion.

    Returns:
        list of tuples: Liste des (label, score) retenus.
    """
    return [(label, float(score)) for label, score in zip(labels, predictions) if score >= threshold]
