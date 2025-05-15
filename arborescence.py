import os

def afficher_arborescence(dossier, prefixe="", niveau=0, max_niveaux=None):
    """Affiche récursivement l'arborescence d'un dossier jusqu'à un niveau donné."""
    if max_niveaux is not None and niveau >= max_niveaux:
        return

    fichiers_et_dossiers = sorted(os.listdir(dossier))
    for index, nom in enumerate(fichiers_et_dossiers):
        chemin_complet = os.path.join(dossier, nom)
        is_last = index == len(fichiers_et_dossiers) - 1
        branche = "└── " if is_last else "├── "
        print(prefixe + branche + nom)
        if os.path.isdir(chemin_complet):
            extension = "    " if is_last else "│   "
            afficher_arborescence(chemin_complet, prefixe + extension, niveau + 1, max_niveaux)
