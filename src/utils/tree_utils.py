import os

def afficher_arborescence(dossier: str = '..', niveau_max: int = 2, prefixe: str = '') -> None:
    """
    Affiche récursivement l'arborescence d'un dossier jusqu'à un certain niveau de profondeur.

    Paramètres
    ----------
    dossier : str
        Chemin du dossier racine à explorer.
    niveau_max : int
        Profondeur maximale de l'affichage (0 = seulement la racine).
    prefixe : str
        Préfixe utilisé pour formater l'affichage (géré automatiquement en récursif).
    """
    if niveau_max < 0:
        return
    try:
        fichiers = sorted(os.listdir(dossier))
    except PermissionError:
        print(f"{prefixe}📛 Accès refusé à {dossier}")
        return
    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        print(f"{prefixe}├── {fichier}")
        if os.path.isdir(chemin):
            afficher_arborescence(chemin, niveau_max - 1, prefixe + '│   ')

## - Utilisation de la fonction

# from src.utils.tree_utils import afficher_arborescence

## - Affiche 3 niveaux à partir du dossier courant
# afficher_arborescence('.', niveau_max=3)