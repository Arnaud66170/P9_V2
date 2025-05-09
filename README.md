# 🧠 Projet P9_V2 – Preuve de Concept NLP & MLOps

Ce projet vise à démontrer qu’un modèle **récent et léger** (ALBERT fine-tuné) peut **dépasser une baseline classique** (TF-IDF + LogReg) sur une tâche de classification de sentiments sur Twitter, tout en étant compatible avec un **déploiement CPU**.

## 🎯 Objectifs
- Comparer **ALBERT fine-tuné** à :
  - **TF-IDF + LogReg** (baseline déployée en prod dans le P7)
  - **DistilBERT fine-tuné** (modèle performant mais trop lourd pour Hugging Face Spaces)
- Prouver que ALBERT offre un **compromis optimal poids/performance**
- Intégrer les bonnes pratiques **MLOps** (MLflow, modularité, tracking, déploiement)

## 🧱 Arborescence du projet
```
P9_V2/
├── data/                        # Données brutes (tweets.csv)
├── models/                      # Modèles entraînés
├── models_from_P7/             # Modèles importés du projet précédent
├── notebooks/                  # Notebooks exploratoires et d'entraînement
├── results/                    # Résultats, métriques, courbes, logs
├── dashboards/                 # Interfaces Gradio / Streamlit
├── src/                        # Code modulaire (prétraitement, modèles, utils)
│   └── utils/
├── requirements.txt            # Dépendances du projet
├── .gitignore
├── README.md
```

## 📊 Modèles comparés
| Modèle               | Type          | Poids approx. | F1-score visé | Compatible CPU |
|----------------------|---------------|---------------|----------------|----------------|
| TF-IDF + LogReg      | Classique     | ~0.5 Mo       | ~0.76          | ✅             |
| DistilBERT fine-tuné | Transformer   | >300 Mo       | ~0.84          | ❌             |
| ALBERT fine-tuné     | Transformer++ | ~12 Mo        | ~0.83–0.84     | ✅✅✅           |

## ⚙️ Environnement
- Entraînement local avec GPU : `GTX 1060` (6 Go)
- Déploiement CPU : Hugging Face Spaces et AWS EC2
- Tracking des expériences avec **MLflow**

## 🚀 Déploiement prévu
- Dashboard interactif (Gradio / Streamlit)
- Accessibilité **WCAG**
- Déploiement web (Spaces ou EC2)
- Intégration continue via GitHub Actions (tests, build, push)

## 📄 Livrables attendus
- ✅ Plan prévisionnel (.docx + .pdf)
- ✅ Note méthodologique (10 pages max)
- 📓 Notebook entraînement baseline + ALBERT
- 📊 Comparaison complète (accuracy, F1, temps, poids, etc.)
- 💻 Dashboard interactif déployé
- 🧪 Tests unitaires
- 🎤 Présentation soutenance (30 slides max)

---

© Projet réalisé dans le cadre de la formation Data Scientist – OpenClassrooms
