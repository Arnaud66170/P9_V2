# 🧠 Projet P9_V2 – Preuve de Concept NLP & MLOps

Ce projet vise à démontrer qu’un modèle **léger et moderne** (ELECTRA fine-tuné) peut **dépasser une baseline classique** (TF-IDF + LogReg) pour la classification de sentiments et l’analyse fine des émotions sur Twitter, tout en restant **entièrement déployable sur CPU**.

## 🎯 Objectifs
- Comparer **ELECTRA fine-tuné** à :
  - **TF-IDF + LogReg** (baseline utilisée en production dans le P7)
  - **DistilBERT fine-tuné** (modèle performant mais trop lourd pour un déploiement léger)
- Démontrer que **ELECTRA offre le meilleur compromis poids/performance**
- Étendre le projet avec une **analyse fine des émotions** (GoEmotions)
- Intégrer les bonnes pratiques **MLOps** : modularité, MLflow, feedback utilisateur, alertes, CI/CD

## 🧱 Arborescence du projet (structure finale)
```
P9_V2/
├── app.py                        # Interface Gradio principale (ELECTRA, audit, feedback)
├── src/                         # Modules métiers (utils, prétraitement, logger...)
│   ├── constants/emotions.py
│   ├── utils/emotion_utils.py
│   ├── utils/logger.py
│   └── utils/alert_email.py
├── models/electra_model/        # Modèle fine-tuné local (config + tokenizer + weights)
├── notebooks/                   # Notebooks d'entraînement et de benchmark
├── outputs/                     # Exports CSV, audit, graphiques, logs
│   ├── emotion_stats_2000.csv
│   └── emotion_errors_flagged.csv
├── feedback_log_emotions.csv   # Retours utilisateurs (👍 / 👎)
├── mlruns/                      # MLflow tracking local
├── requirements.txt
├── lancer_app_ngrok.bat         # Script lancement local avec Ngrok
├── stop_app_ngrok.bat           # Script fermeture app + tunnel
├── deploiement_gradio_ngrok.docx
└── docs/
    └── deploiement_ngrok.md
```

## 📊 Modèles comparés
| Modèle               | Type          | Poids approx. | F1-score visé | Compatible CPU |
|----------------------|---------------|---------------|----------------|----------------|
| TF-IDF + LogReg      | Classique     | ~0.5 Mo       | ~0.76          | ✅             |
| DistilBERT fine-tuné | Transformer   | >300 Mo       | ~0.84          | ❌             |
| ELECTRA fine-tuné    | Transformer++ | ~12 Mo        | ~0.84          | ✅✅✅           |

## ⚙️ Environnement
- Entraînement local sur GPU (GTX 1060)
- Déploiement sur CPU (Ngrok)
- Tracking des runs avec **MLflow**
- Audit automatique sur des tweets types
- Pipeline d’analyse des émotions + alertes sur cas sensibles

## 🚀 Déploiement et UX
- Interface Gradio unique (`app.py`) avec :
  - Détection multi-label avec ELECTRA (GoEmotions)
  - Feedback utilisateur (👍 / 👎)
  - Audit contextuel (mots sensibles)
  - Génération de CSV + visualisation
- Accessibilité WCAG (onglets, lisibilité, emojis)
- Tunnel d’accès externe via **Ngrok** (auto-déployable en local)
- CI/CD via GitHub Actions (tests unitaires + push auto)

## 📄 Livrables
- ✅ Plan prévisionnel + scénario (Word)
- ✅ Note méthodologique (10 pages max)
- ✅ Interface interactive Gradio (`app.py`)
- ✅ Pipeline d’audit (`scripts/audit_emotions.py`)
- ✅ Feedback CSV + système d’alertes
- ✅ Déploiement local Ngrok + documentation `.docx` et `.md`
- 📊 Notebook comparatif TF-IDF / ELECTRA / DistilBERT
- 🎤 Présentation finale (slides synthétiques)

---

© Projet réalisé dans le cadre de la formation AI Engineer – OpenClassrooms
