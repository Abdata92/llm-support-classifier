Voici l'architecture révisée et ajustée pour intégrer **Poetry** (pour la gestion des dépendances et de l'environnement virtuel) ainsi qu'un module **UI** (Interface Utilisateur sous Gradio ou Streamlit) pour tester l'application en direct de manière conviviale.

---

### 📂 Structure du projet : `llm-support-classifier/`

```text
llm-support-classifier/
│
├── .github/                         # Pipelines CI/CD (GitHub Actions)
│   └── workflows/
│       ├── ci.yml                   # Tests unitaires, linting (Ruff/Black) via Poetry
│       └── model_evaluation.yml     # Validation automatique du F1-score >= 92%
│
├── data/                            # Gestion des données (non versionné sur Git)
│   ├── raw/
│   │   └── support_tickets_raw.csv  # Dataset source des 600 demandes
│   └── processed/
│       ├── train.jsonl              # Dataset d'entraînement (prompts)
│       └── test.jsonl               # Jeu de test stratifié
│
├── notebooks/                       # Prototypage & EDA
│   ├── 01_data_exploration.ipynb    # EDA (distribution des queues, langues, business_type)
│   └── 02_prompt_experiments.ipynb # Prototypage des prompts multilingues
│
├── src/                             # Code source modulaire
│   ├── __init__.py
│   ├── config.py                    # Hyperparamètres, chemins, seuils (F1_THRESHOLD = 0.92)
│   │
│   ├── data/                        # Ingestion & Traitement strict
│   │   ├── __init__.py
│   │   ├── dataset_loader.py        # Filtrage strict des 5 colonnes
│   │   ├── prompt_formatter.py      # Conversion en prompts LLM
│   │   └── split_stratified.py      # Echantillonnage stratifié selon `queue`
│   │
│   ├── models/                      # Entraînement & Inférence LLM
│   │   ├── __init__.py
│   │   ├── baseline_evaluator.py    # Évaluation du LLM de base non personnalisé
│   │   ├── finetune.py              # Script de fine-tuning (QLoRA / PEFT)
│   │   └── predictor.py             # Pipeline d'inférence unifiée
│   │
│   └── evaluation/                  # Métriques & Benchmark
│       ├── __init__.py
│       ├── metrics.py               # Calcul strict du F1_j et du Weighted F1-score
│       └── visualize.py             # Matrice de confusion et métriques pour captures
│
├── api/                             # API Backend (Serving)
│   ├── __init__.py
│   ├── app.py                       # Serveur FastAPI pour la classification en temps réel
│   ├── schemas.py                   # Schémas Pydantic (entrées / sorties)
│   └── Dockerfile                   # Containerisation API
│
├── ui/                              # Interface Utilisateur Démo (Gradio / Streamlit)
│   ├── app_ui.py                    # Interface web de démonstration et de test
│   └── Dockerfile.ui                # Containerisation de l'IHM
│
├── tests/                           # Tests unitaires & d'intégration
│   ├── test_data_processing.py      # Validation du filtrage strict des 5 colonnes
│   ├── test_prompt_formatter.py     # Validation des prompts multilingues
│   ├── test_metrics.py              # Test unitaire de la formule du Weighted F1-score
│   └── test_api.py                  # Tests d'intégration API
│
├── pyproject.toml                   # Fichier principal Poetry (dépendances & configurations)
├── poetry.lock                      # Verrouillage exact des versions
├── .gitignore
├── .dockerignore
└── README.md                        # Documentation et instructions de déploiement

```

---

### 🛠️ Nouveautés apportées :

1. **`pyproject.toml` (Poetry)** :
* Remplace `requirements.txt` et `setup.py`.
* Gère les dépendances de développement (`pytest`, `ruff`, `black`), de ML (`torch`, `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`), d'API (`fastapi`, `uvicorn`) et d'UI (`gradio` ou `streamlit`).


2. **Module `ui/` (Interface Utilisateur)** :
* Contient `app_ui.py` basant son fonctionnement soit sur Gradio soit sur Streamlit.
* Permet de saisir manuellement les champs autorisés : `subject`, `body`, `language` (FR, EN, DE, PT) et `business_type`.
* Affiche en temps réel la catégorie (`queue`) prédite, le score de confiance et la comparaison entre le modèle **Baseline** et le modèle **Fine-tuné**.



---

### 🚀 Étape 1 : Initialisation avec Poetry

Pour créer cette structure et configurer l'environnement Poetry, vous pouvez exécuter ces commandes dans votre terminal :

```bash
# 1. Initialiser le projet Poetry
poetry new llm-support-classifier --src
cd llm-support-classifier

# 2. Ajouter les dépendances principales
poetry add transformers peft datasets accelerate bitsandbytes fastapi uvicorn gradio scikit-learn pandas pydantic

# 3. Ajouter les dépendances de dev
poetry add --group dev pytest ruff black

```

Voulez-vous qu'on génère le fichier **`pyproject.toml`** complet ou qu'on passe directement au code de **`src/data/dataset_loader.py`** pour traiter les 600 entrées ?