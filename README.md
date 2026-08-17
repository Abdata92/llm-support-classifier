# 🚀 Classification Multilingue de Tickets Support via LLM Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Dependency%20Manager-Poetry-blueviolet.svg)](https://python-poetry.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PEFT%20%2F%20Transformers-orange)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF5500.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 À propos du Projet

Ce projet implémente une solution de **classification automatique et multilingue** pour un service support client gérant des milliers de requêtes quotidiennes.

Grâce au **Fine-Tuning efficace (QLoRA / PEFT)** d'un modèle de langage Open-Weight (Llama-3-8B / Mistral-7B), le système analyse le contenu des tickets et les oriente instantanément vers le bon département spécialisé (*queue*), réduisant considérablement le temps de traitement manuel et améliorant la réactivité client.

### 🌟 Points Forts & Valeur Ajoutée
- **Multilingue Natively Supported** : Traitement fluide du Français, Anglais, Allemand et Portugais.
- **Haute Précision (Weighted F1-Score $\ge 92\%$)** : Optimisation sur jeu de test stratifié pour surpasser les performances des LLM non personnalisés (Zero-Shot/Few-Shot).
- **Entraînement Efficient (QLoRA 4-bit)** : Quantification et adaptation LoRA permettant d'entraîner un modèle de 8B paramètres sur une seule carte GPU de 24 Go.
- **Architecture Production-Ready** : Separation des responsabilités (Ingestion, Prompts, Fine-tuning, API FastAPI, IHM Gradio, Tests Pytest).
- **Pipeline CI/CD & MLOps** : Validation automatique du code et du seuil de performance avant tout merge sur la branche principale.

---

## 🎯 Métriques & Exigences Métier

### 1. Ingestion & Respect Strict du Contexte
Pour garantir la confidentialité et la robustesse en production, l'ingestion isole strictement **5 colonnes autorisées** :
- `subject` : Objet de la demande.
- `body` : Message détaillé rédigé par l'utilisateur.
- `language` : Langue de la requête (`FR`, `EN`, `DE`, `PT`).
- `business_type` : Secteur d'activité de l'entreprise cliente.
- `queue` *(Variable cible)* : Catégorie d'orientation affectée par les opérateurs.

### 2. Évaluation & Performance Cible
Le modèle est évalué sur un **jeu de test stratifié** pour conserver la distribution exacte des classes. La métrique clé est le **Weighted F1-score** :

$$F1 = \sum_{j=1}^{\vert{}C\vert{}} \alpha_j F1_j \quad \text{avec} \quad \alpha_j = \frac{n_j}{n}$$

- **Objectif de performance** : $F1 \ge 0.92$ ($92\%$).

---

## 📂 Architecture du Projet


```text
llm-support-classifier/
│
├── .github/workflows/          # CI/CD (Linting, Tests unitaires & Seuil F1)
│   ├── ci.yml
│   └── model_evaluation.yml
│
├── data/                       # Datasets (non versionnés)
│   ├── raw/                    # Données brutes (600 tickets)
│   └── processed/              # Splits stratifiés Train / Test au format JSONL
│
├── src/                        # Code source principal
│   ├── config.py               # Configuration centralisée & Hyperparamètres
│   ├── data/                   # Ingestion stricte & Formateur de prompts multilingues
│   ├── models/                 # Evaluation Baseline & Script Fine-tuning QLoRA
│   └── evaluation/             # Metrics Weighted F1 & Visualisations
│
├── api/                        # Serveur Backend d'Inférence (FastAPI + Pydantic)
├── ui/                         # Interface Utilisateur interactive (Gradio)
├── tests/                      # Tests unitaires et d'intégration (Pytest)
├── pyproject.toml              # Gestionnaire de dépendances Poetry
└── README.md

```

---

## 🛠️ Stack Technique

* **Langage & Environnement** : Python 3.10+, Poetry
* **LLM & Fine-Tuning** : Hugging Face `transformers`, `peft` (LoRA), `bitsandbytes` (4-bit quantization), `accelerate`
* **Machine Learning & Evaluation** : PyTorch, Scikit-Learn, Pandas
* **Serving & Web UI** : FastAPI, Uvicorn, Gradio
* **Qualité & CI/CD** : Pytest, Ruff, GitHub Actions

---

## 🚀 Installation & Prise en Main

### 1. Prérequis

Assurez-vous d'avoir Python 3.10+ et Poetry installés.

```bash
# Cloner le dépôt
git clone [https://github.com/votre-compte/llm-support-classifier.git](https://github.com/votre-compte/llm-support-classifier.git)
cd llm-support-classifier

# installation de poetry 
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Activer la création du .venv localement (recommandé)
poetry config virtualenvs.in-project true

# générer le lock fil e
poetry lock

# Installer les dépendances
poetry install



```

### 2. Préparation des Données & Split Stratifié

Placez votre fichier `support_tickets_raw.csv` dans `data/raw/` puis lancez :

```bash
poetry run python -m src.data.prepare_data

```

### 3. Entraînement / Fine-Tuning (Sur Sandbox GPU PyTorch)

Pour lancer le fine-tuning QLoRA :

```bash
poetry run python -m src.models.finetune

```

### 4. Lancement de l'API & de l'Interface Utilisateur (IHM)

**Démarrer l'API FastAPI :**

```bash
poetry run uvicorn api.app:app --reload --port 8000

```

**Démarrer l'Interface Gradio :**

```bash
poetry run python ui/app_ui.py

```

---

## 📊 Résultats & Comparatif

| Modèle | Weighted F1-Score | Temps de réponse moyen | Remarques |
| --- | --- | --- | --- |
| **Baseline (Zero-Shot)** | ~68% - 74% | ~450 ms | Erreurs sur les nuances métier & langues secondaires |
| **LLM Fine-Tuné (QLoRA)** | **$\ge 92.5\%$** | ~120 ms | Qualification précise et respect des contraintes multilingues |

---

## 👤 Auteur

* **Abel FOUOBE** - *LLM Engineer* - [GitHub](https://www.google.com/search?q=https://github.com/votre-compte) | [LinkedIn](https://www.google.com/search?q=https://linkedin.com/in/votre-profil)

```

```
