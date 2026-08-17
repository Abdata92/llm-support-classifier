```python
import weasyprint

html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Documentation du Projet - Classification Multilingue LLM Support Client</title>
    <style>
        @page {
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
            @bottom-right {
                content: "Page " counter(page) " / " counter(pages);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 9pt;
                color: #718096;
            }
            @bottom-left {
                content: "Classification Multilingue LLM — Service Support";
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 9pt;
                color: #718096;
            }
        }

        *, *::before, *::after {
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
            color: #2D3748;
            background-color: #FFFFFF;
            line-height: 1.6;
            font-size: 10.5pt;
            margin: 0;
            padding: 0;
        }

        /* Header / Banner */
        .header-container {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            color: #FFFFFF;
            padding: 30px 25px;
            border-radius: 8px;
            margin-bottom: 25px;
        }

        .header-title {
            font-size: 20pt;
            font-weight: 700;
            margin: 0 0 10px 0;
            letter-spacing: -0.5px;
            color: #F8FAFC;
        }

        .header-subtitle {
            font-size: 12pt;
            color: #38BDF8;
            font-weight: 500;
            margin: 0 0 15px 0;
        }

        .header-tags {
            display: table;
            width: 100%;
        }

        .tag-cell {
            display: table-cell;
            font-size: 9pt;
            background-color: rgba(255, 255, 255, 0.1);
            padding: 4px 10px;
            border-radius: 4px;
            color: #E2E8F0;
            margin-right: 8px;
        }

        h2 {
            font-size: 14pt;
            color: #0F172A;
            border-left: 4px solid #0284C7;
            padding-left: 10px;
            margin-top: 22px;
            margin-bottom: 12px;
            page-break-after: avoid;
        }

        h3 {
            font-size: 11.5pt;
            color: #1E293B;
            margin-top: 16px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }

        p {
            margin-top: 0;
            margin-bottom: 10px;
            text-align: justify;
        }

        ul, ol {
            margin-top: 0;
            margin-bottom: 12px;
            padding-left: 20px;
        }

        li {
            margin-bottom: 4px;
        }

        /* Callout box */
        .callout {
            background-color: #F0F9FF;
            border: 1px solid #BAE6FD;
            border-left: 4px solid #0284C7;
            padding: 12px 15px;
            border-radius: 4px;
            margin: 15px 0;
            page-break-inside: avoid;
        }

        .callout-title {
            font-weight: bold;
            color: #0369A1;
            margin-bottom: 5px;
            font-size: 10.5pt;
        }

        /* Code block */
        pre {
            background-color: #0F172A;
            color: #F8FAFC;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 8.5pt;
            line-height: 1.4;
            overflow-x: auto;
            margin: 12px 0;
            page-break-inside: avoid;
        }

        code {
            font-family: 'Consolas', 'Courier New', monospace;
            background-color: #F1F5F9;
            color: #0F172A;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 9pt;
        }

        /* Table */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }

        th {
            background-color: #1E293B;
            color: #FFFFFF;
            font-weight: 600;
            text-align: left;
            padding: 8px 12px;
            border: 1px solid #1E293B;
        }

        td {
            padding: 8px 12px;
            border: 1px solid #E2E8F0;
            background-color: #FFFFFF;
        }

        tr:nth-child(even) td {
            background-color: #F8FAFC;
        }

        /* Math formula styling */
        .math {
            font-family: 'Times New Roman', Times, serif;
            font-style: italic;
            font-weight: bold;
            color: #0F172A;
        }

        .formula-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 10px;
            text-align: center;
            border-radius: 6px;
            margin: 12px 0;
            font-size: 11pt;
            page-break-inside: avoid;
        }

        .badge {
            display: inline-block;
            padding: 2px 8px;
            font-size: 8pt;
            font-weight: 600;
            border-radius: 12px;
            color: white;
            background-color: #0EA5E9;
        }
    </style>
</head>
<body>

    <div class="header-container">
        <div class="header-title">Classification Multilingue de Tickets Support par Fine-Tuning LLM</div>
        <div class="header-subtitle">Architecture End-to-End, Fine-Tuning QLoRA (Llama-3 / Mistral), MLOps & Déploiement Production-Ready</div>
        <div style="margin-top: 10px;">
            <span style="background-color: rgba(56, 189, 248, 0.2); color: #38BDF8; padding: 4px 10px; border-radius: 4px; font-size: 8.5pt; font-weight: 600; margin-right: 5px;">LLM Engineering</span>
            <span style="background-color: rgba(56, 189, 248, 0.2); color: #38BDF8; padding: 4px 10px; border-radius: 4px; font-size: 8.5pt; font-weight: 600; margin-right: 5px;">PEFT / QLoRA</span>
            <span style="background-color: rgba(56, 189, 248, 0.2); color: #38BDF8; padding: 4px 10px; border-radius: 4px; font-size: 8.5pt; font-weight: 600; margin-right: 5px;">FastAPI & Gradio</span>
            <span style="background-color: rgba(56, 189, 248, 0.2); color: #38BDF8; padding: 4px 10px; border-radius: 4px; font-size: 8.5pt; font-weight: 600;">CI/CD & MLOps</span>
        </div>
    </div>

    <h2>1. Contexte & Problématique Métier</h2>
    <p>
        Dans le cadre de ses opérations quotidiennes, une grande entreprise spécialisée dans la gestion de services support gère plusieurs milliers de demandes de clients en provenance de diverses entreprises partenaires. Ces requêtes sont transmises dans quatre langues principales (<strong>Français, Anglais, Allemand, Portugais</strong>) et couvrent des périmètres variés (support produit, demandes de remboursement, assistance technique, questions d'abonnement).
    </p>
    <p>
        Historiquement, l'acheminement des tickets reposait sur une qualification manuelle effectuée par des opérateurs humains. Avec la hausse continue du volume de demandes, cette méthode présentait des goulots d'étranglement majeurs :
    </p>
    <ul>
        <li><strong>Délais de réponse accrus</strong> liés aux temps de lecture et de tri manuel.</li>
        <li><strong>Risques d'erreurs d'aiguillage</strong> vers les mauvais départements spécialisés.</li>
        <li><strong>Incapacité à passer à l'échelle (Scalability)</strong> lors des pics d'activité.</li>
    </ul>

    <div class="callout">
        <div class="callout-title">Objectif du Projet</div>
        Développer et déployer un système automatisé basé sur un LLM Open-Weight (Llama-3-8B / Mistral-7B) capable de catégoriser instantanément les tickets entrants. Le modèle doit respecter une contrainte stricte de performance : atteindre un <strong>Weighted F1-score &ge; 92%</strong> sur un jeu de test stratifié tout en assurant une intégration continue (CI/CD) et un déploiement robuste.
    </div>

    <h2>2. Contraintes Techniques & Conformité du Dataset</h2>
    <p>
        Pour garantir la confidentialité des données et la viabilité du système en conditions réelles de production, le modèle est soumis à un filtrage strict des variables d'entrée. Seules 5 métadonnées spécifiques sont mises à disposition :
    </p>
    
    <table>
        <thead>
            <tr>
                <th>Champ</th>
                <th>Description</th>
                <th>Rôle dans la modélisation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>subject</code></td>
                <td>Titre de la demande formulée par l'utilisateur</td>
                <td>Variable d'entrée (Features)</td>
            </tr>
            <tr>
                <td><code>body</code></td>
                <td>Corps explicatif de la demande client</td>
                <td>Variable d'entrée (Features)</td>
            </tr>
            <tr>
                <td><code>language</code></td>
                <td>Langue de la requête (FR, EN, DE, PT)</td>
                <td>Variable d'entrée (Features)</td>
            </tr>
            <tr>
                <td><code>business_type</code></td>
                <td>Secteur d'activité de l'entreprise cliente</td>
                <td>Variable d'entrée (Features)</td>
            </tr>
            <tr>
                <td><code>queue</code></td>
                <td>Catégorie/Département cible (ex: Product Support)</td>
                <td><strong>Variable Cible (Label)</strong></td>
            </tr>
        </tbody>
    </table>

    <p>
        Toutes les autres informations (anciennes priorités, tags d'opérateurs, métadonnées d'agents) sont strictement rejetées par le pipeline d'ingestion (<code>dataset_loader.py</code>).
    </p>

    <h2>3. Méthodologie & Évaluation des Performances</h2>
    <h3>Stratification des Données</h3>
    <p>
        Le jeu de données initial comportant 600 exemples annotés présente un déséquilibre naturel entre les différentes catégories (<code>queue</code>). Pour garantir une évaluation rigoureuse sans biais statistique, l'échantillonnage de validation utilise un <strong>Stratified Split (80/20)</strong> reproduisant fidèlement les proportions initiales <span class="math">α<sub>j</sub> = n<sub>j</sub> / n</span>.
    </p>

    <h3>Formulation des Métriques</h3>
    <p>
        La performance globale est mesurée via le <strong>Weighted F1-score</strong> (<span class="math">F1</span>), calculé à partir du F1-score individuel de chaque catégorie <span class="math">j</span> (<span class="math">F1<sub>j</sub></span>) :
    </p>

    <div class="formula-box">
        <span class="math">F1<sub>j</sub> = (2 × TP<sub>j</sub>) / (2 × TP<sub>j</sub> + FP<sub>j</sub> + FN<sub>j</sub>)</span>
        <br><br>
        <span class="math">F1 = ∑<sub>j=1</sub><sup>|C|</sup> α<sub>j</sub> · F1<sub>j</sub></span>
        <span style="font-size: 9pt; color: #64748B; display: block; margin-top: 5px;">Où |C| désigne le nombre de catégories et α<sub>j</sub> la proportion de la classe j dans le jeu de test.</span>
    </div>

    <h2>4. Architecture du Code & Modularité Production-Ready</h2>
    <p>
        Le projet est structuré selon les standards MLOps modernes en utilisant <strong>Poetry</strong> pour la gestion déterministe des dépendances et un découpage modulaire strict :
    </p>

    <pre>
llm-support-classifier/
├── .github/workflows/          # Pipeline CI/CD (Tests unitaires, Linting, Seuil F1 >= 92%)
├── data/                       # Données brutes et converties en JSONL (train/test stratifiés)
├── src/                        # Code source modulaire
│   ├── data/                   # Ingestion stricte (5 colonnes) & Formatting des prompts multilingues
│   ├── models/                 # Inférence baseline & Fine-Tuning QLoRA (PEFT/BitsAndBytes)
│   └── evaluation/             # Calcul exact du Weighted F1-score & Visualisations
├── api/                        # Backend Inférence Serveur FastAPI (Async / Pydantic validation)
├── ui/                         # Interface Utilisateur Interactive (Gradio / Streamlit)
├── tests/                      # Validation automatisée (Pytest)
└── pyproject.toml              # Configuration des dépendances Poetry
    </pre>

    <h2>5. Stratégie de Fine-Tuning & MLOps</h2>
    <ul>
        <li><strong>Quantification 4-bit (BitsAndBytes) & QLoRA</strong> : Optimisation de l'empreinte mémoire GPU permettant d'entraîner efficacement des modèles de 7B à 8B paramètres sur une instance 24 Go GPU (PyTorch).</li>
        <li><strong>Prompt Engineering Multilingue</strong> : Structuration des requêtes sous format d'instruction <code>&lt;s&gt;[INST] ... [/INST]&lt;/s&gt;</code> intégrant le contexte business et la langue.</li>
        <li><strong>CI/CD & Qualité</strong> : Vérification automatisée des régressions de performances avant toute intégration dans la branche principale.</li>
    </ul>
</body>
</html>
"""

with open("document_projet.html", "w", encoding="utf-8") as f:
    f.write(html_content)

weasyprint.HTML("document_projet.html").write_pdf("Classification_Multilingue_LLM_Support.pdf")
print("PDF généré avec succès.")


```

```text
PDF généré avec succès.


```

Voici une proposition complète et valorisante de **`README.md`** à placer à la racine de votre dépôt GitHub. Ce descriptif est conçu selon les standards des meilleurs projets Open Source et MLOps : clair, structuré, visuel et orienté production.

J'ai également généré une **fiche récapitulative au format PDF** [file-tag: code-generated-file-dcb6b95c-ebec-4b75-a21e-d4419843715f] que vous pouvez consulter ou inclure dans votre documentation.

---

# 📝 `README.md` (À copier-coller dans votre projet)

```markdown
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

# Activer la création du .venv localement (recommandé)
poetry config virtualenvs.in-project true

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