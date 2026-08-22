#!/bin/bash
set -e

echo "======================================================="
echo "🚀 PIPELINE BOOTSTRAP - CLASSIFICATION LLM SUPPORT"
echo "======================================================="

# --- 1. CONFIGURATION POETRY & ENVIRONNEMENT ---
echo -e "\n🔧 1/5. Configuration de Poetry et dépendances..."
pip install poetry --quiet
poetry config virtualenvs.in-project true

# --- 2. INSTALLATION PYTORCH CUDA & NETTOYAGE DÉPENDANCES ---
echo -e "\n⚡ 2/5. Alignement des roues PyTorch CUDA..."
poetry run pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
poetry run pip install "fsspec>=2023.1.0,<=2024.6.1" "markupsafe>=2.0,<3.0" --quiet
poetry run pip install "accelerate>=0.26.0,<1.0.0" "bitsandbytes>=0.46.1" trl peft scikit-learn pandas triton --quiet

# Fix des permissions GPU UVM
mknod -m 666 /dev/nvidia-uvm c $(grep nvidia-uvm /proc/devices | awk '{print $1}') 0 2>/dev/null || true
mknod -m 666 /dev/nvidia-uvm-tools c $(grep nvidia-uvm /proc/devices | awk '{print $1}') 1 2>/dev/null || true
chmod 666 /dev/nvidia* 2>/dev/null || true

# Verification CUDA (en évitant le point d'exclamation)
poetry run python -c "import torch; assert torch.cuda.is_available(), 'CUDA INDISPONIBLE'; print('✅ GPU Détecté :', torch.cuda.get_device_name(0))"

# --- 3. PRÉPARATION DES DONNÉES ---
echo -e "\n📊 3/5. Exécution du pipeline de préparation des données..."
mkdir -p data/raw data/processed

# Alignement automatique des noms de fichiers CSV
if [ -f "data/raw/support_tickets_raw.csv" ] && [ ! -f "data/raw/helpdesk_customer_tickets.csv" ]; then
    cp data/raw/support_tickets_raw.csv data/raw/helpdesk_customer_tickets.csv
elif [ -f "data/raw/helpdesk_customer_tickets.csv" ] && [ ! -f "data/raw/support_tickets_raw.csv" ]; then
    cp data/raw/helpdesk_customer_tickets.csv data/raw/support_tickets_raw.csv
fi

poetry run python -m src.data.prepare_data

# --- 4. ÉVALUATION BASELINE (ZERO-SHOT) ---
echo -e "\n⚖️ 4/5. Évaluation Baseline (Zero-Shot)..."
poetry run python -m src.models.baseline_evaluator

# --- 5. FINE-TUNING QLORA ET ÉVALUATION FINALE ---
echo -e "\n🔥 5/5. Lancement du Fine-Tuning QLoRA..."
poetry run python -m src.models.finetune

echo -e "\n🎯 Évaluation finale du modèle Fine-Tuné..."
poetry run python -m src.models.evaluate_finetuned

echo -e "\n======================================================="
echo "🎉 PIPELINE EXÉCUTÉ AVEC SUCCÈS !"
echo "======================================================="