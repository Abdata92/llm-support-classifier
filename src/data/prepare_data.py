from pathlib import Path
import sys

# Ajout dynamique de la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset_loader import load_and_filter_dataset
from src.data.prompt_formatter import export_to_jsonl
from src.data.split_stratified import create_stratified_split


def main():
    raw_path = "data/raw/helpdesk_customer_tickets.csv"
    train_out = "data/processed/train.jsonl"
    test_out = "data/processed/test.jsonl"

    print("🚀 Execution du pipeline de donnees...\n")
    df = load_and_filter_dataset(raw_path)
    train_df, test_df = create_stratified_split(df)
    export_to_jsonl(train_df, train_out)
    export_to_jsonl(test_df, test_out)
    print("\n✅ Termine ! Fichiers train.jsonl et test.jsonl generes.")


if __name__ == "__main__":
    main()