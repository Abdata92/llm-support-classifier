from pathlib import Path
import pandas as pd

ALLOWED_COLUMNS = ["subject", "body", "queue", "language", "business_type"]


def load_and_filter_dataset(raw_csv_path: str) -> pd.DataFrame:
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {raw_csv_path}")

    df = pd.read_csv(path)

    missing = [c for c in ALLOWED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes dans le CSV : {missing}")

    # Conservation stricte des 5 colonnes autorisees
    df_filtered = df[ALLOWED_COLUMNS].copy()
    df_filtered.dropna(subset=["subject", "body", "queue"], inplace=True)

    print(f"[Dataset] {len(df_filtered)} lignes retenues sur {len(df)} originales.")
    return df_filtered