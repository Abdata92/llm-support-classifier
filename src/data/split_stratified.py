import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["queue"], random_state=seed
    )

    print(f"[Split] Train : {len(train_df)} échantillons | Test : {len(test_df)} échantillons")
    return train_df, test_df