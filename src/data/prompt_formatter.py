import json
from pathlib import Path
import pandas as pd


def create_instruction_prompt(row: pd.Series) -> str:
    system_prompt = (
        "Vous etes un assistant expert en classification de tickets support. "
        "Votre tache est de categoriser le message entrant dans l'unique queue appropriee."
    )
    user_input = (
        f"Langue : {row['language']}\n"
        f"Secteur d'activite : {row['business_type']}\n"
        f"Objet : {row['subject']}\n"
        f"Message : {row['body']}\n\n"
        "Quelle est la queue correspondant a ce ticket ?"
    )
    return f"<s>[INST] {system_prompt}\n\n{user_input} [/INST] {row['queue']}</s>"


def export_to_jsonl(df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in df.iterrows():
        records.append(
            {"text": create_instruction_prompt(row), "queue": row["queue"]}
        )
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Export] Sauvegarde dans : {output_path}")