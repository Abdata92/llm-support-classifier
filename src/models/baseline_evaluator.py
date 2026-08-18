import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.metrics import print_evaluation_report

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def load_test_data(test_path: str):
    prompts, y_true = [], []
    
    # 1. Extraire d'abord la liste unique de toutes les catégories réelles
    with open(test_path, "r", encoding="utf-8") as f:
        all_queues = list(set(json.loads(line)["queue"] for line in f))
    
    queues_str = "\n".join([f"- {q}" for q in all_queues])

    # 2. Construire des prompts contraignants avec la liste des choix
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            base_prompt = item["text"].split("[/INST]")[0]
            
            strict_prompt = (
                f"{base_prompt}\n\n"
                f"CHOIX POSSIBLES :\n{queues_str}\n\n"
                "Consigne : Répondez UNIQUEMENT et EXACTEMENT par l'un des choix ci-dessus, sans aucun autre mot.[/INST]"
            )
            prompts.append(strict_prompt)
            y_true.append(item["queue"])
            
    return prompts, y_true, all_queues


def parse_prediction(raw_text: str, valid_queues: list) -> str:
    cleaned = raw_text.strip()
    # Recherche de la classe valide la plus proche présente dans le texte
    for q in valid_queues:
        if q.lower() in cleaned.lower():
            return q
    # En cas d'échec total, retourner la première classe par défaut
    return valid_queues[0]


def run_baseline_evaluation(test_path: str = "data/processed/test.jsonl"):
    print(f"🚀 Chargement du modèle baseline : {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )

    prompts, y_true, valid_queues = load_test_data(test_path)
    y_pred = []

    print("⏳ Inférence sur le jeu de test stratifié...")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id
            )
        raw_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        
        pred_label = parse_prediction(raw_text, valid_queues)
        y_pred.append(pred_label)

    print_evaluation_report(y_true, y_pred, model_name="Baseline (Zero-Shot)")


if __name__ == "__main__":
    run_baseline_evaluation()