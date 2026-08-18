import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import print_evaluation_summary

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "models/final_qlora_adapter"


def load_test_data(test_path: str = "data/processed/test.jsonl"):
    prompts, y_true = [], []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            full_text = item["text"]

            # Reconstruction exacte du prompt d'instruction sans la réponse cible
            if "[/INST]" in full_text:
                prompt_part = full_text.split("[/INST]")[0] + "[/INST]"
            else:
                prompt_part = full_text

            prompts.append(prompt_part)
            y_true.append(item["queue"])
    return prompts, y_true


def run_finetuned_evaluation(test_path: str = "data/processed/test.jsonl"):
    print("🚀 Chargement du modèle de base et de l'adaptateur QLoRA...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = (
        "left"  # Indispensable pour l'inférence autoregressive
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    prompts, y_true = load_test_data(test_path)
    y_pred = []

    print(
        f"⏳ Inférence Fine-Tunée sur les {len(prompts)} tickets de test stratifiés..."
    )

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Inférence déterministe
            )

        # Extraction uniquement des nouveaux tokens générés après [/INST]
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        prediction = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        # Nettoyage de la chaîne générée
        cleaned_pred = prediction.split("\n")[0].replace("</s>", "").strip()
        y_pred.append(cleaned_pred)

        if (i + 1) % 20 == 0 or (i + 1) == len(prompts):
            print(f" Progression : {i + 1}/{len(prompts)} tickets traités")

    print_evaluation_summary(y_true, y_pred)


if __name__ == "__main__":
    run_finetuned_evaluation()