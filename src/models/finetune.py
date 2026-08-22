import accelerate
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# 🛠️ PATCH : Correctif pour le conflit de signature unwrap_model entre Transformers & Accelerate
_orig_unwrap = accelerate.Accelerator.unwrap_model


def _patched_unwrap(self, model, *args, **kwargs):
    kwargs.pop("keep_torch_compile", None)
    return _orig_unwrap(self, model, *args, **kwargs)


accelerate.Accelerator.unwrap_model = _patched_unwrap


def train_qlora():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # 1. Configuration BitsAndBytes 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Chargement du Modèle et Tokenizer
    print("🚀 Chargement du modèle de base Mistral-7B...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Préparation du modèle pour QLoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Chargement des données d'entraînement
    dataset = load_dataset(
        "json", data_files={"train": "data/processed/train.jsonl"}
    )

    # 5. Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir="models/qlora_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("🔥 Lancement du Fine-Tuning QLoRA...")
    trainer.train()

    # 7. Sauvegarde de l'adaptateur final
    print("💾 Sauvegarde de l'adaptateur QLoRA...")
    trainer.model.save_pretrained("models/final_qlora_adapter")
    tokenizer.save_pretrained("models/final_qlora_adapter")
    print("✅ Fine-tuning terminé avec succès !")


if __name__ == "__main__":
    train_qlora()