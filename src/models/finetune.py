import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def train_qlora():
    print("🚀 Initialisation du Fine-Tuning QLoRA...")

    # 1. Quantification 4-bit (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 512

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    # 2. Configuration PEFT / LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. Charger le jeu de données d'entraînement JSONL
    dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl"})

    # 4. Configuration SFTConfig (compatible TRL v0.12+)
    sft_config = SFTConfig(
        dataset_text_field="text",
        output_dir="models/qlora_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        max_length=512,  # Paramètre de longueur propre à SFTConfig dans les versions récentes
    )

    # 5. Instanciation unifiée du Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("🔥 Lancement de l'entraînement...")
    trainer.train()

    # 6. Sauvegarde des poids adaptateurs LoRA
    print("💾 Sauvegarde de l'adaptateur final...")
    trainer.model.save_pretrained("models/final_qlora_adapter")
    tokenizer.save_pretrained("models/final_qlora_adapter")
    print("✅ Entraînement terminé avec succès !")


if __name__ == "__main__":
    train_qlora()