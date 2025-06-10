import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk

def main():
    # ===========================
    # Environment & Device Info
    # ===========================
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("Available GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i+1}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===========================
    # Setup cache & scratch space
    # ===========================
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    cache_dir = os.path.join(tmp_dir, "seansal")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # ===========================
    # Load model & tokenizer
    # ===========================
    model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token  # Required for llama models

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"   # optional if you later switch to accelerate
    )

    # ===========================
    # Load datasets
    # ===========================
    train_dataset = load_from_disk("/projectnb/scottml/seansal2/data/datasets/blood_memory_clm_train")
    test_dataset = load_from_disk("/projectnb/scottml/seansal2/data/datasets/blood_memory_clm_test")
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])
    print("Dataset columns:", train_dataset.column_names)

    # ===========================
    # Data Collator
    # ===========================
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ===========================
    # Training Arguments
    # ===========================
    training_args = TrainingArguments(
        output_dir=f"./marthabot_models/llama2_7b_finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2,
        bf16=True,
        logging_dir="./logs",
        logging_steps=50,
    )

    # ===========================
    # Trainer
    # ===========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ===========================
    # Train & Save
    # ===========================
    torch.cuda.empty_cache()

    print("Starting training ...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(f"./marthabot_models/llama2_7b_finetuned")
    tokenizer.save_pretrained(f"./marthabot_models/llama2_7b_finetuned")
    print("Model and tokenizer saved.")

# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    main()
