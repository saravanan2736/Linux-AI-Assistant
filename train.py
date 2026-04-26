import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ---------------------------
# Model
# ---------------------------
model_name = "microsoft/phi-2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ---------------------------
# LoRA Config
# ---------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print("\nTrainable parameters:")
model.print_trainable_parameters()

# ---------------------------
# Dataset (CodeAlpaca)
# ---------------------------
print("\nLoading CodeAlpaca dataset...")
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

# Limit dataset for faster training
dataset = dataset.select(range(300))

# ---------------------------
# Format dataset
# ---------------------------
def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"]

    if input_text:
        instruction = instruction + " " + input_text

    response = example["output"]

    return {
        "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    }

dataset = dataset.map(format_example)

# ---------------------------
# Tokenization
# ---------------------------
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=192
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names
)

# ---------------------------
# Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="no"
)

# ---------------------------
# Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ---------------------------
# Train
# ---------------------------
print("\nStarting training...\n")
trainer.train()

# ---------------------------
# Save LoRA Adapter
# ---------------------------
model.save_pretrained("lora_adapter")

print("\nTraining complete.")
print("Adapter saved to ./lora_adapter")
