#!/usr/bin/env python
# coding: utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset
from itertools import chain
from tqdm import tqdm
import torch

# -------------------------------
# Load chunk 2 dataset
# -------------------------------
file_path = "merged_chunks.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()
paragraphs = [p.strip() for p in text_data.split("\n\n") if p.strip()]
dataset = Dataset.from_dict({"text": paragraphs})
print(f"Loaded {len(paragraphs)} paragraphs.")

# -------------------------------
# Tokenizer
# -------------------------------
model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# Tokenize dataset
# -------------------------------
batch_size = 1000
all_tokenized = []

for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i+batch_size]
    tokenized = tokenizer(batch["text"], truncation=True, max_length=512, padding=False)
    all_tokenized.append(tokenized)

input_ids = list(chain.from_iterable([t["input_ids"] for t in all_tokenized]))
attention_mask = list(chain.from_iterable([t["attention_mask"] for t in all_tokenized]))

tokenized_dataset = Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": attention_mask
})

# -------------------------------
# Group into 512-token chunks
# -------------------------------
def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    chunk_size = 512
    chunks = [concatenated[i:i+chunk_size] for i in range(0, len(concatenated), chunk_size)]
    return {
        "input_ids": chunks,
        "attention_mask": [[1]*len(c) for c in chunks]
    }

tokenized_dataset_grouped = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    desc="Grouping sequences"
)

print(f"Dataset ready with {len(tokenized_dataset_grouped)} chunks of 512 tokens.")

# -------------------------------
# Load 8-bit base model
# -------------------------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

# -------------------------------
# Load LoRA checkpoint
# -------------------------------
checkpoint_path = "./finetuned_redpajama/checkpoint-5661"
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.train()  # <-- IMPORTANT

# -------------------------------
# Ensure LoRA parameters require gradients
# -------------------------------
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

# -------------------------------
# Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./finetuned_redpajama_final",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2
)

# -------------------------------
# Data collator
# -------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_grouped,
    data_collator=data_collator
)

# -------------------------------
# Train
# -------------------------------
trainer.train()  # Do NOT pass resume_from_checkpoint

# -------------------------------
# Save model & tokenizer
# -------------------------------
model.save_pretrained("./finetuned_redpajama_final")
tokenizer.save_pretrained("./finetuned_redpajama_final")
print("Training on chunk 2 complete.")

