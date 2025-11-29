from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import torch
import os

dataset_path = "./tokenized_ais_dataset"
model_name = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
max_length = 1024
output_dir = "./lora_finetuned_ais"
save_strategy = "steps"
save_steps = 500
logging_steps = 50
batch_size = 2
learning_rate = 3e-4
num_train_epochs = 3
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(dataset_path):
    dataset = load_from_disk(dataset_path)
else:
    dataset = load_dataset("nolanplatt/ais-qa-dataset")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["query_key_value"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)

def tokenize_fn(examples):
    return tokenizer(
        examples["question"] + "\n" + examples["answer"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

config = SFTConfig(
    train_dataset=tokenized_dataset,
    model=model,
    peft_config=lora_config,
    max_seq_length=max_length,
    batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    logging_steps=logging_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    output_dir=output_dir
)

trainer = SFTTrainer(config)
trainer.train()
trainer.save_model(output_dir)
