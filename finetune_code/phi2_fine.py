import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from trl import SFTTrainer
from safetensors.torch import load_file as safe_load_file 
import os

model_name = "microsoft/phi-3.5-mini-instruct"
checkpoint_dir = "./phi3-ais-ft/final"
adapter_weights_path = os.path.join(checkpoint_dir, "adapter_model.safetensors") 

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if torch.cuda.is_available():
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
)

if torch.cuda.is_available():
    model.to(torch.device("cuda"))

model.gradient_checkpointing_enable() 

linear_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

# --- CRITICAL FIX: REVERTED LoRA CONFIGURATION TO MATCH SAVED CHECKPOINT (r=16, alpha=32) ---
lora_config = LoraConfig(
    r=16,                      
    lora_alpha=32,             
    target_modules=linear_layers,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

if os.path.exists(adapter_weights_path):
    print(f"Loading previous adapter weights from {adapter_weights_path}...")
    adapters_weights = safe_load_file(adapter_weights_path) 
    set_peft_model_state_dict(model, adapters_weights)
    print("Previous training weights successfully injected for continuation.")
else:
    print(f"FATAL ERROR: Adapter weights not found at {adapter_weights_path}. Check filename/path.")


# --- LOAD DATASETS INCLUDING VALIDATION SPLIT ---
dataset_splits = load_dataset("nolanplatt/ais-qa-dataset")
train_dataset = dataset_splits["train"]
eval_dataset = dataset_splits["validation"] 

def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=True
    )

training_args = TrainingArguments(
    output_dir="./phi3-ais-ft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_steps=100,
    max_grad_norm=1.0,
    report_to="none",
    gradient_checkpointing=True,
    eval_strategy="steps", 
    eval_steps=20,               
    load_best_model_at_end=False 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,        
    eval_dataset=eval_dataset,          
    formatting_func=formatting_func,
    args=training_args
)

print("\nStarting continued training with validation monitoring...")
trainer.train() 

trainer.save_model(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)
print(f"\nTraining successfully completed and saved to {checkpoint_dir}")