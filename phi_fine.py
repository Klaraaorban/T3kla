from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch.nn as nn

model_name = "microsoft/phi-3.5-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

linear_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

lora_config = LoraConfig(from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer
import torch

base_model_name = "microsoft/phi-3.5-mini-instruct"
checkpoint_dir = "./phi3-ais-ft/final"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"
)

# **load the LoRA checkpoint directly**
model = PeftModel.from_pretrained(model, checkpoint_dir)
model.eval()

dataset = load_dataset("nolanplatt/ais-qa-dataset")["train"]

def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=True
    )

training_args = TrainingArguments(
    output_dir="./phi3-ais-ft-continued",
    per_device_train_batch_size=2,  # lower to avoid OOM
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear",
    warmup_steps=100,
    max_grad_norm=1.0,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args
)

trainer.train()
trainer.save_model("./phi3-ais-ft-continued/final")
tokenizer.save_pretrained("./phi3-ais-ft-continued/final")

    r=16,
    lora_alpha=32,
    target_modules=linear_layers,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("nolanplatt/ais-qa-dataset")["train"]

def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=True
    )

training_args = TrainingArguments(
    output_dir="./phi3-ais-ft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear",
    warmup_steps=100,
    max_grad_norm=1.0,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args
)

trainer.train()
trainer.save_model("./phi3-ais-ft/final")
tokenizer.save_pretrained("./phi3-ais-ft/final")
