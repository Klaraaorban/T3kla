#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from itertools import chain

# In[2]:


file_path = "dataset/chunk_2.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()
paragraphs = [p.strip() for p in text_data.split("\n\n") if p.strip()]
print(f"Loaded {len(paragraphs)} paragraphs.")
dataset = Dataset.from_dict({"text": paragraphs})


# In[3]:


model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[4]:


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("done")


# In[6]:


from tqdm import tqdm

batch_size = 1000
all_tokenized = []

for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i+batch_size]
    tokenized = tokenizer(batch["text"], truncation=True, max_length=512, padding=False)
    all_tokenized.append(tokenized)

# Merge all batches
input_ids = list(chain.from_iterable([t["input_ids"]  for t in all_tokenized]))
attention_mask = list(chain.from_iterable([t["attention_mask"] for t in all_tokenized]))

# Convert to HuggingFace Dataset
tokenized_dataset = Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": attention_mask
})

print(f"Tokenized dataset ready: {len(tokenized_dataset)} sequences")


# In[10]:


def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    chunk_size = 512
    chunks = [concatenated[i:i+chunk_size] for i in range(0, len(concatenated), chunk_size)]

    return {
        "input_ids": chunks,
        "attention_mask": [[1]*len(c) for c in chunks]
    }

# Use batching to avoid memory issues
tokenized_dataset_grouped = tokenized_dataset.map(
    group_texts, 
    batched=True, 
    batch_size=1000,  # adjust if needed
    desc="Grouping sequences"
)

print(f"Dataset ready with {len(tokenized_dataset_grouped)} sequences of 512 tokens each.")


# In[14]:


from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
checkpoint_path = "./finetuned_redpajama/checkpoint-5625"

# Load model in 8-bit to save GPU memory
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)

model = PeftModel.from_pretrained(
	base_model,
	checkpoint_path
)

print("model with checkpoint loaded")


# In[15]:


training_args = TrainingArguments(
    output_dir="./finetuned_redpajama",
    per_device_train_batch_size=2,    # adjust to fit your GPU
    gradient_accumulation_steps=8,    # effective batch size = batch_size * accum_steps
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,                        # mixed precision
    logging_steps=50,
    save_steps=500,
    save_total_limit=2
)


# In[19]:


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_grouped,
    data_collator=data_collator
)

trainer.train(resume_from_checkpoint="./finetuned_redpajama/checkpoint-5625")


# In[ ]:


model.save_pretrained("./finetuned_redpajama")
tokenizer.save_pretrained("./finetuned_redpajama")


