from IPython.display import display
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
import transformers
from datasets import load_dataset
import peft
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os
import torch

# Auto-detect best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


dataset = "openai/gsm8k"
data = load_dataset(dataset, 'main')

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Quantization with training only works reliably on CUDA
# For MPS/CPU, use FP16 without quantization
if torch.cuda.is_available():
    print("Loading model with 8-bit quantization (CUDA)")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    quantized_model = prepare_model_for_kbit_training(
        quantized_model,
        use_gradient_checkpointing=True
    )
else:
    print(f"Loading model in FP16 for {device} (quantization training not fully supported on MPS/CPU)")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        low_cpu_mem_usage=True
    )
    quantized_model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

input = tokenizer("Natalia sold clips to 48 of her friends in April, and then she sold half as \
many clips in May. How many clips did Natalia sell altogether in April and May?", return_tensors="pt").to(device)

# Format dataset for SFTTrainer - combine question and answer and tokenize
def format_and_tokenize(sample):
    text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

data = data.map(format_and_tokenize, batched=False, remove_columns=data["train"].column_names)
train_sample = data["train"].select(range(400))

# LoRA configurations
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
quantized_model = get_peft_model(quantized_model, lora_config)
quantized_model.print_trainable_parameters()

working_dir = './'
output_directory = os.path.join(working_dir, "lora")

training_args = TrainingArguments(
    output_dir = output_directory,
    auto_find_batch_size = True,
    learning_rate = 3e-4,
    num_train_epochs = 5,
    gradient_checkpointing = True,
    save_steps = 100,
    logging_steps = 10,
)

## set the trainer
# Data collator for causal language modeling
data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

trainer = Trainer(
    model = quantized_model,
    args = training_args,
    train_dataset = train_sample,
    data_collator = data_collator
)

print("start training")
trainer.train()