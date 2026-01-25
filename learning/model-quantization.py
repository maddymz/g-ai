from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Force CPU for compatibility/testing (set to 'mps' or 'cuda' if desired)
device = "cpu"
print(f"Using device: {device}")

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_enable_fp32_cpu_offload = True
)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": "cpu"}
)

param_dtypes = [param.dtype for param in quantized_model.parameters()]
print("Parameter dtypes:", param_dtypes)
print(f"Memory footprint: {quantized_model.get_memory_footprint() / 1e9:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer("Portugal is", return_tensors="pt").to(device)

response = quantized_model.generate(**input, max_new_tokens = 50)
print(tokenizer.batch_decode(response, skip_special_tokens=True))