import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

vocab_size = tokenizer.vocab_size
hidden_size = model.config.hidden_size
print(f"Model loaded successfully on {model.device}!")
print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")
print(f"VRAM used: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB")
