import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Num hidden layers: {num_layers}")
print(f"Hidden size: {hidden_size}")
print(f"Layers: model.model.layers[0] ... model.model.layers[{num_layers-1}]")

# Layer 6, 14, 22 の確認
for l_idx in [num_layers // 4, num_layers // 2, (num_layers * 3) // 4]:
    layer_mod = model.model.layers[l_idx]
    print(f"Layer {l_idx}: {layer_mod.__class__.__name__}")
