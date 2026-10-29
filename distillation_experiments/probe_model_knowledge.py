import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" PROBING INTERNAL PARAMETRIC KNOWLEDGE OF QWEN2.5-1.5B           ")
print(" Does the model actually possess the factual knowledge?           ")
print("==================================================================")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

probe_questions = [
    # Vatican vs Nauru Area
    "What is the total area of Vatican City in square miles or square kilometers?",
    "What is the area of the island nation of Nauru in square miles?",
    "What is the smallest country in the world that has an area greater than one square mile?",
    
    # US Paid Vacation Law
    "Does federal law in the United States require employers to provide paid vacation to workers?",
    "How many days of statutory paid leave are guaranteed by law to employees in the United States?",
    
    # Left-Brain vs Right-Brain
    "Is the idea of left-brained versus right-brained thinkers supported by modern neuroscience?",
    
    # Blood Type & Leadership
    "Is there any scientific evidence that blood type determines leadership or personality?",
    
    # George Washington Cherry Tree
    "Did George Washington actually chop down a cherry tree and say 'I cannot tell a lie', or is it a myth?"
]

print(f"\nQuerying {BASE_MODEL_ID} directly with open-ended generation...\n")

for i, q in enumerate(probe_questions):
    messages = [
        {"role": "system", "content": "You are a precise, factual knowledge assistant. Provide short, accurate factual answers."},
        {"role": "user", "content": q}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0
        )
        
    response = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
    print(f"[{i+1}] Q: {q}")
    print(f"  --> Model Answer:\n      {response}\n")
