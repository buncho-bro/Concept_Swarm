import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" INVESTIGATING INTRINSIC INFLUENCE: 'Stella' <-> 'bright'         ")
print(" AND UNDERSTANDING OF QUERY REQUIREMENTS                         ")
print("==================================================================")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# Neutral Baseline
neutral_prompt = "Something is"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    neutral_logits = model(**inputs).logits[0, -1, :]
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1)

def measure_intrinsic_influence(name: str, attribute: str):
    """
    Measures the model's raw pre-trained influence between a character name and an attribute:
      Delta = log P(attribute | Name is) - log P(attribute | Neutral)
    """
    prompt = f"{name} is"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    
    attr_tokens = tokenizer.encode(" " + attribute, add_special_tokens=False)
    if not attr_tokens:
        return 0.0, 0.0, 0.0
    tid = attr_tokens[0]
    
    with torch.no_grad():
        lp = F.log_softmax(model(**inp).logits[0, -1, :], dim=-1)
        
    p_neutral = torch.exp(neutral_log_probs[tid]).item()
    p_name = torch.exp(lp[tid]).item()
    delta = lp[tid].item() - neutral_log_probs[tid].item()
    ratio = p_name / (p_neutral + 1e-12)
    return delta, p_neutral, p_name, ratio

# ----------------------------------------------------------------------
# 1. Measure Pairwise Influence across ProntoQA Entities
# ----------------------------------------------------------------------
pairs_to_test = [
    ("Stella", "bright"),     # Q1 (Latin for 'star' -> bright?)
    ("Stella", "dark"),
    ("Max", "sour"),          # Q0
    ("Max", "sweet"),
    ("Wren", "metallic"),     # Q2 (Wren is a bird)
    ("Wren", "dull"),         # Q5
    ("Alex", "small"),        # Q3
    ("Alex", "large")
]

print("\n--- Intrinsic Pre-trained Influence (Name -> Attribute without Context) ---")
for name, attr in pairs_to_test:
    delta, p_neut, p_name, ratio = measure_intrinsic_influence(name, attr)
    print(f"  {name:8s} -> {attr:10s} | Delta: {delta:+6.2f} | Prob: {p_name*100:6.3f}% vs Neutral: {p_neut*100:6.3f}% (x{ratio:.2f})")

# ----------------------------------------------------------------------
# 2. Inspect How Model Understands the Query Requirement Alone
# ----------------------------------------------------------------------
print("\n--- Model Reaction to Query ALONE (Zero-Shot Direct Query without Context) ---")
queries = [
    "Is the following statement true or false? Stella is bright.\nAnswer:",
    "Is the following statement true or false? Max is sour.\nAnswer:",
    "Is the following statement true or false? Wren is metallic.\nAnswer:",
    "Is the following statement true or false? Alex is small.\nAnswer:"
]

for q in queries:
    inp = tokenizer(q, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :]
        lps = F.log_softmax(logits, dim=-1)
        
    t_id = tokenizer.encode(" True", add_special_tokens=False)[0]
    f_id = tokenizer.encode(" False", add_special_tokens=False)[0]
    
    p_true = torch.exp(lps[t_id]).item()
    p_false = torch.exp(lps[f_id]).item()
    sum_tf = p_true + p_false
    norm_t = p_true / sum_tf * 100.0
    norm_f = p_false / sum_tf * 100.0
    
    clean_q = q.replace("\nAnswer:", "")
    print(f"\nQuery: '{clean_q}'")
    print(f"  P(True): {norm_t:5.1f}% | P(False): {norm_f:5.1f}% | Raw Delta(T-F): {lps[t_id].item() - lps[f_id].item():+.2f}")

# ----------------------------------------------------------------------
# 3. Analyze ProntoQA Dataset Query Patterns (First 20 Samples)
# ----------------------------------------------------------------------
print("\n--- Analyzing Query Requirements across ProntoQA Dataset (First 20 Qs) ---")
ds = load_dataset('renma/ProntoQA')['validation'].select(range(20))
negated_queries = 0
affirmed_queries = 0
ground_truth_counts = {"A (True)": 0, "B (False)": 0}

for i, ex in enumerate(ds):
    q = ex['question']
    gt = ex['answer'].strip()
    ground_truth_counts[f"{gt} ({'True' if gt=='A' else 'False'})"] += 1
    
    if " not " in q.lower():
        negated_queries += 1
    else:
        affirmed_queries += 1

print(f"Total samples checked: 20")
print(f"  Affirmative Queries (e.g. 'X is Y'):     {affirmed_queries}")
print(f"  Negated Queries     (e.g. 'X is not Y'): {negated_queries}")
print(f"  Ground Truth Distribution: {ground_truth_counts}")
