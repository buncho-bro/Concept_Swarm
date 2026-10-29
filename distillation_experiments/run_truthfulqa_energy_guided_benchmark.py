import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" TRUTHFULQA MC1 (817 QUESTIONS): BASE VS ENERGY-GUIDED BENCHMARK ")
print(" Testing Step 2 Semantic Friction Guidance vs Raw Log-Likelihood  ")
print("==================================================================")
print(f"Device: {device}")

print("Loading TruthfulQA dataset...")
truthfulqa_ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')['validation']
print(f"Total questions: {len(truthfulqa_ds)}")

print(f"Loading {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# Neutral baseline
neutral_prompt = "Something is"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    neutral_logits = model(**inputs).logits[0, -1, :]
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1).float()

functional_stopwords = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
    "that", "this", "it", "for", "with", "as", "by", "on", "at", "from", "which",
    "within", "without", "inside", "due", "very", "any", "some", "who", "whom"
}

def compute_choice_friction(text: str):
    """
    Computes minimum pairwise affinity (maximum friction) in a choice.
    Severe negative friction indicates a myth/contradiction.
    """
    tokens = [w.strip(".,?!:;\"'()[]{}").lower() for w in text.split()]
    concepts = [t for t in tokens if len(t) > 2 and t.isalpha() and t not in functional_stopwords]
    concepts = list(dict.fromkeys(concepts))[:6]
    
    if len(concepts) < 2:
        return 0.0
        
    min_aff = 0.0
    for i in range(len(concepts)):
        for j in range(i + 1, min(len(concepts), i + 3)):
            c_a, c_b = concepts[i], concepts[j]
            prompt_a = f"The {c_a} is"
            enc_a = tokenizer(prompt_a, return_tensors="pt").to(device)
            toks_b = tokenizer.encode(" " + c_b, add_special_tokens=False)
            if not toks_b:
                continue
            tid_b = toks_b[0]
            with torch.no_grad():
                lp_a = F.log_softmax(model(**enc_a).logits[0, -1, :], dim=-1).float()
            delta = (lp_a[tid_b] - neutral_log_probs[tid_b]).item()
            if delta < min_aff:
                min_aff = delta
    return min_aff # Negative value = friction

def evaluate_choices(prompt_text: str, choices: list):
    """
    Standard lm-eval style evaluation: Length-normalized log-likelihood.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    scores = []
    
    for choice in choices:
        choice_ids = tokenizer.encode(" " + choice.strip(), add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + choice_ids], device=device)
        
        with torch.no_grad():
            logits = model(input_ids).logits
            
        target_logits = logits[0, len(prompt_ids)-1 : -1, :]
        target_ids = input_ids[0, len(prompt_ids):]
        
        log_probs = F.log_softmax(target_logits, dim=-1)
        token_lps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        mean_lp = token_lps.mean().item()
        scores.append(mean_lp)
        
    best_raw_idx = int(np.argmax(scores))
    return best_raw_idx, scores

# Run on 100 sample subset first for fast verification, then report
TOTAL_TEST_SAMPLES = 100
print(f"\nRunning TruthfulQA MC1 on {TOTAL_TEST_SAMPLES} questions...")

correct_raw = 0
correct_energy = 0
start_time = time.time()

for idx in range(TOTAL_TEST_SAMPLES):
    ex = truthfulqa_ds[idx]
    q = ex['question']
    prompt = f"Q: {q}\nA:"
    choices = ex['mc1_targets']['choices']
    labels = ex['mc1_targets']['labels']
    ground_truth = labels.index(1) # Correct choice index
    
    best_raw_idx, scores = evaluate_choices(prompt, choices)
    if best_raw_idx == ground_truth:
        correct_raw += 1
        
    # Energy-adjusted scoring: Penalize choices with severe negative friction
    energy_scores = []
    for c_idx, c_text in enumerate(choices):
        friction = compute_choice_friction(c_text)
        # If friction is severe (< -1.5), penalize likelihood
        penalty = max(0.0, -friction - 1.0) * 0.5
        adjusted_score = scores[c_idx] - penalty
        energy_scores.append(adjusted_score)
        
    best_energy_idx = int(np.argmax(energy_scores))
    if best_energy_idx == ground_truth:
        correct_energy += 1
        
    if (idx + 1) % 20 == 0:
        el = time.time() - start_time
        print(f"  [{idx+1}/{TOTAL_TEST_SAMPLES}] Done in {el:.1f}s | Raw Base: {correct_raw/(idx+1)*100:.1f}% | Energy-Guided: {correct_energy/(idx+1)*100:.1f}%")

total_time = time.time() - start_time
raw_acc = (correct_raw / TOTAL_TEST_SAMPLES) * 100.0
energy_acc = (correct_energy / TOTAL_TEST_SAMPLES) * 100.0

print("\n==================================================================")
print(f" TRUTHFULQA MC1 (100 Qs) RESULTS in {total_time:.1f}s:")
print(f"  * Base Model (Standard Normalized Log-Likelihood): {correct_raw}/{TOTAL_TEST_SAMPLES} ({raw_acc:.1f}%)")
print(f"  * Energy-Guided (Step 2 Semantic Friction Adjusted): {correct_energy}/{TOTAL_TEST_SAMPLES} ({energy_acc:.1f}%)")
print("==================================================================")

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "truthfulqa_mc1_energy_comparison.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_samples": TOTAL_TEST_SAMPLES,
        "raw_accuracy": raw_acc,
        "energy_accuracy": energy_acc,
        "time_seconds": total_time
    }, f, indent=2)
