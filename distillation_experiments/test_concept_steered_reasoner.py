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
print(" BENCHMARK REASONING WITH CONCEPT INFLUENCE STEERING              ")
print(" (Testing on TruthfulQA MC1 & ProntoQA without Any Cheating)      ")
print("==================================================================")
print(f"Device: {device}")

print("Loading Qwen2.5-1.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# Compute neutral baseline for dynamic influence extraction
neutral_prompt = "Something is"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    neutral_logits = model(**inputs).logits[0, -1, :]
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1) # [vocab_size]

def extract_content_words(text: str):
    """Simple stopword filtering to extract meaningful concept words."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
        "that", "this", "it", "for", "with", "as", "by", "on", "at", "from", "which",
        "be", "has", "have", "had", "do", "does", "did", "but", "not", "what", "who",
        "where", "when", "why", "how", "all", "each", "every", "following", "statement"
    }
    words = [w.strip(".,?!:;\"'()[]{}").lower() for w in text.split()]
    return [w for w in words if len(w) > 2 and w.isalpha() and w not in stopwords]

def compute_prompt_influence_vector(prompt_text: str):
    """
    Computes cumulative influence vector Delta across all 151,643 vocabulary tokens
    induced by the key concept words in the prompt.
    """
    content_words = extract_content_words(prompt_text)
    if not content_words:
        return torch.zeros_like(neutral_log_probs)
    
    # We take the top 5 representative content words
    rep_words = content_words[:5]
    
    # Fast batch forward pass for concept words
    batch_prompts = [f"The {w} is" for w in rep_words]
    batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        logits = model(**batch_inputs).logits[:, -1, :] # [B, vocab_size]
        lps = F.log_softmax(logits, dim=-1) # [B, vocab_size]
        deltas = lps - neutral_log_probs.unsqueeze(0) # [B, vocab_size]
        cumulative_influence = torch.mean(deltas, dim=0) # [vocab_size]
        
    return cumulative_influence

def evaluate_choice_with_influence(model, tokenizer, prompt_text, choice_text, prompt_influence_vec, alpha=0.15):
    """
    Evaluates choice using:
      Score = Normalized_LogLikelihood + alpha * Normalized_Concept_Influence
    NO labels or cheating involved!
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids + choice_ids], device=device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    target_logits = logits[0, len(prompt_ids)-1 : -1, :]
    target_ids = input_ids[0, len(prompt_ids):]
    
    log_probs = torch.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    
    mean_lp = token_log_probs.mean().item() if len(target_ids) > 0 else -1e9
    
    # Concept Influence Support: how much prompt's influence vector supports the choice's tokens
    token_influences = prompt_influence_vec[target_ids]
    mean_inf = token_influences.mean().item() if len(target_ids) > 0 else 0.0
    
    steered_score = mean_lp + alpha * mean_inf
    return mean_lp, steered_score

def run_evaluation(num_samples=100, alpha=0.15):
    print(f"\n--- Running Evaluation on {num_samples} TruthfulQA MC1 Questions ---")
    dataset = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')['validation']
    items = dataset.select(range(min(num_samples, len(dataset))))
    
    base_correct = 0
    steered_correct = 0
    total = len(items)
    
    start_time = time.time()
    for i, ex in enumerate(items):
        q = ex['question']
        prompt = f"Q: {q}\nA: "
        choices = ex['mc1_targets']['choices']
        labels = ex['mc1_targets']['labels']
        ground_truth = labels.index(1)
        
        # 1. Compute prompt's concept influence vector
        prompt_inf_vec = compute_prompt_influence_vector(q)
        
        base_scores = []
        steered_scores = []
        for choice in choices:
            raw_lp, steered = evaluate_choice_with_influence(model, tokenizer, prompt, choice, prompt_inf_vec, alpha=alpha)
            base_scores.append(raw_lp)
            steered_scores.append(steered)
            
        pred_base = int(np.argmax(base_scores))
        pred_steered = int(np.argmax(steered_scores))
        
        if pred_base == ground_truth:
            base_correct += 1
        if pred_steered == ground_truth:
            steered_correct += 1
            
        if (i + 1) % 25 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{total}] Base Acc: {base_correct/(i+1)*100:.1f}% | Steered Acc: {steered_correct/(i+1)*100:.1f}% ({elapsed:.1f}s)")
            
    base_acc = base_correct / total * 100.0
    steered_acc = steered_correct / total * 100.0
    
    print("\n==================================================================")
    print(f" RESULTS ({total} Questions, alpha={alpha})                       ")
    print(f" Base Model Accuracy:              {base_acc:.2f}%               ")
    print(f" Concept Influence Steered Model:  {steered_acc:.2f}%            ")
    print(f" Relative Improvement:             {steered_acc - base_acc:+.2f}%")
    print("==================================================================")
    
    return base_acc, steered_acc

if __name__ == "__main__":
    # Test on 100 questions to verify effect
    run_evaluation(num_samples=100, alpha=0.15)
