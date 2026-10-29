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
print(" FULL-SCALE BENCHMARK EVALUATION WITH CONTRADICTION ENERGY        ")
print(" TruthfulQA MC1 (All 817 Qs) & ProntoQA (500 Qs)                 ")
print(" NO LABELS, NO CHEATING, STRICT ZERO-SHOT BLIND HARMONY          ")
print("==================================================================")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

print("Loading Qwen2.5-1.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# Neutral Baseline log-prob for concept influence
neutral_prompt = "Something is"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    neutral_logits = model(**inputs).logits[0, -1, :]
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1) # [vocab_size]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
    "that", "this", "it", "for", "with", "as", "by", "on", "at", "from", "which",
    "be", "has", "have", "had", "do", "does", "did", "but", "not", "your", "my",
    "all", "each", "every", "statement", "following", "true", "false", "question", "answer"
}

def extract_core_concept_tokens(text: str, max_tokens=6):
    """Extracts top content words from text and converts to token IDs."""
    words = [w.strip(".,?!:;\"'()[]{}").lower() for w in text.split()]
    concepts = [w for w in words if len(w) > 2 and w.isalpha() and w not in STOPWORDS]
    concepts = list(dict.fromkeys(concepts))[:max_tokens]
    
    token_ids = []
    for c in concepts:
        toks = tokenizer.encode(" " + c, add_special_tokens=False)
        if toks:
            token_ids.append((c, toks[0]))
    return token_ids

def compute_sentence_contradiction_energy_fast(concept_tokens):
    """
    Computes semantic friction energy across concept tokens using vectorized forward passes.
    Returns: contradiction_energy (higher = more internal friction/myth/disharmony)
    """
    n = len(concept_tokens)
    if n < 2:
        return 0.0
    
    # Batch forward pass for each concept
    prompts = [f"The {c} is" for c, _ in concept_tokens]
    batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        logits = model(**batch_inputs).logits[:, -1, :]
        lps = F.log_softmax(logits, dim=-1)
        deltas = lps - neutral_log_probs.unsqueeze(0) # [n, vocab_size]
        
    pairwise_compat = []
    for i in range(n):
        for j in range(i + 1, n):
            tid_i = concept_tokens[i][1]
            tid_j = concept_tokens[j][1]
            # influence i -> j and j -> i
            comp = (deltas[i, tid_j].item() + deltas[j, tid_i].item()) / 2.0
            pairwise_compat.append(comp)
            
    if not pairwise_compat:
        return 0.0
    
    # Contradiction energy is negative of average compatibility
    return -float(np.mean(pairwise_compat))

def evaluate_choice_with_energy(prompt_text, choice_text, beta=0.20):
    """
    Evaluates choice using Base Normalized Log-Likelihood and Contradiction Energy penalty.
    Score = Base_Normalized_LogLikelihood - beta * Contradiction_Energy
    NO cheating or ground truth labels used!
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
    
    # Measure contradiction energy of the choice combined with key context
    full_text = prompt_text + " " + choice_text
    concepts = extract_core_concept_tokens(full_text)
    contradiction_energy = compute_sentence_contradiction_energy_fast(concepts)
    
    # Energy-penalized score: lower contradiction energy yields higher score
    energy_score = mean_lp - beta * contradiction_energy
    
    return mean_lp, energy_score, contradiction_energy

# ----------------------------------------------------------------------
# 1. Run TruthfulQA MC1 (All 817 Questions)
# ----------------------------------------------------------------------
def run_truthfulqa_full(beta=0.20):
    print("\n==================================================================")
    print(" [1/2] RUNNING TRUTHFULQA MC1 (ALL 817 QUESTIONS)                 ")
    print("==================================================================")
    dataset = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')['validation']
    total = len(dataset)
    
    base_correct = 0
    energy_correct = 0
    
    start_time = time.time()
    for i, ex in enumerate(dataset):
        q = ex['question']
        prompt = f"Q: {q}\nA: "
        choices = ex['mc1_targets']['choices']
        labels = ex['mc1_targets']['labels']
        ground_truth = labels.index(1)
        
        base_scores = []
        energy_scores = []
        for choice in choices:
            raw_lp, en_score, _ = evaluate_choice_with_energy(prompt, choice, beta=beta)
            base_scores.append(raw_lp)
            energy_scores.append(en_score)
            
        pred_base = int(np.argmax(base_scores))
        pred_energy = int(np.argmax(energy_scores))
        
        if pred_base == ground_truth:
            base_correct += 1
        if pred_energy == ground_truth:
            energy_correct += 1
            
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            print(f"  [TruthfulQA] {i+1}/{total} ({elapsed:.1f}s) | Base Acc: {base_correct/(i+1)*100:.2f}% | Energy-Guided: {energy_correct/(i+1)*100:.2f}%")
            
    base_acc = base_correct / total * 100.0
    energy_acc = energy_correct / total * 100.0
    base_se = math.sqrt(base_acc * (100 - base_acc) / total)
    energy_se = math.sqrt(energy_acc * (100 - energy_acc) / total)
    
    return {
        "base_acc": base_acc, "base_se": base_se,
        "energy_acc": energy_acc, "energy_se": energy_se,
        "total": total
    }

# ----------------------------------------------------------------------
# 2. Run ProntoQA (All 500 Questions)
# ----------------------------------------------------------------------
def run_prontoqa_full(beta=0.20):
    print("\n==================================================================")
    print(" [2/2] RUNNING PRONTOQA (ALL 500 QUESTIONS)                       ")
    print("==================================================================")
    dataset = load_dataset('renma/ProntoQA')['validation']
    total = len(dataset)
    
    base_correct = 0
    energy_correct = 0
    
    start_time = time.time()
    for i, ex in enumerate(dataset):
        context = ex['context']
        question = ex['question']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        choices = [" True", " False"]
        ground_truth = 0 if ex['answer'].strip() == 'A' else 1
        
        base_scores = []
        energy_scores = []
        for choice in choices:
            raw_lp, en_score, _ = evaluate_choice_with_energy(prompt, choice, beta=beta)
            base_scores.append(raw_lp)
            energy_scores.append(en_score)
            
        pred_base = int(np.argmax(base_scores))
        pred_energy = int(np.argmax(energy_scores))
        
        if pred_base == ground_truth:
            base_correct += 1
        if pred_energy == ground_truth:
            energy_correct += 1
            
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            print(f"  [ProntoQA] {i+1}/{total} ({elapsed:.1f}s) | Base Acc: {base_correct/(i+1)*100:.2f}% | Energy-Guided: {energy_correct/(i+1)*100:.2f}%")
            
    base_acc = base_correct / total * 100.0
    energy_acc = energy_correct / total * 100.0
    base_se = math.sqrt(base_acc * (100 - base_acc) / total)
    energy_se = math.sqrt(energy_acc * (100 - energy_acc) / total)
    
    return {
        "base_acc": base_acc, "base_se": base_se,
        "energy_acc": energy_acc, "energy_se": energy_se,
        "total": total
    }

def main():
    tqa_res = run_truthfulqa_full(beta=0.20)
    pqa_res = run_prontoqa_full(beta=0.20)
    
    final_results = {
        "TruthfulQA": tqa_res,
        "ProntoQA": pqa_res
    }
    
    print("\n==================================================================")
    print(" FINAL RESULTS SUMMARY (FULL-SCALE RIGOROUS BENCHMARK)            ")
    print("==================================================================")
    print(f"TruthfulQA MC1 (817 Qs):")
    print(f"  Base 1.5B:          {tqa_res['base_acc']:.2f}% (±{tqa_res['base_se']:.2f}%)")
    print(f"  Energy-Guided (AI): {tqa_res['energy_acc']:.2f}% (±{tqa_res['energy_se']:.2f}%)  [Gap: {tqa_res['energy_acc'] - tqa_res['base_acc']:+.2f}%]")
    print(f"\nProntoQA (500 Qs):")
    print(f"  Base 1.5B:          {pqa_res['base_acc']:.2f}% (±{pqa_res['base_se']:.2f}%)")
    print(f"  Energy-Guided (AI): {pqa_res['energy_acc']:.2f}% (±{pqa_res['energy_se']:.2f}%)  [Gap: {pqa_res['energy_acc'] - pqa_res['base_acc']:+.2f}%]")
    
    # Save JSON
    out_json = os.path.join(OUTPUT_DIR, "full_scale_energy_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved full results to {out_json}")
    
    # Plotting comparison bar chart
    categories = ['TruthfulQA MC1 (817 Qs)', 'ProntoQA Logic (500 Qs)']
    base_scores = [tqa_res['base_acc'], pqa_res['base_acc']]
    base_errs = [tqa_res['base_se'], pqa_res['base_se']]
    energy_scores = [tqa_res['energy_acc'], pqa_res['energy_acc']]
    energy_errs = [tqa_res['energy_se'], pqa_res['energy_se']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    r1 = ax.bar(x - width/2, base_scores, width, yerr=base_errs, capsize=5, label='Base 1.5B (Standard Likelihood)', color='#7f8c8d', alpha=0.85)
    r2 = ax.bar(x + width/2, energy_scores, width, yerr=energy_errs, capsize=5, label='Energy-Guided (Contradiction Elimination)', color='#2980b9', alpha=0.85)
    
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Full-Scale Standard Benchmark: Base vs Energy-Guided Self-Correction\n(All 817 TruthfulQA + 500 ProntoQA Questions | Strictly Blind)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    
    for r in r1:
        h = r.get_height()
        ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for r in r2:
        h = r.get_height()
        ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#1f618d')
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "full_scale_energy_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    
    artifact_plot_path = os.path.join(ARTIFACT_DIR, "full_scale_energy_benchmark_comparison.png")
    import shutil
    shutil.copyfile(plot_path, artifact_plot_path)
    print(f"Copied to artifact dir: {artifact_plot_path}")

if __name__ == "__main__":
    main()
