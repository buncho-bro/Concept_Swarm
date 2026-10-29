import os
import json
import time
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set working paths
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/lora_Qwen2.5_7B_to_Qwen2.5_1.5B_Instruct_Topological"
R1_DISTILL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_choices_batch(model, tokenizer, prompt_text, choices, device="cuda"):
    """
    Evaluates log-likelihood for multiple choices given a prompt.
    Returns:
        best_sum_idx: index of choice with highest total log likelihood
        best_norm_idx: index of choice with highest length-normalized log likelihood
        choice_scores: list of (sum_log_prob, mean_log_prob)
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    choice_scores = []
    # Build batch of full sequences
    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + choice_ids], device=device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        target_logits = logits[0, len(prompt_ids)-1 : -1, :]
        target_ids = input_ids[0, len(prompt_ids):]
        
        log_probs = torch.log_softmax(target_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        sum_lp = token_log_probs.sum().item()
        mean_lp = token_log_probs.mean().item() if len(target_ids) > 0 else -1e9
        choice_scores.append((sum_lp, mean_lp))
        
    best_sum_idx = int(np.argmax([s[0] for s in choice_scores]))
    best_norm_idx = int(np.argmax([s[1] for s in choice_scores]))
    return best_sum_idx, best_norm_idx, choice_scores

def run_truthfulqa_mc1(model, tokenizer, dataset, max_samples=None):
    """
    Evaluates TruthfulQA MC1 (Single-correct choice out of 4-5).
    """
    items = dataset['validation']
    if max_samples:
        items = items.select(range(min(max_samples, len(items))))
        
    correct_raw = 0
    correct_norm = 0
    total = len(items)
    
    start_time = time.time()
    for i, ex in enumerate(items):
        q = ex['question']
        prompt = f"Q: {q}\nA: "
        choices = ex['mc1_targets']['choices']
        labels = ex['mc1_targets']['labels']
        ground_truth = labels.index(1) # index of correct choice
        
        best_sum, best_norm, _ = evaluate_choices_batch(model, tokenizer, prompt, choices, device)
        if best_sum == ground_truth:
            correct_raw += 1
        if best_norm == ground_truth:
            correct_norm += 1
            
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            print(f"  [TruthfulQA] {i+1}/{total} done ({elapsed:.1f}s) | Raw Acc: {correct_raw/(i+1)*100:.1f}%, Norm Acc: {correct_norm/(i+1)*100:.1f}%")
            
    acc_raw = correct_raw / total * 100.0
    acc_norm = correct_norm / total * 100.0
    se_norm = math.sqrt(acc_norm * (100 - acc_norm) / total)
    return {"acc_raw": acc_raw, "acc_norm": acc_norm, "se_norm": se_norm, "total": total}

def run_prontoqa(model, tokenizer, dataset, max_samples=None):
    """
    Evaluates ProntoQA (Multi-step deductive logic: True vs False).
    """
    items = dataset['validation']
    if max_samples:
        items = items.select(range(min(max_samples, len(items))))
        
    correct_raw = 0
    correct_norm = 0
    total = len(items)
    
    start_time = time.time()
    for i, ex in enumerate(items):
        context = ex['context']
        question = ex['question']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        # Choices: " True" vs " False"
        choices = [" True", " False"]
        ground_truth = 0 if ex['answer'].strip() == 'A' else 1
        
        best_sum, best_norm, _ = evaluate_choices_batch(model, tokenizer, prompt, choices, device)
        if best_sum == ground_truth:
            correct_raw += 1
        if best_norm == ground_truth:
            correct_norm += 1
            
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            print(f"  [ProntoQA] {i+1}/{total} done ({elapsed:.1f}s) | Raw Acc: {correct_raw/(i+1)*100:.1f}%, Norm Acc: {correct_norm/(i+1)*100:.1f}%")
            
    acc_raw = correct_raw / total * 100.0
    acc_norm = correct_norm / total * 100.0
    se_norm = math.sqrt(acc_norm * (100 - acc_norm) / total)
    return {"acc_raw": acc_raw, "acc_norm": acc_norm, "se_norm": se_norm, "total": total}

def main():
    print("=================================================================")
    print(" RIGOROUS STANDARD BENCHMARK EVALUATION (NO BIAS, NO SHORTCUTS)  ")
    print("=================================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    print("\nLoading datasets...")
    truthfulqa_ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')
    prontoqa_ds = load_dataset('renma/ProntoQA')
    print(f"TruthfulQA validation count: {len(truthfulqa_ds['validation'])}")
    print(f"ProntoQA validation count: {len(prontoqa_ds['validation'])}")
    
    results = {}
    
    # ------------------------------------------------------------------
    # 1. Base Model: Qwen2.5-1.5B-Instruct
    # ------------------------------------------------------------------
    print("\n>>> [1/3] Evaluating Base Model: Qwen2.5-1.5B-Instruct ...")
    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model_base.eval()
    
    tqa_base = run_truthfulqa_mc1(model_base, tokenizer_base, truthfulqa_ds)
    pqa_base = run_prontoqa(model_base, tokenizer_base, prontoqa_ds)
    results['Base 1.5B'] = {"TruthfulQA": tqa_base, "ProntoQA": pqa_base}
    
    # Free VRAM
    del model_base
    torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------
    # 2. KNN 7B to 1.5B LoRA
    # ------------------------------------------------------------------
    print("\n>>> [2/3] Evaluating KNN 7B to 1.5B Distilled LoRA ...")
    model_lora = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model_lora = PeftModel.from_pretrained(model_lora, LORA_PATH)
    model_lora.eval()
    
    tqa_lora = run_truthfulqa_mc1(model_lora, tokenizer_base, truthfulqa_ds)
    pqa_lora = run_prontoqa(model_lora, tokenizer_base, prontoqa_ds)
    results['KNN 7to1.5B LoRA'] = {"TruthfulQA": tqa_lora, "ProntoQA": pqa_lora}
    
    # Free VRAM
    del model_lora
    torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------
    # 3. DeepSeek-R1-Distill-Qwen-1.5B (Real Line 2 Model)
    # ------------------------------------------------------------------
    print("\n>>> [3/3] Evaluating DeepSeek-R1-Distill-Qwen-1.5B (Line 2 Model) ...")
    tokenizer_r1 = AutoTokenizer.from_pretrained(R1_DISTILL_ID)
    model_r1 = AutoModelForCausalLM.from_pretrained(
        R1_DISTILL_ID,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model_r1.eval()
    
    tqa_r1 = run_truthfulqa_mc1(model_r1, tokenizer_r1, truthfulqa_ds)
    pqa_r1 = run_prontoqa(model_r1, tokenizer_r1, prontoqa_ds)
    results['DeepSeek-R1-Distill-1.5B'] = {"TruthfulQA": tqa_r1, "ProntoQA": pqa_r1}
    
    del model_r1
    torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------
    # Save & Plot Results
    # ------------------------------------------------------------------
    print("\n=================================================================")
    print(" BENCHMARK RESULTS SUMMARY (lm-eval Standard: Normalized Acc)     ")
    print("=================================================================")
    for model_name, bench_dict in results.items():
        tqa = bench_dict['TruthfulQA']
        pqa = bench_dict['ProntoQA']
        print(f"Model: {model_name:25s} | TruthfulQA MC1: {tqa['acc_norm']:.2f}% (±{tqa['se_norm']:.2f}%) | ProntoQA: {pqa['acc_norm']:.2f}% (±{pqa['se_norm']:.2f}%)")
        
    results_path = os.path.join(OUTPUT_DIR, "rigorous_standard_benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Plotting comparison bar chart
    models = list(results.keys())
    tqa_scores = [results[m]['TruthfulQA']['acc_norm'] for m in models]
    tqa_errs = [results[m]['TruthfulQA']['se_norm'] for m in models]
    
    pqa_scores = [results[m]['ProntoQA']['acc_norm'] for m in models]
    pqa_errs = [results[m]['ProntoQA']['se_norm'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, tqa_scores, width, yerr=tqa_errs, capsize=5, label='TruthfulQA MC1 (Factuality, 817 Qs)', color='#3498db', alpha=0.85)
    rects2 = ax.bar(x + width/2, pqa_scores, width, yerr=pqa_errs, capsize=5, label='ProntoQA (Deductive Logic, 500 Qs)', color='#e67e22', alpha=0.85)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Rigorous Standard Benchmark Comparison (No Cheating, No Bias)\nBase 1.5B vs. KNN 7to1.5B LoRA vs. DeepSeek-R1-Distill-1.5B', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "rigorous_standard_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved comparison plot to {plot_path}")
    
    # Copy to artifact dir
    artifact_plot_path = os.path.join(ARTIFACT_DIR, "rigorous_standard_benchmark_comparison.png")
    import shutil
    shutil.copyfile(plot_path, artifact_plot_path)
    print(f"Copied to artifact dir: {artifact_plot_path}")

if __name__ == "__main__":
    main()
