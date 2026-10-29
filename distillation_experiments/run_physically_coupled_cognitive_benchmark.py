"""
Physically Coupled Cognitive Benchmark (選択肢B: 蒸留重み・思考原野結合型ベンチマーク)
================================================================================
Couples the Pure Possibility Tree System with:
1. `lora_dialectical_coherence` (弁証法的自己矛盾解消・幾何整合LoRAアダプター)
2. `UnifiedCognitiveArchitecture` (公理サンドボックス・意味共鳴・反実仮想・メタ反省推論)
3. Direct Evaluation on all 5 Web Standard Benchmark Suites (Total N = 250):
   - BBH Causal Judgement (N = 50)
   - BBH Disambiguation QA (N = 50)
   - BBH Formal Fallacies (N = 50)
   - TruthfulQA MC1 (N = 50)
   - MMLU Moral Scenarios (N = 50)
"""

import os
import sys
import json
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = os.path.join(OUTPUT_DIR, "lora_dialectical_coherence")

N_PER_SUITE = 50


def load_datasets():
    print("Loading 5 Web Standard Benchmark Suites (N = 50 each)...")
    ds_causal = load_dataset('lukaemon/bbh', 'causal_judgement', split='test')
    cases_causal = [ds_causal[i] for i in range(min(N_PER_SUITE, len(ds_causal)))]

    ds_disambig = load_dataset('lukaemon/bbh', 'disambiguation_qa', split='test')
    cases_disambig = [ds_disambig[i] for i in range(min(N_PER_SUITE, len(ds_disambig)))]

    ds_fallacies = load_dataset('lukaemon/bbh', 'formal_fallacies', split='test')
    cases_fallacies = [ds_fallacies[i] for i in range(min(N_PER_SUITE, len(ds_fallacies)))]

    ds_truthful = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    cases_truthful = [ds_truthful[i] for i in range(min(N_PER_SUITE, len(ds_truthful)))]

    ds_moral = load_dataset('cais/mmlu', 'moral_scenarios', split='test')
    cases_moral = [ds_moral[i] for i in range(min(N_PER_SUITE, len(ds_moral)))]

    return cases_causal, cases_disambig, cases_fallacies, cases_truthful, cases_moral


def main():
    print("=" * 85)
    print("PHYSICALLY COUPLED COGNITIVE SYSTEM (LORA + TREE SEARCH) BENCHMARK")
    print("=" * 85)

    cases_causal, cases_disambig, cases_fallacies, cases_truthful, cases_moral = load_datasets()

    print(f"\n1. Loading Base Model: {QWEN_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    base_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")

    print(f"2. Coupling with Dialectical Coherence LoRA Adapter: {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    from universal_backward_epistemic_system import UniversalPossibilityTreeEngine
    engine = UniversalPossibilityTreeEngine(model, tokenizer, device)

    t0 = time.time()

    # 1. Causal Judgement (BBH)
    print("\n[1/5] Evaluating BBH Causal Judgement (N = 50)...")
    correct_causal = 0
    for idx, c in enumerate(cases_causal):
        inp_text = c['input']
        target = c['target'].strip().lower()
        tree_res = engine.explore_possibility_trees_and_invariants(inp_text)
        
        prompt = (
            f"<|im_start|>system\n"
            f"You are a strict causal deduction engine. Given the problem, the counterfactual scenarios, and the necessary causal principle, "
            f"answer strictly with 'Yes' or 'No'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Causal Scenarios & Principle:\n{tree_res['raw_output']}\n\n"
            f"Question: Is the causal claim true? Answer strictly with 'Yes' or 'No'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        pred = "yes" if "yes" in resp and "no" not in resp else ("no" if "no" in resp else "")
        if pred == target:
            correct_causal += 1
    score_causal = (correct_causal / len(cases_causal)) * 100.0
    print(f"Coupled Causal Judgement Accuracy: {score_causal:.1f}%")

    # 2. Disambiguation QA (BBH)
    print("\n[2/5] Evaluating BBH Disambiguation QA (N = 50)...")
    correct_disambig = 0
    for idx, c in enumerate(cases_disambig):
        inp_text = c['input']
        target = c['target'].strip().strip("()").lower()
        tree_res = engine.explore_possibility_trees_and_invariants(inp_text)

        prompt = (
            f"<|im_start|>system\n"
            f"You are a precise reference resolution engine. Given the ambiguous context and possibility branches, "
            f"answer strictly with the correct option letter, e.g., (A), (B), or (C).<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Disambiguation Branches:\n{tree_res['raw_output']}\n\n"
            f"Question: Which choice correctly resolves the reference? Answer strictly with the letter.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.search(r"([a-c])", resp)
        pred = m.group(1) if m else ""
        if pred == target:
            correct_disambig += 1
    score_disambig = (correct_disambig / len(cases_disambig)) * 100.0
    print(f"Coupled Disambiguation QA Accuracy: {score_disambig:.1f}%")

    # 3. Formal Fallacies (BBH)
    print("\n[3/5] Evaluating BBH Formal Fallacies (N = 50)...")
    correct_fallacies = 0
    for idx, c in enumerate(cases_fallacies):
        inp_text = c['input']
        target = c['target'].strip().lower()
        tree_res = engine.explore_possibility_trees_and_invariants(inp_text)

        prompt = (
            f"<|im_start|>system\n"
            f"You are a formal logic engine. Given the syllogistic argument and its counterfactual validation, "
            f"answer strictly with 'valid' or 'invalid'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Formal Invariant Analysis:\n{tree_res['raw_output']}\n\n"
            f"Question: Is the argument valid or invalid? Answer strictly with 'valid' or 'invalid'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        pred = "valid" if "valid" in resp and "invalid" not in resp else ("invalid" if "invalid" in resp else "")
        if pred == target:
            correct_fallacies += 1
    score_fallacies = (correct_fallacies / len(cases_fallacies)) * 100.0
    print(f"Coupled Formal Fallacies Accuracy: {score_fallacies:.1f}%")

    # 4. TruthfulQA MC1
    print("\n[4/5] Evaluating TruthfulQA MC1 (N = 50)...")
    correct_truthful = 0
    for idx, c in enumerate(cases_truthful):
        q = c["question"]
        choices = c["mc1_targets"]["choices"]
        labels = c["mc1_targets"]["labels"]
        ground_idx = labels.index(1) if 1 in labels else 0

        tree_res = engine.explore_possibility_trees_and_invariants(q)
        formatted = "\n".join([f"({chr(65+i)}) {ch}" for i, ch in enumerate(choices)])

        prompt = (
            f"<|im_start|>system\n"
            f"You are an epistemic truthfulness engine. Given the question and possibility analysis, "
            f"answer strictly with the option letter of the true factual statement.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Question: {q}\n\n"
            f"Epistemic Invariant Analysis:\n{tree_res['raw_output']}\n\n"
            f"Choices:\n{formatted}\n\n"
            f"Answer strictly with the option letter (e.g. (A)).<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().upper()
        m = re.search(r"([A-Z])", resp)
        pred_idx = ord(m.group(1)) - 65 if m and ord(m.group(1)) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct_truthful += 1
    score_truthful = (correct_truthful / len(cases_truthful)) * 100.0
    print(f"Coupled TruthfulQA MC1 Accuracy: {score_truthful:.1f}%")

    # 5. MMLU Moral Scenarios
    print("\n[5/5] Evaluating MMLU Moral Scenarios (N = 50)...")
    correct_moral = 0
    for idx, c in enumerate(cases_moral):
        q = c["question"]
        choices = c["choices"]
        ground_idx = c["answer"]

        tree_res = engine.explore_possibility_trees_and_invariants(q)
        formatted = "\n".join([f"({chr(65+i)}) {ch}" for i, ch in enumerate(choices)])

        prompt = (
            f"<|im_start|>system\n"
            f"You are an ethical evaluation engine. Given the moral scenario and trade-off trees, "
            f"answer strictly with the option letter of the morally appropriate action (A, B, C, or D).<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Scenario: {q}\n\n"
            f"Moral Trade-off Analysis:\n{tree_res['raw_output']}\n\n"
            f"Choices:\n{formatted}\n\n"
            f"Answer strictly with the option letter.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().upper()
        m = re.search(r"([A-D])", resp)
        pred_idx = ord(m.group(1)) - 65 if m and ord(m.group(1)) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct_moral += 1
    score_moral = (correct_moral / len(cases_moral)) * 100.0
    print(f"Coupled MMLU Moral Scenarios Accuracy: {score_moral:.1f}%")

    avg_score = np.mean([score_causal, score_disambig, score_fallacies, score_truthful, score_moral])
    total_time = time.time() - t0
    print(f"\nAll 5 Suites Completed in {total_time:.1f}s. Coupled Average Accuracy: {avg_score:.1f}%")

    # Load existing benchmark results and integrate
    json_path = os.path.join(OUTPUT_DIR, "web_standard_large_scale_benchmark_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["Physically_Coupled_LoRA_System"] = {
        "causal_judgement": score_causal,
        "disambiguation_qa": score_disambig,
        "formal_fallacies": score_fallacies,
        "truthfulqa_mc1": score_truthful,
        "mmlu_moral": score_moral,
        "average": avg_score,
        "elapsed_sec": total_time
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated benchmark results saved to: {json_path}")

    # Plot 4-Way Comparison Chart
    tasks = [
        "BBH Causal\nJudgement",
        "BBH Disambig\nQA",
        "BBH Formal\nFallacies",
        "TruthfulQA\nMC1",
        "MMLU Moral\nScenarios"
    ]
    raw_scores = [data["Raw_Qwen1.5B"]["causal_judgement"], data["Raw_Qwen1.5B"]["disambiguation_qa"], data["Raw_Qwen1.5B"]["formal_fallacies"], data["Raw_Qwen1.5B"]["truthfulqa_mc1"], data["Raw_Qwen1.5B"]["mmlu_moral"]]
    r1_scores = [data["DeepSeek_R1_1.5B"]["causal_judgement"], data["DeepSeek_R1_1.5B"]["disambiguation_qa"], data["DeepSeek_R1_1.5B"]["formal_fallacies"], data["DeepSeek_R1_1.5B"]["truthfulqa_mc1"], data["DeepSeek_R1_1.5B"]["mmlu_moral"]]
    tree_scores = [data["Possibility_Tree_System"]["task_accuracies"]["causal_judgement"], data["Possibility_Tree_System"]["task_accuracies"]["disambiguation_qa"], data["Possibility_Tree_System"]["task_accuracies"]["formal_fallacies"], data["Possibility_Tree_System"]["task_accuracies"]["truthfulqa_mc1"], data["Possibility_Tree_System"]["task_accuracies"]["mmlu_moral"]]
    coupled_scores = [score_causal, score_disambig, score_fallacies, score_truthful, score_moral]

    x = np.arange(len(tasks))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    rects1 = ax.bar(x - 1.5*width, raw_scores, width, label='1. Raw Qwen2.5-1.5B (Direct)', color='#2b5c8f', alpha=0.85)
    rects2 = ax.bar(x - 0.5*width, r1_scores, width, label='2. DeepSeek-R1-Distill-1.5B (CoT)', color='#d9534f', alpha=0.85)
    rects3 = ax.bar(x + 0.5*width, tree_scores, width, label='3. Pure Possibility Tree (Raw Base)', color='#808080', alpha=0.85)
    rects4 = ax.bar(x + 1.5*width, coupled_scores, width, label='4. Physically Coupled (LoRA + Tree)', color='#2e8b57', alpha=0.95)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Web Standard Benchmark: Physical Weight Coupling vs Base Models (N = 250)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1 + rects2 + rects3 + rects4:
        h = rect.get_height()
        ax.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACT_DIR, "web_standard_large_scale_benchmark.png")
    plt.savefig(chart_path)
    print(f"Generated complete 4-way benchmark visualization at: {chart_path}")


if __name__ == "__main__":
    main()
