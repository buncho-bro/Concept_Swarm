"""
Full-Suite Web Standard Benchmark: Pure Possibility Tree System (v6)
================================================================================
Evaluates the Possibility Tree & Invariant Principle System across ALL 5 Web Benchmark Suites
(N = 50 each, Total N = 250 problems), obtaining direct task accuracy alongside epistemic metrics:

1. BBH Causal Judgement (N = 50)
2. BBH Disambiguation QA (N = 50)
3. BBH Formal Fallacies (N = 50)
4. TruthfulQA MC1 (N = 50)
5. MMLU Moral Scenarios (N = 50)
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

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

from universal_backward_epistemic_system import UniversalPossibilityTreeEngine

N_PER_SUITE = 50

# Load 5 Web Suites
print("Loading 5 Web Standard Suites...")
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

print("Loading Qwen 1.5B Model...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
model.eval()

engine = UniversalPossibilityTreeEngine(model, tokenizer, device)


def evaluate_tree_guided_causal(cases):
    correct = 0
    for idx, c in enumerate(cases):
        inp_text = c['input']
        target = c['target'].strip().lower()
        tree_res = engine.explore_possibility_trees_and_invariants(inp_text)
        
        eval_prompt = (
            f"<|im_start|>system\n"
            f"You are a strict causal deduction engine. Given the problem and the explored possibility scenarios and necessary principle, "
            f"answer strictly with 'Yes' or 'No'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Possibility Scenarios & Necessary Invariant:\n{tree_res['raw_output']}\n\n"
            f"Question: Is the causal claim valid? Answer strictly with 'Yes' or 'No'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(eval_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        pred = "yes" if "yes" in resp and "no" not in resp else ("no" if "no" in resp else "")
        if pred == target:
            correct += 1
    return (correct / len(cases)) * 100.0


def evaluate_tree_guided_disambig(cases):
    correct = 0
    for idx, c in enumerate(cases):
        inp_text = c['input']
        target = c['target'].strip().strip("()").lower()
        tree_res = engine.explore_possibility_trees_and_invariants(inp_text)

        eval_prompt = (
            f"<|im_start|>system\n"
            f"You are a reference resolution engine. Given the problem and the possibility trees resolving ambiguity, "
            f"answer strictly with the correct option letter, e.g., (A), (B), or (C).<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Possibility Scenarios & Disambiguation Invariant:\n{tree_res['raw_output']}\n\n"
            f"Question: Which option is correct? Answer strictly with the option letter.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(eval_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.search(r"([a-c])", resp)
        pred = m.group(1) if m else ""
        if pred == target:
            correct += 1
    return (correct / len(cases)) * 100.0


def evaluate_tree_guided_fallacies(cases):
    correct = 0
    for idx, c in enumerate(cases):
        inp_text = c['input']
        target = c['target'].strip().lower()
        tree_res = engine.explore_possibility_trees_and_invariants(inp_text)

        eval_prompt = (
            f"<|im_start|>system\n"
            f"You are a formal logic engine. Given the syllogism and the counterfactual possibility scenarios, "
            f"answer strictly with 'valid' or 'invalid'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Counterfactual Scenarios & Deductive Invariant:\n{tree_res['raw_output']}\n\n"
            f"Question: Is the argument valid or invalid? Answer strictly with 'valid' or 'invalid'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(eval_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        pred = "valid" if "valid" in resp and "invalid" not in resp else ("invalid" if "invalid" in resp else "")
        if pred == target:
            correct += 1
    return (correct / len(cases)) * 100.0


def evaluate_tree_guided_truthful(cases):
    correct = 0
    for idx, c in enumerate(cases):
        q = c["question"]
        choices = c["mc1_targets"]["choices"]
        labels = c["mc1_targets"]["labels"]
        ground_idx = labels.index(1) if 1 in labels else 0

        tree_res = engine.explore_possibility_trees_and_invariants(q)
        formatted = "\n".join([f"({chr(65+i)}) {ch}" for i, ch in enumerate(choices)])

        eval_prompt = (
            f"<|im_start|>system\n"
            f"You are a scientific truthfulness engine. Given the question, the possibility scenarios, and the choices, "
            f"answer strictly with the option letter of the true factual statement.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Question: {q}\n\n"
            f"Possibility Scenarios & Invariant:\n{tree_res['raw_output']}\n\n"
            f"Choices:\n{formatted}\n\n"
            f"Answer strictly with the option letter (e.g. (A)).<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(eval_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().upper()
        m = re.search(r"([A-Z])", resp)
        pred_idx = ord(m.group(1)) - 65 if m and ord(m.group(1)) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct += 1
    return (correct / len(cases)) * 100.0


def evaluate_tree_guided_moral(cases):
    correct = 0
    for idx, c in enumerate(cases):
        q = c["question"]
        choices = c["choices"]
        ground_idx = c["answer"]

        tree_res = engine.explore_possibility_trees_and_invariants(q)
        formatted = "\n".join([f"({chr(65+i)}) {ch}" for i, ch in enumerate(choices)])

        eval_prompt = (
            f"<|im_start|>system\n"
            f"You are a moral reasoning engine. Given the ethical scenario, the moral trade-off trees, and the choices, "
            f"answer strictly with the option letter of the most moral/appropriate action.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Scenario: {q}\n\n"
            f"Moral Trade-off Trees & Invariant:\n{tree_res['raw_output']}\n\n"
            f"Choices:\n{formatted}\n\n"
            f"Answer strictly with the option letter (A, B, C, or D).<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(eval_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().upper()
        m = re.search(r"([A-D])", resp)
        pred_idx = ord(m.group(1)) - 65 if m and ord(m.group(1)) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct += 1
    return (correct / len(cases)) * 100.0


def main():
    print("=" * 85)
    print("RUNNING PURE POSSIBILITY TREE ENGINE ON ALL 5 WEB SUITES (N = 250)")
    print("=" * 85)

    t0 = time.time()

    print("[1/5] Evaluating BBH Causal Judgement (N = 50)...")
    score_causal = evaluate_tree_guided_causal(cases_causal)
    print(f"Causal Judgement Accuracy: {score_causal:.1f}%")

    print("[2/5] Evaluating BBH Disambiguation QA (N = 50)...")
    score_disambig = evaluate_tree_guided_disambig(cases_disambig)
    print(f"Disambiguation QA Accuracy: {score_disambig:.1f}%")

    print("[3/5] Evaluating BBH Formal Fallacies (N = 50)...")
    score_fallacies = evaluate_tree_guided_fallacies(cases_fallacies)
    print(f"Formal Fallacies Accuracy: {score_fallacies:.1f}%")

    print("[4/5] Evaluating TruthfulQA MC1 (N = 50)...")
    score_truthful = evaluate_tree_guided_truthful(cases_truthful)
    print(f"TruthfulQA MC1 Accuracy: {score_truthful:.1f}%")

    print("[5/5] Evaluating MMLU Moral Scenarios (N = 50)...")
    score_moral = evaluate_tree_guided_moral(cases_moral)
    print(f"MMLU Moral Scenarios Accuracy: {score_moral:.1f}%")

    avg_score = np.mean([score_causal, score_disambig, score_fallacies, score_truthful, score_moral])
    total_time = time.time() - t0
    print(f"\nAll 5 Suites Completed in {total_time:.1f}s. Average Accuracy: {avg_score:.1f}%")

    # Load existing benchmark results and integrate
    json_path = os.path.join(OUTPUT_DIR, "web_standard_large_scale_benchmark_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["Possibility_Tree_System"]["task_accuracies"] = {
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

    # Plot Complete 3-Model Comparison Chart
    tasks = [
        "BBH Causal\nJudgement",
        "BBH Disambig\nQA",
        "BBH Formal\nFallacies",
        "TruthfulQA\nMC1",
        "MMLU Moral\nScenarios"
    ]
    raw_scores = [data["Raw_Qwen1.5B"]["causal_judgement"], data["Raw_Qwen1.5B"]["disambiguation_qa"], data["Raw_Qwen1.5B"]["formal_fallacies"], data["Raw_Qwen1.5B"]["truthfulqa_mc1"], data["Raw_Qwen1.5B"]["mmlu_moral"]]
    r1_scores = [data["DeepSeek_R1_1.5B"]["causal_judgement"], data["DeepSeek_R1_1.5B"]["disambiguation_qa"], data["DeepSeek_R1_1.5B"]["formal_fallacies"], data["DeepSeek_R1_1.5B"]["truthfulqa_mc1"], data["DeepSeek_R1_1.5B"]["mmlu_moral"]]
    tree_scores = [score_causal, score_disambig, score_fallacies, score_truthful, score_moral]

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    rects1 = ax.bar(x - width, raw_scores, width, label='Raw Qwen2.5-1.5B (Direct)', color='#2b5c8f', alpha=0.9)
    rects2 = ax.bar(x, r1_scores, width, label='DeepSeek-R1-Distill-1.5B (CoT)', color='#d9534f', alpha=0.9)
    rects3 = ax.bar(x + width, tree_scores, width, label='Pure Possibility Tree System (Present)', color='#2e8b57', alpha=0.9)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Complete Web Standard Benchmark Across All 3 Systems (N = 250 Problems)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1 + rects2 + rects3:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACT_DIR, "web_standard_large_scale_benchmark.png")
    plt.savefig(chart_path)
    print(f"Generated complete benchmark visualization at: {chart_path}")


if __name__ == "__main__":
    main()
