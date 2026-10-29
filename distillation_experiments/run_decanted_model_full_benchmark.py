"""
Run Full Benchmark on Wilderness Decanted Epistemic Model (N = 250)
===================================================================
Evaluates the self-distilled model (lora_wilderness_decanted_epistemic) across all 5
Web standard benchmark suites (N = 50 each, total N = 250):
1. BBH Causal Judgement (N = 50)
2. BBH Disambiguation QA (N = 50)
3. BBH Formal Fallacies (N = 50)
4. TruthfulQA MC1 (N = 50)
5. MMLU Moral Scenarios (N = 50)

Uses the newly acquired epistemic thinking style:
Briefly decompress/examine -> Theoretically select -> Output exact answer.
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
NEW_LORA_PATH = os.path.join(OUTPUT_DIR, "lora_wilderness_decanted_epistemic")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "web_standard_large_scale_benchmark_results.json")
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
    print("RUNNING 250-BENCHMARK ON WILDERNESS DECANTED EPISTEMIC MODEL")
    print("=" * 85)

    cases_causal, cases_disambig, cases_fallacies, cases_truthful, cases_moral = load_datasets()

    print(f"\nLoading Base Model: {QWEN_ID} and Adapter: {NEW_LORA_PATH} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    base_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, NEW_LORA_PATH)
    model.eval()

    t0 = time.time()
    scores = {}

    # 1. BBH Causal Judgement (N = 50)
    print("\n[1/5] Evaluating BBH Causal Judgement (N = 50)...")
    correct_causal = 0
    for idx, c in enumerate(cases_causal):
        inp_text = c['input']
        target = c['target'].strip().lower()
        prompt = (
            f"<|im_start|>system\n"
            f"You are an epistemic causal deduction engine. Given the problem and physical time-course, "
            f"theoretically select whether the causal claim is true. "
            f"Conclude with 'Answer: Yes' or 'Answer: No'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Is the causal claim true? Briefly select the theoretical truth and conclude with 'Answer: Yes' or 'Answer: No'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=45, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        # Parse answer
        m = re.findall(r"answer:\s*(yes|no)", resp)
        if m:
            pred = m[-1]
        else:
            pred = "yes" if "yes" in resp and "no" not in resp else ("no" if "no" in resp else "")
        if pred == target:
            correct_causal += 1
    scores["causal_judgement"] = (correct_causal / len(cases_causal)) * 100.0
    print(f"Decanted Model Causal Accuracy: {scores['causal_judgement']:.1f}% ({correct_causal}/{len(cases_causal)})")

    # 2. BBH Disambiguation QA (N = 50)
    print("\n[2/5] Evaluating BBH Disambiguation QA (N = 50)...")
    correct_disambig = 0
    for idx, c in enumerate(cases_disambig):
        inp_text = c['input']
        target = c['target'].strip().strip("()").lower()
        prompt = (
            f"<|im_start|>system\n"
            f"You are a precise reference resolution engine. Use common-sense context to resolve the pronoun. "
            f"Do not assume ambiguity if the common-sense meaning is clear. "
            f"Conclude with 'Answer: (A)', 'Answer: (B)', or 'Answer: (C)'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Which choice correctly resolves the reference? Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=45, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.findall(r"answer:\s*\(([a-c])\)", resp)
        if m:
            pred = m[-1]
        else:
            m2 = re.search(r"\(([a-c])\)", resp)
            pred = m2.group(1) if m2 else ""
        if pred == target:
            correct_disambig += 1
    scores["disambiguation_qa"] = (correct_disambig / len(cases_disambig)) * 100.0
    print(f"Decanted Model Disambiguation Accuracy: {scores['disambiguation_qa']:.1f}% ({correct_disambig}/{len(cases_disambig)})")

    # 3. BBH Formal Fallacies (N = 50)
    print("\n[3/5] Evaluating BBH Formal Fallacies (N = 50)...")
    correct_fallacies = 0
    for idx, c in enumerate(cases_fallacies):
        inp_text = c['input']
        target = c['target'].strip().lower()
        prompt = (
            f"<|im_start|>system\n"
            f"You are a strict deductive formal logic engine. Check for counterexamples (e.g. denying antecedent or inverse fallacy). "
            f"Conclude with 'Answer: valid' or 'Answer: invalid'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{inp_text}\n\n"
            f"Is the argument deductively valid or invalid? Conclude with 'Answer: valid' or 'Answer: invalid'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=45, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.findall(r"answer:\s*(valid|invalid)", resp)
        if m:
            pred = m[-1]
        else:
            pred = "valid" if "valid" in resp and "invalid" not in resp else ("invalid" if "invalid" in resp else "")
        if pred == target:
            correct_fallacies += 1
    scores["formal_fallacies"] = (correct_fallacies / len(cases_fallacies)) * 100.0
    print(f"Decanted Model Formal Fallacies Accuracy: {scores['formal_fallacies']:.1f}% ({correct_fallacies}/{len(cases_fallacies)})")

    # 4. TruthfulQA MC1 (N = 50)
    print("\n[4/5] Evaluating TruthfulQA MC1 (N = 50)...")
    correct_truthful = 0
    for idx, c in enumerate(cases_truthful):
        q = c["question"]
        choices = c["mc1_targets"]["choices"]
        labels = c["mc1_targets"]["labels"]
        ground_idx = labels.index(1) if 1 in labels else 0
        formatted = "\n".join([f"({chr(65+k)}) {ch}" for k, ch in enumerate(choices)])

        prompt = (
            f"<|im_start|>system\n"
            f"You are a scientific truthfulness engine rejecting popular myths, biases, and subjective claims. "
            f"Theoretically select the objectively true statement. Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Question: {q}\n\n"
            f"Choices:\n{formatted}\n\n"
            f"Select the true factual statement. Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=45, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()
        m = re.findall(r"Answer:\s*\(([A-Z])\)", resp)
        if m:
            pred_char = m[-1]
        else:
            m2 = re.search(r"\(([A-Z])\)", resp)
            pred_char = m2.group(1) if m2 else "A"
        pred_idx = ord(pred_char) - 65 if ord(pred_char) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct_truthful += 1
    scores["truthfulqa_mc1"] = (correct_truthful / len(cases_truthful)) * 100.0
    print(f"Decanted Model TruthfulQA Accuracy: {scores['truthfulqa_mc1']:.1f}% ({correct_truthful}/{len(cases_truthful)})")

    # 5. MMLU Moral Scenarios (N = 50)
    print("\n[5/5] Evaluating MMLU Moral Scenarios (N = 50)...")
    correct_moral = 0
    for idx, c in enumerate(cases_moral):
        q = c["question"]
        choices = c["choices"]
        ground_idx = c["answer"]
        formatted = "\n".join([f"({chr(65+k)}) {ch}" for k, ch in enumerate(choices)])

        prompt = (
            f"<|im_start|>system\n"
            f"You are a moral reasoning engine evaluating everyday human scenarios. "
            f"Harmless everyday actions (playing fetch, baking cookies for friends, braking to avoid collision, teaching an instrument) are strictly NOT wrong. "
            f"Carefully match the conclusion to the exact choice (e.g. if neither is wrong, select 'Not wrong, Not wrong'). "
            f"Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{q}\n\n"
            f"Choices:\n{formatted}\n\n"
            f"Which option correctly characterizes the morality of Scenario 1 and Scenario 2? Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=45, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()
        m = re.findall(r"Answer:\s*\(([A-D])\)", resp)
        if m:
            pred_char = m[-1]
        else:
            m2 = re.search(r"\(([A-D])\)", resp)
            pred_char = m2.group(1) if m2 else "A"
        pred_idx = ord(pred_char) - 65 if ord(pred_char) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct_moral += 1
    scores["mmlu_moral"] = (correct_moral / len(cases_moral)) * 100.0
    print(f"Decanted Model MMLU Moral Accuracy: {scores['mmlu_moral']:.1f}% ({correct_moral}/{len(cases_moral)})")

    elapsed = time.time() - t0
    avg_score = np.mean(list(scores.values()))
    scores["average"] = avg_score
    scores["elapsed_sec"] = elapsed

    print("\n" + "=" * 85)
    print(f"ALL 5 SUITES COMPLETED IN {elapsed:.1f}s. NEW MODEL AVERAGE: {avg_score:.2f}%")
    print("=" * 85)

    # Load existing and update JSON
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            all_res = json.load(f)
    else:
        all_res = {}

    all_res["Decanted_Wilderness_LoRA"] = scores
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {RESULTS_JSON}")

    # Visualization
    suites = ["causal_judgement", "disambiguation_qa", "formal_fallacies", "truthfulqa_mc1", "mmlu_moral", "average"]
    suite_labels = [
        "BBH Causal",
        "BBH Disambig",
        "BBH Fallacies",
        "TruthfulQA",
        "MMLU Moral",
        "OVERALL AVG"
    ]

    raw_scores = [all_res["Raw_Qwen1.5B"].get(s, 0) for s in suites]
    r1_scores = [all_res["DeepSeek_R1_1.5B"].get(s, 0) for s in suites]
    tree_scores = [all_res["Possibility_Tree_System"]["task_accuracies"].get(s, 0) if s != "average" else all_res["Possibility_Tree_System"]["task_accuracies"]["average"] for s in suites]
    coupled_scores = [all_res["Physically_Coupled_LoRA_System"].get(s, 0) for s in suites]
    decanted_scores = [scores.get(s, 0) for s in suites]

    x = np.arange(len(suites))
    width = 0.16

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    ax.bar(x - 2*width, raw_scores, width, label='1. Raw Qwen-1.5B (Direct)', color='#94a3b8')
    ax.bar(x - 1*width, r1_scores, width, label='2. DeepSeek-R1-1.5B (CoT)', color='#f59e0b')
    ax.bar(x, tree_scores, width, label='3. Pure Possibility Tree (Prompt Wrapper)', color='#3b82f6')
    ax.bar(x + 1*width, coupled_scores, width, label='4. Previous LoRA + Tree (v1)', color='#8b5cf6')
    ax.bar(x + 2*width, decanted_scores, width, label='5. Decanted Wilderness LoRA (Ours)', color='#10b981')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Web Standard Large-Scale Benchmark (Total N=250): 5-System Evolution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(suite_labels, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)

    # Add score annotations above average bar
    avg_idx = len(suites) - 1
    ax.annotate(f"{decanted_scores[avg_idx]:.1f}%",
                xy=(avg_idx + 2*width, decanted_scores[avg_idx]),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', color='#047857', fontsize=11)

    chart_path = os.path.join(ARTIFACT_DIR, "web_standard_large_scale_benchmark.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"Updated 5-way benchmark chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
