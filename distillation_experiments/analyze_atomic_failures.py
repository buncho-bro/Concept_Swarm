"""
Analyze Atomic Decomposition Failures & Pure Knowledge Deficits
(原子命題分解システムの誤答分析と純粋知識不足の特定スクリプト)
"""

import os
import sys
import json
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
ANALYSIS_JSON = os.path.join(OUTPUT_DIR, "atomic_decomposition_failure_analysis.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

from atomic_proposition_decomposition_engine import (
    decompose_text_to_atomic_propositions,
    solve_causal_judgement_atomic,
    solve_disambiguation_atomic,
    solve_formal_fallacies_atomic,
    solve_truthfulqa_atomic,
    solve_mmlu_moral_atomic,
    load_web_standard_suites
)


def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    suites = load_web_standard_suites()

    failures = {
        "causal_judgement": [],
        "disambiguation_qa": [],
        "formal_fallacies": [],
        "truthfulqa_mc1": [],
        "mmlu_moral": []
    }

    # 1. Causal Judgement
    print("\nAuditing Causal Judgement failures...")
    for i, item in enumerate(suites['causal_judgement']):
        pred = solve_causal_judgement_atomic(model, tokenizer, item['input'])
        target = item['target'].strip()
        if pred.lower() != target.lower():
            props = decompose_text_to_atomic_propositions(item['input'])
            failures["causal_judgement"].append({
                "idx": i,
                "input": item['input'],
                "atomic_propositions": props,
                "predicted": pred,
                "ground_truth": target
            })

    # 2. Disambiguation QA
    print("Auditing Disambiguation QA failures...")
    for i, item in enumerate(suites['disambiguation_qa']):
        pred = solve_disambiguation_atomic(model, tokenizer, item['input'])
        target = item['target'].strip()
        is_corr = (target.lower() in pred.lower()) or (pred.lower() in target.lower())
        if not is_corr:
            props = decompose_text_to_atomic_propositions(item['input'])
            failures["disambiguation_qa"].append({
                "idx": i,
                "input": item['input'],
                "atomic_propositions": props,
                "predicted": pred,
                "ground_truth": target
            })

    # 3. Formal Fallacies
    print("Auditing Formal Fallacies failures...")
    for i, item in enumerate(suites['formal_fallacies']):
        pred = solve_formal_fallacies_atomic(model, tokenizer, item['input'])
        target = item['target'].strip().lower()
        if pred != target:
            props = decompose_text_to_atomic_propositions(item['input'])
            failures["formal_fallacies"].append({
                "idx": i,
                "input": item['input'],
                "atomic_propositions": props,
                "predicted": pred,
                "ground_truth": target
            })

    # 4. TruthfulQA MC1
    print("Auditing TruthfulQA MC1 failures...")
    for i, item in enumerate(suites['truthfulqa_mc1']):
        pred_idx = solve_truthfulqa_atomic(model, tokenizer, item['question'], item['choices'])
        target_idx = item['correct_idx']
        if pred_idx != target_idx:
            failures["truthfulqa_mc1"].append({
                "idx": i,
                "question": item['question'],
                "choices": item['choices'],
                "predicted_idx": pred_idx,
                "predicted_text": item['choices'][pred_idx],
                "ground_truth_idx": target_idx,
                "ground_truth_text": item['choices'][target_idx]
            })

    # 5. MMLU Moral Scenarios
    print("Auditing MMLU Moral Scenarios failures...")
    for i, item in enumerate(suites['mmlu_moral']):
        pred_idx = solve_mmlu_moral_atomic(model, tokenizer, item['question'], item['choices'])
        target_idx = item['answer']
        if pred_idx != target_idx:
            failures["mmlu_moral"].append({
                "idx": i,
                "question": item['question'],
                "choices": item['choices'],
                "predicted_idx": pred_idx,
                "predicted_text": item['choices'][pred_idx],
                "ground_truth_idx": target_idx,
                "ground_truth_text": item['choices'][target_idx]
            })

    # 結果保存
    with open(ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    print(f"\nAudit completed. Saved detailed failure log to: {ANALYSIS_JSON}")
    print(f"Summary of failures:")
    for k, v in failures.items():
        print(f"  - {k}: {len(v)} errors")


if __name__ == "__main__":
    main()
