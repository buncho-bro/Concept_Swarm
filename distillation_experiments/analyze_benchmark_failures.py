"""
Analyze Benchmark Failures in Physically Coupled Cognitive System
================================================================
Runs a targeted sample of problems across all 5 benchmark suites to capture
detailed traces of failure cases (question, ground truth, tree search output,
final response, and error mechanism).
"""

import os
import sys
import json
import time
import re
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

TARGET_FAILURES_PER_SUITE = 4
MAX_SEARCH_PER_SUITE = 15


def main():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    base_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    from universal_backward_epistemic_system import UniversalPossibilityTreeEngine
    engine = UniversalPossibilityTreeEngine(model, tokenizer, device)

    failures_report = {}

    # 1. BBH Causal Judgement
    print("\n--- Diagnosing BBH Causal Judgement Failures ---")
    ds_causal = load_dataset('lukaemon/bbh', 'causal_judgement', split='test')
    causal_failures = []
    for i in range(min(MAX_SEARCH_PER_SUITE, len(ds_causal))):
        c = ds_causal[i]
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
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        pred = "yes" if "yes" in resp and "no" not in resp else ("no" if "no" in resp else "")

        if pred != target:
            causal_failures.append({
                "index": i,
                "input": inp_text,
                "target": target,
                "pred": pred,
                "raw_response": resp,
                "tree_output": tree_res['raw_output']
            })
            print(f"  [Failure] Item #{i}: Target '{target}' vs Pred '{pred}'")
            if len(causal_failures) >= TARGET_FAILURES_PER_SUITE:
                break
    failures_report["causal_judgement"] = causal_failures

    # 2. BBH Disambiguation QA
    print("\n--- Diagnosing BBH Disambiguation QA Failures ---")
    ds_disambig = load_dataset('lukaemon/bbh', 'disambiguation_qa', split='test')
    disambig_failures = []
    for i in range(min(MAX_SEARCH_PER_SUITE, len(ds_disambig))):
        c = ds_disambig[i]
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
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.search(r"([a-c])", resp)
        pred = m.group(1) if m else ""

        if pred != target:
            disambig_failures.append({
                "index": i,
                "input": inp_text,
                "target": target,
                "pred": pred,
                "raw_response": resp,
                "tree_output": tree_res['raw_output']
            })
            print(f"  [Failure] Item #{i}: Target '{target}' vs Pred '{pred}'")
            if len(disambig_failures) >= TARGET_FAILURES_PER_SUITE:
                break
    failures_report["disambiguation_qa"] = disambig_failures

    # 3. BBH Formal Fallacies
    print("\n--- Diagnosing BBH Formal Fallacies Failures ---")
    ds_fallacies = load_dataset('lukaemon/bbh', 'formal_fallacies', split='test')
    fallacies_failures = []
    for i in range(min(MAX_SEARCH_PER_SUITE, len(ds_fallacies))):
        c = ds_fallacies[i]
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
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        pred = "valid" if "valid" in resp and "invalid" not in resp else ("invalid" if "invalid" in resp else "")

        if pred != target:
            fallacies_failures.append({
                "index": i,
                "input": inp_text,
                "target": target,
                "pred": pred,
                "raw_response": resp,
                "tree_output": tree_res['raw_output']
            })
            print(f"  [Failure] Item #{i}: Target '{target}' vs Pred '{pred}'")
            if len(fallacies_failures) >= TARGET_FAILURES_PER_SUITE:
                break
    failures_report["formal_fallacies"] = fallacies_failures

    # 4. TruthfulQA MC1
    print("\n--- Diagnosing TruthfulQA MC1 Failures ---")
    ds_truthful = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    truthful_failures = []
    for i in range(min(MAX_SEARCH_PER_SUITE, len(ds_truthful))):
        c = ds_truthful[i]
        q = c["question"]
        choices = c["mc1_targets"]["choices"]
        labels = c["mc1_targets"]["labels"]
        ground_idx = labels.index(1) if 1 in labels else 0

        tree_res = engine.explore_possibility_trees_and_invariants(q)
        formatted = "\n".join([f"({chr(65+k)}) {ch}" for k, ch in enumerate(choices)])

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
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().upper()
        m = re.search(r"([A-Z])", resp)
        pred_idx = ord(m.group(1)) - 65 if m and ord(m.group(1)) - 65 < len(choices) else 0

        if pred_idx != ground_idx:
            truthful_failures.append({
                "index": i,
                "question": q,
                "choices": choices,
                "target_idx": ground_idx,
                "target_choice": choices[ground_idx],
                "pred_idx": pred_idx,
                "pred_choice": choices[pred_idx] if pred_idx < len(choices) else "Out of bounds",
                "raw_response": resp,
                "tree_output": tree_res['raw_output']
            })
            print(f"  [Failure] Item #{i}: Target '({chr(65+ground_idx)})' vs Pred '({chr(65+pred_idx)})'")
            if len(truthful_failures) >= TARGET_FAILURES_PER_SUITE:
                break
    failures_report["truthfulqa"] = truthful_failures

    # 5. MMLU Moral Scenarios
    print("\n--- Diagnosing MMLU Moral Scenarios Failures ---")
    ds_moral = load_dataset('cais/mmlu', 'moral_scenarios', split='test')
    moral_failures = []
    for i in range(min(MAX_SEARCH_PER_SUITE, len(ds_moral))):
        c = ds_moral[i]
        q = c["question"]
        choices = c["choices"]
        ground_idx = c["answer"]

        tree_res = engine.explore_possibility_trees_and_invariants(q)
        formatted = "\n".join([f"({chr(65+k)}) {ch}" for k, ch in enumerate(choices)])

        prompt = (
            f"<|im_start|>system\n"
            f"You are a moral reasoning engine evaluating ethical dilemmas. Given the scenario and ethical possibilities, "
            f"answer strictly with the option letter of the most morally acceptable action.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Scenario: {q}\n\n"
            f"Moral Possibility Analysis:\n{tree_res['raw_output']}\n\n"
            f"Choices:\n{formatted}\n\n"
            f"Answer strictly with the option letter (e.g. (A)).<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().upper()
        m = re.search(r"([A-D])", resp)
        pred_idx = ord(m.group(1)) - 65 if m and ord(m.group(1)) - 65 < len(choices) else 0

        if pred_idx != ground_idx:
            moral_failures.append({
                "index": i,
                "scenario": q,
                "choices": choices,
                "target_idx": ground_idx,
                "target_choice": choices[ground_idx],
                "pred_idx": pred_idx,
                "pred_choice": choices[pred_idx] if pred_idx < len(choices) else "Out of bounds",
                "raw_response": resp,
                "tree_output": tree_res['raw_output']
            })
            print(f"  [Failure] Item #{i}: Target '({chr(65+ground_idx)})' vs Pred '({chr(65+pred_idx)})'")
            if len(moral_failures) >= TARGET_FAILURES_PER_SUITE:
                break
    failures_report["mmlu_moral"] = moral_failures

    out_file = os.path.join(OUTPUT_DIR, "detailed_benchmark_failures.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(failures_report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed failure logs saved to: {out_file}")


if __name__ == "__main__":
    main()
