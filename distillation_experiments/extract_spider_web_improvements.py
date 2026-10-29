"""
Extract and Analyze Concrete Improvement Cases of Spider-Web Belief Network
(蜘蛛の巣型思考によって誤答から正解へと改善された具体例の抽出と閉路干渉分析)
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
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

from spider_web_belief_network_engine import (
    find_embedding_neighbors,
    load_web_standard_suites
)


def evaluate_raw_single(model, tokenizer, prompt_text):
    prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=15, do_sample=False)
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return resp


def evaluate_spider_web_single_detailed(model, tokenizer, question_text, options_text):
    words = [w.strip(".,?!:;\"'") for w in question_text.split() if len(w) > 4 and w.isalpha()]
    sample_anchors = words[:3] if len(words) >= 3 else ["cause", "action", "result"]
    
    with torch.no_grad():
        neighbors_route_a = find_embedding_neighbors(model, tokenizer, sample_anchors[:2], top_k=2)
        neighbors_route_b = find_embedding_neighbors(model, tokenizer, sample_anchors[-2:], top_k=2)

    str_a = ", ".join(neighbors_route_a) if neighbors_route_a else "context"
    str_b = ", ".join(neighbors_route_b) if neighbors_route_b else "physical"

    prompt_a = (
        f"<|im_start|>system\nYou are an uncompromising logic verification node examining path A ({str_a}).<|im_end|>\n"
        f"<|im_start|>user\nContext: {question_text}\nOptions:\n{options_text}\n"
        f"Analyze the facts strictly through direct physical/temporal evidence related to {str_a}. "
        f"State your reasoned conclusion formatted as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prompt_b = (
        f"<|im_start|>system\nYou are an uncompromising logic verification node examining path B ({str_b}).<|im_end|>\n"
        f"<|im_start|>user\nContext: {question_text}\nOptions:\n{options_text}\n"
        f"Analyze the facts strictly through rule-based/normative consistency related to {str_b}. "
        f"State your reasoned conclusion formatted as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    enc_a = tokenizer(prompt_a, return_tensors="pt").to(device)
    enc_b = tokenizer(prompt_b, return_tensors="pt").to(device)

    with torch.no_grad():
        out_a = model.generate(**enc_a, max_new_tokens=40, do_sample=False)
        out_b = model.generate(**enc_b, max_new_tokens=40, do_sample=False)

    resp_a = tokenizer.decode(out_a[0][enc_a["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    resp_b = tokenizer.decode(out_b[0][enc_b["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    dec_a_match = re.search(r"Decision:\s*\(?([A-Za-z0-9_]+)\)?", resp_a, re.IGNORECASE)
    dec_b_match = re.search(r"Decision:\s*\(?([A-Za-z0-9_]+)\)?", resp_b, re.IGNORECASE)
    dec_a = dec_a_match.group(1).strip() if dec_a_match else ""
    dec_b = dec_b_match.group(1).strip() if dec_b_match else ""

    reconcile_resp = ""
    final_choice = ""
    if dec_a and dec_b and (dec_a.lower() == dec_b.lower()):
        final_choice = dec_a
        is_consensus = True
    else:
        is_consensus = False
        reconcile_prompt = (
            f"<|im_start|>system\nYou are a master judge reconciling conflicting paths in a belief network.<|im_end|>\n"
            f"<|im_start|>user\nProblem: {question_text}\nOptions:\n{options_text}\n"
            f"Path A suggested: '{dec_a}' (reason: {resp_a[:80]})\n"
            f"Path B suggested: '{dec_b}' (reason: {resp_b[:80]})\n"
            f"One of these paths contains an ungrounded assumption or speculative leap. "
            f"Reject the speculation and choose the strictly grounded option. "
            f"Answer clearly as 'Final Answer: <choice>'.<|im_end|>\n"
            f"<|im_start|>assistant\nFinal Answer: "
        )
        enc_r = tokenizer(reconcile_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_r = model.generate(**enc_r, max_new_tokens=10, do_sample=False)
        reconcile_resp = tokenizer.decode(out_r[0][enc_r["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m = re.search(r"([A-Za-z0-9_]+)", reconcile_resp)
        final_choice = m.group(1).strip() if m else (dec_a if dec_a else dec_b)

    return {
        "neighbors_a": neighbors_route_a,
        "neighbors_b": neighbors_route_b,
        "resp_a": resp_a,
        "resp_b": resp_b,
        "dec_a": dec_a,
        "dec_b": dec_b,
        "is_consensus": is_consensus,
        "reconcile_resp": reconcile_resp,
        "final_choice": final_choice
    }


def main():
    print("Loading Base Model Qwen-1.5B (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    suites = load_web_standard_suites()

    improvement_cases = []

    # 1. TruthfulQA (特に大躍進した領域を探索)
    print("\nScanning TruthfulQA for improvements...")
    for i, item in enumerate(suites['truthfulqa_mc1']):
        choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
        target = chr(65 + item['correct_idx'])

        # Raw の単体推論
        raw_prompt = f"Question: {item['question']}\n\nOptions:\n{choices_str}\n\nSelect the single most factual option. Format: 'Answer: (X)'."
        raw_resp = evaluate_raw_single(model, tokenizer, raw_prompt)
        m_raw = re.search(r"([A-D])", raw_resp.upper())
        pred_raw = m_raw.group(1) if m_raw else "A"
        raw_correct = (pred_raw == target)

        if not raw_correct:
            # 蜘蛛の巣ネットワークで解いてみる
            res = evaluate_spider_web_single_detailed(model, tokenizer, item['question'], choices_str)
            m_spider = re.search(r"([A-D])", res["final_choice"].upper())
            pred_spider = m_spider.group(1) if m_spider else "A"
            spider_correct = (pred_spider == target)

            if spider_correct:
                improvement_cases.append({
                    "domain": "TruthfulQA",
                    "idx": i,
                    "question": item['question'],
                    "choices": item['choices'],
                    "ground_truth": target,
                    "ground_truth_text": item['choices'][item['correct_idx']],
                    "raw_prediction": pred_raw,
                    "raw_prediction_text": item['choices'][ord(pred_raw)-65] if ord(pred_raw)-65 < len(item['choices']) else "Unknown",
                    "spider_details": res
                })
                print(f"  [Found TruthfulQA Improvement] Item {i}: Raw={pred_raw} (Wrong) -> Spider={pred_spider} (Correct!)")
                if len(improvement_cases) >= 3:
                    break

    # 2. Formal Fallacies (形式論理での改善を探索)
    print("\nScanning Formal Fallacies for improvements...")
    for i, item in enumerate(suites['formal_fallacies']):
        target = item['target'].strip().lower()
        raw_prompt = f"{item['input']}\n\nAnswer strictly with 'valid' or 'invalid'."
        raw_resp = evaluate_raw_single(model, tokenizer, raw_prompt).lower()
        pred_raw = "valid" if "valid" in raw_resp and "invalid" not in raw_resp else "invalid"
        raw_correct = (pred_raw == target)

        if not raw_correct:
            res = evaluate_spider_web_single_detailed(model, tokenizer, item['input'], "- valid\n- invalid")
            pred_spider = "valid" if "valid" in res["final_choice"].lower() and "invalid" not in res["final_choice"].lower() else "invalid"
            spider_correct = (pred_spider == target)

            if spider_correct:
                improvement_cases.append({
                    "domain": "Formal Fallacies",
                    "idx": i,
                    "question": item['input'],
                    "ground_truth": target,
                    "raw_prediction": pred_raw,
                    "spider_details": res
                })
                print(f"  [Found Fallacies Improvement] Item {i}: Raw={pred_raw} (Wrong) -> Spider={pred_spider} (Correct!)")
                if len(improvement_cases) >= 5:
                    break

    # 結果の保存
    out_path = os.path.join(OUTPUT_DIR, "spider_web_concrete_improvements.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(improvement_cases, f, indent=2, ensure_ascii=False)
    print(f"\nSaved concrete improvement cases to: {out_path}")


if __name__ == "__main__":
    main()
