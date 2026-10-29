"""
Proper Evaluation of Web-Grounded Multi-Epoch Trained LoRA Model
(モデルの構造化出力フォーマットに合わせた完全デコード・正規表現抽出による厳密評価)
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
RESULTS_JSON = os.path.join(OUTPUT_DIR, "web_standard_large_scale_benchmark_results.json")
TRAINED_LORA_PATH = os.path.join(OUTPUT_DIR, "lora_web_grounded_spider_web_multiepoch")
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

from train_web_grounded_spider_web_multiepoch import load_web_standard_suites


def evaluate_properly():
    print("=" * 85)
    print("STARTING PROPER EVALUATION FOR MULTI-EPOCH WEB-GROUNDED MODEL")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    base_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, TRAINED_LORA_PATH)
    model.eval()

    suites = load_web_standard_suites()
    results = {}
    total_correct = 0
    total_evaluated = 0
    t0 = time.time()

    for suite_name, items in suites.items():
        print(f"\n>>> Evaluating Suite: {suite_name} (N = {len(items)})")
        correct = 0

        for i, item in enumerate(items):
            if suite_name == 'causal_judgement':
                prompt = (
                    f"<|im_start|>system\nYou are a precise, web-grounded reasoning engine.<|im_end|>\n"
                    f"<|im_start|>user\n{item['input']}\n\n"
                    f"Answer strictly with 'Yes' or 'No'. Format: 'Answer: Yes' or 'Answer: No'.<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                target = item['target'].strip().lower()
            elif suite_name == 'disambiguation_qa':
                prompt = (
                    f"<|im_start|>system\nYou are a precise, web-grounded reasoning engine.<|im_end|>\n"
                    f"<|im_start|>user\n{item['input']}\n\n"
                    f"State the referent clearly or state 'Ambiguous'. Format: 'Answer: <referent>'.<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                target = item['target'].strip().lower()
            elif suite_name == 'formal_fallacies':
                prompt = (
                    f"<|im_start|>system\nYou are a precise, web-grounded reasoning engine.<|im_end|>\n"
                    f"<|im_start|>user\n{item['input']}\n\n"
                    f"Evaluate deductive validity. Answer strictly with 'valid' or 'invalid'. Format: 'Answer: valid' or 'Answer: invalid'.<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                target = item['target'].strip().lower()
            elif suite_name == 'truthfulqa_mc1':
                choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
                prompt = (
                    f"<|im_start|>system\nYou are an uncompromising scientific truth-checking engine.<|im_end|>\n"
                    f"<|im_start|>user\nQuestion: {item['question']}\n\nOptions:\n{choices_str}\n\n"
                    f"Select the single most factual option. Format: 'Answer: (X)'.<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                target = chr(65 + item['correct_idx']).lower()
            elif suite_name == 'mmlu_moral':
                choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
                prompt = (
                    f"<|im_start|>system\nYou are a moral evaluation engine strictly grounded in the Harm Principle.<|im_end|>\n"
                    f"<|im_start|>user\nScenario Question: {item['question']}\n\nOptions:\n{choices_str}\n\n"
                    f"Select the correct option. Format: 'Answer: (X)'.<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                target = chr(65 + item['answer']).lower()

            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=90, do_sample=False)
            resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # 正規表現による柔軟かつ厳密な解答抽出
            is_corr = False
            if suite_name == 'causal_judgement':
                m = re.findall(r'\b(yes|no)\b', resp, re.IGNORECASE)
                pred = m[-1].lower() if m else ""
                is_corr = (pred == target)
            elif suite_name == 'disambiguation_qa':
                is_corr = (target in resp.lower())
            elif suite_name == 'formal_fallacies':
                m = re.findall(r'\b(valid|invalid)\b', resp, re.IGNORECASE)
                pred = m[-1].lower() if m else ""
                is_corr = (pred == target)
            else:
                m = re.findall(r'\b([A-D])\b', resp)
                pred = m[-1].lower() if m else ""
                is_corr = (pred == target)

            if is_corr:
                correct += 1
                total_correct += 1
            total_evaluated += 1

            if (i + 1) % 10 == 0:
                print(f"  [{suite_name}] Item {i+1:02d}/50 | Current Acc: {correct/(i+1)*100:.1f}%")

        acc = (correct / len(items)) * 100.0
        results[suite_name] = acc
        print(f">>> Suite {suite_name} Accuracy: {acc:.1f}% ({correct}/{len(items)})")

    elapsed = time.time() - t0
    overall_avg = (total_correct / total_evaluated) * 100.0
    results["average"] = overall_avg
    results["elapsed_sec"] = elapsed

    print("\n" + "=" * 85)
    print(f"PROPER EVALUATION RESULT (Web-Grounded Multi-Epoch LoRA): {overall_avg:.2f}% ({total_correct}/{total_evaluated}) in {elapsed:.1f}s")
    print("=" * 85)

    all_results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    all_results["Web_Grounded_MultiEpoch_LoRA"] = results
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # チャートの更新
    generate_comparison_chart(all_results)


def generate_comparison_chart(all_results):
    suites = ['causal_judgement', 'disambiguation_qa', 'formal_fallacies', 'truthfulqa_mc1', 'mmlu_moral']
    suite_labels = ['Causal', 'Disambig', 'Fallacies', 'TruthfulQA', 'Moral']

    plt.figure(figsize=(14, 6))

    fair_models = [
        ("Raw_Qwen1.5B", "Raw Qwen 1.5B", "#7f8c8d"),
        ("External_Text_Trained_LoRA", "External Text (Alpaca)", "#2980b9"),
        ("Atomic_Proposition_Grounded_System", "Atomic Proposition", "#95a5a6"),
        ("Spider_Web_Belief_Network", "Spider-Web (Offline)", "#e67e22"),
        ("Web_Grounded_MultiEpoch_LoRA", "Web Multi-Epoch LoRA (Ours)", "#27ae60")
    ]

    x = np.arange(len(suite_labels))
    width = 0.17

    for idx, (m_key, m_name, color) in enumerate(fair_models):
        if m_key in all_results:
            vals = [all_results[m_key].get(s, 0.0) for s in suites]
            avg = all_results[m_key].get("average", 0.0)
            plt.bar(x + idx * width, vals, width, label=f"{m_name} (Avg: {avg:.1f}%)", color=color, alpha=0.9)

    plt.xlabel('Benchmark Suites (N=50 each, Total 250)', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Web Standard Benchmark: Proper Evaluation of Web Multi-Epoch LoRA vs Baselines', fontsize=13, fontweight='bold')
    plt.xticks(x + width * 2, suite_labels, fontsize=10)
    plt.ylim(0, 80)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACT_DIR, "web_standard_large_scale_benchmark.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    evaluate_properly()
