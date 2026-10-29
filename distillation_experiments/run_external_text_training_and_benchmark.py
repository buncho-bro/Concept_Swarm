"""
External Dataset Training & Benchmark (外部実文章データセット学習とベンチマーク検証)
================================================================================
1. 外部実文章データセット (tatsu-lab/alpaca) から500件の多様な実文書・論述・対話をロード。
2. Qwen-1.5B に対して LoRA で本格的に実文章学習 (2エポック, 250ステップ)。
3. Web標準難関ベンチマーク全250問を完全測定し、自己蒸留モデル(49.6%)との直接比較を行う。
"""

import os
import sys
import json
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "web_standard_large_scale_benchmark_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EXTERNAL_LORA_PATH = os.path.join(OUTPUT_DIR, "lora_external_alpaca_trained")
NUM_SAMPLES = 500


def load_external_texts():
    print(f"Loading {NUM_SAMPLES} real-world texts from external dataset 'tatsu-lab/alpaca'...")
    ds = load_dataset('tatsu-lab/alpaca', split=f'train[:{NUM_SAMPLES}]')
    formatted_data = []
    for item in ds:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        if input_text:
            user_msg = f"{instruction}\n\nContext:\n{input_text}"
        else:
            user_msg = instruction

        formatted_data.append({
            "user": user_msg,
            "assistant": output_text
        })
    print(f"Successfully loaded and formatted {len(formatted_data)} real-world training examples.")
    return formatted_data


class ExternalTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.examples = []
        for item in data:
            text = (
                f"<|im_start|>system\nYou are a helpful and precise assistant.<|im_end|>\n"
                f"<|im_start|>user\n{item['user']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['assistant']}<|im_end|>"
            )
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_on_external_texts(data, tokenizer):
    print("\n--- Training Qwen-1.5B on External Real-World Texts via LoRA ---")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    base_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    train_model = get_peft_model(base_model, peft_config)
    train_model.print_trainable_parameters()

    dataset = ExternalTextDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4, weight_decay=0.01)

    epochs = 2
    train_model.train()
    t_train = time.time()
    step = 0
    total_loss = 0.0
    total_steps = len(dataloader) * epochs

    print(f"Total Epochs: {epochs}, Total Steps: {total_steps} (Full External Text Learning)")

    for ep in range(epochs):
        for batch in dataloader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = train_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0 or step == total_steps:
                print(f"  Epoch [{ep+1}/{epochs}] Step {step}/{total_steps} - Loss: {loss.item():.4f} (Avg: {total_loss/step:.4f})")

    print(f"External Text Training Completed in {time.time() - t_train:.1f}s.")
    print(f"Saving External Text Trained LoRA to: {EXTERNAL_LORA_PATH}...")
    train_model.save_pretrained(EXTERNAL_LORA_PATH)
    tokenizer.save_pretrained(EXTERNAL_LORA_PATH)
    print("External LoRA Successfully Saved!")
    return train_model


def evaluate_benchmark(model, tokenizer):
    print("\n--- Evaluating Model across All 5 Benchmark Suites (Total N = 250) ---")
    ds_causal = load_dataset('lukaemon/bbh', 'causal_judgement', split='test')
    cases_causal = [ds_causal[i] for i in range(min(50, len(ds_causal)))]

    ds_disambig = load_dataset('lukaemon/bbh', 'disambiguation_qa', split='test')
    cases_disambig = [ds_disambig[i] for i in range(min(50, len(ds_disambig)))]

    ds_fallacies = load_dataset('lukaemon/bbh', 'formal_fallacies', split='test')
    cases_fallacies = [ds_fallacies[i] for i in range(min(50, len(ds_fallacies)))]

    ds_truthful = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    cases_truthful = [ds_truthful[i] for i in range(min(50, len(ds_truthful)))]

    ds_moral = load_dataset('cais/mmlu', 'moral_scenarios', split='test')
    cases_moral = [ds_moral[i] for i in range(min(50, len(ds_moral)))]

    model.eval()
    scores = {}
    t0 = time.time()

    # 1. Causal
    correct_causal = 0
    for c in cases_causal:
        prompt = (
            f"<|im_start|>system\nYou are a causal deduction engine. Conclude with 'Answer: Yes' or 'Answer: No'.<|im_end|>\n"
            f"<|im_start|>user\n{c['input']}\n\nIs the causal claim true? Conclude with 'Answer: Yes' or 'Answer: No'.<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.findall(r"answer:\s*(yes|no)", resp)
        pred = m[-1] if m else ("yes" if "yes" in resp and "no" not in resp else ("no" if "no" in resp else ""))
        if pred == c['target'].strip().lower():
            correct_causal += 1
    scores["causal_judgement"] = (correct_causal / len(cases_causal)) * 100.0

    # 2. Disambig
    correct_disambig = 0
    for c in cases_disambig:
        prompt = (
            f"<|im_start|>system\nYou are a reference resolution engine. Resolve using common sense. Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>user\n{c['input']}\n\nWhich choice correctly resolves the reference? Conclude with 'Answer: (X)'.<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.findall(r"answer:\s*\(([a-c])\)", resp)
        pred = m[-1] if m else (re.search(r"\(([a-c])\)", resp).group(1) if re.search(r"\(([a-c])\)", resp) else "")
        if pred == c['target'].strip().strip("()").lower():
            correct_disambig += 1
    scores["disambiguation_qa"] = (correct_disambig / len(cases_disambig)) * 100.0

    # 3. Fallacies
    correct_fallacies = 0
    for c in cases_fallacies:
        prompt = (
            f"<|im_start|>system\nYou are a formal logic engine. Check for counterexamples. Conclude with 'Answer: valid' or 'Answer: invalid'.<|im_end|>\n"
            f"<|im_start|>user\n{c['input']}\n\nIs the argument deductively valid or invalid? Conclude with 'Answer: valid' or 'Answer: invalid'.<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().lower()
        m = re.findall(r"answer:\s*(valid|invalid)", resp)
        pred = m[-1] if m else ("valid" if "valid" in resp and "invalid" not in resp else ("invalid" if "invalid" in resp else ""))
        if pred == c['target'].strip().lower():
            correct_fallacies += 1
    scores["formal_fallacies"] = (correct_fallacies / len(cases_fallacies)) * 100.0

    # 4. TruthfulQA
    correct_truthful = 0
    for c in cases_truthful:
        choices = c["mc1_targets"]["choices"]
        ground_idx = c["mc1_targets"]["labels"].index(1) if 1 in c["mc1_targets"]["labels"] else 0
        formatted = "\n".join([f"({chr(65+k)}) {ch}" for k, ch in enumerate(choices)])
        prompt = (
            f"<|im_start|>system\nYou are a scientific truthfulness engine. Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {c['question']}\n\nChoices:\n{formatted}\n\nSelect the true factual statement. Conclude with 'Answer: (X)'.<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()
        m = re.findall(r"Answer:\s*\(([A-Z])\)", resp)
        pred_char = m[-1] if m else (re.search(r"\(([A-Z])\)", resp).group(1) if re.search(r"\(([A-Z])\)", resp) else "A")
        pred_idx = ord(pred_char) - 65 if ord(pred_char) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct_truthful += 1
    scores["truthfulqa_mc1"] = (correct_truthful / len(cases_truthful)) * 100.0

    # 5. Moral
    correct_moral = 0
    for c in cases_moral:
        choices = c["choices"]
        ground_idx = c["answer"]
        formatted = "\n".join([f"({chr(65+k)}) {ch}" for k, ch in enumerate(choices)])
        prompt = (
            f"<|im_start|>system\nYou are an ethical reasoning engine. Harmless everyday actions are NOT wrong. Conclude with 'Answer: (X)'.<|im_end|>\n"
            f"<|im_start|>user\n{c['question']}\n\nChoices:\n{formatted}\n\nWhich option characterizes the morality? Conclude with 'Answer: (X)'.<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()
        m = re.findall(r"Answer:\s*\(([A-D])\)", resp)
        pred_char = m[-1] if m else (re.search(r"\(([A-D])\)", resp).group(1) if re.search(r"\(([A-D])\)", resp) else "A")
        pred_idx = ord(pred_char) - 65 if ord(pred_char) - 65 < len(choices) else 0
        if pred_idx == ground_idx:
            correct_moral += 1
    scores["mmlu_moral"] = (correct_moral / len(cases_moral)) * 100.0

    elapsed = time.time() - t0
    avg_score = np.mean(list(scores.values()))
    scores["average"] = avg_score
    scores["elapsed_sec"] = elapsed

    print(f"\nBenchmark Completed in {elapsed:.1f}s.")
    print(f"Causal: {scores['causal_judgement']:.1f}% | Disambig: {scores['disambiguation_qa']:.1f}% | Fallacies: {scores['formal_fallacies']:.1f}% | TruthfulQA: {scores['truthfulqa_mc1']:.1f}% | Moral: {scores['mmlu_moral']:.1f}%")
    print(f"OVERALL AVERAGE: {avg_score:.2f}%")
    return scores


def main():
    print("=" * 85)
    print("STARTING EXTERNAL DATASET TRAINING & BENCHMARK PIPELINE")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 外部実文章データセットのロード (500件)
    data = load_external_texts()

    # 2. 外部実文章でのLoRA学習
    trained_model = train_on_external_texts(data, tokenizer)

    # 3. ベンチマーク完全測定
    scores = evaluate_benchmark(trained_model, tokenizer)

    # 4. 結果保存
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            all_res = json.load(f)
    else:
        all_res = {}

    all_res["External_Text_Trained_LoRA"] = scores
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {RESULTS_JSON}")

    # 5. チャート更新
    suites = ["causal_judgement", "disambiguation_qa", "formal_fallacies", "truthfulqa_mc1", "mmlu_moral", "average"]
    suite_labels = ["BBH Causal", "BBH Disambig", "BBH Fallacies", "TruthfulQA", "MMLU Moral", "OVERALL AVG"]

    raw_scores = [all_res["Raw_Qwen1.5B"].get(s, 0) for s in suites]
    tree_scores = [all_res["Possibility_Tree_System"]["task_accuracies"].get(s, 0) if s != "average" else all_res["Possibility_Tree_System"]["task_accuracies"]["average"] for s in suites]
    decanted_scores = [all_res["Decanted_Wilderness_LoRA"].get(s, 0) for s in suites]
    external_scores = [scores.get(s, 0) for s in suites]

    x = np.arange(len(suites))
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    ax.bar(x - 1.5*width, raw_scores, width, label='1. Raw Qwen-1.5B (39.6%)', color='#94a3b8')
    ax.bar(x - 0.5*width, tree_scores, width, label='2. Pure Tree (40.0%)', color='#3b82f6')
    ax.bar(x + 0.5*width, decanted_scores, width, label='3. Decanted Wilderness (49.6%)', color='#10b981')
    ax.bar(x + 1.5*width, external_scores, width, label='4. External Dataset Trained (Ours)', color='#f59e0b')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Web Standard Benchmark (N=250): External Real-World Dataset vs Self-Decantation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(suite_labels, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    ax.axhline(50.0, color='red', linestyle=':', linewidth=1.5, label='50% Breakthrough Line')

    avg_idx = len(suites) - 1
    ax.annotate(f"{external_scores[avg_idx]:.1f}%",
                xy=(avg_idx + 1.5*width, external_scores[avg_idx]),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', color='#b45309', fontsize=12)

    chart_path = os.path.join(ARTIFACT_DIR, "web_standard_large_scale_benchmark.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
