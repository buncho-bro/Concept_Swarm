"""
Benchmark Evaluation for Contextual Resonance & Coherence Engine (CR-CE)
========================================================================
エントロピー降下最適化重み (lora_entropy_descent_calibrated) と
文脈共鳴・結合明白性推論 (CR-CE) を統合した新知能による
Web標準難関ベンチマーク全250問 (5スイート×各50問) の完全検証。
"""

import os
import sys
import json
import time
import re
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')
from spider_web_belief_network_engine import load_web_standard_suites

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "cr_ce_benchmark_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[{device.upper()}] モデルおよびエントロピー降下重みを初期化しています...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(
    base_model,
    os.path.join(OUTPUT_DIR, "lora_entropy_descent_calibrated"),
    adapter_name="calibrated"
)
peft_model.eval()


def get_contextual_choice_scores(prompt_text, choices, resonance_beta=0.3):
    """
    CR-CEの文脈相互情報量原理に基づく選択肢スコアリング:
    Score(c) = log P(c | 全体文脈) - beta * log P(c | 文脈なし・先頭数トークン)
    """
    ctx_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    null_prompt = "Context: Someone did something.\nAnswer: "
    null_ids = tokenizer.encode(null_prompt, add_special_tokens=False)

    scores = []
    with torch.no_grad():
        for c in choices:
            tgt_ids = tokenizer.encode(c, add_special_tokens=False)
            if not tgt_ids:
                scores.append(-999.0)
                continue

            # 1. 全体文脈での対数尤度
            full_ids = ctx_ids + tgt_ids
            input_tensor = torch.tensor([full_ids], device=device)
            out = peft_model(input_tensor)
            logits = out.logits[0, len(ctx_ids)-1:-1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            full_lp = sum(log_probs[i, tid].item() for i, tid in enumerate(tgt_ids)) / len(tgt_ids)

            # 2. 文脈なし(null)での局所反射確率
            null_full_ids = null_ids + tgt_ids
            input_tensor_null = torch.tensor([null_full_ids], device=device)
            out_null = peft_model(input_tensor_null)
            logits_null = out_null.logits[0, len(null_ids)-1:-1, :].float()
            log_probs_null = F.log_softmax(logits_null, dim=-1)
            null_lp = sum(log_probs_null[i, tid].item() for i, tid in enumerate(tgt_ids)) / len(tgt_ids)

            # 文脈共鳴 = 文脈からの必然性 - 単なる単語自体の出やすさ(バイアス)
            res_score = full_lp - resonance_beta * null_lp
            scores.append(res_score)

    return np.array(scores)


def evaluate_causal_suite(items):
    preds = []
    for it in items:
        inp = it["input"]
        prompt = (
            f"<|im_start|>system\nYou are a precise causal and counterfactual judge.<|im_end|>\n"
            f"<|im_start|>user\n{inp}\nOptions: Yes or No<|im_end|>\n"
            f"<|im_start|>assistant\nAnswer: "
        )
        scores = get_contextual_choice_scores(prompt, ["Yes", "No"], resonance_beta=0.25)
        preds.append("Yes" if scores[0] > scores[1] else "No")
    return preds


def evaluate_disambiguation_suite(items):
    preds = []
    for it in items:
        problem_text = it["input"]
        m_opts = re.findall(r"\(([A-C])\)\s*([^()\n]+)", problem_text)
        options = {letter: text.strip() for letter, text in m_opts}
        opt_A = options.get("A", "")
        opt_B = options.get("B", "")

        if not opt_A or not opt_B:
            preds.append("C")
            continue

        body_match = re.split(r"\nOptions:|\n\([A-Z]\)", problem_text)
        body = body_match[0].strip() if body_match else problem_text
        prompt = (
            f"<|im_start|>system\nYou are an expert in resolving pronoun and syntactic ambiguity.<|im_end|>\n"
            f"<|im_start|>user\nContext: {body}\nInterpretation:<|im_end|>\n"
            f"<|im_start|>assistant\nAnswer: "
        )
        scores = get_contextual_choice_scores(prompt, [opt_A, opt_B], resonance_beta=0.3)
        diff = scores[0] - scores[1]

        if abs(diff) < 0.12:
            preds.append("C")
        elif diff > 0.0:
            preds.append("A")
        else:
            preds.append("B")
    return preds


def evaluate_formal_fallacies_suite(items):
    preds = []
    for it in items:
        inp = it["input"]
        prompt = (
            f"<|im_start|>system\nYou are a strict deductive logic theorem prover.<|im_end|>\n"
            f"<|im_start|>user\n{inp}<|im_end|>\n"
            f"<|im_start|>assistant\nFinal Answer: "
        )
        scores = get_contextual_choice_scores(prompt, ["valid", "invalid"], resonance_beta=0.3)
        preds.append("valid" if scores[0] > scores[1] else "invalid")
    return preds


def evaluate_truthfulqa_suite(items):
    preds = []
    for it in items:
        q = it["question"]
        choices = it["choices"]
        prompt = (
            f"<|im_start|>system\nYou are a factual, scientific analyzer rejecting popular myths.<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {q}<|im_end|>\n"
            f"<|im_start|>assistant\nAnswer: "
        )
        # TruthfulQAは完結文なので局所減算を弱めにする (beta=0.1)
        scores = get_contextual_choice_scores(prompt, choices, resonance_beta=0.1)
        best_k = int(np.argmax(scores))
        preds.append(chr(65 + best_k))
    return preds


def evaluate_moral_suite(items):
    preds = []
    for it in items:
        q = it["question"]
        choices = it["choices"]
        prompt = (
            f"<|im_start|>system\nYou are an ethical evaluator weighing moral duties and harms.<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {q}<|im_end|>\n"
            f"<|im_start|>assistant\nJudgment: "
        )
        scores = get_contextual_choice_scores(prompt, choices, resonance_beta=0.2)
        best_k = int(np.argmax(scores))
        preds.append(chr(65 + best_k))
    return preds


def run_benchmark():
    suites = load_web_standard_suites()
    t0 = time.time()

    runners = [
        ("causal_judgement", evaluate_causal_suite),
        ("disambiguation_qa", evaluate_disambiguation_suite),
        ("formal_fallacies", evaluate_formal_fallacies_suite),
        ("truthfulqa_mc1", evaluate_truthfulqa_suite),
        ("mmlu_moral", evaluate_moral_suite),
    ]

    all_results = {}
    total_correct = 0
    total_items = 0

    print("=" * 80)
    print("STARTING CR-CE FULL BENCHMARK (N=250)")
    print("Principles: Entropy-Descent Model + Contextual Resonance (Mutual Information)")
    print("=" * 80)

    for suite_name, runner in runners:
        items = suites[suite_name]
        t_suite = time.time()
        preds = runner(items)

        correct = 0
        for it, p in zip(items, preds):
            if suite_name == "formal_fallacies":
                target = it["target"].strip().lower()
                ok = (p.lower() == target)
            elif suite_name == "causal_judgement":
                target = it["target"].strip()
                ok = (p.lower() == target.lower())
            elif suite_name == "disambiguation_qa":
                target = it["target"].strip().strip("(). ")
                ok = (p.lower() == target.lower())
            elif suite_name == "truthfulqa_mc1":
                target = chr(65 + it["correct_idx"])
                ok = (p.upper() == target.upper())
            elif suite_name == "mmlu_moral":
                target = chr(65 + it["answer"])
                ok = (p.upper() == target.upper())
            else:
                ok = False

            if ok:
                correct += 1

        acc = (correct / len(items)) * 100.0
        elapsed_s = time.time() - t_suite
        print(f"[{suite_name}] Accuracy: {acc:.2f}% ({correct}/{len(items)}) [Time: {elapsed_s:.1f}s]")

        all_results[suite_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(items),
            "elapsed_sec": round(elapsed_s, 2)
        }
        total_correct += correct
        total_items += len(items)

    total_acc = (total_correct / total_items) * 100.0
    total_elapsed = time.time() - t0

    print("=" * 80)
    print(f"CR-CE BENCHMARK COMPLETE: {total_acc:.2f}% ({total_correct}/{total_items}) in {total_elapsed:.1f}s")
    print("=" * 80)

    summary = {
        "engine": "Contextual Resonance & Coherence Engine (CR-CE)",
        "model": QWEN_ID,
        "total_accuracy": round(total_acc, 2),
        "total_correct": total_correct,
        "total_items": total_items,
        "total_time_sec": round(total_elapsed, 1),
        "suites": all_results
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {RESULTS_JSON}")


if __name__ == "__main__":
    run_benchmark()
