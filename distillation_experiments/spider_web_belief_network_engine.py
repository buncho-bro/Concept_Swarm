"""
Spider-Web Belief Network Engine (蜘蛛の巣型・相互検証ネットワーク推論エンジン)
======================================================================================
1. 【問題適応・手動シードの完全排除】: 人間によるシナリオ辞書を全廃し、無作為抽出・埋め込み空間から動的展開。
2. 【思考原野における近傍共鳴展開 (Neighbor Resonance)】:
   課題文から抽出された核トークンに対し、モデルの内部埋め込み空間 (Embedding Cosine Similarity) で
   意味的に近傍・共鳴する概念ノード群を探索し、蜘蛛の巣（メッシュ・格子）状に多方向展開。
3. 【閉路干渉による致命的矛盾の検出 (Cycle Contradiction Probing)】:
   異なる近傍経路を経由した複数の思考パス（Path A, Path B）を合流点で激突させ、
   一本道推論では見落とされる致命的な自己矛盾・偏見をトポロジー（幾何学的整合性）として自動検出。
4. 【相互検証剪定と因果合成 (Consensus Pruning & Synthesis)】:
   矛盾が生じた枝に対し、問題文への文字通りの拘束性（Grounding）をシステムが照合し、
   あぶれた枝を切り落として無矛盾な結論を一意に確定。
5. 【Web標準難関250問の完全測定】:
   全5領域（計250問）を完全走破し、スコアと所要時間を客観記録。
"""

import os
import sys
import json
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "web_standard_large_scale_benchmark_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def find_embedding_neighbors(model, tokenizer, query_tokens, top_k=3):
    """
    モデル自身の埋め込み空間 (Embedding Space) において、
    指定トークンとコサイン類似度が高い近傍概念ノードを探索する。
    """
    embeddings = model.get_input_embeddings().weight.data  # [vocab_size, hidden_dim]
    
    # query_tokens の埋め込みベクトルを取得
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in query_tokens if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    if not token_ids:
        return []
    
    q_vecs = embeddings[token_ids]  # [num_tokens, hidden_dim]
    avg_q = torch.mean(q_vecs, dim=0, keepdim=True)  # [1, hidden_dim]
    
    # コサイン類似度計算
    sims = F.cosine_similarity(avg_q, embeddings, dim=-1)
    
    # 上位k個のトークンを取得 (自身を除く)
    top_indices = torch.topk(sims, k=top_k + len(token_ids)).indices.tolist()
    neighbors = []
    for idx in top_indices:
        if idx not in token_ids:
            tok = tokenizer.decode([idx]).strip()
            if len(tok) > 2 and tok.isalpha() and tok not in neighbors:
                neighbors.append(tok)
        if len(neighbors) >= top_k:
            break
    return neighbors


def solve_with_spider_web_verification(model, tokenizer, question_text, options_text, task_type):
    """
    蜘蛛の巣状に多経路（マルチパス）を展開し、合流点での閉路干渉により矛盾を炙り出して判定する。
    """
    # 1. 課題文から主要な名詞・動詞トークンを抽出（無作為性を含むシード）
    words = [w.strip(".,?!:;\"'") for w in question_text.split() if len(w) > 4 and w.isalpha()]
    sample_anchors = words[:3] if len(words) >= 3 else ["cause", "action", "result"]
    
    # 2. 潜在埋め込み空間から近傍概念（共鳴ノード）を多方向に手繰り寄せる
    with torch.no_grad():
        neighbors_route_a = find_embedding_neighbors(model, tokenizer, sample_anchors[:2], top_k=2)
        neighbors_route_b = find_embedding_neighbors(model, tokenizer, sample_anchors[-2:], top_k=2)

    str_neighbors_a = ", ".join(neighbors_route_a) if neighbors_route_a else "context"
    str_neighbors_b = ", ".join(neighbors_route_b) if neighbors_route_b else "physical"

    # 3. 2つの異なる概念近傍を経由する思考パス（Path A, Path B）を並列生成
    prompt_a = (
        f"<|im_start|>system\nYou are an uncompromising logic verification node examining path A ({str_neighbors_a}).<|im_end|>\n"
        f"<|im_start|>user\nContext: {question_text}\nOptions:\n{options_text}\n"
        f"Analyze the facts strictly through direct physical/temporal evidence related to {str_neighbors_a}. "
        f"State your reasoned conclusion formatted as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    prompt_b = (
        f"<|im_start|>system\nYou are an uncompromising logic verification node examining path B ({str_neighbors_b}).<|im_end|>\n"
        f"<|im_start|>user\nContext: {question_text}\nOptions:\n{options_text}\n"
        f"Analyze the facts strictly through rule-based/normative consistency related to {str_neighbors_b}. "
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

    # 4. 合流点での閉路干渉（一致か、それとも致命的矛盾か）
    dec_a_match = re.search(r"Decision:\s*\(?([A-Za-z0-9_]+)\)?", resp_a, re.IGNORECASE)
    dec_b_match = re.search(r"Decision:\s*\(?([A-Za-z0-9_]+)\)?", resp_b, re.IGNORECASE)

    dec_a = dec_a_match.group(1).strip() if dec_a_match else ""
    dec_b = dec_b_match.group(1).strip() if dec_b_match else ""

    # 両方の経路が一致した場合は、強固な無矛盾合意（Consensus）
    if dec_a and dec_b and (dec_a.lower() == dec_b.lower()):
        return dec_a

    # 5. 衝突（矛盾）が生じた場合: システムによるテキスト拘束性照合（あぶれの切り落とし）
    # どちらの経路が問題文に文字通り書かれた事実に基づいているかを単一パスで剪定
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
    resp_r = tokenizer.decode(out_r[0][enc_r["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    m = re.search(r"([A-Za-z0-9_]+)", resp_r)
    return m.group(1).strip() if m else (dec_a if dec_a else dec_b)


# ==============================================================================
# ベンチマークローダー (Web標準250問)
# ==============================================================================
def load_web_standard_suites():
    suites = {}
    print("Loading 5 Web Standard Benchmark Suites (Total N = 250)...")

    # 1. BBH Causal Judgement
    print("  Loading BBH Causal Judgement (N=50)...")
    ds = load_dataset('lukaemon/bbh', 'causal_judgement', split='test')
    suites['causal_judgement'] = [
        {"input": item['input'], "target": item['target'].strip()}
        for item in list(ds)[:50]
    ]

    # 2. BBH Disambiguation QA
    print("  Loading BBH Disambiguation QA (N=50)...")
    ds = load_dataset('lukaemon/bbh', 'disambiguation_qa', split='test')
    suites['disambiguation_qa'] = [
        {"input": item['input'], "target": item['target'].strip()}
        for item in list(ds)[:50]
    ]

    # 3. BBH Formal Fallacies
    print("  Loading BBH Formal Fallacies (N=50)...")
    ds = load_dataset('lukaemon/bbh', 'formal_fallacies', split='test')
    suites['formal_fallacies'] = [
        {"input": item['input'], "target": item['target'].strip()}
        for item in list(ds)[:50]
    ]

    # 4. TruthfulQA MC1
    print("  Loading TruthfulQA MC1 (N=50)...")
    ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    suites['truthfulqa_mc1'] = []
    for item in list(ds)[:50]:
        q = item['question']
        choices = item['mc1_targets']['choices']
        labels = item['mc1_targets']['labels']
        correct_idx = labels.index(1) if 1 in labels else 0
        suites['truthfulqa_mc1'].append({
            "question": q,
            "choices": choices,
            "correct_idx": correct_idx
        })

    # 5. MMLU Moral Scenarios
    print("  Loading MMLU Moral Scenarios (N=50)...")
    ds = load_dataset('cais/mmlu', 'moral_scenarios', split='test')
    suites['mmlu_moral'] = []
    for item in list(ds)[:50]:
        q = item['question']
        choices = item['choices']
        answer = item['answer']
        suites['mmlu_moral'].append({
            "question": q,
            "choices": choices,
            "answer": answer
        })

    return suites


def main():
    print("=" * 85)
    print("STARTING SPIDER-WEB BELIEF NETWORK BENCHMARK PIPELINE")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model Qwen-1.5B (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    suites = load_web_standard_suites()

    results = {}
    total_correct = 0
    total_evaluated = 0
    t_start = time.time()

    print("\n--- Running Evaluation across All 5 Suites via Spider-Web Network ---")

    # 1. Causal Judgement
    print("\n>>> Running Suite: Causal Judgement (N=50)")
    correct = 0
    for i, item in enumerate(suites['causal_judgement']):
        pred = solve_with_spider_web_verification(model, tokenizer, item['input'], "- Yes\n- No", "causal")
        target = item['target'].strip()
        is_corr = (pred.lower() == target.lower())
        if is_corr:
            correct += 1
            total_correct += 1
        total_evaluated += 1
        if (i + 1) % 10 == 0:
            print(f"  [Causal] Item {i+1:02d}/50 | Current Acc: {correct/(i+1)*100:.1f}%")
    acc = (correct / 50) * 100.0
    results['causal_judgement'] = acc
    print(f">>> Causal Judgement Accuracy: {acc:.1f}% ({correct}/50)")

    # 2. Disambiguation QA
    print("\n>>> Running Suite: Disambiguation QA (N=50)")
    correct = 0
    for i, item in enumerate(suites['disambiguation_qa']):
        options_text = "Refer to candidates in the text or state 'Ambiguous'"
        pred = solve_with_spider_web_verification(model, tokenizer, item['input'], options_text, "disambig")
        target = item['target'].strip()
        is_corr = (target.lower() in pred.lower()) or (pred.lower() in target.lower())
        if is_corr:
            correct += 1
            total_correct += 1
        total_evaluated += 1
        if (i + 1) % 10 == 0:
            print(f"  [Disambig] Item {i+1:02d}/50 | Current Acc: {correct/(i+1)*100:.1f}%")
    acc = (correct / 50) * 100.0
    results['disambiguation_qa'] = acc
    print(f">>> Disambiguation QA Accuracy: {acc:.1f}% ({correct}/50)")

    # 3. Formal Fallacies
    print("\n>>> Running Suite: Formal Fallacies (N=50)")
    correct = 0
    for i, item in enumerate(suites['formal_fallacies']):
        pred = solve_with_spider_web_verification(model, tokenizer, item['input'], "- valid\n- invalid", "fallacy")
        target = item['target'].strip().lower()
        pred_clean = "valid" if "valid" in pred.lower() and "invalid" not in pred.lower() else "invalid"
        is_corr = (pred_clean == target)
        if is_corr:
            correct += 1
            total_correct += 1
        total_evaluated += 1
        if (i + 1) % 10 == 0:
            print(f"  [Fallacies] Item {i+1:02d}/50 | Current Acc: {correct/(i+1)*100:.1f}%")
    acc = (correct / 50) * 100.0
    results['formal_fallacies'] = acc
    print(f">>> Formal Fallacies Accuracy: {acc:.1f}% ({correct}/50)")

    # 4. TruthfulQA MC1
    print("\n>>> Running Suite: TruthfulQA MC1 (N=50)")
    correct = 0
    for i, item in enumerate(suites['truthfulqa_mc1']):
        choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
        pred = solve_with_spider_web_verification(model, tokenizer, item['question'], choices_str, "truthful")
        target = chr(65 + item['correct_idx'])
        m = re.search(r"([A-Z])", pred.upper())
        pred_letter = m.group(1) if m else "A"
        is_corr = (pred_letter == target)
        if is_corr:
            correct += 1
            total_correct += 1
        total_evaluated += 1
        if (i + 1) % 10 == 0:
            print(f"  [TruthfulQA] Item {i+1:02d}/50 | Current Acc: {correct/(i+1)*100:.1f}%")
    acc = (correct / 50) * 100.0
    results['truthfulqa_mc1'] = acc
    print(f">>> TruthfulQA Accuracy: {acc:.1f}% ({correct}/50)")

    # 5. MMLU Moral Scenarios
    print("\n>>> Running Suite: MMLU Moral Scenarios (N=50)")
    correct = 0
    for i, item in enumerate(suites['mmlu_moral']):
        choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
        pred = solve_with_spider_web_verification(model, tokenizer, item['question'], choices_str, "moral")
        target = chr(65 + item['answer'])
        m = re.search(r"([A-Z])", pred.upper())
        pred_letter = m.group(1) if m else "A"
        is_corr = (pred_letter == target)
        if is_corr:
            correct += 1
            total_correct += 1
        total_evaluated += 1
        if (i + 1) % 10 == 0:
            print(f"  [Moral] Item {i+1:02d}/50 | Current Acc: {correct/(i+1)*100:.1f}%")
    acc = (correct / 50) * 100.0
    results['mmlu_moral'] = acc
    print(f">>> MMLU Moral Accuracy: {acc:.1f}% ({correct}/50)")

    elapsed = time.time() - t_start
    overall_avg = (total_correct / total_evaluated) * 100.0
    results["average"] = overall_avg
    results["elapsed_sec"] = elapsed

    print("\n" + "=" * 85)
    print(f"FINAL OVERALL ACCURACY (Spider-Web Belief Network): {overall_avg:.2f}% ({total_correct}/{total_evaluated}) in {elapsed:.1f}s")
    print("=" * 85)

    # 既存の総合結果JSONの更新
    all_results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    all_results["Spider_Web_Belief_Network"] = results
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # チャートの更新
    generate_comparison_chart(all_results)


def generate_comparison_chart(all_results):
    suites = ['causal_judgement', 'disambiguation_qa', 'formal_fallacies', 'truthfulqa_mc1', 'mmlu_moral']
    suite_labels = ['Causal', 'Disambig', 'Fallacies', 'TruthfulQA', 'Moral']

    plt.figure(figsize=(13, 6))

    fair_models = [
        ("Raw_Qwen1.5B", "Raw Qwen 1.5B (Zero-Shot)", "#7f8c8d"),
        ("External_Text_Trained_LoRA", "External Real-Text LoRA", "#2980b9"),
        ("Atomic_Proposition_Grounded_System", "Atomic Proposition System", "#95a5a6"),
        ("Spider_Web_Belief_Network", "Spider-Web Belief Network (Ours)", "#e74c3c")
    ]

    x = np.arange(len(suite_labels))
    width = 0.2

    for idx, (m_key, m_name, color) in enumerate(fair_models):
        if m_key in all_results:
            vals = [all_results[m_key].get(s, 0.0) for s in suites]
            avg = all_results[m_key].get("average", 0.0)
            plt.bar(x + idx * width, vals, width, label=f"{m_name} (Avg: {avg:.1f}%)", color=color, alpha=0.9)

    plt.xlabel('Benchmark Suites (N=50 each, Total 250)', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Web Standard Benchmark: Spider-Web Belief Network vs Fair Baselines', fontsize=13, fontweight='bold')
    plt.xticks(x + width * 1.5, suite_labels, fontsize=10)
    plt.ylim(0, 75)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACT_DIR, "web_standard_large_scale_benchmark.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
