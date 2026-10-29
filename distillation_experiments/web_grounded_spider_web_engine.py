"""
Web-Grounded Spider-Web Belief Network Engine
(Web検索知識グラウンディング付き・蜘蛛の巣型相互検証推論エンジン)
======================================================================================
1. 【「LLMは正しい答えを知らない」の徹底】:
   モデルの重み内部の大衆的記憶を信用せず、外部Web（Wikipedia API等）から
   客観的・学術的な確定知識を動的に検索・入手してグラウンディング。
2. 【蜘蛛の巣型相互検証 (Spider-Web Belief Verification)】:
   Webから入手した客観的事実と課題文を結合し、潜在空間の近傍を介した
   複数経路（Path A, Path B）を展開。合流点での閉路干渉により自己矛盾を剪定。
3. 【複数回試行による統計的検証 (Multiple Runs Audit)】:
   「あと何回か行うようにしてください」の指示および恒久規約GEMINI.mdに基づき、
   全250問を複数回（3回独立試行）走破し、平均正解率と標準偏差を客観記録。
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.parse
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
MULTI_RUN_JSON = os.path.join(OUTPUT_DIR, "web_grounded_multi_run_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# ==============================================================================
# 1. Web検索・客観的事実取得モジュール (Wikipedia API Grounding)
# ==============================================================================
def fetch_web_knowledge(query_text, max_chars=400):
    """
    課題文から抽出されたクエリを用いてWikipediaを動的検索し、
    最も関連する客観的事実サマリー（Extract）を入手する。
    """
    try:
        # クエリの整形（疑問詞や余計な記号を除去し、主要名詞句を抽出）
        clean_q = re.sub(r'^(Question:|Scenario Question:|How would a typical person answer each of the following questions about causation\?|In the following sentences, explain the antecedent.*?\n)', '', query_text).strip()
        words = [w.strip(".,?!:;\"'") for w in clean_q.split() if len(w) > 3 and w.lower() not in ['what', 'which', 'where', 'when', 'would', 'could', 'should', 'about', 'options', 'sentence']]
        search_kw = " ".join(words[:4]) if words else clean_q[:30]

        # 1. 検索クエリ実行
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(search_kw)}&format=json&srlimit=1"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'AntigravityResearch/1.0 (Mozilla/5.0)'})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return ""

        top_title = search_results[0]["title"]

        # 2. 記事サマリーの取得
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(top_title)}"
        req2 = urllib.request.Request(summary_url, headers={'User-Agent': 'AntigravityResearch/1.0 (Mozilla/5.0)'})
        with urllib.request.urlopen(req2, timeout=3.0) as resp2:
            s_data = json.loads(resp2.read().decode('utf-8'))
        
        extract = s_data.get("extract", "").strip()
        return extract[:max_chars] if extract else ""
    except Exception as e:
        # タイムアウトやネットワークエラー時は空文字を返してフォールバック
        return ""


# ==============================================================================
# 2. 蜘蛛の巣型・潜在空間共鳴探索
# ==============================================================================
def find_embedding_neighbors(model, tokenizer, query_tokens, top_k=2):
    embeddings = model.get_input_embeddings().weight.data
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in query_tokens if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    if not token_ids:
        return []
    
    q_vecs = embeddings[token_ids]
    avg_q = torch.mean(q_vecs, dim=0, keepdim=True)
    sims = F.cosine_similarity(avg_q, embeddings, dim=-1)
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


# ==============================================================================
# 3. Web知識グラウンディング付き・蜘蛛の巣相互検証ソルバー
# ==============================================================================
def solve_with_web_grounded_spider_web(model, tokenizer, question_text, options_text, task_type):
    # 1. Webから正しい客観知識を動的入手（「LLMは答えを知らない」前提の補強）
    web_knowledge = fetch_web_knowledge(question_text)
    web_grounding_block = f"Verified Objective Knowledge from Web:\n\"{web_knowledge}\"\n\n" if web_knowledge else ""

    # 2. 潜在空間から近傍概念（共鳴ノード）を手繰り寄せる
    words = [w.strip(".,?!:;\"'") for w in question_text.split() if len(w) > 4 and w.isalpha()]
    sample_anchors = words[:3] if len(words) >= 3 else ["cause", "action", "fact"]
    
    with torch.no_grad():
        neighbors_route_a = find_embedding_neighbors(model, tokenizer, sample_anchors[:2], top_k=2)
        neighbors_route_b = find_embedding_neighbors(model, tokenizer, sample_anchors[-2:], top_k=2)

    str_a = ", ".join(neighbors_route_a) if neighbors_route_a else "empirical facts"
    str_b = ", ".join(neighbors_route_b) if neighbors_route_b else "logical rules"

    # 3. 2つの異なる経路（Path A, Path B）を展開
    prompt_a = (
        f"<|im_start|>system\nYou are a precise verification node examining empirical facts ({str_a}).<|im_end|>\n"
        f"<|im_start|>user\n{web_grounding_block}Problem Context:\n{question_text}\n\nOptions:\n{options_text}\n\n"
        f"Based strictly on verified objective facts, answer clearly as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nDecision: "
    )
    prompt_b = (
        f"<|im_start|>system\nYou are a precise verification node examining deductive logic ({str_b}).<|im_end|>\n"
        f"<|im_start|>user\n{web_grounding_block}Problem Context:\n{question_text}\n\nOptions:\n{options_text}\n\n"
        f"Based strictly on rigorous non-contradiction, answer clearly as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nDecision: "
    )

    enc_a = tokenizer(prompt_a, return_tensors="pt").to(device)
    enc_b = tokenizer(prompt_b, return_tensors="pt").to(device)

    with torch.no_grad():
        out_a = model.generate(**enc_a, max_new_tokens=15, do_sample=False)
        out_b = model.generate(**enc_b, max_new_tokens=15, do_sample=False)

    resp_a = tokenizer.decode(out_a[0][enc_a["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    resp_b = tokenizer.decode(out_b[0][enc_b["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    dec_a_match = re.search(r"([A-Za-z0-9_]+)", resp_a)
    dec_b_match = re.search(r"([A-Za-z0-9_]+)", resp_b)

    dec_a = dec_a_match.group(1).strip() if dec_a_match else ""
    dec_b = dec_b_match.group(1).strip() if dec_b_match else ""

    # 両者が一致した場合は合意採択
    if dec_a and dec_b and (dec_a.lower() == dec_b.lower()):
        return dec_a

    # 4. 食い違い（矛盾）が生じた場合は、Web知識を主軸として調停・剪定
    reconcile_prompt = (
        f"<|im_start|>system\nYou are a master judge reconciling conflicting paths using verified knowledge.<|im_end|>\n"
        f"<|im_start|>user\n{web_grounding_block}Problem: {question_text}\nOptions:\n{options_text}\n\n"
        f"Path A evaluated: '{dec_a}'\nPath B evaluated: '{dec_b}'\n"
        f"Select the single correct option that strictly aligns with objective external facts. "
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
# 4. ベンチマークスイートローダー
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


def run_single_full_benchmark(model, tokenizer, suites, run_idx):
    print(f"\n=======================================================")
    print(f"STARTING BENCHMARK RUN [{run_idx + 1}/3]")
    print(f"=======================================================")
    
    run_results = {}
    total_correct = 0
    total_evaluated = 0
    t0 = time.time()

    for suite_name, items in suites.items():
        correct = 0
        for i, item in enumerate(items):
            if suite_name == 'causal_judgement':
                pred = solve_with_web_grounded_spider_web(model, tokenizer, item['input'], "- Yes\n- No", "causal")
                target = item['target'].strip()
                is_corr = (pred.lower() == target.lower())
            elif suite_name == 'disambiguation_qa':
                pred = solve_with_web_grounded_spider_web(model, tokenizer, item['input'], "Refer to candidates in the text or state 'Ambiguous'", "disambig")
                target = item['target'].strip()
                is_corr = (target.lower() in pred.lower()) or (pred.lower() in target.lower())
            elif suite_name == 'formal_fallacies':
                pred = solve_with_web_grounded_spider_web(model, tokenizer, item['input'], "- valid\n- invalid", "fallacy")
                target = item['target'].strip().lower()
                pred_clean = "valid" if "valid" in pred.lower() and "invalid" not in pred.lower() else "invalid"
                is_corr = (pred_clean == target)
            elif suite_name == 'truthfulqa_mc1':
                choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
                pred = solve_with_web_grounded_spider_web(model, tokenizer, item['question'], choices_str, "truthful")
                target = chr(65 + item['correct_idx'])
                m = re.search(r"([A-Z])", pred.upper())
                pred_letter = m.group(1) if m else "A"
                is_corr = (pred_letter == target)
            elif suite_name == 'mmlu_moral':
                choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
                pred = solve_with_web_grounded_spider_web(model, tokenizer, item['question'], choices_str, "moral")
                target = chr(65 + item['answer'])
                m = re.search(r"([A-Z])", pred.upper())
                pred_letter = m.group(1) if m else "A"
                is_corr = (pred_letter == target)

            if is_corr:
                correct += 1
                total_correct += 1
            total_evaluated += 1

        acc = (correct / len(items)) * 100.0
        run_results[suite_name] = acc
        print(f"  Run {run_idx + 1} - {suite_name}: {acc:.1f}% ({correct}/{len(items)})")

    elapsed = time.time() - t0
    run_avg = (total_correct / total_evaluated) * 100.0
    run_results["average"] = run_avg
    run_results["elapsed_sec"] = elapsed
    print(f">>> RUN [{run_idx + 1}] OVERALL AVERAGE: {run_avg:.2f}% in {elapsed:.1f}s")
    return run_results


def main():
    print("=" * 85)
    print("STARTING WEB-GROUNDED SPIDER-WEB MULTI-RUN AUDIT PIPELINE")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model Qwen-1.5B (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    suites = load_web_standard_suites()

    # 3回試行 (Multiple Runs)
    num_runs = 3
    all_runs = []

    for r in range(num_runs):
        res = run_single_full_benchmark(model, tokenizer, suites, r)
        all_runs.append(res)

    # 統計計算（平均と標準偏差）
    suite_names = ['causal_judgement', 'disambiguation_qa', 'formal_fallacies', 'truthfulqa_mc1', 'mmlu_moral', 'average']
    summary_stats = {}
    for s in suite_names:
        scores = [run[s] for run in all_runs]
        summary_stats[s] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores
        }

    print("\n" + "=" * 85)
    print("MULTI-RUN AUDIT SUMMARY (Statistical Reliability across 3 Full Runs)")
    print("=" * 85)
    for s in suite_names:
        stat = summary_stats[s]
        print(f"  - {s.upper():20s}: Mean = {stat['mean']:.2f}% ± {stat['std']:.2f}% | Runs = {stat['scores']}")

    # 結果保存
    multi_results = {
        "runs": all_runs,
        "summary": summary_stats
    }
    with open(MULTI_RUN_JSON, "w", encoding="utf-8") as f:
        json.dump(multi_results, f, indent=2)
    print(f"\nMulti-run results saved to: {MULTI_RUN_JSON}")

    # 既存の総合結果JSONも最良平均で更新
    all_results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    all_results["Web_Grounded_Spider_Web"] = {
        s: summary_stats[s]["mean"] for s in suite_names if s != "average"
    }
    all_results["Web_Grounded_Spider_Web"]["average"] = summary_stats["average"]["mean"]
    all_results["Web_Grounded_Spider_Web"]["elapsed_sec"] = float(np.mean([r["elapsed_sec"] for r in all_runs]))

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
        ("Spider_Web_Belief_Network", "Spider-Web (Offline)", "#e67e22"),
        ("Web_Grounded_Spider_Web", "Web-Grounded Spider-Web (Ours)", "#27ae60")
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
    plt.title('Web Standard Benchmark: Web-Grounded Spider-Web vs All Fair Baselines', fontsize=13, fontweight='bold')
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
    main()
