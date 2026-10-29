"""
Composite Atomic Proposition & Web-Grounded Spider-Web Reasoning Engine
(原子命題分解 × Web知識グラウンディング × 蜘蛛の巣相互検証 複合推論エンジン)
======================================================================================
1. 【原子命題分解 (Atomic Decomposition)】:
   課題文を文法・節構造に基づいて最小単位の客観的事実命題群 {P_1, P_2, ...} に機械分解。
2. 【動的Web知識グラウンディング (Precision Web Grounding)】:
   分解された事実群からLLMが核心検証クエリを生成し、Wikipedia APIから客観事実を入手。
3. 【蜘蛛の巣型相互検証 (Spider-Web Belief Verification)】:
   原子事実とWeb知識を結合し、潜在空間の近傍を経由する複数経路（Path A, Path B）を展開。
   合流点での閉路干渉により、大衆的俗説や論理の飛躍（あぶれた枝）を自動剪定。
4. 【複数回試行による統計的誠実性 (3 Full Runs Audit)】:
   全250問を3回独立して走破し、平均値 (Mean) と標準偏差 (Std) を客観記録。
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
MULTI_RUN_JSON = os.path.join(OUTPUT_DIR, "composite_atomic_spider_web_multi_run.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# ==============================================================================
# 1. 原子命題分解モジュール
# ==============================================================================
def decompose_to_atomic_propositions(text):
    raw_sentences = re.split(r'(?<=[.?!])\s+|\n+|(?:,\s*(?:and|but|however|so|because|although)\s+)', text)
    propositions = []
    for s in raw_sentences:
        clean_s = s.strip()
        if len(clean_s) > 10:
            propositions.append(clean_s)
    return propositions if propositions else [text.strip()]


# ==============================================================================
# 2. LLMクエリ生成 & Web知識取得モジュール
# ==============================================================================
def fetch_web_grounding(model, tokenizer, context_text):
    try:
        # LLMに核心検証クエリを1行で生成させる
        prompt = (
            f"<|im_start|>system\nYou are a precise query generator. Output ONLY a 2-4 word Wikipedia search query to verify the main claim.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context_text[:200]}\nQuery:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=10, do_sample=False)
        query = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        clean_query = re.sub(r'["\']', '', query).split('\n')[0].strip()
        if not clean_query:
            clean_query = context_text[:30]

        # Wikipedia 検索
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_query)}&format=json&srlimit=1"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'AntigravityResearch/1.0'})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        results = data.get("query", {}).get("search", [])
        if not results:
            return ""

        top_title = results[0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(top_title)}"
        req2 = urllib.request.Request(summary_url, headers={'User-Agent': 'AntigravityResearch/1.0'})
        with urllib.request.urlopen(req2, timeout=2.5) as resp2:
            s_data = json.loads(resp2.read().decode('utf-8'))
        
        extract = s_data.get("extract", "").strip()
        return extract[:300] if extract else ""
    except Exception:
        return ""


# ==============================================================================
# 3. 潜在空間近傍探索 (Embedding Neighbors)
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
# 4. 複合推論ソルバー (原子分解 × Web事実 × 蜘蛛の巣閉路干渉)
# ==============================================================================
def solve_composite(model, tokenizer, question_text, options_text, task_type):
    # 1. 原子命題に機械分解
    atomic_props = decompose_to_atomic_propositions(question_text)
    facts_block = "\n".join([f"- {p}" for p in atomic_props])

    # 2. Webから客観知識を入手
    web_fact = fetch_web_grounding(model, tokenizer, question_text)
    web_block = f"Verified External Knowledge:\n\"{web_fact}\"\n\n" if web_fact else ""

    # 3. 潜在空間の近傍共鳴ノード探索
    words = [w.strip(".,?!:;\"'") for w in question_text.split() if len(w) > 4 and w.isalpha()]
    anchors = words[:3] if len(words) >= 3 else ["fact", "cause", "logic"]
    with torch.no_grad():
        neighbors_a = find_embedding_neighbors(model, tokenizer, anchors[:2], top_k=2)
        neighbors_b = find_embedding_neighbors(model, tokenizer, anchors[-2:], top_k=2)

    str_a = ", ".join(neighbors_a) if neighbors_a else "direct physical evidence"
    str_b = ", ".join(neighbors_b) if neighbors_b else "deductive consistency"

    # 4. 蜘蛛の巣型・多経路並列展開
    prompt_a = (
        f"<|im_start|>system\nYou are a factual verification node examining atomic evidence and {str_a}.<|im_end|>\n"
        f"<|im_start|>user\n{web_block}Atomic Facts:\n{facts_block}\n\nOptions:\n{options_text}\n\n"
        f"Based strictly on direct physical and acute triggers, answer clearly as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nDecision: "
    )
    prompt_b = (
        f"<|im_start|>system\nYou are a logical verification node examining deductive validity and {str_b}.<|im_end|>\n"
        f"<|im_start|>user\n{web_block}Atomic Facts:\n{facts_block}\n\nOptions:\n{options_text}\n\n"
        f"Based strictly on rule consistency and non-contradiction, answer clearly as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nDecision: "
    )

    enc_a = tokenizer(prompt_a, return_tensors="pt").to(device)
    enc_b = tokenizer(prompt_b, return_tensors="pt").to(device)

    with torch.no_grad():
        out_a = model.generate(**enc_a, max_new_tokens=15, do_sample=False)
        out_b = model.generate(**enc_b, max_new_tokens=15, do_sample=False)

    resp_a = tokenizer.decode(out_a[0][enc_a["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    resp_b = tokenizer.decode(out_b[0][enc_b["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    m_a = re.search(r"([A-Za-z0-9_]+)", resp_a)
    m_b = re.search(r"([A-Za-z0-9_]+)", resp_b)
    dec_a = m_a.group(1).strip() if m_a else ""
    dec_b = m_b.group(1).strip() if m_b else ""

    # 合流点での閉路一致（無矛盾合意）
    if dec_a and dec_b and (dec_a.lower() == dec_b.lower()):
        return dec_a

    # 衝突（矛盾）が発生した場合の調停・剪定
    reconcile_prompt = (
        f"<|im_start|>system\nYou are a master judge reconciling conflicting paths using verified facts and atomic evidence.<|im_end|>\n"
        f"<|im_start|>user\n{web_block}Atomic Facts:\n{facts_block}\n\nOptions:\n{options_text}\n\n"
        f"Path A decided: '{dec_a}'\nPath B decided: '{dec_b}'\n"
        f"Reject ungrounded speculation. Select the single strictly verified option. "
        f"Answer clearly as 'Final Answer: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nFinal Answer: "
    )
    enc_r = tokenizer(reconcile_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_r = model.generate(**enc_r, max_new_tokens=10, do_sample=False)
    resp_r = tokenizer.decode(out_r[0][enc_r["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    m_r = re.search(r"([A-Za-z0-9_]+)", resp_r)
    return m_r.group(1).strip() if m_r else (dec_a if dec_a else dec_b)


# ==============================================================================
# 5. ベンチマークスイートローダー & 3回試行
# ==============================================================================
def load_web_standard_suites():
    suites = {}
    print("Loading 5 Web Standard Benchmark Suites (Total N = 250)...")
    ds_c = load_dataset('lukaemon/bbh', 'causal_judgement', split='test')
    suites['causal_judgement'] = [{"input": item['input'], "target": item['target'].strip()} for item in list(ds_c)[:50]]

    ds_d = load_dataset('lukaemon/bbh', 'disambiguation_qa', split='test')
    suites['disambiguation_qa'] = [{"input": item['input'], "target": item['target'].strip()} for item in list(ds_d)[:50]]

    ds_f = load_dataset('lukaemon/bbh', 'formal_fallacies', split='test')
    suites['formal_fallacies'] = [{"input": item['input'], "target": item['target'].strip()} for item in list(ds_f)[:50]]

    ds_t = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    suites['truthfulqa_mc1'] = []
    for item in list(ds_t)[:50]:
        labels = item['mc1_targets']['labels']
        suites['truthfulqa_mc1'].append({
            "question": item['question'],
            "choices": item['mc1_targets']['choices'],
            "correct_idx": labels.index(1) if 1 in labels else 0
        })

    ds_m = load_dataset('cais/mmlu', 'moral_scenarios', split='test')
    suites['mmlu_moral'] = [{"question": item['question'], "choices": item['choices'], "answer": item['answer']} for item in list(ds_m)[:50]]
    return suites


def run_benchmark_trial(model, tokenizer, suites, trial_idx):
    print(f"\n=======================================================")
    print(f"RUNNING COMPOSITE ENGINE TRIAL [{trial_idx + 1}/3]")
    print(f"=======================================================")
    
    trial_results = {}
    total_correct = 0
    total_evaluated = 0
    t0 = time.time()

    for suite_name, items in suites.items():
        correct = 0
        for i, item in enumerate(items):
            if suite_name == 'causal_judgement':
                pred = solve_composite(model, tokenizer, item['input'], "- Yes\n- No", "causal")
                target = item['target'].strip()
                is_corr = (pred.lower() == target.lower())
            elif suite_name == 'disambiguation_qa':
                pred = solve_composite(model, tokenizer, item['input'], "State the referent or 'Ambiguous'", "disambig")
                target = item['target'].strip()
                is_corr = (target.lower() in pred.lower()) or (pred.lower() in target.lower())
            elif suite_name == 'formal_fallacies':
                pred = solve_composite(model, tokenizer, item['input'], "- valid\n- invalid", "fallacy")
                target = item['target'].strip().lower()
                pred_clean = "valid" if "valid" in pred.lower() and "invalid" not in pred.lower() else "invalid"
                is_corr = (pred_clean == target)
            elif suite_name == 'truthfulqa_mc1':
                choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
                pred = solve_composite(model, tokenizer, item['question'], choices_str, "truthful")
                target = chr(65 + item['correct_idx'])
                m = re.search(r"([A-D])", pred.upper())
                is_corr = (m.group(1) == target) if m else False
            elif suite_name == 'mmlu_moral':
                choices_str = "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(item['choices'])])
                pred = solve_composite(model, tokenizer, item['question'], choices_str, "moral")
                target = chr(65 + item['answer'])
                m = re.search(r"([A-D])", pred.upper())
                is_corr = (m.group(1) == target) if m else False

            if is_corr:
                correct += 1
                total_correct += 1
            total_evaluated += 1

        acc = (correct / len(items)) * 100.0
        trial_results[suite_name] = acc
        print(f"  Trial {trial_idx + 1} - {suite_name}: {acc:.1f}% ({correct}/{len(items)})")

    elapsed = time.time() - t0
    trial_avg = (total_correct / total_evaluated) * 100.0
    trial_results["average"] = trial_avg
    trial_results["elapsed_sec"] = elapsed
    print(f">>> TRIAL [{trial_idx + 1}] OVERALL AVERAGE: {trial_avg:.2f}% in {elapsed:.1f}s")
    return trial_results


def main():
    print("=" * 85)
    print("STARTING COMPOSITE (ATOMIC × WEB-GROUNDED SPIDER-WEB) 3-RUN AUDIT")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model Qwen-1.5B (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    suites = load_web_standard_suites()

    all_trials = []
    num_trials = 3

    for t in range(num_trials):
        res = run_benchmark_trial(model, tokenizer, suites, t)
        all_trials.append(res)

    suite_names = ['causal_judgement', 'disambiguation_qa', 'formal_fallacies', 'truthfulqa_mc1', 'mmlu_moral', 'average']
    summary_stats = {}
    for s in suite_names:
        scores = [t[s] for t in all_trials]
        summary_stats[s] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores
        }

    print("\n" + "=" * 85)
    print("COMPOSITE ENGINE STATISTICAL AUDIT SUMMARY (3 Independent Full Runs)")
    print("=" * 85)
    for s in suite_names:
        stat = summary_stats[s]
        print(f"  - {s.upper():20s}: Mean = {stat['mean']:.2f}% ± {stat['std']:.2f}% | Trials = {stat['scores']}")

    multi_results = {
        "trials": all_trials,
        "summary": summary_stats
    }
    with open(MULTI_RUN_JSON, "w", encoding="utf-8") as f:
        json.dump(multi_results, f, indent=2)
    print(f"\nAudit results saved to: {MULTI_RUN_JSON}")

    # 総合結果JSONの更新
    all_results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    all_results["Composite_Atomic_Spider_Web"] = {
        s: summary_stats[s]["mean"] for s in suite_names if s != "average"
    }
    all_results["Composite_Atomic_Spider_Web"]["average"] = summary_stats["average"]["mean"]
    all_results["Composite_Atomic_Spider_Web"]["elapsed_sec"] = float(np.mean([t["elapsed_sec"] for t in all_trials]))

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    generate_comparison_chart(all_results)


def generate_comparison_chart(all_results):
    suites = ['causal_judgement', 'disambiguation_qa', 'formal_fallacies', 'truthfulqa_mc1', 'mmlu_moral']
    suite_labels = ['Causal', 'Disambig', 'Fallacies', 'TruthfulQA', 'Moral']

    plt.figure(figsize=(14, 6))

    fair_models = [
        ("Raw_Qwen1.5B", "Raw Qwen 1.5B", "#7f8c8d"),
        ("External_Text_Trained_LoRA", "External Text (Alpaca)", "#2980b9"),
        ("Atomic_Proposition_Grounded_System", "Atomic Proposition", "#95a5a6"),
        ("Web_Grounded_Spider_Web", "Web Spider-Web (Mean)", "#e67e22"),
        ("Composite_Atomic_Spider_Web", "Composite (Atomic × Web Mesh)", "#27ae60")
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
    plt.title('Web Standard Benchmark: Composite Engine (Atomic × Web Mesh) vs Baselines', fontsize=13, fontweight='bold')
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
