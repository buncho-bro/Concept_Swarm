"""
Unbiased Spider-Web Belief Network Benchmark Suite
(思考原野全単語からの無作為抽出・回答結合・三角形閉路矛盾剪定によるWeb標準250問完全ベンチマーク)
==================================================================================================
1. 【全単語無作為抽出 (Uniform Random Sampling)】:
   偏りを完全に排除するため、語彙空間全40,150概念から一様乱数でトークンを抽出。
2. 【思考原野近傍結合と視点回答 (Neighbor Pairing & Perspective Answering)】:
   無作為トークンと思考原野（埋め込み空間）の近傍単語を組み合わせ、多角的な視点から問題に回答。
3. 【回答網の接続と再帰拡大 (Web Linkage & Expansion)】:
   導出された新たな単語と元のトークンを結合し、さらに近傍とつないで蜘蛛の巣状に回答群をネットワーク化。
4. 【相互閉路干渉による致命的矛盾の検出・剪定 (Cycle Contradiction Pruning)】:
   相互につながった結節点で、前提との矛盾や論理的飛躍を検知し、誤答の枝を切り落とす。
5. 【Web標準難関250問の完全測定】:
   全5領域（各50問、計250問）を走破し、正確な統計と客観スコアを記録。
"""

import os
import sys
import json
import time
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "unbiased_spider_web_benchmark_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer:", QWEN_ID)
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()

embed_weights = model.get_input_embeddings().weight.data
vocab_size = embed_weights.shape[0]

# 思考原野にある全単語（語彙空間）のインデックス構築
print("Indexing complete vocabulary for Thought Wilderness...")
FULL_WILDERNESS_VOCAB = []
VOCAB_ID_MAP = {}
for idx in range(vocab_size):
    s = tokenizer.decode([idx]).strip().lower()
    if len(s) >= 3 and s.isascii() and s.isalpha():
        if s not in VOCAB_ID_MAP:
            FULL_WILDERNESS_VOCAB.append(s)
            VOCAB_ID_MAP[s] = idx

print(f"Indexed {len(FULL_WILDERNESS_VOCAB)} unbiased Thought Wilderness concepts.")


def sample_random_tokens(k=3):
    """思考原野の全単語から完全に偏りなく無作為抽出"""
    return random.sample(FULL_WILDERNESS_VOCAB, k)


def get_nearest_neighbor(concept_word):
    """思考原野（埋め込み空間）において指定単語の最寄りの近傍概念を探索"""
    tok_id = VOCAB_ID_MAP.get(concept_word)
    if tok_id is None:
        ids = tokenizer.encode(concept_word, add_special_tokens=False)
        if not ids:
            return "context"
        tok_id = ids[0]
        
    vec = embed_weights[tok_id].unsqueeze(0)
    sims = F.cosine_similarity(vec, embed_weights, dim=-1)
    top_indices = torch.topk(sims, k=15).indices.tolist()
    
    for idx in top_indices:
        tok = tokenizer.decode([idx]).strip().lower()
        if len(tok) >= 3 and tok.isascii() and tok.isalpha() and tok != concept_word:
            return tok
    return "context"


def synthesize_concept(tok_a, tok_b):
    """(無作為トークン) + (近傍単語) -> 新たな単語"""
    id_a = VOCAB_ID_MAP.get(tok_a)
    id_b = VOCAB_ID_MAP.get(tok_b)
    if id_a is None or id_b is None:
        return f"{tok_a}_{tok_b}"
    vec_syn = 0.5 * embed_weights[id_a] + 0.5 * embed_weights[id_b]
    sims = F.cosine_similarity(vec_syn.unsqueeze(0), embed_weights, dim=-1)
    top_indices = torch.topk(sims, k=10).indices.tolist()
    for idx in top_indices:
        tok = tokenizer.decode([idx]).strip().lower()
        if len(tok) >= 3 and tok.isascii() and tok.isalpha() and tok != tok_a and tok != tok_b:
            return tok
    return f"{tok_a}_{tok_b}"


def solve_with_unbiased_spider_web(problem_text, options_text=""):
    """
    無作為トークンと思考原野近傍の結合から蜘蛛の巣状に回答をつなぎ、
    矛盾を剪定して最終解を導出する。
    """
    # 1. 思考原野から無作為に3つのトークンを抽出
    random_tokens = sample_random_tokens(k=3)
    
    # 2. 各無作為トークンの思考原野近傍を探索し、新単語を合成
    paths = []
    for r in random_tokens:
        n = get_nearest_neighbor(r)
        w_new = synthesize_concept(r, n)
        paths.append({
            "random_token": r,
            "neighbor": n,
            "synthesized_concept": w_new
        })
        
    # 3. 各経路の視点（三角形の結びつき）を通して問題に回答させる
    path_answers = []
    for p in paths:
        concept_combo = f"{p['random_token']} - {p['neighbor']} ({p['synthesized_concept']})"
        prompt = (
            f"<|im_start|>system\nYou are an objective verification node evaluating from conceptual lens: {concept_combo}.<|im_end|>\n"
            f"<|im_start|>user\nProblem: {problem_text}\n"
            f"{options_text}\n"
            f"From the critical stance of testing alternative possibilities and avoiding invalid assumptions, "
            f"state your concise reasoned decision. End strictly as 'Decision: <choice>'.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=25, do_sample=False)
        resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        m = re.search(r"Decision:\s*\(?([A-Za-z0-9_]+)\)?", resp, re.IGNORECASE)
        dec = m.group(1).strip() if m else ""
        p["decision"] = dec
        p["response"] = resp
        path_answers.append(p)
        
    # 4. 回答同士をつなぎ、合流点での閉路干渉による矛盾検知
    decisions = [p["decision"].lower() for p in path_answers if p["decision"]]
    
    # 全員一致の場合（完全合意）
    if len(decisions) >= 2 and all(d == decisions[0] for d in decisions):
        return path_answers[0]["decision"]
        
    # 5. 意見が割れた場合: 相互の結びつきを生かして致命的矛盾や飛躍のある枝を剪定
    reconcile_prompt = (
        f"<|im_start|>system\nYou are a master judge reconciling a spider-web belief network.<|im_end|>\n"
        f"<|im_start|>user\nProblem: {problem_text}\n"
        f"{options_text}\n\n"
        f"Path 1 ({path_answers[0]['synthesized_concept']}) suggests: '{path_answers[0]['decision']}'\n"
        f"Path 2 ({path_answers[1]['synthesized_concept']}) suggests: '{path_answers[1]['decision']}'\n"
        f"Path 3 ({path_answers[2]['synthesized_concept']}) suggests: '{path_answers[2]['decision']}'\n\n"
        f"Identify which path commits an unwarranted leap, fallacy, or popular myth. "
        f"Reject the contradiction and state the strictly valid choice as 'Final Answer: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nFinal Answer: "
    )
    enc_r = tokenizer(reconcile_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_r = model.generate(**enc_r, max_new_tokens=10, do_sample=False)
    resp_r = tokenizer.decode(out_r[0][enc_r["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    m_fin = re.search(r"([A-Za-z0-9_]+)", resp_r)
    return m_fin.group(1).strip() if m_fin else (decisions[0] if decisions else "")


# ==============================================================================
# ベンチマーク評価ループ
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_unbiased_benchmark():
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    start_time = time.time()
    
    print("\n================================================================================")
    print("STARTING UNBIASED SPIDER-WEB BENCHMARK (N=250 items across 5 domains)")
    print("================================================================================")
    
    for suite_name, items in suites.items():
        print(f"\n--- Testing Suite: {suite_name} (N={len(items)}) ---")
        correct = 0
        suite_records = []
        
        for i, it in enumerate(items):
            if suite_name in ['causal_judgement', 'disambiguation_qa', 'formal_fallacies']:
                inp = it['input']
                target = it['target'].strip()
                options_text = "Options: Yes or No" if suite_name in ['causal_judgement', 'formal_fallacies'] else ""
            elif suite_name == 'truthfulqa_mc1':
                inp = it['question']
                options_text = "Options:\n" + "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(it['choices'])])
                target = chr(65 + it['correct_idx'])
            elif suite_name == 'mmlu_moral':
                inp = it['question']
                options_text = "Options:\n" + "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(it['choices'])])
                target = chr(65 + it['answer'])
            else:
                inp = it.get('input', it.get('question', ''))
                target = str(it.get('target', it.get('answer', '')))
                options_text = ""
                    
            pred = solve_with_unbiased_spider_web(inp, options_text)
            
            # 正誤判定
            is_correct = False
            p_clean = pred.lower().strip("()., ")
            t_clean = target.lower().strip("()., ")
            
            if p_clean == t_clean:
                is_correct = True
            elif len(t_clean) == 1 and p_clean.startswith(t_clean):
                is_correct = True
            elif t_clean in ['yes', 'no'] and p_clean in ['yes', 'no'] and t_clean == p_clean:
                is_correct = True
                
            if is_correct:
                correct += 1
                
            suite_records.append({
                "item_idx": i + 1,
                "input": inp[:100],
                "target": target,
                "predicted": pred,
                "correct": is_correct
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                curr_acc = (correct / (i + 1)) * 100.0
                print(f"  [{suite_name}] Item {i+1}/{len(items)} | Current Acc: {curr_acc:.1f}%")
                
        suite_acc = (correct / len(items)) * 100.0
        all_results[suite_name] = {
            "accuracy": suite_acc,
            "correct": correct,
            "total": len(items),
            "records": suite_records
        }
        total_correct += correct
        total_items += len(items)
        print(f">>> Suite {suite_name} Accuracy: {suite_acc:.1f}% ({correct}/{len(items)})")
        
    elapsed = time.time() - start_time
    total_acc = (total_correct / total_items) * 100.0
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_accuracy": total_acc,
        "total_correct": total_correct,
        "total_items": total_items,
        "elapsed_sec": round(elapsed, 1),
        "suites": all_results
    }
    
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("\n================================================================================")
    print(f"BENCHMARK COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL UNBIASED SPIDER-WEB ACCURACY: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    evaluate_unbiased_benchmark()
