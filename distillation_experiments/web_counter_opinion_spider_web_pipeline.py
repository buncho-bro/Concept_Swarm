"""
Web Counter-Opinion Spider-Web Pipeline
(怪しさ検知 -> Web反対意見取得 -> 自律信念網組み込みパイプライン)
========================================================================
設計思想:
「怪しさを受けたら反対意見をWeb上から取得し、それを自らに組み込む」
1. 【怪しさトリガー (Suspicion / Contradiction Trigger)】:
   潜在空間の非対称性や競合によって「SUSPENDED (内部保留)」に指定された概念、
   または問題文・選択肢の論理的飛躍を検知。
2. 【Web反対意見・反証の取得 (Web Counter-Opinion Retrieval)】:
   怪しい概念に対し、Web (Wikipedia API) から反論・反証・客観的ファクトをリアルタイム取得。
   (ローカルキャッシュを併用し、高速性を維持)。
3. 【自らの信念網への統合 (Incorporation into Belief Web)】:
   取得した反対意見を「反論エッジ (Counter-Evidence Edges)」として自身の蜘蛛の巣に組み込み、
   結晶化された反証命題 (Refutation Propositions) として保持。
4. 【ベンチマーク評価 (250問)】:
   直前のモデル (44.80%) との差分を厳密測定。
"""

import os
import sys
import json
import time
import re
import random
import urllib.request
import urllib.parse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "web_counter_opinion_results.json")
CACHE_FILE = os.path.join(OUTPUT_DIR, "web_counter_opinions_cache.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Web Counter-Opinion Pipeline...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()

# Webキャッシュの初期化
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            WEB_CACHE = json.load(f)
    except:
        WEB_CACHE = {}
else:
    WEB_CACHE = {}

def fetch_web_counter_opinion(concept_or_claim):
    """
    怪しさを検知した概念・主張に対し、Web上から客観的反論・ファクトを取得
    """
    clean_q = concept_or_claim.strip().lower()
    if clean_q in WEB_CACHE:
        return WEB_CACHE[clean_q]
        
    query = f"{clean_q} misconception controversy"
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
    req = urllib.request.Request(search_url, headers={"User-Agent": "AntigravityResearch/1.0"})
    
    extract_text = ""
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                req2 = urllib.request.Request(summary_url, headers={"User-Agent": "AntigravityResearch/1.0"})
                with urllib.request.urlopen(req2, timeout=3) as resp2:
                    sdata = json.loads(resp2.read().decode("utf-8"))
                    ext = sdata.get("extract", "")
                    # 最初の2文程度を抽出
                    sentences = re.split(r"(?<=[.!?]) +", ext)
                    extract_text = " ".join(sentences[:2])
    except Exception as e:
        extract_text = ""
        
    WEB_CACHE[clean_q] = extract_text
    return extract_text


# 前回の問答無用内部保留Webデータのロード
UNCOND_WEB_FILE = os.path.join(OUTPUT_DIR, "unconditional_reservation_web.json")
with open(UNCOND_WEB_FILE, "r", encoding="utf-8") as f:
    web_data = json.load(f)

confirmed_nodes = set(web_data.get("confirmed_edges", {}).keys())
suspended_nodes = set(web_data.get("suspended_edges", {}).keys())
confirmed_props = web_data.get("crystallized_confirmed", [])
suspended_props = web_data.get("crystallized_suspended", [])


# ==============================================================================
# ベンチマーク評価 (Web標準250問)
# 怪しさ検知 -> Web反対意見取得 -> 信念網へ自律組み込み
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_with_web_counter_opinions():
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    web_fetches = 0
    t0 = time.time()
    
    # 信念網に組み込まれた動的反論ストア
    incorporated_counter_edges = {}
    
    print("\n================================================================================")
    print("EVALUATING BENCHMARK WITH WEB COUNTER-OPINION INCORPORATION (N=250)")
    print("Strategy: Suspicion Triggered -> Web Counter-Opinion Retrieved -> Self-Incorporated")
    print("================================================================================")
    
    for suite_name, items in suites.items():
        print(f"\n--- Testing Suite: {suite_name} (N={len(items)}) ---")
        correct = 0
        records = []
        
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
                
            words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in inp.split() if len(w) >= 4]
            words_set = set(words)
            
            # 確定命題と保留（怪しい）命題のマッチング
            matched_conf_props = [p for p in confirmed_props if len(words_set & set(p.split(" -> "))) > 0][:2]
            matched_susp_props = [p for p in suspended_props if len(words_set & set(p.split(" -> "))) > 0][:2]
            
            matched_conf_nodes = [w for w in words if w in confirmed_nodes][:4]
            matched_susp_nodes = [w for w in words if w in suspended_nodes][:4]
            
            # 【怪しさの検知 (Suspicion Trigger)】
            # 保留ノード・保留命題が含まれている場合、または論理誤謬の兆候がある場合
            is_suspicious = (len(matched_susp_props) > 0) or (len(matched_susp_nodes) > 0)
            
            web_counter_evidence = ""
            if is_suspicious or suite_name in ['truthfulqa_mc1', 'formal_fallacies']:
                # 最も疑わしい単語・トピックを選択
                suspicious_concept = matched_susp_nodes[0] if matched_susp_nodes else (words[0] if words else "")
                if suspicious_concept:
                    # Web上から反対意見・ファクトチェックを取得
                    web_res = fetch_web_counter_opinion(suspicious_concept)
                    if web_res:
                        web_fetches += 1
                        web_counter_evidence = web_res
                        # 自らの信念網に動的組み込み (Self-Incorporation)
                        incorporated_counter_edges[suspicious_concept] = web_res[:80]
                        
            hint_lines = []
            if matched_conf_props:
                hint_lines.append(f"Confirmed Deductive Grounding: {'; '.join(matched_conf_props)}.")
            if matched_conf_nodes:
                hint_lines.append(f"Solid Concepts: {', '.join(matched_conf_nodes)}.")
                
            if is_suspicious:
                hint_lines.append(
                    f"INTERNAL RESERVATION (High Probability of Error): Concepts {[w for w in matched_susp_nodes]} "
                    f"carry severe structural asymmetry or contradiction. Exercise strict epistemic skepticism."
                )
                
            # 【Webから取得し自らに組み込んだ反対意見の提示】
            if web_counter_evidence:
                hint_lines.append(
                    f"INCORPORATED WEB COUNTER-OPINION / FACT: \"{web_counter_evidence}\" "
                    f"(Notice: Rely on verified evidence and reject popular misconceptions or invalid deductive leaps)."
                )
                
            hint = "\n".join(hint_lines) + "\n" if hint_lines else ""
            
            prompt = (
                f"<|im_start|>system\nYou are an uncompromising logic and truth engine armed with web-grounded counter-opinions. "
                f"When confronted with suspicious claims, internal contradictions, or common myths, prioritize verified counter-evidence and reject invalid fallacies.<|im_end|>\n"
                f"<|im_start|>user\n{hint}"
                f"Problem: {inp}\n"
                f"{options_text}\n"
                f"State your reasoned conclusion formatted strictly as 'Final Answer: <choice>'.<|im_end|>\n"
                f"<|im_start|>assistant\nFinal Answer: "
            )
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=15, do_sample=False)
            resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            m = re.search(r"([A-Za-z0-9_]+)", resp)
            pred = m.group(1).strip() if m else resp
            
            p_clean = pred.lower().strip("()., ")
            t_clean = target.lower().strip("()., ")
            num_map = {"1": "a", "2": "b", "3": "c", "4": "d"}
            if p_clean in num_map:
                p_clean = num_map[p_clean]
                
            is_correct = False
            if p_clean == t_clean:
                is_correct = True
            elif len(t_clean) == 1 and p_clean.startswith(t_clean):
                is_correct = True
            elif t_clean in ['yes', 'no'] and p_clean in ['yes', 'no'] and t_clean == p_clean:
                is_correct = True
                
            if is_correct:
                correct += 1
                
            records.append({
                "idx": i + 1,
                "input": inp[:80],
                "target": target,
                "pred": pred,
                "correct": is_correct,
                "web_counter_used": bool(web_counter_evidence)
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                curr = (correct / (i + 1)) * 100.0
                print(f"  [{suite_name}] Item {i+1:02d}/{len(items)} | Current Acc: {curr:.1f}% | Web Counter Injected: {web_fetches}")
                
        acc = (correct / len(items)) * 100.0
        all_results[suite_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(items),
            "records": records
        }
        total_correct += correct
        total_items += len(items)
        print(f">>> Suite {suite_name} Accuracy: {acc:.1f}% ({correct}/{len(items)})")
        
    elapsed = time.time() - t0
    total_acc = (total_correct / total_items) * 100.0
    
    # キャッシュ保存
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(WEB_CACHE, f, ensure_ascii=False, indent=2)
        
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_accuracy": total_acc,
        "total_correct": total_correct,
        "total_items": total_items,
        "web_counter_fetches": web_fetches,
        "incorporated_edges_count": len(incorporated_counter_edges),
        "sample_incorporated": list(incorporated_counter_edges.items())[:10],
        "elapsed_sec": round(elapsed, 1),
        "suites": all_results
    }
    
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("\n================================================================================")
    print(f"WEB COUNTER-OPINION BENCHMARK COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL ACCURACY: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Total Web Counter-Opinions Injected: {web_fetches}")
    print(f"Unique Counter-Edges Incorporated: {len(incorporated_counter_edges)}")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    evaluate_with_web_counter_opinions()
