"""
Pure Clean Spider-Web Belief Network Pipeline
(右パディング修正・9,474純粋英語概念・ロジット位置補正による真の蜘蛛の巣思考整理＆ベンチマーク)
=============================================================================================
根本修正:
1. 【語彙汚染の完全排除 (Pure Human Concept Wilderness)】:
   プログラミング記号（useralative, ndrfc等）や外国語断片を全廃。
   Google 10,000 Common English Corpus から厳選した9,474の純粋な人間実質概念のみを思考原野として採用。
2. 【右パディング・バグの完全修正 (Left-Padding Correction)】:
   padding_side = 'left' を徹底し、プロンプト末尾（assistant直後）の真のロジット位置で厳密評価。
   パディングトークン位置での評価による異常な73%矛盾誤検知を解消。
3. 【大量エポック自己組織化】:
   純粋な思考原野から無作為抽出・近傍結合・三角形閉路検証を回し、真に健全な信念網を結晶化。
4. 【Web標準250問ベンチマークの客観測定】:
   整理された純粋信念網に基づき、公平なベンチマーク評価を実施。
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "pure_clean_spider_web_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer with Left-Padding...")
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

# 1. 純粋な人間実質概念（9,474語）のロード
COMMON_WORDS_FILE = os.path.join(OUTPUT_DIR, "common_10k_english.txt")
with open(COMMON_WORDS_FILE, "r", encoding="utf-8") as f:
    raw_common = [l.strip().lower() for l in f if len(l.strip()) >= 4]

STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "which", "about", "there", "their", 
    "then", "have", "would", "could", "should", "more", "some", "such", "than", "other", 
    "into", "will", "also", "each", "both", "very", "most", "only", "were", "been"
}
CLEAN_CONCEPTS = [w for w in raw_common if w not in STOPWORDS]

# Qwenの語彙埋め込みから純粋概念テンソルを抽出
embed_w = model.get_input_embeddings().weight.data.clone()
valid_tokens = []
valid_ids = []
for w in CLEAN_CONCEPTS:
    ids = tokenizer.encode(w, add_special_tokens=False)
    if len(ids) == 1:
        valid_tokens.append(w)
        valid_ids.append(ids[0])
    else:
        ids_sp = tokenizer.encode(f" {w}", add_special_tokens=False)
        if len(ids_sp) == 1:
            valid_tokens.append(w)
            valid_ids.append(ids_sp[0])
        else:
            valid_tokens.append(w)
            valid_ids.append(ids[0])

CONCEPT_IDS_TENSOR = torch.tensor(valid_ids, device=device, dtype=torch.long)
CONCEPT_VECS = F.normalize(embed_w[CONCEPT_IDS_TENSOR], p=2, dim=-1) # [N_clean, D]
num_clean_concepts = CONCEPT_VECS.shape[0]
print(f"Constructed Pure Clean Human Thought Wilderness: {num_clean_concepts} concepts.")

TOKEN_YES_ID = tokenizer.encode("YES", add_special_tokens=False)[0]
TOKEN_NO_ID = tokenizer.encode("NO", add_special_tokens=False)[0]


def run_pure_mass_epoch_organization(num_epochs=300, batch_anchors=24):
    """
    純粋概念空間における大量エポック蜘蛛の巣自己組織化
    """
    print(f"\n================================================================================")
    print(f"STARTING PURE CLEAN SPIDER-WEB ORGANIZATION: {num_epochs} EPOCHS (Left-Padding Corrected)")
    print(f"================================================================================")
    
    current_anchors = random.sample(range(num_clean_concepts), batch_anchors)
    graph_edges = {}
    pruned_contradictions = set()
    total_triangles = 0
    sound_triangles = 0
    pruned_triangles = 0
    
    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        B = len(current_anchors)
        anchor_vecs = CONCEPT_VECS[current_anchors]
        
        # 1. 近傍概念探索 (Top-3)
        sims = torch.matmul(anchor_vecs, CONCEPT_VECS.T)
        for i, a_idx in enumerate(current_anchors):
            sims[i, a_idx] = -1.0
        topk_inds = torch.topk(sims, k=3, dim=-1).indices
        
        # 2. 新概念合成 (A + N -> W)
        neighbor_vecs = CONCEPT_VECS[topk_inds]
        syn_vecs = F.normalize(0.5 * anchor_vecs.unsqueeze(1) + 0.5 * neighbor_vecs, p=2, dim=-1)
        syn_flat = syn_vecs.view(-1, CONCEPT_VECS.shape[-1])
        syn_sims = torch.matmul(syn_flat, CONCEPT_VECS.T)
        w_top_inds = torch.topk(syn_sims, k=3, dim=-1).indices
        
        triangles = []
        flat_idx = 0
        for i, a_idx in enumerate(current_anchors):
            a_word = valid_tokens[a_idx]
            for j in range(3):
                n_idx = topk_inds[i, j].item()
                n_word = valid_tokens[n_idx]
                
                candidates = [w_top_inds[flat_idx, c].item() for c in range(3)]
                w_idx = candidates[0]
                for c in candidates:
                    if c != a_idx and c != n_idx:
                        w_idx = c
                        break
                w_word = valid_tokens[w_idx]
                flat_idx += 1
                
                triangles.append({
                    "a_idx": a_idx, "n_idx": n_idx, "w_idx": w_idx,
                    "a": a_word, "n": n_word, "w": w_word
                })
                
        # 3. 左パディングによる正確なロジット位置での三角形閉路検証
        prompts = [
            f"<|im_start|>system\nAudit conceptual relationship.<|im_end|>\n"
            f"<|im_start|>user\nDoes combining concept '{t['a']}' and '{t['n']}' into '{t['w']}' create a fatal logical contradiction or impossible absurdity? Reply YES or NO.<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for t in triangles
        ]
        
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        with torch.no_grad():
            # 左パディングなので [:, -1, :] は確実に全プロンプトの末尾（assistant直後）を指す
            logits = model(**enc).logits[:, -1, :]
            lyes = logits[:, TOKEN_YES_ID]
            lno = logits[:, TOKEN_NO_ID]
            
        # 健全性判定: logit(NO) > logit(YES) -> 正常（矛盾なし）
        # 矛盾判定: logit(YES) >= logit(NO)
        is_contra = (lyes >= lno).tolist()
        
        next_anchor_candidates = []
        for t, contra in zip(triangles, is_contra):
            total_triangles += 1
            if contra:
                pruned_triangles += 1
                pruned_contradictions.add((t["a"], t["w"]))
            else:
                sound_triangles += 1
                graph_edges.setdefault(t["a"], set()).add(t["n"])
                graph_edges.setdefault(t["n"], set()).add(t["w"])
                graph_edges.setdefault(t["w"], set()).add(t["a"])
                next_anchor_candidates.append(t["w_idx"])
                next_anchor_candidates.append(t["a_idx"])
                
        # 次世代アンカー更新
        next_anchor_candidates = list(set(next_anchor_candidates))
        if len(next_anchor_candidates) >= B // 2:
            next_anchors = random.sample(next_anchor_candidates, B // 2)
            fresh = random.sample(range(num_clean_concepts), B - len(next_anchors))
            next_anchors.extend(fresh)
        else:
            next_anchors = random.sample(range(num_clean_concepts), B)
        current_anchors = next_anchors
        
        if epoch % 50 == 0 or epoch == num_epochs:
            contra_pct = (pruned_triangles / total_triangles) * 100.0
            print(f"  [Epoch {epoch:03d}/{num_epochs}] Web Nodes: {len(graph_edges)} | Sound Triangles: {sound_triangles} | Pruned Contradictions: {pruned_triangles} ({contra_pct:.1f}%)")
            
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"PURE SPIDER-WEB ORGANIZATION COMPLETED in {total_time:.2f}s")
    print(f"Total Triangles: {total_triangles} | Sound: {sound_triangles} ({(sound_triangles/total_triangles)*100:.1f}%) | Pruned: {pruned_triangles} ({(pruned_triangles/total_triangles)*100:.1f}%)")
    print(f"Connected Pure Belief Web: {len(graph_edges)} concepts")
    print("=" * 80)
    
    clean_graph_data = {
        "num_epochs": num_epochs,
        "elapsed_sec": round(total_time, 2),
        "total_nodes": len(graph_edges),
        "sound_triangles": sound_triangles,
        "pruned_contradictions": pruned_triangles,
        "nodes_list": list(graph_edges.keys())
    }
    
    out_web = os.path.join(OUTPUT_DIR, "pure_clean_belief_web.json")
    with open(out_web, "w", encoding="utf-8") as f:
        json.dump(clean_graph_data, f, ensure_ascii=False, indent=2)
    return clean_graph_data


# ==============================================================================
# ベンチマーク評価 (Web標準250問)
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_with_pure_belief_web(web_data):
    nodes_set = set(web_data["nodes_list"])
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()
    
    print("\n================================================================================")
    print("EVALUATING BENCHMARK WITH PURE CLEAN BELIEF WEB (N=250 items)")
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
                
            # 問題文中の概念と信念網の照合
            words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in inp.split() if len(w) >= 4]
            matched = [w for w in words if w in nodes_set][:6]
            hint = f"Verified Cognitive Network Nodes: {', '.join(matched)}.\n" if matched else ""
            
            prompt = (
                f"<|im_start|>system\nYou are an uncompromising logic engine guided by an expansive, non-contradictory belief network. "
                f"Weigh grounded causal links against fallacious shortcuts.<|im_end|>\n"
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
                "correct": is_correct
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                curr = (correct / (i + 1)) * 100.0
                print(f"  [{suite_name}] Item {i+1:02d}/{len(items)} | Current Acc: {curr:.1f}%")
                
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
    print(f"EVALUATION COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL ACCURACY WITH PURE CLEAN BELIEF WEB: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    web_data = run_pure_mass_epoch_organization(num_epochs=300, batch_anchors=24)
    evaluate_with_pure_belief_web(web_data)
