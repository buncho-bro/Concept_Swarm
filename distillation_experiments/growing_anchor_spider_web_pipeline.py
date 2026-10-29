"""
Growing Compound Anchor Spider-Web Belief Network Pipeline
(累積的アンカー拡大・ビーム幅マネジメント・GPU高速並列パイプライン)
========================================================================
計算量マネジメントの徹底:
1. 【指数爆発の抑制 (Fixed Beam Budget Management)】:
   分岐係数による指数爆発 O(b^k) を防ぐため、各世代のアクティブな成長アンカー数を
   固定予算（M=16 または 24本）に厳格にクリッピング。
   最も整合性の高い健全な成長鎖（Compound Thought Chains）のみが次世代へ進む。
2. 【最大成長深度のキャッピング (Max Chain Length Cap)】:
   単語が累積されて一定の長さ（K=4〜5単語：1つの完全な命題・意味文）に達した鎖は、
   「検証済み結晶化信念（Crystallized Proposition）」として信念網に確定・保存され、
   スロットを解放して新たな無作為シードの探索へリセットする。
3. 【増分埋め込み高速化 (Incremental Embedding Updates)】:
   累積アンカーの潜在表現を GPU 上で差分平均ベクトルとして高速更新 (O(D))。
4. 【左パディング・バッチ評価 (Left-Padded Batched Evaluation)】:
   成長した文脈プロンプトを一括バッチ（32本）で左パディング評価し、
   長文になっても1ステップあたり数十ミリ秒の超高速性を維持。
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
RESULTS_JSON = os.path.join(OUTPUT_DIR, "growing_anchor_spider_web_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Growing Anchor Pipeline...")
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

# 9,474純粋概念のロード
COMMON_WORDS_FILE = os.path.join(OUTPUT_DIR, "common_10k_english.txt")
with open(COMMON_WORDS_FILE, "r", encoding="utf-8") as f:
    raw_common = [l.strip().lower() for l in f if len(l.strip()) >= 4]

STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "which", "about", "there", "their", 
    "then", "have", "would", "could", "should", "more", "some", "such", "than", "other", 
    "into", "will", "also", "each", "both", "very", "most", "only", "were", "been"
}
CLEAN_CONCEPTS = [w for w in raw_common if w not in STOPWORDS]

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
print(f"Loaded {num_clean_concepts} Pure Concepts in Thought Wilderness.")

TOKEN_YES_ID = tokenizer.encode("YES", add_special_tokens=False)[0]
TOKEN_NO_ID = tokenizer.encode("NO", add_special_tokens=False)[0]


class GrowingAnchorSpiderWebOrganizer:
    """
    計算量を厳格に管理しながら、アンカーに徐々に単語を累積・拡大させていくエンジン
    """
    def __init__(self, beam_budget=16, max_chain_length=4, neighbors_per_step=2):
        self.beam_budget = beam_budget # 同時並行で成長させるアクティブ鎖の最大本数
        self.max_chain_length = max_chain_length # 鎖の最大語数 (これに達したら結晶化・アーカイブ)
        self.neighbors_per_step = neighbors_per_step
        self.active_chains = [] # 各要素: {"tokens": [w1, w2, ...], "vec": tensor, "id_indices": [idx1, ...]}
        self.crystallized_propositions = [] # 完全に結晶化した思考鎖
        self.graph_edges = {}
        self.total_expansions_audited = 0
        self.pruned_count = 0
        self.sound_count = 0
        
    def seed_fresh_chains(self, needed_count):
        """新しい単語から新規の種を注入"""
        fresh_indices = random.sample(range(num_clean_concepts), needed_count)
        for idx in fresh_indices:
            w = valid_tokens[idx]
            v = CONCEPT_VECS[idx].unsqueeze(0) # [1, D]
            self.active_chains.append({
                "tokens": [w],
                "vec": v,
                "id_indices": [idx]
            })
            
    def step_growth_epoch(self):
        """
        1世代分の成長ステップ:
        各アクティブなアンカー鎖に対し、思考原野から近傍を探索し、新単語を累積結合
        """
        # 不足分の種を補充して常に beam_budget 本のアクティブ鎖を維持
        if len(self.active_chains) < self.beam_budget:
            self.seed_fresh_chains(self.beam_budget - len(self.active_chains))
            
        current_chains = self.active_chains[:self.beam_budget]
        
        # 1. 各累積アンカーの現在ベクトルを一括スタック [M, D]
        chain_vecs = torch.cat([c["vec"] for c in current_chains], dim=0)
        
        # 2. 思考原野から近傍概念を並列探索 [M, num_clean_concepts]
        sims = torch.matmul(chain_vecs, CONCEPT_VECS.T)
        for i, c in enumerate(current_chains):
            for seen_idx in c["id_indices"]:
                sims[i, seen_idx] = -1.0 # 既出の単語は除外
                
        topk_inds = torch.topk(sims, k=self.neighbors_per_step, dim=-1).indices # [M, neighbors]
        
        # 3. 候補展開の構築 (アンカー鎖 + 近傍単語 -> 新単語)
        candidate_expansions = []
        for i, c in enumerate(current_chains):
            chain_str = " ".join(c["tokens"])
            for j in range(self.neighbors_per_step):
                n_idx = topk_inds[i, j].item()
                n_word = valid_tokens[n_idx]
                
                # 新単語の合成 (現在の鎖ベクトルと近傍ベクトルの重心)
                syn_v = F.normalize(0.6 * c["vec"] + 0.4 * CONCEPT_VECS[n_idx].unsqueeze(0), p=2, dim=-1)
                syn_sims = torch.matmul(syn_v, CONCEPT_VECS.T)
                for seen_idx in c["id_indices"] + [n_idx]:
                    syn_sims[0, seen_idx] = -1.0
                w_idx = torch.argmax(syn_sims).item()
                w_word = valid_tokens[w_idx]
                
                candidate_expansions.append({
                    "parent_chain": c,
                    "neighbor": n_word,
                    "neighbor_idx": n_idx,
                    "new_word": w_word,
                    "new_word_idx": w_idx,
                    "accumulated_str": f"{chain_str} {w_word}",
                    "new_tokens": c["tokens"] + [w_word],
                    "new_indices": c["id_indices"] + [w_idx],
                    "new_vec": syn_v
                })
                
        # 4. バッチ左パディングによる累積文脈の整合性監査 (計算量マネジメント: 一括フォワード)
        prompts = [
            f"<|im_start|>system\nYou are a rigorous logical consistency auditor checking a growing chain of thought.<|im_end|>\n"
            f"<|im_start|>user\nChain of thought so far: \"{' '.join(cand['parent_chain']['tokens'])}\"\n"
            f"Next connected concept: \"{cand['new_word']}\" (via context '{cand['neighbor']}').\n"
            f"Does appending \"{cand['new_word']}\" introduce a fatal logical contradiction, factual absurdity, or fallacy to this chain? Reply YES or NO.<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for cand in candidate_expansions
        ]
        
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
            lyes = logits[:, TOKEN_YES_ID]
            lno = logits[:, TOKEN_NO_ID]
            
        # 判定: NO (矛盾なし) の確信度
        margins = (lno - lyes).float().tolist()
        
        # 5. ビーム幅マネジメント: 健全な候補から上位 M 本を選抜
        surviving_chains = []
        for cand, margin in zip(candidate_expansions, margins):
            self.total_expansions_audited += 1
            if margin >= 0.0: # 健全 (矛盾なし)
                self.sound_count += 1
                cand["score"] = margin
                surviving_chains.append(cand)
                
                # グラフにエッジを記録
                prev_w = cand["parent_chain"]["tokens"][-1]
                curr_w = cand["new_word"]
                self.graph_edges.setdefault(prev_w, set()).add(curr_w)
                self.graph_edges.setdefault(curr_w, set()).add(prev_w)
            else:
                self.pruned_count += 1
                
        # スコア順にソートし、最大 beam_budget 本に絞り込み（指数爆発を物理的に遮断）
        surviving_chains.sort(key=lambda x: x["score"], reverse=True)
        selected = surviving_chains[:self.beam_budget]
        
        # 6. 最大長に達した鎖の結晶化・アーカイブ
        next_active = []
        for cand in selected:
            if len(cand["new_tokens"]) >= self.max_chain_length:
                # 完全に成長した命題として保存
                prop_str = " -> ".join(cand["new_tokens"])
                self.crystallized_propositions.append(prop_str)
            else:
                next_active.append({
                    "tokens": cand["new_tokens"],
                    "vec": cand["new_vec"],
                    "id_indices": cand["new_indices"]
                })
                
        self.active_chains = next_active


def run_growing_anchor_training(num_epochs=150, beam_budget=16, max_chain_length=4):
    print(f"\n================================================================================")
    print(f"STARTING GROWING ANCHOR SPIDER-WEB TRAINING ({num_epochs} Epochs)")
    print(f"Strict Complexity Management: Beam Budget={beam_budget}, Max Length={max_chain_length}")
    print(f"================================================================================")
    
    organizer = GrowingAnchorSpiderWebOrganizer(
        beam_budget=beam_budget,
        max_chain_length=max_chain_length,
        neighbors_per_step=2
    )
    
    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        organizer.step_growth_epoch()
        dt = time.time() - t0
        
        if epoch % 25 == 0 or epoch == num_epochs:
            print(f"  [Epoch {epoch:03d}/{num_epochs}] dt: {dt*1000:.1f}ms | Active Chains: {len(organizer.active_chains)} | "
                  f"Crystallized Propositions: {len(organizer.crystallized_propositions)} | "
                  f"Web Nodes: {len(organizer.graph_edges)} | Sound: {organizer.sound_count} | Pruned: {organizer.pruned_count}")
            
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"GROWING ANCHOR TRAINING COMPLETED in {total_time:.2f}s!")
    print(f"Total Propositions Crystallized (Length 4-5): {len(organizer.crystallized_propositions)}")
    print(f"Sample Crystallized Chains:")
    for prop in organizer.crystallized_propositions[:8]:
        print(f"   * {prop}")
    print(f"Connected Web Nodes: {len(organizer.graph_edges)} concepts")
    print("=" * 80)
    
    save_data = {
        "num_epochs": num_epochs,
        "elapsed_sec": round(total_time, 2),
        "total_nodes": len(organizer.graph_edges),
        "crystallized_propositions_count": len(organizer.crystallized_propositions),
        "crystallized_propositions": organizer.crystallized_propositions,
        "sample_crystallized": organizer.crystallized_propositions[:50],
        "all_nodes": list(organizer.graph_edges.keys())
    }
    
    out_file = os.path.join(OUTPUT_DIR, "growing_anchor_belief_web.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    return organizer, save_data


# ==============================================================================
# ベンチマーク評価 (Web標準250問)
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_with_growing_web(web_data):
    nodes_set = set(web_data.get("all_nodes", []))
    propositions = web_data.get("crystallized_propositions", web_data.get("sample_crystallized", []))
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()
    
    print("\n================================================================================")
    print("EVALUATING BENCHMARK WITH GROWING COMPOUND ANCHOR BELIEF WEB (N=250)")
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
            # 累積された命題から問題と合致する思考鎖を検索
            scored_props = []
            for p in propositions:
                p_words = set(p.split(" -> "))
                overlap = len(words_set & p_words)
                if overlap > 0:
                    scored_props.append((overlap, p))
            scored_props.sort(key=lambda x: x[0], reverse=True)
            matched_props = [p for _, p in scored_props[:2]]
            
            matched_nodes = [w for w in words if w in nodes_set][:4]
            
            hint_parts = []
            if matched_props:
                hint_parts.append(f"Crystallized Conceptual Chains: {'; '.join(matched_props)}.")
            if matched_nodes:
                hint_parts.append(f"Cognitive Grounding Anchors: {', '.join(matched_nodes)}.")
            hint = "\n".join(hint_parts) + "\n" if hint_parts else ""
                
            prompt = (
                f"<|im_start|>system\nYou are an uncompromising logic engine guided by crystallized conceptual chains. "
                f"Distinguish valid deductions from affirming the consequent, and facts from myths.<|im_end|>\n"
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
    print(f"BENCHMARK COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL ACCURACY: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    organizer, save_data = run_growing_anchor_training(num_epochs=150, beam_budget=16, max_chain_length=4)
    evaluate_with_growing_web(save_data)
