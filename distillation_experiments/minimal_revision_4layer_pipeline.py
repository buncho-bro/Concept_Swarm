"""
Hierarchical Minimal Revision Belief Pipeline (4 Layers, 5 Epochs)
(~+~→~ 式の論理解析と最小修正探索エンジン)
========================================================================
設計仕様:
1. 【~+~→~ 式の論理解析 (Logical Coherence Analysis)】:
   A + B -> C の合成に対し、
   - 順方向適合度 S(A+B, C)
   - 逆方向整合度 S_back(C -> A, B) (非対称性の検知)
   - 内部反発ペナルティ Conflict(A, B, C)
   を総合評価。
2. 【最小修正の探索 (Minimal Revision / Minimal Repair)】:
   破綻や不整合、反発が検知された場合、全否定や除外（IF文）を行わず、
   元の概念 C からの意味的変化量 ||v_C' - v_C|| を最小に抑えつつ
   論理的健全度を最大化する「最小修正先 C*」を自律探索。
   (C -> C* への最小修正: AGM信念修正の幾何的実現)
3. 【4階層展開 (4 Hierarchical Layers)】:
   Level 1: (A + B) -> C1* (最小修正)
   Level 2: (C1* + D) -> C2* (最小修正)
   Level 3: (C2* + E) -> C3* (最小修正)
   Level 4: (C3* + F) -> C4* (最小修正・4階層命題の結晶化)
4. 【5エポックの実行 (5 Epochs)】:
   GPU高速並列バッチ処理により超高速で4階層5エポックを完走。
5. 【ベンチマーク評価 (Web標準250問)】:
   直前のモデル (48.80%) との差分を厳密測定。
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
RESULTS_JSON = os.path.join(OUTPUT_DIR, "minimal_revision_results.json")
CACHE_FILE = os.path.join(OUTPUT_DIR, "web_counter_opinions_cache.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Minimal Revision Pipeline...")
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

# 概念語彙ロード
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
CONCEPT_VECS = F.normalize(embed_w[CONCEPT_IDS_TENSOR], p=2, dim=-1) # [N, D]
num_clean_concepts = CONCEPT_VECS.shape[0]
word_to_idx = {w: i for i, w in enumerate(valid_tokens)}
print(f"Loaded {num_clean_concepts} Concepts in Wilderness.")


class MinimalRevisionEngine:
    """
    ~+~->~ 式の論理解析と最小修正 (Minimal Revision) 探索エンジン
    """
    def __init__(self, revision_weight=0.35):
        self.revision_weight = revision_weight # 原点 C からの距離ペナルティ
        self.revision_log = []
        
    def analyze_and_repair(self, v_A, v_B, candidate_idx, seen_indices):
        """
        A + B -> C (candidate_idx) の論理解析を行い、
        不整合がある場合は最小修正先 C* を探索して返す。
        """
        # 1. 順方向合成ベクトル
        v_syn = F.normalize(0.5 * v_A + 0.5 * v_B, p=2, dim=-1) # [1, D]
        
        # 2. 初期候補 C の評価
        v_C = CONCEPT_VECS[candidate_idx].unsqueeze(0)
        c_word = valid_tokens[candidate_idx]
        
        # 順方向親和度
        forward_score = torch.sum(v_syn * v_C).item()
        # 逆方向整合度 (C -> A, B)
        back_score = 0.5 * (torch.sum(v_C * v_A).item() + torch.sum(v_C * v_B).item())
        # 分岐反発 (A と B の反発度)
        repulsion = 1.0 - torch.sum(v_A * v_B).item()
        
        # 論理健全度スコア
        coherence = forward_score + 0.8 * back_score - 0.3 * repulsion
        
        # 健全度基準 (閾値 0.70 以上かつ逆方向 0.35 以上なら修正不要)
        if coherence >= 0.70 and back_score >= 0.35:
            return candidate_idx, c_word, False, 0.0, coherence
            
        # --- 最小修正探索 (Minimal Revision Search) ---
        # 目的関数: max_{C'} [ Coherence(A+B -> C') - alpha * ||v_C' - v_C||^2 ]
        # ||v_C' - v_C||^2 = 2 - 2 * (v_C' . v_C)
        # したがって max_{C'} [ Coherence(A+B -> C') + 2*alpha * (v_C' . v_C) ]
        
        # 全概念に対するスコアを一括GPU計算
        # 順方向: [1, N]
        f_all = torch.matmul(v_syn, CONCEPT_VECS.T) # [1, N]
        # 逆方向: [1, N]
        b_all = 0.5 * (torch.matmul(v_A, CONCEPT_VECS.T) + torch.matmul(v_B, CONCEPT_VECS.T))
        # 原点 C との近接度 (最小変更の維持): [1, N]
        proximity_to_C = torch.matmul(v_C, CONCEPT_VECS.T)
        
        # 修正目的関数
        objective = f_all + 0.8 * b_all + (2.0 * self.revision_weight) * proximity_to_C
        
        # 既出インデックスは除外
        for idx in seen_indices:
            objective[0, idx] = -999.0
            
        best_repaired_idx = torch.argmax(objective).item()
        repaired_word = valid_tokens[best_repaired_idx]
        
        # 変化距離
        v_repaired = CONCEPT_VECS[best_repaired_idx].unsqueeze(0)
        dist = torch.norm(v_repaired - v_C).item()
        new_coherence = f_all[0, best_repaired_idx].item() + 0.8 * b_all[0, best_repaired_idx].item() - 0.3 * repulsion
        
        is_repaired = (best_repaired_idx != candidate_idx)
        if is_repaired:
            self.revision_log.append({
                "original": c_word,
                "repaired": repaired_word,
                "distance": round(dist, 4),
                "original_coherence": round(coherence, 4),
                "repaired_coherence": round(new_coherence, 4)
            })
            
        return best_repaired_idx, repaired_word, is_repaired, dist, new_coherence


def run_4layer_5epoch_pipeline(beam_size=16, num_epochs=5):
    print("\n" + "=" * 80)
    print(f"STARTING 4-LAYER HIERARCHICAL MINIMAL REVISION PIPELINE ({num_epochs} Epochs)")
    print(f"Structure: 4 Layers of ~+~->~ with Logical Analysis & Minimal Repair")
    print("=" * 80)
    
    engine = MinimalRevisionEngine(revision_weight=0.35)
    all_crystallized_chains = []
    
    t_start = time.time()
    
    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()
        print(f"\n>>> [Epoch {epoch:02d}/{num_epochs:02d}] Initializing {beam_size} Hierarchical Trees...")
        
        # シード概念ペアのサンプリング
        seed_indices = random.sample(range(num_clean_concepts), beam_size * 2)
        active_beams = []
        for b in range(beam_size):
            i_a = seed_indices[b * 2]
            i_b = seed_indices[b * 2 + 1]
            active_beams.append({
                "layer": 0,
                "history": [valid_tokens[i_a], valid_tokens[i_b]],
                "curr_indices": [i_a, i_b],
                "v_curr": F.normalize(0.5 * CONCEPT_VECS[i_a].unsqueeze(0) + 0.5 * CONCEPT_VECS[i_b].unsqueeze(0), p=2, dim=-1),
                "formula_trace": [f"({valid_tokens[i_a]} + {valid_tokens[i_b]})"]
            })
            
        # 4階層の段階的合成と最小修正
        for layer in range(1, 5):
            new_beams = []
            for beam in active_beams:
                v_A = beam["v_curr"]
                prev_w = beam["history"][-1]
                
                # 次の概念パートナー B を探索 (近傍から選抜)
                sims_B = torch.matmul(v_A, CONCEPT_VECS.T)
                for seen_i in beam["curr_indices"]:
                    sims_B[0, seen_i] = -1.0
                i_B = torch.argmax(sims_B).item()
                w_B = valid_tokens[i_B]
                v_B = CONCEPT_VECS[i_B].unsqueeze(0)
                
                # 式: A + B -> C
                v_syn = F.normalize(0.5 * v_A + 0.5 * v_B, p=2, dim=-1)
                sims_C = torch.matmul(v_syn, CONCEPT_VECS.T)
                for seen_i in beam["curr_indices"] + [i_B]:
                    sims_C[0, seen_i] = -1.0
                init_cand_idx = torch.argmax(sims_C).item()
                
                # 【論理解析と最小修正の実行】
                repaired_idx, repaired_w, was_repaired, dist, coh = engine.analyze_and_repair(
                    v_A, v_B, init_cand_idx, beam["curr_indices"] + [i_B]
                )
                
                # 階層の更新
                v_C_star = CONCEPT_VECS[repaired_idx].unsqueeze(0)
                repaired_tag = f"*[revised from {valid_tokens[init_cand_idx]}]" if was_repaired else ""
                
                formula_step = f"[{beam['formula_trace'][-1]} + {w_B} -> {repaired_w}{repaired_tag}]"
                
                new_beams.append({
                    "layer": layer,
                    "history": beam["history"] + [w_B, repaired_w],
                    "curr_indices": beam["curr_indices"] + [i_B, repaired_idx],
                    "v_curr": v_C_star,
                    "formula_trace": beam["formula_trace"] + [formula_step],
                    "was_repaired": was_repaired,
                    "coherence": coh
                })
                
            active_beams = new_beams
            
        dt_ep = time.time() - t_ep
        rep_count_ep = sum(1 for b in active_beams if any("revised" in s for s in b["formula_trace"]))
        print(f"  Completed Epoch {epoch} in {dt_ep*1000:.1f}ms | 4-Layer Chains: {len(active_beams)} | Repaired: {rep_count_ep}/{len(active_beams)}")
        
        for b in active_beams:
            all_crystallized_chains.append({
                "epoch": epoch,
                "chain": " -> ".join(b["history"]),
                "full_formula": b["formula_trace"][-1]
            })
            
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"HIERARCHICAL 4-LAYER 5-EPOCH ENGINE COMPLETED in {total_time:.2f}s!")
    print(f"Total 4-Layer Crystallized Propositions: {len(all_crystallized_chains)}")
    print(f"Total Minimal Revisions Performed: {len(engine.revision_log)}")
    print("Sample Minimal Revisions (Original -> Repaired):")
    for r in engine.revision_log[:6]:
        print(f"   * '{r['original']}' -> '{r['repaired']}' (Dist: {r['distance']:.3f}, Coh: {r['original_coherence']} -> {r['repaired_coherence']})")
    print("\nSample 4-Layer Crystallized Formulae:")
    for c in all_crystallized_chains[:4]:
        print(f"   * {c['chain']}")
    print("=" * 80)
    
    save_data = {
        "num_epochs": num_epochs,
        "layers": 4,
        "elapsed_sec": round(total_time, 2),
        "total_chains": len(all_crystallized_chains),
        "total_revisions": len(engine.revision_log),
        "sample_revisions": engine.revision_log[:20],
        "crystallized_chains": [c["chain"] for c in all_crystallized_chains]
    }
    
    out_file = os.path.join(OUTPUT_DIR, "minimal_revision_chains.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    return save_data


# ==============================================================================
# ベンチマーク評価 (Web標準250問)
# 4階層・最小修正で結晶化された論理式体系を用いた推論
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_with_minimal_revision(save_data):
    propositions = save_data["crystallized_chains"]
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()
    
    print("\n================================================================================")
    print("EVALUATING BENCHMARK WITH 4-LAYER MINIMAL REVISION BELIEF WEB (N=250)")
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
            
            # 4階層・最小修正命題からの照合
            matched_props = []
            for p in propositions:
                p_words = set(p.split(" -> "))
                overlap = len(words_set & p_words)
                if overlap > 0:
                    matched_props.append((overlap, p))
            matched_props.sort(key=lambda x: x[0], reverse=True)
            top_props = [p for _, p in matched_props[:2]]
            
            hint = ""
            if top_props:
                hint = f"Hierarchically Repaired Conceptual Chains (Minimal Revision): {'; '.join(top_props)}.\n"
                
            prompt = (
                f"<|im_start|>system\nYou are a truth and logic engine guided by hierarchically repaired conceptual chains. "
                f"Use minimal logical revision to reconcile contradictions and reject fallacies.<|im_end|>\n"
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
    print(f"MINIMAL REVISION BENCHMARK COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL ACCURACY: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    save_data = run_4layer_5epoch_pipeline(beam_size=16, num_epochs=5)
    evaluate_with_minimal_revision(save_data)
