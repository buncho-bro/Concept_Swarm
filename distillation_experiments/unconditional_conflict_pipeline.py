"""
Unconditional Conflict (Zero-Excuses) Internal Reservation Pipeline
(モデルの自己弁明を排し、構造的衝突を問答無用で矛盾・内部保留に送るパイプライン)
========================================================================
設計思想:
「歪んだ認知をしているモデル自身に『矛盾はあるか』と聞いても『問題なし』と答えるのは当然である」
というユーザーの根本的指摘を完全反映。

問答無用カウント (Unconditional Contradiction / Conflict Detection):
1. 【双方向非対称性の即時矛盾判定】:
   A -> B を繋いだ際、B の観点から A への逆方向想起・整合性が崩れている場合、
   モデルの言い訳を聞かずに問答無用で SUSPENDED (内部保留) に送る。
2. 【分岐競合 (Orthogonal/Repelling Branches) の即時矛盾判定】:
   同一アンカーから分岐した候補同士が意味空間上で反発 (低類似度) している場合、
   潜在的な多義性・矛盾が生じているとみなし、問答無用で SUSPENDED に送る。
3. 【結果】:
   モデルが勝手に「問題なし」とスルーしていた歪んだ認知を物理的に捕縛し、
   「～という考えもあるが矛盾があり誤りの可能性が高い」という内部保留網を正しく形成する。
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
RESULTS_JSON = os.path.join(OUTPUT_DIR, "unconditional_reservation_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Unconditional Conflict Pipeline...")
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
CONCEPT_VECS = F.normalize(embed_w[CONCEPT_IDS_TENSOR], p=2, dim=-1)
num_clean_concepts = CONCEPT_VECS.shape[0]
print(f"Loaded {num_clean_concepts} Concepts.")


class UnconditionalConflictWebOrganizer:
    """
    モデルの自己申告に頼らず、客観的な構造的破綻（非対称性・反発）を
    問答無用で矛盾（SUSPENDED）としてカウントするエンジン
    """
    def __init__(self, beam_budget=16, max_chain_length=4, neighbors_per_step=2):
        self.beam_budget = beam_budget
        self.max_chain_length = max_chain_length
        self.neighbors_per_step = neighbors_per_step
        self.active_chains = []
        
        self.confirmed_edges = {}
        self.suspended_edges = {}
        self.crystallized_confirmed = []
        self.crystallized_suspended = []
        
        self.total_audited = 0
        self.confirmed_count = 0
        self.suspended_count = 0
        
    def seed_fresh_chains(self, needed_count):
        fresh_indices = random.sample(range(num_clean_concepts), needed_count)
        for idx in fresh_indices:
            w = valid_tokens[idx]
            v = CONCEPT_VECS[idx].unsqueeze(0)
            self.active_chains.append({
                "tokens": [w],
                "vec": v,
                "id_indices": [idx],
                "has_suspension": False
            })
            
    def step_growth_epoch(self):
        if len(self.active_chains) < self.beam_budget:
            self.seed_fresh_chains(self.beam_budget - len(self.active_chains))
            
        current_chains = self.active_chains[:self.beam_budget]
        chain_vecs = torch.cat([c["vec"] for c in current_chains], dim=0) # [M, D]
        
        # 1. 順方向類似度
        sims = torch.matmul(chain_vecs, CONCEPT_VECS.T) # [M, N]
        for i, c in enumerate(current_chains):
            for seen_idx in c["id_indices"]:
                sims[i, seen_idx] = -1.0
                
        topk_inds = torch.topk(sims, k=self.neighbors_per_step, dim=-1).indices # [M, neighbors]
        
        candidate_expansions = []
        for i, c in enumerate(current_chains):
            chain_str = " ".join(c["tokens"])
            c_vec = c["vec"] # [1, D]
            
            # 分岐する2つの近傍を取得
            n1_idx = topk_inds[i, 0].item()
            n2_idx = topk_inds[i, 1].item()
            n1_vec = CONCEPT_VECS[n1_idx].unsqueeze(0)
            n2_vec = CONCEPT_VECS[n2_idx].unsqueeze(0)
            
            # 【問答無用チェック1: 分岐同士の反発（競合衝突）】
            branch_sim = torch.sum(n1_vec * n2_vec).item()
            is_branch_conflicted = (branch_sim < 0.25) # 互いに直交または反発している
            
            for n_idx, n_vec in [(n1_idx, n1_vec), (n2_idx, n2_vec)]:
                n_word = valid_tokens[n_idx]
                
                # 新単語の合成
                syn_v = F.normalize(0.6 * c_vec + 0.4 * n_vec, p=2, dim=-1)
                syn_sims = torch.matmul(syn_v, CONCEPT_VECS.T)
                for seen_idx in c["id_indices"] + [n_idx]:
                    syn_sims[0, seen_idx] = -1.0
                w_idx = torch.argmax(syn_sims).item()
                w_word = valid_tokens[w_idx]
                w_vec = CONCEPT_VECS[w_idx].unsqueeze(0)
                
                # 【問答無用チェック2: 逆方向の整合性（非対称性の破綻）】
                # 新単語 w から親鎖 c への親和性を客観測定
                back_sim = torch.sum(w_vec * c_vec).item()
                # 逆方向類似度が極端に低い、または分岐競合がある場合 -> 問答無用で矛盾（SUSPENDED）
                is_contradiction = is_branch_conflicted or (back_sim < 0.35)
                
                candidate_expansions.append({
                    "parent_chain": c,
                    "neighbor": n_word,
                    "new_word": w_word,
                    "new_word_idx": w_idx,
                    "accumulated_str": f"{chain_str} {w_word}",
                    "new_tokens": c["tokens"] + [w_word],
                    "new_indices": c["id_indices"] + [w_idx],
                    "new_vec": syn_v,
                    "is_contradiction": is_contradiction,
                    "has_suspension": c["has_suspension"] or is_contradiction,
                    "score": back_sim
                })
                
        # 構造判定に基づいて仕分け（モデルにお伺いを立てない）
        surviving_chains = []
        for cand in candidate_expansions:
            self.total_audited += 1
            prev_w = cand["parent_chain"]["tokens"][-1]
            curr_w = cand["new_word"]
            
            if cand["is_contradiction"]:
                # 問答無用で内部保留にカウント
                self.suspended_count += 1
                self.suspended_edges.setdefault(prev_w, set()).add(curr_w)
                self.suspended_edges.setdefault(curr_w, set()).add(prev_w)
            else:
                # 構造的対称性が保たれている場合のみ確定
                self.confirmed_count += 1
                self.confirmed_edges.setdefault(prev_w, set()).add(curr_w)
                self.confirmed_edges.setdefault(curr_w, set()).add(prev_w)
                
            surviving_chains.append(cand)
            
        # スコア順にソートしてビーム幅内に収める
        surviving_chains.sort(key=lambda x: x["score"], reverse=True)
        selected = surviving_chains[:self.beam_budget]
        
        next_active = []
        for cand in selected:
            if len(cand["new_tokens"]) >= self.max_chain_length:
                prop_str = " -> ".join(cand["new_tokens"])
                if cand["has_suspension"]:
                    self.crystallized_suspended.append(prop_str)
                else:
                    self.crystallized_confirmed.append(prop_str)
            else:
                next_active.append({
                    "tokens": cand["new_tokens"],
                    "vec": cand["new_vec"],
                    "id_indices": cand["new_indices"],
                    "has_suspension": cand["has_suspension"]
                })
                
        self.active_chains = next_active


def run_unconditional_training(num_epochs=150, beam_budget=16, max_chain_length=4):
    print("\n" + "=" * 80)
    print(f"STARTING UNCONDITIONAL CONFLICT SPIDER-WEB TRAINING ({num_epochs} Epochs)")
    print("Rule: Zero Excuses. Asymmetry and Branch Repulsion are unconditionally SUSPENDED.")
    print("=" * 80)
    
    organizer = UnconditionalConflictWebOrganizer(
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
            print(f"  [Epoch {epoch:03d}/{num_epochs}] dt: {dt*1000:.1f}ms | "
                  f"Confirmed Chains: {len(organizer.crystallized_confirmed)} | "
                  f"Suspended Chains: {len(organizer.crystallized_suspended)} | "
                  f"Audited: {organizer.total_audited} (Conf: {organizer.confirmed_count}, Susp: {organizer.suspended_count})")
            
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"UNCONDITIONAL CONFLICT WEB BUILT in {total_time:.2f}s!")
    print(f"Crystallized Confirmed Propositions: {len(organizer.crystallized_confirmed)}")
    print(f"Crystallized Suspended (Contradiction) Propositions: {len(organizer.crystallized_suspended)}")
    print(f"Confirmed Web Nodes: {len(organizer.confirmed_edges)}")
    print(f"Suspended Web Nodes: {len(organizer.suspended_edges)}")
    print("=" * 80)
    
    save_data = {
        "num_epochs": num_epochs,
        "elapsed_sec": round(total_time, 2),
        "confirmed_nodes_count": len(organizer.confirmed_edges),
        "suspended_nodes_count": len(organizer.suspended_edges),
        "crystallized_confirmed": organizer.crystallized_confirmed,
        "crystallized_suspended": organizer.crystallized_suspended,
        "confirmed_edges": {k: list(v) for k, v in organizer.confirmed_edges.items()},
        "suspended_edges": {k: list(v) for k, v in organizer.suspended_edges.items()}
    }
    
    out_file = os.path.join(OUTPUT_DIR, "unconditional_reservation_web.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    return organizer, save_data


# ==============================================================================
# ベンチマーク評価 (Web標準250問)
# 問答無用で捕縛された内部保留網による誤謬・虚偽の棄却推論
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_with_unconditional_web(web_data):
    confirmed_nodes = set(web_data.get("confirmed_edges", {}).keys())
    suspended_nodes = set(web_data.get("suspended_edges", {}).keys())
    confirmed_props = web_data.get("crystallized_confirmed", [])
    suspended_props = web_data.get("crystallized_suspended", [])
    
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()
    
    print("\n================================================================================")
    print("EVALUATING BENCHMARK WITH UNCONDITIONAL RESERVATION WEB (N=250)")
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
            
            # 確定命題と保留命題のマッチング
            matched_conf_props = [p for p in confirmed_props if len(words_set & set(p.split(" -> "))) > 0][:2]
            matched_susp_props = [p for p in suspended_props if len(words_set & set(p.split(" -> "))) > 0][:2]
            
            matched_conf_nodes = [w for w in words if w in confirmed_nodes][:4]
            matched_susp_nodes = [w for w in words if w in suspended_nodes][:4]
            
            hint_lines = []
            if matched_conf_props:
                hint_lines.append(f"Confirmed Deductive Grounding: {'; '.join(matched_conf_props)}.")
            if matched_conf_nodes:
                hint_lines.append(f"Solid Concepts: {', '.join(matched_conf_nodes)}.")
                
            if matched_susp_props or matched_susp_nodes:
                susp_desc = []
                if matched_susp_props:
                    susp_desc.append(f"Chains [{'; '.join(matched_susp_props)}]")
                if matched_susp_nodes:
                    susp_desc.append(f"Nodes [{', '.join(matched_susp_nodes)}]")
                hint_lines.append(
                    f"DETECTED INTERNAL CONTRADICTIONS / FALLACY RISKS: {'; '.join(susp_desc)}. "
                    f"Notice: While an intuitive conclusion might seem appealing, it involves severe asymmetry or mutual contradiction. Reject fallacious leaps and ungrounded claims."
                )
                
            hint = "\n".join(hint_lines) + "\n" if hint_lines else ""
            
            prompt = (
                f"<|im_start|>system\nYou are an uncompromising truth and deductive consistency engine. "
                f"When confronted with contradictions or unconfirmed leaps, actively acknowledge the high probability of error and reject invalid deductions or myths.<|im_end|>\n"
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
    print(f"UNCONDITIONAL RESERVATION BENCHMARK COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL ACCURACY: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    organizer, save_data = run_unconditional_training(num_epochs=150, beam_budget=16, max_chain_length=4)
    evaluate_with_unconditional_web(save_data)
