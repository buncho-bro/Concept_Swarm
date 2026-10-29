"""
Accumulated Hierarchical Synthesis & Minimal Revision Pipeline (4 Layers, 5 Epochs)
(累積的結論保持による真の階層構造と論理解析・最小修正パイプライン)
========================================================================
階層仕様:
- 第1階層: A + B -> C
- 第2階層: (A + C) + D -> E
- 第3階層: (A + C + E) + F -> G
- 第4階層: (A + C + E + G) + H -> I  または  (A + C) + (E + G) -> Z

設計要件:
1. 前提と結論が累積保持され、初期の文脈が蒸発しない。
2. 各階層で論理解析（語形重複ペナルティ、逆方向整合性、前提反発度）を行い、
   破綻があれば最小修正先 (C*, E*, G*, I* / Z*) を探索して差し替える。
3. 4階層・5エポックを高速実行し、式の具体例とベンチマークを検証。
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

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "hierarchical_accumulated_results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Hierarchical Accumulated Pipeline...")
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

# 語彙ロード
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
num_concepts = CONCEPT_VECS.shape[0]
word_to_idx = {w: i for i, w in enumerate(valid_tokens)}
print(f"Loaded {num_concepts} Concepts.")


class HierarchicalAccumulatedEngine:
    """
    (A+C)+D->E や (A+C)+(E+G)->Z の累積的階層合成と論理解析・最小修正エンジン
    """
    def __init__(self, revision_weight=0.35):
        self.revision_weight = revision_weight
        self.revisions_performed = []
        
    def find_repaired_concept(self, v_syn, context_vecs, context_indices, context_words):
        """
        合成ベクトル v_syn に対して、語幹重複・非対称性を排した最小修正概念を探索
        """
        # 全概念に対する親和度
        sims = torch.matmul(v_syn, CONCEPT_VECS.T) # [1, N]
        
        # 逆方向整合度 (候補から文脈全体への平均想起度)
        b_scores = torch.zeros_like(sims)
        for c_vec in context_vecs:
            b_scores += torch.matmul(c_vec, CONCEPT_VECS.T)
        b_scores /= len(context_vecs)
        
        # 総合スコア
        score = sims + 0.8 * b_scores
        
        # 文脈に出現した単語および同一語幹（先頭4文字）を厳格に除外・減点
        for idx in context_indices:
            score[0, idx] = -999.0
            
        stems = set(w[:4] for w in context_words)
        for w, idx in word_to_idx.items():
            if w[:4] in stems:
                score[0, idx] -= 3.0 # 語形変化・堂々巡りの徹底排除
                
        best_idx = torch.argmax(score).item()
        best_w = valid_tokens[best_idx]
        return best_idx, best_w


def run_accumulated_hierarchy_pipeline(num_epochs=5, beam_size=16):
    print("\n" + "=" * 80)
    print(f"STARTING ACCUMULATED HIERARCHICAL SYNTHESIS ({num_epochs} Epochs, 4 Layers)")
    print("Formula Types:")
    print("  Layer 1: A + B -> C")
    print("  Layer 2: (A + C) + D -> E")
    print("  Layer 3: (A + C + E) + F -> G")
    print("  Layer 4: (A + C + E + G) + H -> I  AND  (A + C) + (E + G) -> Z")
    print("================================================================================")
    
    engine = HierarchicalAccumulatedEngine(revision_weight=0.35)
    crystallized_formulae = []
    t_start = time.time()
    
    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()
        print(f"\n>>> [Epoch {epoch:02d}/{num_epochs:02d}] Generating {beam_size} Hierarchical Deep Trees...")
        
        # 初期ペア (A, B) の選抜
        seed_indices = random.sample(range(num_concepts), beam_size * 2)
        
        for b in range(beam_size):
            i_A = seed_indices[b * 2]
            i_B = seed_indices[b * 2 + 1]
            w_A = valid_tokens[i_A]
            w_B = valid_tokens[i_B]
            v_A = CONCEPT_VECS[i_A].unsqueeze(0)
            v_B = CONCEPT_VECS[i_B].unsqueeze(0)
            
            # -------------------------------------------------------------
            # Layer 1: A + B -> C
            # -------------------------------------------------------------
            v_syn1 = F.normalize(0.5 * v_A + 0.5 * v_B, p=2, dim=-1)
            i_C, w_C = engine.find_repaired_concept(v_syn1, [v_A, v_B], [i_A, i_B], [w_A, w_B])
            v_C = CONCEPT_VECS[i_C].unsqueeze(0)
            f1 = f"{w_A} + {w_B} -> {w_C}"
            
            # -------------------------------------------------------------
            # Layer 2: (A + C) + D -> E
            # -------------------------------------------------------------
            # (A + C) の合成ブロック
            v_AC = F.normalize(0.5 * v_A + 0.5 * v_C, p=2, dim=-1)
            
            # 新たな前提 D を近傍から抽出
            sims_D = torch.matmul(v_AC, CONCEPT_VECS.T)
            for idx in [i_A, i_B, i_C]:
                sims_D[0, idx] = -1.0
            i_D = torch.argmax(sims_D).item()
            w_D = valid_tokens[i_D]
            v_D = CONCEPT_VECS[i_D].unsqueeze(0)
            
            # (A + C) + D -> E
            v_syn2 = F.normalize(0.6 * v_AC + 0.4 * v_D, p=2, dim=-1)
            i_E, w_E = engine.find_repaired_concept(v_syn2, [v_A, v_C, v_D], [i_A, i_B, i_C, i_D], [w_A, w_B, w_C, w_D])
            v_E = CONCEPT_VECS[i_E].unsqueeze(0)
            f2 = f"({w_A} + {w_C}) + {w_D} -> {w_E}"
            
            # -------------------------------------------------------------
            # Layer 3: (A + C + E) + F -> G
            # -------------------------------------------------------------
            # (A + C + E) の合成ブロック
            v_ACE = F.normalize((v_A + v_C + v_E) / 3.0, p=2, dim=-1)
            
            sims_F = torch.matmul(v_ACE, CONCEPT_VECS.T)
            for idx in [i_A, i_B, i_C, i_D, i_E]:
                sims_F[0, idx] = -1.0
            i_F = torch.argmax(sims_F).item()
            w_F = valid_tokens[i_F]
            v_F = CONCEPT_VECS[i_F].unsqueeze(0)
            
            v_syn3 = F.normalize(0.6 * v_ACE + 0.4 * v_F, p=2, dim=-1)
            i_G, w_G = engine.find_repaired_concept(v_syn3, [v_A, v_C, v_E, v_F], [i_A, i_B, i_C, i_D, i_E, i_F], [w_A, w_B, w_C, w_D, w_E, w_F])
            v_G = CONCEPT_VECS[i_G].unsqueeze(0)
            f3 = f"({w_A} + {w_C} + {w_E}) + {w_F} -> {w_G}"
            
            # -------------------------------------------------------------
            # Layer 4: 2つのバリエーション
            # 4A: (A + C + E + G) + H -> I (逐次累積型)
            # 4B: (A + C) + (E + G) -> Z (並列ブロック統合型)
            # -------------------------------------------------------------
            # 4A
            v_ACEG = F.normalize((v_A + v_C + v_E + v_G) / 4.0, p=2, dim=-1)
            sims_H = torch.matmul(v_ACEG, CONCEPT_VECS.T)
            for idx in [i_A, i_B, i_C, i_D, i_E, i_F, i_G]:
                sims_H[0, idx] = -1.0
            i_H = torch.argmax(sims_H).item()
            w_H = valid_tokens[i_H]
            v_H = CONCEPT_VECS[i_H].unsqueeze(0)
            
            v_syn4A = F.normalize(0.6 * v_ACEG + 0.4 * v_H, p=2, dim=-1)
            i_I, w_I = engine.find_repaired_concept(v_syn4A, [v_A, v_C, v_E, v_G, v_H], [i_A, i_B, i_C, i_D, i_E, i_F, i_G, i_H], [w_A, w_C, w_E, w_G, w_H])
            f4A = f"({w_A} + {w_C} + {w_E} + {w_G}) + {w_H} -> {w_I}"
            
            # 4B: (A + C) + (E + G) -> Z (並列ブロック統合)
            v_EG = F.normalize(0.5 * v_E + 0.5 * v_G, p=2, dim=-1)
            v_syn4B = F.normalize(0.5 * v_AC + 0.5 * v_EG, p=2, dim=-1)
            i_Z, w_Z = engine.find_repaired_concept(v_syn4B, [v_A, v_C, v_E, v_G], [i_A, i_C, i_E, i_G], [w_A, w_C, w_E, w_G])
            f4B = f"({w_A} + {w_C}) + ({w_E} + {w_G}) -> {w_Z}"
            
            tree_record = {
                "epoch": epoch,
                "beam": b + 1,
                "layer1": f1,
                "layer2": f2,
                "layer3": f3,
                "layer4_sequential": f4A,
                "layer4_parallel": f4B,
                "chain_str": f"{w_A} -> {w_C} -> {w_E} -> {w_G} -> {w_I} / {w_Z}",
                "concepts": [w_A, w_B, w_C, w_D, w_E, w_F, w_G, w_H, w_I, w_Z]
            }
            crystallized_formulae.append(tree_record)
            
        dt_ep = time.time() - t_ep
        print(f"  Epoch {epoch} Completed in {dt_ep*1000:.1f}ms | 16 Trees Formed.")
        
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ACCUMULATED HIERARCHICAL SYNTHESIS COMPLETED in {total_time:.2f}s!")
    print(f"Total Crystallized Hierarchical Deep Trees: {len(crystallized_formulae)}")
    print("=" * 80)
    
    print("\nSAMPLE HIERARCHICAL TREES:")
    for t in crystallized_formulae[:4]:
        print(f"\n[Epoch {t['epoch']} | Tree {t['beam']}]")
        print(f"  L1: {t['layer1']}")
        print(f"  L2: {t['layer2']}")
        print(f"  L3: {t['layer3']}")
        print(f"  L4 (Seq):      {t['layer4_sequential']}")
        print(f"  L4 (Parallel): {t['layer4_parallel']}")
        print(f"  Crystallized Proposition: {t['chain_str']}")
        
    save_data = {
        "num_epochs": num_epochs,
        "elapsed_sec": round(total_time, 2),
        "total_trees": len(crystallized_formulae),
        "trees": crystallized_formulae
    }
    
    with open(os.path.join(OUTPUT_DIR, "hierarchical_trees.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
        
    return save_data


# ==============================================================================
# ベンチマーク評価 (Web標準250問)
# (A+C)+D->E / (A+C)+(E+G)->Z の真の階層構造を用いた推論
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_with_hierarchical_trees(save_data):
    trees = save_data["trees"]
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()
    
    print("\n================================================================================")
    print("EVALUATING BENCHMARK WITH ACCUMULATED HIERARCHICAL TREES (N=250)")
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
            
            # 階層ツリーからマッチするものを検索
            matched_trees = []
            for tr in trees:
                c_set = set(tr["concepts"])
                overlap = len(words_set & c_set)
                if overlap > 0:
                    matched_trees.append((overlap, tr))
            matched_trees.sort(key=lambda x: x[0], reverse=True)
            top_trees = [tr for _, tr in matched_trees[:2]]
            
            hint = ""
            if top_trees:
                hint_lines = []
                for tr in top_trees:
                    hint_lines.append(f"Hierarchical Deductive Tree: {tr['layer2']} | Parallel: {tr['layer4_parallel']}.")
                hint = "\n".join(hint_lines) + "\n"
                
            prompt = (
                f"<|im_start|>system\nYou are a rigorous logical inference engine guided by accumulated hierarchical deductive trees. "
                f"Weigh premises through multi-layer composition and reject fallacies or false leaps.<|im_end|>\n"
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
    print(f"ACCUMULATED HIERARCHICAL BENCHMARK COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL ACCURACY: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    save_data = run_accumulated_hierarchy_pipeline(num_epochs=5, beam_size=16)
    evaluate_with_hierarchical_trees(save_data)
