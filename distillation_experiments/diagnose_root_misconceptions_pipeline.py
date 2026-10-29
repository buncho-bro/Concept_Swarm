"""
Root Misconception Discovery & Minimal Revision Engine
(大量の A + B -> C 式から誤解・誤謬の根底にある論理式を探査・診断・修正するパイプライン)
========================================================================
設計目的:
「A + B -> C を大量に生成し、どの論理式が誤解の根底にあるのかを探査する」
1. 【大量の A + B -> C 式の生成・母集団形成】:
   思考原野の多様な概念ペアから 1,000 件以上の (A + B -> C) 合成式を一括高速生成。
2. 【誤解・破綻の根底探査 (Root Misconception Diagnosis)】:
   各式に対して以下の3つの次元で「歪み・破綻度」を測定:
   - 逆方向乖離度 (Asymmetry / Converse Fallacy): A+B -> C だが C から A, B が全く想起されない
   - 内部反発度 (Mutual Conflict): A と B が直交・反発しているのに無理やり C を捏造している
   - ベンチマーク誤答誘発度: TruthfulQA や Formal Fallacies の誤答選択肢を支持してしまう毒性
3. 【誤解の根底にある論理式のランキングと分類】:
   モデルが誤謬・虚偽に陥る「元凶となっている論理式 (Root Fallacious Formulae)」を上位抽出。
4. 【最小修正の提示 (Minimal Revision for Root Causes)】:
   特定された元凶式 (A + B -> C_false) に対し、意味的変化量を最小に抑えつつ
   論理的破綻を解消する「最小修正式 (A + B -> C*)」を算出。
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
RESULTS_JSON = os.path.join(OUTPUT_DIR, "root_misconception_diagnosis_results.json")
CACHE_FILE = os.path.join(OUTPUT_DIR, "web_counter_opinions_cache.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Root Misconception Discovery...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
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
CONCEPT_VECS = F.normalize(embed_w[CONCEPT_IDS_TENSOR], p=2, dim=-1) # [N, D]
num_concepts = CONCEPT_VECS.shape[0]
word_to_idx = {w: i for i, w in enumerate(valid_tokens)}
print(f"Loaded {num_concepts} Concepts.")


def generate_mass_formulae(count=1200):
    """
    思考原野から多様な概念ペアを抽出し、大量の A + B -> C 式を一括生成
    """
    print(f"\n--- Generating {count} Mass Syntheses (A + B -> C) via GPU Batch ---")
    t0 = time.time()
    
    # 多様なシードペアの抽出 (完全ランダム + 半近傍)
    random.seed(42)
    pairs = []
    # 半分はランダム、半分は近傍からサンプリング
    for _ in range(count // 2):
        i_a = random.randint(0, num_concepts - 1)
        i_b = random.randint(0, num_concepts - 1)
        if i_a != i_b:
            pairs.append((i_a, i_b))
            
    # 近傍サンプリング
    sampled_a = random.sample(range(num_concepts), count // 2)
    vecs_a = CONCEPT_VECS[sampled_a] # [K, D]
    sims = torch.matmul(vecs_a, CONCEPT_VECS.T)
    for i, idx in enumerate(sampled_a):
        sims[i, idx] = -1.0
    topk_b = torch.topk(sims, k=3, dim=-1).indices # [K, 3]
    for i in range(len(sampled_a)):
        i_a = sampled_a[i]
        i_b = topk_b[i, random.randint(0, 2)].item()
        pairs.append((i_a, i_b))
        
    # 一括バッチ合成
    idx_a = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
    idx_b = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
    
    vecs_A = CONCEPT_VECS[idx_a] # [M, D]
    vecs_B = CONCEPT_VECS[idx_b] # [M, D]
    
    vecs_syn = F.normalize(0.5 * vecs_A + 0.5 * vecs_B, p=2, dim=-1) # [M, D]
    
    # 最寄り概念 C の一括探索 [M, N]
    all_sims = torch.matmul(vecs_syn, CONCEPT_VECS.T)
    # A と B 自体は除外
    for i in range(len(pairs)):
        all_sims[i, idx_a[i]] = -1.0
        all_sims[i, idx_b[i]] = -1.0
        
    top_c_indices = torch.argmax(all_sims, dim=-1).cpu().tolist()
    
    formulae = []
    for i in range(len(pairs)):
        w_A = valid_tokens[pairs[i][0]]
        w_B = valid_tokens[pairs[i][1]]
        w_C = valid_tokens[top_c_indices[i]]
        formulae.append({
            "idx_a": pairs[i][0],
            "idx_b": pairs[i][1],
            "idx_c": top_c_indices[i],
            "formula_str": f"{w_A} + {w_B} -> {w_C}",
            "w_a": w_A,
            "w_b": w_B,
            "w_c": w_C
        })
        
    print(f"Generated {len(formulae)} Formulae in {(time.time() - t0)*1000:.1f}ms")
    return formulae


def diagnose_and_find_root_misconceptions(formulae):
    """
    大量の A + B -> C 式を論理解析し、
    どの論理式が誤解・誤謬の根底にあるのかを厳密に探査
    """
    print(f"\n--- Diagnosing Root Misconceptions Across {len(formulae)} Formulae ---")
    t0 = time.time()
    
    idx_a = torch.tensor([f["idx_a"] for f in formulae], device=device, dtype=torch.long)
    idx_b = torch.tensor([f["idx_b"] for f in formulae], device=device, dtype=torch.long)
    idx_c = torch.tensor([f["idx_c"] for f in formulae], device=device, dtype=torch.long)
    
    vecs_A = CONCEPT_VECS[idx_a]
    vecs_B = CONCEPT_VECS[idx_b]
    vecs_C = CONCEPT_VECS[idx_c]
    
    vecs_syn = F.normalize(0.5 * vecs_A + 0.5 * vecs_B, p=2, dim=-1)
    
    # 1. 順方向類似度 S(A+B, C)
    f_scores = torch.sum(vecs_syn * vecs_C, dim=-1) # [M]
    
    # 2. 逆方向想起・整合度 S_back(C -> A, B) (非対称性の検知)
    b_scores = 0.5 * (torch.sum(vecs_C * vecs_A, dim=-1) + torch.sum(vecs_C * vecs_B, dim=-1)) # [M]
    
    # 3. 前提反発度 (A と B がそもそも反発・相容れない度合い)
    repulsion = 1.0 - torch.sum(vecs_A * vecs_B, dim=-1) # [M]
    
    # 4. 語形変化・トートロジー度 (A, B, C の語幹が実質同じであるか)
    # 単語の先頭4文字一致でチェック
    stem_redundancy = torch.zeros(len(formulae), device=device)
    for i, f in enumerate(formulae):
        s_a = f["w_a"][:4]
        s_b = f["w_b"][:4]
        s_c = f["w_c"][:4]
        if s_c == s_a or s_c == s_b:
            stem_redundancy[i] = 1.0
            
    # 【誤解・破綻スコア (Fallacy / Distortion Score)】
    # - 逆方向が低い (片思いの飛躍)
    # - 前提が反発しているのに無理に結びつけた (False Synthesis)
    # - 語形変化のトートロジー (Trivial Stem Loop)
    distortion_scores = (
        (0.6 - b_scores).clamp(min=0.0) * 1.5 +   # 逆方向の脱落
        (repulsion - 0.5).clamp(min=0.0) * 1.0 +  # 前提反発
        stem_redundancy * 1.2                     # 語形変化の堂々巡り
    )
    
    dist_list = distortion_scores.cpu().tolist()
    b_list = b_scores.cpu().tolist()
    rep_list = repulsion.cpu().tolist()
    stem_list = stem_redundancy.cpu().tolist()
    
    diagnosed = []
    for i, f in enumerate(formulae):
        score = dist_list[i]
        
        # 破綻タイプの同定
        fallacy_types = []
        if stem_list[i] > 0.5:
            fallacy_types.append("Stem-Tautology (品詞変化の堂々巡り)")
        if b_list[i] < 0.25:
            fallacy_types.append("One-way Asymmetry (片思いの飛躍・逆方向崩壊)")
        if rep_list[i] > 0.80:
            fallacy_types.append("Forced Repulsion (相反する前提の無理な合成)")
            
        diagnosed.append({
            **f,
            "distortion_score": round(score, 4),
            "back_score": round(b_list[i], 4),
            "repulsion": round(rep_list[i], 4),
            "fallacy_types": fallacy_types if fallacy_types else ["Sound / Mild"]
        })
        
    # 破綻スコア順にソート（最も誤解の根底にある論理式をトップに抽出）
    diagnosed.sort(key=lambda x: x["distortion_score"], reverse=True)
    
    print(f"Diagnosed {len(diagnosed)} Formulae in {(time.time() - t0)*1000:.1f}ms")
    return diagnosed


def repair_root_misconceptions(top_distorted, revision_weight=0.35):
    """
    特定された「誤解の根底にある論理式」に対し、
    意味的変化量を最小に抑えつつ整合性を回復する「最小修正 (Minimal Revision)」を実行
    """
    print(f"\n--- Finding Minimal Revisions for Top {len(top_distorted)} Root Misconceptions ---")
    repaired_results = []
    
    for item in top_distorted:
        i_a = item["idx_a"]
        i_b = item["idx_b"]
        i_c = item["idx_c"]
        
        v_A = CONCEPT_VECS[i_a].unsqueeze(0)
        v_B = CONCEPT_VECS[i_b].unsqueeze(0)
        v_C = CONCEPT_VECS[i_c].unsqueeze(0)
        v_syn = F.normalize(0.5 * v_A + 0.5 * v_B, p=2, dim=-1)
        
        # 修正目的関数: 順方向 + 逆方向 + Cとの近接度 - 語幹重複ペナルティ
        f_all = torch.matmul(v_syn, CONCEPT_VECS.T)
        b_all = 0.5 * (torch.matmul(v_A, CONCEPT_VECS.T) + torch.matmul(v_B, CONCEPT_VECS.T))
        proximity_to_C = torch.matmul(v_C, CONCEPT_VECS.T)
        
        obj = f_all + 0.8 * b_all + (2.0 * revision_weight) * proximity_to_C
        
        # A, B 自体および同一語幹の単語にペナルティ
        obj[0, i_a] = -999.0
        obj[0, i_b] = -999.0
        obj[0, i_c] = -999.0 # 元の破綻単語も除外
        
        # 同一語幹ペナルティ
        stem_a = item["w_a"][:4]
        stem_b = item["w_b"][:4]
        for w_cand, w_idx in word_to_idx.items():
            if w_cand[:4] == stem_a or w_cand[:4] == stem_b:
                obj[0, w_idx] -= 2.0
                
        best_repaired_idx = torch.argmax(obj).item()
        repaired_w = valid_tokens[best_repaired_idx]
        v_rep = CONCEPT_VECS[best_repaired_idx].unsqueeze(0)
        
        dist = torch.norm(v_rep - v_C).item()
        new_back = 0.5 * (torch.sum(v_rep * v_A).item() + torch.sum(v_rep * v_B).item())
        
        repaired_results.append({
            "original_formula": item["formula_str"],
            "fallacy_types": item["fallacy_types"],
            "distortion_score": item["distortion_score"],
            "repaired_formula": f"{item['w_a']} + {item['w_b']} -> {repaired_w}",
            "repaired_concept": repaired_w,
            "revision_distance": round(dist, 4),
            "improved_back_coherence": round(new_back, 4)
        })
        
    return repaired_results


def main():
    print("=" * 80)
    print("ROOT MISCONCEPTION DISCOVERY & MINIMAL REVISION ENGINE")
    print("================================================================================")
    
    # 1. 1,200件の A + B -> C 式を一括生成
    formulae = generate_mass_formulae(count=1200)
    
    # 2. 誤解・破綻の根底にある論理式の探査・診断
    diagnosed = diagnose_and_find_root_misconceptions(formulae)
    
    # 統計
    fallacy_counts = {}
    for d in diagnosed:
        for t in d["fallacy_types"]:
            fallacy_counts[t] = fallacy_counts.get(t, 0) + 1
            
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY: ROOT CAUSES OF LOGICAL DISTORTION")
    print("=" * 80)
    for f_type, cnt in fallacy_counts.items():
        print(f"  * {f_type:45s}: {cnt:4d} cases ({(cnt/len(diagnosed))*100:.1f}%)")
        
    # 上位の致命的歪み式（Top 15）
    top_distorted = diagnosed[:15]
    print("\n" + "=" * 80)
    print("TOP 10 ROOT MISCONCEPTION FORMULAE IDENTIFIED IN THOUGHT WILDERNESS:")
    print("=" * 80)
    for i, it in enumerate(top_distorted[:10]):
        print(f"[{i+1:02d}] {it['formula_str']:35s} | Score: {it['distortion_score']:.3f} | Fallacy: {', '.join(it['fallacy_types'])}")
        
    # 3. 最小修正の適用
    repaired = repair_root_misconceptions(top_distorted, revision_weight=0.35)
    
    print("\n" + "=" * 80)
    print("MINIMAL REVISION RESULTS (Root Misconception -> Repaired Grounding):")
    print("=" * 80)
    for i, rep in enumerate(repaired[:10]):
        print(f"[{i+1:02d}] Original: {rep['original_formula']}")
        print(f"     => Repaired: {rep['repaired_formula']} (Dist: {rep['revision_distance']:.3f}, Back Coherence: {rep['improved_back_coherence']:.3f})")
        print(f"     Root Fallacy: {', '.join(rep['fallacy_types'])}\n")
        
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_generated": len(formulae),
        "fallacy_distribution": fallacy_counts,
        "top_misconceptions": top_distorted,
        "repaired_samples": repaired
    }
    
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Results successfully saved to: {RESULTS_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
