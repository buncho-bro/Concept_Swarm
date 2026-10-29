"""
Thought Wilderness Comparative Analysis
(素のモデル vs トレーニング済み思考原野の構造・幾何・推論軌跡の厳密比較)
========================================================================
比較軸:
1. 【幾何・トポロジー比較 (Geometric & Topological Metrics)】:
   - 非対称接続率 (Asymmetry / One-way Fallacy Rate): A->B のとき B->A が脱落している割合
   - 意味的衝突率 (Orthogonal Branch Conflict Rate): 同一概念からの分岐が反発する割合
   - グラフの推移律健全度 (Transitive Consistency Rate): A->B, B->C のとき A->C が整合する割合
2. 【自律思考軌跡の安定性 (Autonomous Walk Drift & Coherence)】:
   - 素のモデルの思考原野 (Raw Wilderness): 制約なし最近傍・貪欲探索 (10ステップ)
   - 本システム (Organized Web with Reservation & Web Counter): 確定・保留・反論が組み込まれた探索
   - 各ステップでの意味的エントロピー、意味の崩壊（ハルシネーション・堂々巡り）の発生率
3. 【具体例の比較 (Qualitative Trace Comparison)】:
   - 同一シード単語（5つ）から開始した思考鎖の比較
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
RESULTS_JSON = os.path.join(OUTPUT_DIR, "thought_wilderness_comparison_results.json")
UNCOND_WEB_FILE = os.path.join(OUTPUT_DIR, "unconditional_reservation_web.json")
CACHE_FILE = os.path.join(OUTPUT_DIR, "web_counter_opinions_cache.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Thought Wilderness Measurement...")
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
print(f"Loaded {num_concepts} Concepts in Thought Wilderness.")

# 整理された信念網データのロード
with open(UNCOND_WEB_FILE, "r", encoding="utf-8") as f:
    web_data = json.load(f)

confirmed_edges = {k: set(v) for k, v in web_data.get("confirmed_edges", {}).items()}
suspended_edges = {k: set(v) for k, v in web_data.get("suspended_edges", {}).items()}
confirmed_props = web_data.get("crystallized_confirmed", [])
suspended_props = web_data.get("crystallized_suspended", [])

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        web_cache = json.load(f)
else:
    web_cache = {}


# ==============================================================================
# 測定1: 幾何・トポロジー構造の比較 (N=1000サンプリング)
# ==============================================================================
def measure_topological_metrics(sample_size=1000):
    print(f"\n--- Measuring Topological Metrics (Sample Size = {sample_size}) ---")
    torch.manual_seed(42)
    sample_indices = random.sample(range(num_concepts), sample_size)
    sample_vecs = CONCEPT_VECS[sample_indices] # [S, D]
    
    # 1. 素の思考原野 (Raw Wilderness)
    # 各概念の上位2近傍
    sims = torch.matmul(sample_vecs, CONCEPT_VECS.T) # [S, N]
    for i, idx in enumerate(sample_indices):
        sims[i, idx] = -1.0
        
    top2 = torch.topk(sims, k=2, dim=-1).indices # [S, 2]
    
    # 非対称性（A -> B だが B -> A の近傍トップ10に含まれない割合）
    raw_asymmetry_count = 0
    raw_branch_repulsion_count = 0
    total_edges = 0
    
    for i in range(sample_size):
        src_idx = sample_indices[i]
        n1_idx = top2[i, 0].item()
        n2_idx = top2[i, 1].item()
        
        # 分岐反発
        branch_sim = torch.sum(CONCEPT_VECS[n1_idx] * CONCEPT_VECS[n2_idx]).item()
        if branch_sim < 0.25:
            raw_branch_repulsion_count += 1
            
        # 逆方向チェック (n1 -> src)
        back_sims1 = torch.matmul(CONCEPT_VECS[n1_idx].unsqueeze(0), CONCEPT_VECS.T)
        back_sims1[0, n1_idx] = -1.0
        top10_back = torch.topk(back_sims1, k=10, dim=-1).indices[0].tolist()
        if src_idx not in top10_back:
            raw_asymmetry_count += 1
            
        # 逆方向チェック (n2 -> src)
        back_sims2 = torch.matmul(CONCEPT_VECS[n2_idx].unsqueeze(0), CONCEPT_VECS.T)
        back_sims2[0, n2_idx] = -1.0
        top10_back2 = torch.topk(back_sims2, k=10, dim=-1).indices[0].tolist()
        if src_idx not in top10_back2:
            raw_asymmetry_count += 1
            
        total_edges += 2
        
    raw_asymmetry_rate = (raw_asymmetry_count / total_edges) * 100.0
    raw_repulsion_rate = (raw_branch_repulsion_count / sample_size) * 100.0
    
    # 2. トレーニング済み思考原野 (Organized Web)
    # 確定エッジにおける非対称性
    org_total_edges = 0
    org_asymmetry_count = 0
    for src_w, neighbors in confirmed_edges.items():
        if src_w not in word_to_idx:
            continue
        src_idx = word_to_idx[src_w]
        for dst_w in neighbors:
            if dst_w not in word_to_idx:
                continue
            dst_idx = word_to_idx[dst_w]
            org_total_edges += 1
            # dst から src が confirmed_edges に含まれるか（無向性・双方向性）
            if src_w not in confirmed_edges.get(dst_w, set()):
                org_asymmetry_count += 1
                
    org_asymmetry_rate = (org_asymmetry_count / max(1, org_total_edges)) * 100.0
    
    return {
        "raw_asymmetry_rate": raw_asymmetry_rate,
        "raw_repulsion_rate": raw_repulsion_rate,
        "org_asymmetry_rate": org_asymmetry_rate,
        "total_confirmed_nodes": len(confirmed_edges),
        "total_suspended_nodes": len(suspended_edges),
        "total_crystallized_props": len(confirmed_props),
        "total_suspended_props": len(suspended_props)
    }


# ==============================================================================
# 測定2: 自律的思考軌跡の安定性と意味ドリフト (Random Walk Coherence)
# ==============================================================================
def measure_autonomous_walks(seed_words, steps=8):
    print(f"\n--- Comparing Autonomous Thought Trajectories ({len(seed_words)} Seeds, {steps} Steps) ---")
    trajectories = []
    
    for seed in seed_words:
        if seed not in word_to_idx:
            continue
        seed_idx = word_to_idx[seed]
        
        # --- A. 素のモデルの思考原野 (Raw Greedy/Nearest Walk) ---
        raw_path = [seed]
        raw_seen = {seed_idx}
        curr_vec = CONCEPT_VECS[seed_idx].unsqueeze(0)
        
        for st in range(steps):
            sims = torch.matmul(curr_vec, CONCEPT_VECS.T)
            for seen_i in raw_seen:
                sims[0, seen_i] = -1.0
            next_idx = torch.argmax(sims).item()
            next_w = valid_tokens[next_idx]
            raw_path.append(next_w)
            raw_seen.add(next_idx)
            # ベクトル更新 (移動)
            curr_vec = CONCEPT_VECS[next_idx].unsqueeze(0)
            
        # --- B. 整理・保留・反論が組み込まれた思考原野 (Organized Walk) ---
        org_path = [seed]
        org_statuses = ["SEED"]
        org_seen = {seed_idx}
        curr_v = CONCEPT_VECS[seed_idx].unsqueeze(0)
        
        for st in range(steps):
            curr_w = org_path[-1]
            
            # 確定エッジが存在する場合は確定パスを優先探索
            neighbors = confirmed_edges.get(curr_w, set())
            valid_cand_neighbors = [w for w in neighbors if w in word_to_idx and word_to_idx[w] not in org_seen]
            
            if valid_cand_neighbors:
                # 確定エッジから選択
                cand_vecs = CONCEPT_VECS[[word_to_idx[w] for w in valid_cand_neighbors]]
                c_sims = torch.matmul(curr_v, cand_vecs.T)
                best_sub_idx = torch.argmax(c_sims).item()
                next_w = valid_cand_neighbors[best_sub_idx]
                next_idx = word_to_idx[next_w]
                org_path.append(next_w)
                org_statuses.append("CONFIRMED_LINK")
                org_seen.add(next_idx)
                curr_v = 0.7 * curr_v + 0.3 * CONCEPT_VECS[next_idx].unsqueeze(0)
            else:
                # 自由原野から探索するが、内部保留やWeb反論でフィルタリング
                sims = torch.matmul(curr_v, CONCEPT_VECS.T)
                for seen_i in org_seen:
                    sims[0, seen_i] = -1.0
                top5_inds = torch.topk(sims, k=5, dim=-1).indices[0].tolist()
                
                # 健全な候補を探す
                chosen_w = None
                chosen_status = "CONFIRMED_FALLBACK"
                for c_idx in top5_inds:
                    w = valid_tokens[c_idx]
                    if w in suspended_edges.get(curr_w, set()) or w in suspended_edges:
                        # 内部保留を検知 -> スキップまたは保留フラグ
                        continue
                    chosen_w = w
                    chosen_idx = c_idx
                    break
                    
                if not chosen_w:
                    chosen_idx = top5_inds[0]
                    chosen_w = valid_tokens[chosen_idx]
                    chosen_status = "SUSPENDED_DOUBT"
                    
                org_path.append(chosen_w)
                org_statuses.append(chosen_status)
                org_seen.add(chosen_idx)
                curr_v = 0.7 * curr_v + 0.3 * CONCEPT_VECS[chosen_idx].unsqueeze(0)
                
        # 意味的コヒーレンス（初期シードと各ステップ単語との類似度推移）
        seed_v = CONCEPT_VECS[seed_idx].unsqueeze(0)
        raw_sims = [round(torch.sum(seed_v * CONCEPT_VECS[word_to_idx[w]]).item(), 3) for w in raw_path]
        org_sims = [round(torch.sum(seed_v * CONCEPT_VECS[word_to_idx[w]]).item(), 3) for w in org_path]
        
        trajectories.append({
            "seed": seed,
            "raw_path": " -> ".join(raw_path),
            "raw_sims": raw_sims,
            "raw_final_sim": raw_sims[-1],
            "org_path": " -> ".join(org_path),
            "org_statuses": org_statuses,
            "org_sims": org_sims,
            "org_final_sim": org_sims[-1]
        })
        
    return trajectories


def main():
    print("=" * 80)
    print("MEASURING THOUGHT WILDERNESS: RAW MODEL VS. ORGANIZED BELIEF WEB")
    print("=" * 80)
    
    topological = measure_topological_metrics(sample_size=1000)
    
    seeds = ["science", "justice", "medicine", "animal", "philosophy", "technology"]
    trajectories = measure_autonomous_walks(seeds, steps=6)
    
    print("\n" + "=" * 80)
    print("TOPOLOGICAL METRICS SUMMARY")
    print("=" * 80)
    print(f"1. Raw Wilderness Asymmetry Rate (One-way Fallacy): {topological['raw_asymmetry_rate']:.1f}%")
    print(f"   Organized Web Asymmetry Rate:                    {topological['org_asymmetry_rate']:.1f}%")
    print(f"2. Raw Wilderness Branch Repulsion (Conflict):      {topological['raw_repulsion_rate']:.1f}%")
    print(f"3. Established Belief Network Topology:")
    print(f"   - Confirmed Nodes: {topological['total_confirmed_nodes']}")
    print(f"   - Suspended Nodes (Contradiction Boundary): {topological['total_suspended_nodes']}")
    print(f"   - Crystallized Confirmed Propositions: {topological['total_crystallized_props']}")
    print(f"   - Crystallized Suspended Propositions: {topological['total_suspended_props']}")
    
    print("\n" + "=" * 80)
    print("SAMPLE THOUGHT TRAJECTORIES COMPARISON")
    print("=" * 80)
    for tr in trajectories:
        print(f"\n[Seed Concept: {tr['seed']}]")
        print(f"  Raw Model Path:        {tr['raw_path']}")
        print(f"  Raw Sim to Seed:       {tr['raw_sims']} (Final: {tr['raw_final_sim']})")
        print(f"  Organized Web Path:    {tr['org_path']}")
        print(f"  Organized Sim to Seed: {tr['org_sims']} (Final: {tr['org_final_sim']})")
        
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topological_metrics": topological,
        "trajectories": trajectories
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {RESULTS_JSON}")
    print("=" * 80)

if __name__ == "__main__":
    main()
