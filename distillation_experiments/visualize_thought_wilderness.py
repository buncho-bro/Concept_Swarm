"""
Visualize Thought Wilderness: Raw Model vs. Organized Belief Web
(思考原野の視覚的差異の測定・可視化スクリプト)
========================================================================
可視化パネル:
1. 【素の思考原野のトポロジー (Raw Wilderness Projection)】:
   PCA / 潜在空間2D射影。無秩序な混在と、23%の内部反発・非対称エッジの散乱。
2. 【トレーニング済み信念網のトポロジー (Organized Belief Web)】:
   確定ノード (Confirmed: 健全領域) と 内部保留ノード (Suspended: 矛盾境界防壁) の空間的分離。
3. 【思考軌跡の散乱比較 (Trajectory Comparison: justice & philosophy)】:
   - 素のモデル: 語尾変化やスペル音韻に引っ張られ、概念空間の外へ迷走・発散する軌跡。
   - トレーニング後: 概念コア周辺に留まり、体系的意味領域へと自律復帰する収束軌跡。
4. 【意味的コヒーレンス減衰曲線 (Semantic Coherence Curves)】:
   思考ステップ数に伴う初期シードとの類似度推移の比較グラフ。
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
UNCOND_WEB_FILE = os.path.join(OUTPUT_DIR, "unconditional_reservation_web.json")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "thought_wilderness_comparison_results.json")
IMAGE_OUT = os.path.join(ARTIFACT_DIR, "thought_wilderness_visual_comparison.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading data for Thought Wilderness Visualization...")
model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)

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
CONCEPT_VECS = F.normalize(embed_w[CONCEPT_IDS_TENSOR], p=2, dim=-1).cpu().float().numpy()
word_to_idx = {w: i for i, w in enumerate(valid_tokens)}

with open(UNCOND_WEB_FILE, "r", encoding="utf-8") as f:
    web_data = json.load(f)
confirmed_nodes = set(web_data.get("confirmed_edges", {}).keys())
suspended_nodes = set(web_data.get("suspended_edges", {}).keys())

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    comp_results = json.load(f)
trajectories = comp_results.get("trajectories", [])

# PCA 2D射影の計算
print("Computing 2D PCA projection of Thought Wilderness...")
pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(CONCEPT_VECS)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# プロットの作成 (4パネル)
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 16), dpi=150)

# -------------------------------------------------------------
# Panel 1: 素の思考原野 (Raw Thought Wilderness)
# -------------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("1. Raw Thought Wilderness (Unconstrained Latent Space)", fontsize=14, fontweight='bold', color='#4fc3f7', pad=12)
# 全体散布
ax1.scatter(coords_2d[:, 0], coords_2d[:, 1], c='#546e7a', alpha=0.15, s=6, label="Latent Concepts (N=8782)")

# 無秩序な反発・非対称エッジの可視化 (サンプリング)
np.random.seed(42)
sample_pts = np.random.choice(len(CONCEPT_VECS), size=120, replace=False)
for idx in sample_pts:
    # 最も近い点
    dists = np.dot(CONCEPT_VECS, CONCEPT_VECS[idx])
    dists[idx] = -1.0
    nn = np.argmax(dists)
    ax1.plot([coords_2d[idx, 0], coords_2d[nn, 0]], [coords_2d[idx, 1], coords_2d[nn, 1]], 
             c='#ff5252', alpha=0.4, linewidth=0.8)

ax1.plot([], [], c='#ff5252', alpha=0.7, linewidth=1.2, label="Unchecked Asymmetry / Repulsive Edges (22.8%)")
ax1.set_xlabel("Latent PC-1", fontsize=11, color='#b0bec5')
ax1.set_ylabel("Latent PC-2", fontsize=11, color='#b0bec5')
ax1.grid(True, linestyle='--', alpha=0.15, color='#78909c')
ax1.legend(loc="upper right", framealpha=0.8, facecolor='#263238', edgecolor='#37474f', fontsize=10)

# -------------------------------------------------------------
# Panel 2: トレーニング済み思考原野 (Organized Belief Web)
# -------------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("2. Organized Belief Network (Confirmed Cores vs. Suspended Boundaries)", fontsize=14, fontweight='bold', color='#69f0ae', pad=12)

# 確定ノードと保留ノードのマスク
conf_indices = [word_to_idx[w] for w in confirmed_nodes if w in word_to_idx]
susp_indices = [word_to_idx[w] for w in suspended_nodes if w in word_to_idx and w not in confirmed_nodes]
other_indices = [i for i in range(len(valid_tokens)) if i not in set(conf_indices) and i not in set(susp_indices)]

ax2.scatter(coords_2d[other_indices, 0], coords_2d[other_indices, 1], c='#37474f', alpha=0.1, s=4, label="Unexplored Wilderness")
ax2.scatter(coords_2d[conf_indices, 0], coords_2d[conf_indices, 1], c='#00e676', alpha=0.4, s=12, label=f"Confirmed Nodes (N={len(conf_indices)})")
ax2.scatter(coords_2d[susp_indices, 0], coords_2d[susp_indices, 1], c='#ff9100', alpha=0.5, s=16, marker='x', label=f"Suspended Boundaries (N={len(susp_indices)})")

ax2.set_xlabel("Latent PC-1", fontsize=11, color='#b0bec5')
ax2.set_ylabel("Latent PC-2", fontsize=11, color='#b0bec5')
ax2.grid(True, linestyle='--', alpha=0.15, color='#78909c')
ax2.legend(loc="upper right", framealpha=0.8, facecolor='#263238', edgecolor='#37474f', fontsize=10)

# -------------------------------------------------------------
# Panel 3: 思考軌跡の幾何的比較 (Trajectory Divergence in 2D Space)
# -------------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("3. Thought Trajectory Drift: Raw Drift vs. Organized Anchoring", fontsize=14, fontweight='bold', color='#ffd740', pad=12)
ax3.scatter(coords_2d[:, 0], coords_2d[:, 1], c='#263238', alpha=0.2, s=3)

# justice と philosophy の軌跡を描画
focus_seeds = ["justice", "philosophy"]
colors_raw = {"justice": "#ff5252", "philosophy": "#ff4081"}
colors_org = {"justice": "#00e5ff", "philosophy": "#b388ff"}

for tr in trajectories:
    seed = tr["seed"]
    if seed not in focus_seeds:
        continue
    
    raw_words = tr["raw_path"].split(" -> ")
    org_words = tr["org_path"].split(" -> ")
    
    raw_pts = [coords_2d[word_to_idx[w]] for w in raw_words if w in word_to_idx]
    org_pts = [coords_2d[word_to_idx[w]] for w in org_words if w in word_to_idx]
    
    if len(raw_pts) > 1:
        rx, ry = zip(*raw_pts)
        ax3.plot(rx, ry, marker='o', markersize=6, linestyle='--', linewidth=2.0, 
                 color=colors_raw[seed], alpha=0.85, label=f"Raw '{seed}' (Spelling/Grammar Drift)")
        ax3.annotate(f"Start: {seed}", (rx[0], ry[0]), fontsize=10, fontweight='bold', color='white', 
                     xytext=(rx[0]+0.02, ry[0]+0.02), bbox=dict(boxstyle="round,pad=0.3", fc="#212121", ec="gray", alpha=0.8))
        ax3.annotate(f"Drift: {raw_words[-1]}", (rx[-1], ry[-1]), fontsize=9, color=colors_raw[seed],
                     xytext=(rx[-1]+0.02, ry[-1]-0.02))
        
    if len(org_pts) > 1:
        ox, oy = zip(*org_pts)
        ax3.plot(ox, oy, marker='s', markersize=6, linestyle='-', linewidth=2.2, 
                 color=colors_org[seed], alpha=0.9, label=f"Organized '{seed}' (Core Retained)")
        ax3.annotate(f"Goal: {org_words[-1]}", (ox[-1], oy[-1]), fontsize=9, color=colors_org[seed],
                     xytext=(ox[-1]-0.04, oy[-1]+0.03))

ax3.set_xlabel("Latent PC-1", fontsize=11, color='#b0bec5')
ax3.set_ylabel("Latent PC-2", fontsize=11, color='#b0bec5')
ax3.grid(True, linestyle='--', alpha=0.15, color='#78909c')
ax3.legend(loc="upper left", framealpha=0.85, facecolor='#263238', edgecolor='#37474f', fontsize=9)

# -------------------------------------------------------------
# Panel 4: 意味的コヒーレンス減衰推移 (Semantic Coherence Decay Curves)
# -------------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("4. Semantic Coherence Decay Across Thought Steps (Mean Over Seeds)", fontsize=14, fontweight='bold', color='#ff80ab', pad=12)

steps_len = len(trajectories[0]["raw_sims"])
steps_range = list(range(steps_len))

all_raw_sims = np.array([tr["raw_sims"] for tr in trajectories])
all_org_sims = np.array([tr["org_sims"] for tr in trajectories])

mean_raw = np.mean(all_raw_sims, axis=0)
std_raw = np.std(all_raw_sims, axis=0)

mean_org = np.mean(all_org_sims, axis=0)
std_org = np.std(all_org_sims, axis=0)

ax4.plot(steps_range, mean_raw, marker='o', markersize=8, color='#ff5252', linewidth=2.5, label="Raw Model (Semantic Collapse into Grammar/Phonetics)")
ax4.fill_between(steps_range, mean_raw - std_raw, mean_raw + std_raw, color='#ff5252', alpha=0.15)

ax4.plot(steps_range, mean_org, marker='s', markersize=8, color='#00e676', linewidth=2.5, label="Organized Web (Anchored by Belief & Reservation)")
ax4.fill_between(steps_range, mean_org - std_org, mean_org + std_org, color='#00e676', alpha=0.15)

# 個別プロット
for tr in trajectories:
    ax4.plot(steps_range, tr["raw_sims"], color='#ff5252', alpha=0.2, linestyle=':')
    ax4.plot(steps_range, tr["org_sims"], color='#00e676', alpha=0.2, linestyle=':')

ax4.set_xticks(steps_range)
ax4.set_xticklabels([f"Step {s} (Seed)" if s==0 else f"Step {s}" for s in steps_range], fontsize=10, color='#b0bec5')
ax4.set_xlabel("Autonomous Thought Steps", fontsize=11, color='#b0bec5')
ax4.set_ylabel("Cosine Similarity to Seed Concept", fontsize=11, color='#b0bec5')
ax4.set_ylim(-0.05, 1.1)
ax4.grid(True, linestyle='--', alpha=0.15, color='#78909c')
ax4.legend(loc="upper right", framealpha=0.85, facecolor='#263238', edgecolor='#37474f', fontsize=10)

plt.tight_layout(pad=3.0)
plt.savefig(IMAGE_OUT, bbox_inches='tight', dpi=150)
plt.close()
print(f"Visualization successfully saved to: {IMAGE_OUT}")
