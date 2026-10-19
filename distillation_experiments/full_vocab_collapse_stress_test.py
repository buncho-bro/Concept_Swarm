import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Full-Vocab Collapse Stress Test] Initializing model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()  # (151936, 2048)
vocab_size = W_embed.shape[0]
print(f"[Input Embeddings] Loaded full vocabulary table: {vocab_size} tokens.")

# 1. 自立語（概念）フィルタリング: 意味を持つ全単語を抽出
clean_word_to_id = {}
for tid in range(vocab_size):
    decoded = tokenizer.decode([tid])
    s_strip = decoded.strip()
    if len(s_strip) >= 2 and s_strip.isalpha():
        clean_word = s_strip.lower()
        if clean_word not in clean_word_to_id:
            clean_word_to_id[clean_word] = tid

selected_words = list(clean_word_to_id.keys())
selected_ids = torch.tensor([clean_word_to_id[w] for w in selected_words], device=device, dtype=torch.long)
N_total = len(selected_words)
print(f"[Filtered Concepts] Total meaningful words in vocabulary: {N_total}")

# 入力層の埋め込みベクトル (N_total, 2048)
W_clean = F.normalize(W_embed[selected_ids], p=2, dim=-1)

# 2. 全単語の「線の類似度（軌道相関）」の高速サンプリング・推定
# 全N_total x N_totalは行列が巨大 (74563 x 74563 = 55億要素, ~22GB) のため、
# 主成分（PCA / SVD）を用いて高次元の線の多様体構造を圧縮抽出
print("[Step 2] Projecting full input vocabulary to initial Wilderness surface map...")
# 特異値分解 (SVD) による最良2D平面射影
U, S, V = torch.pca_lowrank(W_clean, q=2)
init_coords = (U * S).cpu().numpy()  # (N_total, 2)

# スケール正規化 [-20, 20]
coords_std = init_coords.std(axis=0) + 1e-6
init_coords = (init_coords - init_coords.mean(axis=0)) / coords_std * 10.0

# 3. 「すべてが沈み込む破綻（重力崩壊）」の物理シミュレーション検証
# 全単語が円形引力（ポテンシャル井戸）を持ったとき、空間がどうなるかを検証
print("\n[Step 3] Simulating Gravitational Interaction & Testing for Collapse...")

# サンプリングによる相互作用密度の解析 (大規模10,000点バッチ)
sample_size = min(10000, N_total)
np.random.seed(42)
sample_indices = np.random.choice(N_total, sample_size, replace=False)
sample_coords = torch.tensor(init_coords[sample_indices], device=device, dtype=torch.float32)

# 物理緩和（引力 vs 斥力）をシミュレート
# パターンA: 引力のみ（万有引力・類似度引き合いのみ）➔ 重力崩壊が起きるか？
# パターンB: 引力 ＋ 斥力（体積排除・パウリの排他律）➔ 安定した原野が保たれるか？

def simulate_field_evolution(coords_init, use_repulsion=False, steps=50):
    pos = coords_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=0.1)
    
    radii_history = []
    density_history = []
    
    for step in range(steps):
        opt.zero_grad()
        # ペアワイズ距離
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (S, S, 2)
        dist = torch.norm(diff, dim=-1) + 1e-6  # (S, S)
        
        # 引力ポテンシャル: 類似度や近接に応じた引き込み
        # V_attr = - exp(-dist^2 / 2sigma^2)
        sigma = 2.0
        v_attr = - torch.exp(- (dist**2) / (2.0 * sigma**2)).sum()
        
        if use_repulsion:
            # 斥力（体積排除: 近すぎると反発する Lennard-Jones / パウリ排他）
            # V_rep = (sigma_rep / dist)^2
            v_rep = (0.8 / dist).clamp(max=10.0).sum()
            loss = v_attr + v_rep
        else:
            loss = v_attr  # 純粋な引力のみ
            
        loss.backward()
        opt.step()
        
        # 空間の広がり（平均半径）と局所密度の記録
        mean_radius = torch.norm(pos - pos.mean(dim=0), dim=-1).mean().item()
        min_dist = dist[dist > 1e-4].min().item()
        radii_history.append(mean_radius)
        density_history.append(min_dist)
        
    return pos.detach().cpu().numpy(), radii_history, density_history

print("  Running Simulation A: Attraction ONLY (Testing for Gravitational Collapse)...")
pos_attr_only, r_hist_A, d_hist_A = simulate_field_evolution(sample_coords, use_repulsion=False, steps=60)

print("  Running Simulation B: Attraction + Volume Exclusion (Stabilized Field)...")
pos_stabilized, r_hist_B, d_hist_B = simulate_field_evolution(sample_coords, use_repulsion=True, steps=60)

# 4. 「すべてが沈み込む破綻」の判定
collapse_ratio_A = r_hist_A[-1] / r_hist_A[0]
collapse_ratio_B = r_hist_B[-1] / r_hist_B[0]

print(f"\n================== COLLAPSE STRESS-TEST RESULTS ==================")
print(f"Initial Mean Space Radius: {r_hist_A[0]:.3f}")
print(f"Pattern A (Attraction Only):")
print(f"  Final Mean Radius: {r_hist_A[-1]:.3f} (Shrinkage: {collapse_ratio_A*100:.1f}%)")
print(f"  Minimum Word Distance: {d_hist_A[-1]:.6f}")
if collapse_ratio_A < 0.2 or d_hist_A[-1] < 1e-3:
    print("  🚨 [ALERT] Pattern A collapsed into a singular black hole! (引力のみでは1点へ重力崩壊)")
else:
    print("  ✅ Pattern A maintained shape.")

print(f"Pattern B (Attraction + Volume Exclusion):")
print(f"  Final Mean Radius: {r_hist_B[-1]:.3f} (Shrinkage: {collapse_ratio_B*100:.1f}%)")
print(f"  Minimum Word Distance: {d_hist_B[-1]:.6f}")
if collapse_ratio_B >= 0.7 and d_hist_B[-1] > 0.05:
    print("  💎 [STABLE] Pattern B completely prevented collapse! Rich wilderness maintained.")
else:
    print("  ⚠️ Pattern B partially compressed.")

# 5. ポテンシャル面の起伏（テクスチャ）の測定: 底なし沼化していないか？
# 空間全体のポテンシャル起伏（分散・峰と谷の深さ）を計算
grid_x, grid_y = np.meshgrid(np.linspace(-15, 15, 200), np.linspace(-15, 15, 200))

def compute_potential_relief(positions):
    pot = np.zeros_like(grid_x)
    for p in positions[:1000]:  # 代表1000語
        r2 = (grid_x - p[0])**2 + (grid_y - p[1])**2
        pot -= np.exp(-r2 / (2 * 1.5**2))
    # 起伏の深さ（Max - Min）と標準偏差
    relief_depth = pot.max() - pot.min()
    relief_std = pot.std()
    return pot, relief_depth, relief_std

pot_A, depth_A, std_A = compute_potential_relief(pos_attr_only)
pot_B, depth_B, std_B = compute_potential_relief(pos_stabilized)

print(f"\n[Potential Field Relief Analysis]")
print(f"Pattern A: Depth = {depth_A:.2f}, Std (Curvature/Ruggedness) = {std_A:.2f}")
print(f"Pattern B: Depth = {depth_B:.2f}, Std (Curvature/Ruggedness) = {std_B:.2f}")

# 6. 比較可視化の生成
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

# A: 崩壊パターン
axes[0].contourf(grid_x, grid_y, pot_A, levels=30, cmap="inferno")
axes[0].scatter(pos_attr_only[:500, 0], pos_attr_only[:500, 1], s=5, color="cyan", alpha=0.5)
axes[0].set_title(f"Pattern A: Attraction Only\n(Gravity Collapse Risk: Shrinkage {collapse_ratio_A*100:.1f}%)", fontsize=12, weight="bold")
axes[0].grid(True, linestyle="--", alpha=0.3)

# B: 安定パターン
cs = axes[1].contourf(grid_x, grid_y, pot_B, levels=30, cmap="viridis")
axes[1].scatter(pos_stabilized[:500, 0], pos_stabilized[:500, 1], s=5, color="magenta", alpha=0.5)
axes[1].set_title(f"Pattern B: Attraction + Exclusion\n(Stabilized Wilderness: Preserved {collapse_ratio_B*100:.1f}%)", fontsize=12, weight="bold")
axes[1].grid(True, linestyle="--", alpha=0.3)
fig.colorbar(cs, ax=axes.ravel().tolist(), label="Wilderness Potential")

out_img = "full_vocab_collapse_comparison.png"
plt.savefig(out_img, bbox_inches="tight")
plt.close()
print(f"\nSaved stress-test comparison visualization to: {out_img}")
