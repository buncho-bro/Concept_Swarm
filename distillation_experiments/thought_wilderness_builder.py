import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Thought Wilderness Builder] Initializing model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

# 1. ベンチマークからコア概念単語（入力語＋期待される因果語）を抽出
from massive_causal_benchmark import BENCHMARK_PAIRS

concept_words = set()
for item in BENCHMARK_PAIRS:
    for w in item["pair"]:
        concept_words.add(w.lower())
    for exp in item["expected"]:
        if len(exp) >= 3 and exp.isalpha():
            concept_words.add(exp.lower())

concept_words = sorted(list(concept_words))
print(f"[Step 1] Selected {len(concept_words)} core concept words for Thought Wilderness.")

# 2. 各単語の「線（止まるまでの全33層軌跡）」を抽出
def extract_trajectory(word: str):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h_states = [F.normalize(h[0, -1, :].float(), p=2, dim=-1) for h in out.hidden_states]
        traj = torch.stack(h_states, dim=0)  # (33, 2048)
    return traj

print("[Step 2] Extracting full-layer trajectories (lines) for all words...")
trajectories = {}
for idx, w in enumerate(concept_words):
    trajectories[w] = extract_trajectory(w)
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(concept_words)} trajectories...")

# 3. 「線の類似度（軌道の相関・アライメント）」を計算
print("[Step 3] Computing Line-to-Line (Trajectory) Similarity Matrix...")
N = len(concept_words)
traj_list = [trajectories[w] for w in concept_words]

# 軌道間の位置類似度と速度類似度の統合
sim_matrix = np.zeros((N, N), dtype=np.float32)

for i in range(N):
    traj_i = traj_list[i]  # (33, 2048)
    v_i = traj_i[1:] - traj_i[:-1]  # (32, 2048)
    v_i_norm = F.normalize(v_i, p=2, dim=-1)
    
    for j in range(i, N):
        traj_j = traj_list[j]
        v_j = traj_j[1:] - traj_j[:-1]
        v_j_norm = F.normalize(v_j, p=2, dim=-1)
        
        # 1. 各層での位置のコサイン類似度の全層平均
        pos_sim = (traj_i * traj_j).sum(dim=-1).mean().item()
        
        # 2. 各層での速度ベクトル（曲がり・運動の向き）のコサイン類似度の全層平均
        vel_sim = (v_i_norm * v_j_norm).sum(dim=-1).mean().item()
        
        # 統合線の類似度 (0〜1)
        line_sim = 0.5 * (pos_sim + max(0.0, vel_sim))
        sim_matrix[i, j] = line_sim
        sim_matrix[j, i] = line_sim

# 4. 「線の類似度」から「面のマップ（2次元平面）」を構築
print("[Step 4] Projecting line similarities to 2D Surface Map via Multidimensional Scaling (MDS)...")
dist_matrix = 1.0 - sim_matrix
np.fill_diagonal(dist_matrix, 0.0)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=500)
coords_2d = mds.fit_transform(dist_matrix)  # (N, 2)

# 座標を [-10, 10] に正規化
coords_2d = (coords_2d - coords_2d.mean(axis=0)) / (coords_2d.std(axis=0) + 1e-6) * 5.0

# 5. 各単語の円形引力分布を空間にばらまき、力学緩和で「思考原野」を安定化
print("[Step 5] Distributing Circular Attraction Fields and stabilizing the Thought Wilderness...")
coords_tensor = torch.tensor(coords_2d, dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([coords_tensor], lr=0.05)

# 力学安定化: 線の類似度が高い単語同士は引力、近すぎる単語同士は斥力
target_dists = torch.tensor(dist_matrix, dtype=torch.float32)

for step in range(120):
    optimizer.zero_grad()
    # ペアワイズ距離
    diff = coords_tensor.unsqueeze(0) - coords_tensor.unsqueeze(1)  # (N, N, 2)
    current_dists = torch.norm(diff, dim=-1)  # (N, N)
    
    # 円形引力の相互作用エネルギー最小化
    loss = F.mse_loss(current_dists, target_dists * 7.0)
    loss.backward()
    optimizer.step()

final_coords = coords_tensor.detach().numpy()
word_to_coord = {w: final_coords[i] for i, w in enumerate(concept_words)}

# 6. 「思考原野」の連続ポテンシャル面（引力の重ね合わせ場）を生成
print("[Step 6] Rendering Thought Wilderness Potential Field...")
grid_x, grid_y = np.meshgrid(np.linspace(-15, 15, 300), np.linspace(-15, 15, 300))
potential_field = np.zeros_like(grid_x)

sigma = 1.8  # 円形引力の広がり半径
for i, w in enumerate(concept_words):
    cx, cy = final_coords[i]
    r2 = (grid_x - cx)**2 + (grid_y - cy)**2
    # 各単語の円形引力ポテンシャル（引力は窪み）
    potential_field -= np.exp(-r2 / (2 * sigma**2))

# 7. 可視化画像の作成
plt.figure(figsize=(14, 12), dpi=150)
plt.contourf(grid_x, grid_y, potential_field, levels=40, cmap="viridis", alpha=0.85)
plt.colorbar(label="Thought Wilderness Potential (Attraction Wells)")

# 代表的な単語のプロットとラベル表示
highlight_keys = [
    "sun", "ice", "water", "melt", "glacier",
    "fire", "wood", "burn", "ash", "smoke",
    "crime", "police", "arrest",
    "earthquake", "ocean", "tsunami", "flood",
    "cow", "grass", "pasture",
    "match", "candle", "wax",
    "cloud", "cold", "snow", "rain",
    "seed", "soil", "sprout", "plant"
]

for w in concept_words:
    cx, cy = word_to_coord[w]
    if w in highlight_keys:
        plt.scatter(cx, cy, color="red", s=60, edgecolors="white", linewidth=1.5, zorder=5)
        plt.text(cx + 0.25, cy + 0.25, w, fontsize=10, weight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7), zorder=6)
    else:
        plt.scatter(cx, cy, color="cyan", s=15, alpha=0.4, zorder=4)

plt.title("Thought Wilderness (思考原野): Stabilized Field from Line-Similarity Map", fontsize=14, weight="bold", pad=15)
plt.xlabel("Wilderness Dimension X", fontsize=11)
plt.ylabel("Wilderness Dimension Y", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)

out_png = "thought_wilderness_map.png"
plt.savefig(out_png, bbox_inches="tight")
plt.close()
print(f"[Done] Thought Wilderness successfully built and saved to: {out_png}")

# 8. 代表的な因果ペアの原野上での位置関係を数値検証
print("\n================== WILDERNESS CAUSAL LOCALITY VERIFICATION ==================")
def inspect_wilderness_causality(pair, target):
    w1, w2 = pair
    p1 = word_to_coord.get(w1)
    p2 = word_to_coord.get(w2)
    pt = word_to_coord.get(target)
    if p1 is not None and p2 is not None and pt is not None:
        mid = 0.5 * (p1 + p2)
        dist_to_mid = np.linalg.norm(pt - mid)
        dist_w1_w2 = np.linalg.norm(p2 - p1)
        print(f"[{w1} + {w2}] ➔ Target: '{target}'")
        print(f"  Word A: {p1}, Word B: {p2}, Target: {pt}")
        print(f"  Distance Target to Midpoint: {dist_to_mid:.3f} (Distance A to B: {dist_w1_w2:.3f})")

inspect_wilderness_causality(("sun", "ice"), "melt")
inspect_wilderness_causality(("fire", "wood"), "ash")
inspect_wilderness_causality(("crime", "police"), "arrest")
inspect_wilderness_causality(("cloud", "cold"), "snow")
inspect_wilderness_causality(("earthquake", "ocean"), "tsunami")
inspect_wilderness_causality(("match", "candle"), "wax")
inspect_wilderness_causality(("cow", "grass"), "pasture")

# データ保存
summary_data = {
    "words": concept_words,
    "coords": {w: word_to_coord[w].tolist() for w in concept_words}
}
with open("thought_wilderness_data.json", "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)
print("Saved Thought Wilderness coordinates to: thought_wilderness_data.json")
