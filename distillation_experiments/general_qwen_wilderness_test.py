import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 汎用Qwen (非Coder版): Qwen2.5-7B-Instruct
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print(f"\n========================================================")
print(f"Building Thought Wilderness for General Qwen: {MODEL_ID}...")
print(f"========================================================")

tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True
).eval()

W_embed = model.get_input_embeddings().weight.data.float()
vocab_size = W_embed.shape[0]
print(f"Vocab size: {vocab_size}, Embed dim: {W_embed.shape[1]}")

clean_ids = []
words = []
raw_tokens = []
for tid in range(min(50000, vocab_size)):
    raw = tok.convert_ids_to_tokens(tid)
    s = tok.decode([tid]).strip()
    if len(s) >= 2 and s.isalpha():
        clean_ids.append(tid)
        words.append(s.lower())
        raw_tokens.append(raw)

sample_k = 6000
sample_ids = clean_ids[:sample_k]
sample_words = words[:sample_k]
sample_raws = raw_tokens[:sample_k]

W_sample = F.normalize(W_embed[torch.tensor(sample_ids, device=device)], p=2, dim=-1)

# SVDによる最良2D射影
U, S, V = torch.pca_lowrank(W_sample, q=2)
coords = (U * S).cpu().numpy()
coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6) * 10.0

# 力学緩和 (引力 + 体積排除斥力)
pos = torch.tensor(coords, device=device, dtype=torch.float32, requires_grad=True)
opt = torch.optim.Adam([pos], lr=0.1)

for step in range(60):
    opt.zero_grad()
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1) + 1e-6
    v_attr = - torch.exp(- (dist**2) / (2.0 * 2.0**2)).sum()
    v_rep = (0.8 / dist).clamp(max=10.0).sum()
    loss = v_attr + v_rep
    loss.backward()
    opt.step()

final_pos = pos.detach().cpu().numpy()

# ポテンシャル面の計算
grid_x, grid_y = np.meshgrid(np.linspace(-25, 25, 200), np.linspace(-25, 25, 200))
pot = np.zeros_like(grid_x)
for p in final_pos[:800]:
    r2 = (grid_x - p[0])**2 + (grid_y - p[1])**2
    pot -= np.exp(-r2 / (2 * 1.5**2))

# クラスタ分析
top_pole_idx = [i for i, (x, y) in enumerate(final_pos) if y > 4]
bottom_pole_idx = [i for i, (x, y) in enumerate(final_pos) if y < -4]
center_idx = [i for i, (x, y) in enumerate(final_pos) if -3 <= x <= 3 and -3 <= y <= 3]

print('\n[General Qwen Top Region (y > 4)] Sample Words:')
print([sample_words[i] for i in top_pole_idx[:15]])

print('\n[General Qwen Bottom Region (y < -4)] Sample Words:')
print([sample_words[i] for i in bottom_pole_idx[:15]])

print('\n[General Qwen Center Region (-3 <= x,y <= 3)] Sample Words:')
print([sample_words[i] for i in center_idx[:15]])

# 可視化画像の生成
plt.figure(figsize=(10, 9), dpi=150)
plt.contourf(grid_x, grid_y, pot, levels=35, cmap="viridis")
plt.scatter(final_pos[:500, 0], final_pos[:500, 1], s=6, color="magenta", alpha=0.6)
plt.title(f"General Qwen2.5-7B (Non-Coder)\nThought Wilderness Topology", fontsize=14, weight="bold", pad=15)
plt.xlabel("Dimension X", fontsize=11)
plt.ylabel("Dimension Y", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)
plt.colorbar(label="Potential Depth")

out_img = "general_qwen_wilderness_map.png"
plt.savefig(out_img, bbox_inches="tight")
plt.close()
print(f"\nSaved General Qwen wilderness map to: {out_img}")
