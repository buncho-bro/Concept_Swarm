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

def build_model_wilderness(model_id, sample_k=8000):
    print(f"\n========================================================")
    print(f"Building Wilderness for: {model_id}...")
    print(f"========================================================")
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()
    
    W_embed = model.get_input_embeddings().weight.data.float()
    vocab_size = W_embed.shape[0]
    print(f"Vocab size: {vocab_size}, Embed dim: {W_embed.shape[1]}")
    
    # 自立概念単語の抽出 (2文字以上、アルファベット)
    clean_ids = []
    for tid in range(vocab_size):
        s = tok.decode([tid]).strip()
        if len(s) >= 2 and s.isalpha():
            clean_ids.append(tid)
            
    print(f"Filtered clean concept words: {len(clean_ids)}")
    sample_ids = clean_ids[:sample_k] if len(clean_ids) > sample_k else clean_ids
    sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
    
    W_clean = F.normalize(W_embed[sample_ids_t].to(device), p=2, dim=-1)
    
    # SVDによる最良2D射影
    U, S, V = torch.pca_lowrank(W_clean, q=2)
    coords = (U * S).cpu().numpy()
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6) * 10.0
    
    # 力学緩和 (引力 + 体積排除斥力)
    pos = torch.tensor(coords, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([pos], lr=0.1)
    
    for step in range(60):
        opt.zero_grad()
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1) + 1e-6
        
        # 引力
        v_attr = - torch.exp(- (dist**2) / (2.0 * 2.0**2)).sum()
        # 斥力 (体積排除)
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
        
    # メモリ解放
    del model
    del W_embed
    torch.cuda.empty_cache()
    
    return final_pos, grid_x, grid_y, pot

# 2つの異なる系譜のモデルで比較
pos_qwen, gx_q, gy_q, pot_qwen = build_model_wilderness("Qwen/Qwen2.5-Coder-3B-Instruct", sample_k=6000)
pos_mistral, gx_m, gy_m, pot_mistral = build_model_wilderness("mistralai/Mistral-7B-v0.3", sample_k=6000)

# 並列可視化
print("\n[Rendering Multi-Model Comparison Plot]...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)

# Qwen
cs1 = axes[0].contourf(gx_q, gy_q, pot_qwen, levels=35, cmap="viridis")
axes[0].scatter(pos_qwen[:400, 0], pos_qwen[:400, 1], s=6, color="magenta", alpha=0.6)
axes[0].set_title("Qwen2.5-Coder-3B (Multilingual / Asia Base)\nThought Wilderness Topology", fontsize=13, weight="bold")
axes[0].set_xlabel("Dimension X", fontsize=11)
axes[0].set_ylabel("Dimension Y", fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.3)
fig.colorbar(cs1, ax=axes[0], label="Potential Depth")

# Mistral
cs2 = axes[1].contourf(gx_m, gy_m, pot_mistral, levels=35, cmap="plasma")
axes[1].scatter(pos_mistral[:400, 0], pos_mistral[:400, 1], s=6, color="cyan", alpha=0.6)
axes[1].set_title("Mistral-7B-v0.3 (Western / European Base)\nThought Wilderness Topology", fontsize=13, weight="bold")
axes[1].set_xlabel("Dimension X", fontsize=11)
axes[1].set_ylabel("Dimension Y", fontsize=11)
axes[1].grid(True, linestyle="--", alpha=0.3)
fig.colorbar(cs2, ax=axes[1], label="Potential Depth")

out_file = "cross_model_wilderness_comparison.png"
plt.savefig(out_file, bbox_inches="tight")
plt.close()
print(f"Successfully generated cross-model comparison image: {out_file}")
