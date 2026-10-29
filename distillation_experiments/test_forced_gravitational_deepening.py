"""
test_forced_gravitational_deepening.py
--------------------------------------
高密度領域の局所重力深化（Density-Driven Gravitational Well Deepening）

【ユーザー着想の数理化】
「集まっているところにわずかに引力を強めるようにして、強制的に深く掘る」
- 局所密度 ρ_i に比例して、集まったノード同士の引力とポテンシャル深度を非線形にブースト。
- 全体のブラックホール崩壊を防ぐため、体積排除斥力との絶妙な平衡を保ちつつ、
  アトラクターの谷底だけを鋭利に深掘り（Sharpening）する。
"""

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

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_deepened_wilderness(model_id="Qwen/Qwen2.5-1.5B-Instruct", apply_deepening=False, boost_beta=0.12, sample_k=4000):
    mode_str = "【強制深掘り適用 (Deepened)】" if apply_deepening else "【通常 (Standard)】"
    print(f"\n[*] 思考原野を構築中: {mode_str} ...")

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()

    W_embed = model.get_input_embeddings().weight.data.float()
    vocab_size = W_embed.shape[0]

    clean_ids = []
    for tid in range(vocab_size):
        s = tok.decode([tid]).strip()
        if len(s) >= 2 and s.isalpha():
            clean_ids.append(tid)

    sample_ids = clean_ids[:sample_k] if len(clean_ids) > sample_k else clean_ids
    sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
    W_clean = F.normalize(W_embed[sample_ids_t].to(device), p=2, dim=-1)

    U, S, V = torch.pca_lowrank(W_clean, q=2)
    coords = (U * S).cpu().numpy()
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6) * 10.0

    pos = torch.tensor(coords, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([pos], lr=0.1)

    for step in range(60):
        opt.zero_grad()
        diff = pos.unsqueeze(0) - pos.unsqueeze(1) # (N, N, 2)
        dist = torch.norm(diff, dim=-1) + 1e-6     # (N, N)

        # ガウシアン近接度
        gauss_kernel = torch.exp(- (dist**2) / (2.0 * 2.0**2))

        if apply_deepening:
            # 局所密度 ρ_i の計算 (自分以外の近傍の集まり度合い)
            density = gauss_kernel.sum(dim=-1, keepdim=True) # (N, 1)
            # 密度が高いペアほど引力をわずかにブースト (1.0 + β * (ρ_i * ρ_j)^0.25)
            density_boost = 1.0 + boost_beta * torch.sqrt(density * density.T).clamp(max=4.0)
            v_attr = - (density_boost * gauss_kernel).sum()
        else:
            v_attr = - gauss_kernel.sum()

        # 斥力 (体積排除: ブラックホール化を阻止)
        v_rep = (0.8 / dist).clamp(max=10.0).sum()

        loss = v_attr + v_rep
        loss.backward()
        opt.step()

    final_pos = pos.detach().cpu().numpy()

    # ポテンシャル面の計算 (グリッド)
    grid_x, grid_y = np.meshgrid(np.linspace(-25, 25, 200), np.linspace(-25, 25, 200))
    pot = np.zeros_like(grid_x)

    # 各粒子の局所密度に応じた谷掘り深さ
    sub_pos = final_pos[:800]
    p_diff = sub_pos[:, np.newaxis, :] - sub_pos[np.newaxis, :, :]
    p_dist = np.linalg.norm(p_diff, axis=-1)
    p_density = np.exp(- (p_dist**2) / (2.0 * 2.0**2)).sum(axis=-1)

    for idx, p in enumerate(sub_pos):
        r2 = (grid_x - p[0])**2 + (grid_y - p[1])**2
        depth_weight = 1.0
        if apply_deepening:
            # 密集している粒子ほどポテンシャルを深く彫刻する
            depth_weight = 1.0 + 0.15 * min(5.0, p_density[idx])
        pot -= depth_weight * np.exp(-r2 / (2 * 1.5**2))

    min_pot = float(pot.min())
    mean_pot = float(pot.mean())
    std_pot = float(pot.std())
    diffs = final_pos[:400, np.newaxis, :] - final_pos[np.newaxis, :400, :]
    mean_dist = float(np.mean(np.linalg.norm(diffs, axis=-1)))

    print(f"  ・最深谷深度 (Min Potential): {min_pot:.3f}")
    print(f"  ・平均ポテンシャル: {mean_pot:.3f}")
    print(f"  ・起伏度 (Ruggedness STD): {std_pot:.3f}")
    print(f"  ・粒子間平均距離: {mean_dist:.3f}")

    del model
    del W_embed
    torch.cuda.empty_cache()

    metrics = {
        "min_potential": min_pot,
        "mean_potential": mean_pot,
        "ruggedness_std": std_pot,
        "mean_particle_distance": mean_dist
    }

    return final_pos, grid_x, grid_y, pot, metrics

def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. 通常の思考原野 (Standard)
    pos_std, gx_s, gy_s, pot_std, m_std = build_deepened_wilderness(model_id, apply_deepening=False)

    # 2. 強制重力深化・思考原野 (Deepened)
    pos_deep, gx_d, gy_d, pot_deep, m_deep = build_deepened_wilderness(model_id, apply_deepening=True, boost_beta=0.15)

    # 並列可視化 (ビフォーアフター)
    print("\n[Rendering Forced Deepening Comparison Plot]...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

    # Standard
    cs1 = axes[0].contourf(gx_s, gy_s, pot_std, levels=35, cmap="viridis")
    axes[0].scatter(pos_std[:350, 0], pos_std[:350, 1], s=7, color="magenta", alpha=0.6)
    axes[0].set_title(f"1. Standard Wilderness (Base Qwen-1.5B)\nDepth: {m_std['min_potential']:.1f} | Ruggedness: {m_std['ruggedness_std']:.2f}", fontsize=12, weight="bold")
    axes[0].set_xlabel("Dimension X", fontsize=10)
    axes[0].set_ylabel("Dimension Y", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(cs1, ax=axes[0], label="Potential Depth")

    # Deepened
    cs2 = axes[1].contourf(gx_d, gy_d, pot_deep, levels=35, cmap="viridis")
    axes[1].scatter(pos_deep[:350, 0], pos_deep[:350, 1], s=7, color="cyan", alpha=0.6)
    axes[1].set_title(f"2. Forced Gravitational Deepening (Density Boost β=0.15)\nDepth: {m_deep['min_potential']:.1f} | Ruggedness: {m_deep['ruggedness_std']:.2f}", fontsize=12, weight="bold")
    axes[1].set_xlabel("Dimension X", fontsize=10)
    axes[1].set_ylabel("Dimension Y", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(cs2, ax=axes[1], label="Potential Depth")

    plt.tight_layout()
    out_png = "distillation_experiments/forced_deepening_wilderness_comparison.png"
    plt.savefig(out_png, dpi=300)
    print(f"\n[+] 強制深掘り比較マップを保存しました: {out_png}")

    # アーティファクトへコピー
    import shutil
    art_path = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/forced_deepening_wilderness_comparison.png"
    shutil.copyfile(out_png, art_path)
    print(f"[+] アーティファクトディレクトリにコピー完了: {art_path}")

    # メトリクスJSON
    with open("distillation_experiments/forced_deepening_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "standard": m_std,
            "deepened": m_deep,
            "depth_improvement_percent": round((m_deep['min_potential'] - m_std['min_potential']) / abs(m_std['min_potential']) * 100, 2),
            "ruggedness_improvement_percent": round((m_deep['ruggedness_std'] - m_std['ruggedness_std']) / m_std['ruggedness_std'] * 100, 2)
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
