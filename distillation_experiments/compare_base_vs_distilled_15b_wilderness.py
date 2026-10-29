"""
compare_base_vs_distilled_15b_wilderness.py
-------------------------------------------
元の Qwen2.5-1.5B vs 蒸留後（7B➔1.5B KNNトポロジカル蒸留）の思考原野比較
（Mistral-7B と同じ形式の 2D ポテンシャル等高線マップによる 3画面並列解析）
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
from peft import PeftModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_wilderness_surface(model_id, lora_path=None, sample_k=4000):
    title_name = model_id
    if lora_path:
        title_name += f" + LoRA ({os.path.basename(lora_path)})"

    print(f"\n========================================================")
    print(f"[*] 思考原野を構築中: {title_name}")
    print(f"========================================================")

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    if lora_path:
        print(f"[*] LoRAアダプタをロード・適用中: {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload() # 重みマージ

    model.eval()

    W_embed = model.get_input_embeddings().weight.data.float()
    vocab_size = W_embed.shape[0]
    print(f"  ・語彙サイズ: {vocab_size}, 埋め込み次元: {W_embed.shape[1]}")

    # 自立概念単語の抽出 (2文字以上、アルファベット)
    clean_ids = []
    for tid in range(vocab_size):
        s = tok.decode([tid]).strip()
        if len(s) >= 2 and s.isalpha():
            clean_ids.append(tid)

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
        v_attr = - torch.exp(- (dist**2) / (2.0 * 2.0**2)).sum()
        v_rep = (0.8 / dist).clamp(max=10.0).sum()
        loss = v_attr + v_rep
        loss.backward()
        opt.step()

    final_pos = pos.detach().cpu().numpy()

    # ポテンシャル面の計算 (グリッド)
    grid_x, grid_y = np.meshgrid(np.linspace(-25, 25, 200), np.linspace(-25, 25, 200))
    pot = np.zeros_like(grid_x)
    for p in final_pos[:800]:
        r2 = (grid_x - p[0])**2 + (grid_y - p[1])**2
        pot -= np.exp(-r2 / (2 * 1.5**2))

    # 地形統計指標の算出
    min_pot = float(pot.min())
    mean_pot = float(pot.mean())
    std_pot = float(pot.std())
    # 粒子間の平均距離
    diffs = final_pos[:400, np.newaxis, :] - final_pos[np.newaxis, :400, :]
    mean_dist = float(np.mean(np.linalg.norm(diffs, axis=-1)))

    print(f"  ・最深谷深度 (Min Potential): {min_pot:.3f}")
    print(f"  ・平均ポテンシャル: {mean_pot:.3f}")
    print(f"  ・ポテンシャル分散 (起伏度): {std_pot:.3f}")
    print(f"  ・粒子間平均距離 (空間スプレッド): {mean_dist:.3f}")

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
    base_15b_id = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/lora_Qwen2.5_7B_to_Qwen2.5_1.5B_Instruct_Topological"
    mistral_id = "mistralai/Mistral-7B-v0.3"

    sample_k = 4000

    # 1. 元の 1.5B
    pos_base, gx_b, gy_b, pot_base, m_base = build_wilderness_surface(base_15b_id, lora_path=None, sample_k=sample_k)

    # 2. 7B➔1.5B KNNトポロジカル蒸留後
    pos_dist, gx_d, gy_d, pot_dist, m_dist = build_wilderness_surface(base_15b_id, lora_path=lora_path, sample_k=sample_k)

    # 3. Mistral-7B (参考基準)
    pos_mis, gx_m, gy_m, pot_mis, m_mis = build_wilderness_surface(mistral_id, lora_path=None, sample_k=sample_k)

    # 3画面並列可視化 (Mistral比較スクリプトと同じ形式)
    print("\n[Rendering Multi-Model Wilderness Comparison Plot]...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)

    # Panel 1: 元の Qwen2.5-1.5B
    cs1 = axes[0].contourf(gx_b, gy_b, pot_base, levels=35, cmap="viridis")
    axes[0].scatter(pos_base[:350, 0], pos_base[:350, 1], s=7, color="magenta", alpha=0.6)
    axes[0].set_title(f"1. Base Qwen2.5-1.5B (Before Distill)\nDepth: {m_base['min_potential']:.1f} | Ruggedness: {m_base['ruggedness_std']:.2f}", fontsize=12, weight="bold")
    axes[0].set_xlabel("Dimension X", fontsize=10)
    axes[0].set_ylabel("Dimension Y", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(cs1, ax=axes[0], label="Potential Depth")

    # Panel 2: 蒸留後 Qwen2.5-1.5B (KNN Topo)
    cs2 = axes[1].contourf(gx_d, gy_d, pot_dist, levels=35, cmap="viridis")
    axes[1].scatter(pos_dist[:350, 0], pos_dist[:350, 1], s=7, color="cyan", alpha=0.6)
    axes[1].set_title(f"2. Qwen2.5-1.5B + 7B-KNN Distilled (Topological)\nDepth: {m_dist['min_potential']:.1f} | Ruggedness: {m_dist['ruggedness_std']:.2f}", fontsize=12, weight="bold")
    axes[1].set_xlabel("Dimension X", fontsize=10)
    axes[1].set_ylabel("Dimension Y", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(cs2, ax=axes[1], label="Potential Depth")

    # Panel 3: Mistral-7B
    cs3 = axes[2].contourf(gx_m, gy_m, pot_mis, levels=35, cmap="viridis")
    axes[2].scatter(pos_mis[:350, 0], pos_mis[:350, 1], s=7, color="yellow", alpha=0.6)
    axes[2].set_title(f"3. Mistral-7B-v0.3 (Reference Model)\nDepth: {m_mis['min_potential']:.1f} | Ruggedness: {m_mis['ruggedness_std']:.2f}", fontsize=12, weight="bold")
    axes[2].set_xlabel("Dimension X", fontsize=10)
    axes[2].set_ylabel("Dimension Y", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(cs3, ax=axes[2], label="Potential Depth")

    plt.tight_layout()
    out_png = "distillation_experiments/base_vs_distilled_15b_wilderness_comparison.png"
    plt.savefig(out_png, dpi=300)
    print(f"\n[+] 思考原野比較マップを保存しました: {out_png}")

    # アーティファクトへコピー
    import shutil
    art_path = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/base_vs_distilled_15b_wilderness_comparison.png"
    shutil.copyfile(out_png, art_path)
    print(f"[+] アーティファクトディレクトリにコピー完了: {art_path}")

    # レポートJSON
    with open("distillation_experiments/base_vs_distilled_15b_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_15b": m_base,
            "distilled_15b": m_dist,
            "mistral_7b": m_mis
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
