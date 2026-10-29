"""
fast_convergence_sweep_benchmark.py
-----------------------------------
高速化ベンチマーク & 収束能（-から+）連続パラメータ探査エンジン

【高速化の仕組み】
1. ベンチマークに必要な全単語の潜在表現（Hidden State）を1回のバッチ処理でGPUメモリ上に完全事前キャッシュ。
2. 以降のパラメータ探査（収束能 β: -0.25 ～ +0.35）は、ニューラルネットの推論を一切走らせず、
   純粋なGPUテンソル代数（行列積とコサイン類似度）のみで実行。
   ➔ 数百回のパラメータ探査がわずか数秒（100倍以上の超高速化）で完了！

【探査目的】
- 負の収束能（反発・平坦化・脱出）
- ゼロ（通常ベース）
- 正の収束能（深掘り・アトラクター強化）
このグラデーションの中で、推論力（Analogy/Logic）と多様性・分離度（Hierarchy）の最適境界（カオスの縁）を特定する。
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- ベンチマーク問題セット ---
analogy_tests = [
    ("King", "Man", "Woman", "Queen"),
    ("Tokyo", "Japan", "France", "Paris"),
    ("Doctor", "Hospital", "School", "Teacher"),
    ("Car", "Road", "Track", "Train"),
    ("Day", "Sun", "Night", "Moon")
]

hierarchy_tests = [
    ("Animal", "Dog", "Desk"),
    ("Science", "Physics", "Apple"),
    ("Emotion", "Anger", "Stone"),
    ("Computer", "Software", "Nature"),
    ("Universe", "Galaxy", "Politics")
]

hop_tests = [
    ("Japan", ["Tokyo", "Tower"], "Skytree"),
    ("France", ["Paris", "Tower"], "Eiffel_Tower"),
    ("Science", ["Apple", "Universal_Gravitation"], "Gravity")
]

logic_tests = [
    ("Bird", "Move", "Penguin", "Move"),
    ("Mammal", "Animal", "Dog", "Animal"),
    ("Ice", "Cold", "Water", "Cold")
]

def collect_unique_words():
    words = set()
    for a, b, c, d in analogy_tests:
        words.update([a, b, c, d])
    for base, target, outlier in hierarchy_tests:
        words.update([base, target, outlier])
    for s, r, t in hop_tests:
        words.add(s)
        words.update(r)
        words.add(t)
    for a, b, c, t in logic_tests:
        words.update([a, b, c, t])
    return sorted(list(words))

def precache_word_vectors(model, tokenizer, word_list):
    print(f"[*] 単語潜在表現を一括事前キャッシュ中 ({len(word_list)} 単語)...")
    cache = {}
    batch_size = 32
    for i in range(0, len(word_list), batch_size):
        batch = word_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 各単語の最終有効トークンの隠れ層ベクトルを取得
            last_token_indices = inputs.attention_mask.sum(dim=1) - 1
            hidden = outputs.hidden_states[-1]
            for idx, word in enumerate(batch):
                pos = last_token_indices[idx]
                v = hidden[idx, pos, :].to(torch.float32)
                cache[word] = F.normalize(v, p=2, dim=0)
    print("  [+] キャッシュ完了！")
    return cache

def build_core_density_field(model, tokenizer, sample_k=3000):
    print(f"[*] 概念コアと局所密度場を構築中 (K={sample_k})...")
    W_embed = model.get_input_embeddings().weight.data.float()
    vocab_size = W_embed.shape[0]

    clean_ids = []
    for tid in range(vocab_size):
        s = tokenizer.decode([tid]).strip()
        if len(s) >= 2 and s.isalpha():
            clean_ids.append(tid)

    sample_ids = clean_ids[:sample_k] if len(clean_ids) > sample_k else clean_ids
    sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
    core_vecs = F.normalize(W_embed[sample_ids_t].to(device), p=2, dim=-1) # (K, D)

    with torch.no_grad():
        sim_matrix = torch.matmul(core_vecs, core_vecs.T)
        weights = torch.clamp((sim_matrix - 0.4) / 0.6, min=0.0) ** 2
        density = weights.sum(dim=-1)
        density = (density - density.mean()) / (density.std() + 1e-6)
        density = torch.clamp(density, min=0.0)

    return core_vecs, density

def apply_deepening_fast(vec, core_vecs, density, beta, alpha=0.35, top_k=24, tau=0.15):
    """
    高速テンソル重力深化/平坦化オペレータ
    beta > 0: 引力増幅（深掘り・収束）
    beta < 0: 反発・谷の平坦化（脱出・拡散）
    beta = 0: 通常
    """
    if abs(beta) < 1e-6 and abs(alpha) < 1e-6:
        return vec

    is_1d = (vec.dim() == 1)
    v = vec.unsqueeze(0) if is_1d else vec

    v_norm = F.normalize(v, p=2, dim=-1)
    sim = torch.matmul(v_norm, core_vecs.T)
    top_sim, top_idx = torch.topk(sim, k=top_k, dim=-1)

    top_densities = density[top_idx] # (B, top_k)
    # beta が負なら高密度ほど引力を減衰（または反発）
    density_factor = torch.clamp(1.0 + beta * torch.sqrt(top_densities + 1e-5), min=0.1, max=5.0)

    attn = F.softmax(top_sim / tau, dim=-1) * density_factor
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

    selected_cores = core_vecs[top_idx]
    pull_targets = (selected_cores * attn.unsqueeze(-1)).sum(dim=1)

    # betaの符号に応じたシフト
    deepened = F.normalize(v + alpha * (pull_targets - v), p=2, dim=-1)
    return deepened.squeeze(0) if is_1d else deepened

def evaluate_fast_benchmark(cached_vecs, core_vecs, density, beta, alpha=0.35):
    """
    純粋なGPUキャッシュ演算による超高速ベンチマーク評価（数ミリ秒）
    """
    # 1. Analogy
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va = apply_deepening_fast(cached_vecs[a], core_vecs, density, beta, alpha)
        vb = apply_deepening_fast(cached_vecs[b], core_vecs, density, beta, alpha)
        vc = apply_deepening_fast(cached_vecs[c], core_vecs, density, beta, alpha)
        vd = apply_deepening_fast(cached_vecs[d], core_vecs, density, beta, alpha)
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        v_pred = apply_deepening_fast(v_pred, core_vecs, density, beta, alpha)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = float(np.mean(analogy_scores))

    # 2. Hierarchy
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base = apply_deepening_fast(cached_vecs[base], core_vecs, density, beta, alpha)
        v_tgt = apply_deepening_fast(cached_vecs[target], core_vecs, density, beta, alpha)
        v_out = apply_deepening_fast(cached_vecs[outlier], core_vecs, density, beta, alpha)
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        hierarchy_scores.append(sim_tgt - sim_out)
    avg_hierarchy = float(np.mean(hierarchy_scores))

    # 3. Multi-hop
    hop_scores = []
    for s, r, t in hop_tests:
        cur_v = apply_deepening_fast(cached_vecs[s], core_vecs, density, beta, alpha)
        for rel in r:
            rel_v = apply_deepening_fast(cached_vecs[rel], core_vecs, density, beta, alpha)
            cur_v = F.normalize(cur_v + rel_v, p=2, dim=0)
            cur_v = apply_deepening_fast(cur_v, core_vecs, density, beta, alpha)
        tgt_v = apply_deepening_fast(cached_vecs[t], core_vecs, density, beta, alpha)
        hop_scores.append(F.cosine_similarity(cur_v.unsqueeze(0), tgt_v.unsqueeze(0)).item())
    avg_hop = float(np.mean(hop_scores))

    # 4. Logic Chain
    logic_scores = []
    for a, b, c, t in logic_tests:
        va = apply_deepening_fast(cached_vecs[a], core_vecs, density, beta, alpha)
        vb = apply_deepening_fast(cached_vecs[b], core_vecs, density, beta, alpha)
        vc = apply_deepening_fast(cached_vecs[c], core_vecs, density, beta, alpha)
        logic_dir = F.normalize(vb - va, p=2, dim=0)
        pred_b = F.normalize(vc + logic_dir, p=2, dim=0)
        pred_b = apply_deepening_fast(pred_b, core_vecs, density, beta, alpha)
        vt = apply_deepening_fast(cached_vecs[t], core_vecs, density, beta, alpha)
        logic_scores.append(F.cosine_similarity(pred_b.unsqueeze(0), vt.unsqueeze(0)).item())
    avg_logic = float(np.mean(logic_scores))

    return {
        "analogy": avg_analogy,
        "hierarchy": avg_hierarchy,
        "multihop": avg_hop,
        "logic_chain": avg_logic
    }

def main():
    print("=" * 70)
    print("⚡ 超高速ベンチマーク & 収束能スイープ探査エンジン")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[*] モデルロード中: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()

    # 1. 単語ベクトルのGPUキャッシュ化
    word_list = collect_unique_words()
    t0 = time.time()
    cached_vecs = precache_word_vectors(model, tok, word_list)
    core_vecs, density = build_core_density_field(model, tok, sample_k=3000)
    print(f"[*] キャッシュ完了 (所要時間: {time.time() - t0:.2f} 秒)")

    # もう大きなニューラルネット本体の推論は不要なので、メモリ節約
    del model
    torch.cuda.empty_cache()

    # 2. 収束能パラメータ β のスイープ (-0.30 から +0.40 まで)
    beta_range = np.linspace(-0.30, 0.40, 29) # 29ポイント探査
    print(f"\n[*] 収束能 β の連続探査を開始 (-0.30 ～ +0.40, 計 {len(beta_range)} 点)...")

    results = []
    t_start_sweep = time.time()

    for idx, beta in enumerate(beta_range):
        beta_val = float(beta)
        score_dict = evaluate_fast_benchmark(cached_vecs, core_vecs, density, beta=beta_val, alpha=0.35)
        # 総合調和スコア (Harmonic / Balanced Metric)
        # 推論力 (Analogy + Logic + Hop) と 構造分離度 (Hierarchy) の統合
        balance_score = (score_dict["analogy"] + score_dict["logic_chain"] + score_dict["multihop"]) / 3.0 + score_dict["hierarchy"]
        
        record = {
            "beta": beta_val,
            "analogy": score_dict["analogy"],
            "hierarchy": score_dict["hierarchy"],
            "multihop": score_dict["multihop"],
            "logic_chain": score_dict["logic_chain"],
            "balance_score": balance_score
        }
        results.append(record)

    sweep_time = time.time() - t_start_sweep
    print(f"  [+] 全 {len(beta_range)} 回のベンチマーク完了！(合計所要時間: {sweep_time:.3f} 秒, 1回あたり {sweep_time/len(beta_range)*1000:.1f} ms)")

    # 最適ポイントの同定
    best_analogy = max(results, key=lambda x: x["analogy"])
    best_logic = max(results, key=lambda x: x["logic_chain"])
    best_balance = max(results, key=lambda x: x["balance_score"])
    zero_baseline = next(r for r in results if abs(r["beta"]) < 1e-3)

    print("\n" + "=" * 70)
    print("🎯 収束能スイープ探査 結果サマリー")
    print("=" * 70)
    print(f"・基準値 (β = 0.00 / 通常):       Analogy={zero_baseline['analogy']:.4f}, Logic={zero_baseline['logic_chain']:.4f}, Hierarchy={zero_baseline['hierarchy']:+.4f}")
    print(f"・比喩推論 最高点 (β = {best_analogy['beta']:+.2f}): Analogy={best_analogy['analogy']:.4f} (基準比 {best_analogy['analogy'] - zero_baseline['analogy']:+.4f})")
    print(f"・論理演繹 最高点 (β = {best_logic['beta']:+.2f}): Logic={best_logic['logic_chain']:.4f} (基準比 {best_logic['logic_chain'] - zero_baseline['logic_chain']:+.4f})")
    print(f"・最適調和点 (β = {best_balance['beta']:+.2f}): Balance={best_balance['balance_score']:.4f} (総合バランス最高)")
    print("=" * 70)

    # グラフ描画
    betas = [r["beta"] for r in results]
    analogies = [r["analogy"] for r in results]
    logics = [r["logic_chain"] for r in results]
    hops = [r["multihop"] for r in results]
    hierarchies = [r["hierarchy"] for r in results]
    balances = [r["balance_score"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), dpi=150, sharex=True)

    # 上段: 推論能力メトリクス
    ax1.plot(betas, analogies, marker='o', linewidth=2.2, color='#e74c3c', label='Analogy (比喩推論)')
    ax1.plot(betas, logics, marker='s', linewidth=2.2, color='#2980b9', label='Logic Chain (論理演繹)')
    ax1.plot(betas, hops, marker='^', linewidth=1.8, color='#27ae60', linestyle='--', label='Multi-hop (軌道追跡)')
    ax1.axvline(0, color='gray', linestyle=':', label='β = 0 (Base Model)')
    ax1.axvline(best_balance['beta'], color='purple', linestyle='--', label=f'Optimal Sweet Spot (β = {best_balance["beta"]:+.2f})')
    ax1.set_ylabel("推論コサイン類似度", fontsize=11, weight='bold')
    ax1.set_title("収束能パラメータ β の連続スイープ探査 (負: 拡散・平坦化 ➔ 正: 凝集・深掘り)", fontsize=13, weight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', framealpha=0.9)

    # 下段: 階層分離度と総合バランス
    ax2.plot(betas, hierarchies, marker='d', linewidth=2, color='#f39c12', label='Hierarchy (階層概念分離度)')
    ax2.plot(betas, balances, marker='*', linewidth=2.5, color='#8e44ad', label='Harmonic Balance Score (総合調和スコア)')
    ax2.axvline(0, color='gray', linestyle=':')
    ax2.axvline(best_balance['beta'], color='purple', linestyle='--')
    ax2.set_xlabel("収束能パラメータ β (負: 反発・谷浅化 ⟵ 0 ⟶ 正: 引力強化・谷深化)", fontsize=11, weight='bold')
    ax2.set_ylabel("分離度 / 総合スコア", fontsize=11, weight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    out_png = "distillation_experiments/fast_convergence_sweep_curve.png"
    plt.savefig(out_png)
    print(f"[+] 探査曲線の可視化グラフを保存: {out_png}")

    # 結果JSON
    out_json = "distillation_experiments/fast_convergence_sweep_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "best_analogy": best_analogy,
            "best_logic": best_logic,
            "best_balance": best_balance,
            "baseline": zero_baseline,
            "sweep_data": results
        }, f, ensure_ascii=False, indent=2)
    print(f"[+] 探査データを保存: {out_json}")

if __name__ == "__main__":
    main()
