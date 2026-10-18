"""
distill_glm53_zero_vram_great_circle.py
=======================================
GLM-5.3 (zai-org/GLM-5.3) 0 VRAM 全語彙大円干渉 真の自律概念核抽出

【核心】
・過去の25単語辞書アンカー (Function, Variable, Loop...) を完全撤廃。
・GLM-5.3 の全 154,880 トークン × 6,144 次元埋め込み全体から、
  0 MB VRAM (CPU/mmap) で大円干渉場を構築。
・全宇宙の干渉極大点 (真のイデア核 = Layer 0 吸収点群) を自律析出させ、
  Layer 1 (因果核) ➔ Layer 2 (統合核) へと自律積層。
・結果を保存し、Student モデルへの同期・蒸留に備える。
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

# パス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'concept_core')))
from shallow_extractor import ShallowWeights
from transformers import AutoTokenizer

def run_glm53_zero_vram_extraction():
    model_id = "zai-org/GLM-5.3"
    print("=" * 70)
    print(f"🌌 GLM-5.3 (6,144次元) 0 VRAM 全語彙大円干渉 自律概念核抽出")
    print("   ※ 辞書アンカー・単語指定 100% 排除 / 全 154,880 トークン直接解析")
    print("=" * 70)

    # 1. 0 VRAM 埋め込み層アクセス
    t0 = time.perf_counter()
    print("\n[Step 1] Loading GLM-5.3 embedding weights via ShallowWeights (mmap / 0 MB VRAM)...")
    sw = ShallowWeights(model_id=model_id, verbose=True)
    vocab_size = sw.vocab_rows
    dim = sw.dim
    print(f"✅ Loaded: {vocab_size} tokens, {dim} dimensions (Elapsed: {time.perf_counter()-t0:.2f}s, 0 MB VRAM)")

    # 2. 全語彙ベクトルの展開 (CPU メモリ上)
    t1 = time.perf_counter()
    print("\n[Step 2] Extracting and normalizing all 154,880 token vectors (Chunked, CPU)...")
    chunk_size = 16384
    all_vecs = []
    with sw._open() as f:
        sl = f.get_slice(sw.embed_key)
        for s in range(0, vocab_size, chunk_size):
            e = min(s + chunk_size, vocab_size)
            raw = sl[s:e].float().numpy()
            norm_b = sw.preprocess_block(raw)
            all_vecs.append(norm_b.astype(np.float32))
            if (s // chunk_size) % 3 == 0 or e == vocab_size:
                print(f"  -> Processed {e}/{vocab_size} tokens...")

    V = np.vstack(all_vecs) # (154880, 6144)
    print(f"✅ Vocabulary Matrix V ready: shape={V.shape}, footprint={V.nbytes / 1e6:.1f} MB ({time.perf_counter()-t1:.2f}s)")

    # 3. 6,144次元空間における多軸大円接方向 U の構築
    t2 = time.perf_counter()
    print("\n[Step 3] Constructing Multi-Axis Context Tangent Field U for Great-Circle span...")
    rng = np.random.default_rng(42)
    # 6144次元空間の主軸 (SVD) を抽出
    sub_sample = V[rng.choice(len(V), size=min(12000, len(V)), replace=False)]
    _, _, Vt = np.linalg.svd(sub_sample, full_matrices=False)
    top_axes = Vt[:16] # 上位16主軸 (16, 6144)

    # 各トークンに割り当てる大円接平面
    weights = rng.standard_normal((len(V), 16), dtype=np.float32)
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)
    raw_U = weights @ top_axes

    # 直交化: U_i = raw_U_i - (raw_U_i . v_i) * v_i
    proj = np.sum(raw_U * V, axis=1, keepdims=True) * V
    U = raw_U - proj
    norm_u = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norm_u, 1e-12)
    print(f"✅ Tangent Field U ready: shape={U.shape} ({time.perf_counter()-t2:.2f}s)")

    # 4. 全大円干渉 球面リーマン勾配上昇 (Layer 0 吸収点群の自律析出)
    t3 = time.perf_counter()
    n_seeds = 64
    ascent_steps = 25
    sigma = 0.22
    merge_sim = 0.85
    print(f"\n[Step 4] Running Great-Circle Riemannian Gradient Ascent (n_seeds={n_seeds}, sigma={sigma:.2f})...")

    # 初期シード候補 (SVD 主軸 + 直交ランダム)
    seeds = []
    for i in range(min(n_seeds // 2, Vt.shape[0])):
        seeds.append(Vt[i])
    while len(seeds) < n_seeds:
        v_rnd = rng.standard_normal(dim).astype(np.float32)
        v_rnd /= np.linalg.norm(v_rnd)
        seeds.append(v_rnd)
    
    X = np.array(seeds, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    step = 0.4 * sigma
    for s_idx in range(ascent_steps):
        # A: (M, N), B: (M, N) -> (64, 154880) float32 = 39 MB
        A = X @ V.T
        B = X @ U.T
        
        r_sq = np.clip(A * A + B * B, 0.0, 1.0)
        r = np.sqrt(r_sq)
        theta = np.arccos(np.clip(r, -1.0, 1.0))
        
        w = np.exp(- (theta ** 2) / (2.0 * (sigma ** 2)))
        phi = np.sum(w, axis=1)
        
        s = np.sqrt(np.clip(1.0 - r_sq, 0.0, None))
        ratio = np.where(s > 1e-6, theta / np.maximum(s, 1e-12), 1.0)
        c = w * ratio / (sigma * sigma * np.maximum(r, 1e-12))
        
        G = (c * A) @ V + (c * B) @ U
        
        proj_g = np.sum(G * X, axis=1, keepdims=True) * X
        tangent_G = G - proj_g
        norm_g = np.linalg.norm(tangent_G, axis=1, keepdims=True)
        
        X_new = X + step * tangent_G / np.maximum(norm_g, 1e-12)
        X = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)

    # 重複イデアのクラスタリング
    final_A = X @ V.T
    final_B = X @ U.T
    final_r = np.clip(np.sqrt(final_A**2 + final_B**2), 0.0, 1.0)
    final_theta = np.arccos(final_r)
    final_phi = np.sum(np.exp(- (final_theta ** 2) / (2.0 * (sigma ** 2))), axis=1)

    unique_absorbers = []
    unique_potentials = []
    for idx in np.argsort(-final_phi):
        cand = X[idx]
        if len(unique_absorbers) == 0:
            unique_absorbers.append(cand)
            unique_potentials.append(final_phi[idx])
        else:
            existing = np.stack(unique_absorbers)
            if np.max(cand @ existing.T) < merge_sim:
                unique_absorbers.append(cand)
                unique_potentials.append(final_phi[idx])

    A0 = np.stack(unique_absorbers) # (K0, 6144)
    print(f"✅ Layer 0 Absorbers extracted: {len(A0)} focal points ({time.perf_counter()-t3:.2f}s)")

    # 5. 自律積層カスケード (Layer 1 ➔ Layer 2)
    t4 = time.perf_counter()
    print("\n[Step 5] Cascading to Layer 1 (Relations) and Layer 2 (Integration)...")
    
    def cascade(prev, sig, m_sim):
        pts = prev.copy()
        ls = prev.copy()
        for _ in range(25):
            sims = pts @ ls.T
            dist_sq = 2.0 - 2.0 * np.clip(sims, -1.0, 1.0)
            pulls = np.exp(- dist_sq / (2.0 * (sig ** 2)))
            weights_c = pulls / (np.sum(pulls, axis=1, keepdims=True) + 1e-7)
            t_dirs = weights_c @ ls
            pr = np.sum(t_dirs * pts, axis=-1, keepdims=True) * pts
            tan = t_dirs - pr
            tan /= np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-7)
            pts = pts + 0.08 * tan
            pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
        uq = []
        for p in pts:
            if len(uq) == 0:
                uq.append(p)
            else:
                if np.max(p @ np.stack(uq).T) < m_sim:
                    uq.append(p)
        return np.stack(uq)

    A1 = cascade(A0, sig=0.28, m_sim=0.82)
    A2 = cascade(A1, sig=0.34, m_sim=0.78)
    print(f"✅ Layer 1 Absorbers: {len(A1)} points")
    print(f"✅ Layer 2 Absorbers: {len(A2)} points ({time.perf_counter()-t4:.2f}s)")

    # 6. GLM-5.3 のトークナイザーで共鳴イデアを可視化
    print("\n" + "=" * 70)
    print(f"🔍 [GLM-5.3 脳内イデアの可視化] トップ共鳴トークン一覧 (辞書指定ゼロ):")
    print("=" * 70)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    for k_idx, a in enumerate(A0):
        # 全語彙との類似度
        dots = V @ a
        top_k = 6
        top_indices = np.argsort(-dots)[:top_k]
        
        tok_strs = []
        for tid in top_indices:
            try:
                s = tok.decode([tid]).strip().replace('\n', '\\n')
                if not s:
                    s = f"token_{tid}"
            except Exception:
                s = f"token_{tid}"
            tok_strs.append(f"'{s}' ({dots[tid]:.3f})")
            
        print(f"  イデア核 #{k_idx+1:02d} [共鳴強度 {dots[top_indices[0]]:.3f}]: {', '.join(tok_strs)}")

    # 7. マスター多層概念地形の保存
    save_path = "glm5.3_zero_vram_pure_concept_terrains.pt"
    save_dict = {
        "model_id": model_id,
        "vocab_size": vocab_size,
        "dim": dim,
        "Layer_0": torch.tensor(A0, dtype=torch.float32),
        "Layer_1": torch.tensor(A1, dtype=torch.float32),
        "Layer_2": torch.tensor(A2, dtype=torch.float32),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(save_dict, save_path)
    print(f"\n💾 Saved GLM-5.3 Pure Stacked Terrains to: {save_path}")
    print(f"   ・Layer 0: {A0.shape}")
    print(f"   ・Layer 1: {A1.shape}")
    print(f"   ・Layer 2: {A2.shape}")
    print("\n🎉 GLM-5.3 0-VRAM Full-Vocab Great-Circle Distillation Genesis SUCCESSFUL!")

if __name__ == "__main__":
    run_glm53_zero_vram_extraction()
