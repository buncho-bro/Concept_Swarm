"""
zero_vram_great_circle_genesis.py
=================================
0 VRAM 全語彙大円干渉 自律積層概念地形エンジン
(Zero-VRAM Great-Circle Interference Stacked Terrain Genesis)

【設計思想】
1. 人間の恣意的な単語選定（辞書アンカー・文章アンカー）を 100% 排除。
2. 教師または対象モデルの全語彙（例: 151,936 トークン）の埋め込み層を
   ShallowWeights (mmap / CPU) で参照し、VRAM 消費ゼロ (0 MB) で動作。
3. 全トークン空間上で大円干渉場 Phi(x) を張り、球面リーマン勾配上昇により、
   全言語知識から自律的に極大干渉点（真のイデア核 = Layer 0 吸収点群）を析出。
4. Layer 0 吸収点 ➔ Layer 1 初期配置 ➔ Layer 2 自律成長 の純粋幾何カスケード。
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

# パス解決: concept_core の shallow_extractor をインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'concept_core')))
from shallow_extractor import ShallowWeights

class ZeroVRAMVocabGreatCircleGenesis:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", verbose=True):
        self.model_id = model_id
        self.verbose = verbose
        self._log(f"Initializing Zero-VRAM ShallowWeights for {model_id}...")
        t0 = time.perf_counter()
        self.sw = ShallowWeights(model_id=model_id, verbose=verbose)
        self.vocab_size = self.sw.vocab_rows
        self.dim = self.sw.dim
        self._log(f"Loaded ShallowWeights: {self.vocab_size} tokens, dim={self.dim} ({time.perf_counter()-t0:.2f}s, 0 MB VRAM)")

    def _log(self, msg):
        if self.verbose:
            print(f"[ZeroVRAM-Genesis] {msg}")

    def load_full_vocab_vectors(self, chunk=16384):
        """全語彙の単位ベクトル V と接方向 U を CPU メモリに展開 (約 500 MB)"""
        self._log(f"Extracting all {self.vocab_size} vocabulary unit vectors (0 MB VRAM)...")
        t0 = time.perf_counter()
        
        all_vecs = []
        with self.sw._open() as f:
            sl = f.get_slice(self.sw.embed_key)
            for s in range(0, self.vocab_size, chunk):
                e = min(s + chunk, self.vocab_size)
                raw_chunk = sl[s:e].float().numpy()
                norm_chunk = self.sw.preprocess_block(raw_chunk)
                all_vecs.append(norm_chunk.astype(np.float32))
                
        V = np.vstack(all_vecs) # (N, D) float32
        
        # 多軸主成分を用いた多様な大円接方向 U の構築
        self._log("Building diverse multi-axis context tangent field U (Great-Circle span)...")
        rng = np.random.default_rng(42)
        # SVD による上位 16 主軸を抽出
        sub_sample = V[rng.choice(len(V), size=min(10000, len(V)), replace=False)]
        _, _, Vt = np.linalg.svd(sub_sample, full_matrices=False) # (D, D)
        top_axes = Vt[:16] # (16, D)
        
        # 各トークンに割り当てる主変動方向: 近傍軸の投影
        # 各トークンにランダムな上位主軸の混合を割り当てることで空間全体の直交大円を形成
        weights = rng.standard_normal((len(V), 16), dtype=np.float32)
        weights /= np.linalg.norm(weights, axis=1, keepdims=True)
        raw_U = weights @ top_axes # (N, D)
        
        # v_i に直交化: U_i = raw_U_i - (raw_U_i . v_i) * v_i
        proj = np.sum(raw_U * V, axis=1, keepdims=True) * V
        U = raw_U - proj
        norm_u = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(norm_u, 1e-12)
        
        self._log(f"Vocab Field Ready: V={V.shape}, U={U.shape} in {time.perf_counter()-t0:.2f}s")
        return V, U

    def extract_layer0_absorbers(self, V, U, n_seeds=64, ascent_steps=25, sigma=0.22, merge_sim=0.85):
        """
        全15万語の大円干渉場 Phi(x) から、球面リーマン勾配上昇で Layer 0 吸収点群を抽出。
        """
        self._log(f"--- [Layer 0] Generating Absorbers from 150k Vocab (n_seeds={n_seeds}, sigma={sigma:.2f}) ---")
        t0 = time.perf_counter()
        N, D = V.shape
        rng = np.random.default_rng(1234)

        # 多様な初期シード候補 (SVD 主軸方向 + 直交ランダム)
        seeds = []
        sub_sample = V[rng.choice(N, size=min(10000, N), replace=False)]
        _, _, Vt = np.linalg.svd(sub_sample, full_matrices=False)
        for i in range(min(n_seeds // 2, Vt.shape[0])):
            seeds.append(Vt[i])
        while len(seeds) < n_seeds:
            v_rnd = rng.standard_normal(D).astype(np.float32)
            v_rnd /= np.linalg.norm(v_rnd)
            seeds.append(v_rnd)
        
        X = np.array(seeds, dtype=np.float32) # (M, D)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        step = 0.4 * sigma
        for step_idx in range(ascent_steps):
            A = X @ V.T # (M, N) cos(x, v_i)
            B = X @ U.T # (M, N) cos(x, u_i)
            
            r_sq = np.clip(A * A + B * B, 0.0, 1.0)
            r = np.sqrt(r_sq)
            theta = np.arccos(np.clip(r, -1.0, 1.0))
            
            w = np.exp(- (theta ** 2) / (2.0 * (sigma ** 2))) # (M, N)
            phi = np.sum(w, axis=1)
            
            s = np.sqrt(np.clip(1.0 - r_sq, 0.0, None))
            ratio = np.where(s > 1e-6, theta / np.maximum(s, 1e-12), 1.0)
            c = w * ratio / (sigma * sigma * np.maximum(r, 1e-12))
            
            G = (c * A) @ V + (c * B) @ U
            
            proj = np.sum(G * X, axis=1, keepdims=True) * X
            tangent_G = G - proj
            norm_g = np.linalg.norm(tangent_G, axis=1, keepdims=True)
            
            X_new = X + step * tangent_G / np.maximum(norm_g, 1e-12)
            X = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)

        # 3. 角度クラスタリング (重複イデアの統合)
        unique_absorbers = []
        unique_potentials = []
        # ポテンシャルが高い順にソート
        final_A = X @ V.T
        final_B = X @ U.T
        final_r = np.clip(np.sqrt(final_A**2 + final_B**2), 0.0, 1.0)
        final_theta = np.arccos(final_r)
        final_phi = np.sum(np.exp(- (final_theta ** 2) / (2.0 * (sigma ** 2))), axis=1)
        
        sorted_indices = np.argsort(-final_phi)
        for idx in sorted_indices:
            cand = X[idx]
            if len(unique_absorbers) == 0:
                unique_absorbers.append(cand)
                unique_potentials.append(final_phi[idx])
            else:
                existing = np.stack(unique_absorbers)
                sims = cand @ existing.T
                if np.max(sims) < merge_sim:
                    unique_absorbers.append(cand)
                    unique_potentials.append(final_phi[idx])

        A0 = np.stack(unique_absorbers) # (K0, D)
        self._log(f"Layer 0 Absorbers extracted: {len(A0)} focal points ({time.perf_counter()-t0:.2f}s)")
        return A0, unique_potentials

    def build_stacked_terrains(self, V, U):
        """
        0 VRAM 全語彙大円干渉 ➔ Layer 0 吸収点 ➔ Layer 1 自律積層 ➔ Layer 2 自律積層
        """
        self._log("=== Starting 0-VRAM Full-Vocab Great-Circle Stacked Genesis ===")
        t_start = time.perf_counter()
        
        # 1. Layer 0 吸収点群 (全15万語からの干渉極大点)
        A0, phi0 = self.extract_layer0_absorbers(V, U, n_seeds=64, ascent_steps=25, sigma=0.22, merge_sim=0.85)
        
        # 2. Layer 1 吸収点群: A₀ を初期配置 X₁ として大円干渉自律成長
        self._log(f"--- [Layer 1] Cascading from A0 ({len(A0)} points) to Relations Landscape ---")
        A1 = self._cascade_layer(A0, sigma=0.28, steps=25, merge_sim=0.82)
        
        # 3. Layer 2 吸収点群: A₁ を初期配置 X₂ として大円干渉自律成長
        self._log(f"--- [Layer 2] Cascading from A1 ({len(A1)} points) to Integration Landscape ---")
        A2 = self._cascade_layer(A1, sigma=0.34, steps=25, merge_sim=0.78)
        
        total_time = time.perf_counter() - t_start
        self._log(f"=== Stacked Terrains Built in {total_time:.2f}s (0 MB VRAM) ===")
        self._log(f"  ・Layer 0 (素イデア核): {len(A0)} 点")
        self._log(f"  ・Layer 1 (因果・関係核): {len(A1)} 点")
        self._log(f"  ・Layer 2 (統合・大域核): {len(A2)} 点")
        
        return A0, A1, A2

    def _cascade_layer(self, prev_absorbers, sigma=0.40, steps=25, merge_sim=0.82):
        """前層の吸収点群を初期配置として、次の層の地形を自律成長させる"""
        points = prev_absorbers.copy()
        landscape = prev_absorbers.copy()
        lr = 0.08
        
        for step in range(steps):
            # 点同士の球面ポテンシャル滑走
            sims = points @ landscape.T
            dist_sq = 2.0 - 2.0 * np.clip(sims, -1.0, 1.0)
            pulls = np.exp(- dist_sq / (2.0 * (sigma ** 2)))
            
            weights = pulls / (np.sum(pulls, axis=1, keepdims=True) + 1e-7)
            target_dirs = weights @ landscape
            
            proj = np.sum(target_dirs * points, axis=-1, keepdims=True) * points
            tangent = target_dirs - proj
            norm_t = np.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent = tangent / np.maximum(norm_t, 1e-7)
            
            points = points + lr * tangent
            points /= np.linalg.norm(points, axis=-1, keepdims=True)
            
        unique = []
        for p in points:
            if len(unique) == 0:
                unique.append(p)
            else:
                existing = np.stack(unique)
                if np.max(p @ existing.T) < merge_sim:
                    unique.append(p)
                    
        return np.stack(unique)

    def inspect_absorber_concepts(self, A0, top_k=5):
        """自律析出した吸収点群が、15万語のどのトークンと最も強く共鳴しているか検証"""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model_id)
        
        self._log("\n🔍 [Inspection] Top Resonating Tokens for Layer 0 Absorbers (辞書指定ゼロ):")
        # 吸収点 A0 (K, D) と 全語彙の類似度を計算
        all_ids = []
        results = []
        
        # チャンク計算でトップトークンを探す
        for k_idx, a in enumerate(A0):
            best_sims = []
            best_tokens = []
            
            # 全語彙との内積
            with self.sw._open() as f:
                sl = f.get_slice(self.sw.embed_key)
                chunk = 16384
                scores = []
                for s in range(0, self.vocab_size, chunk):
                    e = min(s + chunk, self.vocab_size)
                    raw_chunk = sl[s:e].float().numpy()
                    norm_chunk = self.sw.preprocess_block(raw_chunk)
                    dot = norm_chunk @ a
                    scores.append(dot)
                all_dots = np.concatenate(scores)
                
            top_indices = np.argsort(-all_dots)[:top_k]
            tokens_str = [tok.decode([idx]).strip().replace('\n', '\\n') for idx in top_indices]
            sims_str = [f"{all_dots[idx]:.3f}" for idx in top_indices]
            
            token_summary = ", ".join([f"'{t}' ({s})" for t, s in zip(tokens_str, sims_str)])
            print(f"  イデア核 #{k_idx+1} [共鳴値 {all_dots[top_indices[0]]:.3f}]: {token_summary}")
            results.append((k_idx, tokens_str, all_dots[top_indices]))
            
        return results

if __name__ == "__main__":
    genesis = ZeroVRAMVocabGreatCircleGenesis(model_id="Qwen/Qwen2.5-0.5B-Instruct", verbose=True)
    V, U = genesis.load_full_vocab_vectors()
    A0, A1, A2 = genesis.build_stacked_terrains(V, U)
    genesis.inspect_absorber_concepts(A0, top_k=6)
    print("\n✅ Zero-VRAM Great-Circle Stacked Genesis Test Completed Successfully!")
