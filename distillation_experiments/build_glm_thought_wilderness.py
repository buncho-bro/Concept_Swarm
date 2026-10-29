"""
GLM Thought Wilderness Builder - 0-VRAM GLM-5.3 概念地形からの思考原野生成
-------------------------------------------------------------------------
GLM-5.3 (154,880語 x 6,144次元) から0-VRAM大円干渉で析出された
純粋概念地形データ (glm5.3_zero_vram_pure_concept_terrains.pt) を読み込み、
層間フライトパス（Layer 0 -> Layer 1 -> Layer 2）の幾何学的干渉から、
普遍的な3D思考原野マップ（GLM Thought Wilderness）を自己組織化・可視化する。
"""

import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def build_glm_thought_wilderness():
    data_path = "distillation_experiments/glm5.3_zero_vram_pure_concept_terrains.pt"
    print(f"[*] GLM-5.3 概念地形データをロード中: {data_path}...")
    
    # weights_only=False で安全にロード
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    
    print(f"[+] モデルID: {data['model_id']}")
    print(f"[+] 全語彙規模: {data['vocab_size']:,} 語")
    print(f"[+] 潜在空間次元: {data['dim']} 次元")
    
    # 共通するノード数 (Layer 2 は 53 個)
    num_nodes = min(data['Layer_0'].shape[0], data['Layer_1'].shape[0], data['Layer_2'].shape[0])
    l0 = data['Layer_0'][:num_nodes].float()
    l1 = data['Layer_1'][:num_nodes].float()
    l2 = data['Layer_2'][:num_nodes].float()
    
    print(f"[+] 整合イデア核数: {num_nodes} 個 (Layer 0〜2 連鎖)")

    # 1. 各イデア核のフライトパス（運動ベクトル）の計算
    # Layer 0 -> Layer 1 -> Layer 2 の推進力と曲率
    v01 = l1 - l0
    v12 = l2 - l1
    flightpaths = (v01 + v12)  # (53, 6144)
    flightpaths = flightpaths / (torch.norm(flightpaths, dim=-1, keepdim=True) + 1e-8)
    
    print("[*] イデア核間のフライトパス直交干渉行列（測地線距離）を計算中...")
    # コサイン類似度行列
    sim_matrix = torch.mm(flightpaths, flightpaths.t()).numpy()
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

    # 2. 思考原野（3D / 2D 面マップ）への幾何学的展開
    print("[*] 多次元尺度構成法 (MDS) による思考原野の展開...")
    mds_3d = MDS(n_components=3, dissimilarity="precomputed", random_state=42, n_init=10, max_iter=1000)
    coords_3d = mds_3d.fit_transform(dist_matrix)

    mds_2d = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=10, max_iter=1000)
    coords_2d = mds_2d.fit_transform(dist_matrix)

    # 3. パウリの排他律（近接斥力）による安定化シミュレーション
    print("[*] パウリ排他律（体積排除）による思考原野の緩和と安定化...")
    coords = coords_3d.copy()
    num_points = coords.shape[0]
    
    # スケールを正規化
    coords = coords / (np.std(coords) + 1e-6)
    
    dt = 0.01
    for step in range(60):
        forces = np.zeros_like(coords)
        for i in range(num_points):
            for j in range(num_points):
                if i == j:
                    continue
                diff = coords[i] - coords[j]
                r = np.linalg.norm(diff) + 1e-4
                target_dist = dist_matrix[i, j]
                
                # スプリング引力
                f_spring = -0.5 * (r - target_dist) * (diff / r)
                # パウリ斥力 (短距離のみ)
                f_pauli = 0.02 / (r**2 + 0.1) * (diff / r)
                
                forces[i] += f_spring + f_pauli
        coords += np.clip(forces * dt, -0.2, 0.2)
        # ドリフト防止
        coords -= coords.mean(axis=0)

    print("[+] 思考原野の自己組織化完了！")

    # 4. 可視化プロット（3D & 2D マップ）
    fig = plt.figure(figsize=(16, 7))
    
    # 3D 思考原野プロット
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    p3 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                     c=np.linalg.norm(coords, axis=1), cmap='plasma', s=90, alpha=0.85, edgecolors='w')
    ax1.set_title("GLM-5.3 Pure Thought Wilderness (3D Geodesic Manifold)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Curvature X")
    ax1.set_ylabel("Curvature Y")
    ax1.set_zlabel("Curvature Z")
    fig.colorbar(p3, ax=ax1, shrink=0.6, label="Attractor Depth / Radius")

    # 2D 投影 & 因果の等高線（ポテンシャル場）
    ax2 = fig.add_subplot(1, 2, 2)
    p2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                     c=np.linalg.norm(coords_2d, axis=1), cmap='plasma', s=90, alpha=0.85, edgecolors='k')
    
    # イデア核同士の主要な力線（引力線）を描画
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if sim_matrix[i, j] > 0.45:  # 強い共鳴関係
                ax2.plot([coords_2d[i, 0], coords_2d[j, 0]], 
                         [coords_2d[i, 1], coords_2d[j, 1]], 
                         'c-', alpha=0.25, linewidth=1.2)
                
    ax2.set_title("GLM-5.3 Thought Wilderness (2D Potential Plane with Resonant Force Lines)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Manifold Dimension 1")
    ax2.set_ylabel("Manifold Dimension 2")
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig.colorbar(p2, ax=ax2, label="Attractor Depth")

    plt.tight_layout()
    output_image = "distillation_experiments/glm53_thought_wilderness_map.png"
    plt.savefig(output_image, dpi=300)
    print(f"[+] 思考原野マップ画像を保存しました: {output_image}")

    # 5. 普遍幾何学ハブとしてのエクスポート
    wilderness_export = {
        "source_model": "GLM-5.3",
        "num_idea_cores": num_nodes,
        "original_dim": 6144,
        "wilderness_dim": 3,
        "coordinates_3d": coords.tolist(),
        "coordinates_2d": coords_2d.tolist(),
        "geodesic_distance_matrix": dist_matrix.tolist()
    }
    output_json = "distillation_experiments/glm53_thought_wilderness_hub.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(wilderness_export, f, ensure_ascii=False, indent=2)
    print(f"[+] 普遍思考原野ハブデータをエクスポートしました: {output_json}")

if __name__ == "__main__":
    build_glm_thought_wilderness()
