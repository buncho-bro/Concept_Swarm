import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42)
torch.manual_seed(42)

# --- 1. 高次元球面ポテンシャル場 ---
class SphericalConceptField:
    def __init__(self, dim=384):
        self.dim = dim
        self.centers = []      # List of tensor (D,)
        self.weights = []      # List of float
        self.sigmas = []       # List of float
        self.labels = []       # List of str

    def add_attractor(self, center, weight=1.0, sigma=0.25, label="anchor"):
        center = center / torch.norm(center)
        self.centers.append(center.clone().detach())
        self.weights.append(float(weight))
        self.sigmas.append(float(sigma))
        self.labels.append(label)

    def potential(self, x):
        if not self.centers:
            return torch.zeros(x.shape[0])
        centers_tensor = torch.stack(self.centers)
        weights_tensor = torch.tensor(self.weights, dtype=x.dtype, device=x.device)
        sigmas_tensor = torch.tensor(self.sigmas, dtype=x.dtype, device=x.device)

        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        cos_sim = torch.matmul(x_norm, centers_tensor.T)
        dist_sq = 2.0 - 2.0 * cos_sim
        return - torch.sum(weights_tensor * torch.exp(- dist_sq / (2 * sigmas_tensor**2)), dim=-1)

    def roll_marble(self, start_pos, steps=80, lr=0.04, momentum=0.7):
        if not self.centers:
            return start_pos / torch.norm(start_pos)
        pos = (start_pos / torch.norm(start_pos)).clone().detach().requires_grad_(True)
        v = torch.zeros_like(pos)
        for _ in range(steps):
            H = self.potential(pos.unsqueeze(0))
            grad = torch.autograd.grad(H, pos)[0]
            proj_grad = grad - torch.dot(pos, grad) * pos
            v = momentum * v - lr * proj_grad
            pos_new = pos + v
            pos = (pos_new / torch.norm(pos_new)).detach().requires_grad_(True)
            if torch.norm(v) < 1e-4:
                break
        return pos.detach()


# --- 2. 全大円干渉型 知識延長器 ---
class GreatCircleConceptExtender:
    def __init__(self, field: SphericalConceptField, dim=384):
        self.field = field
        self.dim = dim
        self.W1 = torch.randn(dim, dim) * 0.05

    def extend_concept(self, v_init, label="new_concept", K=3, steps=15, lr=0.08):
        v_0 = (v_init / torch.norm(v_init)).clone().detach()

        # 地形が完全に空の場合: そのまま第1概念として登録
        if len(self.field.centers) == 0:
            self.field.add_attractor(v_0, weight=1.0, sigma=0.25, label=f"Ext:{label}")
            return {"x_star": v_0, "w_star": 1.0, "sigma_star": 0.25}

        # 地形に1〜2個しかない場合は利用可能なアンカー全てを使用
        actual_K = min(K, len(self.field.centers))

        # Step 1: 力学初速予測
        F_0 = torch.matmul(self.W1, v_0)
        x_pred_0 = (v_0 + 0.2 * F_0) / torch.norm(v_0 + 0.2 * F_0)

        # Step 2: KNN局所アンカー選定
        centers_tensor = torch.stack(self.field.centers)
        cos_sims = torch.matmul(x_pred_0, centers_tensor.T)
        topk_indices = torch.topk(cos_sims, actual_K).indices.tolist()

        anchor_centers = [self.field.centers[idx] for idx in topk_indices]
        anchor_weights = [self.field.weights[idx] for idx in topk_indices]

        # Step 3: 全大円干渉場
        normals = []
        for c_k in anchor_centers:
            proj = torch.dot(c_k, v_0) * v_0
            ortho = c_k - proj
            norm_ortho = torch.norm(ortho)
            normals.append(ortho / norm_ortho if norm_ortho > 1e-5 else torch.randn(self.dim) / 10.0)

        x_curr = x_pred_0.clone().detach().requires_grad_(True)
        sigma_ortho = 0.25

        for it in range(steps):
            phi_interf = torch.zeros(1)
            for n_k in normals:
                dot_val = torch.dot(x_curr, n_k)
                phi_interf += torch.exp(- (dot_val**2) / (2 * sigma_ortho**2))

            phi_attr = torch.zeros(1)
            for idx, c_k in enumerate(anchor_centers):
                phi_attr += anchor_weights[idx] * torch.dot(x_curr, c_k)

            phi_total = phi_interf + 0.4 * phi_attr
            grad = torch.autograd.grad(phi_total, x_curr)[0]
            proj_grad = grad - torch.dot(x_curr, grad) * x_curr
            x_next = x_curr + lr * proj_grad
            x_curr = (x_next / torch.norm(x_next)).detach().requires_grad_(True)

        x_star = x_curr.detach()

        # Step 4: インプラント深さと幅
        sims_to_star = torch.matmul(x_star, centers_tensor.T)
        avg_weight = np.mean(anchor_weights)
        w_star = 1.0  # 全て対等な強さ (1.0) で統一

        max_sim = float(torch.max(sims_to_star).item())
        min_angle = np.arccos(np.clip(max_sim, -1.0, 1.0))
        sigma_star = max(0.10, min_angle * 0.42)

        self.field.add_attractor(x_star, weight=w_star, sigma=sigma_star, label=f"Ext:{label}")

        return {
            "x_star": x_star,
            "w_star": w_star,
            "sigma_star": sigma_star
        }


# --- 3. データセットの準備 (前回の全60語) ---
def load_dataset():
    boundary_words = [
        "Biochemistry", "Bioinformatics", "Quantum Computing", "Computational Economics",
        "Neuroeconomics", "Biophysics", "Medical Informatics", "Philosophy of Science",
        "Pharmacogenomics", "Algorithmic Trading", "Cognitive Linguistics", "Chemical Physics",
        "Synthetic Biology", "Digital Humanities", "Econophysics", "Health Economics",
        "Bioethics", "Chemoinformatics", "Neurophilosophy", "Financial Engineering"
    ]
    deep_niche_words = [
        "Topological Insulator", "Superconducting Qubit", "CRISPR-Cas9 Gene Editing",
        "Ribosome Biogenesis", "Convolutional Neural Network", "Transformer Attention Mechanism",
        "Interest Rate Swap Derivative", "Monopolistic Competition", "Monoclonal Antibody Therapy",
        "Pharmacokinetics and Metabolism", "Deconstruction and Post-structuralism", "Phenomenology of Perception",
        "Non-equilibrium Thermodynamics", "Epigenetic DNA Methylation", "Cache Invalidation Algorithm",
        "Liquidity Preference Hypothesis", "Cardiac Electrophysiology", "Angiotensin Converting Enzyme",
        "Hermeneutic Circle", "Epistemological Relativism"
    ]
    outlier_words = [
        "Ceramic Pottery Glaze Technique", "Deep Sea Hydrothermal Vent Ecosystem",
        "Traditional Origami Folding Pattern", "Gourmet Coffee Roasting Profile",
        "Alpine Mountaineering Crampons", "Ancient Egyptian Hieroglyph Decipherment",
        "Medieval Blacksmithing Metallurgy", "Sourdough Fermentation Yeast",
        "Symphonic Orchestration Woodwinds", "Bonsai Tree Pruning Aesthetics",
        "Vintage Mechanical Watch Escapement", "Desert Sand Dune Geomorphology",
        "Volcanic Magma Chamber Petrology", "Handcrafted Leather Shoe Stitching",
        "Scuba Diving Buoyancy Control", "Permafrost Thaw Methane Release",
        "Calligraphy Inkstone Grinding", "Traditional Wooden Shipbuilding Joinery",
        "Meteorological Tornado Vortex Dynamics", "Artisanal Cheese Aging Bacteria"
    ]
    return boundary_words + deep_niche_words + outlier_words


# --- 4. 空の初期地形における実験実行 ---
def run_empty_landscape_experiment():
    print("==================================================")
    print(" 実験: 『初期地形が空（ゼロ）』の場合の知識延長挙動")
    print(" (Zero-Base Empty Landscape Self-Organization Test)")
    print("==================================================")

    print("\n[1] Sentence-Transformers モデル (all-MiniLM-L6-v2) から埋め込みを生成...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    test_words = load_dataset() # 60語
    test_embs = embedder.encode(test_words, convert_to_tensor=True).cpu()
    print(f"    生成完了: 全 {len(test_words)} 語 (384次元)")

    # 初期地形: 完全な空 (アンカー数 0)
    field = SphericalConceptField(dim=384)
    extender = GreatCircleConceptExtender(field, dim=384)
    print("\n[2] 初期地形: 完全なフラット球面 (アンカー数 = 0)")

    test_stages = [5, 10, 15, 20, 25, 30, 40, 50, 60]
    metrics = {
        "N": [],
        "recall_rate": [],       # 自分の谷に正しく落ちた割合 (%)
        "crosstalk_rate": [],    # 他の谷に吸い取られた割合 (%)
        "unique_attractors": [], # 独立して生き残った谷の数
        "unique_ratio": []       # 有効アトラクター比率 (%)
    }

    current_count = 0
    for target_N in test_stages:
        while current_count < target_N:
            word = test_words[current_count]
            emb = test_embs[current_count]
            extender.extend_concept(emb, label=word[:15], K=3, steps=15)
            current_count += 1

        # 評価: 現在存在する target_N 個の概念の想起テスト
        centers_t = torch.stack(field.centers) # (target_N, D)
        recalled = 0
        crosstalk = 0
        converged_destinations = set()

        for i in range(target_N):
            emb = test_embs[i]
            pos = field.roll_marble(emb, steps=60)
            sims = torch.matmul(pos, centers_t.T)
            top_idx = torch.argmax(sims).item()

            converged_destinations.add(top_idx)

            if top_idx == i:
                recalled += 1
            else:
                crosstalk += 1

        recall_pct = (recalled / target_N) * 100.0
        crosstalk_pct = (crosstalk / target_N) * 100.0
        unique_cnt = len(converged_destinations)
        unique_pct = (unique_cnt / target_N) * 100.0

        metrics["N"].append(target_N)
        metrics["recall_rate"].append(recall_pct)
        metrics["crosstalk_rate"].append(crosstalk_pct)
        metrics["unique_attractors"].append(unique_cnt)
        metrics["unique_ratio"].append(unique_pct)

        print(f"  [N = {target_N:2d} 語自律蓄積] 想起成功率: {recall_pct:5.1f}% | 有効独立アトラクター数: {unique_cnt:2d}/{target_N:2d} ({unique_pct:5.1f}%) | 混線率: {crosstalk_pct:5.1f}%")

    # グラフプロット
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(metrics["N"], metrics["recall_rate"], 'g-o', linewidth=2, label="Recall Rate (% / 正しい谷への想起)")
    plt.plot(metrics["N"], metrics["unique_ratio"], 'b-s', linewidth=2, label="Unique Attractors Ratio (% / 独立谷残存率)")
    plt.plot(metrics["N"], metrics["crosstalk_rate"], 'r--^', linewidth=1.5, label="Crosstalk Rate (% / 他の谷への誤吸着)")

    plt.title("Zero-Base Empty Landscape: Self-Organization Scaling (D=384)", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Sequentially Implanted Concepts (N)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='center right', fontsize=10)
    plt.tight_layout()

    out_png = "scratch/empty_landscape_scaling_curves.png"
    plt.savefig(out_png)
    plt.close()
    print(f"\n[3] 評価曲線を保存しました: {out_png}")

if __name__ == '__main__':
    run_empty_landscape_experiment()
