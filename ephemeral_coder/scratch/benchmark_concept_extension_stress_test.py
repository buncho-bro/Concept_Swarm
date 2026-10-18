import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42)
torch.manual_seed(42)

# --- 1. 高次元球面ポテンシャル場 (Spherical Concept Field) ---
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


# --- 2. 全大円干渉型 知識延長器 (Great Circle Concept Extender) ---
class GreatCircleConceptExtender:
    def __init__(self, field: SphericalConceptField, dim=384):
        self.field = field
        self.dim = dim
        self.W1 = torch.randn(dim, dim) * 0.05

    def extend_concept(self, v_init, label="new_concept", K=3, steps=15, lr=0.08):
        t0 = time.time()
        v_0 = (v_init / torch.norm(v_init)).clone().detach()

        # Step 1: 力学初速予測
        F_0 = torch.matmul(self.W1, v_0)
        x_pred_0 = (v_0 + 0.2 * F_0) / torch.norm(v_0 + 0.2 * F_0)

        # Step 2: KNN局所アンカー選定
        centers_tensor = torch.stack(self.field.centers)
        cos_sims = torch.matmul(x_pred_0, centers_tensor.T)
        topk_indices = torch.topk(cos_sims, min(K, len(self.field.centers))).indices.tolist()

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
        calc_time = (time.time() - t0) * 1000

        # Step 4: インプラント深さと幅
        sims_to_star = torch.matmul(x_star, centers_tensor.T)
        avg_weight = np.mean(anchor_weights)
        w_star = float(avg_weight * 1.35)

        max_sim = float(torch.max(sims_to_star).item())
        min_angle = np.arccos(np.clip(max_sim, -1.0, 1.0))
        sigma_star = max(0.10, min_angle * 0.42)

        self.field.add_attractor(x_star, weight=w_star, sigma=sigma_star, label=f"Ext:{label}")

        return {
            "x_star": x_star,
            "calc_time_ms": calc_time,
            "w_star": w_star,
            "sigma_star": sigma_star
        }


# --- 3. 実テキストデータセットの準備 (60語の多角的分散セット) ---
def load_dataset():
    # 6つの基幹概念 (Base Anchors)
    base_anchors = [
        "Physics and Quantum Mechanics",
        "Biology and Genetics",
        "Computer Science and Artificial Intelligence",
        "Economics and Finance",
        "Medicine and Pharmacology",
        "Philosophy and Literature"
    ]

    # Type A: 境界語 (Hybrid / Boundary - 20語)
    boundary_words = [
        "Biochemistry", "Bioinformatics", "Quantum Computing", "Computational Economics",
        "Neuroeconomics", "Biophysics", "Medical Informatics", "Philosophy of Science",
        "Pharmacogenomics", "Algorithmic Trading", "Cognitive Linguistics", "Chemical Physics",
        "Synthetic Biology", "Digital Humanities", "Econophysics", "Health Economics",
        "Bioethics", "Chemoinformatics", "Neurophilosophy", "Financial Engineering"
    ]

    # Type B: 局所ニッチ語 (Deep Niche - 20語)
    deep_niche_words = [
        "Topological Insulator", "Superconducting Qubit", "CRISPR-Cas9 Gene Editing",
        "Ribosome Biogenesis", "Convolutional Neural Network", "Transformer Attention Mechanism",
        "Interest Rate Swap Derivative", "Monopolistic Competition", "Monoclonal Antibody Therapy",
        "Pharmacokinetics and Metabolism", "Deconstruction and Post-structuralism", "Phenomenology of Perception",
        "Non-equilibrium Thermodynamics", "Epigenetic DNA Methylation", "Cache Invalidation Algorithm",
        "Liquidity Preference Hypothesis", "Cardiac Electrophysiology", "Angiotensin Converting Enzyme",
        "Hermeneutic Circle", "Epistemological Relativism"
    ]

    # Type C: 孤立語・外れ値 (Outliers - 20語)
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

    return base_anchors, boundary_words, deep_niche_words, outlier_words


# --- 4. 包括的ベンチマーク・ストレステストの実行 ---
def run_benchmark():
    print("==================================================")
    print(" 全大円干渉型 知識延長: 大規模実テキスト多角検証")
    print(" (Multi-Word Real Text Embedding Stress Test)")
    print("==================================================")

    # 埋め込みモデルのロード (384次元)
    print("\n[1] Sentence-Transformers モデル (all-MiniLM-L6-v2) をロード中...")
    t0 = time.time()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"    ロード完了 ({time.time()-t0:.2f}秒, 次元数: 384)")

    base_anchors, boundary_words, deep_niche_words, outlier_words = load_dataset()
    test_words = boundary_words + deep_niche_words + outlier_words # 合計 60語
    word_types = ["Boundary"] * 20 + ["DeepNiche"] * 20 + ["Outlier"] * 20

    # 埋め込み生成
    print(f"[2] 語彙埋め込みを生成中 (基幹アンカー: {len(base_anchors)}語, テスト周辺語: {len(test_words)}語)...")
    base_embs = embedder.encode(base_anchors, convert_to_tensor=True).cpu()
    test_embs = embedder.encode(test_words, convert_to_tensor=True).cpu()

    # フィールド初期化（小型モデル想定: アンカー引力 weight=0.6, sigma=0.28）
    field = SphericalConceptField(dim=384)
    for idx, (label, emb) in enumerate(zip(base_anchors, base_embs)):
        field.add_attractor(emb, weight=0.6, sigma=0.28, label=f"Base:{label[:15]}")

    # --- Phase 1: インプラント前 (Before) のベースライン測定 ---
    print("\n--- [Phase 1] インプラント前 (Before) の小型モデル挙動測定 ---")
    before_results = {"Boundary": [], "DeepNiche": [], "Outlier": []}
    
    for i, (word, emb, wtype) in enumerate(zip(test_words, test_embs, word_types)):
        pos = field.roll_marble(emb, steps=60)
        # 最近傍アンカーとのコサイン類似度
        centers_t = torch.stack(field.centers)
        sims = torch.matmul(pos, centers_t.T)
        best_sim = torch.max(sims).item()
        best_idx = torch.argmax(sims).item()
        
        # 初期位置と収束位置の移動角距離 (rad)
        move_angle = np.arccos(np.clip(torch.dot(emb, pos).item(), -1.0, 1.0))
        before_results[wtype].append({
            "word": word,
            "best_anchor": field.labels[best_idx],
            "sim_to_anchor": best_sim,
            "drift_angle": move_angle
        })

    print(f"  [Before 測定完了] 全60語の初期引き込み状況:")
    for wtype in ["Boundary", "DeepNiche", "Outlier"]:
        drifts = [r["drift_angle"] for r in before_results[wtype]]
        print(f"    - {wtype:10s} (20語): 平均ドリフト角 = {np.mean(drifts):.4f} rad (固有の谷がないため親概念に滑落・漂流)")

    # --- Phase 2: 段階的連続インプラント・ストレステスト ---
    print("\n--- [Phase 2] 段階的インプラント・ストレステスト (N = 10, 20, 30, 40, 50, 60) ---")
    extender = GreatCircleConceptExtender(field, dim=384)

    test_stages = [10, 20, 30, 40, 50, 60]
    stage_metrics = {
        "N": [],
        "new_recall": [],       # 延長概念の想起成功率 (%)
        "anchor_retention": [], # 既存基幹アンカーの非破壊率 (%)
        "crosstalk_rate": [],   # 谷の縮退・混線率 (%)
        "avg_calc_time_ms": []  # 平均計算時間 (ms)
    }

    calc_times = []
    
    current_implanted = 0
    for target_N in test_stages:
        # target_N までインプラントを進める
        while current_implanted < target_N:
            w_idx = current_implanted
            word = test_words[w_idx]
            emb = test_embs[w_idx]
            res = extender.extend_concept(emb, label=word[:15], K=3, steps=15)
            calc_times.append(res["calc_time_ms"])
            current_implanted += 1

        # 評価 1: 延長した概念群の想起成功率（自分の谷に落ちるか）
        recalled = 0
        crosstalk = 0
        centers_t = torch.stack(field.centers) # (6 + target_N, D)

        for i in range(target_N):
            emb = test_embs[i]
            expected_idx = 6 + i # インプラントされた自分のインデックス
            pos = field.roll_marble(emb, steps=60)
            sims = torch.matmul(pos, centers_t.T)
            top_idx = torch.argmax(sims).item()

            if top_idx == expected_idx:
                recalled += 1
            else:
                # 別の延長アトラクターに落ちた場合は混線（Crosstalk）
                if top_idx >= 6:
                    crosstalk += 1

        recall_pct = (recalled / target_N) * 100.0
        crosstalk_pct = (crosstalk / target_N) * 100.0

        # 評価 2: 既存基幹アンカー（6個）の非破壊率（元のアンカーに正しく落ちるか）
        anchor_kept = 0
        for i in range(6):
            base_emb = base_embs[i]
            pos = field.roll_marble(base_emb, steps=60)
            sims = torch.matmul(pos, centers_t.T)
            top_idx = torch.argmax(sims).item()
            if top_idx == i:
                anchor_kept += 1
        retention_pct = (anchor_kept / 6) * 100.0

        stage_metrics["N"].append(target_N)
        stage_metrics["new_recall"].append(recall_pct)
        stage_metrics["anchor_retention"].append(retention_pct)
        stage_metrics["crosstalk_rate"].append(crosstalk_pct)
        stage_metrics["avg_calc_time_ms"].append(np.mean(calc_times[-10:]))

        print(f"  [N = {target_N:2d} 語インプラント] 想起成功率: {recall_pct:5.1f}% | 基幹アンカー非破壊率: {retention_pct:5.1f}% | 混線率: {crosstalk_pct:4.1f}% | 平均時間: {np.mean(calc_times[-10:]):.2f}ms")

    # --- Phase 3: タイプ別 (Boundary, DeepNiche, Outlier) の最終精度分析 ---
    print("\n--- [Phase 3] 単語タイプ別の最終想起精度分析 (N = 60 インプラント後) ---")
    type_recalls = {"Boundary": 0, "DeepNiche": 0, "Outlier": 0}
    type_angles = {"Boundary": [], "DeepNiche": [], "Outlier": []}

    centers_t = torch.stack(field.centers)
    for i, (word, emb, wtype) in enumerate(zip(test_words, test_embs, word_types)):
        expected_idx = 6 + i
        pos = field.roll_marble(emb, steps=60)
        sims = torch.matmul(pos, centers_t.T)
        top_idx = torch.argmax(sims).item()
        
        target_center = field.centers[expected_idx]
        angle_err = np.arccos(np.clip(torch.dot(pos, target_center).item(), -1.0, 1.0))
        type_angles[wtype].append(angle_err)

        if top_idx == expected_idx:
            type_recalls[wtype] += 1

    for wtype in ["Boundary", "DeepNiche", "Outlier"]:
        acc = (type_recalls[wtype] / 20) * 100.0
        avg_err = np.mean(type_angles[wtype])
        print(f"  - {wtype:10s} (20語): 想起成功率 = {acc:5.1f}% (平均角度誤差: {avg_err:.4f} rad)")

    # --- 5. 評価曲線のプロットと保存 ---
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(stage_metrics["N"], stage_metrics["new_recall"], 'g-o', linewidth=2, label="New Concept Recall Rate (%)")
    plt.plot(stage_metrics["N"], stage_metrics["anchor_retention"], 'b-s', linewidth=2, label="Base Anchor Retention Rate (%)")
    plt.plot(stage_metrics["N"], stage_metrics["crosstalk_rate"], 'r--^', linewidth=1.5, label="Crosstalk / Merge Rate (%)")

    plt.title("Stress Test: Great Circle Concept Extension Scaling Curves (D=384)", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Implanted Concepts (N)", fontsize=11)
    plt.ylabel("Rate (%)", fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()

    out_curve_png = "scratch/stress_test_capacity_curves.png"
    plt.savefig(out_curve_png)
    plt.close()
    print(f"\n[4] スケーリング評価曲線を保存しました: {out_curve_png}")

    return stage_metrics

if __name__ == '__main__':
    run_benchmark()
