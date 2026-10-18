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

# --- 1. 両面・磁気浮上ポテンシャル場 (Dual-Surface Magnetic Levitation Field) ---
class DualSurfaceConceptField:
    """
    地表（重力の谷）と上空（磁石の引力）の両面化ポテンシャル場
    """
    def __init__(self, dim=384):
        self.dim = dim
        # 地表（基幹アンカー群: 重力の谷）
        self.ground_centers = []  # List of tensor (D,)
        self.ground_weights = []  # List of float
        self.ground_sigmas = []   # List of float
        self.ground_labels = []   # List of str

        # 上空（新概念群: 磁気浮上トラップ）
        self.sky_centers = []     # List of tensor (D,)
        self.sky_weights = []     # List of float (磁力強度)
        self.sky_sigmas = []      # List of float (磁界有効半径)
        self.sky_labels = []      # List of str

    def add_ground_anchor(self, center, weight=0.6, sigma=0.28, label="ground"):
        center = center / torch.norm(center)
        self.ground_centers.append(center.clone().detach())
        self.ground_weights.append(float(weight))
        self.ground_sigmas.append(float(sigma))
        self.ground_labels.append(label)

    def add_sky_magnet(self, center, weight=0.8, sigma=0.20, label="magnet"):
        center = center / torch.norm(center)
        self.sky_centers.append(center.clone().detach())
        self.sky_weights.append(float(weight))
        self.sky_sigmas.append(float(sigma))
        self.sky_labels.append(label)

    def potential(self, x):
        """
        全有効ポテンシャル U(x) = H_ground(x) + H_sky(x)
        ※ 磁石同士の合体（ブラックホール化）を防ぐため、
           上空磁石は線形単純加算ではなく「LogSumExp (Smooth Max)」で局所独立性を担保する！
        """
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        B = x_norm.shape[0]

        # 1. 地表の重力ポテンシャル H_ground (基幹アンカー群)
        val_ground = torch.zeros(B, dtype=x.dtype, device=x.device)
        if self.ground_centers:
            gc = torch.stack(self.ground_centers)
            gw = torch.tensor(self.ground_weights, dtype=x.dtype, device=x.device)
            gs = torch.tensor(self.ground_sigmas, dtype=x.dtype, device=x.device)
            cos_sim_g = torch.matmul(x_norm, gc.T)
            dist_sq_g = 2.0 - 2.0 * cos_sim_g
            val_ground = - torch.sum(gw * torch.exp(- dist_sq_g / (2 * gs**2)), dim=-1)

        # 2. 上空の磁気浮上ポテンシャル H_sky (分水嶺から見上げた空の磁石)
        # 各磁石は独立した浮上ポケットとして振る舞い、合体しない
        val_sky = torch.zeros(B, dtype=x.dtype, device=x.device)
        if self.sky_centers:
            sc = torch.stack(self.sky_centers)
            sw = torch.tensor(self.sky_weights, dtype=x.dtype, device=x.device)
            ss = torch.tensor(self.sky_sigmas, dtype=x.dtype, device=x.device)
            cos_sim_s = torch.matmul(x_norm, sc.T)
            dist_sq_s = 2.0 - 2.0 * cos_sim_s
            
            # 各磁石の吸引力: sw * exp(- dist_sq / (2 * ss^2))
            magnet_pulls = sw * torch.exp(- dist_sq_s / (2 * ss**2)) # (B, N_sky)
            
            # Smooth-Max (LogSumExp) による局所排他合成（磁石同士が足し算で無限に深くならない！）
            temperature = 5.0
            val_sky = - (1.0 / temperature) * torch.logsumexp(temperature * magnet_pulls, dim=-1)

        return val_ground + val_sky

    def roll_marble(self, start_pos, steps=80, lr=0.04, momentum=0.7):
        pos = (start_pos / torch.norm(start_pos)).clone().detach().requires_grad_(True)
        v = torch.zeros_like(pos)
        for _ in range(steps):
            U = self.potential(pos.unsqueeze(0))
            grad = torch.autograd.grad(U, pos)[0]
            proj_grad = grad - torch.dot(pos, grad) * pos
            v = momentum * v - lr * proj_grad
            pos_new = pos + v
            pos = (pos_new / torch.norm(pos_new)).detach().requires_grad_(True)
            if torch.norm(v) < 1e-4:
                break
        return pos.detach()


# --- 2. 全大円干渉による上空磁石の配置器 ---
class SkyMagnetExtender:
    def __init__(self, field: DualSurfaceConceptField, dim=384):
        self.field = field
        self.dim = dim
        self.W1 = torch.randn(dim, dim) * 0.05

    def deploy_sky_magnet(self, v_init, label="new_concept", K=3, steps=15, lr=0.08):
        v_0 = (v_init / torch.norm(v_init)).clone().detach()

        # Step 1: 力学初速予測
        F_0 = torch.matmul(self.W1, v_0)
        x_pred_0 = (v_0 + 0.2 * F_0) / torch.norm(v_0 + 0.2 * F_0)

        # Step 2: 地表の基幹アンカーからKNN選定 (分水嶺を特定)
        gc_tensor = torch.stack(self.field.ground_centers)
        cos_sims = torch.matmul(x_pred_0, gc_tensor.T)
        topk_indices = torch.topk(cos_sims, min(K, len(self.field.ground_centers))).indices.tolist()

        anchor_centers = [self.field.ground_centers[idx] for idx in topk_indices]

        # Step 3: 全大円干渉によって分水嶺の真上の焦点 (x*) を同定
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

            grad = torch.autograd.grad(phi_interf, x_curr)[0]
            proj_grad = grad - torch.dot(x_curr, grad) * x_curr
            x_next = x_curr + lr * proj_grad
            x_curr = (x_next / torch.norm(x_next)).detach().requires_grad_(True)

        x_star = x_curr.detach()

        # Step 4: 上空に磁石を配置（地表は一切削らない！）
        # 最も近い基幹アンカーとの角距離を測定
        sims_to_ground = torch.matmul(x_star, gc_tensor.T)
        max_sim = float(torch.max(sims_to_ground).item())
        min_angle = np.arccos(np.clip(max_sim, -1.0, 1.0))

        # 磁力強度: 地表の重力(0.6)に対して、分水嶺上で粒子を引き上げるのに十分な磁力(0.75〜0.85)
        m_weight = 0.80
        # 磁界の有効半径: 既存の谷を乱さないよう、分水嶺近傍に限定 (0.15 rad 程度)
        m_sigma = max(0.12, min_angle * 0.38)

        self.field.add_sky_magnet(x_star, weight=m_weight, sigma=m_sigma, label=f"Mag:{label}")

        return {
            "x_star": x_star,
            "m_weight": m_weight,
            "m_sigma": m_sigma
        }


# --- 3. データセットの準備 (60語の多角的分散セット) ---
def load_dataset():
    base_anchors = [
        "Physics and Quantum Mechanics",
        "Biology and Genetics",
        "Computer Science and Artificial Intelligence",
        "Economics and Finance",
        "Medicine and Pharmacology",
        "Philosophy and Literature"
    ]
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
    return base_anchors, boundary_words + deep_niche_words + outlier_words


# --- 4. 磁気浮上ストレステストの実行 ---
def run_magnetic_levitation_test():
    print("==================================================")
    print(" 実験: 『分水嶺から見上げた空の巨大磁石』磁気浮上検証")
    print(" (Dual-Surface Magnetic Levitation Stress Test)")
    print("==================================================")

    print("\n[1] Sentence-Transformers モデル (all-MiniLM-L6-v2) をロード中...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    base_anchors, test_words = load_dataset()
    base_embs = embedder.encode(base_anchors, convert_to_tensor=True).cpu()
    test_embs = embedder.encode(test_words, convert_to_tensor=True).cpu()

    # フィールド初期化（地表の基幹アンカー6つを配置: weight=0.6, sigma=0.28）
    field = DualSurfaceConceptField(dim=384)
    for label, emb in zip(base_anchors, base_embs):
        field.add_ground_anchor(emb, weight=0.6, sigma=0.28, label=f"Ground:{label[:12]}")

    print(f"\n[2] 地表の基幹概念核 (6個) を固定配置完了 (地表は一切削らない)")
    print(f"    テスト周辺語: {len(test_words)} 語")

    extender = SkyMagnetExtender(field, dim=384)

    test_stages = [10, 20, 30, 40, 50, 60]
    metrics = {
        "N": [],
        "magnet_recall": [],    # 新概念が頭上の磁石に正しく浮遊捕捉された率 (%)
        "anchor_retention": [], # 地表の基幹アンカーが100%保持されている率 (%)
        "crosstalk_rate": []    # 別の磁石や谷へ誤吸着された率 (%)
    }

    current_deployed = 0
    for target_N in test_stages:
        while current_deployed < target_N:
            word = test_words[current_deployed]
            emb = test_embs[current_deployed]
            extender.deploy_sky_magnet(emb, label=word[:15], K=3, steps=15)
            current_deployed += 1

        # 評価 1: 吊るした磁石 (target_N 個) の浮遊捕捉テスト
        recalled = 0
        crosstalk = 0
        sc_tensor = torch.stack(field.sky_centers)

        for i in range(target_N):
            emb = test_embs[i]
            pos = field.roll_marble(emb, steps=60)
            
            # 磁石とのコサイン類似度
            sims = torch.matmul(pos, sc_tensor.T)
            top_magnet_idx = torch.argmax(sims).item()

            if top_magnet_idx == i:
                recalled += 1
            else:
                crosstalk += 1

        recall_pct = (recalled / target_N) * 100.0
        crosstalk_pct = (crosstalk / target_N) * 100.0

        # 評価 2: 地表の基幹アンカー（6個）が100%守られているか？
        gc_tensor = torch.stack(field.ground_centers)
        anchor_kept = 0
        for i in range(6):
            base_emb = base_embs[i]
            pos = field.roll_marble(base_emb, steps=60)
            # 地表アンカーとのコサイン類似度
            sims_g = torch.matmul(pos, gc_tensor.T)
            top_g_idx = torch.argmax(sims_g).item()
            if top_g_idx == i:
                anchor_kept += 1
        retention_pct = (anchor_kept / 6) * 100.0

        metrics["N"].append(target_N)
        metrics["magnet_recall"].append(recall_pct)
        metrics["anchor_retention"].append(retention_pct)
        metrics["crosstalk_rate"].append(crosstalk_pct)

        print(f"  [N = {target_N:2d} 磁石配置] 浮遊捕捉成功率: {recall_pct:5.1f}% | ★地表アンカー非破壊率: {retention_pct:5.1f}% | 混線率: {crosstalk_pct:5.1f}%")

    # グラフプロット
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(metrics["N"], metrics["magnet_recall"], 'm-o', linewidth=2.5, label="Sky Magnet Levitation Recall (% / 浮遊捕捉成功率)")
    plt.plot(metrics["N"], metrics["anchor_retention"], 'b-s', linewidth=2.5, label="Ground Anchor Retention (% / 地表基幹非破壊率)")
    plt.plot(metrics["N"], metrics["crosstalk_rate"], 'r--^', linewidth=1.5, label="Crosstalk Rate (% / 誤吸着率)")

    plt.title("Dual-Surface Magnetic Levitation Scaling Benchmark (D=384)", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Sky Magnets Deployed (N)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='center right', fontsize=10)
    plt.tight_layout()

    out_png = "scratch/magnetic_levitation_scaling_curves.png"
    plt.savefig(out_png)
    plt.close()
    print(f"\n[3] 評価曲線を保存しました: {out_png}")

if __name__ == '__main__':
    run_magnetic_levitation_test()
