import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 乱数シード固定
np.random.seed(42)
torch.manual_seed(42)

class SphericalConceptField:
    """
    高次元球面上 (S^{D-1}) のエネルギー地形（ポテンシャル場）
    """
    def __init__(self, dim=64):
        self.dim = dim
        self.centers = []      # List of tensor (D,)
        self.weights = []      # List of float (深さ)
        self.sigmas = []       # List of float (引力幅)
        self.labels = []       # List of str

    def add_attractor(self, center, weight=1.0, sigma=0.25, label="anchor"):
        # L2正規化して球面上に配置
        center = center / torch.norm(center)
        self.centers.append(center.clone().detach())
        self.weights.append(weight)
        self.sigmas.append(sigma)
        self.labels.append(label)

    def potential(self, x):
        """
        x: (1, D) または (B, D) の単位球面上のベクトル
        H(x) = - \sum w_i * exp( - ||x - c_i||^2 / (2 * sigma_i^2) )
        """
        if not self.centers:
            return torch.zeros(x.shape[0])
        
        centers_tensor = torch.stack(self.centers) # (M, D)
        weights_tensor = torch.tensor(self.weights, dtype=x.dtype, device=x.device)
        sigmas_tensor = torch.tensor(self.sigmas, dtype=x.dtype, device=x.device)

        # 距離二乗 ||x - c_i||^2 = ||x||^2 + ||c_i||^2 - 2 (x . c_i) = 2 - 2 (x . c_i)
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        cos_sim = torch.matmul(x_norm, centers_tensor.T) # (B, M)
        dist_sq = 2.0 - 2.0 * cos_sim

        # ガウス谷の足し合わせ
        val = - torch.sum(weights_tensor * torch.exp(- dist_sq / (2 * sigmas_tensor**2)), dim=-1)
        return val

    def roll_marble(self, start_pos, steps=100, lr=0.05, momentum=0.7):
        """
        球面上でのビー玉転がし（リーマン勾配降下 + モメンタム）
        """
        pos = (start_pos / torch.norm(start_pos)).clone().detach().requires_grad_(True)
        v = torch.zeros_like(pos)
        trajectory = [(pos.detach().clone().cpu().numpy())]

        for step in range(steps):
            H = self.potential(pos.unsqueeze(0))
            grad = torch.autograd.grad(H, pos)[0]

            # 球面の接平面への射影: grad_S = grad - (pos . grad) * pos
            proj_grad = grad - torch.dot(pos, grad) * pos

            # モメンタム更新 & 移動
            v = momentum * v - lr * proj_grad
            pos_new = pos + v
            # 球面へのリトラクション
            pos = (pos_new / torch.norm(pos_new)).detach().requires_grad_(True)
            trajectory.append(pos.detach().clone().cpu().numpy())

            if torch.norm(v) < 1e-4:
                break

        return pos.detach(), np.array(trajectory)


class GreatCircleConceptExtender:
    """
    全大円干渉型 知識延長器 (Great Circle Concept Extender)
    """
    def __init__(self, field: SphericalConceptField, dim=64):
        self.field = field
        self.dim = dim
        # 初期層の重み変換（引力マトリクス）のシミュレーション
        self.W1 = torch.randn(dim, dim) * 0.1

    def extend_concept(self, v_init, label="new_concept", K=3, steps=15, lr=0.1):
        """
        Step 1: 初期力学外挿 -> Step 2: KNN選定 -> Step 3: 全大円干渉極大点同定 -> Step 4: インプラント
        """
        t0 = time.time()
        v_0 = (v_init / torch.norm(v_init)).clone().detach()

        # Step 1: 初期力学予測 F_0 = W1 * v_0
        F_0 = torch.matmul(self.W1, v_0)
        x_pred_0 = (v_0 + 0.3 * F_0) / torch.norm(v_0 + 0.3 * F_0)

        # Step 2: KNN局所アンカー選定
        centers_tensor = torch.stack(self.field.centers) # (M, D)
        cos_sims = torch.matmul(x_pred_0, centers_tensor.T) # (M,)
        topk_indices = torch.topk(cos_sims, min(K, len(self.field.centers))).indices.tolist()

        anchor_centers = [self.field.centers[idx] for idx in topk_indices]
        anchor_weights = [self.field.weights[idx] for idx in topk_indices]

        # Step 3: 全大円干渉場の構築
        # 各アンカー c_k と v_0 が張る平面の法線 n_k
        normals = []
        for c_k in anchor_centers:
            # グラム・シュミット直交化で法線基底を抽出
            proj = torch.dot(c_k, v_0) * v_0
            ortho = c_k - proj
            norm_ortho = torch.norm(ortho)
            if norm_ortho > 1e-5:
                normals.append(ortho / norm_ortho)
            else:
                normals.append(torch.randn(self.dim) / 10.0)

        # リーマン勾配上昇で干渉焦点 x* を収束計算
        x_curr = x_pred_0.clone().detach().requires_grad_(True)
        sigma_ortho = 0.3

        for it in range(steps):
            # 干渉ポテンシャル Phi(x)
            # 大円干渉項: exp( - (x . n_k)^2 / (2 * sigma^2) )
            phi_interference = torch.zeros(1)
            for n_k in normals:
                dot_val = torch.dot(x_curr, n_k)
                phi_interference += torch.exp(- (dot_val**2) / (2 * sigma_ortho**2))

            # アンカー引力拘束項: \sum w_k * (x . c_k)
            phi_attraction = torch.zeros(1)
            for idx, c_k in enumerate(anchor_centers):
                phi_attraction += anchor_weights[idx] * torch.dot(x_curr, c_k)

            phi_total = phi_interference + 0.5 * phi_attraction

            # 勾配上昇
            grad = torch.autograd.grad(phi_total, x_curr)[0]
            # 接平面射影
            proj_grad = grad - torch.dot(x_curr, grad) * x_curr
            x_next = x_curr + lr * proj_grad
            x_curr = (x_next / torch.norm(x_next)).detach().requires_grad_(True)

        x_star = x_curr.detach()
        calc_time = (time.time() - t0) * 1000

        # Step 4: インプラントする谷の深さと幅を決定
        # 周辺アンカーの引力に応じた深さ w*
        sims_to_star = torch.matmul(x_star, centers_tensor.T)
        avg_weight = np.mean(anchor_weights)
        w_star = float(avg_weight * 1.5) # 周辺語を確実に捕まえるため、十分な深さを付与

        # 最も近い既存アンカーとの距離の半分を幅 sigma* とする
        max_sim = float(torch.max(sims_to_star).item())
        min_angle = np.arccos(np.clip(max_sim, -1.0, 1.0))
        sigma_star = max(0.12, min_angle * 0.45)

        # インプラント実行
        self.field.add_attractor(x_star, weight=w_star, sigma=sigma_star, label=f"Ext:{label}")

        return {
            "x_star": x_star,
            "calc_time_ms": calc_time,
            "anchors_used": [self.field.labels[i] for i in topk_indices],
            "w_star": w_star,
            "sigma_star": sigma_star
        }


def run_experiment():
    print("==================================================")
    print(" 全大円干渉型 知識延長アルゴリズム 実証実験")
    print(" (Great Circle Concept Extender Prototype)")
    print("==================================================")

    DIM = 64
    field = SphericalConceptField(dim=DIM)

    # 1. 既知の概念核（アンカー）を配置（小型モデルを模して引力を弱めに設定: weight=0.6）
    # アンカーA: 「物理・自然科学」
    anchor_A = torch.randn(DIM)
    anchor_A[0] += 3.0
    field.add_attractor(anchor_A, weight=0.6, sigma=0.3, label="Anchor_A(Physics)")

    # アンカーB: 「情報・計算機」
    anchor_B = torch.randn(DIM)
    anchor_B[1] += 3.0
    field.add_attractor(anchor_B, weight=0.6, sigma=0.3, label="Anchor_B(Computation)")

    # アンカーC: 「数学・論理」
    anchor_C = torch.randn(DIM)
    anchor_C[0] += 2.0
    anchor_C[1] += 2.0
    field.add_attractor(anchor_C, weight=0.6, sigma=0.3, label="Anchor_C(Mathematics)")

    # 無関係な巨大ノイズアトラクター（遠くにあるが引力が強い: weight=1.5）
    # 小型モデルで「無関係な概念に引きずられるハルシネーション」の温床
    anchor_distractor = torch.randn(DIM)
    anchor_distractor[2] += 4.0
    field.add_attractor(anchor_distractor, weight=1.5, sigma=0.4, label="Distractor(GeneralChat)")

    # 2. 周辺語・ニッチ概念の生成: 「量子情報科学（Quantum Information）」
    # A(物理) と B(情報) の間に位置するが、小型モデルの浅い谷では引力不足で素通りする
    peripheral_word = (anchor_A * 0.5 + anchor_B * 0.5 + torch.randn(DIM) * 0.2)
    peripheral_word = peripheral_word / torch.norm(peripheral_word)

    print(f"\n[1] テスト対象の周辺単語: 『量子情報科学 (Quantum Information)』")
    print(f"    - 初期位置: A(物理)とB(情報)の中間・境界領域")
    print(f"    - 小型モデルの既存アトラクター引力: weight = 0.6 (浅い谷)")

    # 3. インプラント前の推論テスト (Before)
    print("\n--- [2] インプラント前のビー玉転がし推論 (Before) ---")
    pos_before, traj_before = field.roll_marble(peripheral_word, steps=100)
    
    # 収束先チェック
    centers_t = torch.stack(field.centers)
    sims_before = torch.matmul(pos_before, centers_t.T)
    best_idx_before = torch.argmax(sims_before).item()
    best_label_before = field.labels[best_idx_before]
    best_sim_before = sims_before[best_idx_before].item()
    best_dist_before = np.arccos(np.clip(best_sim_before, -1.0, 1.0))

    print(f"  結果: ビー玉の停止先 ➔ 『{best_label_before}』 (コサイン類似度: {best_sim_before:.4f}, 角距離: {best_dist_before:.4f} rad)")
    if "Distractor" in best_label_before or best_dist_before > 0.4:
        print("  ❌ 【課題の再現】引力不足のため目的の概念に留まれず、素通りして無関係な谷に滑落・または彷徨いました！")

    # 4. 全大円干渉型 知識延長の実行
    print("\n--- [3] 全大円干渉型 知識延長の実行 (Extension Algorithm) ---")
    extender = GreatCircleConceptExtender(field, dim=DIM)
    ext_result = extender.extend_concept(peripheral_word, label="QuantumInfo", K=3, steps=15)

    print(f"  ⚡ CPU計算所要時間: {ext_result['calc_time_ms']:.2f} ms (0VRAM / 完全ミリ秒即答)")
    print(f"  📡 参照されたアンカー: {ext_result['anchors_used']}")
    print(f"  ✨ 同定された干渉焦点 x* に新たなアトラクターを彫刻: 深さ w*={ext_result['w_star']:.2f}, 幅 σ*={ext_result['sigma_star']:.3f}")

    # 5. インプラント後の推論テスト (After)
    print("\n--- [4] インプラント後のビー玉転がし推論 (After) ---")
    pos_after, traj_after = field.roll_marble(peripheral_word, steps=100)

    sims_after = torch.matmul(pos_after, centers_t.T if len(field.centers) == len(centers_t) else torch.stack(field.centers).T)
    best_idx_after = torch.argmax(sims_after).item()
    best_label_after = field.labels[best_idx_after]
    best_sim_after = sims_after[best_idx_after].item()
    best_dist_after = np.arccos(np.clip(best_sim_after, -1.0, 1.0))

    print(f"  結果: ビー玉の停止先 ➔ 『{best_label_after}』 (コサイン類似度: {best_sim_after:.4f}, 角距離: {best_dist_after:.4f} rad)")
    if "QuantumInfo" in best_label_after and best_dist_after < 0.15:
        print("  🎉 【大成功】新設された概念核の重力圏に100%吸着され、周辺語が確実に把持・捕捉されました！")

    # 6. PCAによる幾何可視化とプロット保存
    all_points = []
    # 既存アンカー
    for c in field.centers:
        all_points.append(c.cpu().numpy())
    all_points.append(peripheral_word.cpu().numpy())
    for p in traj_before:
        all_points.append(p)
    for p in traj_after:
        all_points.append(p)
    
    pca = PCA(n_components=2)
    pca.fit(np.array(all_points))

    plt.figure(figsize=(10, 8), dpi=150)
    plt.title("Concept Core Extension: Great Circle Interference Simulation", fontsize=14, fontweight='bold')

    # 各アンカーをプロット
    for idx, c in enumerate(field.centers):
        cp = pca.transform(c.cpu().numpy().reshape(1, -1))[0]
        lbl = field.labels[idx]
        if "Ext:" in lbl:
            plt.scatter(cp[0], cp[1], color='magenta', s=250, marker='*', edgecolors='black', label=f"Extracted: {lbl}", zorder=5)
        elif "Distractor" in lbl:
            plt.scatter(cp[0], cp[1], color='red', s=180, marker='s', edgecolors='black', label=f"Distractor: {lbl}", zorder=4)
        else:
            plt.scatter(cp[0], cp[1], color='blue', s=150, marker='o', edgecolors='black', label=f"Anchor: {lbl}", zorder=4)
        plt.text(cp[0] + 0.02, cp[1] + 0.02, lbl, fontsize=9, fontweight='bold')

    # 初期位置
    pw_p = pca.transform(peripheral_word.cpu().numpy().reshape(1, -1))[0]
    plt.scatter(pw_p[0], pw_p[1], color='gold', s=160, marker='D', edgecolors='black', label="Peripheral Word (Start)", zorder=6)

    # インプラント前の軌跡 (赤破線)
    tb_p = pca.transform(traj_before)
    plt.plot(tb_p[:, 0], tb_p[:, 1], 'r--', linewidth=2, alpha=0.8, label="Trajectory (Before: Slipped away)")

    # インプラント後の軌跡 (緑実線)
    ta_p = pca.transform(traj_after)
    plt.plot(ta_p[:, 0], ta_p[:, 1], 'g-', linewidth=2.5, alpha=0.9, label="Trajectory (After: Captured perfectly)")

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    out_png = "scratch/concept_extension_simulation.png"
    plt.savefig(out_png)
    plt.close()
    print(f"\n[5] 可視化プロットを保存しました: {out_png}")

if __name__ == '__main__':
    run_experiment()
