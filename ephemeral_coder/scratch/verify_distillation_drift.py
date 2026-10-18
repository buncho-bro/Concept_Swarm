import os
import sys
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 検証実験: 『全大円干渉・概念蒸留における重力崩壊・位置ズレの検証』
# ==============================================================================

def analyze_distillation_gravitational_drift():
    print("==================================================================")
    print(" 検証: 全大円干渉・概念蒸留における重力崩壊・位置ズレ (Gravitational Drift)")
    print("==================================================================")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dim = 384

    # 蒸留対象の概念群 (密接に関連し合うコーディング・計算機科学の概念)
    concept_words = [
        "Function", "Variable", "Loop", "Recursion", "Asynchronous", 
        "Concurrency", "Thread", "Process", "Lock", "Mutex",
        "Database", "Query", "Table", "Index", "Transaction",
        "Network", "Socket", "Protocol", "TCP", "HTTP"
    ]
    N = len(concept_words)

    # 1. 教師空間の真のターゲットベクトル (Ground Truth Teacher Vectors)
    raw_embs = embedder.encode(concept_words, convert_to_tensor=True).cpu()
    teacher_v = raw_embs / torch.norm(raw_embs, dim=-1, keepdim=True)

    print(f"\n[1] 教師空間の真の概念ベクトル取得完了 (N = {N})")

    # 2. 全大円干渉による概念核抽出における「共存相互引力による位置ズレ」
    # 各概念 x_i を抽出する際、周囲の他概念 {x_j} (j != i) が干渉面に存在する場合の焦点ズレをシミュレート
    isolated_foci = []
    coupled_foci = []

    W_proj = torch.randn(dim, dim) * 0.05

    for i in range(N):
        v_i = teacher_v[i]
        
        # --- A. 孤立抽出 (理想状態: 他概念の干渉なし、基幹座標系のみとの干渉) ---
        # 基底ノイズ直交面との干渉
        torch.manual_seed(42 + i)
        ortho_basis = torch.randn(5, dim)
        ortho_basis = ortho_basis - torch.matmul(ortho_basis, v_i.unsqueeze(-1)) * v_i
        ortho_basis = ortho_basis / torch.norm(ortho_basis, dim=-1, keepdim=True)

        x_iso = (v_i + 0.1 * torch.matmul(W_proj, v_i)).clone().detach().requires_grad_(True)
        for _ in range(25):
            phi = torch.zeros(1)
            for k in range(5):
                dot = torch.dot(x_iso, ortho_basis[k])
                phi += torch.exp(- (dot**2) / (2 * 0.25**2))
            grad = torch.autograd.grad(phi, x_iso)[0]
            proj_grad = grad - torch.dot(x_iso, grad) * x_iso
            x_next = x_iso + 0.08 * proj_grad
            x_iso = (x_next / torch.norm(x_next)).detach().requires_grad_(True)
        isolated_foci.append(x_iso.detach())

        # --- B. 連成抽出 (実際の蒸留: 近接する他概念 {v_j} の大円干渉が重畳) ---
        # 近傍の他概念 3 個
        cos_sims = torch.matmul(v_i, teacher_v.T)
        cos_sims[i] = -999.0
        topk = torch.topk(cos_sims, 3).indices.tolist()
        coupled_neighbors = [teacher_v[k] for k in topk]

        coupled_normals = []
        for c_k in coupled_neighbors:
            proj = torch.dot(c_k, v_i) * v_i
            ortho = c_k - proj
            n_ortho = torch.norm(ortho)
            coupled_normals.append(ortho / n_ortho if n_ortho > 1e-5 else torch.randn(dim)/10.0)

        x_coupled = (v_i + 0.1 * torch.matmul(W_proj, v_i)).clone().detach().requires_grad_(True)
        for _ in range(25):
            phi = torch.zeros(1)
            # 他概念からの干渉重畳
            for n_k in coupled_normals:
                dot = torch.dot(x_coupled, n_k)
                phi += torch.exp(- (dot**2) / (2 * 0.25**2))
            grad = torch.autograd.grad(phi, x_coupled)[0]
            proj_grad = grad - torch.dot(x_coupled, grad) * x_coupled
            x_next = x_coupled + 0.08 * proj_grad
            x_coupled = (x_next / torch.norm(x_next)).detach().requires_grad_(True)
        coupled_foci.append(x_coupled.detach())

    iso_tensor = torch.stack(isolated_foci)
    coupled_tensor = torch.stack(coupled_foci)

    # ズレの定量化 (Drift Angle & Cosine Sim)
    drifts_cos = [torch.dot(coupled_tensor[i], teacher_v[i]).item() for i in range(N)]
    drifts_angle_deg = [np.degrees(np.arccos(np.clip(drifts_cos[i], -1.0, 1.0))) for i in range(N)]
    avg_drift_angle = np.mean(drifts_angle_deg)

    print(f"\n[2] 大円干渉抽出時の概念位置ズレ (Gravitational Drift):")
    print(f"    真のTeacherベクトルとの平均角偏位: {avg_drift_angle:.2f} 度 (最大 {np.max(drifts_angle_deg):.2f} 度)")
    print(f"    真のTeacherベクトルとの平均コサイン類似度: {np.mean(drifts_cos):.4f}")

    print("\n    --- 代表的なズレの具体例 ---")
    for idx in [0, 4, 6, 8, 10, 14]:
        w = concept_words[idx]
        deg = drifts_angle_deg[idx]
        cos_sim = drifts_cos[idx]
        # 引っ張られた先の最近傍他概念
        cross_sims = torch.matmul(coupled_tensor[idx], teacher_v.T)
        cross_sims[idx] = -999.0
        puller_idx = torch.argmax(cross_sims).item()
        puller_w = concept_words[puller_idx]
        puller_sim = cross_sims[puller_idx].item()
        print(f"    ・概念 '{w:12s}': ズレ {deg:5.2f}° (cos={cos_sim:.4f}) => 引力で引っ張られた相手: '{puller_w}' (cos={puller_sim:.4f})")

    # 3. 蒸留モデル（生徒地形）における重力崩壊テスト
    # 抽出した 20 個の概念核を、従来の単一地表（線形加算ポテンシャル）に配置してアトラクターを調べる
    weights = [0.8] * N
    sigmas = [0.25] * N

    def potential_ground(x):
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        cos_s = torch.matmul(x_norm, coupled_tensor.T)
        dist_sq = 2.0 - 2.0 * cos_s
        w_t = torch.tensor(weights, dtype=x.dtype, device=x.device)
        s_t = torch.tensor(sigmas, dtype=x.dtype, device=x.device)
        return - torch.sum(w_t * torch.exp(- dist_sq / (2 * s_t**2)), dim=-1)

    def roll_marble_ground(start_pos, steps=60):
        pos = (start_pos / torch.norm(start_pos)).clone().detach().requires_grad_(True)
        v = torch.zeros_like(pos)
        for _ in range(steps):
            U = potential_ground(pos.unsqueeze(0))
            grad = torch.autograd.grad(U, pos)[0]
            proj_grad = grad - torch.dot(pos, grad) * pos
            v = 0.7 * v - 0.04 * proj_grad
            pos_new = pos + v
            pos = (pos_new / torch.norm(pos_new)).detach().requires_grad_(True)
            if torch.norm(v) < 1e-4:
                break
        return pos.detach()

    # 20個の概念核から球を転がして、生き残っているアトラクター数を調べる
    converged_attractors = []
    merged_count = 0
    for i in range(N):
        pos = roll_marble_ground(coupled_tensor[i])
        # 既存アトラクターとの合体判定 (コサイン > 0.95)
        found = False
        for att in converged_attractors:
            if torch.dot(pos, att).item() > 0.95:
                found = True
                merged_count += 1
                break
        if not found:
            converged_attractors.append(pos)

    surviving_attractors = len(converged_attractors)
    extinction_rate = (1.0 - surviving_attractors / N) * 100.0

    print(f"\n[3] 単一面蒸留（生徒モデル地形）での重力崩壊測定:")
    print(f"    蒸留した概念核数: {N} 個")
    print(f"    生き残った独立アトラクター数: {surviving_attractors} 個")
    print(f"    重力合体による概念消滅率: {extinction_rate:.1f}% ({N - surviving_attractors} 概念が近接概念に飲み込まれて消失)")

    return {
        "avg_drift_angle": avg_drift_angle,
        "drifts_cos": drifts_cos,
        "surviving_attractors": surviving_attractors,
        "extinction_rate": extinction_rate
    }

if __name__ == '__main__':
    analyze_distillation_gravitational_drift()
