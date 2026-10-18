import os
import sys
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class DualSurfaceConceptLearner:
    """
    ユーザー着想『地表重力（基幹知識）＋上空磁石（新概念トラップ）』に基づく
    自律型・破局的忘却ゼロのオンライン概念学習エンジン
    """
    def __init__(self, embedder=None, dim=384, temperature=5.0):
        self.dim = dim
        self.temperature = temperature
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")

        # 地表（基幹概念核: フリーズされた重力の谷）
        self.ground_centers = []    # List of Tensor(D,)
        self.ground_weights = []    # List of float
        self.ground_sigmas = []     # List of float
        self.ground_labels = []     # List of str

        # 上空（自律学習された磁石群: 磁気浮上トラップ）
        self.sky_centers = []       # List of Tensor(D,)
        self.sky_weights = []       # List of float
        self.sky_sigmas = []        # List of float
        self.sky_labels = []        # List of str
        self.sky_aliases = []       # List of list of str (同化された同義語・周辺語)
        self.sky_counts = []        # 活性化回数

        # 大円干渉プロジェクター（初速変換器）
        self.W1 = torch.randn(dim, dim) * 0.05

        # テンソルキャッシュ
        self._ground_tensor = None
        self._sky_tensor = None

    def add_ground_anchor(self, text, weight=0.6, sigma=0.28):
        """基幹概念核を地表に配置（二度と削られない）"""
        emb = self.embedder.encode(text, convert_to_tensor=True).cpu()
        emb = emb / torch.norm(emb)
        self.ground_centers.append(emb)
        self.ground_weights.append(float(weight))
        self.ground_sigmas.append(float(sigma))
        self.ground_labels.append(text)
        self._ground_tensor = torch.stack(self.ground_centers)

    def _sync_sky_tensor(self):
        if self.sky_centers:
            self._sky_tensor = torch.stack(self.sky_centers)
        else:
            self._sky_tensor = None

    def potential(self, x):
        """全有効ポテンシャル U(x) = H_ground(x) + H_sky(x) (Smooth-Max)"""
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        B = x_norm.shape[0]

        # 1. 地表重力
        val_ground = torch.zeros(B, dtype=x.dtype, device=x.device)
        if self.ground_centers:
            gc = self._ground_tensor
            gw = torch.tensor(self.ground_weights, dtype=x.dtype, device=x.device)
            gs = torch.tensor(self.ground_sigmas, dtype=x.dtype, device=x.device)
            cos_g = torch.matmul(x_norm, gc.T)
            dist_sq_g = 2.0 - 2.0 * cos_g
            val_ground = - torch.sum(gw * torch.exp(- dist_sq_g / (2 * gs**2)), dim=-1)

        # 2. 上空磁石 (LogSumExp Smooth-Max)
        val_sky = torch.zeros(B, dtype=x.dtype, device=x.device)
        if self.sky_centers:
            sc = self._sky_tensor
            sw = torch.tensor(self.sky_weights, dtype=x.dtype, device=x.device)
            ss = torch.tensor(self.sky_sigmas, dtype=x.dtype, device=x.device)
            cos_s = torch.matmul(x_norm, sc.T)
            dist_sq_s = 2.0 - 2.0 * cos_s
            pulls = sw * torch.exp(- dist_sq_s / (2 * ss**2))
            val_sky = - (1.0 / self.temperature) * torch.logsumexp(self.temperature * pulls, dim=-1)

        return val_ground + val_sky

    def roll_marble(self, start_pos, steps=60, lr=0.04, momentum=0.7):
        """ポテンシャル場を粒子が滑走し、捕捉点（アトラクター）を探索"""
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

    def identify_watershed_focus(self, v_0, K=3, steps=15, lr=0.08):
        """全大円直交干渉により、分水嶺上空の焦点 x* を同定"""
        # 初速予測
        F_0 = torch.matmul(self.W1, v_0)
        x_pred_0 = (v_0 + 0.2 * F_0) / torch.norm(v_0 + 0.2 * F_0)

        # 近接する地表基幹アンカー
        cos_sims = torch.matmul(x_pred_0, self._ground_tensor.T)
        topk = torch.topk(cos_sims, min(K, len(self.ground_centers))).indices.tolist()
        anchor_centers = [self.ground_centers[idx] for idx in topk]

        # 直交干渉面法線
        normals = []
        for c_k in anchor_centers:
            proj = torch.dot(c_k, v_0) * v_0
            ortho = c_k - proj
            n_ortho = torch.norm(ortho)
            normals.append(ortho / n_ortho if n_ortho > 1e-5 else torch.randn(self.dim) / 10.0)

        x_curr = x_pred_0.clone().detach().requires_grad_(True)
        sigma_ortho = 0.25
        for _ in range(steps):
            phi = torch.zeros(1)
            for n_k in normals:
                dot_val = torch.dot(x_curr, n_k)
                phi += torch.exp(- (dot_val**2) / (2 * sigma_ortho**2))
            grad = torch.autograd.grad(phi, x_curr)[0]
            proj_grad = grad - torch.dot(x_curr, grad) * x_curr
            x_next = x_curr + lr * proj_grad
            x_curr = (x_next / torch.norm(x_next)).detach().requires_grad_(True)

        return x_curr.detach()

    def learn_concept(self, text, novelty_sim_threshold=0.82):
        """
        オンライン自律概念学習ループ:
        未知テキストを提示されたとき、
        1. 既知地表アンカーへの吸収か？
        2. 既存上空磁石への同化か？
        3. 新たな上空磁石の自律配備か？
        を即座に判定して学習を実行する。
        """
        v_raw = self.embedder.encode(text, convert_to_tensor=True).cpu()
        v_0 = v_raw / torch.norm(v_raw)

        # A. 既存の上空磁石との類似度チェック（同化判定）
        if self.sky_centers:
            sims_sky = torch.matmul(v_0, self._sky_tensor.T)
            best_sky_idx = torch.argmax(sims_sky).item()
            best_sky_sim = sims_sky[best_sky_idx].item()
            if best_sky_sim >= novelty_sim_threshold:
                # 既存磁石のエイリアスとして同化（磁石の引力強度を微増・重心を微調整）
                self.sky_aliases[best_sky_idx].append(text)
                self.sky_counts[best_sky_idx] += 1
                # 重心を少し引き寄せる (指数移動平均)
                alpha = 0.10
                new_c = (1 - alpha) * self.sky_centers[best_sky_idx] + alpha * v_0
                self.sky_centers[best_sky_idx] = new_c / torch.norm(new_c)
                self._sync_sky_tensor()
                return {
                    "action": "ASSIMILATED_TO_SKY",
                    "concept": text,
                    "target_magnet": self.sky_labels[best_sky_idx],
                    "sim": best_sky_sim,
                    "total_magnets": len(self.sky_centers)
                }

        # B. 地表の基幹アンカーとの類似度チェック（基幹概念の直接配下判定）
        sims_ground = torch.matmul(v_0, self._ground_tensor.T)
        best_g_idx = torch.argmax(sims_ground).item()
        best_g_sim = sims_ground[best_g_idx].item()
        if best_g_sim >= 0.88:
            # 基幹アンカーそのもの、あるいは極めて直接的な下位概念
            return {
                "action": "GROUND_ANCHOR_DIRECT",
                "concept": text,
                "target_ground": self.ground_labels[best_g_idx],
                "sim": best_g_sim,
                "total_magnets": len(self.sky_centers)
            }

        # C. 【真の新概念】分水嶺上空に新たな巨大磁石を自律配備！
        x_star = self.identify_watershed_focus(v_0, K=3, steps=15)

        # 磁界パラメータの決定
        min_angle_to_ground = np.arccos(np.clip(best_g_sim, -1.0, 1.0))
        m_weight = 0.80
        m_sigma = max(0.12, float(min_angle_to_ground * 0.38))

        self.sky_centers.append(x_star)
        self.sky_weights.append(m_weight)
        self.sky_sigmas.append(m_sigma)
        self.sky_labels.append(text)
        self.sky_aliases.append([text])
        self.sky_counts.append(1)
        self._sync_sky_tensor()

        return {
            "action": "NEW_SKY_MAGNET_DEPLOYED",
            "concept": text,
            "magnet_idx": len(self.sky_centers) - 1,
            "min_ground_dist_angle": float(min_angle_to_ground),
            "m_sigma": m_sigma,
            "total_magnets": len(self.sky_centers)
        }

    def recall(self, query_text):
        """
        入力クエリが地表の谷に落ちるか、上空の磁石に浮遊捕捉されるかを判定
        """
        v_raw = self.embedder.encode(query_text, convert_to_tensor=True).cpu()
        pos = self.roll_marble(v_raw, steps=60)

        # 磁石群との類似度
        best_sky = None
        if self.sky_centers:
            sims_sky = torch.matmul(pos, self._sky_tensor.T)
            s_idx = torch.argmax(sims_sky).item()
            best_sky = (s_idx, self.sky_labels[s_idx], sims_sky[s_idx].item())

        # 地表アンカーとの類似度
        sims_ground = torch.matmul(pos, self._ground_tensor.T)
        g_idx = torch.argmax(sims_ground).item()
        best_ground = (g_idx, self.ground_labels[g_idx], sims_ground[g_idx].item())

        # 浮遊捕捉 vs 地表吸着の判定
        if best_sky and best_sky[2] >= best_ground[2]:
            return {
                "type": "SKY_LEVITATION",
                "label": best_sky[1],
                "magnet_idx": best_sky[0],
                "sim": best_sky[2],
                "aliases": self.sky_aliases[best_sky[0]],
                "ground_affinity": best_ground
            }
        else:
            return {
                "type": "GROUND_ANCHOR",
                "label": best_ground[1],
                "ground_idx": best_ground[0],
                "sim": best_ground[2],
                "sky_affinity": best_sky
            }
