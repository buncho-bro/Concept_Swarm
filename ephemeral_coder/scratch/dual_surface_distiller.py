import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 脱重力化・概念蒸留パイプライン (Zero-Gravity Concept Distillation Pipeline)
# ==============================================================================

class OrthogonalDecoupledExtractor:
    """
    【フェーズ1: 抽出の脱重力化】
    Teacherモデル空間から概念核を抽出する際、
    近接概念群の相互引力（重力ドリフト）を Gram-Schmidt 直交化 ＋ アンチドリフト正規化で
    完全にキャンセルし、純粋な真の概念焦点（ズレ0°付近）をピンポイント抽出する。
    """
    def __init__(self, dim=384):
        self.dim = dim
        self.W_proj = torch.randn(dim, dim) * 0.05

    def extract_pure_core(self, target_vec, neighbor_vecs=None, steps=25, lr=0.08, sigma_ortho=0.25):
        """
        target_vec: 抽出対象の真のTeacherベクトル (D,)
        neighbor_vecs: 周囲に存在する他概念ベクトル群 (K, D)
        """
        v_0 = (target_vec / torch.norm(target_vec)).clone().detach()

        # 1. 初速予測
        F_0 = torch.matmul(self.W_proj, v_0)
        x_pred_0 = (v_0 + 0.1 * F_0) / torch.norm(v_0 + 0.1 * F_0)

        # 2. 周囲の他概念から直交干渉法線を構築し、Gram-Schmidtで相互引力をデカップリング
        normals = []
        if neighbor_vecs is not None and len(neighbor_vecs) > 0:
            for c_k in neighbor_vecs:
                proj = torch.dot(c_k, v_0) * v_0
                ortho = c_k - proj
                # 既存法線との相互引力成分を除去
                for prev_n in normals:
                    ortho = ortho - torch.dot(ortho, prev_n) * prev_n
                n_norm = torch.norm(ortho)
                if n_norm > 1e-4:
                    normals.append(ortho / n_norm)

        # 法線が存在しない場合はランダム直交基底
        if len(normals) == 0:
            for _ in range(3):
                rand_v = torch.randn(self.dim)
                rand_v = rand_v - torch.dot(rand_v, v_0) * v_0
                normals.append(rand_v / torch.norm(rand_v))

        # 3. 直交干渉反復 (Smooth-Max LogSumExp + アンチドリフト中心射影)
        x_curr = x_pred_0.clone().detach().requires_grad_(True)
        for it in range(steps):
            pulls = []
            for n_k in normals:
                dot_val = torch.dot(x_curr, n_k)
                pulls.append(torch.exp(- (dot_val**2) / (2 * sigma_ortho**2)))
            
            # Smooth-Max で特定方向への重力偏りを相殺
            phi = torch.logsumexp(torch.stack(pulls) * 4.0, dim=0) / 4.0
            grad = torch.autograd.grad(phi, x_curr)[0]

            # 接平面射影
            proj_grad = grad - torch.dot(x_curr, grad) * x_curr

            # アンチドリフト制約: v_0 から離れすぎる成分にペナルティ
            drift_component = x_curr - torch.dot(x_curr, v_0) * v_0
            anti_drift_restore = - 0.2 * drift_component

            x_next = x_curr + lr * proj_grad + anti_drift_restore
            x_curr = (x_next / torch.norm(x_next)).detach().requires_grad_(True)

        return x_curr.detach()


class DualSurfaceDistillationProcessor:
    """
    【フェーズ2: 蒸留の磁気浮上化】
    Studentモデルの地表面（ベース語彙・言語知識）を1ミリも削らない。
    生徒の潜在／ロジット空間の上空に、抽出した概念核群を
    「磁気浮上トラップ（LogSumExp Smooth-Max）」として配備。
    何十・何百の概念を蒸留しても重力崩壊（ブラックホール化）を完全阻止する。
    """
    def __init__(self, vocab_embeddings, temperature=5.0, alpha=3.0):
        """
        vocab_embeddings: Studentモデルの語彙埋め込み (V, D)
        """
        self.vocab_emb = F.normalize(vocab_embeddings, p=2, dim=-1)
        self.V, self.D = self.vocab_emb.shape
        self.temperature = temperature
        self.alpha = alpha

        # 上空磁石群 (Concept Magnets)
        self.magnets = []    # List of Tensor(D,)
        self.weights = []    # List of float
        self.sigmas = []     # List of float
        self.labels = []     # List of str

        self._magnet_tensor = None

    def add_magnet(self, core_vector, label="concept", weight=1.0, sigma=0.22):
        core = core_vector / torch.norm(core_vector)
        self.magnets.append(core.clone().detach())
        self.weights.append(float(weight))
        self.sigmas.append(float(sigma))
        self.labels.append(label)
        self._magnet_tensor = torch.stack(self.magnets)

    def compute_levitation_bias(self, current_context_vec=None):
        """
        語彙空間全体に対する上空磁気浮上バイアス B_vocab (V,) を計算。
        各磁石は Smooth-Max により局所排他化され、重積ブラックホール化しない。
        """
        if not self.magnets:
            return torch.zeros(self.V, device=self.vocab_emb.device)

        # 語彙ベクトルと各磁石とのコサイン類似度: (V, N_magnets)
        cos_sims = torch.matmul(self.vocab_emb, self._magnet_tensor.T)
        dist_sq = 2.0 - 2.0 * cos_sims # (V, N)

        w_t = torch.tensor(self.weights, dtype=self.vocab_emb.dtype, device=self.vocab_emb.device)
        s_t = torch.tensor(self.sigmas, dtype=self.vocab_emb.dtype, device=self.vocab_emb.device)

        # 各磁石の引力強度: (V, N)
        magnet_pulls = w_t * torch.exp(- dist_sq / (2.0 * s_t**2))

        # コンテキストがある場合、近接磁石をさらに選択的活性化
        if current_context_vec is not None:
            c_norm = current_context_vec / torch.norm(current_context_vec)
            ctx_affinity = torch.matmul(self._magnet_tensor, c_norm) # (N,)
            gate = torch.sigmoid(5.0 * (ctx_affinity - 0.3)) # (N,)
            magnet_pulls = magnet_pulls * gate.unsqueeze(0)

        # Smooth-Max (LogSumExp) による局所排他合成
        # 磁石同士が足し算で無限に深くならない！
        levitation_bias = (1.0 / self.temperature) * torch.logsumexp(self.temperature * magnet_pulls, dim=-1)

        return self.alpha * levitation_bias

    def apply_to_logits(self, base_logits, current_context_vec=None):
        """
        Studentの出力ロジットに磁気浮上バイアスを合成
        logits = base_logits + levitation_bias
        """
        bias = self.compute_levitation_bias(current_context_vec)
        return base_logits + bias.to(base_logits.device)
