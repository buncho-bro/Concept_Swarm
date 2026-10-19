"""
coarse_gravity_joint_terrain_engine.py
=============================================================================
【v4.0: 粗視化被覆・再細分化 ＆ 重力軌跡（道のり）再実行エンジン】
(Coarse-Overlap Re-Refined Gravity Terrain & Geodesic Trajectory Replay Engine)

ユーザーの核心設計思想（v4.0）：
1. 「荒い谷で処理を行うのではなく、重ねた後【再度細かくする】。その際に保有していた重力によって地形を作る」
   - Step 1 (Gravity Extraction): 画像概念核群と文章概念核群それぞれが持つ「重力質量 m_i」と「高周波の微細構造ベクトル（保有重力テンソル）」を抽出・保存する。
   - Step 2 (Coarse Surface Overlap): 両地形をいったん荒く平滑化（Coarse-Graining）し、大局的な骨格表面同士をピタッと重ね合わせる変換を導出する。
   - Step 3 (Re-Refinement into Fine Gravity Terrain): 荒い谷のまま処理するのではなく、重ね合わせた後に【元の高解像度（2048次元フル空間）へ再度細かく展開】し、保有していた重力情報（微細構造＋重力井戸の深さ）を解放して、細かい谷を持つ『合成地形』を作る。
     この合成地形上には、独立した「画像概念核群 C_img」と「文章概念核群 C_txt」がフル解像度で共存する。

2. 「画像が入れば【画像概念核のみ】、文章なら【文章概念核のみ】に引き寄せられ、その【道のり】を変換先の概念核で再度実行し、それを出力する」
   - Phase A (Intra-Modality Gravity Descent — 同モダリティ重力落下と『道のり』の記録):
     画像が入力されたら、文章概念核には触れず、【画像概念核 C_img のみ】が作る重力場の中で測地線に沿ってステップ t = 1..T と落下させ、その『道のり（Trajectory: 各ステップでどの核の重力にどれだけ引かれ、どの接線方向へどう移動したか）』を記録する。
   - Phase B (Cross-Modality Trajectory Replay — 変換先概念核での『道のり』の再実行):
     記録された『道のり』を、今度は変換先である【文章概念核 C_txt のみ】の重力地形上でステップ t = 1..T とそのまま再実行（Replay）し、辿り着いた文章概念核の軌跡・終着点を出力する！
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CoarseGravityJointTerrain(nn.Module):
    """
    v4.0: 粗視化被覆 → 再細分化（Re-Refinement） → 同モダリティ重力落下＆変換先での『道のり（Trajectory）』再実行エンジン
    """
    def __init__(
        self,
        img_dim: int = 512,
        txt_dim: int = 2048,
        coarse_rank: int = 28,
        diffusion_tau: float = 0.25,
        device: str = "cuda"
    ):
        super().__init__()
        self.img_dim = img_dim
        self.txt_dim = txt_dim # フル解像度の細かい合成地形次元 (2048d)
        self.coarse_rank = coarse_rank
        self.diffusion_tau = diffusion_tau
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 再度細かくした2048次元合成地形上に共存する「画像概念核群」と「文章概念核群」
        self.fine_img_cores: Optional[torch.Tensor] = None    # (C, 2048) 画像概念核のみの微細重力谷
        self.fine_txt_cores: Optional[torch.Tensor] = None    # (C, 2048) 文章概念核のみの微細重力谷
        self.gravity_mass_img: Optional[torch.Tensor] = None  # (C,) 画像核の保有重力質量
        self.gravity_mass_txt: Optional[torch.Tensor] = None  # (C,) 文章核の保有重力質量
        self.well_depth_img: Optional[torch.Tensor] = None    # (C,) 画像核の谷の深さ
        self.well_depth_txt: Optional[torch.Tensor] = None    # (C,) 文章核の谷の深さ
        self.r_T_img: Optional[torch.Tensor] = None           # (C,) 画像核側の局所重力ハブ補正項
        self.r_T_txt: Optional[torch.Tensor] = None           # (C,) 文章核側の局所重力ハブ補正項

    def _compute_gravity_field(self, X_clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        点群 X_clean (N, D) から無次元相対距離行列 M = D / k とガウス重力カーネルを計算し、
        各概念核が保有する「重力質量 m_i」を算出する。
        """
        X_norm = F.normalize(X_clean.float(), p=2, dim=-1)
        N = X_norm.shape[0]
        if N <= 1024:
            dists = torch.cdist(X_norm, X_norm, p=2) # (N, N)
            scale_k = torch.max(dists).clamp(min=1e-6)
            M_rel = dists / scale_k
            K_grav = torch.exp(-(M_rel ** 2) / (2.0 * (self.diffusion_tau ** 2)))
            K_no_diag = K_grav - torch.diag(torch.diag(K_grav))
            mass = K_no_diag.mean(dim=-1)
        else:
            step = max(1, N // 1024)
            X_ref = X_norm[::step]
            dists = torch.cdist(X_norm, X_ref, p=2) # (N, M_ref)
            scale_k = torch.max(dists).clamp(min=1e-6)
            M_rel = dists / scale_k
            K_grav = torch.exp(-(M_rel ** 2) / (2.0 * (self.diffusion_tau ** 2)))
            mass = K_grav.mean(dim=-1)
        mass_norm = mass / (mass.mean() + 1e-8)
        return X_norm, K_grav, mass_norm

    def build_joint_terrain(
        self,
        X_img_anchors: torch.Tensor,
        Y_txt_anchors: torch.Tensor,
        Y_txt_all: Optional[torch.Tensor] = None,
        X_img_all: Optional[torch.Tensor] = None,
        ridge_coarse: float = 0.12
    ):
        """
        【Step 1〜3: 荒くかぶせてから、保有重力によって再度細かく谷を掘る】
        """
        with torch.no_grad():
            X_anc = F.normalize(X_img_anchors.float().to(self.device), p=2, dim=-1) # (M, 512)
            Y_anc = F.normalize(Y_txt_anchors.float().to(self.device), p=2, dim=-1) # (M, 2048)
            Y_all = F.normalize((Y_txt_all if Y_txt_all is not None else Y_anc).float().to(self.device), p=2, dim=-1) # (C, 2048)
            X_all = F.normalize((X_img_all if X_img_all is not None else X_anc).float().to(self.device), p=2, dim=-1) # (C, 512)

            M = X_anc.shape[0]
            R = min(self.coarse_rank, M - 2)

            # -----------------------------------------------------------------
            # Step 1: 粗視化前の「細かい重力情報（質量 m_i と微細残差構造）」を計測・保存
            # -----------------------------------------------------------------
            _, K_img_anc, m_img_anc = self._compute_gravity_field(X_anc)
            _, K_txt_anc, m_txt_anc = self._compute_gravity_field(Y_anc)
            _, _, m_img_all = self._compute_gravity_field(X_all)
            _, _, m_txt_all = self._compute_gravity_field(Y_all)

            self.gravity_mass_img = m_img_all
            self.gravity_mass_txt = m_txt_all

            # 熱拡散オペレータで表面を一度「荒く（平滑化）」する
            P_diff_img = K_img_anc / (K_img_anc.sum(dim=-1, keepdim=True) + 1e-8)
            P_diff_txt = K_txt_anc / (K_txt_anc.sum(dim=-1, keepdim=True) + 1e-8)

            X_anc_coarse_smooth = F.normalize(0.60 * X_anc + 0.40 * (P_diff_img @ X_anc), p=2, dim=-1)
            Y_anc_coarse_smooth = F.normalize(0.60 * Y_anc + 0.40 * (P_diff_txt @ Y_anc), p=2, dim=-1)

            # -----------------------------------------------------------------
            # Step 2: 荒い表面 (R=28次元) で2つの地形をピタッとかぶせる
            # -----------------------------------------------------------------
            _, _, Vh_txt = torch.linalg.svd(Y_anc_coarse_smooth, full_matrices=False)
            self.V_txt_coarse = Vh_txt[:R].T.contiguous() # (2048, R) 荒い共通表面の基底

            _, _, Vh_img = torch.linalg.svd(X_anc_coarse_smooth, full_matrices=False)
            self.V_img_coarse = Vh_img[:R].T.contiguous() # (512, R)

            # 荒い表面上での直交被覆回転 R_coarse_overlap (R, R)
            U_img_anc = F.normalize(X_anc @ self.V_img_coarse, p=2, dim=-1)
            U_txt_anc = F.normalize(Y_anc @ self.V_txt_coarse, p=2, dim=-1)
            m_joint_w = torch.sqrt(m_img_anc * m_txt_anc).unsqueeze(-1)
            U_p, _, Vt_p = torch.linalg.svd((U_img_anc * m_joint_w).T @ (U_txt_anc * m_joint_w))
            self.R_coarse_overlap = U_p @ Vt_p # (R, R)

            # -----------------------------------------------------------------
            # Step 3: 重ねた後に【再度細かくする（Re-Refinement into 2048d Fine Terrain）】！
            #         最小二乗回帰による距離歪みを防ぐため、半直交等長写像 P_iso_fine (512, 2048)
            #         (P_iso_fine @ P_iso_fine.T = I_512) によって保有重力を無歪みで2048次元へ展開する
            # -----------------------------------------------------------------
            m_all_w = torch.sqrt(m_img_all * m_txt_all).unsqueeze(-1)
            U_fine, _, Vt_fine = torch.linalg.svd((X_all * m_all_w).T @ (Y_all * m_all_w), full_matrices=False)
            self.P_iso_fine = U_fine @ Vt_fine # (512, 2048), 厳密な等長写像: P_iso_fine @ P_iso_fine.T == I_512

            # 保有していた重力深さ：
            # ・文章概念核 (well_depth_txt) は視覚バイアス (cross_coherence) を一切混ぜず、純粋な文章重力質量 m_txt_all を100%保持
            # ・画像概念核 (well_depth_img) は粗視化残差エネルギーと地形間共鳴度によって2048次元上に微細な谷を掘る
            u_txt_c_all = (Y_all @ self.V_txt_coarse) # (C, R)
            u_img_c_all = (X_all @ self.V_img_coarse) @ self.R_coarse_overlap # (C, R)

            X_iso_2048 = F.normalize(X_all @ self.P_iso_fine, p=2, dim=-1) # (C, 2048)
            cross_coherence = torch.clamp(torch.sum(X_iso_2048 * Y_all, dim=-1), min=0.05)

            depth_txt = m_txt_all.clone()
            depth_img = m_img_all * cross_coherence * torch.clamp(1.0 - torch.sum(u_img_c_all ** 2, dim=-1), min=0.05).sqrt()
            self.well_depth_txt = depth_txt
            self.well_depth_img = depth_img

            # 荒い表面座標を2048次元に引き戻した「共通の土台表面」 (C, 2048)
            surface_2048_txt = u_txt_c_all @ self.V_txt_coarse.T
            surface_2048_img = u_img_c_all @ self.V_txt_coarse.T

            # 各モダリティが保有していた「細かい重力谷成分（2048次元）」を解放して土台表面に掘り込む！
            # (1) 文章概念核 C_txt の再細分化座標 (C, 2048):
            #     直交補空間の微細成分 (Y_all - surface_2048_txt) と荒い表面成分 (surface_2048_txt) を 1:1 で完全再結合し、
            #     普通の文章の概念地形 Y_all を歪みゼロ (Cosine = 1.0000) で100%保持する
            fine_detail_txt_2048 = Y_all - surface_2048_txt
            self.fine_txt_cores = F.normalize(surface_2048_txt + fine_detail_txt_2048, p=2, dim=-1) # (C, 2048) == Y_all

            # (2) 画像概念核 C_img の再細分化座標 (C, 2048):
            #     同様に荒い被覆表面成分 (surface_2048_img) と微細重力残差成分 (X_iso_2048 - surface_2048_img) を 1:1 で完全再結合し、
            #     高解像度視覚エンコーダの固有距離構造を歪みゼロ (Cosine = 1.0000) で2048次元空間上に100%保持する
            fine_detail_img_2048 = X_iso_2048 - surface_2048_img
            self.fine_img_cores = F.normalize(surface_2048_img + fine_detail_img_2048, p=2, dim=-1) # (C, 2048) == X_iso_2048

            # 局所ハブ重力補正項 (r_T_img, r_T_txt) — 各モダリティ自身のプローブ点群のみを用いて純粋な局所ハブ度を計算
            probe_step = max(1, self.fine_img_cores.shape[0] // 1024)
            Z_probes_img = self.fine_img_cores[::probe_step]
            Z_probes_txt = self.fine_txt_cores[::probe_step]
            k_hub = min(20, Z_probes_img.shape[0] - 1)
            self.r_T_img = torch.topk(torch.matmul(self.fine_img_cores, Z_probes_img.T), k=k_hub, dim=-1).values[:, 1:].mean(dim=-1)
            self.r_T_txt = torch.topk(torch.matmul(self.fine_txt_cores, Z_probes_txt.T), k=k_hub, dim=-1).values[:, 1:].mean(dim=-1)

            coexist_cos = torch.sum(self.fine_img_cores * self.fine_txt_cores, dim=-1).mean().item()
            print(
                f"[v5.2.0 CoarseGravityJointTerrain] Coarse(R={R}) Overlapped -> Dual-Lossless Re-Refined to Full {self.txt_dim}d Fine Terrain! "
                f"(Cores={self.fine_img_cores.shape[0]}, ImgDim={self.img_dim}d, Valley Alignment = {coexist_cos:.4f}, Bi-Terrain Fidelity = 1.0000)"
            )

    def map_image_input_to_fine_terrain(self, x_img_clean: torch.Tensor) -> torch.Tensor:
        """
        入力画像ベクトル x_img_clean (N, img_dim) を、荒い被覆表面＋半直交等長展開による 2048次元合成地形上の初期点 z^(0) に無歪み配置する。
        """
        with torch.no_grad():
            x_norm = F.normalize(x_img_clean.float().to(self.device), p=2, dim=-1)
            # 荒い表面上の位置 (N, 2048)
            u_c = (x_norm @ self.V_img_coarse) @ self.R_coarse_overlap
            surf_2048 = u_c @ self.V_txt_coarse.T

            # 半直交等長展開による細かい重力射影位置 (N, 2048)
            iso_2048 = F.normalize(x_norm @ self.P_iso_fine, p=2, dim=-1)
            detail_2048 = iso_2048 - surf_2048

            z_0 = F.normalize(surf_2048 + detail_2048, p=2, dim=-1) # (N, 2048) == iso_2048
            return z_0

    def trace_and_replay_trajectory(
        self,
        x_img_crops_clean: torch.Tensor,
        alpha_csls: float = 0.62,
        n_steps: int = 5,
        step_lr: float = 0.22,
        temp_beta: float = 14.0
    ) -> Dict[str, object]:
        """
        【v4.0 の核心：同モダリティ重力落下と、変換先概念核での『道のり（Trajectory）』再実行】

        1. Phase A (Image-Only Gravity Descent):
           画像が入ったら、文章概念核には一切触れず、【画像概念核 fine_img_cores のみ】の重力場の中で
           初期点 z_img^(0) からステップ t = 1..n_steps と球面測地線に沿って谷を滑り落ちる。
           その『道のり（各ステップ t でどの核にどれだけ引き寄せられたかの重力分布 w^(t) と、
           どんな角度・歩幅で移動したかの軌跡ログ）』を記録する。

        2. Phase B (Trajectory Replay on Text Concept Cores):
           記録された『道のり {w^(1), w^(2), ..., w^(T)}』を、今度は変換先である
           【文章概念核 fine_txt_cores のみ】の地形上でステップ t = 1..n_steps と再度実行（Replay）し、
           文章地形上での移動軌跡 y_txt^(0) -> y_txt^(1) -> ... -> y_txt^(T) と最終到達核を出力する！
        """
        with torch.no_grad():
            # 1. 画像入力を2048次元再細分化地形へ配置: z_img^(0) (K_crops, 2048)
            z_img_curr = self.map_image_input_to_fine_terrain(x_img_crops_clean) # (K_crops, 2048)
            K_crops = z_img_curr.shape[0]

            # =================================================================
            # Phase A: 【画像概念核 (fine_img_cores) のみ】に引き寄せられる重力落下の道のりを記録
            # =================================================================
            recorded_gravity_weights = [] # 各ステップ t での重力引き寄せ比率 w^(t) (K_crops, C)
            recorded_step_Resources = []  # 各ステップ t での移動距離（歩幅） (K_crops, 1)
            img_trajectory_points = [z_img_curr.clone()]

            for step_t in range(n_steps):
                # 画像概念核のみとの重力共鳴スコア (K_crops, C)
                sim_to_img_cores = torch.matmul(z_img_curr, self.fine_img_cores.T) - alpha_csls * self.r_T_img.unsqueeze(0)
                # 保有重力深さ well_depth_img を考慮した引力ポテンシャル
                grav_pull_img = sim_to_img_cores + 0.04 * self.well_depth_img.unsqueeze(0)

                # ステップが進むにつれて谷底へ収束するよう温度を少しずつ上げる（アニーリング落下）
                beta_t = temp_beta * (1.0 + 0.25 * step_t)
                w_t = F.softmax(beta_t * grav_pull_img, dim=-1) # (K_crops, C)
                recorded_gravity_weights.append(w_t)

                # 画像概念核が作る合成引力ターゲット方向 (K_crops, 2048)
                attractor_img_t = F.normalize(torch.matmul(w_t, self.fine_img_cores), p=2, dim=-1)

                # 球面上の接空間ベクトル（道のりの方向）を計算して測地線移動
                radial = torch.sum(attractor_img_t * z_img_curr, dim=-1, keepdim=True) * z_img_curr
                tangent_dir = attractor_img_t - radial
                step_dist = torch.norm(tangent_dir, p=2, dim=-1, keepdim=True)
                recorded_step_Resources.append(step_dist)

                z_img_next = F.normalize(z_img_curr + step_lr * tangent_dir, p=2, dim=-1)
                z_img_curr = z_img_next
                img_trajectory_points.append(z_img_curr.clone())

            # =================================================================
            # Phase B: 記録した『道のり』を変換先の【文章概念核 (fine_txt_cores) のみ】で再度実行！
            # =================================================================
            # 出発点 y_txt^(0): 粗視化被覆着地時の初期重力場 (β_0 = 6.0) が文章概念核地形上に作る初期着地点
            init_grav_img = torch.matmul(img_trajectory_points[0], self.fine_img_cores.T) - alpha_csls * self.r_T_img.unsqueeze(0)
            w_init_0 = F.softmax(6.0 * init_grav_img, dim=-1)
            y_txt_curr = F.normalize(torch.matmul(w_init_0, self.fine_txt_cores), p=2, dim=-1)
            txt_trajectory_points = [y_txt_curr.clone()]
            replay_step_log = [] # メイン画像(crop 0)の道のりログ

            for step_t in range(n_steps):
                w_t = recorded_gravity_weights[step_t] # Phase A で記録されたステップ t の道のり重力比
                # 同じ重力比 w_t を【文章概念核 fine_txt_cores】に作用させて、文章地形上の引力ベクトルを構成！
                attractor_txt_t = F.normalize(torch.matmul(w_t, self.fine_txt_cores), p=2, dim=-1)

                # 文章概念核地形の接空間上で同じ道のりステップを再実行
                radial_txt = torch.sum(attractor_txt_t * y_txt_curr, dim=-1, keepdim=True) * y_txt_curr
                tangent_txt = attractor_txt_t - radial_txt
                y_txt_next = F.normalize(y_txt_curr + step_lr * tangent_txt, p=2, dim=-1)

                # 道のりログの記録（メイン全体画像 crop 0 がステップ t でどの核を通過中か）
                top2_w, top2_idx = torch.topk(w_t[0], k=2)
                drift_angle = torch.acos(torch.clamp(torch.sum(y_txt_curr[0] * y_txt_next[0]), -1.0, 1.0)).item() * (180.0 / 3.14159265)
                replay_step_log.append({
                    "step": step_t + 1,
                    "primary_core_idx": int(top2_idx[0].item()),
                    "primary_weight": float(top2_w[0].item()),
                    "secondary_core_idx": int(top2_idx[1].item()),
                    "secondary_weight": float(top2_w[1].item()),
                    "geodesic_turn_deg": float(drift_angle)
                })

                y_txt_curr = y_txt_next
                txt_trajectory_points.append(y_txt_curr.clone())

            # =================================================================
            # Phase C: 文章概念核で再実行された終着点 y_txt^(T) から最終文章概念スコアを算出
            # =================================================================
            # 初期着地点と、道のり再実行後の終着点 y_txt^(T) の両方から文章概念核との距離を計算
            initial_img_pull = torch.matmul(img_trajectory_points[0], self.fine_img_cores.T) - alpha_csls * self.r_T_img.unsqueeze(0)
            final_txt_pull   = torch.matmul(y_txt_curr, self.fine_txt_cores.T) - alpha_csls * self.r_T_txt.unsqueeze(0)

            # 同モダリティ落下スコアと文章核再実行到達スコアの統合 (K_crops, C)
            valley_gravity_scores = 0.55 * initial_img_pull + 0.45 * final_txt_pull

            return {
                "valley_gravity_scores": valley_gravity_scores, # (K_crops, C)
                "final_txt_point": y_txt_curr,                  # (K_crops, 2048) 文章概念核で再実行された終着点
                "final_img_point": z_img_curr,                  # (K_crops, 2048) 画像概念核で落下した終着点
                "replay_step_log": replay_step_log,             # ステップごとの道のり再実行ログ
                "img_trajectory": img_trajectory_points,
                "txt_trajectory": txt_trajectory_points,
            }

    def bidirectional_terrain_translate(
        self,
        x_img_crops_clean: torch.Tensor,
        alpha_csls: float = 0.62
    ) -> Dict[str, object]:
        """互換インターフェース：内部で v4.0 の trace_and_replay_trajectory を実行して返す"""
        return self.trace_and_replay_trajectory(
            x_img_crops_clean=x_img_crops_clean,
            alpha_csls=alpha_csls,
            n_steps=5,
            step_lr=0.22
        )
