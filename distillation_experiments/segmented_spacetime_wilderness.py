"""
Segmented Spacetime Wilderness - 3D節分割型・時空思考原野エンジン
------------------------------------------------------------------
概念の全層軌跡 (33層) を4つの「節 (Segments)」に分割:
  - 節 1 (Layer 0〜7)   : 初期形態素・語彙知覚
  - 節 2 (Layer 8〜15)  : 文脈結合・属性展開
  - 節 3 (Layer 16〜23) : 因果相互作用・軌道交差
  - 節 4 (Layer 24〜32) : 抽象化合流・最終着地

時間的に「幾つ目の節か」という時間進行を保ちながら、
3次元空間上で2つの概念の節同士が干渉し合い、
最終着地点（必然的未来アトラクター）へ至る「全フライトパス」をシミュレート・可視化する。
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

class SegmentedSpacetimeWilderness:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"[*] 概念幾何学モデルをロード中: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        print(f"[+] ロード完了: 総層数 = {self.num_layers}, デバイス = {self.device}")

        # 真正概念語彙プールの構築
        print("[*] 真正概念語彙空間を準備中...")
        W_embed = self.model.get_input_embeddings().weight.data.to(torch.float32)
        vocab = self.tokenizer.get_vocab()
        
        purified_words = []
        purified_ids = []
        for raw_s, tid in vocab.items():
            decoded = self.tokenizer.decode([tid])
            s_strip = decoded.strip()
            if not s_strip.isalpha() or not s_strip.isascii():
                continue
            has_space = raw_s.startswith('Ġ') or decoded.startswith(' ')
            if not has_space or len(s_strip) < 3:
                continue
            clean_w = s_strip.lower()
            if clean_w not in purified_words:
                purified_words.append(clean_w)
                purified_ids.append(tid)

        self.words_list = purified_words
        self.ids_tensor = torch.tensor(purified_ids, device=self.device, dtype=torch.long)
        
        # 脱中心化
        selected_embeds = W_embed[self.ids_tensor]
        self.mean_vec = selected_embeds.mean(dim=0, keepdim=True)
        centered_embeds = selected_embeds - self.mean_vec
        self.concept_space = F.normalize(centered_embeds, p=2, dim=-1)
        
        # 3次元投影用の基底 (GPU pca_lowrank で一瞬で上位3主軸を算出)
        print("[*] 3次元思考原野プロジェクション基底 (PCA 3D) を計算中...")
        _, _, V = torch.pca_lowrank(self.concept_space.float(), q=3)
        self.pca_basis_3d = V # (D, 3)
        print(f"[+] 3次元時空思考原野の準備完了 (語彙数: {len(self.words_list)})")

    def extract_segmented_trajectory(self, token_str: str, num_segments=4):
        """
        全層軌跡を num_segments 個の「節」に分割して抽出
        各節の代表ベクトル (層間変化量) と 3D 空間座標を計算
        """
        inputs = self.tokenizer(token_str, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            layers = [h[0, -1, :].to(torch.float32) - self.mean_vec.squeeze(0) for h in out.hidden_states]

        total_layers = len(layers) - 1 # 28
        step = total_layers // num_segments
        
        segments = []
        node_positions_3d = []

        for i in range(num_segments):
            start_l = i * step
            end_l = (i + 1) * step if i < num_segments - 1 else total_layers
            
            h_start = layers[start_l]
            h_end = layers[end_l]
            
            seg_vec = h_end - h_start
            seg_vec = F.normalize(seg_vec, p=2, dim=0)
            segments.append(seg_vec)
            
            pos_3d = torch.matmul(h_end, self.pca_basis_3d).cpu().numpy()
            node_positions_3d.append(pos_3d)

        return segments, np.array(node_positions_3d)

    def simulate_spacetime_interaction(self, word_A: str, word_B: str, num_segments=4):
        """
        2つの単語の節同士を時間発展 (t=1..num_segments) に沿って干渉させ、
        思考粒子が3次元空間を飛行して最終着地点へ到達するプロセスをシミュレート
        """
        print(f"\n========================================================")
        print(f"⏱️ 時空節分割シミュレーション: [{word_A}] + [{word_B}]")
        print(f"========================================================")

        segs_A, pos_A = self.extract_segmented_trajectory(word_A, num_segments)
        segs_B, pos_B = self.extract_segmented_trajectory(word_B, num_segments)

        start_pos_3d = (pos_A[0] + pos_B[0]) / 2.0
        particle_path_3d = [start_pos_3d]
        
        current_state_vec = (segs_A[0] + segs_B[0]) / 2.0
        current_state_vec = F.normalize(current_state_vec, p=2, dim=0)

        segment_logs = []

        # 時間ステップ (節 1 -> 節 2 -> 節 3 -> 節 4)
        for t in range(num_segments):
            seg_a = segs_A[t]
            seg_b = segs_B[t]

            # 節同士の相対角度（この時間帯でどれだけ違う方向を向いているか）
            cos_diff = torch.dot(seg_a, seg_b).item()
            deg_diff = np.degrees(np.arccos(np.clip(cos_diff, -1.0, 1.0)))

            # 直交力線の計算（2つの単語を結ぶ軸に垂直な成分）
            axis = seg_b - seg_a
            axis_norm = F.normalize(axis, p=2, dim=0)
            
            # 相互作用の合力: 前進力 + 時間進行に伴う直交加速
            forward_force = (seg_a + seg_b) / 2.0
            
            # 直交偏向: 時間が進むほど (t が大きいほど) 因果の横方向への分岐・合流が強まる
            time_weight = (t + 1) / num_segments
            ortho_nudge = forward_force - torch.dot(forward_force, axis_norm) * axis_norm
            ortho_nudge = F.normalize(ortho_nudge, p=2, dim=0)
            
            # 状態ベクトルの更新
            current_state_vec = F.normalize(current_state_vec * 0.65 + forward_force * 0.75 + ortho_nudge * (0.45 * time_weight), p=2, dim=0)

            # 3次元座標の更新
            next_pos_3d = torch.matmul(current_state_vec, self.pca_basis_3d).cpu().numpy() * 1.6
            particle_path_3d.append(next_pos_3d)

            log_msg = f"節 {t+1} (Layer {t*7}〜{(t+1)*7}): 相対角 = {deg_diff:.1f}° | 因果偏向度 = {time_weight:.2f}"
            print(f"  ・{log_msg}")
            segment_logs.append(log_msg)

        # 最終着地点の同定（全語彙への逆射影）
        proj_a = torch.dot(current_state_vec, segs_A[-1]) * segs_A[-1]
        proj_b = torch.dot(current_state_vec, segs_B[-1]) * segs_B[-1]
        final_attractor_vec = F.normalize(current_state_vec - 0.45 * (proj_a + proj_b), p=2, dim=0)

        sims = torch.mv(self.concept_space, final_attractor_vec)
        topk = torch.topk(sims, k=30)
        
        pair_set = {word_A.lower(), word_B.lower()}
        retrieved_outcomes = []
        for idx in topk.indices.tolist():
            w = self.words_list[idx]
            if w in pair_set or any(pw in w or w in pw for pw in pair_set):
                continue
            if len(w) >= 3 and w not in retrieved_outcomes:
                retrieved_outcomes.append(w)
            if len(retrieved_outcomes) >= 5:
                break

        print(f"\n🎯 最終着地点（第4節の結末アトラクター）: {retrieved_outcomes}")
        return {
            "pair": [word_A, word_B],
            "pos_A": pos_A,
            "pos_B": pos_B,
            "particle_path": np.array(particle_path_3d),
            "outcomes": retrieved_outcomes,
            "logs": segment_logs
        }

def run_spacetime_demo():
    wilderness = SegmentedSpacetimeWilderness("Qwen/Qwen2.5-1.5B-Instruct")

    test_pairs = [
        ("crime", "police"),
        ("cloud", "cold"),
        ("fire", "wood"),
        ("group", "commutative")
    ]

    all_results = []
    
    fig = plt.figure(figsize=(18, 12))

    for idx, (w1, w2) in enumerate(test_pairs, 1):
        res = wilderness.simulate_spacetime_interaction(w1, w2, num_segments=4)
        all_results.append(res)

        # 3D サブプロット描画
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        pA = res["pos_A"]
        pB = res["pos_B"]
        pP = res["particle_path"]

        # 単語Aの節ごとの軌道 (青)
        ax.plot(pA[:, 0], pA[:, 1], pA[:, 2], 'b-o', linewidth=2, label=f"[{w1}] Trajectory A")
        for seg_i, (x, y, z) in enumerate(pA):
            ax.text(x, y, z, f"A{seg_i+1}", color='blue', fontsize=8)

        # 単語Bの節ごとの軌道 (赤)
        ax.plot(pB[:, 0], pB[:, 1], pB[:, 2], 'r-o', linewidth=2, label=f"[{w2}] Trajectory B")
        for seg_i, (x, y, z) in enumerate(pB):
            ax.text(x, y, z, f"B{seg_i+1}", color='red', fontsize=8)

        # 思考粒子の時間発展フライトパス (緑・金)
        ax.plot(pP[:, 0], pP[:, 1], pP[:, 2], 'g--s', linewidth=2.5, label="Causal Particle Path")
        for seg_i, (x, y, z) in enumerate(pP):
            label = "Start" if seg_i == 0 else f"Node {seg_i}"
            ax.text(x, y, z, label, color='green', fontweight='bold', fontsize=9)

        # 最終着地点のマーカー
        ax.scatter([pP[-1, 0]], [pP[-1, 1]], [pP[-1, 2]], color='gold', s=160, edgecolors='black', label=f"Landing: {res['outcomes'][0]}")

        ax.set_title(f"Spacetime Trajectory: [{w1}] + [{w2}]\n➔ Outcome: {res['outcomes'][:3]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("X (Causal Curvature)")
        ax.set_ylabel("Y (Hierarchy Depth)")
        ax.set_zlabel("Z (Orthogonal Shift)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_png = "distillation_experiments/segmented_spacetime_wilderness_3d.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n[+] 3次元時空節分割プロットを保存しました: {output_png}")

if __name__ == "__main__":
    run_spacetime_demo()
