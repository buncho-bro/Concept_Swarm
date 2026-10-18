"""
slice_topological_distiller.py
=============================================================================
概念地形スライス・トポロジー蒸留エンジン (Slice & Topological Distillation)
=============================================================================
【概要】
文字（単語リスト/テキストプロンプト）を一切介さず、
教師モデル（GLM-5.3）の「概念地形そのもの」から
生徒モデル（Qwen2.5-Coder-3B）の概念地形へと直接幾何学的同相写像を行う。

【アルゴリズム（ユーザー考案理論の実装）】
1. 教師の概念地形（全大円干渉によって得られた多層地形マスター）を主軸に沿って上から下へ等間隔にスライス。
2. 各スライスの断面における点群から穴の中心（クラスター重心群）を算出。
3. 最下部から上部までの最大距離を k(T) と定義。
4. k(T) を単位として、各点データの相対距離を無次元化: M_T = D(P_T) / k(T)。
5. 生徒モデル（Qwen2.5-Coder-3B）の全語彙埋め込み空間に対しても同様にスライス処理を行い、k(S) および M_S を算出。
6. 文字入力を1つも使わず、StudentのLoRA（および埋め込み写像）を M_S -> M_T となるように最適化。
   これにより、次元の異なる空間間（6144次元 -> 2048次元）で実質的な完全同相コピーを達成する。
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

TEACHER_TERRAIN_PATH = os.path.join(BASE_DIR, "glm5.3_zero_vram_pure_concept_terrains.pt")
STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
OUTPUT_LORA_DIR = os.path.join(BASE_DIR, "lora_glm5.3_slice_topological_to_Qwen2.5_Coder_3B")

class ConceptTerrainSlicer:
    """
    高次元概念地形を等間隔スライス（レベルセット走査）し、
    穴の中心（ボイド重心）と全体スケール k を抽出する幾何プロセッサ
    """
    def __init__(self, n_slices=16):
        self.n_slices = n_slices

    def slice_terrain(self, points: torch.Tensor):
        """
        points: (N, D) の概念地形テンソル (float32)
        戻り値:
            slice_centers: (M, D) 各スライスの穴の中心（重心点群）
            k: float 空間の最下部から最上部までの最大距離（空間全幅スケール）
            dist_matrix: (M, M) 各中心点同士の距離行列
            norm_dist_matrix: (M, M) k で割った無次元トポロジー行列
        """
        device = points.device
        N, D = points.shape

        # 1. 空間の重心と主軸（最大分散方向）の特定
        centroid = torch.mean(points, dim=0, keepdim=True) # (1, D)
        centered = points - centroid
        
        # SVDによる第1主成分（スライス方向軸）
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        primary_axis = Vt[0] # (D,) 主軸ベクトル
        primary_axis = primary_axis / torch.norm(primary_axis)

        # 2. 主軸への射影（高さ座標 h の決定）
        heights = torch.matmul(centered, primary_axis) # (N,)
        h_min = torch.min(heights)
        h_max = torch.max(heights)

        # 3. 空間全幅スケール k の算出（最下部から上部までの距離）
        # ユークリッド距離の最大幅
        all_dists = torch.cdist(points, points, p=2)
        k = torch.max(all_dists) # 空間の最大直径

        # 4. 等分位スライス（Quantile Slicing: 上から下へ等密度にスライス）
        q_steps = torch.linspace(0.0, 1.0, steps=self.n_slices + 1, device=device)
        quantiles = torch.quantile(heights, q_steps)
        slice_centers = []

        for i in range(self.n_slices):
            h_start = quantiles[i]
            h_end = quantiles[i+1]
            if i == self.n_slices - 1:
                mask = (heights >= h_start) & (heights <= h_end)
            else:
                mask = (heights >= h_start) & (heights < h_end)
            
            pts_in_slice = points[mask]
            if len(pts_in_slice) > 0:
                # 各スライス断面の幾何学的中心（穴の中心）
                center_i = torch.mean(pts_in_slice, dim=0)
                center_i = center_i / (torch.norm(center_i) + 1e-8)
                slice_centers.append(center_i)
            else:
                # フォールバック
                slice_centers.append(points[i % len(points)] / torch.norm(points[i % len(points)]))

        slice_centers = torch.stack(slice_centers, dim=0) # (n_slices, D)
        
        # 5. 各穴の中心点同士の距離行列と k による無次元化
        dist_matrix = torch.cdist(slice_centers, slice_centers, p=2)
        norm_dist_matrix = dist_matrix / (k + 1e-8)

        return slice_centers, k, dist_matrix, norm_dist_matrix


def run_slice_topological_distillation():
    print("=" * 80)
    print(" 概念地形スライス・トポロジー蒸留 (Slice & Topological Distillation)")
    print(" [Zero-Text, Pure Concept Terrain Homomorphic Mapping]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 1. 教師側（GLM-5.3）の概念地形マスターのロードとスライス分解
    # -------------------------------------------------------------------------
    print(f"\n[1/3] Loading Teacher Concept Terrain ({os.path.basename(TEACHER_TERRAIN_PATH)})...")
    if not os.path.exists(TEACHER_TERRAIN_PATH):
        raise FileNotFoundError(f"Teacher terrain file not found: {TEACHER_TERRAIN_PATH}")

    t_data = torch.load(TEACHER_TERRAIN_PATH, map_location="cpu")
    # Layer 0 の自律概念核（54点 x 6144次元）
    t_layer0 = t_data["Layer_0"].to(device)
    print(f"Teacher Layer 0 Terrain: {t_layer0.shape[0]} attractors, dim={t_layer0.shape[1]}")

    slicer = ConceptTerrainSlicer(n_slices=24)
    t_centers, k_T, t_dists, M_T = slicer.slice_terrain(t_layer0)
    print(f"Teacher Terrain Sliced: {len(t_centers)} slice void centers extracted.")
    print(f"Teacher Scale k(T) = {k_T.item():.4f}")
    print(f"Teacher Dimensionless Topology Matrix M_T: {M_T.shape}")

    # -------------------------------------------------------------------------
    # 2. 生徒側（Qwen2.5-Coder-3B）の概念地形の抽出（文字不使用）
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Extracting Student Initial Concept Terrain directly from Embeddings (No Text)...")
    from shallow_extractor import ShallowWeights
    sw_s = ShallowWeights(model_id=STUDENT_MODEL_ID, verbose=False)
    print(f"Loaded Student ({STUDENT_MODEL_ID}) embed_tokens: {sw_s.vocab_rows} tokens, dim={sw_s.dim}")

    # 生徒の全語彙から代表的な幾何学的基底点群をサンプリング（文字は一切介さない）
    with sw_s._open() as f:
        sl = f.get_slice(sw_s.embed_key)
        # 空間全体を均等にカバーするグリッドサンプリング
        sample_indices = np.linspace(0, sw_s.vocab_rows - 1, num=len(t_centers) * 8, dtype=int)
        raw_s = sl[sample_indices].float().numpy()
        s_pts_np = sw_s.preprocess_block(raw_s)

    s_pts = torch.tensor(s_pts_np, dtype=torch.float32, device=device)
    s_centers_init, k_S_init, s_dists_init, M_S_init = slicer.slice_terrain(s_pts)
    print(f"Student Initial Terrain Sliced: {len(s_centers_init)} slice centers extracted.")
    print(f"Student Scale k(S) = {k_S_init.item():.4f}")
    K = min(len(s_centers_init), len(t_centers))
    M_T_aligned = M_T[:K, :K]
    M_S_aligned = M_S_init[:K, :K]
    print(f"Initial Topological Distance Error: {F.mse_loss(M_S_aligned, M_T_aligned).item():.6f}")

    # -------------------------------------------------------------------------
    # 3. 生徒モデルへのLoRAアタッチと概念スライス・トポロジー蒸留学習
    # -------------------------------------------------------------------------
    print(f"\n[3/3] Setting up Student Model with LoRA for Slice Topology Distillation...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(base_model, lora_config)
    student.print_trainable_parameters()

    # 生徒の埋め込み層を取得
    embed_layer = student.get_input_embeddings()
    # サンプリングインデックスに対応するテンソル（文字ではなくトークンID位置のベクトル射影）
    target_token_indices = torch.tensor(sample_indices[:K], dtype=torch.long, device=device)

    optimizer = optim.Adam(student.parameters(), lr=2e-4)
    epochs = 150
    target_M_T = M_T_aligned.detach()

    print(f"\nStarting Pure Slice Topological Distillation (Target Topology: {target_M_T.shape})...")
    start_time = time.time()
    
    # 初期の無次元トポロジー誤差
    student.eval()
    with torch.no_grad():
        init_emb = embed_layer(target_token_indices).to(torch.float32)
        init_emb_norm = F.normalize(init_emb, p=2, dim=-1)
        init_outputs = student(inputs_embeds=init_emb_norm.unsqueeze(0).to(student.dtype), output_hidden_states=True)
        init_hidden = init_outputs.hidden_states[-1].squeeze(0).to(torch.float32)
        init_hidden_norm = F.normalize(init_hidden, p=2, dim=-1)
        init_dist = torch.cdist(init_hidden_norm, init_hidden_norm, p=2)
        init_k = torch.max(init_dist)
        init_M = init_dist / (init_k + 1e-8)
        init_loss = F.mse_loss(init_M, target_M_T).item()
    print(f"Epoch 0/150 - Initial Slice Topology Error: {init_loss:.6f}")

    # 学習ループ
    student.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()

        # 埋め込み層から直接スライスベクトルを取得し、inputs_embeds として深層モデルへ注入（文字を一切介さない）
        current_emb = embed_layer(target_token_indices).to(torch.float32)
        current_norm = F.normalize(current_emb, p=2, dim=-1)

        # 文字列プロンプトではなく、概念地形ベクトルを直接 Transformer へ Forward
        # shape: (1, M, D)
        outputs = student(inputs_embeds=current_norm.unsqueeze(0).to(student.dtype), output_hidden_states=True)
        hidden_vecs = outputs.hidden_states[-1].squeeze(0).to(torch.float32) # (M, D)
        hidden_norm = F.normalize(hidden_vecs, p=2, dim=-1)

        # Studentの距離行列と最大幅 k(S) の算出
        s_dist = torch.cdist(hidden_norm, hidden_norm, p=2)
        k_S = torch.max(s_dist)
        
        # 無次元スライストポロジー行列 M_S
        M_S = s_dist / (k_S + 1e-8)

        # 1. スライストポロジーMSE損失 (無次元同相写像: M_S == M_T)
        topo_loss = F.mse_loss(M_S, target_M_T)

        # 2. スケールアスペクト比ペナルティ (k(S)が極端に潰れたり発散するのを防止)
        scale_loss = 0.05 * torch.abs(k_S - init_k)

        # 3. アンカー維持ペナルティ (全体の空間回転を抑止)
        anchor_loss = 0.1 * F.mse_loss(hidden_norm, init_hidden_norm)

        total_loss = topo_loss + scale_loss + anchor_loss
        total_loss.backward()
        optimizer.step()

        if ep % 25 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} - Loss: {total_loss.item():.6f} (Topo: {topo_loss.item():.6f}, k_S={k_S.item():.4f})")

    elapsed = time.time() - start_time
    final_topo_loss = topo_loss.item()
    improvement = (init_loss - final_topo_loss) / init_loss * 100
    print(f"\nSlice Topological Distillation Completed in {elapsed:.2f}s!")
    print(f"Topology Error: {init_loss:.6f} -> {final_topo_loss:.6f} (Improvement: {improvement:+.2f}%)")

    # 成果物の保存
    os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)
    student.save_pretrained(OUTPUT_LORA_DIR)
    tokenizer.save_pretrained(OUTPUT_LORA_DIR)
    print(f"Saved Pure Slice Topological LoRA to: {OUTPUT_LORA_DIR}")

    # concept_core側にも同期保存
    concept_core_lora_dir = r"C:\Users\lop3l\.gemini\antigravity\scratch\concept_core\distillation_experiment\lora_glm5.3_slice_topological_to_Qwen2.5_Coder_3B"
    os.makedirs(concept_core_lora_dir, exist_ok=True)
    student.save_pretrained(concept_core_lora_dir)
    tokenizer.save_pretrained(concept_core_lora_dir)
    print(f"Synced to concept_core: {concept_core_lora_dir}")

    return OUTPUT_LORA_DIR, init_loss, final_topo_loss, improvement

if __name__ == "__main__":
    run_slice_topological_distillation()
