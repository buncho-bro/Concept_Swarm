"""
diagnose_topological_triangulation.py
=============================================================================
なぜ Cartesian 線形写像 (x @ W) が未見画像で崩壊したのかの数学的解明と、
「無次元トポロジー三角測量 (Geodesic Triangulation: M(x) = D(x, B) / k)」
による完全ゼロショット（100谷・未見6画像）の汎用性検証
=============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from test_zero_shot_generality_stress import (
    HELD_OUT_TEST_CONCEPTS,
    BACKGROUND_CALIBRATION_CONCEPTS,
    load_six_real_images,
    STUDENT_MODEL_ID,
    LORA_PATH
)

def extract_qwen_semantic_vec(model, tokenizer, texts: list[str], batch_size: int = 32) -> torch.Tensor:
    """
    Qwen2.5-Coder-3B に意味概念プロンプトを与え、最終トークンの深層隠れ状態 (L18 + L24 + L30)
    から共通バイアスを除去して純粋な概念地形ベクトルを抽出する。
    """
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompted = [f"The core semantic concept and category of '{t}' is:" for t in texts]
    vecs = []
    for i in range(0, len(prompted), batch_size):
        batch = prompted[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=48).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            # Left padding なので最後のトークン [-1] が常に ':' の直後の概念凝縮位置！
            l_list = []
            for l_idx in [12, 18, 24, 30]:
                h_last = hs[l_idx][:, -1, :].float() # (B, 2048)
                l_list.append(F.normalize(h_last, p=2, dim=-1))
            vecs.append(torch.stack(l_list, dim=1)) # (B, 4, 2048)

    stacked = torch.cat(vecs, dim=0) # (N, 4, 2048)
    mean_l = stacked.mean(dim=0, keepdim=True)
    centered = F.normalize(stacked - mean_l, p=2, dim=-1)
    fused = centered.mean(dim=1)
    return F.normalize(fused, p=2, dim=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_concept_ids = [c[0] for c in HELD_OUT_TEST_CONCEPTS] + [c[0] for c in BACKGROUND_CALIBRATION_CONCEPTS]
    all_concept_texts = [c[1] for c in HELD_OUT_TEST_CONCEPTS] + [c[1] for c in BACKGROUND_CALIBRATION_CONCEPTS]
    N_unseen = len(HELD_OUT_TEST_CONCEPTS) # 6

    print("Loading CLIP & Qwen...")
    clip_model = SentenceTransformer("clip-ViT-B-32", device=device, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
    base_llm = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    llm = PeftModel.from_pretrained(base_llm, LORA_PATH) if os.path.exists(LORA_PATH) else base_llm
    llm.eval()

    real_images = load_six_real_images()
    img_objs = [item[2] for item in real_images]

    # 1. CLIP 空間の抽出
    clip_txt_all = F.normalize(torch.tensor(clip_model.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1)
    clip_img_raw = F.normalize(torch.tensor(clip_model.encode(img_objs), dtype=torch.float32, device=device), p=2, dim=-1)

    # まず診断1: CLIP自身のネイティブ精度（ブリッジを通す前、CLIPがそもそも6枚の画像を100概念中何位に認識しているか）
    print("\n[Diagnosis 1] Native CLIP (Image 512d vs CLIP Text 512d across 100 concepts):")
    for idx_img, (cid, img_name, _, _) in enumerate(real_images):
        sims = torch.matmul(clip_txt_all, clip_img_raw[idx_img])
        rank = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
        top3 = [all_concept_ids[i] for i in torch.topk(sims, k=3).indices.tolist()]
        print(f"  {img_name:18s} -> Native CLIP Rank: #{rank:2d}/100 | Top-3: {top3}")

    # 2. Qwen の概念空間抽出（意味凝縮プロンプトの最終トークン L12+L18+L24+L30）
    print("\n[Diagnosis 2] Extracting Qwen Semantic Last-Token Cascade (L12+L18+L24+L30)...")
    qwen_all = extract_qwen_semantic_vec(llm, tokenizer, all_concept_texts) # (100, 2048)

    # Qwen内部で未見6概念(0..5)の最近傍背景概念(6..99)が何になっているか確認！
    print("  Checking Qwen's internal semantic neighbors for the 6 Held-Out concepts:")
    qwen_bg = qwen_all[N_unseen:] # (94, 2048)
    bg_ids = all_concept_ids[N_unseen:]
    for i in range(N_unseen):
        s_bg = torch.matmul(qwen_bg, qwen_all[i])
        top3_bg = [bg_ids[j] for j in torch.topk(s_bg, k=3).indices.tolist()]
        print(f"    Qwen '{all_concept_ids[i]:18s}' nearest background landmarks: {top3_bg}")

    # 3. 【核心手法】無次元トポロジー三角測量 (Geodesic Triangulation Bridge)
    # 94個の背景ランドマーク B (Index 6..99) のみを使用！テストクラス (0..5) は一切不使用！
    print("\n[Diagnosis 3] Zero-Shot Topological Triangulation (Using ONLY 94 Background Landmarks):")
    B_clip = clip_txt_all[N_unseen:] # (94, 512) - テスト概念は完全除外！
    B_qwen = qwen_all[N_unseen:]     # (94, 2048) - テスト概念は完全除外！

    # 画像 x の 94個の背景ランドマークに対する相対トポロジー座標 M_vis(x) (94次元)
    # Modality Gap (画像とテキストの定数オフセット) を各点の平均・標準偏差(Z-score)および温度付きソフトマックスで無次元化！
    img_to_B = torch.matmul(clip_img_raw, B_clip.T) # (6, 94)
    img_topo_coord = F.normalize((img_to_B - img_to_B.mean(dim=-1, keepdim=True)) / (img_to_B.std(dim=-1, keepdim=True) + 1e-6), p=2, dim=-1)

    # 100個の候補概念 (0..99) の 94個の背景ランドマークに対する相対トポロジー座標
    # CLIP側トポロジーとQwen側トポロジーの無次元一致
    qwen_to_B = torch.matmul(qwen_all, B_qwen.T) # (100, 94)
    # 自分自身(対角成分)が1.0になって有利にならないよう、背景概念自身の対角成分はTop-2値でマスク補間する（完全公平化！）
    for j in range(94):
        row = qwen_to_B[N_unseen + j].clone()
        row[j] = -1e9
        qwen_to_B[N_unseen + j, j] = row.max()
    qwen_topo_coord = F.normalize((qwen_to_B - qwen_to_B.mean(dim=-1, keepdim=True)) / (qwen_to_B.std(dim=-1, keepdim=True) + 1e-6), p=2, dim=-1)

    # さらに 94ランドマーク基底を用いた Full-Rank 直交 + リッジブリッジ (Modality Gap 補正付き)
    # CLIPのテキスト空間と画像空間のModality Gapを背景平均で正規化
    X_bg = F.normalize(B_clip - B_clip.mean(dim=0, keepdim=True), p=2, dim=-1) # (94, 512)
    Y_bg = F.normalize(B_qwen - B_qwen.mean(dim=0, keepdim=True), p=2, dim=-1) # (94, 2048)
    W_bridge = torch.linalg.solve(X_bg.T @ X_bg + 0.1 * torch.eye(512, device=device), X_bg.T @ Y_bg) # (512, 2048)

    img_centered = F.normalize(clip_img_raw - B_clip.mean(dim=0, keepdim=True), p=2, dim=-1)
    qwen_all_centered = F.normalize(qwen_all - B_qwen.mean(dim=0, keepdim=True), p=2, dim=-1)

    top1_cnt, top3_cnt, top5_cnt = 0, 0, 0
    for idx_img, (cid, img_name, _, _) in enumerate(real_images):
        # 1) ブリッジ直接内積
        z_bridged = F.normalize(img_centered[idx_img] @ W_bridge, p=2, dim=-1)
        sim_direct = torch.matmul(qwen_all_centered, z_bridged) # (100,)
        # 2) 無次元トポロジー三角測量内積
        sim_topo = torch.matmul(qwen_topo_coord, img_topo_coord[idx_img]) # (100,)
        
        sim_combined = 0.5 * sim_direct + 0.5 * sim_topo
        rank = (torch.argsort(sim_combined, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
        top5_ids = torch.topk(sim_combined, k=5).indices.tolist()
        if rank == 1: top1_cnt += 1
        if rank <= 3: top3_cnt += 1
        if rank <= 5: top5_cnt += 1
        top3_names = [all_concept_ids[i] for i in top5_ids[:3]]
        print(f"  {img_name:18s} (True: {cid:18s}) -> Zero-Shot Rank: #{rank:2d}/100 | Top-3: {top3_names}")

    print(f"\n  => Zero-Shot Unseen Accuracy (100 Valleys): Top-1 = {top1_cnt}/6 ({top1_cnt/6*100:.1f}%), Top-3 = {top3_cnt}/6 ({top3_cnt/6*100:.1f}%), Top-5 = {top5_cnt}/6 ({top5_cnt/6*100:.1f}%)")

if __name__ == "__main__":
    main()
