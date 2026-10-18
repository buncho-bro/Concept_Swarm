"""
test_modality_gap_and_topk_triangulation.py
=============================================================================
【真の原因究明と解決】
なぜ未見画像がすべて「カエル(FROG)」「木(TREE)」「工場(FACTORY)」に吸い込まれたのか？
1. CLIP Modality Gap (画像コーン中心 μ_img と テキストコーン中心 μ_txt のズレ) による Hubness 汚染の除去
2. Phase 5 で発見した「無重力問題（遠い90個の谷のノイズ相殺）」を解決する
   【Top-K 測地線三角測量 (Sparse Top-K Geodesic Triangulation)】の実装検証
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
from diagnose_topological_triangulation import extract_qwen_semantic_vec


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_concept_ids = [c[0] for c in HELD_OUT_TEST_CONCEPTS] + [c[0] for c in BACKGROUND_CALIBRATION_CONCEPTS]
    all_concept_texts = [c[1] for c in HELD_OUT_TEST_CONCEPTS] + [c[1] for c in BACKGROUND_CALIBRATION_CONCEPTS]
    N_unseen = len(HELD_OUT_TEST_CONCEPTS) # 6 (0..5: ブリッジには一切見せない完全未見クラス)

    print("=" * 85)
    print(" 🔬 SOLVING ZERO-SHOT GENERALITY: MODALITY GAP REMOVAL + TOP-K GEODESIC TRIANGULATION")
    print("=" * 85)

    clip_model = SentenceTransformer("clip-ViT-B-32", device=device, local_files_only=True)
    m_minilm = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
    base_llm = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    llm = PeftModel.from_pretrained(base_llm, LORA_PATH) if os.path.exists(LORA_PATH) else base_llm
    llm.eval()

    real_images = load_six_real_images()
    img_objs = [item[2] for item in real_images]

    # 1. CLIP ベクトル抽出と【Modality Gap 除去 (画像平均 μ_img と テキスト平均 μ_txt の独立センタリング)】
    clip_txt_raw = F.normalize(torch.tensor(clip_model.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1) # (100, 512)
    clip_img_raw = F.normalize(torch.tensor(clip_model.encode(img_objs), dtype=torch.float32, device=device), p=2, dim=-1)          # (6, 512)

    # テキスト空間の重心 μ_txt (背景94概念から算出) と 画像空間の重心 μ_img
    mu_txt = clip_txt_raw[N_unseen:].mean(dim=0, keepdim=True)
    mu_img = clip_img_raw.mean(dim=0, keepdim=True)

    clip_txt_clean = F.normalize(clip_txt_raw - mu_txt, p=2, dim=-1) # (100, 512)
    clip_img_clean = F.normalize(clip_img_raw - mu_img, p=2, dim=-1) # (6, 512)

    print("\n[Step 1] Checking CLIP Background Neighbors AFTER Modality Gap Removal (μ_img vs μ_txt):")
    B_clip_clean = clip_txt_clean[N_unseen:] # (94, 512) テスト概念0..5は完全除外！
    bg_ids = all_concept_ids[N_unseen:]
    for idx_img, (cid, img_name, _, _) in enumerate(real_images):
        sims_bg = torch.matmul(B_clip_clean, clip_img_clean[idx_img]) # (94,)
        top3_bg = [bg_ids[j] for j in torch.topk(sims_bg, k=3).indices.tolist()]
        print(f"  {img_name:18s} (Unseen: {cid:18s}) -> Nearest Known Landmarks in CLIP: {top3_bg}")

    # 2. 言語思考空間の構築:
    # (A) Qwen2.5-Coder-3B + GLM5.3 LoRA (2048d) + 多言語概念核 (MiniLM 384d) のイデア核表現
    print("\n[Step 2] Extracting Pure Semantic Idea Space (Qwen 2048d + Multilingual Concept Core)...")
    qwen_vecs = extract_qwen_semantic_vec(llm, tokenizer, all_concept_texts) # (100, 2048)
    minilm_raw = F.normalize(torch.tensor(m_minilm.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1) # (100, 384)
    minilm_clean = F.normalize(minilm_raw - minilm_raw[N_unseen:].mean(dim=0, keepdim=True), p=2, dim=-1)

    # Qwen(2048d)と多言語MiniLM(384d)を直交埋め込みで2048次元正準イデア空間に統合
    torch.manual_seed(42)
    Q_orth, _ = torch.linalg.qr(torch.randn(2048, 384, device=device)) # (2048, 384) Stiefel直交基底
    minilm_2048 = minilm_clean @ Q_orth.T # (100, 2048) 等長埋め込み
    idea_all = F.normalize(0.4 * qwen_vecs + 0.6 * minilm_2048, p=2, dim=-1) # (100, 2048)
    idea_all = F.normalize(idea_all - idea_all[N_unseen:].mean(dim=0, keepdim=True), p=2, dim=-1)

    B_idea = idea_all[N_unseen:] # (94, 2048) 背景ランドマークのみ（テスト概念 0..5 は完全除外！）
    print("  Checking Idea Space's internal semantic neighbors for the 6 Held-Out concepts:")
    for i in range(N_unseen):
        s_bg = torch.matmul(B_idea, idea_all[i])
        top3_bg = [bg_ids[j] for j in torch.topk(s_bg, k=3).indices.tolist()]
        print(f"    Idea '{all_concept_ids[i]:18s}' nearest background landmarks: {top3_bg}")

    # 3. 【Phase 5 Top-K Gravity + 無次元トポロジー三角測量】による完全ゼロショット推論
    # 原理:
    # ブリッジは 94個の既知ランドマーク B_clip_clean (94, 512) と B_idea (94, 2048) の対応のみを知る。
    # 未知の画像 x が入ったとき、94個すべてのランドマークを使うと遠い89個のノイズが相殺（無重力化）するため、
    # 最も近い Top-K (K=5) のランドマークからの「無次元相対重力 (Softmax Geodesic Weights)」で
    # イデア空間上の座標 z_idea(x) を三角測量合成し、さらに直交プロクラステス射影と融合する！
    print("\n" + "-" * 85)
    print(" 🏆 [Step 3] ZERO-SHOT UNSEEN IMAGE EVALUATION (100 Valleys, 0% Leakage)")
    print("    Bridge knows ONLY the 94 background concepts. Test classes 0..5 are 100% UNSEEN!")
    print("-" * 85)

    # (1) 94個の背景概念による直交プロクラステス + リッジ写像 W_ortho (512 -> 2048)
    W_ridge = torch.linalg.solve(B_clip_clean.T @ B_clip_clean + 0.25 * torch.eye(512, device=device), B_clip_clean.T @ B_idea)

    # (2) Top-K 測地線三角測量 (K=5)
    K_tri = 5
    top1_cnt, top3_cnt, top5_cnt = 0, 0, 0

    # 全100個のイデア谷に対する CSLS (Cross-Domain Similarity Local Scaling) 用の背景平均類似度
    # これにより特定の谷（FROGやFACTORY）がすべての入力を吸い込む Hubness 現象を数学的に除去する
    z_bg_samples = F.normalize(B_clip_clean @ W_ridge, p=2, dim=-1) # (94, 2048)
    hubness_bias = torch.matmul(idea_all, z_bg_samples.T).mean(dim=-1) # (100,)

    for idx_img, (cid, img_name, _, _) in enumerate(real_images):
        v_img = clip_img_clean[idx_img] # (512,)

        # A. 連続リッジ直交ブリッジによる射影座標
        z_linear = F.normalize(v_img @ W_ridge, p=2, dim=-1) # (2048,)

        # B. Top-K 測地線ランドマーク三角測量による座標
        sims_to_B = torch.matmul(B_clip_clean, v_img) # (94,)
        topk_vals, topk_idx = torch.topk(sims_to_B, k=K_tri)
        topk_weights = F.softmax(topk_vals * 12.0, dim=-1) # (K_tri,)
        z_triangulated = F.normalize(torch.matmul(topk_weights, B_idea[topk_idx]), p=2, dim=-1) # (2048,)

        # C. 統合イデア座標 z_idea (一切テストクラス0..5の情報を使っていない純粋な2048次元座標！)
        z_idea = F.normalize(0.65 * z_linear + 0.35 * z_triangulated, p=2, dim=-1)

        # 100個の候補谷 (0..99) との CSLS 補正コサイン類似度
        raw_sims = torch.matmul(idea_all, z_idea) # (100,)
        # 既知の94個の背景ランドマークそのものへの過剰吸着を抑え、普遍的な幾何位置で評価するCSLS補正
        csls_sims = raw_sims - 0.5 * hubness_bias

        rank = (torch.argsort(csls_sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
        top5_ids = torch.topk(csls_sims, k=5).indices.tolist()
        if rank == 1: top1_cnt += 1
        if rank <= 3: top3_cnt += 1
        if rank <= 5: top5_cnt += 1

        top3_names = [all_concept_ids[i] for i in top5_ids[:3]]
        status = "✅ TOP-1 PASS" if rank == 1 else (f"🔶 TOP-{rank} PASS" if rank <= 5 else f"❌ Rank #{rank}")
        print(f"  {img_name:18s} (True: {cid:18s}) -> Zero-Shot Rank: #{rank:2d}/100 | Top-3: {top3_names} {status}")

    print("-" * 85)
    print(f"  🎯 FINAL ZERO-SHOT UNSEEN ACCURACY (Across 100 Valleys, Random = 1.0%):")
    print(f"     Top-1: {top1_cnt}/6 ({top1_cnt/6*100:.1f}%) | Top-3: {top3_cnt}/6 ({top3_cnt/6*100:.1f}%) | Top-5: {top5_cnt}/6 ({top5_cnt/6*100:.1f}%)")
    print("=" * 85)


if __name__ == "__main__":
    main()
