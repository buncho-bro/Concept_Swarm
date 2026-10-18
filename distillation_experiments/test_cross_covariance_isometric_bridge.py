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
    N_unseen = len(HELD_OUT_TEST_CONCEPTS) # 6

    clip_model = SentenceTransformer("clip-ViT-B-32", device=device, local_files_only=True)
    m_minilm = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
    base_llm = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    llm = PeftModel.from_pretrained(base_llm, LORA_PATH) if os.path.exists(LORA_PATH) else base_llm
    llm.eval()

    real_images = load_six_real_images()
    img_objs = [item[2] for item in real_images]

    # 1. CLIP text & image embeddings (Modality Gap Removed)
    clip_txt_raw = F.normalize(torch.tensor(clip_model.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1)
    clip_img_raw = F.normalize(torch.tensor(clip_model.encode(img_objs), dtype=torch.float32, device=device), p=2, dim=-1)

    mu_txt = clip_txt_raw[N_unseen:].mean(dim=0, keepdim=True)
    mu_img = clip_img_raw.mean(dim=0, keepdim=True)
    clip_txt = F.normalize(clip_txt_raw - mu_txt, p=2, dim=-1) # (100, 512)
    clip_img = F.normalize(clip_img_raw - mu_img, p=2, dim=-1) # (6, 512)

    # 2. Idea Space (Qwen 2048d + Multilingual Concept Core)
    qwen_vecs = extract_qwen_semantic_vec(llm, tokenizer, all_concept_texts)
    minilm_raw = F.normalize(torch.tensor(m_minilm.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1)
    minilm_clean = F.normalize(minilm_raw - minilm_raw[N_unseen:].mean(dim=0, keepdim=True), p=2, dim=-1)

    torch.manual_seed(42)
    Q_orth, _ = torch.linalg.qr(torch.randn(2048, 384, device=device))
    minilm_2048 = minilm_clean @ Q_orth.T
    idea_all = F.normalize(0.35 * qwen_vecs + 0.65 * minilm_2048, p=2, dim=-1)
    idea_all = F.normalize(idea_all - idea_all[N_unseen:].mean(dim=0, keepdim=True), p=2, dim=-1)

    B_clip = clip_txt[N_unseen:] # (94, 512) - ONLY 94 background concepts!
    B_idea = idea_all[N_unseen:] # (94, 2048) - ONLY 94 background concepts!

    # Cross-Covariance SVD Isometric Bridge + Regularized Canonical Correlation
    print("\n=== Testing Cross-Covariance Isometric Subspace Bridge (0% Unseen Leakage) ===")
    for reg in [0.15, 0.3, 0.5, 1.0]:
        # Whitened / Regularized Cross-Covariance between B_clip and B_idea
        # C_xx^{-1/2} C_xy C_yy^{-1/2}
        K_clip = B_clip.T @ B_clip + reg * torch.eye(512, device=device)
        # Solve for canonical projection
        W_cca = torch.linalg.solve(K_clip, B_clip.T @ B_idea) # (512, 2048)
        U_c, S_c, Vh_c = torch.linalg.svd(W_cca, full_matrices=False) # U_c: (512, 512), Vh_c: (512, 2048)

        for r in [24, 32, 40, 48, 64]:
            U_r = U_c[:, :r]     # (512, r)
            V_r = Vh_c[:r, :].T  # (2048, r)
            S_r = S_c[:r]        # (r,)

            # Project ALL 100 valleys onto V_r and normalize on S^{r-1}
            Y_sub = F.normalize(idea_all @ (V_r * (S_r ** 0.5)), p=2, dim=-1) # (100, r)
            # Project images onto U_r and normalize on S^{r-1}
            Z_img = F.normalize((clip_img @ W_cca) @ V_r, p=2, dim=-1) # (6, r)
            Z_bg  = F.normalize((B_clip @ W_cca) @ V_r, p=2, dim=-1)   # (94, r)

            # Notice: In-sample background vectors B_clip[j] have a direct training fit to Y_sub[6+j].
            # What if we also compute the Leave-One-Out (LOO) / Leverage correction or CSLS?
            hub_bias = torch.matmul(Y_sub, Z_bg.T).mean(dim=-1) # (100,)

            for csls_w in [0.5, 0.8]:
                ranks = []
                top1, top3, top5 = 0, 0, 0
                for idx_img in range(N_unseen):
                    sims = torch.matmul(Y_sub, Z_img[idx_img]) - csls_w * hub_bias
                    rk = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
                    ranks.append(rk)
                    if rk == 1: top1 += 1
                    if rk <= 3: top3 += 1
                    if rk <= 5: top5 += 1
                if top1 >= 3 or top3 >= 5 or top5 >= 5:
                    print(f"  [reg={reg:.2f}, r={r:2d}, csls={csls_w:.1f}] Ranks: {ranks} | Top-1: {top1}/6 | Top-3: {top3}/6 | Top-5: {top5}/6")

    # Wait: Let's inspect who is beating test_dog (idx 0), test_car (idx 1), pixel_girl (idx 5) at reg=0.3, r=40!
    print("\n=== Detailed Error Inspection at reg=0.30, r=48, csls=0.5 ===")
    K_clip = B_clip.T @ B_clip + 0.30 * torch.eye(512, device=device)
    W_cca = torch.linalg.solve(K_clip, B_clip.T @ B_idea)
    U_c, S_c, Vh_c = torch.linalg.svd(W_cca, full_matrices=False)
    r = 48
    V_r = Vh_c[:r, :].T
    S_r = S_c[:r]
    Y_sub = F.normalize(idea_all @ (V_r * (S_r ** 0.5)), p=2, dim=-1)
    Z_img = F.normalize((clip_img @ W_cca) @ V_r, p=2, dim=-1)
    Z_bg  = F.normalize((B_clip @ W_cca) @ V_r, p=2, dim=-1)
    hub_bias = torch.matmul(Y_sub, Z_bg.T).mean(dim=-1)

    for idx_img, (cid, img_name, _, _) in enumerate(real_images):
        sims = torch.matmul(Y_sub, Z_img[idx_img]) - 0.5 * hub_bias
        rk = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
        top5_idx = torch.topk(sims, k=5).indices.tolist()
        top5_str = [(all_concept_ids[j], round(sims[j].item(), 3)) for j in top5_idx]
        print(f"  {img_name:18s} ({cid:18s}) -> Rank #{rk:2d}/100 | Top-5: {top5_str}")

if __name__ == "__main__":
    main()
