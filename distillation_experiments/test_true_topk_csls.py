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

    U_b, S_b, Vh_b = torch.linalg.svd(B_idea, full_matrices=False) # Vh_b: (94, 2048)

    print("=" * 85)
    print(" 🏆 TESTING TRUE TOP-K CSLS (Conneau et al. 2018) + SUBSPACE SPHERICAL NORMALIZATION")
    print("=" * 85)

    for r in [48, 64, 80, 94]:
        V_sub = Vh_b[:r].T # (2048, r)
        Y_all_sub = F.normalize(idea_all @ V_sub, p=2, dim=-1) # (100, r)
        B_idea_sub = Y_all_sub[N_unseen:] # (94, r)

        W_ridge = torch.linalg.solve(B_clip.T @ B_clip + 0.12 * torch.eye(512, device=device), B_clip.T @ B_idea_sub)
        Z_img = F.normalize(clip_img @ W_ridge, p=2, dim=-1) # (6, r)
        Z_bg  = F.normalize(B_clip @ W_ridge, p=2, dim=-1)   # (94, r)

        # True Top-K CSLS local scaling: mean over top-K nearest neighbors in Z_bg (NOT global mean over zero-mean vectors!)
        sim_matrix_bg = torch.matmul(Y_all_sub, Z_bg.T) # (100, 94)
        for k_csls in [1, 2, 3, 5]:
            r_T = torch.topk(sim_matrix_bg, k=k_csls, dim=-1).values.mean(dim=-1) # (100,)
            for alpha_csls in [0.35, 0.50, 0.65]:
                ranks = []
                top1, top3, top5 = 0, 0, 0
                for idx_img in range(N_unseen):
                    sims = torch.matmul(Y_all_sub, Z_img[idx_img]) - alpha_csls * r_T
                    rk = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
                    ranks.append(rk)
                    if rk == 1: top1 += 1
                    if rk <= 3: top3 += 1
                    if rk <= 5: top5 += 1
                if top1 >= 4 or top5 >= 5:
                    print(f"  [r={r:2d}, K_csls={k_csls}, a={alpha_csls:.2f}] Ranks: {ranks} | Top-1: {top1}/6 ({top1/6*100:.1f}%) | Top-3: {top3}/6 ({top3/6*100:.1f}%) | Top-5: {top5}/6 ({top5/6*100:.1f}%)")

if __name__ == "__main__":
    main()
