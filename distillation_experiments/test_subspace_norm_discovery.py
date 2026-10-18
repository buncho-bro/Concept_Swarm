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

    # Check subspace norm penalty!
    U_b, S_b, Vh_b = torch.linalg.svd(B_idea, full_matrices=False) # Vh_b: (94, 2048)
    proj_norms = torch.norm(idea_all @ Vh_b.T, dim=-1)
    print(f"\n[Mathematical Discovery] Subspace Norm inside span(B_idea) (94D subspace of 2048D):")
    print(f"  Unseen 6 concepts (0..5) norm in span(B_idea): {proj_norms[:N_unseen].mean().item():.4f} (each: {[round(x,3) for x in proj_norms[:N_unseen].tolist()]})")
    print(f"  Seen  94 concepts (6..99) norm in span(B_idea): {proj_norms[N_unseen:].mean().item():.4f}")

    # Notice: If we normalize EVERY candidate valley (all 100) AFTER projecting onto the bridge's observable subspace
    # (or regularized principal subspace of B_idea), unseen concepts no longer suffer the 0.68x norm penalty!
    print("\n--- Method A: Subspace-Normalized Ridge + Canonical Procrustes Bridge ---")
    for r in [24, 32, 48, 64, 80, 94]:
        V_sub = Vh_b[:r].T # (2048, r) observable semantic subspace of B_idea
        # Project all 100 candidate valleys into the observable r-dim semantic subspace and normalize to unit sphere S^{r-1}
        Y_all_sub = F.normalize(idea_all @ V_sub, p=2, dim=-1) # (100, r) — ALL 100 valleys now have norm 1.0!
        B_idea_sub = Y_all_sub[N_unseen:] # (94, r)

        for lam in [0.1, 0.25, 0.5]:
            W_ridge = torch.linalg.solve(B_clip.T @ B_clip + lam * torch.eye(512, device=device), B_clip.T @ B_idea_sub) # (512, r)
            Z_img = F.normalize(clip_img @ W_ridge, p=2, dim=-1) # (6, r)
            Z_bg = F.normalize(B_clip @ W_ridge, p=2, dim=-1)    # (94, r)
            hub_bias = torch.matmul(Y_all_sub, Z_bg.T).mean(dim=-1) # (100,)

            # Also compute Leave-One-Out self-fitting bias for the 94 training landmarks!
            # Wait: for the 94 training landmarks, W_ridge was fit DIRECTLY on them, so <B_idea_sub[j], B_clip[j] @ W_ridge>
            # has an in-sample training boost compared to held-out points!
            # Let's measure the exact in-sample vs LOO boost:
            H_mat = B_clip @ torch.linalg.solve(B_clip.T @ B_clip + lam * torch.eye(512, device=device), B_clip.T) # (94, 94) hat matrix!
            diag_h = torch.diag(H_mat) # (94,) leverage of each training landmark!

            ranks = []
            top1, top3, top5 = 0, 0, 0
            for idx_img in range(N_unseen):
                sims = torch.matmul(Y_all_sub, Z_img[idx_img]) - 0.5 * hub_bias
                rk = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rk)
                if rk == 1: top1 += 1
                if rk <= 3: top3 += 1
                if rk <= 5: top5 += 1
            print(f"  [r={r:2d}, lam={lam:.2f}] Ranks: {ranks} | Top-1: {top1}/6 | Top-3: {top3}/6 | Top-5: {top5}/6")

if __name__ == "__main__":
    main()
