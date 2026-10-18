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

    # =========================================================================
    # MATHEMATICAL DISCOVERY: Leave-One-Out (LOO) Manifold Reconstruction
    # Why did background landmarks (6..99) beat unseen concepts (0..5) by 0.01~0.02?
    # Because for a seen landmark j in {6..99}, y_j is literally one of the basis vectors of B_idea!
    # So y_j has a 1.0 self-identity spike with itself (including idiosyncratic word noise),
    # whereas an unseen concept i in {0..5} is NOT in B_idea, so its projection onto span(B_idea)
    # only uses off-diagonal semantic correlations with the 94 landmarks!
    #
    # By the Leave-One-Out Manifold Principle:
    # Every candidate valley c in {0..99} should be represented by its smooth manifold projection
    # onto the OTHER landmarks (for c in 0..5, it uses all 94 landmarks; for c = 6+j, we use
    # Ridge LOO projection onto the OTHER 93 landmarks so j cannot cheat via its own self-noise spike!).
    # =========================================================================
    print("=" * 85)
    print(" 🔬 TESTING LEAVE-ONE-OUT (LOO) SELF-SPIKE REMOVAL ON THE 100 CONCEPT VALLEYS")
    print("=" * 85)

    for lam_manifold in [0.1, 0.25, 0.5, 1.0]:
        # Gram matrix of the 94 landmarks in Idea Space
        G_bb = B_idea @ B_idea.T # (94, 94)
        # Kernel ridge smoother K_inv = (G_bb + lam * I)^{-1}
        K_inv = torch.linalg.inv(G_bb + lam_manifold * torch.eye(94, device=device))
        # Hat matrix H = G_bb @ K_inv
        H = G_bb @ K_inv # (94, 94)

        # For unseen concepts (0..5): projection onto the 94 landmarks
        G_unseen_b = idea_all[:N_unseen] @ B_idea.T # (6, 94)
        Y_unseen_manifold = F.normalize((G_unseen_b @ K_inv) @ B_idea, p=2, dim=-1) # (6, 2048)

        # For seen background concepts (6..99): Leave-One-Out (LOO) manifold projection!
        # Formula: y_j^{(-j)} = (H y - diag(H) y) / (1 - diag(H))
        diag_H = torch.diag(H).unsqueeze(-1) # (94, 1)
        Y_seen_in_sample = H @ B_idea # (94, 2048)
        Y_seen_loo = F.normalize((Y_seen_in_sample - diag_H * B_idea) / (1.0 - diag_H), p=2, dim=-1) # (94, 2048)

        # Combined 100 candidate valleys on 100% equal Out-of-Sample footing!
        Y_all_fair = torch.cat([Y_unseen_manifold, Y_seen_loo], dim=0) # (100, 2048)

        # Now map CLIP -> B_idea using regularized bridge trained ONLY on the 94 background concepts
        for reg_clip in [0.15, 0.30, 0.50]:
            W_bridge = torch.linalg.solve(B_clip.T @ B_clip + reg_clip * torch.eye(512, device=device), B_clip.T @ B_idea) # (512, 2048)
            Z_img = F.normalize(clip_img @ W_bridge, p=2, dim=-1) # (6, 2048)
            Z_bg  = F.normalize(B_clip @ W_bridge, p=2, dim=-1)   # (94, 2048)
            hub_bias = torch.matmul(Y_all_fair, Z_bg.T).mean(dim=-1) # (100,)

            ranks = []
            top1, top3, top5 = 0, 0, 0
            for idx_img in range(N_unseen):
                sims = torch.matmul(Y_all_fair, Z_img[idx_img]) - 0.5 * hub_bias
                rk = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rk)
                if rk == 1: top1 += 1
                if rk <= 3: top3 += 1
                if rk <= 5: top5 += 1
            print(f"  [lam_m={lam_manifold:.2f}, reg_c={reg_clip:.2f}] Ranks: {ranks} | Top-1: {top1}/6 ({top1/6*100:.1f}%) | Top-3: {top3}/6 ({top3/6*100:.1f}%) | Top-5: {top5}/6 ({top5/6*100:.1f}%)")

if __name__ == "__main__":
    main()
