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
    bg_ids = all_concept_ids[N_unseen:]

    clip_model = SentenceTransformer("clip-ViT-B-32", device=device, local_files_only=True)

    real_images = load_six_real_images()
    img_objs = [item[2] for item in real_images]

    clip_txt_raw = F.normalize(torch.tensor(clip_model.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1) # (100, 512)
    clip_img_raw = F.normalize(torch.tensor(clip_model.encode(img_objs), dtype=torch.float32, device=device), p=2, dim=-1)          # (6, 512)

    print("=== 1. Native Raw CLIP (clip_txt_raw[6:] @ clip_img_raw) Nearest Background Landmarks ===")
    B_raw = clip_txt_raw[N_unseen:] # (94, 512)
    for i, (cid, img_name, _, _) in enumerate(real_images):
        sims = B_raw @ clip_img_raw[i]
        top5 = [bg_ids[j] for j in torch.topk(sims, k=5).indices.tolist()]
        print(f"  {img_name:18s} ({cid:18s}) -> Raw Top-5 BG: {top5}")

    print("\n=== 2. Native Raw CLIP with Text-Hubness Removal (CSLS on Raw CLIP) ===")
    # Each text concept has an intrinsic hubness in CLIP space (some text prompts have higher average cosine with everything)
    txt_hubness = (B_raw @ B_raw.T).mean(dim=-1) # (94,)
    for i, (cid, img_name, _, _) in enumerate(real_images):
        sims = (B_raw @ clip_img_raw[i]) - 0.5 * txt_hubness
        top5 = [bg_ids[j] for j in torch.topk(sims, k=5).indices.tolist()]
        print(f"  {img_name:18s} ({cid:18s}) -> CSLS Top-5 BG: {top5}")

    # Wait: Why did Native CLIP get 6/6 = 100% Top-1 when comparing clip_img_raw to clip_txt_raw (all 100)?
    # Because in Native CLIP, clip_img_raw[i] is ALREADY in the same 512-dim space as clip_txt_raw!
    # The ONLY reason clip_img_raw failed when passed through W_bridge (trained on B_clip -> B_idea)
    # is the Modality Gap vector delta = E[clip_img] - E[clip_txt] which gets multiplied by W_bridge!
    # Wait: What happens if W_bridge is forced to be ORTHOGONAL to the constant/mean text direction mu_txt
    # AND we do NOT subtract the 6-image batch mean, OR we project out the first principal component of the modality shift?!
    print("\n=== 3. Testing Single-Image-Compatible Projection (Orthogonal to mu_txt, NO batch mean!) ===")
    mu_txt = F.normalize(B_raw.mean(dim=0, keepdim=True), p=2, dim=-1) # (1, 512) unit vector of text cone axis!
    # Project both text and any single image orthogonal to mu_txt (Householder / Gram-Schmidt projection I - mu_txt^T mu_txt)
    B_proj = F.normalize(B_raw - (B_raw @ mu_txt.T) @ mu_txt, p=2, dim=-1)
    img_proj = F.normalize(clip_img_raw - (clip_img_raw @ mu_txt.T) @ mu_txt, p=2, dim=-1)
    for i, (cid, img_name, _, _) in enumerate(real_images):
        sims = B_proj @ img_proj[i]
        top5 = [bg_ids[j] for j in torch.topk(sims, k=5).indices.tolist()]
        print(f"  {img_name:18s} ({cid:18s}) -> Ortho-to-mu_txt Top-5 BG: {top5}")

if __name__ == "__main__":
    main()
