"""
Profile GLM Idea Cores
----------------------
53個のGLMイデア核それぞれに対して、GLM-5.3の全語彙から最もコサイン類似度が高い
最近傍トークン（Top-5）を特定し、各イデア核の意味論的プロファイルを可視化する。
"""

import os
import sys
import time
import torch
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'concept_core')))
from shallow_extractor import ShallowWeights
from transformers import AutoTokenizer

def profile_glm_idea_cores():
    model_id = "zai-org/GLM-5.3"
    print(f"[*] ShallowWeights (GLM-5.3) を準備中...")
    sw = ShallowWeights(model_id=model_id, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 概念核データロード
    terrain_path = "distillation_experiments/glm5.3_zero_vram_pure_concept_terrains.pt"
    data = torch.load(terrain_path, map_location="cpu", weights_only=False)
    A0 = data["Layer_0"][:53].float().numpy() # (53, 6144)
    A0 = A0 / np.linalg.norm(A0, axis=1, keepdims=True)

    print(f"[+] イデア核: {A0.shape}。全語彙 {sw.vocab_rows} との内積を計算中 (0 MB VRAM)...")
    
    # チャンクごとに内積計算
    chunk_size = 16384
    scores = np.zeros((53, sw.vocab_rows), dtype=np.float32)
    
    with sw._open() as f:
        sl = f.get_slice(sw.embed_key)
        for s in range(0, sw.vocab_rows, chunk_size):
            e = min(s + chunk_size, sw.vocab_rows)
            raw = sl[s:e].float().numpy()
            norm_b = sw.preprocess_block(raw)
            # (53, chunk) = (53, 6144) @ (6144, chunk)
            scores[:, s:e] = A0 @ norm_b.T

    print("\n" + "=" * 70)
    print("🌟 GLM-5.3 53個のイデア核 プロファイル (代表概念ノード)")
    print("=" * 70)

    idea_core_profiles = []
    for i in range(53):
        top_indices = np.argsort(-scores[i])[:6]
        top_tokens = []
        for tid in top_indices:
            decoded = tokenizer.decode([int(tid)]).strip()
            top_tokens.append(f"{decoded}({scores[i, tid]:.2f})")
        
        profile_str = ", ".join(top_tokens)
        print(f"Core #{i:02d}: {profile_str}")
        idea_core_profiles.append({
            "core_id": i,
            "top_tokens": [tokenizer.decode([int(tid)]).strip() for tid in top_indices],
            "scores": [float(scores[i, tid]) for tid in top_indices]
        })

    import json
    with open("distillation_experiments/glm53_idea_core_labels.json", "w", encoding="utf-8") as f:
        json.dump(idea_core_profiles, f, ensure_ascii=False, indent=2)
    print("\n[+] プロファイル保存完了: distillation_experiments/glm53_idea_core_labels.json")

if __name__ == "__main__":
    profile_glm_idea_cores()
