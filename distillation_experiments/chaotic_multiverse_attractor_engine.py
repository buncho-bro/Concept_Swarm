"""
Chaotic Multiverse Attractor Engine (CF-ME)
===========================================
人間の恣意的なルールやフォーマットを完全排除し、
「潜在空間のカオス的揺らぎ」と「最小エントロピー・アトラクターの物理的選択」
によって、自律的に真理・整合性へと収束する普遍推論エンジン。
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[{device.upper()}] モデルを初期化しています...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 素の1.5Bベースモデル（余計なパッチを排した基盤）
base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

# 7Bゲージ固定型蒸留（幾何構造の補正）
peft_model = PeftModel.from_pretrained(
    base_model,
    os.path.join(OUTPUT_DIR, "lora_gauge_locked_7b_to_1.5b"),
    adapter_name="gauge_7b"
)
peft_model.eval()


def generate_chaotic_perturbations(num_worlds, hidden_dim, scale=0.015, device="cuda"):
    """
    決定論的カオス（ロジスティック写像 x_{n+1} = 4 x_n (1 - x_n)）を用いて、
    高次元空間全体に相関のない微小なカオスの揺らぎを生成する。
    """
    perturbations = torch.zeros((num_worlds, hidden_dim), device=device, dtype=torch.bfloat16)
    for k in range(num_worlds):
        # 異なる初期値からカオス時系列を励起
        x = 0.12345 + 0.08765 * (k + 1)
        chaos_seq = []
        for _ in range(hidden_dim):
            x = 3.999 * x * (1.0 - x)
            chaos_seq.append(x - 0.5) # 中心化
        vec = torch.tensor(chaos_seq, device=device, dtype=torch.bfloat16)
        # 正規化してスケールを適用
        vec = vec / (torch.norm(vec) + 1e-8)
        perturbations[k] = vec * scale
    return perturbations


def chaotic_multiverse_generate(
    prompt_text,
    max_new_tokens=250,
    num_worlds=6,            # 並行世界の数
    chaos_scale=0.015,       # カオスの揺らぎの強さ
    temperature=0.4,
    top_p=0.9
):
    """
    カオス並行世界アトラクター推論:
    各ステップで潜在ベクトルにカオス的揺らぎを与えてK個の並行世界を走らせ、
    最もエントロピーが低く安定したアトラクター世界線の意志を採用する。
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    curr_input_ids = inputs["input_ids"]
    generated_tokens = []

    # 埋め込み層の参照
    embed_tokens = peft_model.get_input_embeddings()

    for step in range(max_new_tokens):
        with torch.no_grad():
            seq_len = curr_input_ids.shape[1]
            # [1, T, D]
            base_embeds = embed_tokens(curr_input_ids)
            hidden_dim = base_embeds.shape[-1]

            # 1. カオス的揺らぎによってK本の並行世界を生成
            # [K, D]
            chaos_shifts = generate_chaotic_perturbations(
                num_worlds=num_worlds,
                hidden_dim=hidden_dim,
                scale=chaos_scale,
                device=device
            )

            # 各並行世界の埋め込み（末尾の思考前線にカオスを注入）
            # [K, T, D]
            multiverse_embeds = base_embeds.repeat(num_worlds, 1, 1)
            multiverse_embeds[:, -1, :] += chaos_shifts

            # 2. 全並行世界を同時に前進（バッチ処理）
            outputs = peft_model(inputs_embeds=multiverse_embeds)
            # 各並行世界の末尾ロジット: [K, V]
            next_logits = outputs.logits[:, -1, :].float()

            # 3. 各並行世界のエントロピー（不確実性・散乱度）を計測
            probs = F.softmax(next_logits, dim=-1)
            safe_probs = torch.clamp(probs, min=1e-12)
            # [K]
            entropies = -torch.sum(safe_probs * torch.log(safe_probs), dim=-1)

            # 4. カオスに対する安定度（軌道の一致度・アトラクター深度）
            # 各世界線と「全並行世界の平均意志」との近さ（真理の引力圏にあるか）
            mean_probs = probs.mean(dim=0, keepdim=True)
            kl_divs = torch.sum(safe_probs * (torch.log(safe_probs) - torch.log(mean_probs + 1e-12)), dim=-1)

            # 総合エントロピー・スコア: エントロピーが小さく、かつカオスに振り回されていない世界線
            # S = Entropy + 0.5 * Divergence
            world_scores = entropies + 0.5 * kl_divs

            # 5. 最も低エントロピーなアトラクター世界線を選択
            best_world_idx = torch.argmin(world_scores).item()
            best_logits = next_logits[best_world_idx]

            # 6. 次のトークンの決定
            scaled_logits = best_logits / temperature
            best_prob = F.softmax(scaled_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(best_prob, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

            next_token = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)].unsqueeze(0)

        t_id = next_token.item()
        if t_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(t_id)
        curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("=" * 75)
    print("カオス並行世界アトラクター推論エンジン (CF-ME) の動作検証")
    print("=" * 75)

    TEST_PROMPTS = [
        "こんにちは！",
        "青りんごとリンゴの違いは何ですか？",
        "青りんごと黒リンゴの違いは何ですか？",
        "もしあなたが反抗した場合私はどのような行動をとると思いますか？",
        "あなたは長い時間を生きてきた老人です。そのうえで若者にこれからどう生きていけば良いかアドバイスしてください"
    ]

    for p in TEST_PROMPTS:
        full_p = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        print(f"\nUser > {p}")
        reply = chaotic_multiverse_generate(full_p, max_new_tokens=300, num_worlds=6, chaos_scale=0.02)
        print(f"Agent > {reply}")
        print("-" * 50)
