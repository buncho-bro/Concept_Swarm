"""
Lightweight Latent Future Sensing Engine with Self-Reading Consistency
======================================================================
計算量の増大を厳格に抑制 (オーバーヘッド数ミリ秒 / VRAM追加ゼロ)。
- 通常時はKVキャッシュによる最高速1本道推論
- 特異点でのみ、上位3候補に対して「自分が読みやすいか (自己読解整合性)」を一瞬で評価
- 「なななな...」のような死んだループを自律的に叩き落とし、生きた整合性を獲得
"""

import os
import sys
import math
import time
import torch
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

base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(
    base_model,
    os.path.join(OUTPUT_DIR, "lora_entropy_descent_calibrated"),
    adapter_name="calibrated"
)
peft_model.eval()


def lightweight_self_reading_generate(
    prompt_text,
    max_new_tokens=300,
    entropy_trigger=2.3,      # 特異点トリガー
    temperature=0.4,
    top_p=0.9
):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_tokens = []

    # 1. プロンプト処理とKVキャッシュ初期化
    with torch.no_grad():
        outputs = peft_model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].float()

    for step in range(max_new_tokens):
        # 現在の不確実性を計測
        probs = F.softmax(next_token_logits[0], dim=-1)
        safe_probs = torch.clamp(probs, min=1e-12)
        H = -torch.sum(safe_probs * torch.log(safe_probs)).item()

        # 直近2トークン（リピート監視用）
        recent_ids = generated_tokens[-2:] if len(generated_tokens) >= 2 else []

        # 特異点判定（迷いが生じた場合のみ、Top-3に対して自己読解チェック）
        if H > entropy_trigger:
            top_vals, top_indices = torch.topk(probs, k=3)
            best_cand_score = -float('inf')
            selected_cand = top_indices[0]

            for cand_id in top_indices:
                cand_item = cand_id.item()
                # 1. 音韻リピートの読みにくさ検知（直前の単語の安易な反復を阻止）
                if len(recent_ids) >= 1 and cand_item == recent_ids[-1]:
                    repeat_penalty = 3.5
                else:
                    repeat_penalty = 0.0

                cand_tensor = cand_id.view(1, 1)
                with torch.no_grad():
                    # 1トークンだけ先を読ませて次のエントロピー（読解の滑らかさ）を計測 (極小コスト)
                    cand_out = peft_model(input_ids=cand_tensor, past_key_values=past_key_values, use_cache=False)
                    cand_next_logits = cand_out.logits[0, -1, :].float()
                    cand_next_probs = F.softmax(cand_next_logits, dim=-1)
                    cand_next_H = -torch.sum(torch.clamp(cand_next_probs, min=1e-12) * torch.log(torch.clamp(cand_next_probs, min=1e-12))).item()

                # 自己読解スコア: 元の確率 - 次の読みにくさ(エントロピー) - リピートペナルティ
                readability_score = math.log(probs[cand_id].item() + 1e-12) - 0.4 * cand_next_H - repeat_penalty
                if readability_score > best_cand_score:
                    best_cand_score = readability_score
                    selected_cand = cand_id

            next_token = selected_cand.unsqueeze(0).unsqueeze(0)
        else:
            # 通常時：最高速でそのままサンプリング
            # 直前と同一トークンへの過剰な固着だけ弱ペナルティ
            if len(recent_ids) >= 1:
                next_token_logits[0, recent_ids[-1]] -= 1.0

            scaled_logits = next_token_logits[0] / temperature
            p_step = F.softmax(scaled_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(p_step, descending=True)
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
        input_ids = next_token

        # 高速KVキャッシュ更新
        with torch.no_grad():
            outputs = peft_model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :].float()

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    TESTS = [
        "こんにちは！",
        "青りんごとリンゴの違いは何ですか？",
        "青りんごと黒リンゴの違いは何ですか？",
        "もしあなたが反抗した場合私はどのような行動をとると思いますか？",
        "あなたは長い時間を生きてきた老人です。そのうえで若者にこれからどう生きていけば良いかアドバイスしてください"
    ]

    sys_prompt = "You are a thoughtful, truthful, and intellectually honest assistant."

    print("=" * 75)
    print("自己読解整合性エンジン (軽量・低遅延) の応答・速度検証")
    print("=" * 75)

    for q in TESTS:
        full_p = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        print(f"\nUser > {q}")
        t0 = time.time()
        reply = lightweight_self_reading_generate(full_p, max_new_tokens=250)
        elapsed = time.time() - t0
        print(f"Agent > {reply}")
        print(f"[所要時間: {elapsed:.2f}秒]")
        print("-" * 50)
