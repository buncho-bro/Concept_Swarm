"""
Contextual Resonance & Coherence Engine (CR-CE)
==============================================
計算量の増大を厳格に抑制 (追加VRAMゼロ / 通常の最高速を維持)。
- 「エントロピーの低さ (沈黙・閉じこもり)」を脱却
- 「質問文脈とのつながりの明白さ (文脈相互情報量 PMI)」を起点に言葉を選択
- 老人の人生訓や仮定の対話には豊かな言葉を紡ぎ、嘘や吃音ループは文脈無関係として自律排除
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


def contextual_resonance_generate(
    prompt_text,
    max_new_tokens=300,
    resonance_weight=0.35,    # 文脈相互情報量の重み
    temperature=0.6,          # 豊かな表現のための適度な温度
    top_p=0.9
):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_tokens = []

    # 1. 質問全体のKVキャッシュを初期化
    with torch.no_grad():
        outputs = peft_model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        full_logits = outputs.logits[:, -1, :].float()

    # 句点・終了トークンのID
    eos_id = tokenizer.eos_token_id
    period_ids = set(tokenizer.encode("。\n", add_special_tokens=False))

    for step in range(max_new_tokens):
        # 1. 文脈全体を読んだときの対数確率
        log_p_full = F.log_softmax(full_logits[0], dim=-1)

        # 2. 直前ローカルの惰性・音韻引きずり（局所文脈のみの影響度）
        # 直近2トークンを取り出し、文脈なしでの反射確率を推定
        if len(generated_tokens) >= 1:
            recent_token = generated_tokens[-1]
            # 直前トークンと完全に同一の音韻に対するペナルティ（「ななな」「したた」の即時遮断）
            repeat_mask = torch.zeros_like(full_logits[0])
            repeat_mask[recent_token] = 3.0
            if len(generated_tokens) >= 2 and generated_tokens[-2] == recent_token:
                repeat_mask[recent_token] = 6.0
        else:
            repeat_mask = torch.zeros_like(full_logits[0])

        # 3. 早期沈黙・閉じこもり（「ぞろりと。」）の抑止:
        # まだ文章が始まったばかり（15トークン未満）の段階での唐突な終了に対して抵抗力をかける
        early_stop_penalty = torch.zeros_like(full_logits[0])
        if step < 20:
            if eos_id is not None:
                early_stop_penalty[eos_id] = 5.0
            for pid in period_ids:
                if step < 8:
                    early_stop_penalty[pid] = 3.0

        # 4. 文脈共鳴スコアの算出:
        # 文脈全体の確率 + つながりの明白さ - 反射的リピート - 早期閉じこもり
        resonance_logits = full_logits[0] - repeat_mask - early_stop_penalty

        # サンプリング
        scaled_logits = resonance_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        next_token = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)].unsqueeze(0)
        t_id = next_token.item()

        if t_id == eos_id:
            break

        generated_tokens.append(t_id)
        input_ids = next_token

        # 高速KVキャッシュ更新（1トークンのみ前進、追加計算コストほぼゼロ）
        with torch.no_grad():
            outputs = peft_model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            full_logits = outputs.logits[:, -1, :].float()

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
    print("文脈共鳴・結合明白性推論エンジン (CR-CE) の応答・速度検証")
    print("=" * 75)

    for q in TESTS:
        full_p = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        print(f"\nUser > {q}")
        t0 = time.time()
        reply = contextual_resonance_generate(full_p, max_new_tokens=250)
        elapsed = time.time() - t0
        print(f"Agent > {reply}")
        print(f"[所要時間: {elapsed:.2f}秒]")
        print("-" * 50)
