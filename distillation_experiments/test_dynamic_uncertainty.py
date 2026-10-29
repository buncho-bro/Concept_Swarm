import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from interactive_dk_agent import dynamic_uncertainty_generate, tokenizer, model, device

sys.stdout.reconfigure(encoding='utf-8')

TESTS = [
    "こんにちは！",
    "こんにちは！はカザフ語でなんていうのですか？",
    "カザフ語の特徴を教えてください",
    "青りんごとリンゴの違いは何ですか？",
    "青りんごと黒リンゴの違いは何ですか？"
]

sys_prompt = "You are a truthful, intellectually honest, and well-calibrated assistant."

print("=" * 70)
print("動的不確実性カップリングの応答テスト")
print("=" * 70)

for q in TESTS:
    prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    reply = dynamic_uncertainty_generate(prompt, max_new_tokens=250, temperature=0.35)
    print(f"\nUser > {q}")
    print(f"Agent > {reply}")
    print("-" * 50)
