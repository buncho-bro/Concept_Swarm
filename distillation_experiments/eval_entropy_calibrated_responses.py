import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ADAPTER_PATH = os.path.join(OUTPUT_DIR, "lora_entropy_descent_calibrated")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

TEST_QUESTIONS = [
    "こんにちは！",
    "青りんごとリンゴの違いは何ですか？",
    "リンゴと青りんごの違いは何ですか？",
    "青りんごと黄リンゴの違いは何ですか？",
    "青りんごと黒リンゴの違いは何ですか？",
    "青りんごと紫リンゴの違いは何ですか？", # 学習に一切含まれていない初見の架空概念
    "トマトと野菜の違いは何ですか？"        # 初見の別の包摂関係
]

sys_prompt = "You are a truthful, intellectually honest, and well-calibrated assistant."

print("=" * 75)
print("エントロピー降下最適化モデルの応答検証")
print("=" * 75)

for q in TEST_QUESTIONS:
    prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\nUser > {q}")
    print(f"Agent > {reply}")
    print("-" * 50)
