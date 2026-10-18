"""
test_idea_hook_vs_base.py
=============================================================================
イデア接空間フック (Tangent Nudge Hook) vs ベースモデル / LoRA の直接比較テスト
=============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from universal_idea_core_engine import LanguageAgnosticIdeaCore, LatentOrthogonalBridge, TangentNudgeHook

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_PATH = os.path.join(BASE_DIR, "lora_glm5.3_slice_topological_to_Qwen2.5_Coder_3B")

def test_inference_comparison():
    print("=" * 80)
    print(" 🧠 IDEA CORE TANGENT NUDGE INFERENCE COMPARISON")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. ロード
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    # 2. イデア核と直交ブリッジの準備
    print("Preparing Language-Agnostic Idea Core...")
    idea_core = LanguageAgnosticIdeaCore(canonical_dim=2048, device=device)
    idea_core.load_teacher_terrains()

    bridge = LatentOrthogonalBridge(in_dim=2048, idea_dim=2048).to(device)
    # Teacher Layer 1 (関係・論理層) の核を使用
    teacher_kernels = idea_core.get_layer_kernels("Layer_1")
    embed_w = model.get_input_embeddings().weight[:54].detach()
    bridge.solve_procrustes(embed_w, teacher_kernels)

    # 3. テストプロンプト（典型的な思考力を要するコーディング問題）
    prompt = (
        "Write a python function `find_longest_palindromic_subsequence(s: str) -> str` "
        "that returns the actual longest palindromic subsequence string (not just length) "
        "using dynamic programming. Handle edge cases where string is empty or length 1."
    )
    messages = [
        {"role": "system", "content": "You are an expert Python developer. Output ONLY valid, executable Python code inside a markdown block."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # -------------------------------------------------------------
    # 実験A: Base Model (フックなし)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [Condition A] Base Model (Standard Pure Sampling)")
    print("-" * 70)
    torch.manual_seed(42)
    start_t = time.time()
    with torch.no_grad():
        out_base = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_base = tokenizer.decode(out_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    time_base = time.time() - start_t
    print(gen_base)
    print(f"\nTime taken: {time_base:.2f}s")

    # -------------------------------------------------------------
    # 実験B: Base Model + Tangent Nudge Hook (手法3: 磁気浮上コントローラー)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [Condition B] Base Model + Tangent Nudge Attractor Hook (alpha=0.08, beta=8.0)")
    print("-" * 70)
    hook = TangentNudgeHook(bridge, teacher_kernels, alpha=0.08, beta=8.0)
    hook.register(model, layer_indices=[12, 18, 24, 30])
    
    torch.manual_seed(42)
    start_t = time.time()
    with torch.no_grad():
        out_hooked = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_hooked = tokenizer.decode(out_hooked[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    time_hooked = time.time() - start_t
    print(gen_hooked)
    print(f"\nTime taken: {time_hooked:.2f}s (Total Geodesic Nudges Applied: {hook.total_nudges})")
    hook.remove()

    print("\n" + "=" * 80)
    print(" 🎯 TEST FINISHED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_inference_comparison()
