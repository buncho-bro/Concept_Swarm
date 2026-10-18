import os
import sys
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
OLD_LORA_PATH = r"C:\Users\lop3l\.gemini\antigravity\scratch\concept_core\distillation_experiment\lora_glm5.3_to_Qwen2.5_Coder_3B_Topological"
PURE_LORA_PATH = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\distillation_experiments\lora_glm5.3_pure_to_Qwen2.5_Coder_3B"

PROMPTS = [
    {
        "title": "スレッドセーフ & TTL付きLRUキャッシュの実装",
        "prompt": "PythonでスレッドセーフかつTTL（ミリ秒単位の有効期限）付きのLRUキャッシュを自作してください。Lock制御、有効期限切れのパージ、OrderedDictまたは双方向連結リストによるO(1)アクセスを含め、解説と動作確認コードを示してください。"
    },
    {
        "title": "分散システムにおけるスプリットブレイン防止（Quorum合意シミュレーション）",
        "prompt": "分散システムで発生するスプリットブレイン（Split-Brain）問題の根本原因と、Quorum（過半数合意）による防御機構を解説し、Pythonでノードのネットワーク分断と合意判定をシミュレートする簡潔なコードを提示してください。"
    }
]

def generate_response(model, tokenizer, prompt_text, max_new_tokens=450):
    messages = [
        {"role": "system", "content": "You are a senior software architect and expert engineer."},
        {"role": "user", "content": prompt_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start_time
    response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    tps = len(response_tokens) / elapsed if elapsed > 0 else 0
    return text.strip(), elapsed, tps

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model

def main():
    print("=" * 75)
    print("GLM-5.3 蒸留LoRA 比較検証 (Base vs Old 25単語アンカー vs Pure 154k語自律大円干渉)")
    print("=" * 75)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    results = {}
    
    # 1. Base Model evaluation
    print("\n[1/3] Loading and Evaluating Base Model (Qwen2.5-Coder-3B)...")
    model = load_base_model()
    results["Base Model"] = []
    for item in PROMPTS:
        print(f"  - Generating for: {item['title']}...")
        ans, elap, tps = generate_response(model, tokenizer, item["prompt"])
        results["Base Model"].append({"title": item["title"], "text": ans, "time": elap, "tps": tps})
    del model
    gc.collect()
    torch.cuda.empty_cache()
        
    # 2. Old LoRA evaluation (25-word dictionary anchor)
    if os.path.exists(OLD_LORA_PATH):
        print("\n[2/3] Loading and Evaluating Old LoRA (25-word Dictionary Anchor)...")
        base = load_base_model()
        model = PeftModel.from_pretrained(base, OLD_LORA_PATH)
        results["Old LoRA (25-word Anchor)"] = []
        for item in PROMPTS:
            print(f"  - Generating for: {item['title']}...")
            ans, elap, tps = generate_response(model, tokenizer, item["prompt"])
            results["Old LoRA (25-word Anchor)"].append({"title": item["title"], "text": ans, "time": elap, "tps": tps})
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\nOld LoRA not found at {OLD_LORA_PATH}, skipping...")

    # 3. Pure LoRA evaluation (154k vocab autonomous great circle)
    if os.path.exists(PURE_LORA_PATH):
        print("\n[3/3] Loading and Evaluating Pure LoRA (154,880-token Autonomous Great Circle)...")
        base = load_base_model()
        model = PeftModel.from_pretrained(base, PURE_LORA_PATH)
        results["Pure LoRA (Autonomous Great Circle)"] = []
        for item in PROMPTS:
            print(f"  - Generating for: {item['title']}...")
            ans, elap, tps = generate_response(model, tokenizer, item["prompt"])
            results["Pure LoRA (Autonomous Great Circle)"].append({"title": item["title"], "text": ans, "time": elap, "tps": tps})
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\nPure LoRA not found at {PURE_LORA_PATH}, skipping...")

    # 出力レポート作成
    report_file = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\distillation_experiments\glm53_lora_comparison_result.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# GLM-5.3 蒸留LoRA A/B/Base 比較検証結果\n\n")
        f.write("> **検証モデル**:\n")
        f.write("> 1. **Base Model**: Qwen2.5-Coder-3B-Instruct\n")
        f.write("> 2. **Old LoRA**: 人間指定の25個のプログラミング辞書単語による蒸留\n")
        f.write("> 3. **Pure LoRA**: GLM-5.3 全154,880トークン埋め込みから大円干渉で自律析出させた54個のイデア核による真の0VRAM蒸留\n\n")
        
        for i, item in enumerate(PROMPTS):
            f.write(f"## タスク {i+1}: {item['title']}\n\n")
            f.write(f"**プロンプト**: {item['prompt']}\n\n")
            for model_name, res_list in results.items():
                res = res_list[i]
                f.write(f"### {model_name} (所要時間: {res['time']:.2f}s, 生成速度: {res['tps']:.1f} tokens/s)\n\n")
                f.write(f"```python\n{res['text']}\n```\n\n")
                f.write("---\n\n")
                
    print(f"\nComparison report written to: {report_file}")
    
    # ターミナルにプレビュー出力
    for i, item in enumerate(PROMPTS):
        print(f"\n{'='*30} TASK {i+1}: {item['title']} {'='*30}")
        for model_name, res_list in results.items():
            print(f"\n--- {model_name} ({res_list[i]['tps']:.1f} tok/s) ---")
            print(res_list[i]['text'][:250] + "\n...(略)...")

if __name__ == "__main__":
    main()
