"""
【ファイル概要】
step3_visualizability_tester.py

プロジェクトの一部として作成されたPythonスクリプト。
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 設定 ---
STUDENT_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(BASE_DIR, "vision_aligned_lora")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_prompt_embedding(model, tokenizer, text):
    """文章（プロンプト）全体のベクトルを抽出する"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # 全単語の平均（Mean Pooling）をやめ、
    # 全ての文脈の重力が一点に収束する「最後のトークン（Last Token）」の座標だけを抽出する
    hidden = outputs.hidden_states[-1][0, -1, :]
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def step3_visualizability_tester():
    print("==================================================")
    print("Cross-modal Intersection (Visualizability Tester v3)")
    print("==================================================")
    
    if not os.path.exists(LORA_DIR):
        print(f"ERROR: LoRA not found at {LORA_DIR}. Please run step 1 first.")
        return

    print("Loading Base Logic Brain (Qwen 1.5B) and Vision Brain (LoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    # LoRAを読み込む
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    
    while True:
        print("\n" + "-"*50)
        target_prompt = input("画像生成プロンプト（文章）を入力してください (終了は exit): ")
        if target_prompt.lower() == "exit":
            break
            
        print("🧠 [Base Brain: Extracting Logical Coordinates...]")
        # 論理の脳での座標
        with model.disable_adapter():
            vec_logic = get_prompt_embedding(model, tokenizer, target_prompt)
        
        print("👁️  [Vision Brain: Extracting Visual Coordinates...]")
        # 視覚の脳での座標
        vec_vision = get_prompt_embedding(model, tokenizer, target_prompt)
        
        # 2つの脳の間での「移動距離（吹っ飛び具合）」を計測
        # Cosine Similarity が高い（1.0に近い）ほど位置が安定している
        stability_score = F.cosine_similarity(vec_logic.unsqueeze(0), vec_vision.unsqueeze(0)).item()
        
        print(f"\n=> Conceptual Stability Score (視覚空間での安定度): {stability_score:.4f}")
        
        # 安定度が閾値を超えているか判定（閾値は実験に応じて調整可能）
        THRESHOLD = 0.60
        if stability_score >= THRESHOLD:
            print("🟢 判定: 【画像生成可能】")
            print("理由: 論理の脳と視覚の脳で概念の位置が安定しています（物理的実体あり）。")
            print(f"      [Simulated] Generating Image for '{target_prompt}'...")
        else:
            print("🔴 判定: 【画像生成拒否】")
            print("理由: 視覚の脳に切り替えた瞬間、視覚的実体がないため空間の果てに吹き飛ばされました（抽象概念）。")

if __name__ == "__main__":
    step3_visualizability_tester()
