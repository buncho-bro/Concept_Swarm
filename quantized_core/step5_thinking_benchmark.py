"""
【ファイル概要】
step5_thinking_benchmark.py

モデルの「思考プロセス（Chain of Thought）」空間の軌跡を幾何学的に評価するためのベンチマーク。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 設定 ---
STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(BASE_DIR, "knn_graph_lora")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :] # Last token
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

# ==================================================
# A: Multi-hop Trajectory (多段推論)
# ==================================================
def test_multihop_routing(model, tokenizer, start, relations, target):
    """スタートから方向ベクトルを足し続け、補助なしでターゲットに到達できるか"""
    current_vec = get_word_embedding(model, tokenizer, start)
    
    for rel in relations:
        rel_vec = get_word_embedding(model, tokenizer, rel)
        # 補助輪（着地）なしで純粋に空中の軌跡を進む
        current_vec = F.normalize(current_vec + rel_vec, p=2, dim=0)
        
    target_vec = get_word_embedding(model, tokenizer, target)
    final_score = F.cosine_similarity(current_vec.unsqueeze(0), target_vec.unsqueeze(0)).item()
    return final_score

# ==================================================
# C: Logic Chain (三段論法・因果関係)
# ==================================================
def test_logic_chain(model, tokenizer, a, a_is_b, c, target_b):
    """
    前提1: AはBである (関係ベクトル1)
    前提2: CはAである
    推論: CはBであるか？ (Cから関係ベクトル1方向に飛んでBに到達できるか)
    """
    vec_a = get_word_embedding(model, tokenizer, a)
    vec_b = get_word_embedding(model, tokenizer, a_is_b)
    vec_c = get_word_embedding(model, tokenizer, c)
    
    # A->B の論理方向ベクトルを抽出
    logic_direction = F.normalize(vec_b - vec_a, p=2, dim=0)
    
    # Cからその論理方向に純粋ジャンプ
    pred_b = F.normalize(vec_c + logic_direction, p=2, dim=0)
    
    # 着地点がターゲット(B)と同じか確認
    target_vec = get_word_embedding(model, tokenizer, target_b)
    final_score = F.cosine_similarity(pred_b.unsqueeze(0), target_vec.unsqueeze(0)).item()
    return final_score

def run_thinking_benchmark(model, tokenizer):
    # --- Test A ---
    hop_tests = [
        ("日本", ["東京", "タワー"], "スカイツリー"),
        ("フランス", ["パリ", "タワー"], "エッフェル塔"),
        ("科学", ["りんご", "万有引力"], "重力")
    ]
    hop_scores = []
    for start, rels, tgt in hop_tests:
        score = test_multihop_routing(model, tokenizer, start, rels, tgt)
        hop_scores.append(score)
        
    # --- Test C ---
    logic_tests = [
        ("鳥", "動く", "ペンギン", "動く"), # 鳥は動く、ペンギンは鳥 -> ペンギンは動く
        ("哺乳類", "動物", "犬", "動物"),
        ("氷", "冷たい", "水", "冷たい")
    ]
    logic_scores = []
    for a, b, c, tgt in logic_tests:
        score = test_logic_chain(model, tokenizer, a, b, c, tgt)
        logic_scores.append(score)
        
    return np.mean(hop_scores), np.mean(logic_scores)

def step5_thinking_benchmark():
    print("==================================================")
    print("Thinking Benchmark (Concept Routing)")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading Models ({STUDENT_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    if os.path.exists(LORA_DIR):
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
    else:
        print(f"WARNING: LoRA not found at {LORA_DIR}. Will test base model only.")
        model = base_model
    
    print("\n[1/2] Running Thinking Benchmark on Base Model (No LoRA)...")
    with model.disable_adapter():
        base_hop, base_logic = run_thinking_benchmark(model, tokenizer)
        
    print("[2/2] Running Thinking Benchmark on KNN Graph Model (LoRA)...")
    quant_hop, quant_logic = run_thinking_benchmark(model, tokenizer)
    
    print("\n==================================================")
    print("                 THINKING RESULTS                 ")
    print("==================================================")
    print("[Base 0.5B]")
    print(f"Multi-hop (A)  : {base_hop:.4f}")
    print(f"Logic Chain (C): {base_logic:.4f}")
    print("-" * 50)
    print("[KNN Graph LoRA 0.5B] (Constellation Engine)")
    print(f"Multi-hop (A)  : {quant_hop:.4f}")
    print(f"Logic Chain (C): {quant_logic:.4f}")
    print("==================================================")
    print("*Score: 1.0 means perfect arrival at the target concept.*")

if __name__ == "__main__":
    step5_thinking_benchmark()
