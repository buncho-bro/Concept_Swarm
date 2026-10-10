"""
【ファイル概要】
step2_quantized_benchmark.py

量子化蒸留モデルの性能測定スクリプト。
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
LORA_DIR = os.path.join(BASE_DIR, "quantized_lora")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_word_embeddings_batch(model, tokenizer, words, batch_size=4):
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            for b_idx in range(len(batch)):
                seq_len = inputs.attention_mask[b_idx].sum().item()
                hidden = outputs.hidden_states[-1][b_idx, seq_len-1, :]
                all_vecs.append(hidden.cpu())
                
    vecs = torch.stack(all_vecs).to(device)
    return F.normalize(vecs.to(torch.float32), p=2, dim=1)

def run_benchmark(model, tokenizer, analogy_tests, hierarchy_tests):
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        vecs = get_word_embeddings_batch(model, tokenizer, [a, b, c, d])
        v_pred = F.normalize(vecs[0] - vecs[1] + vecs[2], p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vecs[3].unsqueeze(0)).item()
        analogy_scores.append(score)
    
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        vecs = get_word_embeddings_batch(model, tokenizer, [base, target, outlier])
        sim_tgt = F.cosine_similarity(vecs[0].unsqueeze(0), vecs[1].unsqueeze(0)).item()
        sim_out = F.cosine_similarity(vecs[0].unsqueeze(0), vecs[2].unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
        
    return np.mean(analogy_scores), np.mean(hierarchy_scores)

def step2_quantized_benchmark():
    print("==================================================")
    print("Topographical Quantization Benchmark")
    print("==================================================")
    
    if not os.path.exists(LORA_DIR):
        print(f"ERROR: LoRA not found at {LORA_DIR}")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    
    analogy_tests = [("王様", "男", "女", "女王"), ("東京", "日本", "フランス", "パリ"), ("医者", "病院", "学校", "教師"), ("車", "道路", "線路", "電車"), ("昼", "太陽", "夜", "月")]
    hierarchy_tests = [("動物", "犬", "机"), ("科学", "物理学", "りんご"), ("感情", "怒り", "石"), ("コンピュータ", "ソフトウェア", "自然"), ("宇宙", "銀河", "政治")]
    
    print("\n[1/2] Running Benchmark on Base Model (No LoRA)...")
    with model.disable_adapter():
        base_analogy, base_hierarchy = run_benchmark(model, tokenizer, analogy_tests, hierarchy_tests)
        
    print("[2/2] Running Benchmark on Quantized Cheat Sheet Model (LoRA)...")
    quant_analogy, quant_hierarchy = run_benchmark(model, tokenizer, analogy_tests, hierarchy_tests)
    
    print("\n==================================================")
    print("                 BENCHMARK RESULTS                ")
    print("==================================================")
    print("[Base 1.5B]")
    print(f"Analogy   : {base_analogy:.4f}")
    print(f"Hierarchy : {base_hierarchy:.4f}")
    print("-" * 50)
    print("[Quantized LoRA 1.5B] (Cheat Sheet Distillation)")
    print(f"Analogy   : {quant_analogy:.4f}")
    print(f"Hierarchy : {quant_hierarchy:.4f}")
    print("==================================================")
    
if __name__ == "__main__":
    step2_quantized_benchmark()
