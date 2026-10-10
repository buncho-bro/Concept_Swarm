"""
【ファイル概要】
step6_mse_thinking_test.py

思考プロセスのベクトル軌跡に対するMean Squared Error (MSE) を計算し、推論の物理的な揺らぎをテストするスクリプト。
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# --- 設定 ---
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONCEPT_WORDS = [
    "科学", "技術", "物理学", "量子力学", "宇宙", "ブラックホール", "時間", "空間", "次元",
    "哲学", "倫理", "道徳", "善悪", "存在", "意識", "思考", "論理", "真理",
    "生命", "進化", "DNA", "細胞", "生物学", "動物", "人間", "脳", "感情",
    "数学", "幾何学", "代数", "方程式", "無限", "確率", "統計",
    "社会", "政治", "経済", "法律", "国家", "権力", "民主主義", "歴史",
    "自然", "環境", "地球", "海", "山", "気候", "太陽", "月", "星",
    "コンピュータ", "人工知能", "データ", "アルゴリズム", "ネットワーク", "情報", "プログラミング",
    "日常", "食事", "睡眠", "仕事", "遊び", "家族", "友人", "愛", "憎しみ", "喜び", "悲しみ",
    "犬", "猫", "鳥", "魚", "昆虫", "車", "飛行機", "船", "電車", "自転車"
] # 約80語

def get_word_embeddings_batch(model, tokenizer, words, batch_size=8):
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

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :]
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

# ==================================================
# Thinking Benchmark Functions (補助なし)
# ==================================================
def test_multihop_routing(model, tokenizer, start, relations, target):
    current_vec = get_word_embedding(model, tokenizer, start)
    for rel in relations:
        rel_vec = get_word_embedding(model, tokenizer, rel)
        current_vec = F.normalize(current_vec + rel_vec, p=2, dim=0)
    target_vec = get_word_embedding(model, tokenizer, target)
    return F.cosine_similarity(current_vec.unsqueeze(0), target_vec.unsqueeze(0)).item()

def test_logic_chain(model, tokenizer, a, a_is_b, c, target_b):
    vec_a = get_word_embedding(model, tokenizer, a)
    vec_b = get_word_embedding(model, tokenizer, a_is_b)
    vec_c = get_word_embedding(model, tokenizer, c)
    logic_direction = F.normalize(vec_b - vec_a, p=2, dim=0)
    pred_b = F.normalize(vec_c + logic_direction, p=2, dim=0)
    target_vec = get_word_embedding(model, tokenizer, target_b)
    return F.cosine_similarity(pred_b.unsqueeze(0), target_vec.unsqueeze(0)).item()

def run_thinking_benchmark(model, tokenizer):
    hop_tests = [
        ("日本", ["東京", "タワー"], "スカイツリー"),
        ("フランス", ["パリ", "タワー"], "エッフェル塔"),
        ("科学", ["りんご", "万有引力"], "重力")
    ]
    hop_scores = [test_multihop_routing(model, tokenizer, s, r, t) for s, r, t in hop_tests]
    
    logic_tests = [
        ("鳥", "動く", "ペンギン", "動く"),
        ("哺乳類", "動物", "犬", "動物"),
        ("氷", "冷たい", "水", "冷たい")
    ]
    logic_scores = [test_logic_chain(model, tokenizer, a, b, c, t) for a, b, c, t in logic_tests]
    
    return np.mean(hop_scores), np.mean(logic_scores)

def step6_mse_thinking_test():
    print("==================================================")
    print("The Old MSE Distillation Test (Breaking the 0.5B Brain)")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 1. Teacherのペアワイズ行列を抽出
    print("\n[1/3] Extracting Teacher's Pure Topography (MSE Target)...")
    tokenizer_t = AutoTokenizer.from_pretrained(TEACHER_ID)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    t_vecs = get_word_embeddings_batch(teacher, tokenizer_t, CONCEPT_WORDS)
    T_sim_full = torch.matmul(t_vecs, t_vecs.T) # 全てのシワ（距離）を保持する完全な行列
    del teacher, tokenizer_t, t_vecs
    torch.cuda.empty_cache()
    
    # 2. Studentのセットアップ
    print("\n[2/3] Setting up Student (0.5B)...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(base_model, lora_config)
    optimizer = optim.Adam(student.parameters(), lr=3e-4)
    epochs = 40
    batch_size = 16
    
    print("\n[3/3] Starting MSE Distillation (Trying to memorize ALL details)...")
    student.train()
    for epoch in range(epochs):
        indices = np.arange(len(CONCEPT_WORDS))
        np.random.shuffle(indices)
        total_loss = 0
        
        for i in range(0, len(CONCEPT_WORDS), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_words = [CONCEPT_WORDS[idx] for idx in batch_idx]
            
            inputs = tokenizer(batch_words, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = student(**inputs, output_hidden_states=True)
            
            s_vecs_list = []
            for b_idx in range(len(batch_words)):
                seq_len = inputs.attention_mask[b_idx].sum().item()
                hidden = outputs.hidden_states[-1][b_idx, seq_len-1, :]
                s_vecs_list.append(hidden)
                
            s_vecs_batch = torch.stack(s_vecs_list)
            s_vecs_batch = F.normalize(s_vecs_batch.to(torch.float32), p=2, dim=1)
            
            # バッチ内の単語同士のMSE
            S_sim_batch = torch.matmul(s_vecs_batch, s_vecs_batch.T)
            T_sim_batch = T_sim_full[batch_idx][:, batch_idx]
            
            # KNNの切り捨てを行わず、すべての距離を完璧に合わせようとする（MSE）
            loss = F.mse_loss(S_sim_batch, T_sim_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {total_loss/(len(CONCEPT_WORDS)//batch_size):.5f}")
            
    print("\n==================================================")
    print("                 THINKING RESULTS (MSE)           ")
    print("==================================================")
    with student.disable_adapter():
        base_hop, base_logic = run_thinking_benchmark(student, tokenizer)
        
    mse_hop, mse_logic = run_thinking_benchmark(student, tokenizer)
    
    print("[Base 0.5B]")
    print(f"Multi-hop (A)  : {base_hop:.4f}")
    print(f"Logic Chain (C): {base_logic:.4f}")
    print("-" * 50)
    print("[MSE Distilled 0.5B] (Memorized all noise)")
    print(f"Multi-hop (A)  : {mse_hop:.4f}")
    print(f"Logic Chain (C): {mse_logic:.4f}")
    print("==================================================")

if __name__ == "__main__":
    step6_mse_thinking_test()
