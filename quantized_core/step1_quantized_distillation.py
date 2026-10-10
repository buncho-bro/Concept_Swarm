"""
【ファイル概要】
step1_quantized_distillation.py

量子化されたモデル重みを用いた蒸留実験の第1ステップ。メモリ帯域を節約しつつ概念トポロジーを転写します。
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import numpy as np

# --- 設定 ---
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ビー玉（地形を観測するための固定アンカーポイント）
ANCHOR_WORDS = [
    "人間", "動物", "植物", "機械", "宇宙", "自然", 
    "感情", "論理", "科学", "芸術", "時間", "空間",
    "物質", "精神", "善", "悪",
    "生命", "死", "水", "火", "光", "闇", 
    "言葉", "数字", "音", "色", "力", "情報", 
    "社会", "個人", "過去", "未来"
]

# 2. 学習させるターゲット概念（今回はプロトタイプとして厳選）
TARGET_WORDS = [
    "犬", "猫", "車", "コンピュータ", "愛", "憎しみ", "りんご", "海",
    "哲学", "物理学", "時計", "星", "正義", "犯罪", "細胞", "ロケット",
    "夢", "現実", "平和", "戦争", "太陽", "月", "男", "女"
]

def get_word_embeddings(model, tokenizer, words, batch_size=8):
    """単語の最終トークンのベクトルを抽出"""
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # 最後のトークンのベクトルを取得
            for b_idx in range(len(batch)):
                # paddingを考慮して実際の最後のトークン位置を探す
                seq_len = inputs.attention_mask[b_idx].sum().item()
                hidden = outputs.hidden_states[-1][b_idx, seq_len-1, :]
                all_vecs.append(hidden.cpu())
                
    vecs = torch.stack(all_vecs).to(device)
    return F.normalize(vecs.to(torch.float32), p=2, dim=1)

def get_cheat_sheet(target_vecs, anchor_vecs, temperature=0.05):
    """
    【カンニングペーパーの作成】
    ターゲット単語が、16個のビー玉（アンカー）のどの谷に落ちたかの確率分布（粗い地形）を作成する
    """
    # Cosine Similarity を計算 [batch, num_anchors]
    sim_matrix = torch.matmul(target_vecs, anchor_vecs.T)
    # Softmax で確率分布（谷への所属度）に変換
    cheat_sheet = F.softmax(sim_matrix / temperature, dim=-1)
    return cheat_sheet

def step1_quantized_distillation():
    print("==================================================")
    print("Topographical Quantization (The Cheat Sheet Method)")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # --------------------------------------------------
    # [Phase 1] Teacher (7B) からカンニングペーパーを作成
    # --------------------------------------------------
    print(f"\nLoading Teacher Model ({TEACHER_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    print("Extracting Teacher's Anchor Coordinates (ビー玉の配置)...")
    t_anchor_vecs = get_word_embeddings(teacher_model, tokenizer, ANCHOR_WORDS)
    
    print("Extracting Teacher's Target Coordinates...")
    t_target_vecs = get_word_embeddings(teacher_model, tokenizer, TARGET_WORDS)
    
    # Teacherのカンニングペーパーを作成
    t_cheat_sheet = get_cheat_sheet(t_target_vecs, t_anchor_vecs)
    
    # メモリ解放
    del teacher_model
    torch.cuda.empty_cache()
    
    # --------------------------------------------------
    # [Phase 2] Student (1.5B) によるカンニングペーパーの学習
    # --------------------------------------------------
    print(f"\nLoading Student Model ({STUDENT_ID})...")
    tokenizer_s = AutoTokenizer.from_pretrained(STUDENT_ID)
    tokenizer_s.pad_token = tokenizer_s.eos_token
    base_student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    print("Extracting Student's Anchor Coordinates (ビー玉の配置)...")
    s_anchor_vecs = get_word_embeddings(base_student, tokenizer_s, ANCHOR_WORDS)
    
    lora_config = LoraConfig(
        r=64, # カンニングペーパーなら解像度が粗いため、ランクも小さくて済む
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(base_student, lora_config)
    
    optimizer = optim.Adam(student.parameters(), lr=3e-4)
    epochs = 60
    
    print("\nStarting Quantized Distillation (KL Divergence)...")
    student.train()
    
    for epoch in range(epochs):
        # Studentの現在のターゲット座標を抽出 (勾配を流すため手動で計算)
        inputs = tokenizer_s(TARGET_WORDS, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = student(**inputs, output_hidden_states=True)
        
        s_vecs_list = []
        for b_idx in range(len(TARGET_WORDS)):
            seq_len = inputs.attention_mask[b_idx].sum().item()
            hidden = outputs.hidden_states[-1][b_idx, seq_len-1, :]
            s_vecs_list.append(hidden)
            
        s_target_vecs = torch.stack(s_vecs_list)
        s_target_vecs = F.normalize(s_target_vecs.to(torch.float32), p=2, dim=1)
        
        # Studentのカンニングペーパーを作成
        s_cheat_sheet = get_cheat_sheet(s_target_vecs, s_anchor_vecs)
        
        # カンニングペーパー同士のズレ（KL Divergence）を計算
        # Teacherの分布に近づける
        loss = F.kl_div(torch.log(s_cheat_sheet + 1e-8), t_cheat_sheet, reduction='batchmean')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss (Cheat Sheet Gap): {loss.item():.5f}")
            
    print("\n=> SUCCESS: The Student has successfully learned the Quantized Cheat Sheet!")
    
    # 保存
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantized_lora")
    print(f"Saving Quantized LoRA to: {save_path}")
    student.save_pretrained(save_path)

if __name__ == "__main__":
    step1_quantized_distillation()
