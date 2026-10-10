"""
【ファイル概要】
step3_knn_graph_distillation.py

高次元空間におけるK近傍(KNN)グラフを用いて、局所的な概念のトポロジー関係をより高精度に小モデルへ蒸留する実験。
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import numpy as np
from tqdm import tqdm

# --- 設定 ---
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ネットワークを形成するための概念（純粋な座標群）
CONCEPT_WORDS = [
    "科学", "技術", "物理学", "量子力学", "宇宙", "ブラックホール", "時間", "空間", "次元",
    "哲学", "倫理", "道徳", "善悪", "存在", "意識", "思考", "論理", "真理",
    "生命", "進化", "DNA", "細胞", "生物学", "動物", "人間", "脳", "感情",
    "数学", "幾何学", "代数", "方程式", "無限", "確率", "統計",
    "芸術", "音楽", "絵画", "文学", "詩", "美", "創造", "表現",
    "社会", "政治", "経済", "法律", "国家", "権力", "民主主義", "歴史",
    "自然", "環境", "地球", "海", "山", "気候", "太陽", "月", "星",
    "コンピュータ", "人工知能", "データ", "アルゴリズム", "ネットワーク", "情報", "プログラミング",
    "宗教", "神", "信仰", "祈り", "魂", "奇跡", "神話",
    "日常", "食事", "睡眠", "仕事", "遊び", "家族", "友人", "愛", "憎しみ", "喜び", "悲しみ",
    "犬", "猫", "鳥", "魚", "昆虫", "車", "飛行機", "船", "電車", "自転車",
    "りんご", "バナナ", "みかん", "肉", "野菜", "パン", "米", "水", "お茶", "コーヒー",
    "赤", "青", "緑", "黄", "白", "黒", "丸", "四角", "三角", "線"
] # 112語

def get_word_embeddings_batch(model, tokenizer, words, batch_size=8):
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(words), batch_size), desc="Extracting"):
            batch = words[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            for b_idx in range(len(batch)):
                seq_len = inputs.attention_mask[b_idx].sum().item()
                hidden = outputs.hidden_states[-1][b_idx, seq_len-1, :]
                all_vecs.append(hidden.cpu())
    vecs = torch.stack(all_vecs).to(device)
    return F.normalize(vecs.to(torch.float32), p=2, dim=1)

def step3_knn_graph_distillation():
    print("==================================================")
    print("KNN Graph Distillation (The Constellation Method)")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # --------------------------------------------------
    # [Phase 1] Teacher (7B) から KNNグラフ（星座）を作成
    # --------------------------------------------------
    print(f"\nLoading Teacher Model ({TEACHER_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    t_vecs = get_word_embeddings_batch(teacher_model, tokenizer, CONCEPT_WORDS)
    del teacher_model
    torch.cuda.empty_cache()
    
    print("Creating K-Nearest Neighbors Graph (星座の生成)...")
    K = 8 # 最も近い8個の隣人だけを接続する（それ以外は切り捨てる）
    temperature = 0.05
    
    # Teacherのペアワイズ行列 [N, N]
    M_T = torch.matmul(t_vecs, t_vecs.T)
    # 上位K個のインデックスを取得
    topk_values, topk_indices = torch.topk(M_T, k=K, dim=-1)
    
    # 遠い単語への引力を完全にゼロ（-inf）にするマスクを作成
    mask = torch.zeros_like(M_T, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values, dtype=torch.bool))
    
    M_T_masked = M_T.clone()
    M_T_masked[~mask] = -1e9
    
    # カクカクしたネットワーク（スパース確率分布）
    P_T = F.softmax(M_T_masked / temperature, dim=-1)
    
    # --------------------------------------------------
    # [Phase 2] Student (0.5B) による星座の学習
    # --------------------------------------------------
    print(f"\nLoading Student Model ({STUDENT_ID})...")
    tokenizer_s = AutoTokenizer.from_pretrained(STUDENT_ID)
    tokenizer_s.pad_token = tokenizer_s.eos_token
    base_student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(base_student, lora_config)
    optimizer = optim.Adam(student.parameters(), lr=3e-4)
    epochs = 40
    batch_size = 16
    
    print("\nStarting KNN Graph Distillation...")
    student.train()
    
    for epoch in range(epochs):
        # メモリ節約のため、バッチごとに学習
        indices = np.arange(len(CONCEPT_WORDS))
        np.random.shuffle(indices)
        total_loss = 0
        
        for i in range(0, len(CONCEPT_WORDS), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_words = [CONCEPT_WORDS[idx] for idx in batch_idx]
            
            inputs = tokenizer_s(batch_words, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = student(**inputs, output_hidden_states=True)
            
            s_vecs_list = []
            for b_idx in range(len(batch_words)):
                seq_len = inputs.attention_mask[b_idx].sum().item()
                hidden = outputs.hidden_states[-1][b_idx, seq_len-1, :]
                s_vecs_list.append(hidden)
            s_vecs = torch.stack(s_vecs_list)
            s_vecs = F.normalize(s_vecs.to(torch.float32), p=2, dim=1)
            
            # 全単語（固定されたStudentベースベクトル）との距離を計算するため、
            # まずベース状態で全ベクトルを抽出（勾配なし）
            with torch.no_grad():
                with student.disable_adapter():
                    s_all_base_vecs = get_word_embeddings_batch(student, tokenizer_s, CONCEPT_WORDS, batch_size=32)
            
            # 学習中のバッチベクトルと、ベース全ベクトルとの類似度行列
            M_S = torch.matmul(s_vecs, s_all_base_vecs.T)
            
            # Teacherと同じマスク（ネットワークの線）だけを学習
            M_S_masked = M_S.clone()
            M_S_masked[~mask[batch_idx]] = -1e9
            P_S = F.softmax(M_S_masked / temperature, dim=-1)
            
            # KL Divergence
            loss = F.kl_div(torch.log(P_S + 1e-8), P_T[batch_idx], reduction='batchmean')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss (Constellation Gap): {total_loss/(len(CONCEPT_WORDS)//batch_size):.5f}")
            
    # 保存
    save_path = os.path.join(BASE_DIR, "knn_graph_lora")
    print(f"\nSaving KNN Graph LoRA to: {save_path}")
    student.save_pretrained(save_path)

if __name__ == "__main__":
    step3_knn_graph_distillation()
