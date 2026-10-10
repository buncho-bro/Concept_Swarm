"""
【ファイル概要】
step1_vision_topography_alignment.py

プロジェクトの一部として作成されたPythonスクリプト。
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPVisionModelWithProjection, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# --- 設定 ---
VISION_TEACHER_ID = "openai/clip-vit-base-patch32"
TEXT_STUDENT_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 の英語ラベルから日本語ラベルへのマッピング
cifar_labels = {
    0: "飛行機", 1: "自動車", 2: "鳥", 3: "猫", 4: "鹿",
    5: "犬", 6: "カエル", 7: "馬", 8: "船", 9: "トラック"
}

def load_cifar10_subset(samples_per_class=5):
    """CIFAR-10から各クラス数枚ずつの画像を抽出"""
    print("Loading CIFAR-10 dataset...")
    # 最近のHuggingFace仕様に合わせて namespace を追加
    dataset = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
    
    images = []
    labels_ja = []
    class_counts = {i: 0 for i in range(10)}
    
    for item in dataset:
        label = item['label']
        if class_counts[label] < samples_per_class:
            images.append(item['img'])
            labels_ja.append(cifar_labels[label])
            class_counts[label] += 1
            
        if all(count >= samples_per_class for count in class_counts.values()):
            break
            
    return images, labels_ja

def get_vision_embeddings(model, processor, images):
    print(f"Extracting Vision Embeddings from {VISION_TEACHER_ID}...")
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # CLIPの投影済み画像特徴量
    vision_embeds = outputs.image_embeds 
    return F.normalize(vision_embeds.to(torch.float32), p=2, dim=1)

def get_text_embeddings_batch(model, tokenizer, words, batch_size=16):
    model.eval()
    all_vecs = []
    
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            for b_idx in range(len(batch)):
                mask = inputs.attention_mask[b_idx].unsqueeze(-1)
                hidden = outputs.hidden_states[-1][b_idx] * mask
                vec = hidden.sum(dim=0) / mask.sum()
                all_vecs.append(vec.cpu())
                
    vecs = torch.stack(all_vecs).to(device)
    return F.normalize(vecs.to(torch.float32), p=2, dim=1)

def evaluate_topographical_alignment(vision_vecs, text_vecs):
    """画像地形とテキスト地形の一致度（MSE）を計算"""
    M_img = torch.matmul(vision_vecs, vision_vecs.T)
    M_txt = torch.matmul(text_vecs, text_vecs.T)
    mse = F.mse_loss(M_img, M_txt).item()
    return mse

def step1_multimodal_alignment():
    print("==================================================")
    print("Topographical Multimodal Alignment (Vision -> Text)")
    print("==================================================")
    
    # 1. データの準備
    # 各クラス10枚（合計100枚）の画像と対応する日本語ラベルを取得
    images, labels_ja = load_cifar10_subset(samples_per_class=10)
    print(f"Loaded {len(images)} images for multimodal topography matching.")
    
    # 2. 視覚地形（Teacher）の抽出
    processor = CLIPProcessor.from_pretrained(VISION_TEACHER_ID)
    vision_teacher = CLIPVisionModelWithProjection.from_pretrained(VISION_TEACHER_ID, use_safetensors=True).to(device)
    vision_teacher.eval()
    
    vision_vecs = get_vision_embeddings(vision_teacher, processor, images)
    
    # メモリ解放
    del vision_teacher, processor
    torch.cuda.empty_cache()
    
    # 3. 論理地形（Student）のロードと初期化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(TEXT_STUDENT_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(TEXT_STUDENT_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    # --- Before Alignment の計測 ---
    print("\nMeasuring topography before alignment...")
    text_vecs_before = get_text_embeddings_batch(base_model, tokenizer, labels_ja)
    mse_before = evaluate_topographical_alignment(vision_vecs, text_vecs_before)
    print(f"[Before] Topographical Discrepancy (MSE): {mse_before:.5f}")
    
    # 4. 地形同期（LoRA Distillation）
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(base_model, lora_config)
    
    optimizer = optim.Adam(student.parameters(), lr=2e-4)
    epochs = 25
    batch_size = 20
    
    print("\nStarting Multimodal Topographical Alignment (Pairwise MSE)...")
    student.train()
    
    for epoch in range(epochs):
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        total_loss = 0
        
        for i in range(0, len(images), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_labels = [labels_ja[idx] for idx in batch_idx]
            batch_v_vecs = vision_vecs[batch_idx]
            
            # テキストベクトルの抽出 (学習用)
            inputs = tokenizer(batch_labels, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = student(**inputs, output_hidden_states=True)
            
            s_vecs_list = []
            for b_idx in range(len(batch_labels)):
                mask = inputs.attention_mask[b_idx].unsqueeze(-1)
                hidden = outputs.hidden_states[-1][b_idx] * mask
                vec = hidden.sum(dim=0) / mask.sum()
                s_vecs_list.append(vec)
            
            s_vecs_batch = torch.stack(s_vecs_list)
            s_vecs_batch = F.normalize(s_vecs_batch.to(torch.float32), p=2, dim=1)
            
            # ペアワイズ行列の計算
            M_img = torch.matmul(batch_v_vecs, batch_v_vecs.T)
            M_txt = torch.matmul(s_vecs_batch, s_vecs_batch.T)
            
            loss = F.mse_loss(M_txt, M_img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss (Topography Gap): {total_loss/(len(images)//batch_size):.5f}")
        
    # --- After Alignment の計測 ---
    print("\nMeasuring topography after alignment...")
    student.eval()
    text_vecs_after = get_text_embeddings_batch(student, tokenizer, labels_ja)
    mse_after = evaluate_topographical_alignment(vision_vecs, text_vecs_after)
    
    # --- 保存 ---
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_aligned_lora")
    print(f"\nSaving Vision-Aligned LoRA to: {save_path}")
    student.save_pretrained(save_path)
    
    # --- ベンチマークテスト ---
    print("\nRunning Logic Benchmark on Vision-Aligned Model...")
    analogy_tests = [("王様", "男", "女", "女王"), ("東京", "日本", "フランス", "パリ"), ("医者", "病院", "学校", "教師"), ("車", "道路", "線路", "電車"), ("昼", "太陽", "夜", "月")]
    hierarchy_tests = [("動物", "犬", "机"), ("科学", "物理学", "りんご"), ("感情", "怒り", "石"), ("コンピュータ", "ソフトウェア", "自然"), ("宇宙", "銀河", "政治")]
    
    def run_benchmark(model, tok):
        analogy_scores = []
        for a, b, c, d in analogy_tests:
            # get_text_embeddings_batch を使って1つずつ抽出
            vecs = get_text_embeddings_batch(model, tok, [a, b, c, d], batch_size=4)
            v_pred = F.normalize(vecs[0] - vecs[1] + vecs[2], p=2, dim=0)
            score = F.cosine_similarity(v_pred.unsqueeze(0), vecs[3].unsqueeze(0)).item()
            analogy_scores.append(score)
        
        hierarchy_scores = []
        for base, target, outlier in hierarchy_tests:
            vecs = get_text_embeddings_batch(model, tok, [base, target, outlier], batch_size=3)
            sim_tgt = F.cosine_similarity(vecs[0].unsqueeze(0), vecs[1].unsqueeze(0)).item()
            sim_out = F.cosine_similarity(vecs[0].unsqueeze(0), vecs[2].unsqueeze(0)).item()
            score = sim_tgt - sim_out
            hierarchy_scores.append(score)
            
        return np.mean(analogy_scores), np.mean(hierarchy_scores)
        
    avg_analogy, avg_hierarchy = run_benchmark(student, tokenizer)
    
    print("\n==================================================")
    print("      MULTIMODAL TOPOGRAPHY ALIGNMENT RESULTS     ")
    print("==================================================")
    print(f"Topographical Gap Before : {mse_before:.5f}")
    print(f"Topographical Gap After  : {mse_after:.5f}")
    print("-" * 50)
    print(f"Analogy (Logical Reasoning) : {avg_analogy:.4f}")
    print(f"Hierarchy (Concept Depth)   : {avg_hierarchy:.4f}")
    print("==================================================")
    
if __name__ == "__main__":
    step1_multimodal_alignment()
