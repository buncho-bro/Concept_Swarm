"""
【ファイル概要】
auto_shallow_distiller.py

浅層モデルへの蒸留を自動化し、複数ドメインの概念地形をバッチ処理で次々と学習させるパイプライン。
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import numpy as np

# ==============================================================================
# Auto Shallow Concept Distiller (True LoRA Topology Training)
# 
# 1. GLMのEmbeddingファイルのみを部分ダウンロードし、概念の「絶対座標」を抽出。
# 2. Qwen2.5(Student)にLoRAをアタッチ。
# 3. QwenがGLMと同じ概念座標(Target Topology)に引力を持つよう、
#    optim.AdamでLoRAウェイトを直接ファインチューニング(焼き付け)。
# 4. 正規のPEFTフォーマット(adapter_model.safetensors)として保存。
# ==============================================================================

# --- CONFIGURATION ---
TEACHER_REPO = "zai-org/GLM-5.3"
# GGUFでの学習は不可能なため、元の3Bモデルを4bit量子化でロードして同等のVRAM節約を実現します
STUDENT_REPO = "Qwen/Qwen2.5-Coder-3B-Instruct"

# 蒸留する概念のアンカー（コーディング特化）
CONCEPT_WORDS = [
    "Function", "Variable", "Loop", "Recursion", "Asynchronous", "Concurrency", 
    "Database", "Query", "Network", "Protocol", "Security", "Encryption", 
    "Thread", "Process", "Lock", "Mutex", "API", "Framework", "Deployment", 
    "Virtualization", "Cloud", "Server", "Cluster", "Scalability", "Optimization"
]

TEACHER_DIR = os.path.abspath(f"./models_shallow/{TEACHER_REPO.split('/')[-1]}_shallow")
OUTPUT_LORA_DIR = "./lora_glm5.3_to_Qwen2.5_Coder_3B_Topological"
EPOCHS = 150
LEARNING_RATE = 2e-4
# ---------------------

import sys
import gc

def step1_2_extract_great_circle_targets():
    print(f"\n[STEP 1 & 2] Extracting Great Circle Concept Cores from {TEACHER_REPO}...")
    from shallow_extractor import ShallowConceptExtractor, InterferenceConfig
    
    # Initialize the new extractor (this handles HTTP Range downloads automatically)
    print("Initializing ShallowConceptExtractor (CPU/0-VRAM)...")
    ex = ShallowConceptExtractor(model_id=TEACHER_REPO, verbose=True)
    
    target_vectors = []
    valid_words = []
    
    print("Extracting Concept Cores using Great Circle Interference...")
    for i, word in enumerate(CONCEPT_WORDS):
        # We extract exactly 1 core (the primary core) for each concept word
        res = ex.extract(word, config=InterferenceConfig(k_max=1))
        target_vectors.append(res.primary)
        valid_words.append(word)
        if (i + 1) % 5 == 0 or (i + 1) == len(CONCEPT_WORDS):
            print(f"  -> Processed {i + 1}/{len(CONCEPT_WORDS)} concepts (Last: {word} - Resonance: {res.cores[0].resonance:.2%})")
            
    # Clean up extractor memory
    del ex
    gc.collect()
            
    # [N, Hidden_Dim] matrix
    target_matrix = torch.tensor(np.array(target_vectors), dtype=torch.float32)
    
    # Create the N x N relative topology (distance/similarity matrix)
    target_topology = torch.matmul(target_matrix, target_matrix.T) # [N, N]
    
    print(f"✅ Extracted Target Topology (N x N shape): {target_topology.shape}")
    return valid_words, target_topology

def step3_train_and_save_lora(valid_words, target_topology):
    print(f"\n[STEP 3] Training Student LoRA ({STUDENT_REPO}) in 16bit Full Precision...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 16bitフル精度での生徒モデルロード（12GB VRAMなら余裕で収まります）
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_REPO)
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_REPO, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # 2. LoRAのアタッチ
    print("Attaching LoRA modules...")
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    student = get_peft_model(student_model, lora_config)
    student.print_trainable_parameters()
    
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    target_topology = target_topology.to(device).half()
    
    print(f"Starting Concept Topology Distillation (Epochs: {EPOCHS})...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Qwen(LoRA)の出力を取得
        inputs = student_tokenizer(valid_words, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = student(**inputs, output_hidden_states=True)
        
        # 最終隠れ層から「概念ベクトル」を取得
        current_vectors = []
        for i in range(len(valid_words)):
            # パディングを無視してシーケンス長方向の平均をとる
            vec = outputs.hidden_states[-1][i].mean(dim=0)
            current_vectors.append(vec)
            
        current_matrix = torch.stack(current_vectors) # [N, 1536 or 3072]
        
        # 🌟 Qwen空間内での N x N の「相対距離行列」を計算
        current_matrix = F.normalize(current_matrix, p=2, dim=1)
        current_topology = torch.matmul(current_matrix, current_matrix.T) # [N, N]
        
        # N x N の行列同士のMSEロス（次元の壁を無視して形状だけを合わせる）
        loss = F.mse_loss(current_topology, target_topology)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Topology Loss: {loss.item():.4f}")
            
    print(f"\n[STEP 4] Saving Regular PEFT LoRA to: {OUTPUT_LORA_DIR}")
    os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)
    student.save_pretrained(OUTPUT_LORA_DIR)
    print("✅ Successfully saved adapter_config.json and adapter_model.safetensors!")
    
if __name__ == "__main__":
    print("==================================================")
    print(" ⚗️ AUTO SHALLOW CONCEPT DISTILLER (TRUE LORA) ")
    print("==================================================")
    try:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        valid_words, target_topology = step1_2_extract_great_circle_targets()
        step3_train_and_save_lora(valid_words, target_topology)
        print("\nAll done. The LoRA is now ready to be loaded by ChatForge or Project Ripple.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
