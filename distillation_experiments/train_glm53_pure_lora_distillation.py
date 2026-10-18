"""
train_glm53_pure_lora_distillation.py
=====================================
GLM-5.3 の全 154,880 語から自律析出した 54 個の純粋イデア星座を、
Qwen2.5-Coder-3B-Instruct の LoRA に焼き付ける真のトポロジー蒸留。

【核心】
・過去の 25 単語辞書アンカーを完全撤廃。
・GLM-5.3 の全語彙大円干渉によって自律析出した 54 個の真のイデア核群の
  相対幾何学トポロジー (54 x 54 無次元距離行列) を教師ターゲットとする。
・Qwen2.5-Coder-3B に LoRA をアタッチし、Topological Scale-Invariant Loss で
  GLM-5.3 の知性トポロジーを完全同相写像として学習・焼き付け。
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 54個のGLM自律イデア核に対応する重心シードトークン群 (辞書アンカーではなく自律析出語)
PURE_GLM_IDEA_SEEDS = [
    "The", "2019", "_ComCallableWrapper", '""', "Corporate", "int", "git", "PostalCodes", "island",
    "));", "Initial", "cmd", "status", "View", "constraints", "year", "system", "tools", "split",
    "Api", "Virtual", "center", "entities", "actions", "internal", "DOMContentLoaded", "content",
    "agents", "Persistent", "channel", "stream", "platform", "Definition", "Results", "()",
    "convert", "document", "element", "Return", "goal", "Digital", "Automatic", "updates",
    "{}:;", "available", "leetcode", "collection", "service", "entity", "Professional",
    "resolved", "completed", "levels", "Config"
]

def run_pure_lora_training():
    print("=" * 70)
    print("🌌 GLM-5.3 ➔ Qwen2.5-Coder-3B 真の 0-VRAM トポロジー LoRA 蒸留")
    print("   ※ 辞書アンカー 100% 排除 / 54個の全語彙自律イデア星座を焼き付け")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. GLM-5.3 の自律多層概念地形マスターの読み込み
    terrain_file = "glm5.3_zero_vram_pure_concept_terrains.pt"
    if not os.path.exists(terrain_file):
        raise FileNotFoundError(f"Terrain file {terrain_file} not found. Run distill_glm53_zero_vram_great_circle.py first.")

    print(f"\n[Step 1] Loading GLM-5.3 pure concept terrain from {terrain_file}...")
    checkpoint = torch.load(terrain_file, map_location="cpu", weights_only=True)
    A0 = checkpoint["Layer_0"] # (54, 6144)
    print(f"✅ Loaded Layer 0 pure absorbers: shape={A0.shape} (54 pure ideas, 6,144 dims)")

    # 2. 教師ターゲットトポロジー (無次元の相対距離行列) の計算
    A0_norm = F.normalize(A0, p=2, dim=-1)
    T_dist_matrix = torch.cdist(A0_norm, A0_norm, p=2) # [54, 54]
    k_T = torch.mean(T_dist_matrix)
    normalized_T_dist = (T_dist_matrix / (k_T + 1e-8)).to(device).half()
    print(f"✅ Target Topology Matrix calculated: shape={normalized_T_dist.shape}, mean_dist={k_T.item():.4f}")

    # 3. 生徒モデル (Qwen2.5-Coder-3B-Instruct) のロード (FP16)
    student_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
    print(f"\n[Step 2] Loading Student Model ({student_id}) in FP16 into GPU...")
    t_load = time.perf_counter()
    student_tok = AutoTokenizer.from_pretrained(student_id)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_id,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    print(f"✅ Student Model Loaded in {time.perf_counter()-t_load:.2f}s! VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

    # 4. LoRA アタッチ
    print("\n[Step 3] Attaching PEFT LoRA modules (r=16, alpha=32)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(student_model, lora_config)
    student.print_trainable_parameters()

    # 5. 初期状態（学習前）の生徒空間の幾何学アンカーを取得
    student.eval()
    inputs = student_tok(PURE_GLM_IDEA_SEEDS, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        init_outputs = student(**inputs, output_hidden_states=True)
        # 最終隠れ層の最終有効トークン位置
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(len(PURE_GLM_IDEA_SEEDS), device=device)
        initial_vecs = F.normalize(init_outputs.hidden_states[-1][batch_idx, seq_lens, :].to(torch.float32), p=2, dim=-1)

    init_dist_matrix = torch.cdist(initial_vecs, initial_vecs, p=2)
    k_init = torch.mean(init_dist_matrix)
    normalized_init_dist = init_dist_matrix / (k_init + 1e-8)
    initial_loss = F.mse_loss(normalized_init_dist.half(), normalized_T_dist).item()
    print(f"📊 Initial Pre-Training Topology Error to GLM-5.3: {initial_loss:.5f}")

    # 6. トポロジー蒸留 学習ループ
    student.train()
    epochs = 120
    lr = 2e-4
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    output_dir = "./lora_glm5.3_pure_to_Qwen2.5_Coder_3B"
    print(f"\n[Step 4] Starting Topological Distillation Training ({epochs} epochs)...")
    t_train = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = student(**inputs, output_hidden_states=True)
        current_h = outputs.hidden_states[-1][batch_idx, seq_lens, :]
        current_vecs_norm = F.normalize(current_h.to(torch.float32), p=2, dim=-1)
        
        # 生徒の距離行列 [54, 54]
        S_dist_matrix = torch.cdist(current_vecs_norm, current_vecs_norm, p=2)
        k_S = torch.mean(S_dist_matrix)
        normalized_S_dist = (S_dist_matrix / (k_S + 1e-8)).half()
        
        # 形状の同相写像誤差 (GLM-5.3 のイデア星座との距離)
        topology_loss = F.mse_loss(normalized_S_dist, normalized_T_dist)
        
        # 星座の回転・崩壊防止アンカーペナルティ (10%の力で繋ぎ止め)
        anchor_penalty = F.mse_loss(current_vecs_norm, initial_vecs)
        
        total_loss = topology_loss + 0.1 * anchor_penalty
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 15 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"  Epoch {epoch+1:03d}/{epochs} | Total Loss: {total_loss.item():.5f} (Topo: {topology_loss.item():.5f}, Anchor: {anchor_penalty.item():.5f})")

    total_train_time = time.perf_counter() - t_train
    print(f"\n✅ Distillation Training Complete in {total_train_time:.2f}s!")

    # 7. 学習後の検証
    student.eval()
    with torch.no_grad():
        final_outputs = student(**inputs, output_hidden_states=True)
        final_vecs = F.normalize(final_outputs.hidden_states[-1][batch_idx, seq_lens, :].to(torch.float32), p=2, dim=-1)
        final_dist = torch.cdist(final_vecs, final_vecs, p=2)
        k_final = torch.mean(final_dist)
        normalized_final_dist = (final_dist / (k_final + 1e-8)).half()
        final_loss = F.mse_loss(normalized_final_dist, normalized_T_dist).item()

    improvement = (initial_loss - final_loss) / initial_loss * 100
    print(f"📈 Pre-training Loss: {initial_loss:.5f} ➔ Post-training Loss: {final_loss:.5f} (Alignment Improvement: +{improvement:.1f}%)")

    # 8. LoRA の保存
    print(f"\n[Step 5] Saving Pure PEFT LoRA Adapter to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    student.save_pretrained(output_dir)
    student_tok.save_pretrained(output_dir)
    print("✅ Successfully saved adapter_config.json and adapter_model.safetensors!")
    print(f"\n🎉 GLM-5.3 Pure Full-Vocab LoRA Distillation to Qwen2.5-Coder-3B COMPLETED!")

if __name__ == "__main__":
    run_pure_lora_training()
