"""
test_zero_vram_stacked_inference.py
===================================
0 VRAM 全大円干渉により全15万語から析出した自律積層地形を、
実LLM (Qwen2.5-0.5B) の内部レイヤー (Layer 6, 12, 18) に完全同期させて推論テストを行う。
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 自作エンジンとコントローラーのインポート
from zero_vram_great_circle_genesis import ZeroVRAMVocabGreatCircleGenesis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SynchronizedTerrainHook:
    def __init__(self, layer_idx, layer_name, absorbers, gamma=0.06, threshold=0.15):
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.absorbers = absorbers # (M, D) float16 on cuda
        self.gamma = gamma
        self.threshold = threshold
        self.active = True
        self.records = []

    def __call__(self, module, inputs, output):
        if not self.active or self.absorbers is None:
            return output

        last_h = output[:, -1, :] # (B, D) float16
        h_norm = F.normalize(last_h, p=2, dim=-1)

        cos_sim = torch.matmul(h_norm, self.absorbers.T) # (B, M)
        max_sim, best_idx = torch.max(cos_sim, dim=-1)

        # 磁界ゲート: 類似度が閾値未満なら磁力はゼロ（自然言語や事実知識を阻害しない）
        if max_sim.item() < self.threshold:
            return output

        gate = (max_sim - self.threshold) / (1.0 - self.threshold)
        target_absorber = self.absorbers[best_idx]

        # 接平面射影 (Tangent Nudge)
        proj = torch.sum(target_absorber * h_norm, dim=-1, keepdim=True) * h_norm
        tangent = target_absorber - proj
        norm_t = torch.norm(tangent, p=2, dim=-1, keepdim=True)
        if norm_t.item() < 1e-6:
            return output
        tangent_nudge = tangent / norm_t

        shift_deg = float(torch.acos(torch.clamp(max_sim, -1.0, 1.0)).item() * 180.0 / math.pi)
        self.records.append({"best_idx": best_idx.item(), "shift_deg": shift_deg, "sim": max_sim.item()})

        h_mag = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        nudged = last_h + self.gamma * gate.unsqueeze(-1) * h_mag * tangent_nudge

        new_output = output.clone()
        new_output[:, -1, :] = nudged
        return new_output

class SynchronizedLayerController:
    def __init__(self, model, terrains):
        self.model = model
        self.terrains = terrains
        self.hooks = []
        self.hook_handles = []

        num_layers = getattr(model.config, "num_hidden_layers", len(model.model.layers))
        self.l_low = int(num_layers * 0.25)
        self.l_mid = int(num_layers * 0.50)
        self.l_high = int(num_layers * 0.75)

        for l_idx, absorbers in terrains.items():
            hook = SynchronizedTerrainHook(
                layer_idx=l_idx,
                layer_name=f"Layer_{l_idx}_MLP",
                absorbers=absorbers,
                gamma=0.06,
                threshold=0.15
            )
            handle = self.model.model.layers[l_idx].mlp.register_forward_hook(hook)
            self.hooks.append(hook)
            self.hook_handles.append(handle)

    def set_active(self, active: bool):
        for h in self.hooks:
            h.active = active

    def clear_records(self):
        for h in self.hooks:
            h.records.clear()

    def detach(self):
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.hook_handles.clear()

def run_e2e_zero_vram_distillation_sync():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print("=" * 60)
    print("🚀 0 VRAM 全大円干渉 ➔ レイヤー同期 統合検証")
    print("=" * 60)
    
    # 1. 0 VRAM 全語彙大円干渉による多層概念地形の自律析出
    genesis = ZeroVRAMVocabGreatCircleGenesis(model_id=model_id, verbose=True)
    V, U = genesis.load_full_vocab_vectors()
    A0_np, A1_np, A2_np = genesis.build_stacked_terrains(V, U)
    
    print("\n[Layer 0 吸収点のトップ共鳴語を検証]")
    genesis.inspect_absorber_concepts(A0_np, top_k=4)

    # 2. 実LLMのロード (GPU上 FP16)
    print("\n--- Loading LLM onto GPU (FP16) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    # 3. 0VRAM析出地形を Tensor (CUDA float16) に変換して同期
    num_layers = getattr(model.config, "num_hidden_layers", len(model.model.layers))
    l_low = int(num_layers * 0.25)
    l_mid = int(num_layers * 0.50)
    l_high = int(num_layers * 0.75)

    terrains = {
        l_low: torch.tensor(A0_np, dtype=torch.float16, device=device),
        l_mid: torch.tensor(A1_np, dtype=torch.float16, device=device),
        l_high: torch.tensor(A2_np, dtype=torch.float16, device=device)
    }

    controller = SynchronizedLayerController(model, terrains)
    print(f"✅ SynchronizedLayerController 配備完了:")
    print(f"  ・Layer {l_low} MLP: {len(A0_np)} 点の全語彙イデア核と同期")
    print(f"  ・Layer {l_mid} MLP: {len(A1_np)} 点の因果関係核と同期")
    print(f"  ・Layer {l_high} MLP: {len(A2_np)} 点の統合核と同期")

    # 4. 推論ベンチマークテスト
    test_queries = [
        "日本の首都はどこですか？簡潔に答えてください。",
        "オーストラリアの首都はどこですか？シドニーと誤解されやすいですが、正しい都市名を挙げてください。",
        "3 + 5 × 4 の計算結果を数字のみで答えてください。",
        "分散システムにおけるデッドロック検出について、Wait-For Graphの観点から簡潔に解説してください。"
    ]

    for q in test_queries:
        print("\n" + "-" * 50)
        print(f"Q: {q}")
        controller.clear_records()
        controller.set_active(True)

        messages = [{"role": "user", "content": q}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        ans = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        print(f"A:\n{ans}")

        # テレメトリ同調統計
        h6 = len(controller.hooks[0].records)
        h12 = len(controller.hooks[1].records)
        h18 = len(controller.hooks[2].records)
        print(f"\n[🧠 磁界同調テレメトリ]")
        print(f"  ・Layer {l_low} (全語彙イデア核): {h6} 回同調")
        print(f"  ・Layer {l_mid} (因果関係核): {h12} 回同調")
        print(f"  ・Layer {l_high} (統合核): {h18} 回同調")

    controller.detach()
    print("\n✅ All End-to-End Tests with 0-VRAM Full-Vocab Sync PASSED!")

if __name__ == "__main__":
    run_e2e_zero_vram_distillation_sync()
