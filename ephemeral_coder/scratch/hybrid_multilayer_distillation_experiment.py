import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 実LLM (Qwen2.5-0.5B) 隠れ層同期・積層地形ハイブリッド蒸留 (Layer 6, 12, 18 MLP Hooks)
# Multi-Layer Synchronized Distillation via MLP Hidden States Forward Hooks
# ==============================================================================

class LayerTerrainHook:
    """
    特定レイヤーの MLP Hidden States に非侵食的な磁気浮上介入を行う Hook
    """
    def __init__(self, layer_name, attractors, gamma=0.08, sigma=0.25):
        """
        attractors: (N_attractors, 896) の正規化アトラクターテンソル
        gamma: 介入強度 (0.05〜0.12: ベースモデルを壊さない微小結合)
        """
        self.layer_name = layer_name
        self.attractors = attractors # (N, D) on cuda
        self.gamma = gamma
        self.sigma = sigma
        self.active = True

    def __call__(self, module, inputs, output):
        if not self.active or self.attractors is None:
            return output

        # output is Tensor of shape (B, seq_len, 896)
        last_h = output[:, -1, :] # (B, 896)
        h_norm = F.normalize(last_h, p=2, dim=-1)

        # アトラクター群とのコサイン類似度 & 引力計算
        cos_sim = torch.matmul(h_norm, self.attractors.T)
        dist_sq = 2.0 - 2.0 * cos_sim
        pulls = torch.exp(- dist_sq / (2.0 * self.sigma**2))

        # Smooth-Max で引力を計算
        weights = F.softmax(pulls * 8.0, dim=-1)
        target_dir = torch.matmul(weights, self.attractors)
        target_dir = F.normalize(target_dir, p=2, dim=-1)

        # 接平面射影
        tangent_nudge = target_dir - torch.sum(target_dir * h_norm, dim=-1, keepdim=True) * h_norm
        tangent_nudge = F.normalize(tangent_nudge, p=2, dim=-1)

        # 微小な磁気浮上補正を加える
        h_magnitude = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        nudged_last_h = last_h + self.gamma * h_magnitude * tangent_nudge

        # テンソル置換
        new_output = output.clone()
        new_output[:, -1, :] = nudged_last_h
        return new_output


class MultiLayerSynchronizedController:
    """
    3つの重要レイヤー (Layer 6, 12, 18) の MLP 出力に積層地形をフックし、
    LLMの自己回帰レイヤー伝播と幾何学的思考滑走を完全同期させるコントローラー
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.hook_handles = []

    def attach_layer_terrains(self, layer_specs):
        """
        layer_specs: dict {layer_idx: attractors_tensor}
        """
        self.detach()
        for l_idx, attractors in layer_specs.items():
            hook = LayerTerrainHook(
                layer_name=f"Layer_{l_idx}_MLP",
                attractors=attractors,
                gamma=0.08,
                sigma=0.25
            )
            # DecoderLayer の MLP 出力にフックをアタッチ
            handle = self.model.model.layers[l_idx].mlp.register_forward_hook(hook)
            self.hooks.append(hook)
            self.hook_handles.append(handle)

    def set_active(self, active: bool):
        for h in self.hooks:
            h.active = active

    def detach(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()


def run_multilayer_experiment():
    print("==================================================================")
    print(" 実LLM隠れ層同期・積層地形ハイブリッド蒸留 (Qwen2.5-0.5B)")
    print(" Multi-Layer Synchronized Distillation: Layer 6, 12, 18 MLP Hooks")
    print("==================================================================")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[1] モデルロード中: {model_id} (CUDA 16-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    device = model.device
    hidden_dim = model.config.hidden_size # 896
    print(f"    モデル準備完了 (VRAM: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB, Hidden Dim: {hidden_dim})")

    # 2. 各階層 (Layer 6, 12, 18) の思考アトラクターを構築
    # 課題: 「複雑な非同期分散システムにおけるデッドロック回避とトランザクション隔離性」
    print("\n[2] 3階層の思考アトラクター（各896次元）を教師プロンプトから生成...")
    def get_layer_hidden_attractors(prompt_list, target_layer_idx):
        attractor_vecs = []
        for p in prompt_list:
            inp = tokenizer(p, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
                # target_layer_idx の最終トークン hidden state
                h = out.hidden_states[target_layer_idx][0, -1, :] # (896,)
                h = F.normalize(h, p=2, dim=-1)
                attractor_vecs.append(h)
        return torch.stack(attractor_vecs)

    l6_prompts = [
        "Primitive locking primitives: mutex, spinlock, compare-and-swap atomic operations",
        "Deadlock detection algorithms using wait-for graphs and cycle detection"
    ]
    l12_prompts = [
        "Two-phase locking protocol (2PL) ensuring strict serializable schedules",
        "Multi-version concurrency control (MVCC) snapshot isolation with timestamp ordering"
    ]
    l18_prompts = [
        "Strict serializability and linearizability in high-throughput distributed architectures",
        "Deadlock-free non-blocking consensus with bounded latency guarantees"
    ]

    attractors_l6 = get_layer_hidden_attractors(l6_prompts, target_layer_idx=6)
    attractors_l12 = get_layer_hidden_attractors(l12_prompts, target_layer_idx=12)
    attractors_l18 = get_layer_hidden_attractors(l18_prompts, target_layer_idx=18)

    print("    ・Layer 6  アトラクター (低次構文・プリミティブ): shape", attractors_l6.shape)
    print("    ・Layer 12 アトラクター (中次関係・MVCC/2PL): shape", attractors_l12.shape)
    print("    ・Layer 18 アトラクター (高次統合・分散直列化可能性): shape", attractors_l18.shape)

    # コントローラーのセットアップ
    controller = MultiLayerSynchronizedController(model)
    layer_specs = {
        6: attractors_l6,
        12: attractors_l12,
        18: attractors_l18
    }
    controller.attach_layer_terrains(layer_specs)

    # 3. テストプロンプト (高度な複雑思考を要求する問題)
    complex_prompt = "Explain in detail how a distributed database avoids deadlocks while maintaining serializable isolation during concurrent transactions:"
    print(f"\n[3] 複雑推論テストプロンプト:\n    \"{complex_prompt}\"")

    inputs = tokenizer(complex_prompt, return_tensors="pt").to(device)

    # --- 条件 A: 素の生徒モデル (Base Qwen-0.5B, 介入なし) ---
    print("\n--- [A. 素の生徒モデル (Base Qwen-0.5B, 介入なし)] ---")
    controller.set_active(False)
    t0 = time.time()
    with torch.no_grad():
        out_base = model.generate(
            **inputs,
            max_new_tokens=130,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    txt_base = tokenizer.decode(out_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    time_base = (time.time() - t0) * 1000.0
    print(f"(生成時間: {time_base:.1f}ms)\n{txt_base}\n")

    # --- 条件 B: ★本手法: 隠れ層同期・積層地形ハイブリッド (Layer 6, 12, 18 同期介入) ---
    print("--- [B. ★本手法: 隠れ層同期・積層地形ハイブリッド (Layer 6, 12, 18 同期介入)] ---")
    controller.set_active(True)
    t0 = time.time()
    with torch.no_grad():
        out_hybrid = model.generate(
            **inputs,
            max_new_tokens=130,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    txt_hybrid = tokenizer.decode(out_hybrid[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    time_hybrid = (time.time() - t0) * 1000.0
    print(f"(生成時間: {time_hybrid:.1f}ms)\n{txt_hybrid}\n")

    controller.detach()
    print("==================================================================")
    print(" 隠れ層同期・積層ハイブリッド蒸留実験が正常に完了しました！")
    print("==================================================================")

if __name__ == '__main__':
    run_multilayer_experiment()
