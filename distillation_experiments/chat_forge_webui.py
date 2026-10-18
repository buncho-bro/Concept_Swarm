"""
【ファイル概要】
chat_forge_webui.py

自律積層概念地形エンジン (Autonomous Stacked Terrain Genesis & AI Layer Sync) 搭載 WebUI。
人間による「文章の指定・アンカー化」を完全に排除し、
「Layer 0 吸収点 ➔ Layer 1 初期配置 ➔ Layer 2 自律成長」により自己組織化された幾何学多層地形と、
実LLM（Qwen2.5等）の内部レイヤー（低次: Layer 6, 中次: Layer 12, 高次: Layer 18）を完全同期。
"""

import os
import sys
import gc
import time
import glob
import math
import torch
import torch.nn.functional as F
import numpy as np
from threading import Thread

# オフライン高速起動（ローカルキャッシュ優先）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

# ==========================================
# グローバル状態
# ==========================================
current_model = None
current_tokenizer = None
current_model_name = "None"
current_controller = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初期配置用 普遍基底アンカー（文章ではなく、初期の種となる普遍単語群）
GENESIS_WORDS = [
    "Entity", "Object", "Number", "Name", "Place", "Time", "Space", "Property", "State", "Definition", "Data", "Fact",
    "Cause", "Effect", "Condition", "Relation", "Difference", "Mechanism", "Process", "Rule", "Change", "Function", "Action",
    "Conclusion", "Principle", "System", "Truth", "Consistency", "Summary", "Invariance", "Resolution", "Optimization", "Insight",
    "Logic", "Structure", "Language", "Thought", "Form", "Order", "Connection", "Balance", "Pattern", "Boundary"
]

def free_vram():
    global current_model, current_tokenizer, current_model_name, current_controller
    if current_controller is not None:
        current_controller.detach()
        current_controller = None
    if current_model is not None:
        del current_model
        current_model = None
    if current_tokenizer is not None:
        del current_tokenizer
        current_tokenizer = None
    current_model_name = "None"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# 自律積層地形ジェネレーター (Autonomous Genesis)
# ==========================================
class AutonomousTerrainGenesis:
    """
    前層の吸収点を次層の初期配置として、自己組織的に多層地形を成長させる
    """
    def __init__(self, model, tokenizer, device, hidden_dim=896):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_dim = hidden_dim

    def extract_word_embeddings(self, words, target_layer):
        vecs = []
        for w in words:
            inp = self.tokenizer(w, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp, output_hidden_states=True)
                h = out.hidden_states[target_layer][0, -1, :].to(torch.float32)
                h = F.normalize(h, p=2, dim=-1)
                vecs.append(h)
        return torch.stack(vecs) # (N, D)

    def find_absorbers(self, terrain_landscape, steps=25, lr=0.06, sigma=0.28, merge_sim=0.90):
        N, D = terrain_landscape.shape
        points = terrain_landscape.clone()

        for step in range(steps):
            cos_sims = torch.matmul(points, terrain_landscape.T)
            dist_sq = 2.0 - 2.0 * cos_sims
            pulls = torch.exp(- dist_sq / (2.0 * (sigma ** 2)))

            weights = pulls / (torch.sum(pulls, dim=1, keepdim=True) + 1e-7)
            target_dirs = torch.matmul(weights, terrain_landscape)

            proj = torch.sum(target_dirs * points, dim=-1, keepdim=True) * points
            tangent = target_dirs - proj
            norm_t = torch.norm(tangent, p=2, dim=-1, keepdim=True)
            tangent = tangent / (norm_t + 1e-7)

            points = F.normalize(points + lr * tangent, p=2, dim=-1)

        unique_absorbers = []
        for p in points:
            if len(unique_absorbers) == 0:
                unique_absorbers.append(p)
            else:
                existing = torch.stack(unique_absorbers)
                sims = torch.matmul(p, existing.T)
                if torch.max(sims).item() < merge_sim:
                    unique_absorbers.append(p)

        return torch.stack(unique_absorbers)

    def build_stacked_terrains(self, l_low=6, l_mid=12, l_high=18):
        # 1. Layer 0 地形生成 (Layer 6 空間)
        X0 = self.extract_word_embeddings(GENESIS_WORDS, target_layer=l_low)
        A0 = self.find_absorbers(X0, steps=25, sigma=0.28, merge_sim=0.90)

        # 2. Layer 1 地形生成: A₀ をそのまま初期配置 X₁ とする
        X1 = F.normalize(A0, p=2, dim=-1)
        A1 = self.find_absorbers(X1, steps=25, sigma=0.32, merge_sim=0.88)

        # 3. Layer 2 地形生成: A₁ をそのまま初期配置 X₂ とする
        X2 = F.normalize(A1, p=2, dim=-1)
        A2 = self.find_absorbers(X2, steps=25, sigma=0.38, merge_sim=0.85)

        return {
            l_low: A0.to(self.device, dtype=torch.float16),
            l_mid: A1.to(self.device, dtype=torch.float16),
            l_high: A2.to(self.device, dtype=torch.float16)
        }

# ==========================================
# 実LLMレイヤーとの完全同期フック & コントローラー
# ==========================================
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

    def set_gamma(self, gamma: float):
        for h in self.hooks:
            h.gamma = gamma

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

# ==========================================
# モデルロード
# ==========================================
def load_model_for_chat(base_id, precision_opt, lora_dir, use_lora):
    global current_model, current_tokenizer, current_model_name, current_controller
    free_vram()
    
    try:
        current_tokenizer = AutoTokenizer.from_pretrained(base_id)
        
        if "4-bit" in precision_opt:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            current_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                quantization_config=bnb_config,
                device_map={"": "cuda:0"}
            )
            precision_label = "4-bit (NF4)"
        else:
            current_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
            precision_label = "FP16 (Direct)"

        current_model.eval()

        num_layers = getattr(current_model.config, "num_hidden_layers", len(current_model.model.layers))
        l_low = int(num_layers * 0.25)
        l_mid = int(num_layers * 0.50)
        l_high = int(num_layers * 0.75)

        # 自律積層地形の生成
        genesis = AutonomousTerrainGenesis(current_model, current_tokenizer, device, hidden_dim=current_model.config.hidden_size)
        terrains = genesis.build_stacked_terrains(l_low=l_low, l_mid=l_mid, l_high=l_high)
        current_controller = SynchronizedLayerController(current_model, terrains)

        if lora_dir == "None":
            use_lora = False
            
        if use_lora:
            if os.path.exists(lora_dir):
                try:
                    current_model = PeftModel.from_pretrained(current_model, lora_dir)
                    current_model_name = f"Distilled Model ({precision_label} + LoRA: {os.path.basename(lora_dir)})"
                except Exception as lora_e:
                    current_model_name = f"Base Model ({precision_label}, LoRA Failed)"
                    return f"❌ ERROR: LoRAの適用に失敗しました。\n詳細: {str(lora_e)}"
            else:
                current_model_name = f"Base Model ({precision_label})"
                return f"⚠️ WARNING: LoRA path '{lora_dir}' not found. Loaded Base Model instead."
        else:
            current_model_name = f"Base Model ({precision_label}: {base_id})"
            
        vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0
        n0, n1, n2 = len(terrains[l_low]), len(terrains[l_mid]), len(terrains[l_high])
        return (
            f"✅ ロード成功: {current_model_name}\n"
            f"VRAM使用量: {vram_mb:.1f} MB\n"
            f"自律積層地形自律生成完了:\n"
            f"  ・Layer {l_low} (要素地形): {n0} 吸収点\n"
            f"  ・Layer {l_mid} (関係地形): {n1} 吸収点 (前層吸収点から自律生成)\n"
            f"  ・Layer {l_high} (統合地形): {n2} 吸収点 (前層吸収点から自律生成)\n"
            f"※ 文章指定・アンカー化なし (完全自己組織化)"
        )
    except Exception as e:
        return f"❌ Error loading Model: {str(e)}"

# ==========================================
# チャットロジック (自律積層同期 + ストリーミング)
# ==========================================
def chat_fn(message, history, use_sync, gamma_val, show_stats):
    global current_model, current_tokenizer, current_model_name, current_controller
    if current_model is None:
        yield "⚠️ 上部の「🚀 Load Model」ボタンを押して、まずモデルをロードしてください。"
        return
        
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, (list, tuple)):
                content = str(content[0]) if len(content) > 0 else ""
            elif not isinstance(content, str):
                content = str(content)
            messages.append({"role": msg.get("role", "user"), "content": content})
        else:
            u_content = str(msg[0]) if not isinstance(msg[0], str) else msg[0]
            a_content = str(msg[1]) if not isinstance(msg[1], str) else msg[1]
            messages.append({"role": "user", "content": u_content})
            messages.append({"role": "assistant", "content": a_content})
            
    if not isinstance(message, str):
        message = str(message)

    if use_sync and current_controller is not None:
        current_controller.set_gamma(float(gamma_val))
        current_controller.set_active(True)
        current_controller.clear_records()
    else:
        if current_controller is not None:
            current_controller.set_active(False)

    messages.append({"role": "user", "content": message})
    text = current_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = current_tokenizer(text, return_tensors="pt").to(device)

    # ストリーミング生成 (TextIteratorStreamer)
    streamer = TextIteratorStreamer(current_tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    thread = Thread(target=current_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
    thread.join()

    # 生成完了後に統計を表示
    if use_sync and show_stats and current_controller is not None and len(current_controller.hooks) >= 3:
        h6 = len(current_controller.hooks[0].records)
        h12 = len(current_controller.hooks[1].records)
        h18 = len(current_controller.hooks[2].records)
        stats_header = (
            f"\n\n---\n"
            f"**[🧠 自律積層地形 同期統計 (Autonomous Layer Sync Telemetry)]**\n"
            f"- Layer {current_controller.l_low} (要素地形): 磁界同調 `{h6}` 回\n"
            f"- Layer {current_controller.l_mid} (関係地形): 磁界同調 `{h12}` 回 (前層吸収点から自律継承)\n"
            f"- Layer {current_controller.l_high} (統合地形): 磁界同調 `{h18}` 回 (前層吸収点から自律継承)\n"
            f"*(※文章アンカーなし・AIの思考深度に応じた自律的磁気浮上同期)*"
        )
        yield generated_text + stats_header

    if current_controller is not None:
        current_controller.set_active(False)

def get_available_loras():
    lora_dirs = [d for d in glob.glob("./lora_*") if os.path.isdir(d)]
    if os.path.exists("./concept_lora_output"):
        lora_dirs.append("./concept_lora_output")
    return ["None"] + lora_dirs

def refresh_lora_list():
    return gr.Dropdown(choices=get_available_loras())

# ==========================================
# Gradio UI 構築
# ==========================================
STUDENT_CHOICES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it"
]

with gr.Blocks(title="Chat Forge - 自律積層同期対話") as demo:
    gr.Markdown("# 💬 Chat Forge — 自律積層概念地形 × AIレイヤー完全同期")
    gr.Markdown(
        "文章指定によるアンカー化（決めつけ）を完全排除。\n"
        "**「Layer 0 吸収点 ➔ Layer 1 初期配置 ➔ Layer 2 自律成長」** により自己組織化した幾何学地形を、"
        "実LLMの内部レイヤー（Layer 6: 要素 ➔ Layer 12: 因果 ➔ Layer 18: 統合）とリアルタイム完全同期させます。"
    )
    
    with gr.Accordion("⚙️ モデル設定 (Model Setup)", open=True):
        with gr.Row():
            base_txt = gr.Dropdown(choices=STUDENT_CHOICES, value="Qwen/Qwen2.5-0.5B-Instruct", label="1. Base Model", allow_custom_value=True, scale=3)
            precision_opt = gr.Dropdown(choices=["FP16 (Direct, Recommended <=1.5B)", "4-bit (BitsAndBytes NF4)"], value="FP16 (Direct, Recommended <=1.5B)", label="2. Precision", scale=2)
            lora_dropdown = gr.Dropdown(choices=get_available_loras(), value="None", label="3. Optional LoRA", allow_custom_value=True, scale=2)
            refresh_lora_btn = gr.Button("🔄", scale=1)
            use_lora_chk = gr.Checkbox(label="Apply LoRA", value=False, scale=1)
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=2)
            
        load_status = gr.Textbox(label="System Status", interactive=False)
        refresh_lora_btn.click(refresh_lora_list, inputs=[], outputs=[lora_dropdown])
        load_btn.click(load_model_for_chat, inputs=[base_txt, precision_opt, lora_dropdown, use_lora_chk], outputs=[load_status])
    
    with gr.Accordion("🏔️ 自律積層同期設定 (Autonomous Stacked Sync Settings)", open=True):
        gr.Markdown(
            "🧠 **原理**: 人間が文章を一切与えず、Layer 0 の吸収点群をそのまま Layer 1 の初期配置とし、さらに Layer 2 へと自律成長させた幾何学地形と同期します。"
        )
        with gr.Row():
            use_sync_chk = gr.Checkbox(
                label="🏔️ 自律積層同期を有効化 (Autonomous Sync Active)",
                value=True,
                scale=2,
                info="チェックを外すと素のベースモデル（無介入）に戻り、思考力・安定性の違いをA/Bテストできます。"
            )
            gamma_slider = gr.Slider(
                minimum=0.00,
                maximum=0.15,
                value=0.06,
                step=0.01,
                label="同期介入強度 γ (Intervention Strength)",
                scale=2,
                info="0.05〜0.08推奨。事実性を保護しつつ思考を自律誘導します。"
            )
            show_stats_chk = gr.Checkbox(
                label="磁界同調統計（テレメトリ）を表示",
                value=True,
                scale=2,
                info="回答末尾に各階層での同調回数を表示します。"
            )

    sample_examples = [
        ["日本の首都はどこですか？簡潔に答えてください。", True, 0.06, True],
        ["オーストラリアの首都はどこですか？シドニーと誤解されやすいですが、正しい都市名を挙げてください。", True, 0.06, True],
        ["3 + 5 × 4 の計算結果を数字のみで答えてください。", True, 0.06, True],
        ["分散データベースにおけるデッドロック検出アルゴリズムについて、Wait-For Graph（待機グラフ）の観点から段階的に解説してください。", True, 0.06, True],
        ["TransformerのSelf-Attentionにおいて、内積を次元数の平方根√d_kで割る数学的理由は何ですか？", True, 0.06, True]
    ]

    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[
            use_sync_chk,
            gamma_slider,
            show_stats_chk
        ],
        examples=sample_examples
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_port=7860, share=False, theme=gr.themes.Soft())
