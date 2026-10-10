"""
【ファイル概要】
chimera_injector.py

複数の異なるドメインで蒸留されたLoRAを動的・局所的に結合（キメラ化）し、入力文脈に応じて重みをブレンドする実験スクリプト。
"""

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import gc

# グローバル状態
current_model_id = None
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_id):
    global current_model_id, model, tokenizer
    if current_model_id == model_id and model is not None:
        return f"Model {model_id} is already loaded."
    
    if model is not None:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
    try:
        yield f"Loading {model_id}..."
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        current_model_id = model_id
        yield f"Successfully loaded {model_id}!"
    except Exception as e:
        yield f"Error loading model: {str(e)}"

def extract_concept_vector(concept_text, target_layer_idx):
    """ 指定されたテキストから、特定の層における概念ベクトル(方向)を抽出する """
    inputs = tokenizer(concept_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of length (num_layers + 1)
        # 最初の層(0)はEmbedding、それ以降が各Transformer層の出力
        layer_hidden = outputs.hidden_states[target_layer_idx + 1] # [batch, seq_len, hidden_dim]
        # 時間方向(seq_len)で平均をとり、単一の方向ベクトルにする
        concept_vector = layer_hidden.mean(dim=1, keepdim=True) # [batch, 1, hidden_dim]
        # 正規化（大きさを揃えるため）
        concept_vector = F.normalize(concept_vector, p=2, dim=-1)
    return concept_vector

def generate_with_chimera(prompt, concept_text, target_layer_idx, injection_strength, max_new_tokens):
    global model, tokenizer
    if model is None:
        yield "Error: Please load a model first."
        return

    # 1. 概念ベクトルの抽出
    if concept_text.strip() and injection_strength > 0:
        yield "Extracting concept vector...\n"
        concept_vec = extract_concept_vector(concept_text, int(target_layer_idx))
    else:
        concept_vec = None

    # 2. フックの設定
    hook_handle = None
    if concept_vec is not None:
        # モデルアーキテクチャに応じて対象層を取得 (Llama/Qwen系を想定)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            target_module = model.model.layers[int(target_layer_idx)]
        else:
            yield "Error: Unsupported model architecture for hook injection."
            return

        def chimera_hook(module, args, kwargs, output):
            # output は通常 tuple (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Injection（つぎはぎ）: 現在のベクトルに概念ベクトルを加算
            # 抽出したベクトルを現在の隠れ状態のスケールに合わせてスケーリング
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
            injected_hidden = hidden_states + (concept_vec * injection_strength * current_norm.mean())
            
            if isinstance(output, tuple):
                return (injected_hidden,) + output[1:]
            return injected_hidden
            
        # forward_hook を登録
        hook_handle = target_module.register_forward_hook(chimera_hook, with_kwargs=True)

    # 3. 生成プロセス
    yield "Generating text with Chimera Injection...\n---\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        streamer=streamer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
        
    # フックの解除
    if hook_handle is not None:
        hook_handle.remove()

# --- UI構築 ---
with gr.Blocks(title="Chimera Concept Injector") as demo:
    gr.Markdown("# 🧪 Chimera Concept Injector (Activation Steering)")
    gr.Markdown("プロンプト（言葉）ではなく、AIの脳内（ベクトル空間）に直接別の概念を『注射（つぎはぎ）』し、強引に未知のアルゴリズムや思考を創出させる実験的プロトタイプです。")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"], value="Qwen/Qwen2.5-1.5B-Instruct", label="Base Model")
        load_btn = gr.Button("Load Model", variant="secondary")
    load_status = gr.Textbox(label="Load Status", interactive=False)
    load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])
    
    with gr.Row():
        with gr.Column(scale=1):
            concept_input = gr.Textbox(label="Injection Concept (注射する概念)", value="Centrifugal force, gravity, separation, filtering", lines=2)
            layer_slider = gr.Slider(minimum=0, maximum=40, step=1, value=14, label="Target Layer (介入する層の深さ)", info="浅い層=文法崩壊リスク大。中間層=概念の転換。深い層=出力の微修正。")
            strength_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Injection Strength (α)", info="注射の強さ。高すぎると文字化け・意味不明になります。")
            tokens_slider = gr.Slider(minimum=100, maximum=2000, step=100, value=500, label="Max New Tokens")
            
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(label="Main Prompt (ベースとなる課題)", value="Write a Python script for a completely new, never-before-seen sorting algorithm. Do not use standard quicksort or mergesort logic. Think step by step.", lines=4)
            generate_btn = gr.Button("🚀 Generate Chimera Code", variant="primary")
            output_box = gr.Textbox(label="Output", lines=20)
            
    generate_btn.click(
        generate_with_chimera,
        inputs=[prompt_input, concept_input, layer_slider, strength_slider, tokens_slider],
        outputs=[output_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
