"""
【ファイル概要】
resonant_vector_coder.py

複数の正解コードベクトルを「共鳴」させ、未知のタスクに対してその共鳴周波数（ベクトルの合成波）からコードを生成させる実験。
"""

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import gc

current_model_id = None
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOMAINS = [
    "Fluid Dynamics (流体力学)", "Quantum Mechanics (量子力学)", "Biological Evolution (生物の進化)",
    "Economic Market Dynamics (経済の市場原理)", "Musical Harmony (音楽の和音)",
    "Celestial Mechanics (天体力学)", "Thermodynamics (熱力学)",
    "Geological Sedimentation (地質堆積)", "Neural Pruning (神経剪定)",
    "Chemical Crystallization (化学結晶化)"
]

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

def extract_base_vector(text, target_layer):
    """ 指定テキストの最後のトークンの隠れ状態ベクトルを抽出 """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 最初の要素がEmbeddingなので +1
        layer_hidden = outputs.hidden_states[target_layer + 1] 
        # 最後のトークンのベクトル
        vec = layer_hidden[:, -1:, :] 
    return vec

def evaluate_resonance(prompt, meta_vec, target_layer, alpha):
    """ メタファーベクトルを注入した時の、解決方向(本来のベクトル)との共鳴度を測る """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 注入せずに本来の解決方向 V_goal を取得
    with torch.no_grad():
        base_outputs = model(**inputs, output_hidden_states=True)
        V_goal = base_outputs.hidden_states[target_layer + 1][:, -1:, :]
    
    # 注入状態でのベクトル V_infused を取得
    V_infused_cache = []
    
    def eval_hook(module, args, kwargs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # プロンプト(seq_len > 1)の時のみ注入
        if hidden_states.shape[1] > 1:
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
            norm_meta = F.normalize(meta_vec, p=2, dim=-1)
            infused = hidden_states + (norm_meta * alpha * current_norm.mean())
            V_infused_cache.append(infused[:, -1:, :].detach())
            if isinstance(output, tuple):
                return (infused,) + output[1:]
            return infused
        return output

    target_module = model.model.layers[target_layer]
    handle = target_module.register_forward_hook(eval_hook, with_kwargs=True)
    
    with torch.no_grad():
        model(**inputs)
        
    handle.remove()
    
    if not V_infused_cache:
        return 0, 0, 0
        
    V_infused = V_infused_cache[0]
    
    # 共鳴スコアの計算
    cos_sim = F.cosine_similarity(V_infused, V_goal, dim=-1).item()
    goal_norm = V_goal.norm(dim=-1).item()
    infused_norm = V_infused.norm(dim=-1).item()
    
    # 力の増分
    force_increase = infused_norm / (goal_norm + 1e-8)
    
    # 解決の方向(cos_sim)を保ったまま、力が強まった(force_increase)ものを評価
    # 類似度が低すぎる(0.5未満など)とタスク崩壊とみなす
    if cos_sim < 0.5:
        resonance_score = 0
    else:
        resonance_score = cos_sim * force_increase
        
    return resonance_score, cos_sim, force_increase

def run_resonant_coder(target_task, target_layer, alpha):
    global model, tokenizer
    if model is None:
        yield "Error: Load model first.", ""
        return
        
    log = "### 🌊 Phase 1: Measuring Baseline (Goal Vector)\n"
    yield log, ""
    
    target_layer = int(target_layer)
    alpha = float(alpha)
    prompt = f"Write a completely new and highly efficient Python code for '{target_task}'. Output the python code inside ```python ``` blocks. Do not explain, just code."
    
    log += "Extracting the core vector for the target task...\n\n"
    yield log, ""
    
    # Phase 2: ノイズの爆撃と共鳴の観測
    log += "### ⚡ Phase 2: Vector Sweeping & Resonance Measurement\n"
    log += "Bombarding the AI's hidden states with metaphor vectors...\n\n"
    yield log, ""
    
    best_score = -1
    best_domain = None
    best_meta_vec = None
    
    results = []
    
    for domain in DOMAINS:
        # メタファーベクトルの抽出 (意味合いを濃くするため少し詳しいテキストにする)
        meta_text = f"The physical principles and abstract concepts of {domain}."
        meta_vec = extract_base_vector(meta_text, target_layer)
        
        # 注入と共鳴測定
        score, cos_sim, f_inc = evaluate_resonance(prompt, meta_vec, target_layer, alpha)
        results.append((domain, score, cos_sim, f_inc, meta_vec))
        
        log += f"Domain: {domain}\n"
        log += f"  -> Direction (Sim): {cos_sim:.4f}, Force (Norm ratio): {f_inc:.4f} => **Resonance: {score:.4f}**\n"
        
        if score > best_score:
            best_score = score
            best_domain = domain
            best_meta_vec = meta_vec
            
        yield log, ""
        
    if best_meta_vec is None or best_score == 0:
        yield log + "\n❌ All vectors caused the AI to escape (Task Collapse). Try adjusting layer or alpha.", ""
        return
        
    log += f"\n🏆 **Maximum Resonance Achieved!**\n"
    log += f"Selected Wave: **{best_domain}** (Score: {best_score:.4f})\n"
    log += "The force towards the solution has been explosively amplified.\n\n"
    yield log, "Crystallizing the resonant wave into code..."
    
    # Phase 3: 結晶化 (生成)
    log += "### 💎 Phase 3: Crystallization (Decoding)\n"
    yield log, ""
    
    target_module = model.model.layers[target_layer]
    
    def chimera_hook(module, args, kwargs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # プロンプト処理時のみ注入 (AIの純粋な思考を壊さないため)
        if hidden_states.shape[1] > 1:
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
            norm_meta = F.normalize(best_meta_vec, p=2, dim=-1)
            infused = hidden_states + (norm_meta * alpha * current_norm.mean())
            if isinstance(output, tuple):
                return (infused,) + output[1:]
            return infused
        return output
        
    hook_handle = target_module.register_forward_hook(chimera_hook, with_kwargs=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=1000,
        streamer=streamer,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    final_code = ""
    for new_text in streamer:
        final_code += new_text
        yield log, final_code
        
    hook_handle.remove()
    
    log += "\n✅ **Crystallization Complete.**"
    yield log, final_code

with gr.Blocks(title="Resonant Vector Coder") as demo:
    gr.Markdown("# 🌊 Resonant Vector Coder (物理ベース飛躍的思考)")
    gr.Markdown("プロンプトという言葉の強制を捨て、AIの脳内(高次元空間)に直接異分野の波(ベクトル)を浴びせます。本来の目的方向への引力を保ちつつ、最もエネルギーが爆発(共鳴)した波だけを数学的に検知し、未知のコードへと結晶化させます。")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], value="Qwen/Qwen2.5-1.5B-Instruct", label="Base Model")
        load_btn = gr.Button("Load Model", variant="secondary")
    load_status = gr.Textbox(label="Load Status", interactive=False)
    load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])
    
    with gr.Row():
        with gr.Column(scale=1):
            target_task = gr.Textbox(label="Target Task", value="Sorting Algorithm", lines=1)
            target_layer = gr.Slider(minimum=1, maximum=40, step=1, value=14, label="Target Layer (介入層)")
            alpha = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, value=1.0, label="Injection Strength (波の強さ)")
            run_btn = gr.Button("🌊 Trigger Vector Resonance", variant="primary")
            
        with gr.Column(scale=2):
            log_box = gr.Markdown("ここに物理空間の観測データ（波の干渉と共鳴の測定）が表示されます。")
            code_box = gr.Code(label="Crystallized Code (結晶化されたコード)", language="python", lines=25)
            
    run_btn.click(
        run_resonant_coder,
        inputs=[target_task, target_layer, alpha],
        outputs=[log_box, code_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7864)
