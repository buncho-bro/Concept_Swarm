"""
【ファイル概要】
evolutionary_coder.py

生成されたコードの実行結果（エラー）をフィードバックとし、概念地形上の座標を進化計算（GA）的に探索・最適化してバグを修正するコーダー。
"""

import gradio as gr
import torch
import random
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import gc

# グローバル状態
current_model_id = None
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 異分野ドメインのリスト（突然変異の種）
DOMAINS = [
    "Fluid Dynamics (流体力学)", "Quantum Mechanics (量子力学)", "Biological Evolution (生物の進化)",
    "Economic Market Dynamics (経済の市場原理)", "Musical Harmony and Acoustics (音楽の和音と音響学)",
    "Celestial Mechanics (天体の軌道力学)", "Thermodynamics and Entropy (熱力学とエントロピー)",
    "Geological Sedimentation (地質の堆積作用)", "Neural Synaptic Pruning (神経シナプスの刈り込み)",
    "Chemical Crystallization (化学的結晶化)", "Ant Colony Optimization (アリの群知能)",
    "Epidemiological Spread (ウイルスの伝播モデル)"
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

def generate_text_sync(prompt, max_new_tokens=150, temperature=1.0, do_sample=True):
    """ 同期的にテキストを生成するヘルパー関数 """
    messages = [
        {"role": "system", "content": "You are a brilliant computer scientist and a polymath who invents new algorithms by combining unrelated fields."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # プロンプト部分を切り取る
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def run_evolutionary_thinking(target_task, num_seeds, gen_temp):
    global model, tokenizer
    if model is None:
        yield "Error: Please load a model first.", ""
        return
        
    log_output = "### Phase 1: Seed Generation (Massive Mutation)\n\n"
    yield log_output, ""
    
    seeds = []
    
    # --- Phase 1: 大量生成 (突然変異) ---
    for i in range(int(num_seeds)):
        domain = random.choice(DOMAINS)
        prompt = (f"Target Task: Create a completely new {target_task}.\n"
                  f"Constraint: You MUST use the metaphor and principles of '{domain}'.\n"
                  f"Do NOT use any existing approaches. Describe the core mechanism in 3-4 sentences.")
        
        log_output += f"**Seed {i+1} [Domain: {domain}]**\n"
        yield log_output, ""
        
        idea = generate_text_sync(prompt, max_new_tokens=200, temperature=gen_temp)
        seeds.append({"id": i+1, "domain": domain, "idea": idea})
        
        log_output += f"{idea}\n\n"
        yield log_output, ""
        
    # --- Phase 2: スクリーニング (自動評価と淘汰) ---
    log_output += "---\n### Phase 2: Automatic Screening (Natural Selection)\n\n"
    yield log_output, ""
    
    best_score = -1
    best_seed = None
    
    for seed in seeds:
        eval_prompt = (f"Evaluate this new approach for '{target_task}':\n\n{seed['idea']}\n\n"
                       "Rate on two scales (1 to 10):\n"
                       "1. Novelty (Is it truly different from existing methods?)\n"
                       "2. Feasibility (Can this actually be implemented as a working logic/code, or is it just nonsense?)\n"
                       "Output ONLY a JSON format: {\"novelty\": <int>, \"feasibility\": <int>}")
        
        # 評価は冷静に行うため temperature を低く設定
        eval_result = generate_text_sync(eval_prompt, max_new_tokens=50, temperature=0.1)
        
        try:
            # 簡単なJSON抽出
            match = re.search(r'\{.*?\}', eval_result, re.DOTALL)
            if match:
                scores = json.loads(match.group(0))
                novelty = int(scores.get("novelty", 0))
                feasibility = int(scores.get("feasibility", 0))
            else:
                novelty, feasibility = 0, 0
        except:
            novelty, feasibility = 0, 0
            
        total_score = novelty * feasibility
        seed["score"] = total_score
        seed["novelty"] = novelty
        seed["feasibility"] = feasibility
        
        log_output += f"Seed {seed['id']} -> Novelty: {novelty}/10, Feasibility: {feasibility}/10 (Total: {total_score})\n"
        yield log_output, ""
        
        if total_score > best_score:
            best_score = total_score
            best_seed = seed
            
    if best_seed is None:
        yield log_output + "\nError: Could not evaluate any seeds properly.", ""
        return
        
    log_output += f"\n🏆 **Winner: Seed {best_seed['id']} (Domain: {best_seed['domain']}) with score {best_score}**\n"
    yield log_output, "Preparing to write code..."
    
    # --- Phase 3: コードの結晶化 ---
    final_prompt = (f"You are an elite programmer. Based on the following revolutionary concept, "
                    f"write a fully functional Python script that implements this exact idea for '{target_task}'.\n\n"
                    f"Concept (Domain: {best_seed['domain']}):\n{best_seed['idea']}\n\n"
                    f"Ensure the code is complete, highly logical, and includes comments explaining how the metaphor translates to code. "
                    f"Do NOT fallback to standard algorithms.")
                    
    messages = [
        {"role": "system", "content": "You are a brilliant software engineer who brings abstract concepts into working Python code."},
        {"role": "user", "content": final_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=800,
        streamer=streamer,
        do_sample=True,
        temperature=0.3, # コーディングなので冷静に
        pad_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    final_code = ""
    for new_text in streamer:
        final_code += new_text
        yield log_output, final_code


# --- UI構築 ---
with gr.Blocks(title="Evolutionary Coder") as demo:
    gr.Markdown("# 🧬 Evolutionary Coder (飛躍的思考ジェネレーター)")
    gr.Markdown("AIの圧倒的な計算速度を活かし、ランダムな異分野メタファーから『大量のアイデア』を生成させ、AI自身に評価・淘汰させることで、全く新しい未知のアルゴリズム（突然変異）を抽出してコード化するシステムです。")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"], value="Qwen/Qwen2.5-1.5B-Instruct", label="Base Model")
        load_btn = gr.Button("Load Model", variant="secondary")
    load_status = gr.Textbox(label="Load Status", interactive=False)
    load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])
    
    with gr.Row():
        with gr.Column(scale=1):
            target_task = gr.Textbox(label="Target Task (目的)", value="Sorting Algorithm", lines=1)
            num_seeds = gr.Slider(minimum=3, maximum=20, step=1, value=5, label="Number of Seeds (大量生成する数)", info="PCのスペックに合わせて調整（最初は5くらい推奨）")
            gen_temp = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.2, label="Mutation Temperature (乱れ度)", info="高いほど奇抜なアイデアになります")
            run_btn = gr.Button("🚀 Run Evolutionary Thinking", variant="primary")
            
        with gr.Column(scale=2):
            log_box = gr.Markdown("ここに生成プロセス（進化と淘汰の記録）が表示されます。")
            code_box = gr.Textbox(label="Final Crystallized Code (最終出力コード)", lines=25)
            
    run_btn.click(
        run_evolutionary_thinking,
        inputs=[target_task, num_seeds, gen_temp],
        outputs=[log_box, code_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
