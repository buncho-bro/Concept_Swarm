"""
【ファイル概要】
project_ripple_core.py

Project Rippleの旧バージョンコア。空間的な波及効果（Ripple）を用いて、一つの概念が他の概念へ物理的に影響を与えるモデル。
"""

import gradio as gr
import json
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import math

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = None
model = None

DOMAINS = {
    "Quantum Mechanics": "The abstract physical principles of Quantum Mechanics, superposition, and entanglement.",
    "Evolution": "The biological principles of evolution, natural selection, and genetic mutation.",
    "Fluid Dynamics": "The physical principles of fluid dynamics, turbulence, and flow.",
    "Thermodynamics": "The laws of thermodynamics, entropy, and heat transfer.",
    "None": "General logical reasoning without specific domain bias."
}

def init_model():
    global tokenizer, model
    if model is None:
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        print("Model loaded successfully.")

def get_terrain_vector(domain_name, layer_idx):
    if domain_name not in DOMAINS or domain_name == "None":
        return None
    text = DOMAINS[domain_name]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer_idx + 1][:, -1:, :]

def get_config_from_ai(user_task):
    sys_prompt = (
        "You are the Environment Configurator for 'Project Ripple'.\n"
        "Analyze the user's task and output physical parameters for the LLM.\n"
        "1. terrain_injection: Choose ONE from ['Quantum Mechanics', 'Evolution', 'Fluid Dynamics', 'Thermodynamics', 'None'].\n"
        "2. viscosity (0.0 to 1.0): Memory retention. High for deep focus/long docs. Low for rapid brainstorm.\n"
        "3. decay (0.0 to 1.0): Speed of context forgetting. Fast(high) for rapid switching. Slow(low) to keep context.\n\n"
        "CRITICAL RULE: The 'reasoning' value MUST be written in the exact same language as the user's Task (e.g., if the task is in Japanese, write reasoning in Japanese).\n\n"
        "Output ONLY JSON: {'terrain_injection': '...', 'viscosity': 0.0, 'decay': 0.0, 'reasoning': '...'}"
    )
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Task: {user_task}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return {"terrain_injection": "None", "viscosity": 0.5, "decay": 0.5, "reasoning": "Fallback config (language auto-detect failed)."}

def run_ripple_engine(user_task):
    init_model()
    
    yield "🤖 Analyzing task to configure environment...", "", "{}"
    
    # Layer 0: Configuration
    config = get_config_from_ai(user_task)
    terrain_name = config.get("terrain_injection", "None")
    viscosity = float(config.get("viscosity", 0.5))
    decay = float(config.get("decay", 0.5))
    
    log = (f"### ⚙️ Layer 0: Environment Configured\n"
           f"**Terrain:** `{terrain_name}`\n"
           f"**Viscosity (粘性):** `{viscosity}`\n"
           f"**Decay (減衰):** `{decay}`\n"
           f"**Reasoning:** {config.get('reasoning', '')}\n\n"
           f"🌊 Running Core Physics Engine...\n")
    yield log, "", json.dumps(config, indent=2)
    
    # Layer 1 & 2 Setup
    target_layer = 14
    terrain_vec = get_terrain_vector(terrain_name, target_layer)
    
    hooks = []
    generation_step = 0
    previous_hidden = None
    
    # Physics Hook
    def ripple_physics_hook(module, args, kwargs, output):
        nonlocal generation_step, previous_hidden
        hidden = output[0] if isinstance(output, tuple) else output
        
        if hidden.shape[1] == 1:
            generation_step += 1
            
            # 1. Decay Calculation
            current_power = math.exp(-decay * generation_step * 0.05)
            
            # 2. Terrain Injection
            if terrain_vec is not None:
                norm_meta = F.normalize(terrain_vec, p=2, dim=-1)
                current_norm = hidden.norm(dim=-1, keepdim=True)
                injection_force = norm_meta * (current_power * 1.5) * current_norm.mean()
                hidden = hidden + injection_force
                
            # 3. Viscosity
            if previous_hidden is not None and previous_hidden.shape == hidden.shape:
                hidden = (viscosity * previous_hidden) + ((1.0 - viscosity) * hidden)
                
            previous_hidden = hidden.detach()
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
            
        return output

    handle = model.model.layers[target_layer].register_forward_hook(ripple_physics_hook, with_kwargs=True)
    
    # Generation
    sys_prompt_gen = "You are the Ripple Physics Engine. You MUST respond in the EXACT SAME LANGUAGE as the user's task."
    prompt = f"Perform the following task.\nTask: {user_task}"
    messages = [{"role": "system", "content": sys_prompt_gen}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
    handle.remove()
    
    final_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    log += "\n✅ **Task Completed**"
    yield log, final_output, json.dumps(config, indent=2)

with gr.Blocks(title="Project Ripple: Core Engine MVP") as demo:
    gr.Markdown("# 🌊 Project Ripple: Core Engine MVP")
    gr.Markdown("タスクを入力すると、Configuratorがパラメータ（地形・粘性・減衰）を決定し、**実際のTransformerのテンソル演算内で物理シミュレーション（Hook）を実行して**推論を行います。")
    
    with gr.Row():
        task_input = gr.Textbox(label="User Task", lines=3, placeholder="ソートアルゴリズムを新しく考えて。")
        run_btn = gr.Button("Execute in Physics Engine", variant="primary")
        
    with gr.Row():
        with gr.Column():
            log_output = gr.Markdown("Status Log")
            json_config = gr.Code(label="Configurator Decision", language="json")
        with gr.Column():
            final_text = gr.Textbox(label="Final Ripple Output", lines=15)
            
    run_btn.click(
        run_ripple_engine,
        inputs=[task_input],
        outputs=[log_output, final_text, json_config]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7868)
