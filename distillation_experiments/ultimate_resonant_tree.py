"""
【ファイル概要】
ultimate_resonant_tree.py

共鳴木探索の最終形態。過去の探索結果をキャッシュし、概念のフラクタル構造上で最速で最適解に到達する。
"""

import gradio as gr
import torch
import torch.nn.functional as F
import random
import re
import json
import subprocess
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Thread
import gc

# グローバル状態
current_model_id = None
model = None
tokenizer = None

DOMAINS = [
    "Fluid Dynamics", "Quantum Mechanics", "Biological Evolution",
    "Economic Market Dynamics", "Musical Harmony",
    "Celestial Mechanics", "Thermodynamics",
    "Geological Sedimentation", "Neural Pruning",
    "Chemical Crystallization"
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

# --- ベクトル操作系 ---
def extract_base_vector(text, target_layer):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[target_layer + 1][:, -1:, :]

def measure_resonance(prompt, meta_vec, target_layer, alpha):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        base_outputs = model(**inputs, output_hidden_states=True)
        V_goal = base_outputs.hidden_states[target_layer + 1][:, -1:, :]
    
    V_infused_cache = []
    
    def eval_hook(module, args, kwargs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.shape[1] > 1:
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
            norm_meta = F.normalize(meta_vec, p=2, dim=-1)
            infused = hidden_states + (norm_meta * alpha * current_norm.mean())
            V_infused_cache.append(infused[:, -1:, :].detach())
            if isinstance(output, tuple):
                return (infused,) + output[1:]
            return infused
        return output

    handle = model.model.layers[target_layer].register_forward_hook(eval_hook, with_kwargs=True)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    V_infused = V_infused_cache[0]
    cos_sim = F.cosine_similarity(V_infused, V_goal, dim=-1).item()
    f_inc = V_infused.norm(dim=-1).item() / (V_goal.norm(dim=-1).item() + 1e-8)
    
    score = (cos_sim * f_inc) if cos_sim >= 0.5 else 0
    return score, cos_sim, f_inc

def generate_chimera_text(prompt, meta_vec, target_layer, alpha, max_tokens=150):
    """ フックをかけて「ひらめき」を抽出する（発散） """
    def chimera_hook(module, args, kwargs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.shape[1] > 1:
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
            norm_meta = F.normalize(meta_vec, p=2, dim=-1)
            infused = hidden_states + (norm_meta * alpha * current_norm.mean())
            if isinstance(output, tuple):
                return (infused,) + output[1:]
            return infused
        return output
        
    handle = model.model.layers[target_layer].register_forward_hook(chimera_hook, with_kwargs=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# --- 通常のテキスト生成系 (収束用) ---
def generate_text_sync(prompt, system_prompt="You are an AI.", max_new_tokens=500, temperature=0.2):
    """ フックなしの純粋な論理生成（収束） """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=(temperature>0), pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def extract_python_code(text):
    match = re.search(r'```python(.*?)```', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

# --- メインシステム ---
def run_ultimate_system(target_task, target_layer, alpha, max_retries):
    global model, tokenizer
    if model is None:
        yield "Error: Load model first.", ""
        return
        
    target_layer = int(target_layer)
    alpha = float(alpha)
    prompt = f"Brainstorm a completely new and revolutionary concept for a {target_task}. Write 3 sentences describing the abstract mechanism."
    
    log = "### 🌊 Phase 1: Vector Resonance Measurement\n"
    yield log, ""
    
    results = []
    for domain in DOMAINS:
        meta_text = f"The physical principles of {domain}."
        meta_vec = extract_base_vector(meta_text, target_layer)
        score, sim, f_inc = measure_resonance(prompt, meta_vec, target_layer, alpha)
        results.append({"domain": domain, "score": score, "sim": sim, "f_inc": f_inc, "vec": meta_vec})
        log += f"[{domain}] Score: {score:.4f} (Sim: {sim:.4f}, Force: {f_inc:.4f})\n"
        yield log, ""
        
    # 上位3つを選出
    results.sort(key=lambda x: x["score"], reverse=True)
    top3 = results[:3]
    
    log += "\n### 🧬 Phase 2: Chimera Extraction (The Sparks of Genius)\n"
    log += f"Top 3 Domains: {top3[0]['domain']}, {top3[1]['domain']}, {top3[2]['domain']}\n\n"
    yield log, ""
    
    seeds = []
    
    # 単独メタファーでのひらめき抽出
    for res in top3:
        log += f"Extracting spark using **{res['domain']}** vector...\n"
        yield log, ""
        chimera_text = generate_chimera_text(prompt, res['vec'], target_layer, alpha)
        seeds.append({"name": res['domain'], "text": chimera_text})
        log += f"Spark: {chimera_text}\n\n"
        yield log, ""
        
    # ハイブリッド（組み合わせ）メタファーの作成とひらめき抽出
    log += f"Extracting spark using **Hybrid Vector (Top 3 Combined)**...\n"
    yield log, ""
    hybrid_vec = (top3[0]['vec'] + top3[1]['vec'] + top3[2]['vec']) / 3.0
    hybrid_text = generate_chimera_text(prompt, hybrid_vec, target_layer, alpha)
    seeds.append({"name": "Hybrid (Combined)", "text": hybrid_text})
    log += f"Hybrid Spark: {hybrid_text}\n\n"
    yield log, ""
    
    sys_tree = "You are an elite Python architect. Translate abstract ideas into flawless, step-by-step logic and code."
    
    for attempt, seed in enumerate(seeds):
        log += f"\n=========================================\n"
        log += f"### 🌳 Phase 3 & 4: Tree Implementation & TDD Gravity (Attempt {attempt+1})\n"
        log += f"**Source Spark: {seed['name']}**\n\n"
        yield log, ""
        
        # 3.1 タスク分解
        log += "Building task tree from spark...\n"
        yield log, ""
        tree_prompt = (f"We must build a '{target_task}' based on this abstract idea:\n{seed['text']}\n\n"
                       f"Break down the implementation into a sequence of logical coding steps. "
                       f"Output ONLY a JSON array of strings representing the tasks.")
        task_tree_res = generate_text_sync(tree_prompt, sys_tree, max_new_tokens=300)
        
        try:
            match = re.search(r'\[.*?\]', task_tree_res, re.DOTALL)
            tasks = json.loads(match.group(0)) if match else ["Implement the core logic and add strict assertions."]
        except:
            tasks = ["Define structures", "Implement logic", "Add assertions"]
            
        log += "Tasks:\n" + "\n".join([f"- {t}" for t in tasks]) + "\n\n"
        yield log, ""
        
        # 3.2 段階的実装
        accumulated_code = ""
        for i, task in enumerate(tasks):
            log += f"Executing Node {i+1}/{len(tasks)}: {task}\n"
            yield log, accumulated_code
            
            step_prompt = (f"Goal: '{target_task}' based on this idea: {seed['text']}\n\n"
                           f"Current Code:\n```python\n{accumulated_code}\n```\n\n"
                           f"Current task: {task}\n"
                           f"Extend the current code to complete this task. Output the ENTIRE updated Python code inside ```python ``` blocks.")
                           
            if i == len(tasks) - 1:
                step_prompt += (f"\n\nCRITICAL RULE: You MUST append a testing block at the end. "
                                f"For sorting, generate a random list, run your algorithm, and do `assert your_result == sorted(original_list)`. "
                                f"If the logic is wrong, the assertion MUST fail.")
                                
            accumulated_code = extract_python_code(generate_text_sync(step_prompt, sys_tree, max_new_tokens=1500))
            yield log, accumulated_code
            
        current_code = accumulated_code
        
        # 3.3 実行と修復 (Self-Healing)
        log += "\n### ⚙️ Phase 5: Execution & Self-Healing\n"
        yield log, current_code
        
        success = False
        for retry in range(int(max_retries)):
            log += f"\nRunning code (Retry {retry}/{max_retries})...\n"
            yield log, current_code
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(current_code)
                temp_file = f.name
                
            try:
                result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=5)
                os.remove(temp_file)
                
                # Check if assert is present in the code (Absolute Gravity)
                if "assert " not in current_code:
                    log += f"❌ Missing Assertion! You tried to escape the test.\n"
                    error_msg = "AssertionError: You MUST include `assert your_result == sorted(original_list)` in the code to prove it works."
                elif result.returncode == 0:
                    log += f"✅ Execution Successful & Assertions Passed!\nOutput:\n{result.stdout[:200]}...\n"
                    success = True
                    break
                else:
                    error_msg = result.stderr.strip()[-1000:]
                    if "AssertionError" in error_msg:
                        log += f"❌ AssertionError! Logic is flawed.\nError:\n{error_msg[-200:]}\n"
                    else:
                        log += f"❌ Execution Failed.\nError:\n{error_msg[-200:]}\n"
            except subprocess.TimeoutExpired:
                os.remove(temp_file)
                error_msg = "TimeoutExpired: Infinite loop detected."
                log += f"❌ {error_msg}\n"
            except Exception as e:
                os.remove(temp_file)
                error_msg = str(e)
                log += f"❌ System Error: {error_msg}\n"
                
            yield log, current_code
            
            log += "Healing code (Self-Correction)...\n"
            yield log, current_code
            heal_prompt = (f"The Python code failed.\n\nCode:\n```python\n{current_code}\n```\n\n"
                           f"Error:\n{error_msg}\n\n"
                           f"Fix the bug logically without giving up the core idea. "
                           f"You MUST include the assertion test. Output the complete corrected code.")
            current_code = extract_python_code(generate_text_sync(heal_prompt, sys_tree, max_new_tokens=1500))
            
        if success:
            log += f"\n🎉 **Masterpiece Successfully Synthesized!**\n"
            yield log, current_code
            return 
        else:
            log += f"\n⚠️ **Failed after {max_retries} retries. Discarding spark and backtracking...**\n"
            yield log, current_code
            
    log += "\n💀 **All sparks failed to crystallize into valid code.**\n"
    yield log, ""

# --- UI ---
with gr.Blocks(title="The Ultimate Resonant Tree Coder") as demo:
    gr.Markdown("# 🌌 The Ultimate Resonant Tree Coder (真・飛躍的思考システム)")
    gr.Markdown("1. **ベクトル共鳴**: 最も親和性の高い上位3つのメタファー（とそれらを合成したハイブリッド）から純粋な「ひらめき」を抽出。\n2. **Tree収束**: そのひらめきをTreeモデルに渡し、タスク分解してコード化。\n3. **絶対重力(TDD)**: アサーションを物理的に強制。サボりは許されず、完全な動作コードのみが結晶化します。")
    
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
            max_retries = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Max Healing Retries (デバッグ回数)")
            run_btn = gr.Button("🌌 Trigger Ultimate System", variant="primary")
            
        with gr.Column(scale=2):
            log_box = gr.Markdown("ここに宇宙開闢から結晶化までのログが表示されます。")
            code_box = gr.Code(label="Current / Final Code", language="python", lines=30)
            
    run_btn.click(
        run_ultimate_system,
        inputs=[target_task, target_layer, alpha, max_retries],
        outputs=[log_box, code_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
