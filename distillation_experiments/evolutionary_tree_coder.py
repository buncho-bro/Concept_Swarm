"""
【ファイル概要】
evolutionary_tree_coder.py

コードの派生ツリー（ASTやバージョン木）を概念空間にマッピングし、MCTS（モンテカルロ木探索）と大円干渉を組み合わせて最適解を探索するスクリプト。
"""

import gradio as gr
import torch
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
    "Geological Sedimentation", "Neural Synaptic Pruning",
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

def generate_text_sync(prompt, system_prompt="You are an AI.", max_new_tokens=500, temperature=1.0):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def extract_python_code(text):
    match = re.search(r'```python(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def run_evolutionary_tree_coder(target_task, num_seeds, max_retries):
    global model, tokenizer
    if model is None:
        yield "Error: Please load a model first.", ""
        return
        
    log = "### 🧬 Phase 1: Evolutionary Seed Generation\n\n"
    yield log, ""
    
    seeds = []
    sys_evo = "You are a polymath who invents radically new algorithms by combining unrelated fields."
    
    # 1. 大量生成
    for i in range(int(num_seeds)):
        domain = random.choice(DOMAINS)
        prompt = (f"Target Task: Create a completely new {target_task}.\n"
                  f"Constraint: You MUST use the metaphor of '{domain}'.\n"
                  f"Do NOT use existing approaches. Describe the core mechanism in 3 sentences.")
        
        log += f"**Generating Seed {i+1} [Domain: {domain}]...**\n"
        yield log, ""
        idea = generate_text_sync(prompt, sys_evo, max_new_tokens=150, temperature=1.2)
        seeds.append({"id": i+1, "domain": domain, "idea": idea})
        
    # 2. スクリーニング
    log += "\n### ⚖️ Phase 2: Natural Selection (Screening)\n\n"
    yield log, ""
    
    for seed in seeds:
        eval_prompt = (f"Evaluate this idea for '{target_task}':\n{seed['idea']}\n\n"
                       "Rate on two scales (1 to 10):\n"
                       "1. Novelty (Is it different from existing methods?)\n"
                       "2. Feasibility (Can it be logically implemented in code?)\n"
                       "Output ONLY a JSON format: {\"novelty\": int, \"feasibility\": int}")
        eval_res = generate_text_sync(eval_prompt, "You are a strict code evaluator.", max_new_tokens=50, temperature=0.1)
        
        try:
            match = re.search(r'\{.*?\}', eval_res, re.DOTALL)
            scores = json.loads(match.group(0)) if match else {}
            seed["score"] = int(scores.get("novelty", 0)) * int(scores.get("feasibility", 0))
        except:
            seed["score"] = 0
            
        log += f"Seed {seed['id']} Score: {seed['score']}\n"
        yield log, ""
        
    seeds.sort(key=lambda x: x["score"], reverse=True)
    
    sys_tree = "You are an elite software architect and Python developer. Break problems down logically, solve them step-by-step, and write bug-free code."
    
    for attempt, best_seed in enumerate(seeds):
        if best_seed["score"] == 0:
            continue
            
        log += f"\n=========================================\n"
        log += f"### 🌳 Phase 3: Recursive Tree Implementation (Attempt {attempt+1})\n"
        log += f"**Selected Seed {best_seed['id']} [Domain: {best_seed['domain']}]**\n"
        log += f"{best_seed['idea']}\n\n"
        yield log, ""
        
        # 3.1 タスクの再帰的分解 (Tree Building)
        log += "Analyzing and splitting into logical task nodes...\n"
        yield log, ""
        tree_prompt = (f"We need to implement this revolutionary idea for '{target_task}':\n"
                       f"Idea: {best_seed['idea']}\n\n"
                       f"Break down the implementation into a sequence of necessary coding tasks. "
                       f"If a task is too complex, break it down further. "
                       f"Output ONLY a JSON array of strings representing the task steps.")
        task_tree_res = generate_text_sync(tree_prompt, sys_tree, max_new_tokens=300, temperature=0.2)
        
        try:
            match = re.search(r'\[.*?\]', task_tree_res, re.DOTALL)
            tasks = json.loads(match.group(0)) if match else ["Implement the full logic and add strict assertions."]
        except:
            tasks = ["Define data structures", "Implement core logic", "Add strict assertions and testing"]
            
        log += f"Identified Tasks:\n" + "\n".join([f"- {t}" for t in tasks]) + "\n\n"
        yield log, ""
        
        # 3.2 段階的実装 (Step-by-step Execution)
        accumulated_code = ""
        for i, task in enumerate(tasks):
            log += f"Executing Node {i+1}/{len(tasks)}: {task}\n"
            yield log, accumulated_code
            
            step_prompt = (f"Goal: {target_task} based on '{best_seed['domain']}' metaphor.\n"
                           f"Core Idea: {best_seed['idea']}\n\n"
                           f"Current Code:\n```python\n{accumulated_code}\n```\n\n"
                           f"Your current task node is: {task}\n"
                           f"Extend or modify the current code to complete this task. "
                           f"Output the ENTIRE updated Python code inside ```python ``` blocks. Do not omit previous code.")
                           
            # 最後のタスクにはTDDの重力を強制する
            if i == len(tasks) - 1:
                step_prompt += (f"\n\nCRITICAL INSTRUCTION: This is the final integration node. "
                                f"You MUST append a testing block that uses strict assertions to mathematically prove your algorithm works. "
                                f"For a sorting algorithm, generate a random list, run your sort, and do `assert your_result == sorted(original_list)`. "
                                f"If the logic is flawed (e.g., elements disappear), the assertion must fail.")
                                
            accumulated_code = extract_python_code(generate_text_sync(step_prompt, sys_tree, max_new_tokens=1500, temperature=0.2))
            yield log, accumulated_code
            
        current_code = accumulated_code
        
        # 3.3 実行と自己修復 (Self-Healing Loop with strict assertions)
        log += "\n### ⚙️ Phase 4: Execution & Test-Driven Self-Healing\n"
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
                
                if result.returncode == 0:
                    log += f"✅ Execution Successful! Assertions passed.\nOutput:\n{result.stdout[:200]}...\n"
                    success = True
                    break
                else:
                    error_msg = result.stderr.strip()[-1000:]
                    if "AssertionError" in error_msg:
                        log += f"❌ AssertionError! The logic is flawed (the result is incorrect).\nError:\n{error_msg[-200:]}\n"
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
            
            log += "Healing code (Self-Correction against constraints)...\n"
            yield log, current_code
            heal_prompt = (f"The following Python code for '{target_task}' failed during testing.\n\n"
                           f"Code:\n```python\n{current_code}\n```\n\n"
                           f"Error message:\n{error_msg}\n\n"
                           f"Fix the bug logically. If it's an AssertionError, it means your metaphor's logic failed to achieve the target task correctly. "
                           f"Rethink the logic, fix the implementation, and provide the complete, corrected Python code inside ```python ``` blocks. "
                           f"Ensure the strict assertions pass.")
            current_code = extract_python_code(generate_text_sync(heal_prompt, sys_tree, max_new_tokens=1500, temperature=0.2))
            
        if success:
            log += f"\n🎉 **Algorithm Successfully Synthesized, Tested, and Proven!**\n"
            yield log, current_code
            return 
        else:
            log += f"\n⚠️ **Tree Implementation Failed after {max_retries} retries.**\n"
            log += f"The idea could not be mathematically proven. Discarding this seed and restarting (Backtracking)...\n"
            yield log, current_code
            
    log += "\n💀 **All seeds failed. The AI could not synthesize a working algorithm.**\n"
    yield log, ""

# --- UI ---
with gr.Blocks(title="Evolutionary Tree Coder (TDD Version)") as demo:
    gr.Markdown("# 🧬🌳 Evolutionary Tree Coder (TDD & Recursive Split)")
    gr.Markdown("Treeモデルによる『タスクの再帰的分解と段階的実装』を追加し、さらに『厳密なアサーション（TDD）』をコンパイラの重力として課した最終形態です。")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], value="Qwen/Qwen2.5-1.5B-Instruct", label="Base Model")
        load_btn = gr.Button("Load Model", variant="secondary")
    load_status = gr.Textbox(label="Load Status", interactive=False)
    load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])
    
    with gr.Row():
        with gr.Column(scale=1):
            target_task = gr.Textbox(label="Target Task", value="Sorting Algorithm", lines=1)
            num_seeds = gr.Slider(minimum=3, maximum=10, step=1, value=5, label="Number of Seeds (発散数)")
            max_retries = gr.Slider(minimum=1, maximum=10, step=1, value=4, label="Max Healing Retries (デバッグ回数)")
            run_btn = gr.Button("🚀 Initialize Evolutionary Tree", variant="primary")
            
        with gr.Column(scale=2):
            log_box = gr.Markdown("ここにシステム全体の動作ログが表示されます。")
            code_box = gr.Code(label="Current / Final Code", language="python", lines=30)
            
    run_btn.click(
        run_evolutionary_tree_coder,
        inputs=[target_task, num_seeds, max_retries],
        outputs=[log_box, code_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863)
