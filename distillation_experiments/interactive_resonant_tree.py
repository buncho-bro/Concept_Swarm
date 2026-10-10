"""
【ファイル概要】
interactive_resonant_tree.py

共鳴木（Resonant Tree）をインタラクティブに操作し、ユーザーが概念の分岐点に介入して推論方向を誘導できるUIバックエンド。
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
import gc

current_model_id = None
model = None
tokenizer = None

DOMAINS = [
    "Fluid Dynamics (流体力学)", "Quantum Mechanics (量子力学)", "Biological Evolution (生物進化)",
    "Economic Market Dynamics (市場原理)", "Musical Harmony (和音)",
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
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        current_model_id = model_id
        yield f"Successfully loaded {model_id}!"
    except Exception as e:
        yield f"Error loading model: {str(e)}"

# --- ベクトル系関数 ---
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

def generate_chimera_text(prompt, meta_vec, target_layer, alpha):
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
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    handle.remove()
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def generate_text_sync(prompt, system_prompt="You are an AI.", max_new_tokens=500, temperature=0.2):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=(temperature>0), pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def extract_python_code(text):
    match = re.search(r'```python(.*?)```', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

# --- インタラクティブ処理 ---
def generate_sparks(target_task, target_layer, alpha, chat_history):
    if model is None:
        chat_history.append({"role": "assistant", "content": "❌ Model is not loaded."})
        return chat_history, None, ""
        
    target_layer = int(target_layer)
    alpha = float(alpha)
    
    prompt = (
        f"Task: Implement a completely new {target_task}.\n"
        f"Input: An unordered array of integers.\n"
        f"Output: The exact same array sorted in strictly ascending order.\n"
        f"Requirement: Describe the core logical mechanism to achieve this sorting in 3 sentences."
    )
    
    chat_history.append({"role": "assistant", "content": "🌊 Measuring Vector Resonance..."})
    yield chat_history, None, ""
    
    results = []
    for domain in DOMAINS:
        meta_text = f"The physical principles of {domain}."
        meta_vec = extract_base_vector(meta_text, target_layer)
        score, sim, f_inc = measure_resonance(prompt, meta_vec, target_layer, alpha)
        results.append({"domain": domain, "score": score, "vec": meta_vec})
        
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    
    chat_history.append({"role": "assistant", "content": f"🏆 Highest Resonance: **{best['domain']}** (Score: {best['score']:.4f})\nExtracting the chimera spark..."})
    yield chat_history, None, ""
    
    spark = generate_chimera_text(prompt, best['vec'], target_layer, alpha)
    
    # 質問回数を管理するフラグを追加
    state_data = {
        "task": target_task, 
        "domain": best['domain'], 
        "spark": spark, 
        "context": f"Idea: {spark}\n\n",
        "question_count": 0
    }
    
    msg = (f"**Spark Generated ({best['domain']})**:\n{spark}\n\n"
           f"--- \n"
           f"Treeモデルに渡す準備ができました。もしこのアイデアに足りない仕様や、追加したい条件があれば入力して送信してください。（そのまま進める場合は空欄で送信してください）")
    chat_history.append({"role": "assistant", "content": msg})
    
    yield chat_history, state_data, ""

def process_reply(user_msg, chat_history, state_data, max_retries):
    if state_data is None:
        chat_history.append({"role": "assistant", "content": "❌ Please generate sparks first."})
        yield chat_history, state_data, ""
        return
        
    if user_msg:
        chat_history.append({"role": "user", "content": user_msg})
        state_data["context"] += f"User specification/clarification: {user_msg}\n\n"
        state_data["question_count"] += 1
    else:
        chat_history.append({"role": "user", "content": "[Proceed empty]"})
        state_data["question_count"] += 1 # 空で進めた場合も質問フェーズ終了とみなす
        
    yield chat_history, state_data, ""
    
    sys_tree = "You are an elite software architect."
    
    # 質問は最大1回（1往復）までに制限する
    allow_question = state_data["question_count"] < 1
    
    if allow_question:
        instruction = (
            f"If the logic is too abstract and lacks necessary parameters or specifications to write real code, "
            f"you MUST ask the user a clarifying question. Prefix your question with 'QUESTION: '.\n"
            f"If you have enough information, break the implementation into coding tasks and output ONLY a JSON array of strings representing the steps."
        )
    else:
        instruction = (
            f"Based on the provided idea and the user's clarification, you now have all the information you need.\n"
            f"DO NOT ask any more questions. Break the implementation into logical coding tasks and output ONLY a JSON array of strings representing the steps."
        )
    
    tree_prompt = (
        f"Task: {state_data['task']}\n"
        f"{state_data['context']}\n"
        f"Analyze the idea above to implement it as Python code.\n"
        f"{instruction}"
    )
    
    response = generate_text_sync(tree_prompt, sys_tree, max_new_tokens=300)
    
    # 質問が許可されていて、かつ QUESTION: が含まれている場合のみ質問ループへ
    if allow_question and "QUESTION:" in response:
        q_text = response.split("QUESTION:")[1].strip()
        chat_history.append({"role": "assistant", "content": f"❓ {q_text}"})
        yield chat_history, state_data, ""
        return
        
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        tasks = json.loads(match.group(0)) if match else ["Implement logic", "Add assertions"]
    except:
        tasks = ["Implement logic", "Add strict assertions"]
        
    chat_history.append({"role": "assistant", "content": f"Tasks defined:\n" + "\n".join([f"- {t}" for t in tasks]) + "\n\nStarting implementation..."})
    yield chat_history, state_data, ""
    
    accumulated_code = ""
    for i, task in enumerate(tasks):
        chat_history.append({"role": "assistant", "content": f"Executing Node {i+1}/{len(tasks)}: {task}"})
        yield chat_history, state_data, accumulated_code
        
        step_prompt = (
            f"Goal: {state_data['task']} based on idea: {state_data['spark']}\n\n"
            f"Current Code:\n```python\n{accumulated_code}\n```\n\n"
            f"Current task node: {task}\n"
            f"Extend the current code to complete this task. Output the ENTIRE updated Python code inside ```python ``` blocks."
        )
        if i == len(tasks) - 1:
            step_prompt += (f"\n\nCRITICAL RULE: You MUST append a testing block at the end. "
                            f"Generate a random list, run your algorithm, and do `assert your_result == sorted(original_list)`. "
                            f"If the logic is wrong, the assertion MUST fail.")
                            
        accumulated_code = extract_python_code(generate_text_sync(step_prompt, sys_tree, max_new_tokens=1500))
        yield chat_history, state_data, accumulated_code
        
    current_code = accumulated_code
    
    chat_history.append({"role": "assistant", "content": "⚙️ Phase 3: Execution & Test-Driven Self-Healing"})
    yield chat_history, state_data, current_code
    
    success = False
    for retry in range(int(max_retries)):
        chat_history.append({"role": "assistant", "content": f"Running code (Retry {retry}/{max_retries})..."})
        yield chat_history, state_data, current_code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(current_code)
            temp_file = f.name
            
        try:
            result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=5)
            os.remove(temp_file)
            
            if "assert " not in current_code:
                error_msg = "AssertionError: You MUST include `assert your_result == sorted(original_list)` in the code."
                chat_history.append({"role": "assistant", "content": "❌ Missing Assertion! You tried to escape the test."})
            elif result.returncode == 0:
                chat_history.append({"role": "assistant", "content": f"✅ Execution Successful & Assertions Passed!\nOutput:\n{result.stdout[:100]}..."})
                success = True
                break
            else:
                error_msg = result.stderr.strip()[-500:]
                if "AssertionError" in error_msg:
                    chat_history.append({"role": "assistant", "content": f"❌ AssertionError! Logic is flawed.\n{error_msg[-100:]}"})
                else:
                    chat_history.append({"role": "assistant", "content": f"❌ Execution Failed.\n{error_msg[-100:]}"})
        except subprocess.TimeoutExpired:
            os.remove(temp_file)
            error_msg = "TimeoutExpired: Infinite loop detected."
            chat_history.append({"role": "assistant", "content": f"❌ {error_msg}"})
        except Exception as e:
            os.remove(temp_file)
            error_msg = str(e)
            chat_history.append({"role": "assistant", "content": f"❌ System Error: {error_msg}"})
            
        yield chat_history, state_data, current_code
        
        chat_history.append({"role": "assistant", "content": "Healing code (Self-Correction)..."})
        heal_prompt = (
            f"The Python code failed.\n\nCode:\n```python\n{current_code}\n```\n\nError:\n{error_msg}\n\n"
            f"Fix the bug logically. You MUST include the assertion test. Output the complete corrected code."
        )
        current_code = extract_python_code(generate_text_sync(heal_prompt, sys_tree, max_new_tokens=1500))
        
    if success:
        chat_history.append({"role": "assistant", "content": "🎉 **Masterpiece Successfully Synthesized!**"})
    else:
        chat_history.append({"role": "assistant", "content": "⚠️ **Failed to verify algorithm.**"})
        
    yield chat_history, state_data, current_code

# --- UI ---
with gr.Blocks(title="Interactive Resonant Tree Coder") as demo:
    gr.Markdown("# 🗣️ Interactive Resonant Tree Coder (対話型・課題解決AI)")
    gr.Markdown("プロンプトを「課題解決型（厳密な入出力）」に強化。ベクトルによる「ひらめき」を生成後、AIが仕様不足を感じれば**あなたに質問**してきます。対話を通じて仕様を固めてから、Treeモデルがコードを書き上げます。")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], value="Qwen/Qwen2.5-1.5B-Instruct", label="Base Model")
        load_btn = gr.Button("Load Model", variant="secondary")
    load_status = gr.Textbox(label="Load Status", interactive=False)
    load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])
    
    state_data = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            target_task = gr.Textbox(label="Target Task", value="Sorting Algorithm", lines=1)
            target_layer = gr.Slider(minimum=1, maximum=40, step=1, value=14, label="Target Layer (介入層)")
            alpha = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, value=1.0, label="Injection Strength (波の強さ)")
            max_retries = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Max Healing Retries")
            btn_sparks = gr.Button("🌊 Step 1: Generate Spark (ひらめき生成)", variant="primary")
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="System & AI Communication", height=400)
            with gr.Row():
                txt_reply = gr.Textbox(label="User Feedback (AIからの質問に回答、またはそのまま進む)", placeholder="回答を入力...", scale=4)
                btn_reply = gr.Button("Send / Proceed", variant="primary", scale=1)
            
            code_box = gr.Code(label="Current / Final Code", language="python", lines=20)
            
    btn_sparks.click(
        generate_sparks,
        inputs=[target_task, target_layer, alpha, chatbot],
        outputs=[chatbot, state_data, code_box]
    )
    
    btn_reply.click(
        process_reply,
        inputs=[txt_reply, chatbot, state_data, max_retries],
        outputs=[chatbot, state_data, code_box]
    ).then(
        lambda: "", None, txt_reply
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866)
