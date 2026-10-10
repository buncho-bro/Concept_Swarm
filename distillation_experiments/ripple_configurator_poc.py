"""
【ファイル概要】
ripple_configurator_poc.py

Ripple環境のセットアップやパラメータチューニングを自動化するProof of Conceptスクリプト。
"""

import gradio as gr
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = None
model = None

def init_model():
    global tokenizer, model
    if model is None:
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        print("Model loaded successfully.")

def configure_environment(user_task):
    # 遅延ロード（Gradio起動後に必要に応じてロードする）
    init_model()
    
    sys_prompt = (
        "You are the Environment Configurator AI for 'Project Ripple'.\n"
        "Your job is to analyze the user's task and determine the optimal physical parameters for the LLM's cognitive engine.\n\n"
        "Parameters to set:\n"
        "1. terrain_injection (string): An abstract concept to inject into the embedding layer to twist the context (e.g., 'Quantum Mechanics', 'Evolution', 'None' if strict logic is needed).\n"
        "2. viscosity (float 0.0 to 1.0): Memory retention. High (0.8-1.0) for long document reading/deep focus. Low (0.0-0.3) for rapid brainstorming.\n"
        "3. decay (float 0.0 to 1.0): Speed of context forgetting. Fast (0.8-1.0) for rapid task switching/troubleshooting. Slow (0.0-0.3) to keep context alive.\n\n"
        "Output ONLY a JSON block with the keys: 'terrain_injection', 'viscosity', 'decay', 'reasoning'."
    )
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Task: {user_task}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3)
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            config = json.loads(match.group(0))
        else:
            raise ValueError("No JSON found")
            
        display_text = (
            f"### ⚙️ 自動セッティング完了\n\n"
            f"**🏔️ 概念地形 (Terrain Vector):** `{config.get('terrain_injection', 'None')}`\n"
            f"**💧 波紋の粘性 (Viscosity):** `{config.get('viscosity', 0.5)}`\n"
            f"**📉 波紋の減衰 (Decay):** `{config.get('decay', 0.5)}`\n\n"
            f"**🧠 AIの判断理由:**\n{config.get('reasoning', 'No reasoning provided.')}"
        )
        return display_text, json.dumps(config, indent=2)
    except Exception as e:
        return f"Failed to parse configuration.\nAI Output: {response}", "{}"

with gr.Blocks(title="Ripple Environment Configurator") as demo:
    gr.Markdown("# 💧 Project Ripple: Environment Configurator AI (Layer 0)")
    gr.Markdown("ユーザーがタスクを入力すると、システムが自動的に裏側で最適な「地形（ベクトル）」と「波紋（粘性・減衰）」のパラメータを算出し、推論環境をセッティングします。")
    
    with gr.Row():
        with gr.Column():
            task_input = gr.Textbox(
                label="User Task (自然言語でのタスク入力)", 
                placeholder="例: この100ページのPDFを読んで、常識を覆すような全く新しい切り口の要約を作って。",
                lines=3
            )
            analyze_btn = gr.Button("Analyze & Configure", variant="primary")
            
        with gr.Column():
            result_display = gr.Markdown("ここにセッティング結果と理由が表示されます。")
            json_output = gr.Code(label="Internal JSON Config", language="json")
            
    analyze_btn.click(
        configure_environment,
        inputs=[task_input],
        outputs=[result_display, json_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7867)
