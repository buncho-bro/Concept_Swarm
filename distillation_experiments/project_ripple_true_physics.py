"""
【ファイル概要】
project_ripple_true_physics.py

Rippleの波及効果に流体力学的な「真の物理式」を取り入れ、ベクトル場としての自然な推論の収束を試みた意欲的な実験コード。
"""

import gradio as gr
import json
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import math

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = None
model = None
vocab_embeddings = None

DOMAINS = {
    "Quantum Mechanics": "The abstract physical principles of Quantum Mechanics, superposition, entanglement, and probability.",
    "Evolution": "The biological principles of evolution, natural selection, adaptation, and genetic mutation.",
    "Fluid Dynamics": "The physical principles of fluid dynamics, liquid, turbulence, and continuous flow.",
    "Thermodynamics": "The laws of thermodynamics, entropy, heat transfer, and energy conservation.",
    "None": "General logical reasoning without specific domain bias."
}

def init_model():
    global tokenizer, model, vocab_embeddings
    if model is None:
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        # 概念地形（全ボキャブラリのベクトル）を取得
        vocab_embeddings = model.get_output_embeddings().weight.detach()
        print("Model and Concept Topography loaded successfully.")

def get_terrain_vector(domain_name):
    if domain_name not in DOMAINS or domain_name == "None":
        return None
    text = DOMAINS[domain_name]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 最終隠れ層の平均を地形ベクトルとする
        hidden = outputs.hidden_states[-1]
        return hidden.mean(dim=1)

def get_config_from_ai(user_task):
    sys_prompt = (
        "You are the Environment Configurator for 'Project Ripple'.\n"
        "Analyze the user's task and output physical parameters for the LLM.\n"
        "1. terrain_injection: Choose ONE from ['Quantum Mechanics', 'Evolution', 'Fluid Dynamics', 'Thermodynamics', 'None'].\n"
        "2. viscosity (0.0 to 1.0): Concept retention power. High for deep focus. Low for rapid brainstorm.\n"
        "3. decay (0.0 to 1.0): Speed of context forgetting. Fast(high) for rapid switching. Slow(low) to keep context.\n\n"
        "CRITICAL RULE: The 'reasoning' value MUST be written in the exact same language as the user's Task.\n\n"
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
    return {"terrain_injection": "None", "viscosity": 0.5, "decay": 0.5, "reasoning": "Fallback config."}

# --- 真の投影モデル (Logits Processor) ---
class RippleTopographyProcessor(LogitsProcessor):
    def __init__(self, terrain_vec, vocab_emb, viscosity, decay, alpha=2.0):
        self.terrain_vec = F.normalize(terrain_vec, p=2, dim=-1) if terrain_vec is not None else None
        self.vocab_emb = F.normalize(vocab_emb, p=2, dim=-1)
        self.viscosity = viscosity
        self.decay = decay
        self.alpha = alpha
        self.step = 0
        
        # 事前に全単語と地形の親和性（コサイン類似度）を計算しておく（地形の起伏）
        if self.terrain_vec is not None:
            self.terrain_affinity = torch.matmul(self.vocab_emb, self.terrain_vec.T).squeeze()
        else:
            self.terrain_affinity = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        
        # 減衰 (Decay) の計算
        # decayが高いほど早く力が失われる
        current_power = math.exp(-self.decay * self.step * 0.1)
        
        if self.terrain_affinity is not None:
            # 波の形（本来の確率分布）に対して、地形の引力（Affinity）を足し合わせる
            # affinityの高い単語（概念に近い単語）が水面から浮かび上がりやすくなる
            gravity_bias = self.terrain_affinity * (self.alpha * current_power)
            
            # 粘性 (Viscosity) の影響
            # 粘性が高いほど、地形の引力が局所的（特定の単語）に集中する効果を持たせる(温度スケーリング的)
            # 粘性が低いほど、広く浅く引力がかかる
            viscosity_factor = 1.0 + (self.viscosity * 2.0)
            gravity_bias = gravity_bias * viscosity_factor
            
            scores = scores + gravity_bias.to(scores.device)
            
        return scores

def run_ripple_engine(user_task):
    init_model()
    
    yield "🤖 Analyzing task to configure environment...", "", "{}"
    
    config = get_config_from_ai(user_task)
    terrain_name = config.get("terrain_injection", "None")
    viscosity = float(config.get("viscosity", 0.5))
    decay = float(config.get("decay", 0.5))
    
    log = (f"### ⚙️ Layer 0: Environment Configured\n"
           f"**Terrain Vector:** `{terrain_name}`\n"
           f"**Viscosity (粘性):** `{viscosity}`\n"
           f"**Decay (減衰):** `{decay}`\n"
           f"**Reasoning:** {config.get('reasoning', '')}\n\n"
           f"🌊 Raindrops falling... Computing wave interference and topography projection...\n")
    yield log, "", json.dumps(config, indent=2)
    
    terrain_vec = get_terrain_vector(terrain_name)
    
    # 投影時のプロセッサ（物理シミュレータ）をセット
    processors = LogitsProcessorList()
    if terrain_vec is not None:
        ripple_processor = RippleTopographyProcessor(
            terrain_vec=terrain_vec,
            vocab_emb=vocab_embeddings,
            viscosity=viscosity,
            decay=decay,
            alpha=3.0 # 地形の引力の基本強度
        )
        processors.append(ripple_processor)
    
    sys_prompt_gen = "You are the Ripple Physics Engine. You MUST respond in the EXACT SAME LANGUAGE as the user's task."
    prompt = f"Perform the following task.\nTask: {user_task}"
    messages = [{"role": "system", "content": sys_prompt_gen}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            temperature=0.7, 
            do_sample=True, 
            logits_processor=processors,
            pad_token_id=tokenizer.eos_token_id
        )
        
    final_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    log += "\n✅ **Projection Complete**"
    yield log, final_output, json.dumps(config, indent=2)

with gr.Blocks(title="Project Ripple: True Physics Engine") as demo:
    gr.Markdown("# 🌊 Project Ripple: True Physics Engine")
    gr.Markdown("雨粒（入力）から生じた波の干渉計算はAIのネイティブ機構に任せ、**「最終的な波の形を概念地形に投影する瞬間（Logits出力時）」**に地形の引力と減衰を介入させる、文章を崩壊させない真の物理モデルです。")
    
    with gr.Row():
        task_input = gr.Textbox(label="User Task", lines=3, placeholder="ソートアルゴリズムの新しいパラダイムを考えて。")
        run_btn = gr.Button("Execute in Physics Engine", variant="primary")
        
    with gr.Row():
        with gr.Column():
            log_output = gr.Markdown("Status Log")
            json_config = gr.Code(label="Configurator Decision", language="json")
        with gr.Column():
            final_text = gr.Textbox(label="Final Ripple Output (投影結果)", lines=15)
            
    run_btn.click(
        run_ripple_engine,
        inputs=[task_input],
        outputs=[log_output, final_text, json_config]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7869)
