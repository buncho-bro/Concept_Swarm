"""
【ファイル概要】
chat_forge_webui.py

蒸留された概念モデルとチャット形式で対話するためのGradioベースのWebUI。
"""

import gradio as gr
import os
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import glob

# ==========================================
# グローバル状態
# ==========================================
current_model = None
current_tokenizer = None
current_model_name = "None"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cached_faiss_index = None
cached_lm_head_weight = None
cached_faiss_model_id = None
cached_concept_vecs = None
cached_model_id = None

# 純度100%の概念アンカー（Universal Language / English）
CONCEPT_WORDS = [
    "Time", "Space", "Existence", "Nothingness", "Truth", "Logic", "Philosophy", "Ethics", "Morality", "Justice", "Freedom", "Equality", "Peace", "War", "History", "Future", "Past", "Eternity",
    "Mathematics", "Chemistry", "Biology", "Geometry", "Algebra", "Probability", "Statistics", "Quantum", "Particle", "Light", "Gravity", "Magnetism", "Electricity", "Energy", "Mass", "Velocity",
    "Economy", "Market", "Capital", "Currency", "Labor", "Industry", "Trade", "Enterprise", "Society", "Culture", "Law", "Nation", "Democracy", "Dictatorship", "Revolution", "Rights",
    "Art", "Music", "Painting", "Sculpture", "Literature", "Poetry", "Theater", "Movie", "Beauty", "Expression", "Creation", "Tradition", "Religion", "Faith", "God", "Spirit", "Soul",
    "Plant", "Tree", "Flower", "Grass", "Forest", "Ocean", "Mountain", "River", "Desert", "Climate", "Weather", "Rain", "Snow", "Wind", "Thunder", "Earth", "Planet", "Star",
    "Machine", "Tool", "Engine", "Robot", "Circuit", "Communication", "Metal", "Iron", "Gold", "Silver", "Copper", "Water", "Fire", "Earth_soil", "Air", "Oxygen", "Hydrogen",
    "Joy", "Sadness", "Fear", "Surprise", "Love", "Hate", "Hope", "Despair", "Loneliness", "Happiness", "Pain", "Pleasure", "Memory", "Thought", "Consciousness", "Unconscious"
]

def free_vram():
    global current_model, current_tokenizer, current_model_name
    if current_model is not None:
        del current_model
        current_model = None
    if current_tokenizer is not None:
        del current_tokenizer
        current_tokenizer = None
    current_model_name = "None"
    gc.collect()
    torch.cuda.empty_cache()

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :] 
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

# ==========================================
# モデルロード
# ==========================================
def load_model_for_chat(base_id, lora_dir, use_lora):
    global current_model, current_tokenizer, current_model_name
    free_vram()
    
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        current_tokenizer = AutoTokenizer.from_pretrained(base_id)
        current_model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb_config, device_map={"": "cuda:0"})
        
        if lora_dir == "None":
            use_lora = False
            
        if use_lora:
            if os.path.exists(lora_dir):
                try:
                    current_model = PeftModel.from_pretrained(current_model, lora_dir)
                    current_model_name = f"Distilled Model (LoRA: {os.path.basename(lora_dir)})"
                except Exception as lora_e:
                    current_model_name = "Base Model (LoRA Load Failed)"
                    return f"❌ ERROR: LoRAの適用に失敗しました。\n詳細: {str(lora_e)}"
            else:
                current_model_name = "Base Model"
                return f"⚠️ WARNING: LoRA path '{lora_dir}' not found. Loaded Base Model instead."
        else:
            current_model_name = "Base Model"
            
        return f"✅ Successfully loaded: {current_model_name}"
    except Exception as e:
        return f"❌ Error loading Model: {str(e)}"

# ==========================================
# チャットロジック (Deepthink & FAISS 搭載)
# ==========================================
def chat_fn(message, history, use_deepthink, use_lightweight):
    global current_model, current_tokenizer, current_model_name
    if current_model is None:
        yield "⚠️ Please load a model first in the options above."
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
        
    system_prompt = ""
    thinking_log = ""
    
    # --- 🧠 Deepthink (Fractal Routing) Mode ---
    if use_deepthink:
        inputs_dt = current_tokenizer(message[-15:], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_dt = current_model(**inputs_dt, output_hidden_states=True)
        user_vec = F.normalize(outputs_dt.hidden_states[-1][0, -1, :].to(torch.float32), p=2, dim=0)
        
        global cached_concept_vecs, cached_model_id
        if 'cached_model_id' not in globals() or cached_model_id != current_model_name or cached_concept_vecs is None:
            yield "Initializing Concept Topography Cache (First time only...)\n"
            concept_vecs = []
            for w in CONCEPT_WORDS:
                concept_vecs.append(get_word_embedding(current_model, current_tokenizer, w))
            cached_concept_vecs = torch.stack(concept_vecs)
            cached_model_id = current_model_name
            
        concept_vecs = cached_concept_vecs
        
        sims_macro = F.cosine_similarity(user_vec.unsqueeze(0), concept_vecs)
        macro_idx = torch.argmax(sims_macro).item()
        c_macro_vec = concept_vecs[macro_idx]
        macro_concept = CONCEPT_WORDS[macro_idx]
        
        dot_u = torch.dot(user_vec, c_macro_vec)
        u_micro = user_vec - dot_u * c_macro_vec
        u_micro = F.normalize(u_micro, p=2, dim=0)
        
        dot_c = torch.matmul(concept_vecs, c_macro_vec) 
        concept_micro_vecs = concept_vecs - dot_c.unsqueeze(1) * c_macro_vec.unsqueeze(0)
        concept_micro_vecs = F.normalize(concept_micro_vecs, p=2, dim=1)
        
        sims_micro = F.cosine_similarity(u_micro.unsqueeze(0), concept_micro_vecs)
        sims_micro[macro_idx] = -1.0 
        
        sorted_sims, sorted_idx = torch.sort(sims_micro, descending=True)
        
        routed_concepts = []
        max_sim = sorted_sims[0].item()
        
        for i in range(5):
            current_sim = sorted_sims[i].item()
            if (max_sim - current_sim) > 0.15 or current_sim <= 0.0:
                break
            routed_concepts.append(CONCEPT_WORDS[sorted_idx[i].item()])
            
        if not routed_concepts:
            routed_concepts.append(CONCEPT_WORDS[sorted_idx[0].item()])
        
        thinking_log = f"**[Fractal Deepthink]**\n"
        thinking_log += f"Macro Valley (重力源): `{macro_concept}`\n"
        thinking_log += f"Micro Routing (重力除去後の突起探索): `{'` -> `'.join(routed_concepts)}`\n\n"
        yield thinking_log
        
        system_prompt = (
            f"System: [Latent Trajectory Suggestion]\n"
            f"- Macro Concept: '{macro_concept}'\n"
            f"- Micro Concepts: {', '.join(routed_concepts)}\n\n"
            f"Instruction: Use these concepts as a creative scaffolding for your reasoning. "
            f"However, you have absolute authority to self-correct: if you determine that any of these concepts lead to a logical dead end or are irrelevant to the user's actual prompt, you MUST ignore them and dynamically correct your trajectory to provide the most accurate answer.\n\n"
        )

    messages.append({"role": "user", "content": system_prompt + message})
    text = current_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = current_tokenizer(text, return_tensors="pt").to(device)
    
    # --- 🍃 Lightweight Mode (FAISS 100-words Decoding) ---
    if use_lightweight:
        import faiss
        global cached_faiss_index, cached_lm_head_weight, cached_faiss_model_id
        
        if 'cached_faiss_model_id' not in globals() or cached_faiss_model_id != current_model_name or cached_faiss_index is None:
            yield thinking_log + "🍃 Initializing Lightweight FAISS Index (150,000 words). First time only...\n"
            cached_lm_head_weight = current_model.lm_head.weight.detach()
            lm_head_np = cached_lm_head_weight.cpu().to(torch.float32).numpy()
            faiss.normalize_L2(lm_head_np)
            D = lm_head_np.shape[1]
            cached_faiss_index = faiss.IndexFlatIP(D)
            cached_faiss_index.add(lm_head_np)
            cached_faiss_model_id = current_model_name
            
        generated_text = thinking_log
        yield generated_text
        
        past_key_values = None
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = current_model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past_key_values = outputs.past_key_values
        hidden = outputs.hidden_states[-1][0, -1, :]
        
        past_generated_tokens = [] # 繰り返しペナルティ用の履歴
        
        for _ in range(2048):
            # 1. 通常探査 (K=2048)
            hidden_np = hidden.cpu().to(torch.float32).numpy().reshape(1, -1)
            faiss.normalize_L2(hidden_np)
            _, I = cached_faiss_index.search(hidden_np, 2048) 
            candidates = I[0]
            
            local_weights = cached_lm_head_weight[candidates]
            local_logits = torch.matmul(hidden.unsqueeze(0), local_weights.T)
            
            # --- 動的拡張探査 (Dynamic Fallback Expansion) ---
            # 候補の中で一番自信がある単語でもロジット(確率の元)が極端に低い場合、
            # 「この中には正解がない」と判断し、探索範囲を一気に拡張する
            if torch.max(local_logits).item() < 5.0: # Qwenの一般的なロジット分布に基づく閾値
                _, I_expand = cached_faiss_index.search(hidden_np, 15000) # 15000語まで緊急拡張
                candidates = I_expand[0]
                local_weights = cached_lm_head_weight[candidates]
                local_logits = torch.matmul(hidden.unsqueeze(0), local_weights.T)
            
            # --- 繰り返しペナルティ (Repetition Penalty) ---
            penalty = 1.15
            for past_token in set(past_generated_tokens[-50:]): 
                if past_token in candidates:
                    idx = np.where(candidates == past_token)[0][0]
                    score = local_logits[0, idx].item()
                    local_logits[0, idx] = score / penalty if score > 0 else score * penalty
            
            # --- 温度付きサンプリング (Temperature Scaling) ---
            temperature = 0.7
            local_logits = local_logits / temperature
            probs = F.softmax(local_logits, dim=-1)
            
            # --- Top-P (Nucleus) Sampling ---
            top_p = 0.9
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            mask = cumulative_probs > top_p
            mask[0, 0] = False 
            
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum() 
            
            sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).item()
            best_local_idx = sorted_indices[0, sampled_sorted_idx].item()
            next_token = candidates[best_local_idx]
            
            past_generated_tokens.append(next_token)
            
            if next_token in [current_tokenizer.eos_token_id, current_tokenizer.pad_token_id] or next_token == 151645:
                break
                
            # 文字化け対策: 全トークンを一気にデコードし直すことで、不完全なバイト列(漢字の欠片)が正しく結合される
            new_text_full = current_tokenizer.decode(past_generated_tokens, skip_special_tokens=True)
            yield thinking_log + new_text_full
            
            input_ids = torch.tensor([[next_token]]).to(device)
            with torch.no_grad():
                outputs = current_model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            hidden = outputs.hidden_states[-1][0, -1, :]
            
    else:
        # --- 従来方式 (全15万語計算) ---
        streamer = TextIteratorStreamer(current_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
        )
        
        thread = Thread(target=current_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated_text = thinking_log
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

def get_available_loras():
    lora_dirs = [d for d in glob.glob("./lora_*") if os.path.isdir(d)]
    if os.path.exists("./concept_lora_output"):
        lora_dirs.append("./concept_lora_output")
    return ["None"] + lora_dirs

def refresh_lora_list():
    return gr.Dropdown(choices=get_available_loras())

# ==========================================
# Gradio UI 構築 (チャット専用)
# ==========================================
STUDENT_CHOICES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it"
]

with gr.Blocks(title="Chat Forge", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💬 Chat Forge")
    gr.Markdown("Concept Forge によって蒸留・生成された概念地形（LoRA）を用いて、フラクタル・ルーティングによる高度な対話を行います。")
    
    with gr.Row():
        base_txt = gr.Dropdown(choices=STUDENT_CHOICES, value="Qwen/Qwen2.5-0.5B-Instruct", label="1. Select Base Model", allow_custom_value=True, scale=2)
        lora_dropdown = gr.Dropdown(choices=get_available_loras(), value="None", label="2. Select Concept Topography (LoRA)", allow_custom_value=True, scale=3)
        refresh_lora_btn = gr.Button("🔄 Refresh", scale=1)
        use_lora_chk = gr.Checkbox(label="Apply LoRA", value=True, scale=1)
        load_btn = gr.Button("🚀 Load Model", variant="primary", scale=2)
        
    refresh_lora_btn.click(refresh_lora_list, inputs=[], outputs=[lora_dropdown])
    
    load_status = gr.Textbox(label="System Status", interactive=False)
    load_btn.click(load_model_for_chat, inputs=[base_txt, lora_dropdown, use_lora_chk], outputs=[load_status])
    
    gr.Markdown("### Interaction Modes")
    with gr.Row():
        deepthink_chk = gr.Checkbox(label="🧠 Deepthink (Fractal Routing)", value=False, info="局所重力をキャンセルし、微細な概念突起を探索して思考を構築します。")
        lightweight_chk = gr.Checkbox(label="🍃 Lightweight (FAISS Edge Decoding)", value=False, info="出力語彙をFAISSで動的に絞り込み、計算量を極限まで削減します。（※語彙制限のトレードオフにより、ハルシネーションの可能性が高まります）")
        
    gr.ChatInterface(fn=chat_fn, additional_inputs=[deepthink_chk, lightweight_chk])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
