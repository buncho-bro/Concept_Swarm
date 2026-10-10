"""
【ファイル概要】
physics_core.py

概念ベクトルの引力・反発、ドーム状の多様体上での大円干渉（測地線計算）など、本プロジェクトの根幹となる高次元物理シミュレーションエンジン。
"""

import torch
import torch.nn.functional as F
import math
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

DOMAINS = {
    "Quantum Mechanics": "The abstract physical principles of Quantum Mechanics, superposition, entanglement, and probability.",
    "Evolution": "The biological principles of evolution, natural selection, adaptation, and genetic mutation.",
    "Fluid Dynamics": "The physical principles of fluid dynamics, liquid, turbulence, and continuous flow.",
    "Thermodynamics": "The laws of thermodynamics, entropy, heat transfer, and energy conservation.",
    "None": "General logical reasoning without specific domain bias."
}

class RippleTopographyProcessor(LogitsProcessor):
    def __init__(self, terrain_vec, vocab_emb, viscosity, decay, alpha=2.0):
        self.terrain_vec = F.normalize(terrain_vec, p=2, dim=-1) if terrain_vec is not None else None
        self.vocab_emb = F.normalize(vocab_emb, p=2, dim=-1)
        self.viscosity = viscosity
        self.decay = decay
        self.alpha = alpha
        self.step = 0
        
        if self.terrain_vec is not None:
            self.terrain_affinity = torch.matmul(self.vocab_emb, self.terrain_vec.T).squeeze()
        else:
            self.terrain_affinity = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        current_power = math.exp(-self.decay * self.step * 0.1)
        
        if self.terrain_affinity is not None:
            gravity_bias = self.terrain_affinity * (self.alpha * current_power)
            viscosity_factor = 1.0 + (self.viscosity * 2.0)
            gravity_bias = gravity_bias * viscosity_factor
            scores = scores + gravity_bias.to(scores.device)
            
        return scores

class PhysicsCoreEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vocab_embeddings = None
        self.current_model_id = None

    def load_model(self, model_id):
        if hasattr(self, 'model') and getattr(self, 'model') is not None:
            print(f"Unloading current model: {self.current_model_id}")
            del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            self.vocab_embeddings = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # 安全のために None で初期化しておく
        self.model = None
        self.tokenizer = None

        print(f"Loading Physics Engine core ({model_id})...")
        try:
            # LoRAパスかどうかの判定
            is_lora = "lora_" in model_id.lower()
            
            if is_lora:
                print("Detected LoRA configuration. Loading base model first...")
                # ファイル名からベースモデルを推測（デフォルトは Qwen2.5-1.5B-Instruct）
                base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
                if "0.5B" in model_id: base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                if "7B" in model_id: base_model_id = "Qwen/Qwen2.5-7B-Instruct"
                if "Coder" in model_id:
                    if "3B" in model_id:
                        base_model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
                    else:
                        base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
                
                print(f"Base model identified as: {base_model_id}")
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype=torch.float16)
                
                print("Applying LoRA weights...")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(base_model, model_id)
                # Vocab Embeddingsはベースモデルから取得
                self.vocab_embeddings = base_model.get_output_embeddings().weight.detach()
                
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
                self.vocab_embeddings = self.model.get_output_embeddings().weight.detach()
                
            self.current_model_id = model_id
            print(f"Model {model_id} loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load model {model_id}: {str(e)}")
            return False

    def get_terrain_vector(self, domain_name):
        if self.model is None or domain_name not in DOMAINS or domain_name == "None":
            return None
        text = DOMAINS[domain_name]
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1].mean(dim=1)

    def generate_with_physics(self, prompt, config):
        if self.model is None:
            return "Error: No model loaded."
            
        terrain_vec = self.get_terrain_vector(config.get("terrain_injection", "None"))
        viscosity = float(config.get("viscosity", 0.5))
        decay = float(config.get("decay", 0.5))
        
        processors = LogitsProcessorList()
        if terrain_vec is not None:
            processors.append(RippleTopographyProcessor(
                terrain_vec=terrain_vec,
                vocab_emb=self.vocab_embeddings,
                viscosity=viscosity,
                decay=decay,
                alpha=3.0
            ))
        
        sys_prompt = "You are the Ripple Physics Engine. You MUST respond in the EXACT SAME LANGUAGE as the user's task."
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        
        # モデルによって apply_chat_template が使えない場合のフォールバック
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"System: {sys_prompt}\nUser: {prompt}\nAssistant:"
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=500, 
                temperature=0.7, 
                do_sample=True, 
                logits_processor=processors,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
