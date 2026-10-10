"""
【ファイル概要】
standalone_concept_distiller.py

外部のサーバーに依存せず、ローカル環境単独でベースモデルの概念地形を抽出し、高速に小規模モデルへ蒸留（Topological Distillation）するスクリプト。
"""

import torch
import json
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

class ConceptExtractionProcessor:
    @staticmethod
    def extract_and_save(model, tokenizer, concept_prompt, output_file="extracted_concept.json", top_k=100):
        print(f"\n[PHASE 1: EXTRACTION]")
        print(f"Extracting concept topography for: '{concept_prompt}'")
        inputs = tokenizer(concept_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            probs = torch.softmax(next_token_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            
            concept_mapping = {}
            for prob, idx in zip(top_probs, top_indices):
                token_str = tokenizer.decode([idx.item()])
                if token_str.strip() and len(token_str.strip()) > 1:
                    concept_mapping[token_str] = prob.item()

        max_prob = max(concept_mapping.values()) if concept_mapping else 1.0
        normalized_mapping = {k: v / max_prob for k, v in concept_mapping.items()}

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(normalized_mapping, f, ensure_ascii=False, indent=2)
            
        print(f"Extraction complete. Saved {len(normalized_mapping)} key tokens to '{output_file}'.")
        return normalized_mapping


class ExtractedTerrainProcessor(LogitsProcessor):
    def __init__(self, tokenizer, concept_file, viscosity=0.5, decay=0.5, alpha=5.0):
        self.viscosity = viscosity
        self.decay = decay
        self.alpha = alpha
        self.step = 0
        
        with open(concept_file, 'r', encoding='utf-8') as f:
            concept_mapping = json.load(f)
            
        vocab_size = tokenizer.vocab_size
        self.terrain_bias = torch.zeros(vocab_size)
        
        mapped_count = 0
        for token_str, weight in concept_mapping.items():
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                self.terrain_bias[token_ids[0]] += weight
                mapped_count += 1
                
        print(f"Loaded terrain bias. Successfully mapped {mapped_count} tokens to local vocabulary.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        current_power = math.exp(-self.decay * self.step * 0.1)
        viscosity_factor = 1.0 + (self.viscosity * 2.0)
        
        bias = self.terrain_bias.to(scores.device) * (self.alpha * current_power * viscosity_factor)
        scores = scores + bias
        
        return scores

if __name__ == "__main__":
    print("Standalone Concept Distiller Toolkit starting local test...\n")
    
    # ローカルのモデルをロード（抽出と適用の両方に同じモデルを使用）
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading local model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    
    # ==========================================
    # Phase 1: 抽出
    # ==========================================
    extract_prompt = "The absolute best practices of asynchronous python architecture:"
    output_json = "test_coding_concept.json"
    ConceptExtractionProcessor.extract_and_save(model, tokenizer, extract_prompt, output_json, top_k=100)
    
    # ==========================================
    # Phase 2: 適用
    # ==========================================
    print(f"\n[PHASE 2: PROJECTION]")
    processors = LogitsProcessorList()
    processors.append(ExtractedTerrainProcessor(
        tokenizer=tokenizer,
        concept_file=output_json,
        viscosity=0.8,
        decay=0.2,
        alpha=5.0
    ))
    
    user_task = "Write a high-performance web scraper script."
    print(f"User Task: '{user_task}'")
    
    # Qwen2.5用のプロンプトフォーマット
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_task}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print("Generating output with Concept Topography applied...\n")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, logits_processor=processors)
    
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("--- Output ---")
    print(result.strip())
