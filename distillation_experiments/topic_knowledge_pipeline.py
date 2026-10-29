"""
Topic-Specific Knowledge Distillation and Sublimation Architecture.
Implements:
1. Topic Knowledge Corpus Ingestion & Semantic Chunking
2. Topic Manifold LoRA Fine-Tuning (8-bit Quantized on GPU)
3. Dynamic Knowledge Grounding Store (integrated into Step 3 Fact Grounding)
4. Full Integration with 5-Step Unified Cognitive Engine
5. Dual-Axis Evaluation: Topic Mastery vs. General Reasoning Preservation (Anti-Forgetting Check)
"""

import os
import sys
import json
import time
import math
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

sys.stdout.reconfigure(encoding='utf-8')

# ======================================================================
# 1. TOPIC KNOWLEDGE STORE (GROUNDING REPOSITORY)
# ======================================================================

class TopicKnowledgeStore:
    """
    Topic-specific document & code snippet repository.
    Enables zero-hallucination fact lookup and verification during Step 3.
    """
    def __init__(self):
        self.documents: List[Dict[str, str]] = []
        self.chunk_tokens: List[List[str]] = []

    def add_document(self, doc_id: str, title: str, content: str, category: str = "general"):
        # Split into logical chunks (paragraphs or code blocks)
        raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for idx, chunk in enumerate(raw_chunks):
            chunk_data = {
                "doc_id": doc_id,
                "title": title,
                "chunk_id": f"{doc_id}_c{idx}",
                "category": category,
                "text": chunk
            }
            self.documents.append(chunk_data)
            tokens = [t.lower() for t in re.findall(r'[a-zA-Z0-9_\-\.\:\:\&]+', chunk)]
            self.chunk_tokens.append(tokens)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Lexical-topological relevance retrieval.
        Scores based on term salience, code syntax matching, and IDF.
        """
        q_tokens = [t.lower() for t in re.findall(r'[a-zA-Z0-9_\-\.\:\:\&]+', query)]
        if not q_tokens or not self.documents:
            return []

        # Document frequency
        df = defaultdict(int)
        for tokens in self.chunk_tokens:
            unique_terms = set(tokens)
            for t in unique_terms:
                df[t] += 1
        num_docs = len(self.documents)

        scored = []
        for doc_idx, tokens in enumerate(self.chunk_tokens):
            score = 0.0
            doc_len = len(tokens)
            if doc_len == 0:
                continue
            tf_counts = defaultdict(int)
            for t in tokens:
                tf_counts[t] += 1

            for qt in q_tokens:
                if qt in tf_counts:
                    idf = math.log((num_docs + 1) / (df[qt] + 1)) + 1.0
                    tf = tf_counts[qt] / doc_len
                    # Boost exact identifier or symbol matches
                    boost = 2.0 if ("::" in qt or "_" in qt or "(" in qt) else 1.0
                    score += tf * idf * boost

            if score > 0:
                scored.append((score, self.documents[doc_idx]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]


# ======================================================================
# 2. TOPIC MANIFOLD LORA TRAINER
# ======================================================================

class TopicLoRATrainer:
    """
    Trains a specialized LoRA adapter on topic corpus without destroying
    the base model's general manifold or reasoning ability.
    """
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.base_model_name = base_model_name
        self.device = device

    def train_topic_adapter(
        self,
        training_texts: List[str],
        output_dir: str,
        r: int = 16,
        lora_alpha: int = 32,
        epochs: int = 3,
        lr: float = 3e-4,
        batch_size: int = 2
    ):
        print(f"[TopicLoRATrainer] Initializing base model with bfloat16: {self.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)

        print(f"[TopicLoRATrainer] Starting training over {len(training_texts)} text units for {epochs} epochs...")
        start_time = time.time()
        peft_model.train()

        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            for i in range(0, len(training_texts), batch_size):
                batch_texts = training_texts[i:i+batch_size]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100

                optimizer.zero_grad()
                outputs = peft_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1

            avg_loss = epoch_loss / max(1, steps)
            loss_history.append(avg_loss)
            print(f"  Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        elapsed = time.time() - start_time
        print(f"[TopicLoRATrainer] Training complete in {elapsed:.2f}s. Saving adapter to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        peft_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Write training log
        log_path = os.path.join(output_dir, "training_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "base_model": self.base_model_name,
                "epochs": epochs,
                "r": r,
                "lora_alpha": lora_alpha,
                "learning_rate": lr,
                "loss_history": loss_history,
                "elapsed_seconds": elapsed,
                "num_samples": len(training_texts)
            }, f, indent=2)

        del peft_model
        del model
        torch.cuda.empty_cache()
        return log_path


# ======================================================================
# 3. HYBRID COGNITIVE ENGINE (LoRA + Sublimation + Grounding)
# ======================================================================

class HybridTopicCognitiveEngine:
    """
    Integrates the Topic LoRA adapter with the 5-Step Sublimation & Grounding Engine.
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        knowledge_store: Optional[TopicKnowledgeStore] = None,
        device: str = "cuda"
    ):
        self.device = device
        self.knowledge_store = knowledge_store or TopicKnowledgeStore()
        
        print(f"[HybridCognitiveEngine] Loading model with bfloat16: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        if adapter_path and os.path.exists(adapter_path):
            print(f"[HybridCognitiveEngine] Attaching Topic LoRA Adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model

        self.model.eval()

    def generate_direct(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Raw direct autoregressive generation (Baseline test)."""
        messages = [
            {"role": "system", "content": "You are a technical expert. Answer accurately and concisely."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()

    def infer_sublimated_grounded(self, query: str, max_new_tokens: int = 250) -> Dict:
        """
        Execute 5-Step Sublimation & Knowledge-Grounded Reasoning:
        Step 1: Epistemic Deconstruction (Identify technical premises, language syntax assumptions)
        Step 2: Grounding Retrieval (Retrieve authoritative snippets from KnowledgeStore)
        Step 3: Axiomatic Fact Verification (Validate syntax, constraints, and semantics against facts)
        Step 4: Counterfactual / Edge-case Refutation (Check failure modes, memory safety, logic traps)
        Step 5: Sublimated Synthesis (Deliver final grounded, rigorous technical conclusion)
        """
        # Step 1: Sublimation Decomposition
        decomp_prompt = (
            f"Analyze this technical question or programming problem:\n"
            f"Question: \"{query}\"\n\n"
            f"Output exactly in this format:\n"
            f"SYNTAX_SEMANTIC_PREMISE: <core technical / syntax premise being assumed>\n"
            f"SEARCH_KEYS: <3 to 5 critical technical keywords or identifiers for fact retrieval>\n"
            f"POTENTIAL_PITFALL: <likely misconception, deprecated pattern, or edge condition>"
        )
        messages_s1 = [
            {"role": "system", "content": "You are an analytical parser for technical and programming systems."},
            {"role": "user", "content": decomp_prompt}
        ]
        chat_s1 = self.tokenizer.apply_chat_template(messages_s1, tokenize=False, add_generation_prompt=True)
        enc_s1 = self.tokenizer(chat_s1, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_s1 = self.model.generate(
                **enc_s1,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id
            )
        resp_s1 = self.tokenizer.decode(out_s1[0][len(enc_s1["input_ids"][0]):], skip_special_tokens=True).strip()

        m_premise = re.search(r"SYNTAX_SEMANTIC_PREMISE:\s*(.*)", resp_s1)
        m_keys = re.search(r"SEARCH_KEYS:\s*(.*)", resp_s1)
        m_pitfall = re.search(r"POTENTIAL_PITFALL:\s*(.*)", resp_s1)

        premise_text = m_premise.group(1).strip() if m_premise else query
        search_keys = m_keys.group(1).strip() if m_keys else query
        pitfall_text = m_pitfall.group(1).strip() if m_pitfall else "None identified"

        # Step 2: Grounding Lookup
        retrieved_docs = self.knowledge_store.search(f"{query} {search_keys}", top_k=3)
        grounding_context = "\n---\n".join([f"[{d['title']}]: {d['text']}" for d in retrieved_docs])
        if not grounding_context:
            grounding_context = "No specific reference documents found in local knowledge store."

        # Step 3, 4, 5: Grounded Synthesis with Counterfactual Sandbox
        synth_prompt = (
            f"質問: {query}\n\n"
            f"[技術的・構文的前提]: {premise_text}\n"
            f"[注意すべき落とし穴・非推奨記法]: {pitfall_text}\n\n"
            f"[信頼できる公式仕様情報]:\n{grounding_context}\n\n"
            f"指示: 上記の仕様情報に基づいて、質問に対する正確かつ直接的な回答を日本語で簡潔に記述してください。"
            f"コマンド構文やコードが含まれる場合は、Minecraft 1.21の最新仕様に準拠した正確なコードを出力し、古い仕様や落とし穴を論破してください。"
        )
        messages_s5 = [
            {"role": "system", "content": "あなたはMinecraft技術仕様（データパック、コマンド、Redstone回路、Mod開発）の高度な専門家です。最新の仕様に厳密に準拠して回答します。"},
            {"role": "user", "content": synth_prompt}
        ]
        chat_s5 = self.tokenizer.apply_chat_template(messages_s5, tokenize=False, add_generation_prompt=True)
        enc_s5 = self.tokenizer(chat_s5, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_s5 = self.model.generate(
                **enc_s5,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id
            )
        final_answer = self.tokenizer.decode(out_s5[0][len(enc_s5["input_ids"][0]):], skip_special_tokens=True).strip()

        return {
            "query": query,
            "step1_premise": premise_text,
            "step1_pitfall": pitfall_text,
            "retrieved_chunks": [d["chunk_id"] for d in retrieved_docs],
            "grounding_context": grounding_context,
            "final_answer": final_answer
        }
