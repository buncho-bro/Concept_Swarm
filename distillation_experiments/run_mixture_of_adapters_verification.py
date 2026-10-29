"""
Mixture of Minecraft Adapters: Dynamic Routing & Sublimated Inference Engine.
Architecture:
1. Router: Deconstructs query and determines the target domain:
   - `domain_datapack`: Routes to `lora_adapter_datapack` + DataPack Knowledge Store
   - `domain_redstone`: Routes to `lora_adapter_redstone` + Redstone Knowledge Store
   - `domain_modding`: Routes to `lora_adapter_modding` + Modding Knowledge Store
2. Dynamic Adapter Switcher: Loads or activates the target LoRA adapter on-the-fly.
3. 5-Step Sublimated Grounding: Executes rigorous premise checks and generates clean, unentangled outputs.
"""

import os
import sys
import json
import re
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from topic_knowledge_pipeline import TopicKnowledgeStore

sys.stdout.reconfigure(encoding='utf-8')

class MixtureOfMinecraftAdaptersEngine:
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.device = device
        print(f"[MoA Engine] Initializing base model with bfloat16: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        base_dir = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
        self.adapter_paths = {
            "domain_datapack": os.path.join(base_dir, "lora_adapter_datapack"),
            "domain_redstone": os.path.join(base_dir, "lora_adapter_redstone"),
            "domain_modding": os.path.join(base_dir, "lora_adapter_modding")
        }

        # Initialize 3 isolated Knowledge Stores
        print("[MoA Engine] Populating 3 isolated domain knowledge stores...")
        self.domain_stores: Dict[str, TopicKnowledgeStore] = {
            "domain_datapack": TopicKnowledgeStore(),
            "domain_redstone": TopicKnowledgeStore(),
            "domain_modding": TopicKnowledgeStore()
        }

        domains_dir = os.path.join(base_dir, "minecraft_domains")
        for dom in ["domain_datapack", "domain_redstone", "domain_modding"]:
            fpath = os.path.join(domains_dir, f"{dom}.json")
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    mods = json.load(f)
                for m in mods:
                    self.domain_stores[dom].add_document(m["doc_id"], m["title"], m["content"], dom)
                print(f"  - {dom}: {len(self.domain_stores[dom].documents)} chunks indexed.")

        # Load first adapter and register others
        first_dom = "domain_datapack"
        print(f"[MoA Engine] Loading primary PEFT model with {first_dom} adapter...")
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            self.adapter_paths[first_dom],
            adapter_name=first_dom
        )

        # Attach the remaining adapters
        for dom in ["domain_redstone", "domain_modding"]:
            print(f"[MoA Engine] Attaching secondary adapter: {dom} ...")
            self.peft_model.load_adapter(self.adapter_paths[dom], adapter_name=dom)

        self.peft_model.eval()
        self.current_adapter = first_dom

    def route_query_domain(self, query: str) -> str:
        q_lower = query.lower()
        # Redstone cues
        if any(k in q_lower for k in ["レッドストーン", "回路", "ピストン", "コンパレーター", "リピーター", "オブザーバー", "準接続", "redstone", "piston", "comparator", "qc", "tick", "ティック"]):
            return "domain_redstone"
        # Modding / Resource cues
        if any(k in q_lower for k in ["fabric", "neoforge", "mixin", "mod", "モッド", "リソースパック", "ブロック状態", "モデル", "java", "settings", "cullface"]):
            return "domain_modding"
        # Default to DataPack / Command
        return "domain_datapack"

    def set_active_domain(self, target_domain: str):
        if self.current_adapter != target_domain:
            self.peft_model.set_adapter(target_domain)
            self.current_adapter = target_domain

    def infer_sublimated(self, query: str) -> Dict:
        # Step 1: Dynamic Domain Routing
        target_domain = self.route_query_domain(query)
        self.set_active_domain(target_domain)
        store = self.domain_stores[target_domain]

        # Step 2: Epistemic Premise Deconstruction
        decomp_prompt = (
            f"Analyze this Minecraft technical inquiry:\n"
            f"Query: \"{query}\"\n"
            f"Domain: {target_domain}\n\n"
            f"Output in this exact format:\n"
            f"SYNTAX_SEMANTIC_PREMISE: <core technical premise and required syntax format (command vs code)>\n"
            f"SEARCH_KEYS: <3 to 5 critical identifiers or technical terms>\n"
            f"POTENTIAL_PITFALL: <obsolete syntax (e.g. pre-1.21 NBT) or domain confusion to strictly avoid>"
        )
        messages_s1 = [
            {"role": "system", "content": f"You are a specialized parser for Minecraft {target_domain}."},
            {"role": "user", "content": decomp_prompt}
        ]
        chat_s1 = self.tokenizer.apply_chat_template(messages_s1, tokenize=False, add_generation_prompt=True)
        enc_s1 = self.tokenizer(chat_s1, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_s1 = self.peft_model.generate(
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
        pitfall_text = m_pitfall.group(1).strip() if m_pitfall else "None"

        # Step 3: Domain-Isolated Knowledge Retrieval
        retrieved_docs = store.search(f"{query} {search_keys}", top_k=3)
        grounding_context = "\n---\n".join([f"[{d['title']}]: {d['text']}" for d in retrieved_docs])
        if not grounding_context:
            grounding_context = "No direct documentation found."

        # Step 4 & 5: Sublimated Synthesis with Zero Cross-Domain Contamination
        domain_instruction = {
            "domain_datapack": "Provide purely Minecraft command / mcfunction syntax (or JSON). Do NOT output Java code.",
            "domain_redstone": "Explain the physical mechanics, block updates, signal strength, and tick timings clearly in Japanese. Avoid unrelated programming code.",
            "domain_modding": "Provide standard, idiomatic Java/Kotlin Modding code (Fabric/NeoForge) or Resource Pack JSON."
        }[target_domain]

        synth_prompt = (
            f"質問: {query}\n\n"
            f"[対象技術ドメイン]: {target_domain}\n"
            f"[構文・技術的前提]: {premise_text}\n"
            f"[排除すべき落とし穴]: {pitfall_text}\n\n"
            f"[公式技術仕様（グラウンディング資料）]:\n{grounding_context}\n\n"
            f"指示: {domain_instruction} 質問に対する正確で直接的な回答を日本語で簡潔に記述してください。"
        )

        messages_s5 = [
            {"role": "system", "content": f"あなたはMinecraft {target_domain} の専任技術エキスパートです。ドメイン混濁を排し、厳密な回答を行います。"},
            {"role": "user", "content": synth_prompt}
        ]
        chat_s5 = self.tokenizer.apply_chat_template(messages_s5, tokenize=False, add_generation_prompt=True)
        enc_s5 = self.tokenizer(chat_s5, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_s5 = self.peft_model.generate(
                **enc_s5,
                max_new_tokens=250,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id
            )
        final_answer = self.tokenizer.decode(out_s5[0][len(enc_s5["input_ids"][0]):], skip_special_tokens=True).strip()

        return {
            "query": query,
            "target_domain": target_domain,
            "premise": premise_text,
            "pitfall": pitfall_text,
            "retrieved_chunks": [d["chunk_id"] for d in retrieved_docs],
            "final_answer": final_answer
        }


def main():
    print("=" * 80)
    print("TESTING MIXTURE OF MINECRAFT ADAPTERS (MoA) ENGINE")
    print("=" * 80)

    engine = MixtureOfMinecraftAdaptersEngine()

    test_queries = [
        # Query 1: DataPack (Command execution) - Previous failure: Output Java code instead of command!
        "最新のexecuteコマンドを使って、半径10ブロック以内にいる最も近いゾンビの足元に雷を落とす正確なコマンド構文を教えてください。",
        
        # Query 2: DataPack (1.21+ components)
        "Minecraft 1.21において、鋭さ5のダイヤモンドの剣をプレイヤーに付与し、カスタムデータとして spell_tier:3 を埋め込む最新のコマンド構文を教えてください。従来のNBT compound記法は使えますか？",
        
        # Query 3: Redstone (QC and 1-tick block spit) - Previous failure: Invented fake words like 粘着ピトニックス
        "Java版における準接続（Quasi-Connectivity）の仕組みと、粘着ピストンに1レッドストーンティック（2ゲームティック）のパルスを入力したときに起きる現象（ブロック吐き出し/T-Flip-Flop）について説明してください。",

        # Query 4: Modding (Fabric Mixin) - Java code expected here
        "Fabric mod開発において、プレイヤーのtickメソッド（ServerPlayer.tick）の先頭に処理を割り込み、特定条件でイベントをキャンセルするMixinコードをJavaで記述してください。"
    ]

    results = []
    for q in test_queries:
        print(f"\n" + "=" * 60)
        print(f"[Query]: {q}")
        res = engine.infer_sublimated(q)
        print(f"[Routed Domain]: {res['target_domain']}")
        print(f"[Premise]: {res['premise']}")
        print(f"[Pitfall Refuted]: {res['pitfall']}")
        print(f"[Answer]:\n{res['final_answer']}")
        results.append(res)

    out_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/moa_verification_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMoA Verification completed and saved to {out_path}")


if __name__ == "__main__":
    main()
