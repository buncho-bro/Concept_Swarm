"""
Pure Sublimated Grounding Engine Test (Zero-LoRA Distortion).
Tests the frozen base model (Qwen2.5-1.5B-Instruct) directly coupled with
the 168-module authoritative Minecraft Knowledge Store using the 5-Step Sublimated Cognitive Architecture.
"""

import os
import sys
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from topic_knowledge_pipeline import TopicKnowledgeStore

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 80)
    print("TESTING PURE SUBLIMATED GROUNDING ENGINE (ZERO LORA DISTORTION)")
    print("=" * 80)

    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Load 168 unified modules
    corpus_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus/minecraft_unified_comprehensive_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    store = TopicKnowledgeStore()
    for m in corpus:
        store.add_document(m["doc_id"], m["title"], m["content"], m.get("category", "general"))

    print(f"Loaded {len(store.documents)} chunks into Knowledge Store.")

    test_queries = [
        # Query 1: DataPack (execute command)
        "最新のexecuteコマンドを使って、半径10ブロック以内にいる最も近いゾンビの足元に雷を落とす正確なコマンド構文を教えてください。",

        # Query 2: DataPack (1.21+ Item Components)
        "Minecraft 1.21において、鋭さ5のダイヤモンドの剣をプレイヤーに付与し、カスタムデータとして spell_tier:3 を埋め込む最新のコマンド構文を教えてください。従来のNBT compound記法は使えますか？",

        # Query 3: Redstone (Quasi-Connectivity & Sticky Piston)
        "Java版における準接続（Quasi-Connectivity）の仕組みと、粘着ピストンに1レッドストーンティック（2ゲームティック）のパルスを入力したときに起きる現象（ブロック吐き出し/T-Flip-Flop）について説明してください。",

        # Query 4: Modding (Fabric Mixin)
        "Fabric mod開発において、プレイヤーのtickメソッド（ServerPlayer.tick）の先頭に処理を割り込み、特定条件でイベントをキャンセルするMixinコードをJavaで記述してください。"
    ]

    results = []
    for q in test_queries:
        print(f"\n" + "=" * 60)
        print(f"[Query]: {q}")

        # Step 1: Deconstruction
        decomp_prompt = (
            f"Analyze this technical query:\n"
            f"Question: \"{q}\"\n\n"
            f"Output in this exact format:\n"
            f"SEARCH_KEYS: <3 to 5 critical technical keywords or command names>\n"
            f"PITFALL: <likely pitfall or deprecated usage to avoid>"
        )
        enc_s1 = tokenizer(decomp_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_s1 = model.generate(**enc_s1, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        resp_s1 = tokenizer.decode(out_s1[0][len(enc_s1["input_ids"][0]):], skip_special_tokens=True).strip()

        m_keys = re.search(r"SEARCH_KEYS:\s*(.*)", resp_s1)
        search_keys = m_keys.group(1).strip() if m_keys else q

        # Step 2: Grounding Lookup
        retrieved = store.search(f"{q} {search_keys}", top_k=3)
        context = "\n---\n".join([f"[{d['title']}]: {d['text']}" for d in retrieved])

        # Step 3, 4, 5: Grounded Synthesis
        synth_prompt = (
            f"質問: {q}\n\n"
            f"[公式技術仕様]:\n{context}\n\n"
            f"指示: 上記の公式仕様に基づいて、正確かつ直接的な回答を日本語で簡潔に出力してください。"
            f"コマンドやコードが含まれる場合は、正確な構文をそのままコードブロックで記述してください。"
        )
        messages = [
            {"role": "system", "content": "あなたはMinecraft技術仕様（コマンド、データパック、レッドストーン回路、Mod開発）の正確な専門家です。公式仕様に基づいて正確に回答します。"},
            {"role": "user", "content": synth_prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=300,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )
        ans = tokenizer.decode(out[0][len(enc["input_ids"][0]):], skip_special_tokens=True).strip()
        print(f"[Answer]:\n{ans}")
        results.append({"query": q, "retrieved": [d["chunk_id"] for d in retrieved], "answer": ans})

    out_file = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/pure_sublimated_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()
