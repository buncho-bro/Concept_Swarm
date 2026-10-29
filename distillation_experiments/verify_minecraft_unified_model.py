"""
Verification Script for the Unified Minecraft Knowledge Model.
Tests the newly trained `lora_minecraft_unified_adapter` integrated with
the 168-module `TopicKnowledgeStore` using the 5-Step Sublimated Cognitive Engine.
"""

import os
import sys
import json
import torch

from topic_knowledge_pipeline import (
    TopicKnowledgeStore,
    HybridTopicCognitiveEngine
)

sys.stdout.reconfigure(encoding='utf-8')

VERIFICATION_QUERIES = [
    {
        "id": "q1_components_ja",
        "query": "Minecraft 1.21において、鋭さ5のダイヤモンドの剣をプレイヤーに付与し、カスタムデータとして spell_tier:3 を埋め込む最新のコマンド構文を教えてください。従来のNBT compound記法は使えますか？"
    },
    {
        "id": "q2_redstone_qc_spit_ja",
        "query": "Java版における準接続（Quasi-Connectivity）の仕組みと、粘着ピストンに1レッドストーンティック（2ゲームティック）のパルスを入力したときに起きる現象（ブロック吐き出し/T-Flip-Flop）について説明してください。"
    },
    {
        "id": "q3_execute_command_ja",
        "query": "最新のexecuteコマンドを使って、半径10ブロック以内にいる最も近いゾンビの足元に雷を落とす正確なコマンド構文を教えてください。"
    }
]

def main():
    print("=" * 80)
    print("VERIFYING UNIFIED MINECRAFT TECHNICAL REASONING ENGINE")
    print("=" * 80)

    corpus_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus/minecraft_unified_comprehensive_corpus.json"
    adapter_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/lora_minecraft_unified_adapter"
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. Initialize Store with 168 modules
    print("\n--- Step 1: Loading 168 modules into TopicKnowledgeStore ---")
    store = TopicKnowledgeStore()
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    for m in corpus:
        store.add_document(m["doc_id"], m["title"], m["content"], m.get("category", "general"))
    print(f"Store loaded with {len(store.documents)} chunked modules.")

    # 2. Initialize Hybrid Cognitive Engine with Adapter
    print("\n--- Step 2: Initializing Hybrid Cognitive Engine ---")
    engine = HybridTopicCognitiveEngine(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        knowledge_store=store
    )

    # 3. Execute Inferences
    print("\n--- Step 3: Executing Grounded Technical Inferences ---")
    results = []
    for item in VERIFICATION_QUERIES:
        print(f"\n[Query]: {item['query']}")
        res = engine.infer_sublimated_grounded(item["query"], max_new_tokens=300)
        print(f"\n[Step 1 Deconstructed Premise]: {res['step1_premise']}")
        print(f"[Step 1 Pitfall]: {res['step1_pitfall']}")
        print(f"[Retrieved Modules]: {res['retrieved_chunks']}")
        print(f"\n[Final Sublimated Answer]:\n{res['final_answer']}")
        print("-" * 60)
        results.append({
            "id": item["id"],
            "query": item["query"],
            "step1_premise": res["step1_premise"],
            "step1_pitfall": res["step1_pitfall"],
            "retrieved_chunks": res["retrieved_chunks"],
            "final_answer": res["final_answer"]
        })

    # Save output
    out_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/unified_minecraft_verification_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nVerification results saved to: {out_path}")

if __name__ == "__main__":
    main()
