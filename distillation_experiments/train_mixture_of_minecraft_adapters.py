"""
Mixture of Adapters: Domain Splitter and Independent LoRA Trainer.
Splits the 168 unified Minecraft modules into 3 clean, orthogonal domains:
1. DataPack & Command Domain (`domain_datapack`)
2. Redstone Circuit & Physics Domain (`domain_redstone`)
3. Mod Development & Resource Pack Domain (`domain_modding`)

Trains 3 separate specialized LoRA adapters on GPU (bfloat16) to eliminate domain entanglement.
"""

import os
import sys
import json
import time
import torch

from topic_knowledge_pipeline import TopicLoRATrainer

sys.stdout.reconfigure(encoding='utf-8')

def categorize_module(mod: dict) -> str:
    title = mod["title"].lower()
    content = mod["content"].lower()
    cat = mod.get("category", "").lower()

    # Redstone domain
    if any(k in title for k in ["レッドストーン", "回路", "ピストン", "コンパレーター", "リピーター", "オブザーバー", "準接続", "redstone", "piston", "comparator", "quasi"]):
        return "domain_redstone"
    if any(k in content[:200] for k in ["レッドストーン回路", "準接続", "ピストン", "コンパレーター"]):
        return "domain_redstone"

    # Modding & Resource domain
    if any(k in title for k in ["fabric", "neoforge", "mixin", "リソースパック", "ブロック状態", "モデル", "resourcepack", "mod"]):
        return "domain_modding"
    if any(k in content[:200] for k in ["fabric", "neoforge", "mixin", "cullface", "pack.mcmeta"]):
        return "domain_modding"

    # Default to DataPack & Commands
    return "domain_datapack"

def main():
    print("=" * 80)
    print("STEP 1: DOMAIN SPLITTING (ELIMINATING DOMAIN ENTANGLEMENT)")
    print("=" * 80)

    corpus_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus/minecraft_unified_comprehensive_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    domains = {
        "domain_datapack": [],
        "domain_redstone": [],
        "domain_modding": []
    }

    for mod in corpus:
        dom = categorize_module(mod)
        domains[dom].append(mod)

    for dom_name, mods in domains.items():
        print(f"  - {dom_name}: {len(mods)} structured modules")

    # Save partitioned datasets
    base_dir = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_domains"
    os.makedirs(base_dir, exist_ok=True)

    for dom_name, mods in domains.items():
        dom_file = os.path.join(base_dir, f"{dom_name}.json")
        with open(dom_file, "w", encoding="utf-8") as f:
            json.dump(mods, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("STEP 2: TRAINING 3 ISOLATED SPECIALIZED LORA ADAPTERS")
    print("=" * 80)

    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_configs = [
        ("domain_datapack", "lora_adapter_datapack", 3),
        ("domain_redstone", "lora_adapter_redstone", 3),
        ("domain_modding", "lora_adapter_modding", 3)
    ]

    trainer = TopicLoRATrainer(base_model_name=base_model_name)

    for dom_name, adapter_folder, epochs in adapter_configs:
        print(f"\n>>> Training Specialized Adapter: {adapter_folder} ({dom_name}) <<<")
        mods = domains[dom_name]
        training_texts = [f"Title: {m['title']}\n\n{m['content']}" for m in mods]
        out_dir = os.path.join("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment", adapter_folder)

        start_t = time.time()
        trainer.train_topic_adapter(
            training_texts=training_texts,
            output_dir=out_dir,
            r=16,
            lora_alpha=32,
            epochs=epochs,
            lr=3e-4,
            batch_size=2
        )
        print(f">>> Completed {adapter_folder} in {time.time() - start_t:.2f}s <<<")

    print("\nAll 3 Domain-Specific Adapters trained successfully!")

if __name__ == "__main__":
    main()
