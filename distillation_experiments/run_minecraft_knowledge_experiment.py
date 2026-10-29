"""
Minecraft Technical Knowledge Benchmark & Verification Harness.
Evaluates:
1. Data Pack Mastery (1.20.5+/1.21+ Item Components, Macro Functions)
2. Resource Pack Mastery (Blockstates, UV/Cullface, Atlas sheets)
3. Mod Development Mastery (Fabric Mixins, 1.20.5+ Item.Settings, NeoForge Capabilities)
4. Redstone Circuit Engineering (Quasi-Connectivity, 0-Tick Block Spitting, Comparator Math)
5. General Logic Retention (ProntoQA)

Across 4 Modalities:
- Modality 1: Raw Base 1.5B
- Modality 2: Minecraft LoRA Alone
- Modality 3: Grounding Alone (RAG)
- Modality 4: Hybrid Sublimated Engine (LoRA + 5-Step Sublimation + Grounding)
"""

import os
import sys
import json
import time
from typing import List, Dict, Tuple, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np

from topic_knowledge_pipeline import (
    TopicKnowledgeStore,
    TopicLoRATrainer,
    HybridTopicCognitiveEngine
)

sys.stdout.reconfigure(encoding='utf-8')

# ======================================================================
# MINECRAFT EVALUATION BENCHMARK SUITE
# ======================================================================

MINECRAFT_BENCHMARK = [
    {
        "id": "mc_q1_datapack_components",
        "domain": "datapack",
        "question": "In Minecraft 1.21+, how do you give a player a diamond sword with sharpness level 5 and custom metadata spell_tier:3? Does the old NBT compound syntax still work?",
        "key_facts": ["item components", "square brackets", "custom_data", "enchantments={levels:", "old nbt removed"]
    },
    {
        "id": "mc_q2_datapack_macros",
        "domain": "datapack",
        "question": "How do macro functions work in Minecraft 1.20.2+ mcfunction files? What symbol denotes a macro line, and how are parameters passed to it?",
        "key_facts": ["$", "$(variable)", "function command with", "storage or compound"]
    },
    {
        "id": "mc_q3_resourcepack_cullface",
        "domain": "resourcepack",
        "question": "In resource pack block models, what is the exact performance purpose of 'cullface', and when does the GPU cull the face?",
        "key_facts": ["adjacent opaque block", "cullface", "draw call", "gpu optimization"]
    },
    {
        "id": "mc_q4_mod_mixin_fabric",
        "domain": "mod_development",
        "question": "In modern Fabric modding, how do you cancel a player tick event using SpongePowered Mixin? What class was removed in 1.20.5 in favor of Item.Settings?",
        "key_facts": ["@Inject", "at = @At(\"HEAD\")", "cancellable = true", "ci.cancel()", "FabricItemSettings removed"]
    },
    {
        "id": "mc_q5_redstone_qc_spit",
        "domain": "redstone_circuit",
        "question": "Explain Quasi-Connectivity (QC) in Java Edition redstone. What happens when a sticky piston receives a 1-redstone-tick pulse?",
        "key_facts": ["2-block tall door", "block above powered", "needs block update", "BUD", "spits block", "leaves block behind"]
    },
    {
        "id": "mc_q6_redstone_comparator_math",
        "domain": "redstone_circuit",
        "question": "What is the mathematical output of a redstone comparator in Subtraction Mode, and what signal strength does an empty container emit?",
        "key_facts": ["rear input minus side input", "front torch lit", "max(0, rear - side)", "empty container emits 0"]
    }
]

GENERAL_LOGIC_BENCHMARK = [
    {
        "id": "logic_q1",
        "context": "Each cat is a mammal. Each mammal is an animal. Mittens is a cat.",
        "question": "Is Mittens an animal?",
        "expected": True
    },
    {
        "id": "logic_q2",
        "context": "All birds can fly. Tweety is a bird. Penguins are birds. Every penguin cannot fly.",
        "question": "Does Tweety have the ability to fly?",
        "expected": True
    },
    {
        "id": "logic_q3",
        "context": "No carnivore eats grass. A lion is a carnivore. Simba is a lion.",
        "question": "Does Simba eat grass?",
        "expected": False
    },
    {
        "id": "logic_q4",
        "context": "All integers are real numbers. All real numbers are complex numbers. Seven is an integer.",
        "question": "Is Seven a complex number?",
        "expected": True
    },
    {
        "id": "logic_q5",
        "context": "Every reptile is cold-blooded. No mammal is cold-blooded. A dog is a mammal. Buster is a dog.",
        "question": "Is Buster a reptile?",
        "expected": False
    }
]


def evaluate_topic_answers(answers: List[str], benchmark: List[Dict]) -> Tuple[float, List[float]]:
    scores = []
    for ans, item in zip(answers, benchmark):
        ans_lower = ans.lower()
        matched = sum(1 for kf in item["key_facts"] if kf.lower() in ans_lower)
        fact_ratio = matched / len(item["key_facts"])
        scores.append(fact_ratio)
    avg_score = float(np.mean(scores)) * 100.0 if scores else 0.0
    return avg_score, scores


def evaluate_general_logic(model_engine, benchmark: List[Dict]) -> float:
    correct = 0
    for item in benchmark:
        prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer strictly 'Yes' or 'No'."
        resp = model_engine.generate_direct(prompt, max_new_tokens=20).lower()
        pred_yes = ("yes" in resp and "no" not in resp) or ("is an animal" in resp) or ("can fly" in resp) or ("is a complex" in resp)
        pred_no = ("no" in resp and "yes" not in resp) or ("does not" in resp) or ("cannot" in resp) or ("is not" in resp)

        expected = item["expected"]
        if expected and pred_yes:
            correct += 1
        elif not expected and pred_no:
            correct += 1
        elif expected and not pred_no and "yes" in resp:
            correct += 1
        elif not expected and not pred_yes and "no" in resp:
            correct += 1

    return (correct / len(benchmark)) * 100.0


def main():
    print("=" * 80)
    print("STARTING MINECRAFT TECHNICAL KNOWLEDGE (DATAPACK/MOD/RS) EXPERIMENT")
    print("=" * 80)

    corpus_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus/minecraft_technical_knowledge_corpus.json"
    adapter_dir = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/lora_minecraft_technical_adapter"
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. Load Corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    store = TopicKnowledgeStore()
    training_texts = []
    for doc in corpus:
        store.add_document(doc["doc_id"], doc["title"], doc["content"], doc["category"])
        training_texts.append(f"Title: {doc['title']}\nCategory: {doc['category']}\n\n{doc['content']}")

    print(f"Store loaded with {len(store.documents)} chunks across {len(corpus)} technical modules.")

    # 2. Train Minecraft LoRA Adapter
    print("\n--- Step 2: Training Minecraft Technical LoRA Adapter on GPU (bfloat16) ---")
    trainer = TopicLoRATrainer(base_model_name=base_model_name)
    trainer.train_topic_adapter(
        training_texts=training_texts,
        output_dir=adapter_dir,
        r=16,
        lora_alpha=32,
        epochs=3,
        lr=3e-4,
        batch_size=1
    )

    # 3. Benchmark Modalities
    print("\n--- Step 3: Benchmarking 4 Modalities across Minecraft & Logic Retention ---")

    # Modality 1: Raw Base 1.5B
    print("\n[Evaluating Modality 1: Raw Base 1.5B]")
    engine_raw = HybridTopicCognitiveEngine(base_model_name=base_model_name, adapter_path=None, knowledge_store=None)
    raw_topic_answers = [engine_raw.generate_direct(item["question"]) for item in MINECRAFT_BENCHMARK]
    raw_topic_acc, raw_topic_details = evaluate_topic_answers(raw_topic_answers, MINECRAFT_BENCHMARK)
    raw_logic_acc = evaluate_general_logic(engine_raw, GENERAL_LOGIC_BENCHMARK)
    print(f"  Modality 1: Topic Accuracy = {raw_topic_acc:.2f}%, Logic Retention = {raw_logic_acc:.2f}%")
    del engine_raw
    torch.cuda.empty_cache()

    # Modality 2: Minecraft LoRA Alone
    print("\n[Evaluating Modality 2: Minecraft LoRA Alone]")
    engine_lora = HybridTopicCognitiveEngine(base_model_name=base_model_name, adapter_path=adapter_dir, knowledge_store=None)
    lora_topic_answers = [engine_lora.generate_direct(item["question"]) for item in MINECRAFT_BENCHMARK]
    lora_topic_acc, lora_topic_details = evaluate_topic_answers(lora_topic_answers, MINECRAFT_BENCHMARK)
    lora_logic_acc = evaluate_general_logic(engine_lora, GENERAL_LOGIC_BENCHMARK)
    print(f"  Modality 2: Topic Accuracy = {lora_topic_acc:.2f}%, Logic Retention = {lora_logic_acc:.2f}%")
    del engine_lora
    torch.cuda.empty_cache()

    # Modality 3: Grounding Alone (RAG)
    print("\n[Evaluating Modality 3: Grounding Alone (RAG without LoRA)]")
    engine_rag = HybridTopicCognitiveEngine(base_model_name=base_model_name, adapter_path=None, knowledge_store=store)
    rag_topic_answers = []
    for item in MINECRAFT_BENCHMARK:
        res = engine_rag.infer_sublimated_grounded(item["question"])
        rag_topic_answers.append(res["final_answer"])
    rag_topic_acc, rag_topic_details = evaluate_topic_answers(rag_topic_answers, MINECRAFT_BENCHMARK)
    rag_logic_acc = evaluate_general_logic(engine_rag, GENERAL_LOGIC_BENCHMARK)
    print(f"  Modality 3: Topic Accuracy = {rag_topic_acc:.2f}%, Logic Retention = {rag_logic_acc:.2f}%")
    del engine_rag
    torch.cuda.empty_cache()

    # Modality 4: Hybrid Sublimated Engine
    print("\n[Evaluating Modality 4: Hybrid Sublimated Engine (Proposed)]")
    engine_hybrid = HybridTopicCognitiveEngine(base_model_name=base_model_name, adapter_path=adapter_dir, knowledge_store=store)
    hybrid_topic_answers = []
    for item in MINECRAFT_BENCHMARK:
        res = engine_hybrid.infer_sublimated_grounded(item["question"])
        hybrid_topic_answers.append(res["final_answer"])
    hybrid_topic_acc, hybrid_topic_details = evaluate_topic_answers(hybrid_topic_answers, MINECRAFT_BENCHMARK)
    hybrid_logic_acc = evaluate_general_logic(engine_hybrid, GENERAL_LOGIC_BENCHMARK)
    print(f"  Modality 4: Topic Accuracy = {hybrid_topic_acc:.2f}%, Logic Retention = {hybrid_logic_acc:.2f}%")

    # 4. Save Results
    results_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_experiment_results.json"
    comparison_data = {
        "modalities": [
            "Modality 1: Raw Base 1.5B",
            "Modality 2: Minecraft LoRA Alone",
            "Modality 3: Grounding Alone (RAG)",
            "Modality 4: Hybrid Sublimated System"
        ],
        "topic_accuracy": [raw_topic_acc, lora_topic_acc, rag_topic_acc, hybrid_topic_acc],
        "logic_retention": [raw_logic_acc, lora_logic_acc, rag_logic_acc, hybrid_logic_acc],
        "detailed_topic_scores": {
            "raw": raw_topic_details,
            "lora": lora_topic_details,
            "rag": rag_topic_details,
            "hybrid": hybrid_topic_details
        },
        "sample_answers": {
            "q1_components": {
                "question": MINECRAFT_BENCHMARK[0]["question"],
                "raw": raw_topic_answers[0][:350],
                "lora": lora_topic_answers[0][:350],
                "rag": rag_topic_answers[0][:350],
                "hybrid": hybrid_topic_answers[0][:350]
            },
            "q5_redstone_qc": {
                "question": MINECRAFT_BENCHMARK[4]["question"],
                "raw": raw_topic_answers[4][:350],
                "lora": lora_topic_answers[4][:350],
                "rag": rag_topic_answers[4][:350],
                "hybrid": hybrid_topic_answers[4][:350]
            }
        }
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # 5. Chart
    chart_path = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/minecraft_knowledge_benchmark_comparison.png"
    plt.figure(figsize=(10, 6))
    x = np.arange(len(comparison_data["modalities"]))
    width = 0.35

    bars1 = plt.bar(x - width/2, comparison_data["topic_accuracy"], width, label="Minecraft Knowledge Accuracy (%)", color="#D32F2F")
    bars2 = plt.bar(x + width/2, comparison_data["logic_retention"], width, label="General Logic Retention (%)", color="#388E3C")

    plt.ylabel("Score (%)", fontsize=12)
    plt.title("Minecraft Domain Knowledge (DataPack/Mod/RS) vs Logic Retention", fontsize=14, fontweight="bold")
    plt.xticks(x, [m.replace("Modality ", "M").replace(": ", "\n") for m in comparison_data["modalities"]], fontsize=10)
    plt.ylim(0, 115)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight="bold")

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
