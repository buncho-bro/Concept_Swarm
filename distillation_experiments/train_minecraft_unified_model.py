"""
Train Unified Minecraft Technical Knowledge Model on GPU (bfloat16).
Uses 168 structured technical modules from ja.minecraft.wiki and modern specifications.
"""

import os
import sys
import json
import time
import torch

from topic_knowledge_pipeline import (
    TopicKnowledgeStore,
    TopicLoRATrainer,
    HybridTopicCognitiveEngine
)

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 80)
    print("TRAINING UNIFIED MINECRAFT TECHNICAL KNOWLEDGE MODEL (GPU bfloat16)")
    print("=" * 80)

    unified_corpus_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus/minecraft_unified_comprehensive_corpus.json"
    adapter_dir = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/lora_minecraft_unified_adapter"
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    with open(unified_corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} technical modules.")
    training_texts = [f"Title: {m['title']}\n\n{m['content']}" for m in corpus]

    trainer = TopicLoRATrainer(base_model_name=base_model_name)
    # Train over the unified corpus
    trainer.train_topic_adapter(
        training_texts=training_texts,
        output_dir=adapter_dir,
        r=16,
        lora_alpha=32,
        epochs=2,
        lr=3e-4,
        batch_size=2
    )

    print(f"\nUnified Minecraft LoRA successfully trained and saved to: {adapter_dir}")

if __name__ == "__main__":
    main()
