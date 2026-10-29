"""
Corpus Merger & Semantic Chunking for Minecraft Technical Domain.
Merges:
1. ja_minecraft_wiki_corpus.json (86,000+ chars from ja.minecraft.wiki)
2. minecraft_technical_knowledge_corpus.json (modern 1.21+ components, Fabric/NeoForge modding)

Produces:
minecraft_corpus/minecraft_unified_comprehensive_corpus.json
Ready for high-density TopicKnowledgeStore and LoRA fine-tuning.
"""

import os
import sys
import json
import re

sys.stdout.reconfigure(encoding='utf-8')

def main():
    corpus_dir = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus"
    wiki_path = os.path.join(corpus_dir, "ja_minecraft_wiki_corpus.json")
    spec_path = os.path.join(corpus_dir, "minecraft_technical_knowledge_corpus.json")
    unified_path = os.path.join(corpus_dir, "minecraft_unified_comprehensive_corpus.json")

    unified_modules = []

    # 1. Load curated modern 1.21 & modding specs
    if os.path.exists(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            specs = json.load(f)
            unified_modules.extend(specs)

    # 2. Load harvested Wiki articles
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki_articles = json.load(f)

        for art in wiki_articles:
            # Chunk large articles by headings (== ... ==)
            sections = re.split(r'\n={2,4}\s*(.*?)\s*={2,4}\n', art["content"])
            if len(sections) > 1:
                intro = sections[0].strip()
                if intro:
                    unified_modules.append({
                        "doc_id": f"{art['doc_id']}_intro",
                        "title": f"{art['title']} (概要)",
                        "category": "wiki_reference",
                        "content": intro
                    })
                for i in range(1, len(sections), 2):
                    sec_title = sections[i].strip()
                    sec_body = sections[i+1].strip() if i+1 < len(sections) else ""
                    if len(sec_body) > 100:
                        unified_modules.append({
                            "doc_id": f"{art['doc_id']}_sec_{i//2}",
                            "title": f"{art['title']} - {sec_title}",
                            "category": "wiki_reference",
                            "content": sec_body
                        })
            else:
                unified_modules.append({
                    "doc_id": art["doc_id"],
                    "title": art["title"],
                    "category": "wiki_reference",
                    "content": art["content"]
                })

    with open(unified_path, "w", encoding="utf-8") as f:
        json.dump(unified_modules, f, indent=2, ensure_ascii=False)

    print(f"[CorpusMerger] Unified corpus successfully built with {len(unified_modules)} structured technical modules.")
    print(f"Saved to: {unified_path}")

if __name__ == "__main__":
    main()
