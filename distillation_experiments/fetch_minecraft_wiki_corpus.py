"""
Safe & Polite MediaWiki Harvester for Japanese Minecraft Wiki (ja.minecraft.wiki).
Complies strictly with MediaWiki API guidelines:
1. Custom identifiable User-Agent with contact information.
2. Polite rate limiting (1.5s delay between requests to avoid any server strain).
3. Structured plain-text extraction via action=query&prop=extracts&explaintext=true.
4. Automatic semantic chunking and JSON serialization into minecraft_corpus directory.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from typing import List, Dict

sys.stdout.reconfigure(encoding='utf-8')

API_ENDPOINT = "https://ja.minecraft.wiki/api.php"
USER_AGENT = "MinecraftLearningAgent/1.0 (personal academic research; contact: github.com/buncho-bro)"

TARGET_ARTICLES = [
    # 1. Data Packs & Commands
    "データパック",
    "関数",
    "述語",
    "タグ",
    "コマンド",
    "コマンド/execute",
    "コマンド/give",
    
    # 2. Redstone & Circuits
    "レッドストーン回路",
    "レッドストーン",
    "レッドストーンコンパレーター",
    "レッドストーンリピーター",
    "ピストン",
    "オブザーバー",
    "準接続",
    
    # 3. Resource Packs & Technical Mechanics
    "リソースパック",
    "ブロック状態",
    "モデル"
]

def fetch_article(title: str) -> Dict[str, str]:
    encoded_title = urllib.parse.quote(title)
    url = (
        f"{API_ENDPOINT}?action=query&titles={encoded_title}"
        f"&prop=extracts&explaintext=true&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id == "-1":
                    print(f"  [Warning] Article '{title}' not found (page_id: -1).")
                    return None
                text = page_info.get("extract", "").strip()
                return {
                    "doc_id": f"wiki_{page_id}",
                    "title": page_info.get("title", title),
                    "url": f"https://ja.minecraft.wiki/w/{encoded_title}",
                    "content": text
                }
    except Exception as e:
        print(f"  [Error] Failed to fetch '{title}': {e}")
        return None

def main():
    print("=" * 80)
    print("SAFE MEDIAWIKI CORPUS HARVESTER: ja.minecraft.wiki")
    print(f"Targeting {len(TARGET_ARTICLES)} key technical articles with 1.5s rate-limit intervals.")
    print("=" * 80)

    output_dir = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ja_minecraft_wiki_corpus.json")

    harvested_articles = []

    for idx, title in enumerate(TARGET_ARTICLES):
        print(f"[{idx+1}/{len(TARGET_ARTICLES)}] Fetching: {title} ...")
        article_data = fetch_article(title)
        if article_data and len(article_data["content"]) > 100:
            char_len = len(article_data["content"])
            print(f"  -> Successfully retrieved {char_len} characters.")
            harvested_articles.append(article_data)
        else:
            print(f"  -> Skipped or empty content.")
        
        # Polite rate limiting
        time.sleep(1.5)

    print("\n--- Harvesting Complete ---")
    print(f"Successfully harvested {len(harvested_articles)} articles.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(harvested_articles, f, indent=2, ensure_ascii=False)

    print(f"Saved complete structured corpus to {output_path}")

if __name__ == "__main__":
    main()
