"""
Test: Pure Dictionary-Free Autonomous Epistemic Pipeline
========================================================
GEMINI.md 第2条（ハードコード辞書の完全禁止）に準拠した完全自律型アーキテクチャ。
- 手動単語リスト（web_cues, meta_patterns 等）を一切使用しない
- モデルの内部ロジットによる自律的行動判定 (Epistemic Action Probe)
- モデルによる自律的検索実体抽出 (Autonomous Entity Extraction)
- Layer 0 埋め込みコサイン類似度による適合性検証 (Dense Relevance Gate)
"""

import os
import sys
import math
import time
import requests
import urllib.parse
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
_FALLBACK_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
OUTPUT_DIR = _LOCAL_DIR if os.path.exists(os.path.join(_LOCAL_DIR, "lora_entropy_descent_calibrated", "adapter_model.safetensors")) else _FALLBACK_DIR
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[{device.upper()}] モデルを初期化しています...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(
    base_model,
    os.path.join(OUTPUT_DIR, "lora_entropy_descent_calibrated"),
    adapter_name="calibrated"
)
peft_model.eval()

# 行動判定トークン (Search vs Direct)
search_tokens = ['search', ' search', 'Search', ' Search']
direct_tokens = ['direct', ' direct', 'Direct', ' Direct']
s_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in search_tokens]
d_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in direct_tokens]

embed_weights = base_model.model.embed_tokens.weight.data


def get_text_dense_embedding(text):
    """Layer 0 単語埋め込みの平均ベクトル"""
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)[0]
    embeds = embed_weights[input_ids]
    mean_vec = embeds.mean(dim=0)
    return F.normalize(mean_vec, p=2, dim=0)


def autonomous_action_probe(user_query, threshold=-4.5):
    """
    辞書ゼロ: モデルの内的不確実性とロジット対比による自律的行動決定
    """
    prompt = (
        f"<|im_start|>system\nDetermine whether this user query requires external factual web search ('search') "
        f"or should be handled directly via conversation, internal memory, or reasoning ('direct').<|im_end|>\n"
        f"<|im_start|>user\nQuery: {user_query}\nAction:<|im_end|>\n"
        f"<|im_start|>assistant\nJudgment:"
    )
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = peft_model(inp["input_ids"]).logits[0, -1]
    
    s_logit = torch.logsumexp(logits[s_ids], dim=0).item()
    d_logit = torch.logsumexp(logits[d_ids], dim=0).item()
    score = s_logit - d_logit
    need_search = score > threshold
    return need_search, score


def autonomous_entity_extractor(user_query):
    """
    辞書ゼロ: 手動置換を排し、モデル自身に検索対象の実体を抽出させる
    """
    prompt = (
        f"<|im_start|>system\nExtract the core search entity from the user query. "
        f"Respond with only the entity keyword without punctuation.<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = peft_model.generate(**inp, max_new_tokens=15, do_sample=False)
    extracted = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return extracted if extracted else user_query


def search_and_verify_dense(query_entity, max_results=1):
    """
    Wikipedia検索 + 埋め込みコサイン類似度による意味論的ノイズ除去
    """
    search_url = f"https://ja.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query_entity)}&format=json&utf8=1"
    headers = {'User-Agent': 'ConceptSwarmAgent/1.0'}
    try:
        r = requests.get(search_url, headers=headers, timeout=4)
        if r.status_code == 200:
            items = r.json().get('query', {}).get('search', [])
            if items:
                top_item = items[0]
                title = top_item['title']
                
                # 意味論的コサイン適合性ゲート
                q_vec = get_text_dense_embedding(query_entity)
                title_vec = get_text_dense_embedding(title)
                cosine_sim = torch.dot(q_vec, title_vec).item()

                if cosine_sim >= 0.40:
                    summary_url = f"https://ja.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    sr = requests.get(summary_url, headers=headers, timeout=4)
                    if sr.status_code == 200:
                        extract = sr.json().get('extract', '')
                        return True, title, extract, cosine_sim
                else:
                    return False, title, f"不適合ノイズ除外 (コサイン類似度: {cosine_sim:.2f})", cosine_sim
    except Exception as e:
        pass
    return False, "", "", 0.0


def main():
    print("=" * 70)
    print("GEMINI.md 第2条準拠: 完全自律・辞書ゼロ行動判定パイプライン検証")
    print("=" * 70)

    test_queries = [
        "あなたは誰ですか？",
        "こんにちは！",
        "自己紹介してください。",
        "三角形の内角の和は190度ですか？",
        "2024年のパリオリンピックの開催日程と開催都市は？",
        "量子コンピュータの仕組みとはどのようなものですか？",
        "オッペンハイマーは誰ですか？",
        "先ほどのコードネームは何でしたか？"
    ]

    for q in test_queries:
        print(f"\n[入力クエリ]: {q}")
        need_search, score = autonomous_action_probe(q)
        print(f"  -> 自律行動判定: {'[SEARCH] 外部知識検索' if need_search else '[DIRECT] 直接対話/内部記憶'} (Score: {score:+.2f})")

        if need_search:
            entity = autonomous_entity_extractor(q)
            print(f"  -> 自律抽出実体: '{entity}'")
            success, title, content, sim = search_and_verify_dense(entity)
            if success:
                print(f"  -> 検索結果適合 [Title: {title} | 類似度: {sim:.2f}]")
                print(f"     内容: {content[:100]}...")
            else:
                print(f"  -> 検索結果不採用 [Title: {title} | 理由: {content}]")

    print("\n検証完了。")


if __name__ == "__main__":
    main()
