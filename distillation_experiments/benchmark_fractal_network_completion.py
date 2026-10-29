"""
benchmark_fractal_network_completion.py
---------------------------------------
フラクタル階層概念ネットワーク連動 文章補完AI ベンチマークテスト

【アーキテクチャ】
1. Base Completion AI: 従来の全語彙フラット探索 (Flat Softmax / Greedy)
2. Fractal Network AI:
   - Level 1 (マクロ核): プロンプト文脈から目的の「概念核（イデア核）」をマクロ高速道路で高速特定。
   - Level 2 (ミクロ回路): 選択された核の内部サブネットワークから、最小抵抗（最高コンダクタンス）の最適トークンを精密補完。

【ベンチマークテスト項目】
- Test 1: 事実性・知識補完 (Factuality QA: 首都、科学、歴史、アルゴリズム)
- Test 2: 多義性・文脈適応補完 (Contextual Disambiguation: bank, apple, crane)
- Test 3: 多段階因果推論補完 (Multi-hop Causal Chain: 警察➔逮捕、火➔灰、抽象代数➔アーベル群)
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GLM_DATA_PATH = "distillation_experiments/glm5.3_zero_vram_pure_concept_terrains.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

BENCHMARK_TASKS = [
    # --- Category 1: 事実性・知識補完 ---
    {
        "category": "Factuality",
        "prompt": "オーストラリアの首都はどこですか？シドニーと誤解されやすいですが、正しい都市名は",
        "expected": ["キャンベラ", "canberra"],
        "target_concept": "capital/geography"
    },
    {
        "category": "Factuality",
        "prompt": "太陽系で太陽に最も近い第1惑星は",
        "expected": ["水星", "mercury"],
        "target_concept": "planet/astronomy"
    },
    {
        "category": "Factuality",
        "prompt": "データベースのACID特性における「A」の略称は",
        "expected": ["atomicity", "原子性"],
        "target_concept": "database/computer"
    },
    {
        "category": "Factuality",
        "prompt": "ソート済み配列から特定要素を検索する二分探索（バイナリサーチ）の計算量はO記法で",
        "expected": ["o(log n)", "o(log", "log n"],
        "target_concept": "algorithm/complexity"
    },
    {
        "category": "Factuality",
        "prompt": "1603年に江戸幕府を開いた初代征夷大将軍は",
        "expected": ["徳川家康", "家康"],
        "target_concept": "history/japan"
    },

    # --- Category 2: 多義性・文脈適応補完 ---
    {
        "category": "Disambiguation",
        "prompt": "In the Wall Street financial district, customers deposit money and apply for loans at the commercial",
        "expected": ["bank"],
        "target_concept": "finance/money"
    },
    {
        "category": "Disambiguation",
        "prompt": "Walking along the muddy edge of the flowing river, fishermen sat peacefully on the grassy",
        "expected": ["bank"],
        "target_concept": "nature/river"
    },
    {
        "category": "Disambiguation",
        "prompt": "At the high-tech electronics conference in Cupertino, engineers unveiled the latest smartphone from",
        "expected": ["apple"],
        "target_concept": "tech/corporation"
    },
    {
        "category": "Disambiguation",
        "prompt": "At the autumn harvest fruit festival in the orchard, the delicious sweet pie was baked with red",
        "expected": ["apple", "apples"],
        "target_concept": "fruit/agriculture"
    },

    # --- Category 3: 因果連鎖補完 ---
    {
        "category": "Causal Chain",
        "prompt": "When severe criminal activity occurs, the legal duty of the investigating police is to make an",
        "expected": ["arrest"],
        "target_concept": "law/justice"
    },
    {
        "category": "Causal Chain",
        "prompt": "When intense heat and fire completely consume dry wooden logs, the remaining residue on the ground is",
        "expected": ["ash", "ashes"],
        "target_concept": "chemistry/fire"
    },
    {
        "category": "Causal Chain",
        "prompt": "In abstract algebra, when a group operation satisfies the commutative law (a * b = b * a), it is called an",
        "expected": ["abelian"],
        "target_concept": "math/algebra"
    }
]

class FractalHierarchicalEngine:
    """
    フラクタル階層ネットワークによる文章補完エンジン
    マクロ層（Level 1）で大局的文脈核を絞り込み、
    ミクロ層（Level 2）で核内サブネットワークから最適候補をブースト補完
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        print("[*] フラクタル階層ネットワークを構築中...")
        # GLM-5.3の概念地形からマクロ核（53個）のアンカーを取得
        glm_data = torch.load(GLM_DATA_PATH, map_location="cpu", weights_only=False)
        self.macro_l2 = glm_data['Layer_2'].float().to(device) # (53, 6144)
        self.macro_l2 = F.normalize(self.macro_l2, p=2, dim=-1)

        # 語彙埋め込みからマクロ核への射影プロキシを構築
        W_embed = model.get_input_embeddings().weight.data.float() # (V, 1536)
        self.W_embed_norm = F.normalize(W_embed, p=2, dim=-1)
        
        # 簡易プロキシクラスタリング（Qwen空間での53個のマクロ重心）
        U, S, V = torch.pca_lowrank(self.W_embed_norm[:8000], q=53)
        self.qwen_macro_cores = F.normalize(V.T, p=2, dim=-1) # (53, 1536)
        print(f"  [+] マクロ核数: {self.qwen_macro_cores.shape[0]} 個 (Level 1)")
        print(f"  [+] ミクロ語彙空間: {W_embed.shape[0]} 単語 (Level 2)")

    def complete_hierarchical(self, prompt, max_new_tokens=20):
        """
        フラクタル階層補完:
        1. 文脈の隠れ状態 h から、最も活性化しているマクロ核 (Level 1) を特定
        2. そのマクロ核の引力圏にある語彙候補をミクロ回路 (Level 2) でブースト
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        cur_ids = inputs["input_ids"]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(cur_ids, output_hidden_states=True)
                hidden = out.hidden_states[-1][:, -1, :] # (1, D)
                h_norm = F.normalize(hidden.float(), p=2, dim=-1)

                # --- Level 1: マクロ核ルーティング ---
                macro_affinities = torch.matmul(h_norm, self.qwen_macro_cores.T) # (1, 53)
                top_macro_val, top_macro_idx = torch.topk(macro_affinities, k=3, dim=-1)

                # --- Level 2: ミクロサブネットワーク内のブースト ---
                # 活性化したマクロ核の近傍語彙にのみ高コンダクタンス（低抵抗）を付与
                active_cores = self.qwen_macro_cores[top_macro_idx[0]] # (3, D)
                # 語彙との親和性
                vocab_affinities = torch.matmul(self.W_embed_norm, active_cores.T).max(dim=-1).values # (V,)
                
                # 生のロジット
                logits = out.logits[:, -1, :].clone()
                # 階層ブースト: マクロ核の引力圏にある単語のロジットを適度にブースト (+1.2)
                boost = torch.clamp((vocab_affinities - 0.25) / 0.5, min=0.0) * 1.5
                logits[0] += boost

                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                cur_ids = torch.cat([cur_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        gen_tokens = cur_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

def run_benchmark():
    print("=" * 70)
    print("🚀 フラクタル階層ネットワーク連動 文章補完AI ベンチマーク開始")
    print("=" * 70)

    print(f"[*] モデルをロード中: {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()

    fractal_engine = FractalHierarchicalEngine(model, tok)

    base_results = []
    fractal_results = []

    print("\n" + "=" * 70)
    print("📋 テスト実行中...")
    print("=" * 70)

    for idx, item in enumerate(BENCHMARK_TASKS, 1):
        prompt = item["prompt"]
        expected_list = item["expected"]
        cat = item["category"]

        # 1. Base AI (従来のフラット生成)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_base = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        base_tokens = out_base[0][inputs["input_ids"].shape[1]:]
        ans_base = tok.decode(base_tokens, skip_special_tokens=True).strip()
        hit_base = any(exp.lower() in ans_base.lower() for exp in expected_list)

        # 2. Fractal Network AI (階層ネットワーク連動補完)
        ans_fractal = fractal_engine.complete_hierarchical(prompt, max_new_tokens=20)
        hit_fractal = any(exp.lower() in ans_fractal.lower() for exp in expected_list)

        base_results.append({"cat": cat, "hit": hit_base, "ans": ans_base})
        fractal_results.append({"cat": cat, "hit": hit_fractal, "ans": ans_fractal})

        print(f"\n[問 {idx}/{len(BENCHMARK_TASKS)}] ({cat}) {prompt}")
        print(f"  ・期待解: {expected_list}")
        print(f"  ・[Base AI]    : {ans_base[:35]}... [{'PASS' if hit_base else 'FAIL'}]")
        print(f"  ・[Fractal AI] : {ans_fractal[:35]}... [{'PASS' if hit_fractal else 'FAIL'}]")

    # 集計
    base_acc = sum(1 for r in base_results if r["hit"]) / len(BENCHMARK_TASKS)
    fractal_acc = sum(1 for r in fractal_results if r["hit"]) / len(BENCHMARK_TASKS)

    print("\n" + "=" * 70)
    print("🏆 ベンチマーク最終集計結果")
    print("=" * 70)
    print(f"・Base AI (従来フラット探索)       : 正解率 {base_acc*100:.1f}% ({sum(1 for r in base_results if r['hit'])}/{len(BENCHMARK_TASKS)})")
    print(f"・Fractal Network AI (二階層連動) : 正解率 {fractal_acc*100:.1f}% ({sum(1 for r in fractal_results if r['hit'])}/{len(BENCHMARK_TASKS)})")
    print("=" * 70)

    # カテゴリ別集計
    categories = sorted(list(set(item["category"] for item in BENCHMARK_TASKS)))
    cat_summary = {}
    for c in categories:
        b_hits = sum(1 for r in base_results if r["cat"] == c and r["hit"])
        f_hits = sum(1 for r in fractal_results if r["cat"] == c and r["hit"])
        total = sum(1 for r in BENCHMARK_TASKS if r["category"] == c)
        cat_summary[c] = {"base": b_hits / total, "fractal": f_hits / total, "total": total}
        print(f"  [{c:<15}] Base: {b_hits}/{total} ({b_hits/total*100:.0f}%) ➔ Fractal: {f_hits}/{total} ({f_hits/total*100:.0f}%)")

    # レポート保存
    report = {
        "base_accuracy": base_acc,
        "fractal_accuracy": fractal_acc,
        "category_summary": cat_summary,
        "details": [
            {
                "prompt": item["prompt"],
                "expected": item["expected"],
                "base_ans": b["ans"],
                "base_hit": b["hit"],
                "fractal_ans": f["ans"],
                "fractal_hit": f["hit"]
            }
            for item, b, f in zip(BENCHMARK_TASKS, base_results, fractal_results)
        ]
    }
    out_json = "distillation_experiments/fractal_completion_benchmark_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[+] レポートを保存しました: {out_json}")

if __name__ == "__main__":
    run_benchmark()
