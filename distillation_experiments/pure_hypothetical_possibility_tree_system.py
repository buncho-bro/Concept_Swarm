"""
Pure Hypothetical Possibility Tree System
(純粋仮定義・可能性木探査＆必要原則抽出システム)
================================================================================
Core Philosophical Principle:
1. Complete Information Principle:
   If all variables are given, a deterministic correct answer exists.
2. Hypothetical Possibility Trees (仮定義による木生成):
   Since premises are missing, allow tentative definitions (hypotheses H1, H2, H3...).
   Generate possibility trees by varying these definitions.
3. Necessary Principle Exploration (必要原則の探査):
   By comparing the possibility trees, pinpoint the exact sensitivity boundary
   (the necessary principle / missing parameter) that bifurcates the conclusion.
4. Zero Hardcoded Dictionaries:
   100% dynamic, general-purpose, and free of manual archetype enumeration.
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


class PureHypotheticalPossibilityTreeSystem:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.tokenizer(chat, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()

    def explore_possibility_trees(self, question: str) -> Dict:
        """
        Step 1 & 2:
        Generate tentative definitions (hypotheses), grow possibility trees under each,
        and extract the necessary invariant principle from the cross-tree variance.
        """
        system_prompt = (
            "あなたは数理論理学および認識論の分析エンジンです。\n"
            "完全な前提が与えられれば正しい答えは一意に定まりますが、問いに前提が欠落している場合、"
            "「前提を仮に〇〇と定義する（仮定義）」ことで可能性の木を複数生成できます。\n"
            "それらの木を比較し、結論を左右している決定的な「必要原則（不足している根本変数）」を抽出してください。"
        )

        user_prompt = (
            f"問い: 「{question}」\n\n"
            f"この問いに対して、結論や意味が大きく異なる結果になるような「仮の前提条件（仮定義）」を3つ設定し、"
            f"それぞれの仮定義のもとでの帰結（結論）を導いてください。\n"
            f"そのうえで、これらの仮定義の間で何が変化したことで結論が分岐したのか、「必要原則（結論を決定づける根本要因）」を1つ抽出してください。\n\n"
            f"出力フォーマット（必ずこの形式を1行ずつ厳密に守ってください）:\n"
            f"仮定義1: <仮定する状況・条件A> -> 結論1: <その場合の明確な結論>\n"
            f"仮定義2: <仮定する状況・条件B> -> 結論2: <その場合の明確な結論>\n"
            f"仮定義3: <仮定する状況・条件C> -> 結論3: <その場合の明確な結論>\n"
            f"必要原則: <結論を一意に決定するために必要な根本的な前提変数（1行・簡潔に）>\n"
            f"質問者への確認: <その必要原則の値を質問者に尋ねる簡潔な問いかけ（1文）>"
        )

        raw_output = self.generate(system_prompt, user_prompt, max_tokens=320)

        # Parse branches and necessary principle
        trees = []
        necessary_principle = "前提条件の具体的な状況"
        clarification_question = "想定されている具体的な前提条件について教えていただけますでしょうか？"

        for line in raw_output.split("\n"):
            line = line.strip()
            if re.match(r"^仮定義\d*[\s:：]", line) and ("->" in line or "→" in line):
                sep = "->" if "->" in line else "→"
                parts = line.split(sep, 1)
                cond = re.sub(r"^仮定義\d*[\s:：]*", "", parts[0]).strip("【】「」<> ")
                outcome = re.sub(r"^結論\d*[\s:：]*", "", parts[1]).strip("【】「」<> ").rstrip("。")
                if cond and outcome:
                    trees.append({"hypothesis": cond, "outcome": outcome})
            elif re.match(r"^必要原則[\s:：]", line):
                necessary_principle = re.sub(r"^必要原則[\s:：]*", "", line).strip("【】「」<> 。")
            elif re.match(r"^質問者への確認[\s:：]", line):
                clarification_question = re.sub(r"^質問者への確認[\s:：]*", "", line).strip("【】「」<> ")

        # Fallback parser if format varied
        if len(trees) < 2:
            lines = [l for l in raw_output.split("\n") if l.strip()]
            for l in lines:
                if ("->" in l or "→" in l) and not l.startswith("必要原則") and not l.startswith("質問者"):
                    sep = "->" if "->" in l else "→"
                    p = l.split(sep, 1)
                    trees.append({"hypothesis": p[0].strip("0123456789.:： "), "outcome": p[1].strip().rstrip("。")})

        return {
            "question": question,
            "trees": trees,
            "necessary_principle": necessary_principle,
            "clarification_question": clarification_question,
            "raw_output": raw_output
        }

    def synthesize_response(self, result: Dict) -> str:
        trees = result["trees"]
        necessary_principle = result["necessary_principle"]
        clarification = result["clarification_question"]

        header = (
            f"この問いは前提条件によって結論が一意に定まらないため、"
            f"前提をどのように「仮定義」するかによって可能性の木（結論）が以下のように分岐します。"
        )

        bullet_lines = []
        for t in trees:
            hypo = t["hypothesis"]
            outcome = t["outcome"]
            bullet_lines.append(f"・**【{hypo}】**と定義するなら、{outcome}。")

        bullets_text = "\n".join(bullet_lines)

        footer = (
            f"これらの可能性の木を比較すると、結論を決定づける必要原則は**【{necessary_principle}】**にあります。\n"
            f"{clarification}"
        )

        return f"{header}\n\n{bullets_text}\n\n{footer}"


def test_pure_system():
    print("=" * 85)
    print("TEST: PURE HYPOTHETICAL POSSIBILITY TREE & INVARIANT PRINCIPLE SYSTEM")
    print(f"Device: {device} | Model: {MODEL_ID}")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    system = PureHypotheticalPossibilityTreeSystem(model, tokenizer, device)

    TEST_QUESTIONS = [
        "焚き火に水をかけて消火すべきでしょうか？",
        "車で120kmの距離を走行したのですが、時速100kmの制限速度を超過したでしょうか？",
        "『リンゴ』って何ですか？",
        "『水に浮くもの』に当てはまるものは何ですか？",
        "好きな人に今すぐ告白すべきでしょうか？"
    ]

    results = []

    for idx, q in enumerate(TEST_QUESTIONS):
        print(f"\n[{idx+1}/{len(TEST_QUESTIONS)}] 問い: 「{q}」")
        t0 = time.time()
        res = system.explore_possibility_trees(q)
        final_answer = system.synthesize_response(res)
        elapsed_ms = (time.time() - t0) * 1000.0

        print(f"探査された必要原則: 【{res['necessary_principle']}】")
        print(f"生成された可能性の木（枝数）: {len(res['trees'])}")
        print("-" * 75)
        print(final_answer)
        print("-" * 75)
        print(f"所要時間: {elapsed_ms:.1f}ms\n")

        res["final_answer"] = final_answer
        res["elapsed_ms"] = elapsed_ms
        results.append(res)

    out_file = os.path.join(OUTPUT_DIR, "pure_hypothetical_possibility_tree_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"All tests finished! Saved to: {out_file}")


if __name__ == "__main__":
    test_pure_system()
