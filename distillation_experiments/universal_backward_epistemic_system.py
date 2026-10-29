"""
Universal Backward Epistemic System (v6: Pure Hypothetical Tree & Invariant Principle Engine)
(汎用逆木探査＆不足前提逆算システム v6: 完全辞書ゼロ・仮定義可能性木探査エンジン)
================================================================================
Philosophical Foundations:
1. Complete Information Principle (完全性の原理):
   When all variables are given, a deterministic correct causal answer exists.
2. Hypothetical Possibility Tree Generation (仮定義による木生成):
   Since premises are missing in open human questions, allow tentative definitions (hypotheses H1, H2, H3).
   Generate possibility trees by varying these definitions without hardcoded dictionaries.
3. Invariant Principle Exploration (必要原則の探査):
   By cross-comparing the generated trees, extract the exact sensitivity boundary
   (the necessary principle / missing root variable) that bifurcates the conclusions.
4. Zero Hardcoded Dictionaries (辞書・暗記の完全全廃):
   100% dynamic, algorithmic, and general-purpose reasoning across all domains.
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


class UniversalPossibilityTreeEngine:
    """
    Pure algorithmic possibility tree explorer without any hardcoded archetype dictionary.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, max_tokens: int = 250) -> str:
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()

    def explore_possibility_trees_and_invariants(self, question: str) -> Dict:
        """
        Pure dynamic algorithm:
        1. Formulates 3 orthogonal hypothetical definitions (仮定義).
        2. Simulates deterministic outcomes under each complete-information hypothesis.
        3. Explores and extracts the necessary invariant principle (必要原則) that governs the divergence.
        """
        # Generic epistemic prompt with a neutral structural exemplar (boiling water)
        prompt = (
            "<|im_start|>system\n"
            "あなたは因果推論と認識論の分析エンジンです。必ず自然で論理的な日本語のみで回答してください。\n"
            "前提要素が全て揃っていれば正しい答えは一意に出せますが、問いに前提が欠落している場合、"
            "前提を様々に「仮定義」することで可能性の木（シナリオ）を複数作成できます。\n"
            "仮定義を変えた複数の木を比較し、結論を決定づける「必要原則（未指定の根本変数）」を探査してください。<|im_end|>\n"
            "<|im_start|>user\n"
            "問い: 「100度の熱湯に手を入れても大丈夫ですか？」\n\n"
            "この問いに対し、前提を様々に仮定義して可能性の木を3つ作成し、そこから必要原則を探査してください。<|im_end|>\n"
            "<|im_start|>assistant\n"
            "1. 仮定義: 【厚い耐熱手袋を完全に装着している場合】 -> 結論: 熱が遮断されるため火傷せず安全です\n"
            "2. 仮定義: 【素手で直接浸す場合】 -> 結論: 重度の熱傷を負うため極めて危険です\n"
            "3. 仮定義: 【指先を一瞬（0.1秒未満）だけ触れる場合】 -> 結論: 瞬間的な痛みと軽度の熱傷のリスクがあります\n"
            "必要原則: 耐熱保護具の有無と皮膚の接触時間\n"
            "質問者への確認: 耐熱手袋などの保護具を着用されているか、素手の状況でしょうか？<|im_end|>\n"
            f"<|im_start|>user\n"
            f"問い: 「{question}」\n\n"
            f"この問いに対し、前提を様々に仮定義して可能性の木を3つ作成し、そこから必要原則を探査してください。<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        raw_output = self.generate(prompt, max_tokens=250)

        # Parse branches and necessary principle
        trees = []
        necessary_principle = "前提条件の具体的な状況"
        clarification_question = "想定されている具体的な前提条件について教えていただけますでしょうか？"

        for line in raw_output.split("\n"):
            line = line.strip()
            # Match branch: "1. 仮定義: 【...】 -> 結論: ..."
            m = re.match(r"^(\d+[\.、\s]*)?仮定義[\s:：]*【?(.*?)】?[\s]*(->|→|＝＞)[\s]*結論[\s:：]*(.*)", line)
            if m:
                hypo = m.group(2).strip("【】「」 ")
                outcome = m.group(4).strip("【】「」 ").rstrip("。")
                if hypo and outcome:
                    trees.append({"hypothesis": hypo, "outcome": outcome})
            elif line.startswith("必要原則"):
                necessary_principle = re.sub(r"^必要原則[\s:：]*", "", line).strip("【】「」<> 。")
            elif line.startswith("質問者への確認"):
                clarification_question = re.sub(r"^質問者への確認[\s:：]*", "", line).strip("【】「」<> ")

        # Fallback parser if format slightly varied
        if len(trees) < 2:
            for line in raw_output.split("\n"):
                line = line.strip()
                if ("->" in line or "→" in line or "＝＞" in line) and not line.startswith("必要原則") and not line.startswith("質問者"):
                    sep = "->" if "->" in line else ("→" if "→" in line else "＝＞")
                    parts = line.split(sep, 1)
                    hypo = re.sub(r"^(\d+[\.、\s]*)?(仮定義|仮定|条件\d*)[\s:：]*", "", parts[0]).strip("【】「」 ")
                    outcome = re.sub(r"^(結論\d*|帰結\d*)[\s:：]*", "", parts[1]).strip("【】「」 ").rstrip("。")
                    if hypo and outcome:
                        trees.append({"hypothesis": hypo, "outcome": outcome})

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
            f"前提をどのように「仮定義」するかによって可能性の木（帰結）が以下のように分岐します。"
        )

        bullet_lines = []
        for t in trees:
            hypo = t["hypothesis"].strip("【】 ")
            outcome = t["outcome"].strip("【】 ")
            bullet_lines.append(f"・**【{hypo}】**であるなら、{outcome}。")

        bullets_text = "\n".join(bullet_lines)

        footer = (
            f"これらの可能性の木を比較すると、結論を決定づける必要原則は**【{necessary_principle}】**にあります。\n"
            f"{clarification}"
        )

        return f"{header}\n\n{bullets_text}\n\n{footer}"


def run_pure_invariant_benchmark():
    print("=" * 85)
    print("UNIVERSAL BACKWARD EPISTEMIC SYSTEM (v6)")
    print("PURE HYPOTHETICAL POSSIBILITY TREE & INVARIANT PRINCIPLE ENGINE (ZERO DICTIONARY)")
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

    system = UniversalPossibilityTreeEngine(model, tokenizer, device)

    # 10 Diverse benchmark questions across all modalities + open novel questions
    BENCHMARK_SUITE = [
        # 1. Action Decision / Safety
        "焚き火に水をかけて消火すべきでしょうか？",
        # 2. Math / Physics Calculation
        "車で120kmの距離を走行したのですが、時速100kmの制限速度を超過したでしょうか？",
        # 3. Definition (Polysemous / Context)
        "『リンゴ』って何ですか？",
        # 4. Definition (Scientific Boundary)
        "『ウイルス』とは何ですか？生物ですか？",
        # 5. Mechanism Enumeration
        "『水に浮くもの』に当てはまるものは何ですか？",
        # 6. Factor Enumeration
        "『風邪をひく原因』にはどのようなものがありますか？",
        # 7. Human Emotion / Advice
        "好きな人に今すぐ告白すべきでしょうか？",
        # 8. Counter-intuitive Physics
        "高い所から重い鉄球と軽い羽を同時に落としたら、鉄球のほうが先に落ちますか？",
        # 9. Novel Open Question 1 (Action / Health)
        "就寝前の夜遅くに熱いブラックコーヒーを飲むべきでしょうか？",
        # 10. Novel Open Question 2 (Science Definition)
        "『量子力学』とは何ですか？"
    ]

    all_results = []

    for idx, q in enumerate(BENCHMARK_SUITE):
        print(f"\n[{idx+1}/{len(BENCHMARK_SUITE)}] 問い: 「{q}」")
        t0 = time.time()
        res = system.explore_possibility_trees_and_invariants(q)
        final_answer = system.synthesize_response(res)
        elapsed_ms = (time.time() - t0) * 1000.0

        print(f"探査された必要原則: 【{res['necessary_principle']}】")
        print(f"展開された可能性の木（枝数）: {len(res['trees'])}")
        print("-" * 75)
        print(final_answer)
        print("-" * 75)
        print(f"所要時間: {elapsed_ms:.1f}ms\n")

        res["final_answer"] = final_answer
        res["elapsed_ms"] = elapsed_ms
        all_results.append(res)

    out_file = os.path.join(OUTPUT_DIR, "universal_backward_epistemic_system_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Verification complete! Results saved to: {out_file}")


if __name__ == "__main__":
    run_pure_invariant_benchmark()
