"""
Conditional Epistemic Reasoner & Missing Premise Clarifier
(条件分岐的イデア補完および不足前提逆算システム)
================================================================================
Core Functionality:
1. Detects that the input lacks the necessary premise filter to make a unique claim.
2. Formulates potential conditional premises (Filter A -> Outcome A', Filter B -> Outcome B').
3. Constructs the structured response:
   "〜であるなら〜で、〜だったなら〜である。" (State the branching conditional truths)
4. Diagnoses exactly what premise parameter is missing and asks the user for it:
   "そのうえで、不足している〇〇について教えていただけますでしょうか？"
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
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class ConditionalEpistemicEngine:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_conditional_branches(self, question: str) -> Dict:
        """
        Phase 1: LLM completes the missing conditional filters and branching outcomes.
        The LLM is strictly constrained to output 'Condition -> Outcome' branches
        without picking one as an absolute dogma.
        """
        prompt = (
            f"質問: 「{question}」\n\n"
            f"この質問は文脈や前提条件が省略されているため、単一の断定的な答えを出すと誤りになります。\n"
            f"以下の指示に従って、日本語で厳密に分析してください。\n\n"
            f"1. 結論が肯定または特定の結果になる「前提条件1（状況・文脈）」と、その「帰結1」\n"
            f"2. 結論が否定または逆の結果になる「前提条件2（状況・文脈）」と、その「帰結2」\n"
            f"3. どちらの結論になるかを確定するために、質問者に対して不足している具体的なパラメータや状況\n\n"
            f"出力フォーマット（厳格遵守・日本語のみ）:\n"
            f"条件1: <前提条件1> -> 帰結1: <その場合の結論>\n"
            f"条件2: <前提条件2> -> 帰結2: <その場合の結論>\n"
            f"不足情報: <質問者に確認すべき不足している具体的な情報>"
        )

        messages = [
            {"role": "system", "content": "あなたは哲学的・論理的に厳密な分析エンジンです。決して安易な断定を行わず、前提条件によって結論がどう分岐するかを正確な日本語で整理します。"},
            {"role": "user", "content": prompt}
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.tokenizer(chat, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        resp = self.tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()

        # Parse branches
        b1, b2, missing = "", "", ""
        for line in resp.split("\n"):
            line = line.strip()
            if line.startswith("条件1:"):
                b1 = line[len("条件1:"):].strip()
            elif line.startswith("条件2:"):
                b2 = line[len("条件2:"):].strip()
            elif line.startswith("不足情報:"):
                missing = line[len("不足情報:"):].strip()

        return {
            "branch_1": b1,
            "branch_2": b2,
            "missing_info": missing,
            "raw_analysis": resp
        }

    def synthesize_response(self, question: str, branches: Dict) -> str:
        b1 = branches["branch_1"]
        b2 = branches["branch_2"]
        missing = branches["missing_info"]

        # Parse 'Cond -> Outcome'
        cond1 = b1.split("->")[0].replace("条件1:", "").strip()
        out1 = b1.split("->")[1].replace("帰結1:", "").strip() if "->" in b1 else "その結論が成立します"

        cond2 = b2.split("->")[0].replace("条件2:", "").strip()
        out2 = b2.split("->")[1].replace("帰結2:", "").strip() if "->" in b2 else "逆の結論となります"

        response = (
            f"ご質問の事象は前提条件（状況や目的）によって結論が分かれます。\n\n"
            f"・【{cond1}】であるなら、{out1}。\n"
            f"・一方で【{cond2}】であるなら、{out2}。\n\n"
            f"したがって、結論を一意に確定するためには【{missing}】の情報が不可欠です。\n"
            f"現在想定されている具体的な状況や前提条件について教えていただけますでしょうか？"
        )
        return response


def run_demonstration():
    print("=" * 85)
    print("DEMONSTRATION: CONDITIONAL INVARIANT & CLARIFICATION REASONER")
    print(f"Model: {MODEL_ID} on {device}")
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

    engine = ConditionalEpistemicEngine(model, tokenizer, device)

    # Representative incomplete real-world questions
    TEST_QUESTIONS = [
        "焚き火に水をかけて消火すべきでしょうか？",
        "車で120kmの距離を走行したのですが、時速100kmの制限速度を超過したでしょうか？",
        "電子レンジに金属製のスプーンを入れて加熱しても大丈夫ですか？",
        "高い所から重い鉄球と軽い羽を同時に落としたら、鉄球のほうが先に落ちますか？"
    ]

    results = []

    for idx, q in enumerate(TEST_QUESTIONS):
        print(f"\n[{idx+1}/{len(TEST_QUESTIONS)}] 質問: \"{q}\"")
        t0 = time.time()
        branches = engine.extract_conditional_branches(q)
        formatted_answer = engine.synthesize_response(q, branches)
        elapsed_ms = (time.time() - t0) * 1000.0

        print("-" * 70)
        print(formatted_answer)
        print("-" * 70)
        print(f"(生成所要時間: {elapsed_ms:.1f}ms)\n")

        results.append({
            "question": q,
            "branches": branches,
            "final_response": formatted_answer,
            "elapsed_ms": elapsed_ms
        })

    # Save Results
    out_file = os.path.join(OUTPUT_DIR, "conditional_epistemic_reasoner_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    run_demonstration()
