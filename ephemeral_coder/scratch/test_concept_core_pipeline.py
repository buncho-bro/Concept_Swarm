import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from engine.ollama_client import query_ollama

def test_concept_core_pipeline():
    prompt = (
        "下のようなゲームを作成してください。元素を使って対戦カードゲーム！"
        "分子にすると強くなるぞ！…でもどこかで聞いたような気が…いいえ！本作最大の特徴は分子を合成できない点。"
        "並べたら手札のなかで勝手くっついてしまう。でも更にくっつきやすい元素が隣に来ると引っ付きなおす。"
        "熱、衝撃等の特殊カードも活かしてより安定した分子を作るもよし！有機系に振るのもよし！"
        "通常のカードゲームとは一風変わったデッキ構築が楽しめます。"
    )

    sys_prompt = (
        "You are a Lead Software & Game Systems Architect practicing 'Concept Core (概念核)' tree decomposition.\n"
        "Core Philosophy:\n"
        "Decompose until reaching LANGUAGE-DEFINED BUILT-IN FUNCTIONS, OPERATORS, OR CONSTANT DEFINITIONS (e.g. sum(), len(), range(), dict, if/else, comparisons >, list slicing, random.shuffle()).\n"
        "There must be NO undefined or ambiguous gaps. Every system must specify:\n"
        "- What (exact criteria, formula, data fields)\n"
        "- When (phase, timing, trigger)\n"
        "- Scope (range of elements, maximum deck/hand size)\n"
        "Output ONLY a JSON array of concrete sub-elements in Japanese, or ONLY: null if irreducible language primitive."
    )

    # 1. Root Decomposition
    print("--- Level 1: Root Decomposition ---")
    p1 = (
        f"全体要件: 『{prompt}』\n"
        f"ゲームシステムを破綻なく構築するために、大分類（元素データ、分子結合、デッキ構築、対戦・勝敗ルール、UI表示）に分解せよ。\n"
        f"出力形式: JSON配列 [\"要素1\", \"要素2\", ...] のみ:"
    )
    r1 = query_ollama("llama3.1:8b", prompt=p1, system_prompt=sys_prompt, timeout=60)
    print("Level 1:", r1)

    # 2. Decompose 勝敗判定
    print("\n--- Level 2: Decompose 勝敗判定システム ---")
    p2 = (
        f"対象要素: 『勝敗判定システム』\n"
        f"未定義領域を残さないため、【判定タイミング（いつ行うか）】と【勝敗条件・スコア集計式（何をもって勝敗とするか）】に分解せよ。\n"
        f"出力形式: JSON配列 [\"要素1\", \"要素2\", ...] のみ:"
    )
    r2 = query_ollama("llama3.1:8b", prompt=p2, system_prompt=sys_prompt, timeout=60)
    print("Level 2 (勝敗):", r2)

    # 3. Decompose 判定条件・スコア集計式 to Primitive
    print("\n--- Level 3: Decompose スコア集計式 to Language Primitive ---")
    p3 = (
        f"対象要素: 『手札内分子の安定度スコア集計と勝敗比較』\n"
        f"これをプログラミング言語の組み込み関数（sum, 比較演算子 >）や具体的計算処理へ分解せよ。\n"
        f"もしこれ以上分解不能な単一組み込み関数（sum()等）や演算であれば null と出力せよ。\n"
        f"出力形式: JSON配列 または null:"
    )
    r3 = query_ollama("llama3.1:8b", prompt=p3, system_prompt=sys_prompt, timeout=60)
    print("Level 3 (Primitives):", r3)

if __name__ == '__main__':
    test_concept_core_pipeline()
