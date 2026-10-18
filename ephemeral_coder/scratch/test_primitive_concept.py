import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from engine.ollama_client import query_ollama

def test_primitive_concept():
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
        "Decompose the user's requirements recursively until reaching LANGUAGE-DEFINED BUILT-IN FUNCTIONS, DATA PRIMITIVES, OR ELEMENTARY CONSTANTS (e.g., dict lookup, sum(), range() loop, comparisons ==/>, const list).\n"
        "There must be NO undefined or ambiguous areas. For every feature, define:\n"
        "- What (exact data fields, formula, criteria)\n"
        "- When (phase, timing, trigger)\n"
        "- Scope (range of elements, maximum deck/hand size)\n"
        "Output ONLY a JSON array of concrete sub-elements, or 'null' if irreducible language primitive."
    )

    # Test Level 1 decomposition of Root
    print("=== Testing Level 1 Decomposition of Root ===")
    p1 = (
        f"全体要件: 『{prompt}』\n"
        f"このゲームを破綻なく動作させるために不可欠な主要構成要素（ゲームシステム、元素、分子結合、デッキ構築、勝敗ルール、UI等）に大分類分解してください。\n"
        f"出力形式: JSON配列 [\"要素1\", \"要素2\", ...] のみ:"
    )
    r1 = query_ollama("llama3.1:8b", prompt=p1, system_prompt=sys_prompt, timeout=60)
    print("Level 1 Output:\n", r1)

    # Test Level 2 decomposition of "分子結合システム"
    print("\n=== Testing Level 2 Decomposition of '分子結合システム' ===")
    p2 = (
        f"要素: 『分子結合システム（手札内での自動結合、親和力による再結合、特殊カード）』\n"
        f"このシステムを構成する具体的サブシステムに分解してください（自動結合判定、再結合優先度、特殊効果等）。\n"
        f"出力形式: JSON配列 [\"要素1\", \"要素2\", ...] のみ:"
    )
    r2 = query_ollama("llama3.1:8b", prompt=p2, system_prompt=sys_prompt, timeout=60)
    print("Level 2 Output:\n", r2)

    # Test Level 3 decomposition of "勝敗判定システム"
    print("\n=== Testing Level 3 Decomposition of '勝敗判定システム' (What / When / Primitives) ===")
    p3 = (
        f"要素: 『勝敗判定システム』\n"
        f"未定義領域を残さないため、【いつ判定を行うか（判定タイミング）】と【何をもって勝敗とするか（判定基準値・計算式）】に分解してください。\n"
        f"出力形式: JSON配列 [\"要素1\", \"要素2\", ...] のみ:"
    )
    r3 = query_ollama("llama3.1:8b", prompt=p3, system_prompt=sys_prompt, timeout=60)
    print("Level 3 Output:\n", r3)

    # Test Primitive / Null check of a primitive
    print("\n=== Testing Primitive Null Check ===")
    p4 = (
        f"要素: 『手札内分子の安定度スコア総和計算 (sum(m.stability for m in hand_molecules))』\n"
        f"言語組み込み関数（sum()）や単一プリミティブ処理に到達している場合、これ以上前提を分解できないため null とだけ出力してください。\n"
        f"出力形式: null または JSON配列:"
    )
    r4 = query_ollama("llama3.1:8b", prompt=p4, system_prompt=sys_prompt, timeout=60)
    print("Primitive Check Output:\n", r4)

if __name__ == '__main__':
    test_primitive_concept()
