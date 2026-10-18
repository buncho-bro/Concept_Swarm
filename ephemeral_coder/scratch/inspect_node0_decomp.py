import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json

from engine.tree_architect import TreeArchitect

ta = TreeArchitect(model_name="llama3.1:8b")
prompt = (
    "下のようなゲームを作成してください。元素を使って対戦カードゲーム！"
    "分子にすると強くなるぞ！…でもどこかで聞いたような気が…いいえ！本作最大の特徴は分子を合成できない点。"
    "並べたら手札のなかで勝手くっついてしまう。でも更にくっつきやすい元素が隣に来ると引っ付きなおす。"
    "熱、衝撃等の特殊カードも活かしてより安定した分子を作るもよし！有機系に振るのもよし！"
    "通常のカードゲームとは一風変わったデッキ構築が楽しめます。"
)

# test query atomic decomposition on node_0
res = ta._query_atomic_decomposition(prompt)
print("DECOMPOSITION RESULT ON ROOT:")
print("res =", repr(res))
