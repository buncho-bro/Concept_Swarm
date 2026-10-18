import sys
import urllib.request
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

url_plan = "http://localhost:8085/api/chat/plan"
prompt = (
    "下のようなゲームを作成してください。元素を使って対戦カードゲーム！"
    "分子にすると強くなるぞ！…でもどこかで聞いたような気が…いいえ！本作最大の特徴は分子を合成できない点。"
    "並べたら手札のなかで勝手くっついてしまう。でも更にくっつきやすい元素が隣に来ると引っ付きなおす。"
    "熱、衝撃等の特殊カードも活かしてより安定した分子を作るもよし！有機系に振るのもよし！"
    "通常のカードゲームとは一風変わったデッキ構築が楽しめます。"
)

req = urllib.request.Request(
    url_plan,
    data=json.dumps({"prompt": prompt}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as resp:
    res = json.loads(resp.read().decode('utf-8'))
t_plan = time.time() - t0

print(f"Plan received in {t_plan:.1f}s, Spec ID: {res['spec_id']}")
dr = res.get("domain_rules", {})
print("\n--- Domain Rules Formulated ---")
if dr.get("elements_spec"):
    print("Elements Scope:", dr["elements_spec"].get("scope"))
    print("Master Table:", [(e.get("symbol"), e.get("valency"), e.get("affinity")) for e in dr["elements_spec"].get("master_table", [])])
if dr.get("binding_spec"):
    print("Binding Spec:", dr["binding_spec"])
if dr.get("deck_spec"):
    print("Deck Spec:", dr["deck_spec"])
if dr.get("victory_spec"):
    print("Victory Spec:", dr["victory_spec"])
if dr.get("ui_spec"):
    print("UI Spec:", dr["ui_spec"])

print(f"\n--- Goal Tree Nodes (Total: {len(res['tree_data']['nodes'])}) ---")
nodes = res["tree_data"]["nodes"]
for nid, n in nodes.items():
    tag = "ATOMIC (null)" if n["is_atomic"] else "BRANCH"
    print(f"  [{n['depth']}] {nid}: {n['name']} [{tag}]")
