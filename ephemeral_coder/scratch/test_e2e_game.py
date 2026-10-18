import sys
import os
import urllib.request
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_e2e():
    url_plan = "http://localhost:8085/api/chat/plan"
    url_approve = "http://localhost:8085/api/chat/approve"
    
    prompt = (
        "下のようなゲームを作成してください。元素を使って対戦カードゲーム！"
        "分子にすると強くなるぞ！…でもどこかで聞いたような気が…いいえ！本作最大の特徴は分子を合成できない点。"
        "並べたら手札のなかで勝手くっついてしまう。でも更にくっつきやすい元素が隣に来ると引っ付きなおす。"
        "熱、衝撃等の特殊カードも活かしてより安定した分子を作るもよし！有機系に振るのもよし！"
        "通常のカードゲームとは一風変わったデッキ構築が楽しめます。"
    )
    
    print("=== 1. Requesting Goal Tree Plan via API ===")
    req = urllib.request.Request(
        url_plan,
        data=json.dumps({"prompt": prompt}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        plan_res = json.loads(resp.read().decode('utf-8'))
    t_plan = time.time() - t0
    
    spec_id = plan_res["spec_id"]
    print(f"Plan received in {t_plan:.1f}s, Spec ID: {spec_id}")
    print(f"Title: {plan_res['title']}")
    
    # Check domain rules
    dr = plan_res.get("domain_rules")
    print("\n--- Domain Rules ---")
    if dr:
        print(f"Rule Title: {dr.get('title')}")
        print(f"Core Mechanics Summary: {dr.get('core_mechanics_summary')}")
        entities = dr.get("entities") or dr.get("elements", [])
        print(f"Entities/Cards count: {len(entities)}")
        for e in entities:
            print(f"  - {e.get('name')}: {e.get('attributes')} ({e.get('role', '')})")
        transitions = dr.get("state_transitions") or []
        print(f"State Transitions: {len(transitions)}")
        for st in transitions:
            print(f"  - [{st.get('event')}] Trigger: {st.get('trigger')} -> Action: {st.get('action')}")
        specials = dr.get("special_rules") or dr.get("special_cards", [])
        print(f"Special Rules: {len(specials)}")
        for sp in specials:
            print(f"  - {sp.get('name')}: {sp.get('effect')}")
        print(f"Evaluation Criteria: {dr.get('evaluation_criteria') or dr.get('win_condition')}")
    else:
        print("WARNING: No domain_rules found!")
        
    # Check nodes for platitudes
    nodes = plan_res["tree_data"]["nodes"]
    print(f"\n--- Goal Tree Nodes (Total: {len(nodes)}) ---")
    platitudes = ["明確さ", "一貫性", "品質", "管理の徹底"]
    platitude_found = []
    for nid, node in nodes.items():
        node_type = 'ATOMIC (null)' if node['is_atomic'] else 'BRANCH'
        print(f"  {nid}: {node['name']} [{node_type}]")
        for p in platitudes:
            if p in node['name']:
                platitude_found.append((nid, node['name'], p))
                
    if platitude_found:
        print(f"\n[FAILED] Platitudes detected: {platitude_found}")
    else:
        print(f"\n[PASSED] No platitudes found! Concrete game mechanics generated.")
        
    # Check acceptance suite
    print(f"\n--- Acceptance Suite ---")
    suite = plan_res.get("acceptance_suite", "")
    print(suite[:300] + ("..." if len(suite) > 300 else ""))
    
    # Test approval execution
    print(f"\n=== 2. Approving Plan and Launching Worker Swarm ===")
    req_app = urllib.request.Request(
        url_approve,
        data=json.dumps({"spec_id": spec_id}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    t1 = time.time()
    with urllib.request.urlopen(req_app, timeout=600) as resp:
        app_res = json.loads(resp.read().decode('utf-8'))
    t_exec = time.time() - t1
    print(f"Approval execution completed in {t_exec:.1f}s")
    print(f"Total lines: {app_res.get('line_count')}")
    print("Logs snippet:")
    for log in app_res.get("logs", [])[-10:]:
        print("  ", log)
        
    print("\nFinal Code Snippet:")
    final_code = app_res.get("final_code", "")
    print(final_code[:400] + "\n...")

    # Save to file
    with open("scratch/e2e_result.json", "w", encoding="utf-8") as f:
        json.dump({"plan": plan_res, "execution": app_res}, f, ensure_ascii=False, indent=2)
    print("\n[SUCCESS] Saved full E2E result to scratch/e2e_result.json")

if __name__ == '__main__':
    test_e2e()
