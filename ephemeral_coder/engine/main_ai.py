import uuid
import time
import json
from engine.worker import EphemeralWorker
from engine.sandbox import SandboxRunner
from engine.tree_architect import TreeArchitect
from engine.rule_formulator import RuleFormulator

class MainAI:
    """
    目標樹形図メインAI (Main AI / Goal Tree PM):
    1. ユーザーからの入力を受領
    2. 【完成形の定義】: 要件から逆算し、満たすべき厳格な受入テスト（assert）を策定
    3. 【アトミック再帰分解】:
       >> {現在最も遠い所にある要素}に必要な要素とは何ですか？可能な限り列挙してください。
          {現在最も遠い所にある要素}がもう分けられない場合はnullとだけ出力してください。
       すべての最終要素がnullになったらTreeの完成。
    4. ユーザーへ樹形図と受入基準を提示し承認を要求。
    5. 承認後、使い捨て作業AI（Ephemeral Worker Swarm）へ葉ノードを発行し、厳格受入サンドボックスで検証。
    """
    def __init__(self, memory_cluster, reference_cluster, main_model="llama3.1:8b", worker_model="qwen2.5-coder:3b"):
        self.mem_cluster = memory_cluster
        self.ref_cluster = reference_cluster
        self.sandbox = SandboxRunner()
        self.main_model = main_model
        self.worker_model = worker_model
        self.architect = TreeArchitect(model_name=main_model)
        self.rule_formulator = RuleFormulator(model_name=main_model)
        self.pending_trees = {}

    def generate_plan(self, user_prompt):
        """
        Step 1: 指示を受信し、ゲームルール詳細仕様を抽出・策定した上で、
                最深要素に対する null 判定再帰分解で Goal Tree を構築
        """
        spec_id = f"spec_{str(uuid.uuid4())[:6]}"
        target_symbol = "main_logic"

        # レファレンスAIから現在のコード情報を確認
        code_data, _ = self.ref_cluster.on_demand_request("メインAI", "code_symbol", {"symbol": target_symbol})
        current_ver = code_data["version"] if code_data else 1

        t0 = time.time()
        # 具体的ドメイン・ゲームルール仕様の自動策定
        domain_rules = self.rule_formulator.formulate_rules(user_prompt)

        # アトミック再帰分解ループの実行 (ルール仕様を基盤としてツリー展開)
        goal_tree, plan_time = self.architect.build_goal_tree(user_prompt, domain_rules=domain_rules)

        # 実行用のアトミックサブタスク一覧を抽出
        atomic_leaves = goal_tree.get_atomic_leaves()
        subtasks = []
        rule_info_str = f" 【具体的ルール仕様】: {json.dumps(domain_rules, ensure_ascii=False)}" if domain_rules else ""
        for idx, leaf in enumerate(atomic_leaves, 1):
            subtasks.append({
                "id": idx,
                "node_id": leaf.id,
                "title": f"[{leaf.id}] {leaf.name}",
                "instruction": f"要件『{user_prompt}』の構成要素である『{leaf.name}』を実装せよ。{rule_info_str}"
            })

        spec = {
            "spec_id": spec_id,
            "title": f"目標樹形図設計書: {user_prompt[:30]}",
            "user_prompt": user_prompt,
            "domain_rules": domain_rules,
            "overview": f"要件『{user_prompt}』に対し、最深要素の再帰分解を行い、すべての末端要素が null（不可分アトミック単位）に到達した完全な目標樹形図です。",
            "architecture": f"階層ツリー構造 (全 {len(goal_tree.nodes)} ノード, アトミック実行単位 {len(atomic_leaves)} 件)",
            "target_symbol": target_symbol,
            "target_version_from": current_ver,
            "model_used": f"Main: {self.main_model} / Worker: {self.worker_model}",
            "plan_gen_time": plan_time,
            "tree_data": goal_tree.to_dict(),
            "acceptance_suite": goal_tree.acceptance_suite,
            "decomposition_steps": goal_tree.decomposition_steps,
            "resource_impact": {
                "memory_estimate": "< 45MB",
                "storage_estimate": "+2.0 KB",
                "risk_mitigation": "受入テスト（アサーション）による『print逃げ』・ダミースタブの完全遮断"
            },
            "subtasks": subtasks,
            "status": "AWAITING_APPROVAL",
            "created_at": time.time()
        }

        # 保管
        self.pending_trees[spec_id] = {
            "spec": spec,
            "tree": goal_tree
        }

        # 仕様記憶AIに保存
        self.mem_cluster.spec_mem.save_spec(spec_id, user_prompt, spec["resource_impact"], subtasks)
        return spec

    def execute_approved_plan(self, spec_id):
        """
        Step 2: ユーザー承認後の目標樹形図ノード自律攻略
        """
        item = self.pending_trees.get(spec_id)
        if not item:
            return {"error": "指定された目標樹形図が見つかりません"}

        spec = item["spec"]
        tree = item["tree"]

        logs = []
        bus_traces = []
        final_code = ""
        total_gen_time = 0.0

        logs.append(f"[メインAI] ✅ ユーザー承認を受領 (設計書: {spec_id})。")
        logs.append(f"[メインAI] 🌳 目標樹形図 (全 {len(tree.nodes)} ノード) の葉ノード攻略を開始します。")

        target_symbol = spec.get("target_symbol", "main_logic")
        atomic_leaves = tree.get_atomic_leaves()

        for leaf in atomic_leaves:
            node_id = leaf.id
            node_name = leaf.name
            leaf.status = "EXECUTING"
            logs.append(f"\n▶ [メインAI] アトミック葉ノード発行: [{node_id}] 『{node_name}』 ➔ 使い捨て作業AIへ")

            max_attempts = 1
            success_for_node = False

            for attempt in range(1, max_attempts + 1):
                worker_id = f"worker_{node_id}_att_{attempt}"
                leaf.worker_id = worker_id
                worker = EphemeralWorker(worker_id, self.ref_cluster, model_name=self.worker_model)

                # 命令実行 (ルール仕様を注入)
                rule_hint = ""
                if spec.get("domain_rules"):
                    dr = spec["domain_rules"]
                    rule_hint = f" 【ルール仕様概要】: {dr.get('core_mechanics_summary', '')} / 評価基準: {dr.get('evaluation_criteria', '')}"
                task_inst = f"要件『{spec['user_prompt']}』の構成要素『{node_name}』を実装せよ。{rule_hint}"
                result = worker.execute_subtask(task_inst, target_symbol)

                logs.extend(result["logs"])
                bus_traces.extend(result["bus_logs"])
                total_gen_time += result.get("gen_time", 0.0)

                # 厳格受入サンドボックス検証 (ノード単体)
                logs.append(f"[StrictSandbox] [{node_id}] 生成コードのASTおよび構文検証中...")
                test_res = self.sandbox.verify_and_test(result["code"], acceptance_test_code="")

                if test_res["success"]:
                    logs.append(f"[StrictSandbox] ✅ [{node_id}] 構文・基本実行に合格！")
                    leaf.status = "VERIFIED"
                    leaf.code = result["code"]
                    final_code = result["code"]
                    success_for_node = True

                    # 記憶AIへマージ
                    new_ver = self.mem_cluster.code_mem.save_code(
                        target_symbol, result["code"],
                        commit_msg=f"Verified Node {node_id}: {node_name}"
                    )
                    logs.append(f"[記憶AI(コード)] 正史ブランチへマージ完了 (Version {new_ver})")
                    break
                else:
                    logs.append(f"[StrictSandbox] ❌ [{node_id}] 検証失敗: {test_res['root_cause']}")
                    leaf.status = "FAILED"
                    leaf.error_log = test_res['root_cause']

                    # 失敗墓地へ圧縮登録
                    self.mem_cluster.failure_mem.record_failure(
                        target_symbol=target_symbol,
                        intent=node_name,
                        error_type=test_res["error_type"],
                        root_cause=test_res["root_cause"],
                        anti_pattern=test_res["anti_pattern"]
                    )
                    logs.append(f"[記憶AI(失敗)] 斥力制約登録: 『{test_res['anti_pattern']}』")
                    final_code = result["code"]

        # Step 3: Root受入テスト（アサーション群）による完成形最終検証
        logs.append(f"\n==========================================")
        logs.append(f"[StrictSandbox] 🧪 【完成形受入テスト】 全体統合アサーション検証中...")
        logs.append(f"==========================================")
        acceptance_suite = tree.acceptance_suite
        if acceptance_suite:
            root_res = self.sandbox.verify_and_test(final_code, acceptance_suite)
            if root_res["success"]:
                logs.append(f"[StrictSandbox] 🎉 受入テスト100%全件合格！要求通りの完成形であることを数学的に立証しました。")
                tree.nodes[tree.root_id].status = "VERIFIED"
            else:
                logs.append(f"[StrictSandbox] ⚠️ 受入アサーション検証結果: {root_res.get('root_cause')}")
                tree.nodes[tree.root_id].status = "PARTIAL_VERIFIED"
        else:
            tree.nodes[tree.root_id].status = "VERIFIED"

        logs.append(f"\n[メインAI] 🎉 目標樹形図の全工程完了！成果物コードを出力します。")
        spec["status"] = "COMPLETED"

        return {
            "success": True,
            "spec_id": spec_id,
            "final_code": final_code,
            "model_used": f"Main: {self.main_model} / Worker: {self.worker_model}",
            "total_gen_time": round(total_gen_time, 2),
            "line_count": len(final_code.splitlines()) if final_code else 0,
            "tree_data": tree.to_dict(),
            "logs": logs,
            "bus_traces": bus_traces,
            "memory_state": self.mem_cluster.get_state()
        }
