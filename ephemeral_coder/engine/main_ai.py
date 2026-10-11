import uuid
import time
import json
import re
from engine.worker import EphemeralWorker
from engine.sandbox import SandboxRunner
from engine.ollama_client import query_ollama

class MainAI:
    """
    メインAI (Main AI / PM):
    1. ユーザーからの入力を受領
    2. ローカルLLMを活用して要件を分析し、詳細な設計書（アーキテクチャ・リスク・サブタスク分割）を作成
    3. ユーザーへ提示し、承認を要求
    4. 承認後、分割された指示を作業AIへディスパッチして進捗を監督
    """
    def __init__(self, memory_cluster, reference_cluster, model_name="qwen2.5-coder:3b"):
        self.mem_cluster = memory_cluster
        self.ref_cluster = reference_cluster
        self.sandbox = SandboxRunner()
        self.model_name = model_name
        self.pending_specs = {}

    def generate_plan(self, user_prompt):
        """
        Step 1: 指示を受信し、ローカルLLMを使って詳細な設計書とタスク分割を作成
        """
        spec_id = f"spec_{str(uuid.uuid4())[:6]}"
        target_symbol = "main_logic"

        # レファレンスAIから現在のコード情報を事前取得
        code_data, _ = self.ref_cluster.on_demand_request("メインAI", "code_symbol", {"symbol": target_symbol})
        current_code = code_data["code"] if code_data else ""
        current_ver = code_data["version"] if code_data else 1

        # レファレンスAIから過去の失敗教訓も事前確認（設計時のリスク判定用）
        fail_data, _ = self.ref_cluster.on_demand_request("メインAI", "failure_history", {"target_symbol": target_symbol})
        known_anti_patterns = [f["anti_pattern"] for f in (fail_data or [])]

        t0 = time.time()
        # ローカルLLMに本格的なアーキテクトとして設計書を作成させる
        plan_data = self._call_llm_architect(user_prompt, current_code, known_anti_patterns)
        plan_time = round(time.time() - t0, 2)

        spec = {
            "spec_id": spec_id,
            "title": plan_data.get("title", f"設計書: {user_prompt[:30]}"),
            "user_prompt": user_prompt,
            "overview": plan_data.get("overview", "要件に基づく詳細設計"),
            "architecture": plan_data.get("architecture", "モジュール構造と関数の定義"),
            "target_symbol": target_symbol,
            "target_version_from": current_ver,
            "model_used": self.model_name,
            "plan_gen_time": plan_time,
            "resource_impact": plan_data.get("resource_impact", {
                "memory_estimate": "< 50MB",
                "storage_estimate": "+1.5 KB",
                "risk_mitigation": "エッジケースのバリデーションと防御的コーディング"
            }),
            "subtasks": plan_data.get("subtasks", [
                {"id": 1, "title": f"要件『{user_prompt}』のコア実装", "instruction": user_prompt}
            ]),
            "status": "AWAITING_APPROVAL",
            "created_at": time.time()
        }

        self.pending_specs[spec_id] = spec
        # 仕様記憶AIに保存
        self.mem_cluster.spec_mem.save_spec(spec_id, user_prompt, spec["resource_impact"], spec["subtasks"])
        return spec

    def _call_llm_architect(self, user_prompt, current_code, known_anti_patterns):
        """ローカルLLMへアーキテクト用プロンプトを投入し、構造化JSON設計書を生成"""
        system_prompt = (
            "You are a Principal Software Architect and Project Manager (Main AI).\n"
            "Analyze the user's requirements, evaluate existing code, and produce a formal, production-grade Software Design Specification and task decomposition.\n"
            "CRITICAL: You must output ONLY a valid JSON object matching the exact schema below, with no conversational preamble or markdown formatting:\n"
            "{\n"
            '  "title": "明確な設計書タイトル(日本語)",\n'
            '  "overview": "要件の分析、設計思想、アルゴリズム選定理由(日本語 2〜3文)",\n'
            '  "architecture": "採用するクラス・関数構成、データ構造、入出力仕様(日本語)",\n'
            '  "resource_impact": {\n'
            '    "memory_estimate": "推定メモリ使用量 (例: < 30MB)",\n'
            '    "storage_estimate": "推定追加ストレージ (例: +2.0 KB)",\n'
            '    "risk_mitigation": "想定されるバグやリスクと、その回避方針(日本語)"\n'
            "  },\n"
            '  "subtasks": [\n'
            '    {"id": 1, "title": "サブタスク1の命令タイトル(日本語)", "instruction": "作業AIへの具体的実装指示(日本語)"},\n'
            '    {"id": 2, "title": "サブタスク2の命令タイトル(日本語)", "instruction": "作業AIへの具体的実装指示(日本語)"}\n'
            "  ]\n"
            "}"
        )

        anti_txt = "\n".join([f"- {ap}" for ap in known_anti_patterns]) if known_anti_patterns else "None"
        prompt = (
            f"# USER REQUIREMENT:\n{user_prompt}\n\n"
            f"# CURRENT CODEBASE CONTEXT:\n{current_code}\n\n"
            f"# KNOWN PAST FAILURES TO AVOID IN ARCHITECTURE:\n{anti_txt}\n\n"
            f"Generate the complete JSON Design Specification now:"
        )

        raw = query_ollama(self.model_name, prompt=prompt, system_prompt=system_prompt, timeout=90)
        
        if raw:
            # JSONブロックの抽出 (```json ... ``` または 生の { ... })
            match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if "subtasks" in data and isinstance(data["subtasks"], list):
                        # 各フィールドがオブジェクトやリストの場合は文字列化して正規化
                        for key in ["title", "overview", "architecture"]:
                            if key in data and not isinstance(data[key], str):
                                data[key] = json.dumps(data[key], ensure_ascii=False, indent=2)
                        
                        if "resource_impact" in data and isinstance(data["resource_impact"], dict):
                            for r_key in ["memory_estimate", "storage_estimate", "risk_mitigation"]:
                                if r_key in data["resource_impact"] and not isinstance(data["resource_impact"][r_key], str):
                                    data["resource_impact"][r_key] = json.dumps(data["resource_impact"][r_key], ensure_ascii=False)
                        
                        return data
                except Exception as e:
                    print(f"[Main AI JSON Parse Error] {e}")

        # フォールバック（安全なデフォルト分解）
        return {
            "title": f"機能設計: {user_prompt[:25]}",
            "overview": f"指示『{user_prompt}』を満たすため、モジュール構造を定義し安全な実行環境を構築します。",
            "architecture": "単一責任原則に基づき、入力検証関数とメイン処理クラスを分離実装します。",
            "resource_impact": {
                "memory_estimate": "< 40MB",
                "storage_estimate": "+1.0 KB",
                "risk_mitigation": "型不一致および空入力に対する防御的ガードを配置"
            },
            "subtasks": [
                {"id": 1, "title": "コア機能および主要アルゴリズムの実装", "instruction": user_prompt},
                {"id": 2, "title": "エッジケース保護・型検証および例外ハンドリングの統合", "instruction": "空データや不正入力時の耐性を強化"}
            ]
        }

    def execute_approved_plan(self, spec_id):
        """
        Step 2: ユーザー承認後の作業命令ディスパッチ
        """
        spec = self.pending_specs.get(spec_id)
        if not spec:
            return {"error": "指定された設計書が見つかりません"}

        logs = []
        bus_traces = []
        final_code = ""
        total_gen_time = 0.0
        logs.append(f"[メインAI] ✅ ユーザー承認を受領 (設計書: {spec_id})。作業命令を開始します。")

        target_symbol = spec.get("target_symbol", "main_logic")

        for task in spec["subtasks"]:
            task_title = task["title"]
            task_instruction = task.get("instruction", task_title)
            logs.append(f"\n▶ [メインAI] サブタスク発行: 『{task_title}』 ➔ 作業AIへ")

            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                worker_id = f"worker_{task['id']}_att_{attempt}"
                worker = EphemeralWorker(worker_id, self.ref_cluster, model_name=self.model_name)
                
                result = worker.execute_subtask(task_instruction, target_symbol)
                logs.extend(result["logs"])
                bus_traces.extend(result["bus_logs"])
                total_gen_time += result.get("gen_time", 0.0)

                # サンドボックス検証
                logs.append(f"[Sandbox] 生成コードのAST検証中...")
                test_res = self.sandbox.verify_and_test(result["code"])

                if test_res["success"]:
                    logs.append(f"[Sandbox] ✅ 検証成功！ 構文・テストに合格しました。")
                    new_ver = self.mem_cluster.code_mem.save_code(target_symbol, result["code"], commit_msg=f"Approved Task: {task_title}")
                    logs.append(f"[記憶AI(コード)] 正史ブランチにマージ (Version {new_ver})")
                    final_code = result["code"]
                    break
                else:
                    logs.append(f"[Sandbox] ❌ 検証失敗: {test_res['root_cause']}")
                    logs.append(f"[記憶AI(失敗)] 失敗要因を圧縮して『失敗墓地（斥力）』へ登録中...")
                    
                    self.mem_cluster.failure_mem.record_failure(
                        target_symbol=target_symbol,
                        intent=task_title,
                        error_type=test_res["error_type"],
                        root_cause=test_res["root_cause"],
                        anti_pattern=test_res["anti_pattern"]
                    )
                    logs.append(f"[記憶AI(失敗)] 登録完了: 忌避制約『{test_res['anti_pattern']}』")
                    final_code = result["code"]

        logs.append(f"\n[メインAI] 🎉 全サブタスクの実行完了。設計書 {spec_id} の全工程を完了しました。")
        spec["status"] = "COMPLETED"

        return {
            "success": True,
            "spec_id": spec_id,
            "final_code": final_code,
            "model_used": self.model_name,
            "total_gen_time": round(total_gen_time, 2),
            "line_count": len(final_code.splitlines()) if final_code else 0,
            "logs": logs,
            "bus_traces": bus_traces,
            "memory_state": self.mem_cluster.get_state()
        }
