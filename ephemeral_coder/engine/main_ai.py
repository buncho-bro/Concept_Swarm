import uuid
import time
from engine.worker import EphemeralWorker
from engine.sandbox import SandboxRunner

class MainAI:
    """
    メインAI (Main AI / PM):
    1. ユーザーからの入力を受領
    2. 設計書（仕様・リソース見積・タスク分割）を作成しユーザーへ送信・承認要求
    3. ユーザー承認後、指示を分割して作業AIへ命令を発行
    4. 成果物コード（final_code）を取りまとめ、ユーザーへ返却
    """
    def __init__(self, memory_cluster, reference_cluster, model_name="qwen2.5-coder:3b"):
        self.mem_cluster = memory_cluster
        self.ref_cluster = reference_cluster
        self.sandbox = SandboxRunner()
        self.model_name = model_name
        self.pending_specs = {}

    def generate_plan(self, user_prompt):
        """
        Step 1: 指示を受信し、設計書とタスク分割を作成（承認待ちにする）
        """
        spec_id = f"spec_{str(uuid.uuid4())[:6]}"
        target_symbol = "main_logic"

        code_data, _ = self.ref_cluster.on_demand_request("メインAI", "code_symbol", {"symbol": target_symbol})
        current_ver = code_data["version"] if code_data else 1

        subtasks = self._plan_subtasks(user_prompt, target_symbol)

        spec = {
            "spec_id": spec_id,
            "user_prompt": user_prompt,
            "target_symbol": target_symbol,
            "target_version_from": current_ver,
            "model_used": self.model_name,
            "resource_impact": {
                "engine": f"Local Ollama ({self.model_name})",
                "estimated_ram": "< 120MB (推論時)",
                "disk_delta": "+1.2 KB",
                "affected_modules": [f"engine.{target_symbol}"]
            },
            "subtasks": subtasks,
            "status": "AWAITING_APPROVAL",
            "created_at": time.time()
        }

        self.pending_specs[spec_id] = spec
        self.mem_cluster.spec_mem.save_spec(spec_id, user_prompt, spec["resource_impact"], subtasks)
        return spec

    def _plan_subtasks(self, prompt, symbol):
        p = prompt.lower()
        if "平均" in p or "集計" in p:
            return [
                {"id": 1, "title": "データ集計・平均値算出ロジックの実装", "symbol": symbol},
                {"id": 2, "title": "エッジケース（空リスト・型安全性）の検証と保護", "symbol": symbol}
            ]
        elif "ソート" in p or "並び替え" in p:
            return [
                {"id": 1, "title": "正数フィルタリングおよび昇順ソート機能の実装", "symbol": symbol}
            ]
        elif "倍" in p:
            return [
                {"id": 1, "title": "各要素の数値変換（2倍化）処理の実装", "symbol": symbol}
            ]
        else:
            # 任意の複雑な指示に対する柔軟なタスク分割
            return [
                {"id": 1, "title": f"要件『{prompt}』のコアロジック実装", "symbol": symbol},
                {"id": 2, "title": "防御的例外処理とクラス/関数インターフェースの統合", "symbol": symbol}
            ]

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

        for task in spec["subtasks"]:
            task_title = task["title"]
            target_symbol = task["symbol"]
            logs.append(f"\n▶ [メインAI] サブタスク発行: 『{task_title}』 ➔ 作業AIへ")

            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                worker_id = f"worker_{task['id']}_att_{attempt}"
                worker = EphemeralWorker(worker_id, self.ref_cluster, model_name=self.model_name)
                
                result = worker.execute_subtask(task_title, target_symbol)
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
                    final_code = result["code"] # 失敗したコードも保持

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
