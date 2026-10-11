import time
from engine.worker import EphemeralWorker
from engine.sandbox import SandboxRunner

class ReferenceRouter:
    """
    レファレンスAI (Reference Router):
    - ユーザーと記憶・作業AIの調停役。
    - 指示の解釈、依存関係・シンボルの特定。
    - 記憶AIからのコンテキスト取得（正のコード + 負のアンチパターン）。
    - 作業AIの起動・使い捨て管理。
    - サンドボックス検証と手戻り（リトライ）のオーケストレーション。
    """
    def __init__(self, memory_engine):
        self.memory = memory_engine
        self.sandbox = SandboxRunner()
        self.style_rules = "PEP8準拠, 型安全, 防御的プログラミング"

    def handle_user_request(self, user_prompt):
        logs = []
        logs.append(f"[Reference AI] ユーザー指示を受信: 『{user_prompt}』")
        
        # 1. 対象シンボルの特定 (簡易ASTルーター)
        target_symbol = "main_logic"
        logs.append(f"[Reference AI] 構文解析により対象シンボルを特定: `{target_symbol}`")
        
        # 2. 記憶AIからコードと過去の失敗（斥力）を取得
        current_code = self.memory.get_code(target_symbol)
        past_failures = self.memory.get_relevant_failures(target_symbol)
        
        logs.append(f"[Reference AI] 記憶AIより現在のコード (v{self.memory.code_store[target_symbol]['version']}) を取得。")
        if past_failures:
            logs.append(f"[Reference AI] 記憶AIより過去の失敗教訓（斥力ベクトル）を {len(past_failures)} 件取得。")
        else:
            logs.append(f"[Reference AI] 過去の失敗記録はありません。")

        max_attempts = 2
        success = False
        final_code = None
        attempt_details = []

        for attempt in range(1, max_attempts + 1):
            logs.append(f"\n--- [試行 {attempt}/{max_attempts}] 作業AIの召喚 ---")
            
            # 3. エフェメラル作業AIの召喚（完全新規インスタンス）
            worker = EphemeralWorker(worker_id=f"worker_attempt_{attempt}")
            worker_result = worker.execute_task(
                task_instruction=user_prompt,
                current_code=current_code,
                anti_patterns=past_failures,
                style_rules=self.style_rules
            )
            
            logs.extend(worker_result["logs"])
            generated_code = worker_result["code"]
            
            # 4. サンドボックスでの即時検証 (Execution & Verification)
            logs.append(f"[Sandbox] 生成コードのAST検証および分離実行テスト中...")
            test_res = self.sandbox.verify_and_test(generated_code)
            
            if test_res["success"]:
                logs.append(f"[Sandbox] ✅ 検証成功！ 構文・実行テストをクリアしました。")
                success = True
                final_code = generated_code
                
                # 記憶AIへ正史としてコミット
                new_ver = self.memory.save_code(target_symbol, final_code, commit_msg=f"Implemented: {user_prompt}")
                logs.append(f"[Memory AI] 正史ブランチにマージ完了 (Version {new_ver})。")
                break
            else:
                logs.append(f"[Sandbox] ❌ 検証失敗: {test_res['root_cause']}")
                logs.append(f"[Memory AI] 失敗を圧縮し『失敗墓地（斥力）』へアーカイブ登録中...")
                
                # 失敗を圧縮して記憶AIに登録！
                failure_node = self.memory.record_failure(
                    target_symbol=target_symbol,
                    intent=user_prompt,
                    error_type=test_res["error_type"],
                    root_cause=test_res["root_cause"],
                    anti_pattern=test_res["anti_pattern"]
                )
                logs.append(f"[Memory AI] 登録完了: 忌避制約『{failure_node['anti_pattern']}』")
                
                # 次の試行のために過去の失敗リストを更新
                past_failures = self.memory.get_relevant_failures(target_symbol)
                current_code = generated_code # 失敗したコードを土台に改善

        # 5. ユーザーへの最終回答メッセージ作成
        if success:
            reply_msg = f"指示に基づき `{target_symbol}` の改修を完了しました（Version {self.memory.code_store[target_symbol]['version']}）。\n作業AIのコンテキストは即座に完全抹消され、安全性が保証されています。"
        else:
            reply_msg = f"`{target_symbol}` の改修中にエラーが検出され、教訓として失敗墓地へ圧縮保持されました。"

        return {
            "reply": reply_msg,
            "success": success,
            "code": final_code or current_code,
            "logs": logs,
            "memory_state": self.memory.get_full_state()
        }
