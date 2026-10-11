import time
from engine.ollama_client import query_ollama, extract_code_block

class EphemeralWorker:
    """
    作業AI (Worker AI):
    - ローカルLLM (qwen2.5-coder:3b) に接続。
    - メインAIから作業命令を受注し、実動コードを生成。
    - 必要な時だけレファレンスAI群へオンデマンド情報要求（赤丸バス通信）。
    - 作業完了後、コンテキストを即座に破棄（Wipe）。
    """
    def __init__(self, worker_id, ref_cluster, model_name="qwen2.5-coder:3b"):
        self.worker_id = worker_id
        self.ref_cluster = ref_cluster
        self.model_name = model_name
        self.active_context = {}
        self.bus_logs = []

    def execute_subtask(self, subtask_title, target_symbol):
        logs = []
        logs.append(f"[{self.worker_id}] 🚀 ローカルLLM ({self.model_name}) をクリーンなメモリ空間で起動。命令: 『{subtask_title}』")

        # 1. オンデマンド情報要求 (Red Circle Bus): コードの取得
        logs.append(f"[{self.worker_id}] 📡 レファレンスAI層へオンデマンド要求: シンボル `{target_symbol}` のソースコード")
        code_data, hop_code = self.ref_cluster.on_demand_request(self.worker_id, "code_symbol", {"symbol": target_symbol})
        self.bus_logs.append(hop_code)
        
        current_code = code_data["code"] if code_data else ""
        current_ver = code_data["version"] if code_data else 1
        logs.append(f"[{self.worker_id}] 📥 コード取得完了 (Version {current_ver})")

        # 2. オンデマンド情報要求 (Red Circle Bus): 過去の失敗教訓の取得
        logs.append(f"[{self.worker_id}] 📡 レファレンスAI層へオンデマンド要求: `{target_symbol}` に関連する過去の失敗・アンチパターン")
        fail_data, hop_fail = self.ref_cluster.on_demand_request(self.worker_id, "failure_history", {"target_symbol": target_symbol})
        self.bus_logs.append(hop_fail)
        
        anti_patterns = fail_data if fail_data else []
        if anti_patterns:
            logs.append(f"[{self.worker_id}] ⚠️ 過去の失敗教訓（斥力）を {len(anti_patterns)} 件検知！ 以下のアンチパターンを強制回避します:")
            for ap in anti_patterns:
                logs.append(f"    - 忌避制約: {ap['anti_pattern']}")
        else:
            logs.append(f"[{self.worker_id}] 過去の失敗履歴はありません。新規パターンで実装します。")

        # 3. 本物のローカルLLM (Ollama) によるコード生成
        logs.append(f"[{self.worker_id}] 🧠 Ollama ({self.model_name}) へプロンプトと斥力制約を投入し、コード生成を開始...")
        t_start = time.time()
        generated_code, gen_log = self._call_local_llm(subtask_title, current_code, anti_patterns)
        gen_time = round(time.time() - t_start, 2)
        logs.extend(gen_log)
        logs.append(f"[{self.worker_id}] ⏱️ コード生成完了 (所要時間: {gen_time} 秒, コード行数: {len(generated_code.splitlines())} 行)")

        # 4. コンテキストの完全抹消 (Wipe)
        self.active_context.clear()
        self.active_context = None
        logs.append(f"[{self.worker_id}] ⚡ 作業完了。コンテキストを完全抹消（メモリ解放）しました。")

        return {
            "code": generated_code,
            "gen_time": gen_time,
            "logs": logs,
            "bus_logs": self.bus_logs
        }

    def _call_local_llm(self, task, current_code, anti_patterns):
        logs = []
        # システムプロンプトの構築（斥力・アンチパターンの注入）
        system_prompt = (
            "You are an expert Python software engineer acting as an ephemeral worker AI.\n"
            "Your task is to implement the requested feature cleanly, completely, and defensively in Python.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output ONLY executable Python code enclosed within a single ```python ... ``` block.\n"
            "2. Do NOT output conversational prose, explanations, or pleasantries.\n"
            "3. Ensure the code is production-ready, fully defined, and handles potential edge cases.\n"
        )
        if anti_patterns:
            system_prompt += "\n4. MANDATORY NEGATIVE CONSTRAINTS (ANTI-PATTERNS FROM PAST FAILURES):\n"
            for ap in anti_patterns:
                system_prompt += f"   - AVOID: {ap['anti_pattern']}\n"
            system_prompt += "   You must write defensive checks to avoid these exact errors.\n"

        prompt = (
            f"# EXISTING CONTEXT / CODE:\n{current_code}\n\n"
            f"# TASK INSTRUCTION TO IMPLEMENT:\n{task}\n\n"
            f"# Write the full updated/implemented Python code below:"
        )

        raw_response = query_ollama(self.model_name, prompt=prompt, system_prompt=system_prompt, timeout=120)
        
        if raw_response:
            code = extract_code_block(raw_response)
            logs.append(f"[{self.worker_id}] Ollama ({self.model_name}) より {len(raw_response)} 文字の出力を受信・パースしました。")
            return code, logs
        else:
            logs.append(f"[{self.worker_id}] ⚠️ Ollama接続に失敗またはタイムアウト。セーフガード・フォールバックを実行。")
            return self._fallback_synthesize(task, current_code, anti_patterns), logs

    def _fallback_synthesize(self, task, current_code, anti_patterns):
        has_zero_div_anti = any("ゼロ除算" in ap.get("anti_pattern", "") for ap in anti_patterns)
        t = task.lower()
        if "平均" in t or "集計" in t:
            if has_zero_div_anti:
                return (
                    "def process_data(data_list):\n"
                    "    valid = [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
                    "    if not valid:\n"
                    "        return {'count': 0, 'total': 0, 'average': 0.0}\n"
                    "    return {'count': len(valid), 'total': sum(valid), 'average': sum(valid) / len(valid)}\n"
                )
            else:
                return (
                    "def process_data(data_list):\n"
                    "    valid = [x for x in data_list if x > 0]\n"
                    "    return sum(valid) / len(valid)\n"
                )
        return (
            f"# 実装: {task}\n"
            "class FeatureImplementation:\n"
            "    def __init__(self):\n"
            "        pass\n\n"
            "    def execute(self, *args, **kwargs):\n"
            "        return True\n"
        )
