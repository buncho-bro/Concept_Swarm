import time

class EphemeralWorker:
    """
    作業AI (Worker AI):
    - メインAIから作業命令を受注。
    - 必要な時だけレファレンスAI群へオンデマンド情報要求（赤丸バス通信）。
    - 作業完了後、コンテキストを即座に破棄（Wipe）。
    """
    def __init__(self, worker_id, ref_cluster):
        self.worker_id = worker_id
        self.ref_cluster = ref_cluster
        self.active_context = {}
        self.bus_logs = []

    def execute_subtask(self, subtask_title, target_symbol):
        logs = []
        logs.append(f"[{self.worker_id}] 🚀 クリーンなメモリで起動。命令: 『{subtask_title}』")

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
            logs.append(f"[{self.worker_id}] ⚠️ 過去の失敗教訓を {len(anti_patterns)} 件検知。以下のアンチパターンを強制回避します:")
            for ap in anti_patterns:
                logs.append(f"    - 忌避制約: {ap['anti_pattern']}")
        else:
            logs.append(f"[{self.worker_id}] 過去の失敗履歴はありません。新規パターンで実装します。")

        # 3. コード生成
        generated_code = self._synthesize_code(subtask_title, current_code, anti_patterns)
        logs.append(f"[{self.worker_id}] コード生成完了。")

        # 4. コンテキストの完全抹消 (Wipe)
        self.active_context.clear()
        self.active_context = None
        logs.append(f"[{self.worker_id}] ⚡ 作業完了。コンテキストを完全抹消（メモリ解放）しました。")

        return {
            "code": generated_code,
            "logs": logs,
            "bus_logs": self.bus_logs
        }

    def _synthesize_code(self, task, current_code, anti_patterns):
        has_zero_div_anti = any("ゼロ除算" in ap.get("anti_pattern", "") for ap in anti_patterns)
        t = task.lower()

        if "平均" in t or "集計" in t or "統計" in t:
            if has_zero_div_anti:
                return (
                    "def process_data(data_list):\n"
                    "    # ゼロ除算・空データ耐性を備えた集計処理\n"
                    "    valid = [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
                    "    if not valid:\n"
                    "        return {'count': 0, 'total': 0, 'average': 0.0}\n"
                    "    return {'count': len(valid), 'total': sum(valid), 'average': sum(valid) / len(valid)}\n"
                )
            else:
                # 初回は素朴に平均を割る（空リストでゼロ除算になるリスクあり）
                return (
                    "def process_data(data_list):\n"
                    "    valid = [x for x in data_list if x > 0]\n"
                    "    return sum(valid) / len(valid)\n"
                )
        elif "ソート" in t or "並び替え" in t:
            return (
                "def process_data(data_list):\n"
                "    # 昇順ソート実装\n"
                "    valid = [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
                "    return sorted(valid)\n"
            )
        elif "2倍" in t or "倍" in t:
            return (
                "def process_data(data_list):\n"
                "    # 要素変換処理\n"
                "    return [x * 2 for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
            )
        else:
            return (
                "def process_data(data_list):\n"
                "    # フィルタリングと拡張メタデータ\n"
                "    valid = [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
                "    return {'processed': valid, 'count': len(valid)}\n"
            )
