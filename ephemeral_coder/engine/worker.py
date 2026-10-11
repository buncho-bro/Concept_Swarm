import time

class EphemeralWorker:
    """
    作業AI (Ephemeral Worker):
    - 完全ステートレス。タスクの実行ごとに生成され、終了時に即座にコンテキスト破棄。
    - 過去の対話履歴やゴミを持たず、レファレンスAIから渡された
      『局所コード』と『過去の失敗の戒め（斥力・アンチパターン）』のみを入力とする。
    """
    def __init__(self, worker_id=None):
        self.worker_id = worker_id or f"worker_{int(time.time()*1000)%10000}"
        self.active_context = {}

    def execute_task(self, task_instruction, current_code, anti_patterns, style_rules):
        """
        タスクの実行
        """
        # 1. 最小限のコンテキストのみをメモリにバインド
        self.active_context = {
            "task": task_instruction,
            "code": current_code,
            "anti_patterns": anti_patterns,
            "style_rules": style_rules
        }
        
        # 思考ログ
        thought_log = []
        thought_log.append(f"[{self.worker_id}] 局所コンテキストを受信。クリーンなメモリ空間で起動。")
        thought_log.append(f"[{self.worker_id}] 適用されたスタイル制約: {style_rules}")
        
        if anti_patterns:
            thought_log.append(f"[{self.worker_id}] ⚠️ 過去の失敗（斥力）を検知！ 以下のアンチパターンを強制回避します:")
            for ap in anti_patterns:
                thought_log.append(f"    - 忌避制約: {ap['anti_pattern']}")
        else:
            thought_log.append(f"[{self.worker_id}] 過去の失敗履歴はありません。新規アプローチで実装します。")

        # 2. コード生成（ヒューリスティック・生成エンジン）
        # 過去の失敗（斥力）があるかどうかに応じて生成コードの安全策・アルゴリズムが変化する
        generated_code = self._generate_solution(task_instruction, current_code, anti_patterns)
        thought_log.append(f"[{self.worker_id}] コード生成完了。")

        # 3. 最重要: コンテキストの完全抹消 (Wipe)
        self._wipe_context()
        thought_log.append(f"[{self.worker_id}] ⚡ 作業完了。コンテキストを完全に消去（メモリ解放）しました。")

        return {
            "code": generated_code,
            "logs": thought_log
        }

    def _generate_solution(self, task, code, anti_patterns):
        """
        アンチパターン（過去の失敗）を回避するようにコードを生成・リファクタリング
        """
        has_zero_div_anti = any("ゼロ除算" in ap.get("anti_pattern", "") for ap in anti_patterns)
        has_type_anti = any("型" in ap.get("anti_pattern", "") for ap in anti_patterns)
        
        # タスク内容に応じた適応
        t = task.lower()
        if "集計" in t or "平均" in t or "average" in t or "統計" in t:
            if has_zero_div_anti:
                return (
                    "def process_data(data_list):\n"
                    "    # 過去のゼロ除算失敗を回避した堅牢な実装\n"
                    "    valid_data = [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
                    "    if not valid_data:\n"
                    "        return {'count': 0, 'total': 0, 'average': 0.0}\n"
                    "    total = sum(valid_data)\n"
                    "    avg = total / len(valid_data)\n"
                    "    return {'count': len(valid_data), 'total': total, 'average': avg}\n"
                )
            else:
                # 初回は敢えてエッジケース（空リストでゼロ除算になる）を含む可能性のあるシンプルなコード
                return (
                    "def process_data(data_list):\n"
                    "    valid_data = [x for x in data_list if x > 0]\n"
                    "    return sum(valid_data) / len(valid_data) # 平均値\n"
                )
                
        elif "ソート" in t or "並び替え" in t or "sort" in t:
            return (
                "def process_data(data_list):\n"
                "    # 昇順ソートと正数フィルタリング\n"
                "    filtered = [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
                "    return sorted(filtered)\n"
            )
        elif "倍" in t or "2倍" in t or "transform" in t:
            return (
                "def process_data(data_list):\n"
                "    # 各要素を2倍にする変換\n"
                "    return [x * 2 for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
            )
        else:
            # 汎用的な機能追加
            return (
                "def process_data(data_list):\n"
                "    # フィルタリングとメタ情報付与\n"
                "    filtered = [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n"
                "    return {'status': 'success', 'data': filtered, 'count': len(filtered)}\n"
            )

    def _wipe_context(self):
        """コンテキストを抹消し、過去のしがらみを断ち切る"""
        self.active_context.clear()
        self.active_context = None
