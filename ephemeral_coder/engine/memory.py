import time
import uuid

class MemoryEngine:
    """
    記憶AI (Memory AI):
    1. コード実体をシンボル/関数単位でDAG・バージョン管理
    2. 失敗の記憶を破棄せず、「死後解剖（ポストモーテム）情報」として圧縮保持
    """
    def __init__(self):
        # symbol -> { "code": str, "version": int, "history": [ {version, code, msg, timestamp} ] }
        self.code_store = {
            "main_logic": {
                "symbol": "main_logic",
                "code": "def process_data(data_list):\n    # TODO: 効率的なデータ集計とフィルタリング\n    return [x for x in data_list if x > 0]\n",
                "version": 1,
                "history": [
                    {
                        "version": 1,
                        "code": "def process_data(data_list):\n    # TODO: 効率的なデータ集計とフィルタリング\n    return [x for x in data_list if x > 0]\n",
                        "msg": "Initial stub implementation",
                        "timestamp": time.time()
                    }
                ]
            }
        }
        
        # 圧縮された失敗の墓地 (Failure Graveyard)
        # 過去の失敗を教訓（アンチパターン・斥力）として蓄積
        self.failure_graveyard = []

    def save_code(self, symbol, code, commit_msg="Update code"):
        if symbol not in self.code_store:
            self.code_store[symbol] = {
                "symbol": symbol,
                "code": code,
                "version": 1,
                "history": []
            }
        else:
            self.code_store[symbol]["version"] += 1
            self.code_store[symbol]["code"] = code
            
        entry = {
            "version": self.code_store[symbol]["version"],
            "code": code,
            "msg": commit_msg,
            "timestamp": time.time()
        }
        self.code_store[symbol]["history"].append(entry)
        return self.code_store[symbol]["version"]

    def get_code(self, symbol, version=None):
        if symbol not in self.code_store:
            return None
        if version is None or version == self.code_store[symbol]["version"]:
            return self.code_store[symbol]["code"]
        # 履歴から特定バージョンを取得
        for h in self.code_store[symbol]["history"]:
            if h["version"] == version:
                return h["code"]
        return None

    def record_failure(self, target_symbol, intent, error_type, root_cause, anti_pattern):
        """
        失敗したコード実体は圧縮し、教訓・アンチパターンとして永久保存する
        """
        failure_node = {
            "id": str(uuid.uuid4())[:8],
            "target_symbol": target_symbol,
            "attempted_intent": intent,
            "error_type": error_type,
            "root_cause": root_cause,
            "anti_pattern": anti_pattern,
            "timestamp": time.time()
        }
        self.failure_graveyard.append(failure_node)
        return failure_node

    def get_relevant_failures(self, target_symbol):
        """
        対象シンボルやタスクに関連する『避けるべき過去の失敗（斥力）』を取得
        """
        return [f for f in self.failure_graveyard if f["target_symbol"] == target_symbol or f["target_symbol"] == "all"]

    def get_full_state(self):
        return {
            "code_store": self.code_store,
            "failure_graveyard": self.failure_graveyard
        }
