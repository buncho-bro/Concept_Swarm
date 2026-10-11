import time
import uuid

class SpecMemoryNode:
    """記憶AI 1: 設計書・仕様書専門"""
    def __init__(self):
        self.specs = {}

    def save_spec(self, spec_id, title, details, tasks):
        self.specs[spec_id] = {
            "id": spec_id,
            "title": title,
            "details": details,
            "tasks": tasks,
            "timestamp": time.time()
        }

    def get_spec(self, spec_id):
        return self.specs.get(spec_id)


class CodeMemoryNode:
    """記憶AI 2: ソースコード・ASTシンボル専門"""
    def __init__(self):
        self.code_store = {
            "main_logic": {
                "symbol": "main_logic",
                "code": "def process_data(data_list):\n    # 基準実装: 正数フィルタリング\n    return [x for x in data_list if isinstance(x, (int, float)) and x > 0]\n",
                "version": 1,
                "history": []
            }
        }

    def save_code(self, symbol, code, commit_msg):
        if symbol not in self.code_store:
            self.code_store[symbol] = {"symbol": symbol, "code": code, "version": 1, "history": []}
        else:
            self.code_store[symbol]["version"] += 1
            self.code_store[symbol]["code"] = code
        self.code_store[symbol]["history"].append({
            "version": self.code_store[symbol]["version"],
            "code": code,
            "msg": commit_msg,
            "timestamp": time.time()
        })
        return self.code_store[symbol]["version"]

    def get_code(self, symbol):
        return self.code_store.get(symbol, {}).get("code")


class FailureMemoryNode:
    """記憶AI 3: 失敗の死後解剖（ポストモーテム）・アンチパターン専門"""
    def __init__(self):
        self.failures = []

    def record_failure(self, target_symbol, intent, error_type, root_cause, anti_pattern):
        node = {
            "id": str(uuid.uuid4())[:8],
            "target_symbol": target_symbol,
            "intent": intent,
            "error_type": error_type,
            "root_cause": root_cause,
            "anti_pattern": anti_pattern,
            "timestamp": time.time()
        }
        self.failures.append(node)
        return node

    def get_failures(self, target_symbol):
        return [f for f in self.failures if f["target_symbol"] == target_symbol or f["target_symbol"] == "all"]


class MemoryCluster:
    """記憶AI群の統合コンテナ"""
    def __init__(self):
        self.spec_mem = SpecMemoryNode()
        self.code_mem = CodeMemoryNode()
        self.failure_mem = FailureMemoryNode()

    def get_state(self):
        return {
            "specs": self.spec_mem.specs,
            "code_store": self.code_mem.code_store,
            "failures": self.failure_mem.failures
        }
