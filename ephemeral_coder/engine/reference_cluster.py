class BaseReferenceAI:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

    def query(self, question_type, params):
        raise NotImplementedError


class SpecReferenceAI(BaseReferenceAI):
    """レファレンスAI 1: 仕様・要件専門"""
    def __init__(self, spec_memory):
        super().__init__("レファレンスAI(仕様)", "design_spec")
        self.memory = spec_memory

    def query(self, question_type, params):
        if question_type != "design_spec":
            return {"known": False, "answer": "【管轄外】仕様・要件以外の質問には答えられません（わかりません）"}
        
        spec_id = params.get("spec_id")
        spec = self.memory.get_spec(spec_id)
        if spec:
            return {"known": True, "answer": spec}
        return {"known": False, "answer": f"指定された仕様ID({spec_id})は見つかりません（わかりません）"}


class CodeReferenceAI(BaseReferenceAI):
    """レファレンスAI 2: コード・構文専門"""
    def __init__(self, code_memory):
        super().__init__("レファレンスAI(コード)", "code_symbol")
        self.memory = code_memory

    def query(self, question_type, params):
        if question_type != "code_symbol":
            return {"known": False, "answer": "【管轄外】ソースコード以外の質問には答えられません（わかりません）"}
            
        symbol = params.get("symbol")
        code = self.memory.get_code(symbol)
        if code:
            ver = self.memory.code_store[symbol]["version"]
            return {"known": True, "answer": {"symbol": symbol, "version": ver, "code": code}}
        return {"known": False, "answer": f"シンボル `{symbol}` は記憶に存在しません（わかりません）"}


class FailureReferenceAI(BaseReferenceAI):
    """レファレンスAI 3: 失敗・アンチパターン専門"""
    def __init__(self, failure_memory):
        super().__init__("レファレンスAI(失敗教訓)", "failure_history")
        self.memory = failure_memory

    def query(self, question_type, params):
        if question_type != "failure_history":
            return {"known": False, "answer": "【管轄外】失敗履歴以外の質問には答えられません（わかりません）"}
            
        target_symbol = params.get("target_symbol")
        failures = self.memory.get_failures(target_symbol)
        if failures:
            return {"known": True, "answer": failures}
        return {"known": True, "answer": []} # 失敗がないのは正常な回答


class ReferenceCluster:
    """
    レファレンスAI群の統合バス (赤丸の『必要があれば情報要求』インターフェース)
    """
    def __init__(self, memory_cluster):
        self.ref_spec = SpecReferenceAI(memory_cluster.spec_mem)
        self.ref_code = CodeReferenceAI(memory_cluster.code_mem)
        self.ref_fail = FailureReferenceAI(memory_cluster.failure_mem)
        self.agents = [self.ref_spec, self.ref_code, self.ref_fail]

    def on_demand_request(self, requester_name, question_type, params):
        """
        作業AIやメインAIからのオンデマンド情報要求（図の赤丸通信）
        各レファレンスAIに問い合わせ、知っている者だけが回答を返す。
        """
        hop_log = {
            "requester": requester_name,
            "question_type": question_type,
            "params": params,
            "responses": []
        }
        
        authoritative_answer = None

        for ref in self.agents:
            res = ref.query(question_type, params)
            hop_log["responses"].append({
                "agent": ref.name,
                "known": res["known"],
                "answer": res["answer"] if not res["known"] else "【管轄内】回答を提供しました"
            })
            if res["known"]:
                authoritative_answer = res["answer"]

        return authoritative_answer, hop_log
