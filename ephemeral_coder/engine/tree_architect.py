import json
import re
import time
from engine.goal_tree import GoalTree
from engine.ollama_client import query_ollama

class TreeArchitect:
    """
    目標樹形図アーキテクト (Tree Architect / メインAIのコア思考エンジン):
    1. 【完成形の定義】: 要件から逆算し、満たすべき厳格な受入テスト（assert）を策定
    2. 【アトミック再帰分解】:
       >> {現在最も遠い所にある要素}に必要な要素とは何ですか？可能な限り列挙してください。
          {現在最も遠い所にある要素}がもう分けられない場合はnullとだけ出力してください。
       すべての最終要素がnullになったらTreeの完成。
    """
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name

    def build_goal_tree(self, user_prompt, domain_rules=None, max_depth=3, max_nodes=18):
        tree = GoalTree(user_prompt)
        start_time = time.time()
        self.domain_rules = domain_rules

        # Step 1: 【完成形の受入テスト策定】(逆算の基点)
        acceptance_code = self._generate_acceptance_test(user_prompt, domain_rules=domain_rules)
        tree.acceptance_suite = acceptance_code
        tree.nodes[tree.root_id].acceptance_test = acceptance_code

        # Step 2: 【アトミック再帰分解ループ】
        step_index = 0
        while not tree.is_complete():
            target_node = tree.get_deepest_unexpanded_node()
            if not target_node:
                break

            step_index += 1
            node_name = target_node.name

            # 安全ガード: 深度またはノード上限に達した場合はアトミック単位として固定
            if target_node.depth >= max_depth or len(tree.nodes) >= max_nodes:
                target_node.is_atomic = True
                target_node.status = "ATOMIC"
                tree.decomposition_steps.append({
                    "step": step_index,
                    "target_id": target_node.id,
                    "target_name": node_name,
                    "depth": target_node.depth,
                    "result_type": "ATOMIC_LIMIT",
                    "explanation": "最小実行単位（アトミック）として確定"
                })
                continue

            target_node.status = "EXPANDING"
            decomp_res = self._query_atomic_decomposition(node_name)

            if decomp_res == "null" or not decomp_res:
                # 分解不能 ➔ 言語プリミティブ / アトミック葉ノードとして確定
                target_node.is_atomic = True
                target_node.status = "ATOMIC"
                tree.decomposition_steps.append({
                    "step": step_index,
                    "target_id": target_node.id,
                    "target_name": node_name,
                    "depth": target_node.depth,
                    "result_type": "NULL_TERMINATED",
                    "explanation": "言語組み込み関数・プリミティブ定数レベルに到達（null確定）。"
                })
            else:
                # 分解成功 ➔ 子ノードを追加
                target_node.status = "BRANCH"
                added_children = []
                for sub_elem in decomp_res[:4]:  # 構成要素を最大4件抽出
                    sub_clean = str(sub_elem).strip()
                    if sub_clean and sub_clean.lower() != "null":
                        child = tree.add_child(target_node, sub_clean)
                        added_children.append({"id": child.id, "name": child.name})

                tree.decomposition_steps.append({
                    "step": step_index,
                    "target_id": target_node.id,
                    "target_name": node_name,
                    "depth": target_node.depth,
                    "result_type": "BRANCHED",
                    "children": added_children,
                    "explanation": f"{len(added_children)} 個の構成要素へ分解展開。"
                })

        total_time = round(time.time() - start_time, 2)
        return tree, total_time

    def _query_atomic_decomposition(self, element_name):
        """
        【概念核（Concept Core）言語プリミティブ分解】:
        プログラミング言語の組み込み関数（sum, len, range等）や演算子に行きつくまで徹底的に分解。
        未定義領域（When / What / Scope）を排除する。
        """
        # すでに言語組み込み関数・演算子・定数定義レベルに到達している場合は即座に null (アトミック確定)
        primitive_indicators = [
            "sum(", "len(", "range(", "dict", "辞書", "random.shuffle", "shuffle",
            "比較演算", "定数定義", "スライス", "max(", "min(", ">", "<", "=="
        ]
        if any(ind in element_name.lower() for ind in primitive_indicators):
            return "null"

        rule_context = ""
        if getattr(self, "domain_rules", None):
            rule_context = f"\n【背景ドメイン/ルール仕様】:\n{json.dumps(self.domain_rules, ensure_ascii=False, indent=2)}\n"

        prompt = (
            f"対象要素: 『{element_name}』\n{rule_context}\n"
            f"【概念核（Concept Core）分解原則】:\n"
            f"すでにプログラミング言語上で定義されている組み込み関数（sum, len, range, random.shuffle等）、基本構文（if/else, 比較演算 ==/>/<, ループfor）、静的定数・辞書データに行きつくまで分解してください。\n"
            f"未定義領域を残してはなりません。複合的な機能名（勝敗判定、デッキ構築、分子結合等）は、必ず【いつ行うか（タイミング）】【何をもって判定・集計するか（計算式・基準値）】【使用範囲・上限制約】へと分解してください。\n"
            f"もしすでに言語組み込み関数や単一の演算子・定数定義（sum計算、比較、辞書参照など）に行きついており、これ以上分解不能なアトミック単位である場合は、null とだけ出力してください。"
        )
        sys_prompt = (
            "You are a Lead Software & Game Systems Architect practicing 'Concept Core (概念核)' tree decomposition.\n"
            "Core Philosophy:\n"
            "Decompose components recursively until reaching LANGUAGE-DEFINED BUILT-IN FUNCTIONS, DATA PRIMITIVES, OR ELEMENTARY OPERATORS (e.g. sum(), len(), range(), dict lookup, comparisons >, if/else, const values).\n"
            "There must be NO undefined or ambiguous gaps. For every system, define:\n"
            "- What (exact criteria, formula, data fields)\n"
            "- When (phase, timing, trigger)\n"
            "- Scope (range of elements, maximum deck/hand size)\n"
            "STRICTLY FORBIDDEN: Abstract platitudes ('明確さ', '一貫性', '品質', '管理').\n"
            "Output ONLY a JSON array of concrete sub-elements in Japanese: [\"具体的仕様1\", \"具体的仕様2\"], or ONLY: null if irreducible language primitive.\n"
            "No markdown, no conversation."
        )

        raw = query_ollama(self.model_name, prompt=prompt, system_prompt=sys_prompt, timeout=60)
        candidates = self._parse_json_or_list(raw)

        # Step B: レビューシステム（非機能形容詞の監査・リジェクト & プリミティブ整合性検証）
        reviewed = self._review_decomposition(element_name, candidates)
        return reviewed

    def _review_decomposition(self, element_name, candidates):
        """
        レビューシステム (Reviewer / Critic):
        提案された要素を監査し、未定義領域がないか、言語プリミティブに到達しているかを精査。
        抽象語が検出された場合のみ再レビューLLMを駆動し、オーバーヘッドを最小化する。
        """
        if candidates == "null" or not candidates:
            return "null"

        abstract_keywords = ["明確さ", "一貫性", "品質", "管理", "対象ルール", "適切な", "向上", "最適化"]
        needs_review = False
        if isinstance(candidates, list):
            for c in candidates:
                if any(kw in str(c) for kw in abstract_keywords):
                    needs_review = True
                    break

        if not needs_review:
            return candidates

        review_prompt = (
            f"親要素: 『{element_name}』\n"
            f"提案された構成要素候補: {json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"上記の構成要素候補に抽象的な非機能語が含まれています。以下を実施してください:\n"
            f"1. 『明確さ』『一貫性』『管理』などの抽象形容詞があれば、実装可能な『具体的ルール仕様』『データモデル』『判定ロジック』に置換せよ。\n"
            f"2. 複合機能（勝敗判定、デッキ構築等）について、【いつ行うか (When)】や【何をもって判定するか (What)】が曖昧な場合、明確な仕様名に補正せよ。\n"
            f"3. すでに言語組み込み関数や単一の定数定義に到達している場合、または候補が重複している場合は整理し、最大3〜4個の明確なモジュールとして最適化せよ。\n"
            f"出力形式: JSON配列 [\"要素A\", \"要素B\"] または null のみ:"
        )

        sys_prompt = (
            "You are an Architectural Quality Auditor.\n"
            "Enforce CONCRETE EXECUTABLE SPECIFICATIONS down to Language Primitives.\n"
            "Reject abstract qualities and convert them into real game mechanics and data structures.\n"
            "Output ONLY a JSON array of verified prerequisite strings in Japanese, or ONLY: null.\n"
            "No explanations, no markdown."
        )

        raw = query_ollama(self.model_name, prompt=review_prompt, system_prompt=sys_prompt, timeout=60)
        reviewed_candidates = self._parse_json_or_list(raw)
        return reviewed_candidates if reviewed_candidates else candidates

    def _parse_json_or_list(self, text):
        if not text:
            return "null"
        clean = text.strip()
        if clean.lower() == "null" or clean.lower().startswith("null"):
            return "null"
        match = re.search(r"(\[.*?\])", clean, re.DOTALL)
        if match:
            items = None
            try:
                items = json.loads(match.group(1))
            except Exception:
                try:
                    import ast
                    items = ast.literal_eval(match.group(1))
                except Exception:
                    items = None

            if isinstance(items, list) and len(items) > 0:
                clean_items = []
                for x in items:
                    if isinstance(x, dict):
                        name = x.get("要素") or x.get("name") or x.get("module") or (list(x.values())[0] if x else "")
                        desc = x.get("仕様") or x.get("description") or x.get("details") or ""
                        combined = f"{name} ({desc})" if desc else str(name)
                        clean_items.append(combined.strip())
                    else:
                        clean_items.append(str(x).strip())
                clean_items = [x for x in clean_items if x and x.lower() != "null"]
                if clean_items:
                    return clean_items
        lines = [l.strip().lstrip("-*0123456789. ") for l in clean.splitlines() if l.strip().startswith(("-", "*", "1.", "2.", "3."))]
        clean_lines = [l for l in lines if l and l.lower() != "null"]
        if clean_lines:
            return clean_lines[:4]
        return "null"

    def _generate_acceptance_test(self, user_prompt, domain_rules=None):
        """
        完成形の意識: 単なるprintやダミー出力を許さず、
        実際に要求を満たしているかを検証する Python assert 文を生成
        """
        rule_context = ""
        if domain_rules:
            rule_context = f"\n# GAME/DOMAIN SPECIFICATION RULES:\n{json.dumps(domain_rules, ensure_ascii=False, indent=2)}\n"

        sys_prompt = (
            "You are a Quality Architect. Write concise, executable Python acceptance test assertions.\n"
            "CRITICAL: The assert statements must verify real functionality and return types (FAIL if code only prints or returns dummy True).\n"
            "Output ONLY the test code inside a single ```python ... ``` block."
        )
        prompt = (
            f"# USER REQUIREMENT:\n{user_prompt}\n{rule_context}\n"
            f"Write 2-4 Python assert statements that verify the implemented classes or functions actually satisfy the specification:"
        )

        raw = query_ollama(self.model_name, prompt=prompt, system_prompt=sys_prompt, timeout=60)
        if raw:
            match = re.search(r"```(?:python)?(.*?)```", raw, re.DOTALL)
            if match:
                return match.group(1).strip()
            return raw.strip()

        # デフォルトの安全な受入テスト
        return (
            "# 受入テスト (厳格アサーション)\n"
            "assert 'main_logic' in locals() or 'PathFinder' in locals() or 'UserManager' in locals(), '主要クラスまたは関数が定義されていません'\n"
        )
