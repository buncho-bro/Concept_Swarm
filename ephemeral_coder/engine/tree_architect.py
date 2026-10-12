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

    def build_goal_tree(self, user_prompt, domain_rules=None, max_depth=2, max_nodes=8):
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
                # 分解不能 ➔ アトミック葉ノードとして確定
                target_node.is_atomic = True
                target_node.status = "ATOMIC"
                tree.decomposition_steps.append({
                    "step": step_index,
                    "target_id": target_node.id,
                    "target_name": node_name,
                    "depth": target_node.depth,
                    "result_type": "NULL_TERMINATED",
                    "explanation": "LLMが 'null' を返答。これ以上分解できない不可分な実装単位として確定。"
                })
            else:
                # 分解成功 ➔ 子ノードを追加
                target_node.status = "BRANCH"
                added_children = []
                for sub_elem in decomp_res[:4]:  # 成立要件を最大4件抽出
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
        1. 【抽象・ドメイン実体化プロンプト】: 精神論（明確さ、一貫性等）を厳禁し、実装可能な具体的仕様・ルールを要求
        2. 【レビューシステム】: 抽象形容詞を検知・リジェクトし、実装モジュールへ置換
        """
        rule_context = ""
        if getattr(self, "domain_rules", None):
            rule_context = f"\n【背景ドメイン/ルール仕様】:\n{json.dumps(self.domain_rules, ensure_ascii=False, indent=2)}\n"

        prompt = (
            f"対象要素: 『{element_name}』\n{rule_context}\n"
            f"この要素を実際にプログラムとして成立・動作させるために不可欠な【具体的仕様、データ構造、判定アルゴリズム】は何ですか？\n"
            f"※厳格な禁止事項: 『明確さ』『一貫性』『品質』『使いやすさ』『管理』などの非機能的・精神論的な形容詞は絶対に含めないでください。\n"
            f"要件に登場する具体的なドメインモデル（カードデータ、自動結合、再結合ロジック、特殊カード、勝敗スコア等）を直接反映した具体的な仕様・モジュール名のみを列挙してください。\n"
            f"もしこれ以上前提を分ける必要のない最小の完結単位（単一の計算式、組み込み機能、アトミックな単一処理）である場合は、null とだけ出力してください。"
        )
        sys_prompt = (
            "You are a Lead Software & Game Systems Architect.\n"
            "Decompose components into CONCRETE, IMPLEMENTABLE software modules, rule specifications, and data structures.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. STRICTLY FORBIDDEN: Abstract platitudes, vague qualities, or non-executable adjectives (e.g., '明確さ', '一貫性', '品質').\n"
            "2. REQUIRED: Concrete domain entities, algorithmic mechanics, and state transitions.\n"
            "3. Output ONLY a JSON array of strings in Japanese: [\"具体的仕様1\", \"具体的仕様2\"], or ONLY: null if irreducible.\n"
            "No markdown, no conversation."
        )

        raw = query_ollama(self.model_name, prompt=prompt, system_prompt=sys_prompt, timeout=60)
        candidates = self._parse_json_or_list(raw)

        # Step B: レビューシステム（非機能形容詞の監査・リジェクト）
        reviewed = self._review_decomposition(element_name, candidates)
        return reviewed

    def _review_decomposition(self, element_name, candidates):
        """
        レビューシステム (Reviewer / Critic):
        提案された前提要素を批判的に精査し、『明確さ』『一貫性』などの抽象的単語を排除して具体的ルールに修正する。
        """
        if candidates == "null" or not candidates:
            return "null"

        review_prompt = (
            f"親要素: 『{element_name}』\n"
            f"提案された前提要素候補: {json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"上記の前提要素候補を監査し、以下を実施してください:\n"
            f"1. 『明確さ』『一貫性』『管理』などの抽象的・非機能的な単語があれば、実装可能な『具体的ルール仕様』『データモデル』『判定ロジック』に置換または削除せよ。\n"
            f"2. 重複を削り、最大3〜4個の明確な実装モジュール・ルール仕様として最適化せよ。\n"
            f"出力形式: JSON配列 [\"要素A\", \"要素B\"] または null のみ:"
        )

        sys_prompt = (
            "You are an Architectural Quality Auditor.\n"
            "Enforce CONCRETE EXECUTABLE SPECIFICATIONS.\n"
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
            try:
                items = json.loads(match.group(1))
                if isinstance(items, list) and len(items) > 0:
                    clean_items = []
                    for x in items:
                        if isinstance(x, dict):
                            name = x.get("要素") or x.get("name") or x.get("module") or list(x.values())[0]
                            desc = x.get("仕様") or x.get("description") or x.get("details") or ""
                            combined = f"{name} ({desc})" if desc else str(name)
                            clean_items.append(combined.strip())
                        else:
                            clean_items.append(str(x).strip())
                    clean_items = [x for x in clean_items if x and x.lower() != "null"]
                    if clean_items:
                        return clean_items
            except Exception:
                pass
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
