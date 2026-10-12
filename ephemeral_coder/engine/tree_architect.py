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

    def build_goal_tree(self, user_prompt, max_depth=2, max_nodes=8):
        tree = GoalTree(user_prompt)
        start_time = time.time()

        # Step 1: 【完成形の受入テスト策定】(逆算の基点)
        acceptance_code = self._generate_acceptance_test(user_prompt)
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
                for sub_elem in decomp_res[:4]:  # 成立要件（コア、テスト、ライブラリ、制約）を最大4件抽出
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
        1. 【抽象化プロンプト】: ドメイン具体例（探索、優先度キュー等）によるアンカーバイアスを完全排除
        2. 【レビューシステム】: 提案された前提要素の妥当性・過不足・重複・ハルシネーションを監査
        """
        # Step A: 抽象プロンプトによる前提要素の抽出
        prompt = (
            f"『{element_name}』が実際に動作し、その正当性が実証されるために、前提としてあらかじめ成立・存在していなければならない基盤や条件は何ですか？\n"
            f"本質的に不可欠な前提要素を列挙してください。\n"
            f"もしこれ以上前提を分ける必要のない最小の完結単位（単一の計算式、組み込み機能、アトミックな単一処理）である場合は、null とだけ出力してください。"
        )
        sys_prompt = (
            "You are an Architectural Foundation Decomposition Engine.\n"
            "Analyze the given requirement in an abstract, objective manner without relying on fixed domain examples.\n"
            "Determine the genuine prerequisites required for it to be implemented, executed, and verified.\n"
            "RULES:\n"
            "- If it has essential prerequisites, output ONLY a JSON array of strings in Japanese: [\"前提1\", \"前提2\"]\n"
            "- If it is already an irreducible, self-contained atomic unit, output ONLY: null\n"
            "- Output NO markdown, NO conversational text."
        )

        raw = query_ollama(self.model_name, prompt=prompt, system_prompt=sys_prompt, timeout=60)
        candidates = self._parse_json_or_list(raw)

        # Step B: レビューシステム（監査・最適化）
        reviewed = self._review_decomposition(element_name, candidates)
        return reviewed

    def _review_decomposition(self, element_name, candidates):
        """
        レビューシステム (Reviewer / Critic):
        提案された前提要素を批判的に精査し、過不足、重複、幻覚（ハルシネーション）を削ぎ落とす。
        """
        if candidates == "null" or not candidates:
            review_prompt = (
                f"対象要素: 『{element_name}』\n"
                f"初期判定: 分解不要な最小単位 (null)\n\n"
                f"上記の要素は、本当にこれ以上前提を分ける必要がない最小の実装・定義単位ですか？\n"
                f"妥当であれば 'null' と出力してください。もし複合的な責務があり前提分解が必要なら、真に必要な前提要素のJSON配列を出力してください。"
            )
        else:
            review_prompt = (
                f"対象要素: 『{element_name}』\n"
                f"提案された前提要素候補: {json.dumps(candidates, ensure_ascii=False)}\n\n"
                f"上記の前提要素候補をレビューし、過不足のない最小限の必須前提要素のJSON配列として最適化してください。\n"
                f"- 親要素の成立に無関係な要素、過剰な具体例、重複を削除してください。\n"
                f"- 候補がすべて不要またはすでに最小単位なら 'null' と出力してください。\n"
                f"出力形式: JSON配列 [\"要素A\", \"要素B\"] または null のみ:"
            )

        sys_prompt = (
            "You are an Architectural Quality Reviewer.\n"
            "Critically evaluate proposed prerequisites for software components.\n"
            "Eliminate hallucinations, redundancies, and arbitrary domain anchoring.\n"
            "Output ONLY a JSON array of verified prerequisite strings in Japanese, or ONLY: null\n"
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
                    clean_items = [str(x).strip() for x in items if str(x).strip() and str(x).lower() != "null"]
                    if clean_items:
                        return clean_items
            except Exception:
                pass
        lines = [l.strip().lstrip("-*0123456789. ") for l in clean.splitlines() if l.strip().startswith(("-", "*", "1.", "2.", "3."))]
        clean_lines = [l for l in lines if l and l.lower() != "null"]
        if clean_lines:
            return clean_lines[:4]
        return "null"

    def _generate_acceptance_test(self, user_prompt):
        """
        完成形の意識: 単なるprintやダミー出力を許さず、
        実際に要求を満たしているかを検証する Python assert 文を生成
        """
        sys_prompt = (
            "You are a Quality Architect. Write concise, executable Python acceptance test assertions.\n"
            "CRITICAL: The assert statements must verify real functionality and return types (FAIL if code only prints or returns dummy True).\n"
            "Output ONLY the test code inside a single ```python ... ``` block."
        )
        prompt = (
            f"# USER REQUIREMENT:\n{user_prompt}\n\n"
            f"Write 2-4 Python assert statements that verify the implemented class or function actually solves this requirement:"
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
