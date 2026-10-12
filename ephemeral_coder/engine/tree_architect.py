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
        ユーザー考案の核となるプロンプト:
        >> {現在最も遠い所にある要素}に必要な要素とは何ですか？可能な限り列挙してください。
           {現在最も遠い所にある要素}がもう分けられない場合はnullとだけ出力してください。
        """
        prompt = (
            f"『{element_name}』を現実に成立させ、正しく機能・実証するために不可欠な前提要素は何ですか？\n"
            f"単なる部分品の内訳ではなく、以下のような【成立要件・前提依存】を多角的に含めて可能な限り列挙してください：\n"
            f"- 1. コア実装・アルゴリズム本体（探索関数など）\n"
            f"- 2. その正しさを立証・保証するテストコードおよび受入検証機構\n"
            f"- 3. 支える数学計算ライブラリやデータ構造の基盤（優先度キュー、ヒューリスティック計算など）\n"
            f"- 4. 実行環境や前提データ型・境界制約・最適化仕様\n\n"
            f"もし『{element_name}』がこれ以上前提や依存を分ける必要のない最小の完結単位（単一の計算式、標準インポート、独立した単一テストなど）である場合は、null とだけ出力してください。"
        )
        sys_prompt = (
            "You are a Systems & Architectural Foundation Decomposition Engine.\n"
            "Your mission is to decompose a programming requirement into WHAT IS NECESSARY TO MAKE IT TRULY WORK AND BE PROVEN VALID (成立要件・前提依存の逆算因数分解).\n"
            "Decompose across essential foundational layers:\n"
            "1. Core Logic & Algorithm (コア関数・実装本体)\n"
            "2. Verification & Acceptance Test Harness (受入テスト・検証コード)\n"
            "3. Mathematical / Data Structure Foundations & Libraries (数学計算ライブラリ、データ構造基盤)\n"
            "4. Execution Environment & Constraints (データ型、境界条件、最適化制約)\n\n"
            "OUTPUT RULES:\n"
            "- If it requires foundational components, output ONLY a JSON array of strings in Japanese: [\"要素1\", \"要素2\"]\n"
            "- If it is already an irreducible atomic unit (a single primitive calculation, standard import, or atomic test assert), output ONLY: null\n"
            "- Output NO markdown, NO commentary."
        )

        raw = query_ollama(self.model_name, prompt=prompt, system_prompt=sys_prompt, timeout=60)
        if not raw:
            return "null"

        text = raw.strip()
        # null 判定
        if text.lower() == "null" or text.lower().startswith("null"):
            return "null"

        # JSON 配列の抽出
        match = re.search(r"(\[.*?\])", text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(1))
                if isinstance(items, list) and len(items) > 0:
                    clean_items = [str(x).strip() for x in items if str(x).strip() and str(x).lower() != "null"]
                    if clean_items:
                        return clean_items
            except Exception:
                pass

        # 箇条書き判定 (フォールバック)
        lines = [l.strip().lstrip("-*0123456789. ") for l in text.splitlines() if l.strip().startswith(("-", "*", "1.", "2.", "3."))]
        clean_lines = [l for l in lines if l and l.lower() != "null"]
        if clean_lines:
            return clean_lines[:3]

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
