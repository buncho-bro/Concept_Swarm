import json
import re
from engine.ollama_client import query_ollama

class RuleFormulator:
    """
    ドメイン・ゲームルール詳細定義エンジン (Rule & Domain Formulator):
    ユーザーのアイデアや要件から、実装・検証に必要な具体的ルール、
    データモデル（カード・パラメータ）、状態遷移、勝敗条件を定量的に策定する。
    """
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name

    def formulate_rules(self, user_prompt):
        # ゲームまたはルール駆動型要件であるかを判定
        is_game_or_rule = any(kw in user_prompt for kw in ["ゲーム", "カード", "ルール", "対戦", "ターン", "スコア", "勝敗", "デッキ", "バトル"])

        if not is_game_or_rule:
            return None

        sys_prompt = (
            "You are a Lead Game Rules & Mechanics Designer.\n"
            "Analyze the user's game vision and produce a rigorous, quantitative Game Rule Specification in JSON.\n"
            "CRITICAL: Define concrete parameter tables, numerical values, state transition rules, and scoring formulas.\n"
            "Output ONLY valid JSON matching the schema below (no markdown, no conversation):\n"
            "{\n"
            '  "title": "ゲーム正式名称",\n'
            '  "core_mechanics_summary": "コアゲームメカニクス要約",\n'
            '  "entities": [\n'
            '    {"name": "エンティティ名/カード名", "attributes": "具体的な属性値（価数、親和度、コスト等）", "role": "役割"}\n'
            '  ],\n'
            '  "state_transitions": [\n'
            '    {"event": "発生イベント（例: 手札隣接）", "trigger": "条件", "action": "発生する具体的処理（例: 自動結合、親和力再結合）"}\n'
            '  ],\n'
            '  "special_rules": [\n'
            '    {"name": "特殊カード/効果名", "effect": "具体的効果（例: 熱による解離、衝撃によるシャッフル）"}\n'
            '  ],\n'
            '  "evaluation_criteria": "勝敗判定・スコアリング計算ロジック（安定度合計、有機系ボーナス等）"\n'
            "}"
        )

        prompt = (
            f"# USER GAME REQUIREMENT:\n{user_prompt}\n\n"
            f"この要件をプログラムとして100%実装可能な、厳密で具体的なゲームルール仕様JSONとして策定してください:"
        )

        raw = query_ollama(self.model_name, prompt=prompt, system_prompt=sys_prompt, timeout=90)
        if raw:
            match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return data
                except Exception:
                    pass
        return None
