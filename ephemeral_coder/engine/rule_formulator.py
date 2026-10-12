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
            "Eliminate ALL ambiguity (e.g. exactly which elements are used, deck max size, victory timing AND formula, UI representation).\n"
            "Output ONLY valid JSON matching the schema below (no markdown, no conversation):\n"
            "{\n"
            '  "title": "ゲーム正式名称",\n'
            '  "core_mechanics_summary": "コアゲームメカニクス要約",\n'
            '  "elements_spec": {\n'
            '    "scope": "定義元素の範囲（例: H, C, N, O等の第1〜2周期代表元素）",\n'
            '    "master_table": [\n'
            '      {"symbol": "H", "valency": 1, "affinity": 2.2, "cost": 1, "description": "水素"},\n'
            '      {"symbol": "C", "valency": 4, "affinity": 2.5, "cost": 2, "description": "炭素 (有機系の核)"},\n'
            '      {"symbol": "N", "valency": 3, "affinity": 3.0, "cost": 2, "description": "窒素"},\n'
            '      {"symbol": "O", "valency": 2, "affinity": 3.4, "cost": 2, "description": "酸素"}\n'
            '    ]\n'
            '  },\n'
            '  "binding_spec": {\n'
            '    "auto_binding": "手札内隣接による自動分子結合走査ロジック",\n'
            '    "re_binding": "隣接元素の親和力優先度による結合組み換え判定"\n'
            '  },\n'
            '  "deck_spec": {\n'
            '    "max_deck_size": 30,\n'
            '    "hand_size": 5,\n'
            '    "initial_draw": 5\n'
            '  },\n'
            '  "special_cards": [\n'
            '    {"name": "熱 (Heat)", "effect": "配置した分子を構成元素カードに解離", "timing": "プレイ時"},\n'
            '    {"name": "衝撃 (Shock)", "effect": "手札順をシャッフルし強制再結合を誘発", "timing": "プレイ時"}\n'
            '  ],\n'
            '  "victory_spec": {\n'
            '    "timing": "各ターン終了時および規定ターン到達時",\n'
            '    "formula": "手札内完成分子の安定度合計 score = sum(m.stability for m in molecules) + 有機系ボーナス",\n'
            '    "condition": "相手スコアを上回る、または目標安定度到達で勝利"\n'
            '  },\n'
            '  "ui_spec": {\n'
            '    "hand_display": "手札元素および結合分子のテキスト/ビジュアル描画",\n'
            '    "battle_log": "結合・再結合・特殊カード発動のログ出力"\n'
            '  },\n'
            '  "entities": [\n'
            '    {"name": "元素カード", "attributes": "価数, 親和度, コスト", "role": "基本カード"}\n'
            '  ],\n'
            '  "evaluation_criteria": "安定度スコアの合計計算式とターン終了時の判定"\n'
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
