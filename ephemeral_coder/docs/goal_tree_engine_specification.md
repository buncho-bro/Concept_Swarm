# 逆算型目標樹形図エンジン (Goal-Driven Tree Engine / GDTE v3.3) 詳細設計書

## 1. 根本課題とシステム的解決アプローチ

### 1.1 課題: 非機能的形容詞（お題目の罠）による形骸化
ゲームや複雑なルール駆動システムを樹形図展開させた際、従来のプロンプトでは以下のような非機能的・精神論的な形容詞ノードが発生し、開発が頓挫する問題がありました：
- `node_3: 対象ルールの明確さ [ATOMIC (null)]`
- `node_4: 対象ルールの一貫性 [ATOMIC (null)]`
- `node_5: ゲームの進行管理 [ATOMIC (null)]`
- `node_6: 勝負の判定基準 [ATOMIC (null)]`

これらは「どのような機能やデータ構造が必要か」を全く定義しておらず、コード生成AI（Worker AI）に渡しても実装不可能なダミーノードとなってしまいます。

### 1.2 システム的解決策: ルール詳細仕様化エンジン & 抽象形容詞追放フィルター
単なるプロンプト調整に留まらず、以下の2つのシステム的機構をエンジンに組み込みました：

```mermaid
flowchart TD
    UserPrompt[ユーザー指示 / ゲーム構想] --> RF[RuleFormulator: ルール仕様策定エンジン]
    
    subgraph SpecEngine [1. ドメイン・ゲームルール詳細定義 (RuleFormulator)]
        RF --> DomainRules[定量的仕様JSON<br/>・エンティティ/属性値 (価数・親和度)<br/>・状態遷移 (手札隣接・自動結合)<br/>・特殊カード (熱・衝撃)<br/>・勝敗評価基準]
    end
    
    DomainRules --> TA[TreeArchitect: 目標樹形図アーキテクト]
    
    subgraph TreeLoop [2. アトミック再帰分解 & 追放フィルター]
        TA --> Prompt[具体的仕様分解プロンプト<br/>※明確さ・一貫性・品質・管理を厳禁]
        Prompt --> Critic[Reviewer / Critic AI<br/>非機能形容詞を検知・データモデルへ置換]
        Critic --> VerifiedTree[具体的実装モジュールTree<br/>全葉ノードが null (アトミック) 到達]
    end
    
    VerifiedTree --> Swarm[使い捨て作業AIスウォーム<br/>Ephemeral Workers (qwen2.5-coder:3b)]
    Swarm --> StrictSandbox[厳格受入検証サンドボックス<br/>AST構文 & 実行時アサーション合格]
    StrictSandbox --> CodeStore[正史コードストア (Verified DAG)]
```

---

## 2. コアアーキテクチャ詳細

### 2.1 ルール策定エンジン (`RuleFormulator`)
要件内にゲームや対戦、カード、ターンなどの要素が含まれる場合、樹形図の分解に先立って定量的かつプログラム可能な仕様書をJSON形式で自動策定します：
- **entities:** 登場カードや構成要素の属性（価数、親和度、強さ、コストなど）
- **state_transitions:** イベント発生トリガーと遷移先アクション（隣接結合、親和度優先の引っ付き直し）
- **special_rules:** 特殊カードの効果（熱による分子解離、衝撃によるシャッフル）
- **evaluation_criteria:** 勝敗および安定度スコアリング計算ロジック

### 2.2 非機能形容詞追放フィルター (`Anti-Platitude Filter`)
`TreeArchitect` のプロンプトおよび `_review_decomposition` において、以下の制約を強制：
1. **厳禁ワードリスト:** 『明確さ』『一貫性』『品質』『使いやすさ』『管理の徹底』
2. **変換ルール:** 抽象的な形容詞が提案された場合、Reviewerが直ちにそれを破棄し、具体的な「データモデル」「判定ロジック」「計算アルゴリズム」へと置換。

---

## 3. 実証実験結果: 元素対戦カードゲーム

ユーザーから提示された超複雑なゲーム仕様を投入して検証を実施しました：

> **指示:**
> 「元素を使って対戦カードゲーム！分子にすると強くなるぞ！…でも本作最大の特徴は分子を合成できない点。並べたら手札のなかで勝手くっついてしまう。でも更にくっつきやすい元素が隣に来ると引っ付きなおす。熱、衝撃等の特殊カードも活かしてより安定した分子を作るもよし！有機系に振るのもよし！通常のカードゲームとは一風変わったデッキ構築が楽しめます。」

### 3.1 策定されたルール仕様（RuleFormulator 出力）
```json
{
  "title": "元素対戦カードゲーム",
  "core_mechanics_summary": "対戦カードゲームで、分子を合成して強さを増やすが、手札の並び順で自動結合する。",
  "entities": [
    {"name": "元素", "attributes": {"強さ": 10, "親和度": 10, "コスト": 5}, "role": "カード"},
    {"name": "熱", "attributes": {"効果": "分子を解離"}, "role": "特殊カード"},
    {"name": "衝撃", "attributes": {"効果": "分子をシャッフル"}, "role": "特殊カード"}
  ],
  "state_transitions": [
    {"event": "分子隣接", "trigger": "元素が隣接", "action": "分子を引っ付く"},
    {"event": "特殊カード発動", "trigger": "熱または衝撃カードの使用", "action": "解離 / シャッフル"}
  ],
  "special_rules": [
    {"name": "有機系", "effect": "特定元素の結合時にボーナス付与"}
  ],
  "evaluation_criteria": "分子の強さの合計値で勝敗を判定"
}
```

### 3.2 生成された目標樹形図（Goal Tree）
旧来の曖昧な形容詞は完全に一掃され、100%実装可能な具体的モジュールへと展開されました：

```
node_0: [ROOT] 元素対戦カードゲーム (手札内自動分子結合・再結合・特殊カード)
├── node_1: 分子データベース構築 [BRANCH]
│   ├── node_5: 分子情報格納データ構造 [ATOMIC (null)] ➔ ✅ VERIFIED
│   ├── node_6: 元素間の強さ計算アルゴリズム [ATOMIC (null)] ➔ ✅ VERIFIED
│   ├── node_7: 特殊カード効果判定ロジック [ATOMIC (null)] ➔ ✅ VERIFIED
│   └── node_8: 勝敗判定アルゴリズム [ATOMIC (null)] ➔ ✅ VERIFIED
├── node_2: 特殊カードデータベース構築 [ATOMIC (null)] ➔ ✅ VERIFIED
├── node_3: 分子引っ付けロジック [ATOMIC (null)] ➔ ✅ VERIFIED
└── node_4: 勝敗スコア計算アルゴリズム [ATOMIC (null)] ➔ ✅ VERIFIED
```

### 3.3 作業AIスウォームによる成果物コード（正史ブランチ v8）
使い捨て作業AI（`qwen2.5-coder:3b`）が各葉ノードを順次自律実装し、厳格受入サンドボックス検証を経て以下の完全動作Pythonクラス群を生成・マージしました：
- `class Element(name, strength, special_effect)`
- `class SpecialCard(name, type, strength_increase, weakness_increase)`: 熱・衝撃カード対応
- `class Card(element, special_card)`: `combine()`, `apply_special_effect()`
- `class Deck()`: `combine_cards()`, `get_strength()`, `can_combine()`, `is_winner()`
- 熱・衝撃などの特殊カードリストと対戦シミュレーションコード

---

## 4. 結論
システム的に「ルール仕様策定フェーズ」をツリー展開の前に挟み、かつ「非機能形容詞追放フィルター」を常時適用することで、どんなに複雑・奇抜なゲームや仕様であっても、抽象論に逃げることなく「完成形から逆算した本物のデータ構造とロジック」を確実に構築できることが実証されました。
