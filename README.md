# ・Antigravity Concept Swarm & Thought Wilderness

> **「巨大な言語モデルから、純粋な『概念のトポロジー』を抽出し、言葉を介さずにミリ秒で合議させ、エッジから自己組織化する思考原野までを解き明かす」**

VRAMゼロで動作する10体並列の「非言語合議型AIエージェントスウォーム」から始まり、極小エッジマイコン（520KB SRAM）での自律推論、巨大80BモデルのHTTPストリーミング解体、人間定義ラベルを全廃した純粋軌跡移転、そして単語の全層軌跡から自己組織化する超並列推論基盤**「思考原野（Thought Wilderness）」**に至るまで、ユーザーとAIが二人三脚で切り拓いた一連の幾何学的AI探求の成果を集約した統合リポジトリです。

---

## ・プロジェクトの理念と免責事項
- **Google Antigravityによる開発:** 本プロジェクトのコアアルゴリズム、スクリプト、およびUIアーキテクチャは、Google Antigravity AIエージェントフレームワークを使用し、主に Gemini との対話・協働を通じて開発されました。
- **公平なAI競争の促進:** 本プロジェクトの最大の目標は、AI開発における公平な競争と民主化の促進です。開発者がローカル環境で効率的に「概念地形」「思考原野」を抽出・自己組織化できる手法を提供することで、AI研究参入への障壁を下げることを目指しています。既存のAIシステムを破壊、搾取、または妨害することを意図したものでは決してありません。あくまで最適化と平等性を追求する学術的・工学的な試みです。
- **オリジナルAIライセンスの遵守:** 本リポジトリのスクリプトを使用して知識の抽出やモデルの蒸留を行う際は、対象となるAIモデルの元の開発者（Hugging Face、OpenAI、Z.ai、Alibaba、Mistral AIなど）が定める利用規約、ライセンス、および利用ルールを厳守してください。

---

## ・時系列リポジトリ構成（目次）

本リポジトリは、対話の進展と技術的ブレイクスルーの時系列順に構成されています。各Pythonファイルの先頭には、詳細な仕組みを記したDocstringが埋め込まれています。

### 1. `core/` (第1章：VRAM-Zero マルチエージェント合議エンジン)
*大次元の潜在ベクトルを共有メモリでやり取りし、テキスト変換なしで10体並列合議を行う中核システム。*
- `shared_memory_loader.py` / `latent_blackboard.py` / `agent_worker.py` / `multi_agent_swarm.py` / `dual_stream_decoder.py`

### 2. `project_ripple/` (第2章：波及効果型 物理推論エンジン)
*概念空間における推論を「波紋（Ripple）」や流体力学的な物理式に見立て、Web UIとWebSocketを通じてリアルタイムに操作・可視化するバックエンド。*
- `server.py` / `physics_core.py` / `standalone_concept_distiller.py` / etc.

### 3. `quantized_core/` & `edge_micro_agent/` (第3章：極限の圧縮とエッジ到達)
*量子化重みを用いた高速化やKNNグラフ蒸留、そしてSRAMわずか520KBの RP2350 (Cortex-M33) マイコン上で局所100語展開（Local Topography）を用いて自律推論する組み込み向けAI。*
- `main.py` / `export_micro_topo.py` / `st7789.py` / `step1_quantized_distillation.py` 〜 `step8_hierarchical_decoding.py`

### 4. `alice_profiler/` & `alice_relay_engine/` (第4章：80B巨大モデルのオンデマンド解体)
*160GBの巨大MoEモデル（AliceAI-80B）から、HTTP Rangeリクエストを用いて必要な数MBだけを直接ストリーミング取得し、数学AIからコードAIへ非言語バトンパスを行う疎結合推論エンジン。*
- `expert_topography_profiler.py` / `alice_relay_engine.py`

### 5. `game_prototype/` & `alife_agent_sim/` (第5章：生命と社会のシミュレーション)
*潜在物理型AI執政官のゲーム応用、および16次元概念ベクトルをDNAとして持つ人工生命（Semantic ALife V1〜V4）の連続空間自律進化シミュレータ。*
- `server.py` / `alife_sim.py` / etc.

### 6. `ephemeral_coder/` (第6章：逆算型目標樹形図エンジン / GDTE v3.4)
*目標ツリーを「言語の組み込みプリミティブ（組み込み関数・演算子）」に行きつくまで再帰分解し、末端Workerが構文的補間のみを行う自律開発エンジン。*
- `server.py` / `engine/tree_architect.py` / `engine/rule_formulator.py` / `docs/goal_tree_engine_specification.md`

### 7. `distillation_experiments/` (第6〜7章：概念フォージ・純粋軌跡移転・思考原野)
*0VRAM全語彙大円干渉、スライストポロジー蒸留、および単語の全層軌跡の類似度から自己組織化する「思考原野（Thought Wilderness）」実験群。*
- **Thought Wilderness (思考原野):**
  - `thought_wilderness_builder.py` : 線の類似度から面のマップを展開し、円形引力とパウリ斥力で自己組織化・安定化する思考原野エンジン。
  - `massive_full_trajectory_benchmark.py` : 各単語が止まるまでの全33層軌跡の直交力線を積分する100組大規模因果ベンチマーク。
  - `full_vocab_collapse_stress_test.py` : 全語彙（7.4万語）規模での重力崩壊（底なし沼化）検証と体積排除安定化シミュレータ。
  - `math_theory_wilderness_v2.py` : Qwen2.5-7Bを用いた数学的理論構築（群+可換➔環、行列+固有値➔分解、多様体+曲率➔テンソル）検証。
  - `model_comparison_wilderness.py` : Qwen-Coder（双極構造） vs Mistral-7B（品詞群島構造） vs 汎用Qwen（渦巻銀河）の比較可視化。

### 8. `docs/` (全体年代記・アーキテクチャ仕様書)
- [`PROJECT_CHRONICLE.md`](docs/PROJECT_CHRONICLE.md) : **全体開発史・総括年代記（第1章〜第7章完結版）**。非言語スウォームから思考原野に至る全軌跡、直面した壁、ブレイクスルーを完全網羅。
- [`THOUGHT_WILDERNESS_CHRONICLE.md`](docs/THOUGHT_WILDERNESS_CHRONICLE.md) : **思考原野の創出史・集中年代記**。単一トークンの固有揺らぎ、全軌道掛け合わせ、重力崩壊とパウリ排他律、モデル別トポロジー解析、数学理論構築と超並列性を時系列順に完全記録。
- `goal_tree_engine_specification.md` : GDTE v3.4 概念核言語プリミティブ分解仕様書。
- `universal_trajectory_transfer_specification.md` : v5.1 粗視化被覆・再細分化重力地形 ＆ 純粋軌跡移転仕様書。

---

## ・使い方
本リポジトリのコード群はモジュール・実験ごとに独立して実行可能です。
**詳しい実行目的とメカニズムは、各 `.py` ファイルの先頭にある解説ブロック（Docstring）および [`docs/PROJECT_CHRONICLE.md`](docs/PROJECT_CHRONICLE.md) をご覧ください。**
