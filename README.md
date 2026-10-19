# ・Antigravity Concept Swarm & Experimental Cores

VRAMゼロで動作する10体並列の「非言語合議型 AIエージェントスウォーム」を中心に、本プロジェクト内で派生・並行して実験されてきたあらゆるコア（量子化推論、マルチモーダル、進化計算、Project Ripple等）を集約した統合リポジトリです。途中でフリーズしたアイデアや未完の実験コードも、将来の発展のためにすべてアーカイブ・ドキュメント化して保存されています。

---

## ・プロジェクトの理念と免責事項
Google Antigravityによる開発 本プロジェクトのコアアルゴリズム、スクリプト、およびUIアーキテクチャは、Google Antigravity AIエージェントフレームワークを使用し、主に Gemini Pro との対話・協働を通じて開発されました。

公平なAI競争の促進 本プロジェクトの最大の目標は、AI開発における公平な競争と民主化の促進です。開発者がローカル環境で効率的に「概念地形」を抽出・蒸留できる手法を提供することで、AI研究参入への障壁を下げることを目指しています。このプロジェクトは、既存のAIシステムを破壊、搾取、または妨害することを意図したものでは決してありません。あくまで最適化と平等性を追求する学術的・工学的な試みです。

オリジナルAIライセンスの遵守 本リポジトリのスクリプトを使用して知識の抽出やモデルの蒸留を行う際は、対象となるAIモデルの元の開発者（Hugging Face、OpenAI、Z.ai、Alibabaなど）が定める利用規約、ライセンス、および利用ルールを厳守してください。特定のモデルのライセンス違反や著作権侵害を目的としたツールの使用は一切容認しません。

---

## ・リポジトリ構成（目次）

各Pythonファイルの先頭には、そのファイルの詳しい仕組みや目的を記載したDocstring（ブロックコメント）が埋め込まれています。各コードの詳細なメカニズムはファイル内を直接参照してください。

### 1. `core/` (VRAM-Zero マルチエージェント合議エンジン)
*大次元の潜在ベクトルを共有メモリでやり取りし、テキスト変換なしで10体並列合議を行う中核システム。*
- `shared_memory_loader.py` / `latent_blackboard.py` / `agent_worker.py` / `multi_agent_swarm.py` / `dual_stream_decoder.py`

### 2. `benchmark/` (論理コーディング性能評価)
*MBPP+データセットを用いて、概念蒸留によるコーディング能力の変容を正確に測定するサンドボックス環境。*
- `run_mbppplus_benchmark.py` / `eval_worker.py`

### 3. `edge_micro_agent/` (SRAM 520KB 限界実装)
*RP2350 (Cortex-M33) などの極小エッジマイコン上で、局所100語展開（Local Topography）を用いて自律推論する組み込み向けAI。*
- `main.py` / `export_micro_topo.py` / `st7789.py`

### 4. `alice_profiler/` (Zero-Full-Load 専門性抽出)
*160GBの巨大MoEモデル（AliceAI-80B）から、HTTP Rangeリクエストを用いて必要な数MBのテンソルだけを直接ストリーミング取得し、ディスクとRAM消費を極小に抑えつつ数学・コードExpertを抽出するプロファイラ。*
- `expert_topography_profiler.py`

### 5. `multimodal/` (視覚と言語のセマンティック・ブリッジ)
*言語の概念空間と、CLIP等から得られる視覚（画像）の概念空間を大円干渉で結びつけるクロスモーダル実験群。*
- `multimodal_topology.py` / `semantic_bridge.py`

### 6. `project_ripple/` (波及効果型 物理推論エンジン)
*概念空間における推論を「波紋（Ripple）」や流体力学的な物理式に見立て、Web UIとWebSocketを通じてリアルタイムに操作・可視化するバックエンド。*
- `server.py` / `physics_core.py` / `standalone_concept_distiller.py` / etc.

### 7. `quantized_core/` (量子化・KNN・フラクタルルーティング)
*量子化重みを用いた高速化実験や、KNNグラフによるトポロジー蒸留、さらにフラクタル構造を用いた再帰的なExpertルーティングなどの基礎実験群 (step1〜step8)。*
- `step1_quantized_distillation.py` 〜 `step8_hierarchical_decoding.py`

### 8. `distillation_experiments/` (概念フォージ・スライストポロジー蒸留・思考原野)
*0VRAM全語彙大円干渉、スライストポロジー蒸留、および単語の全層軌跡の類似度から自己組織化する「思考原野（Thought Wilderness）」実験群。*
- **Thought Wilderness (思考原野):**
  - `thought_wilderness_builder.py` : 線の類似度から面のマップを展開し、円形引力とパウリ斥力で自己組織化・安定化する思考原野エンジン。
  - `massive_full_trajectory_benchmark.py` : 各単語が止まるまでの全33層軌跡の直交力線を積分する100組大規模因果ベンチマーク。
  - `full_vocab_collapse_stress_test.py` : 全語彙（7.4万語）規模での重力崩壊（底なし沼化）検証と体積排除安定化シミュレータ。
  - `math_theory_wilderness_v2.py` : Qwen2.5-7Bを用いた数学的理論構築（群+可換➔環、行列+固有値➔分解、多様体+曲率➔テンソル）検証。
  - `model_comparison_wilderness.py` : Qwen-Coder（双極構造） vs Mistral-7B（品詞群島構造） vs 汎用Qwen（渦巻銀河）の比較可視化。

### 9. `ephemeral_coder/` (逆算型目標樹形図エンジン / GDTE v3.4)
*目標ツリーを「すでに言語上で定義されている組み込み関数・演算子に行きつくまで再帰分解する不変の概念核」と定義し、使い捨てWorker AIスウォームが構文的補間のみを行って厳格受入テストを突破する自律開発エンジン。*
- `server.py` / `engine/tree_architect.py` / `engine/rule_formulator.py` / `docs/goal_tree_engine_specification.md`

### 10. `docs/` (全体年代記・アーキテクチャ仕様書)
- `THOUGHT_WILDERNESS_CHRONICLE.md` (**思考原野の創出史・完全年代記**：単一トークンの固有揺らぎから、軌跡の直交合成、重力崩壊ストレステスト、双極・群島トポロジー解析、数学理論構築への昇華までを時系列順に完全記録)
- `PROJECT_CHRONICLE.md` (全開発史・総括年代記 第1章〜第6章)
- `goal_tree_engine_specification.md` (GDTE v3.4 概念核言語プリミティブ分解仕様書)
- `universal_trajectory_transfer_specification.md` (v5.1 粗視化被覆・再細分化重力地形 ＆ 純粋軌跡移転仕様書)

---

## ・使い方
本リポジトリのコード群はモジュール・実験ごとに独立しています。
**詳しい実行目的とメカニズムは、各 `.py` ファイルの先頭にある解説ブロック（Docstring）および `docs/` 内の仕様書をご覧ください。**
※実験途中のPoC（Proof of Concept）スクリプトや、特定のローカルモデルパスに依存しているコードも含まれているため、実行時は環境に合わせて適宜パスを修正してください。
