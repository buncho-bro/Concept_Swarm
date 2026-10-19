# 🌌 Antigravity Concept Swarm & Thought Wilderness

> **「巨大な言語モデルから、純粋な『概念のトポロジー』を抽出し、言葉を介さずにミリ秒で合議させ、エッジから自己組織化する思考原野までを解き明かす」**

自己組織化ポテンシャル場（Concept Core）による「概念地形そのものの構築」から始まり、人間の辞書・ラベルを1語も使わない「純粋な蒸留」、VRAMゼロで動作する10体並列の「非言語合議型AIエージェントスウォーム」、極小エッジマイコン（520KB SRAM）での自律推論、巨大80BモデルのHTTPストリーミング解体、粗視化被覆による純粋軌跡移転、そして単語の全層軌跡から自己組織化する超並列推論基盤**「思考原野（Thought Wilderness）」**に至るまで、ユーザーとAIが二人三脚で切り拓いた一連の幾何学的AI探求の成果を集約した統合リポジトリです。

---

## 📜 プロジェクトの理念と免責事項
- **Google Antigravityによる開発:** 本プロジェクトのコアアルゴリズム、スクリプト、およびUIアーキテクチャは、Google Antigravity AIエージェントフレームワークを使用し、主に Gemini との対話・協働を通じて開発されました。
- **公平なAI競争の促進:** 本プロジェクトの最大の目標は、AI開発における公平な競争と民主化の促進です。開発者がローカル環境で効率的に「概念地形」「思考原野」を抽出・自己組織化できる手法を提供することで、AI研究参入への障壁を下げることを目指しています。既存のAIシステムを破壊、搾取、または妨害することを意図したものでは決してありません。あくまで最適化と平等性を追求する学術的・工学的な試みです。
- **オリジナルAIライセンスの遵守:** 本リポジトリのスクリプトを使用して知識の抽出やモデルの蒸留を行う際は、対象となるAIモデルの元の開発者（Hugging Face、OpenAI、Z.ai、Alibaba、Mistral AIなど）が定める利用規約、ライセンス、および利用ルールを厳守してください。

---

## 🏛️ 時系列リポジトリ構成（完全目次）

本リポジトリは、対話の進展と技術的ブレイクスルーの時系列順に構成されています。各Pythonファイルの先頭には、詳細な仕組みを記したDocstringが埋め込まれています。

```
Antigravity_Concept_Swarm/
├── concept_core/             # [第0章] 自己組織化ポテンシャル場・浸食/風化・DoG尾根・ビー玉推論 (Pentagon)
├── distillation_experiments/ # [第1章/第6章/第7章] 純粋蒸留 (0-VRAM大円干渉/スライストポロジー) & 思考原野
├── core/                     # [第2章] 潜在空間ブラックボード (2048d) & Dual-Stream デコーダー
├── project_ripple/           # [第2章] 物理コア・水面の波紋干渉・ピクセルアートWebUI
├── quantized_core/           # [第3章] 低ビット量子化・KNNグラフ蒸留・フラクタルルーティング
├── edge_micro_agent/         # [第3章] RP2350 (520KB SRAM) 局所100語マイコンエージェント
├── alice_profiler/           # [第4章] 160GB (AliceAI-80B) のHTTP Rangeストリーミング走査
├── alice_relay_engine/       # [第4章] 疎結合オンデマンドMoE & 数学➔コード非言語バトンパス
├── game_prototype/           # [第5章] 潜在物理型AI執政官・国家指導者アカデミー
├── alife_agent_sim/          # [第5章] 連続空間エージェント・Boids群れ・環境知識淘汰 (V1〜V4)
├── ephemeral_coder/          # [第6章] 逆算型目標樹形図エンジン (GDTE v3.4: 言語プリミティブ分解)
└── docs/                     # [記録] 全体ドキュメンテーション・仕様書・思想年代記
```

---

### 0. `concept_core/` (第0章：原点――概念地形そのものの構築)
*静的コサイン類似度から脱却し、意味を「重力ポテンシャル場」として扱ったプロジェクトの発祥地。*
- `concept_core.py` / `run_experiments.py` : 流体浸食（Erosion）・風化（Weathering）によるY字型合流谷の形成とビー玉推論シミュレーション。
- `high_dim_concept_core.py` / `run_capacity_limits.py` : 64次元空間でのブラックホール崩壊（想起率0%）の発見と、Difference of Gaussians（DoG/メキシカンハット）尾根による分水嶺形成・キャパシティ維持。
- `pentagon_core.py` / `app_pentagon.py` : CLIPマルチモーダル統合、文脈の余熱（Thermal Trace）によるタイポの物理デノイズ、多言語対応GUI推論エンジン（Pentagon）。

### 1. `distillation_experiments/` (第1章：純粋な蒸留の夜明け)
*人間が選んだ単語リストやラベルを全廃し、巨大モデルの潜在トポロジーを無歪みで転写する純粋蒸留技術。*
- `zero_vram_great_circle_genesis.py` : 教師モデル `GLM-5.3`（15.4万語×6144d）から、**VRAM 0MB（CPU mmap）** で全語彙の大円球面干渉により自律的に54個の無名イデア核を析出。
- `slice_topological_distiller.py` : 主軸等分位スライスと相対無次元化距離行列 $\mathbf{M}_T = \mathbf{D}(P_T)/k(T)$ により、文字を1語も介さずに生徒モデル `Qwen2.5-Coder-3B` へ同相LoRA蒸留（**トポロジー誤差97.86%削減**）。

### 2. `core/` & `project_ripple/` (第2章：非言語合議スウォームと波紋干渉)
*大次元の潜在ベクトルを共有メモリでやり取りし、テキスト変換なしで10体並列合議を行う中核システム。*
- `shared_memory_loader.py` / `latent_blackboard.py` / `agent_worker.py` / `multi_agent_swarm.py` : OS共有メモリによるゼロコピー・サブミリ秒合議（< 1ms）。
- `dual_stream_decoder.py` : 概念核（Stream 1）と語彙アンカー（Stream 2）のLogit Bias干渉による文法崩壊防止。
- `project_ripple/physics_core.py` / `server.py` : 水面波紋物理とピクセル少女の感情移入WebUI。

### 3. `quantized_core/` & `edge_micro_agent/` (第3章：極限の圧縮とエッジ到達)
*量子化重みを用いた高速化やKNNグラフ蒸留、そしてSRAMわずか520KBの RP2350 (Cortex-M33) マイコン上で局所100語展開（Local Topography）を用いて自律推論する組み込み向けAI。*
- `main.py` / `export_micro_topo.py` / `st7789.py` / `step1_quantized_distillation.py` 〜 `step8_hierarchical_decoding.py`

### 4. `alice_profiler/` & `alice_relay_engine/` (第4章：80B巨大モデルのオンデマンド解体)
*160GBの巨大MoEモデル（AliceAI-80B）から、HTTP Rangeリクエストを用いて必要な数MBだけを直接ストリーミング取得し、数学AIからコードAIへ非言語バトンパスを行う疎結合推論エンジン。*
- `expert_topography_profiler.py` / `alice_relay_engine.py` : ディスク0GB・RAM 100MB未満での走査と、IPCテキスト変換0回の非言語バトンパス。

### 5. `game_prototype/` & `alife_agent_sim/` (第5章：生命と社会のシミュレーション)
*潜在物理型AI執政官のゲーム応用、および16次元概念ベクトルをDNAとして持つ人工生命（Semantic ALife V1〜V4）の連続空間自律進化シミュレータ。*
- `server.py` / `alife_sim.py` : 概念DNA細胞、遊牧青と捕食要塞赤の二分創発、流体摩擦と老化による生態系安定化。

### 6. `ephemeral_coder/` & `distillation_experiments/` v5.1 (第6章：目標樹形図と言語プリミティブ分解・純粋軌跡移転)
*目標ツリーを「言語の組み込みプリミティブ（組み込み関数・演算子）」に行きつくまで再帰分解し、末端Workerが構文的補間のみを行う自律開発エンジン、および純粋軌跡移転。*
- `ephemeral_coder/engine/tree_architect.py` / `engine/rule_formulator.py` : GDTE v3.4 概念核言語プリミティブ分解。
- `coarse_gravity_joint_terrain_engine.py` / `universal_vision_inspector_webui.py` (v5.1) : 11,845個の無名重力核の熱拡散粗視化（$R=48$）と半直交等長再細分化、画像概念核の落下軌跡を文章概念核へ直接移転（Pure Geodesic Trajectory Transfer）。

### 7. `distillation_experiments/` (第7章：思考原野――Thought Wilderness)
*単語の全層軌跡の類似度から自己組織化する「思考原野（Thought Wilderness）」実験群。*
- `massive_full_trajectory_benchmark.py` : 各単語が止まるまでの全33層軌跡の直交力線を積分する100組大規模因果ベンチマーク（`crime+police➔arrest`, `cloud+cold➔ice` 等の行為・因果射抜）。
- `thought_wilderness_builder.py` : 線の類似度から面のマップを展開し、円形引力とパウリ斥力で自己組織化・安定化する思考原野エンジン。
- `full_vocab_collapse_stress_test.py` : 全語彙（7.4万語）規模での重力崩壊（底なし沼化）検証と体積排除（近接斥力・パウリの排他律）による宇宙安定化。
- `model_comparison_wilderness.py` : Qwen-Coder（双極惑星） vs Mistral-7B（形態素群島） vs 汎用Qwen（渦巻銀河）の知性の指紋解析。
- `math_theory_wilderness_v2.py` : 単語のみで完結する数学的理論構築（群+可換➔環、行列+固有値➔分解、多様体+曲率➔テンソル）と超並列幾何学推論の実証。

---

## 📚 必読ドキュメント

- [`docs/PROJECT_CHRONICLE.md`](docs/PROJECT_CHRONICLE.md) : **全体開発史・総括年代記（第0章〜第7章完結版）**。概念地形そのものの構築、純粋な蒸留、非言語スウォーム、エッジ、巨大モデル解体、人工生命、純粋軌跡移転、そして思考原野に至る全軌跡、直面した壁、ブレイクスルーを時系列順に完全網羅。
- [`docs/THOUGHT_WILDERNESS_CHRONICLE.md`](docs/THOUGHT_WILDERNESS_CHRONICLE.md) : **思考原野の創出史・集中年代記**。単一トークンの固有揺らぎ、全軌道掛け合わせ、重力崩壊とパウリ排他律、モデル別トポロジー解析、数学理論構築と超並列性を時系列順に完全記録。
- `docs/goal_tree_engine_specification.md` : GDTE v3.4 概念核言語プリミティブ分解仕様書。
- `docs/universal_trajectory_transfer_specification.md` : v5.1 粗視化被覆・再細分化重力地形 ＆ 純粋軌跡移転仕様書。

---

## 🚀 使い方
本リポジトリのコード群はモジュール・実験ごとに独立して実行可能です。  
**詳しい実行目的とメカニズムは、各 `.py` ファイルの先頭にある解説ブロック（Docstring）および [`docs/PROJECT_CHRONICLE.md`](docs/PROJECT_CHRONICLE.md) をご覧ください。**
