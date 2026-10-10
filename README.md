# 🌌 Antigravity Concept Swarm & Experimental Cores

VRAMゼロで動作する10体並列の「非言語合議型 AIエージェントスウォーム」を中心に、本プロジェクト内で派生・並行して実験されてきたあらゆるコア（量子化推論、マルチモーダル、進化計算、Project Ripple等）を集約した統合リポジトリです。途中でフリーズしたアイデアや未完の実験コードも、将来の発展のためにすべてアーカイブ・ドキュメント化して保存されています。

---

## 📂 リポジトリ構成（目次）

各Pythonファイルの先頭には、そのファイルの詳しい仕組みや目的を記載したDocstring（ブロックコメント）が埋め込まれています。各コードの詳細なメカニズムはファイル内を直接参照してください。

### 1. 🧠 `core/` (VRAM-Zero マルチエージェント合議エンジン)
*大次元の潜在ベクトルを共有メモリでやり取りし、テキスト変換なしで10体並列合議を行う中核システム。*
- `shared_memory_loader.py` / `latent_blackboard.py` / `agent_worker.py` / `multi_agent_swarm.py` / `dual_stream_decoder.py`

### 2. 📊 `benchmark/` (論理コーディング性能評価)
*MBPP+データセットを用いて、概念蒸留によるコーディング能力の変容を正確に測定するサンドボックス環境。*
- `run_mbppplus_benchmark.py` / `eval_worker.py`

### 3. 💾 `edge_micro_agent/` (SRAM 520KB 限界実装)
*RP2350 (Cortex-M33) などの極小エッジマイコン上で、局所100語展開（Local Topography）を用いて自律推論する組み込み向けAI。*
- `main.py` / `export_micro_topo.py` / `st7789.py`

### 4. 🔬 `alice_profiler/` (Zero-Full-Load 専門性抽出)
*160GBの巨大MoEモデル（AliceAI-80B）から、HTTP Rangeリクエストを用いて必要な数MBのテンソルだけを直接ストリーミング取得し、ディスクとRAM消費を極小に抑えつつ数学・コードExpertを抽出するプロファイラ。*
- `expert_topography_profiler.py`

### 5. 👁️ `multimodal/` (視覚と言語のセマンティック・ブリッジ)
*言語の概念空間と、CLIP等から得られる視覚（画像）の概念空間を大円干渉で結びつけるクロスモーダル実験群。*
- `multimodal_topology.py` / `semantic_bridge.py`

### 6. 💧 `project_ripple/` (波及効果型 物理推論エンジン)
*概念空間における推論を「波紋（Ripple）」や流体力学的な物理式に見立て、Web UIとWebSocketを通じてリアルタイムに操作・可視化するバックエンド。*
- `server.py` / `physics_core.py` / `standalone_concept_distiller.py` / etc.

### 7. 🧩 `quantized_core/` (量子化・KNN・フラクタルルーティング)
*量子化重みを用いた高速化実験や、KNNグラフによるトポロジー蒸留、さらにフラクタル構造を用いた再帰的なExpertルーティングなどの基礎実験群 (step1〜step8)。*
- `step1_quantized_distillation.py` 〜 `step8_hierarchical_decoding.py`

### 8. 🧬 `distillation_experiments/` (概念フォージ・進化計算・キメラ結合)
*プロジェクト初期〜中期にかけて試作された、ポテンシャル地形のGUI操作（Concept Forge）、ASTベースのMCTS探索（Evolutionary Tree Coder）、複数LoRAの局所結合（Chimera Injector）など、多様なプロトタイプ群。*
- `concept_forge_webui.py` / `evolutionary_tree_coder.py` / `chimera_injector.py` / `ultimate_resonant_tree.py` / etc.

---

## 🚀 使い方
本リポジトリのコード群はモジュール・実験ごとに独立しています。
**詳しい実行目的とメカニズムは、各 `.py` ファイルの先頭にある解説ブロック（Docstring）をご覧ください。**
※実験途中のPoC（Proof of Concept）スクリプトや、特定のローカルモデルパスに依存しているコードも含まれているため、実行時は環境に合わせて適宜パスを修正してください。
