import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from concept_learning_engine import DualSurfaceConceptLearner

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# --- 1. 大規模ストリーム概念データセットの構築 (150+ 概念) ---
def get_large_concept_stream():
    # 基幹アンカー (地表にフリーズ)
    ground_anchors = [
        "Physics and Quantum Mechanics",
        "Biology and Genetics",
        "Computer Science and Artificial Intelligence",
        "Economics and Finance",
        "Medicine and Pharmacology",
        "Philosophy and Literature"
    ]

    # オンラインで順次流入する学習ストリーム (多様なドメイン・階層・境界概念)
    # 同義語・バリエーションも含めることで「自律同化（Assimilation）」もテスト
    stream_concepts = [
        # --- ドメインA: 計算物理・ナノテク ---
        "Quantum Computing",
        "Superconducting Qubit",
        "Quantum Annealing",
        "Topological Insulator",
        "Spintronics",
        "Graphene Nanoribbon",
        "Quantum Cryptography",
        "Quantum Teleportation",
        "Photonics",
        "Condensed Matter Physics",

        # --- ドメインB: バイオインフォ・遺伝工学 ---
        "Bioinformatics",
        "Biochemistry",
        "CRISPR-Cas9 Gene Editing",
        "Synthetic Biology",
        "Ribosome Biogenesis",
        "Epigenetics DNA Methylation",
        "Single-cell RNA Sequencing",
        "Metabolic Pathway Engineering",
        "Structural Biology",
        "Molecular Dynamics Simulation",

        # --- ドメインC: AI・数理情報 ---
        "Convolutional Neural Network",
        "Transformer Attention Mechanism",
        "Reinforcement Learning with Human Feedback",
        "Large Language Model Reasoning",
        "Graph Neural Network",
        "Knowledge Graph Embedding",
        "Vector Database Indexing",
        "Diffusion Generative Model",
        "Autoencoder Latent Space",
        "Bayesian Neural Network",

        # --- ドメインD: 経済・金融工学 ---
        "Algorithmic Trading",
        "Econophysics",
        "Computational Economics",
        "Interest Rate Swap Derivative",
        "High-Frequency Market Microstructure",
        "Decentralized Finance Protocol",
        "Black-Scholes Option Pricing",
        "Behavioral Game Theory",
        "Liquidity Preference Hypothesis",
        "Macroeconomic Equilibrium",

        # --- ドメインE: 医療・創薬 ---
        "Pharmacogenomics",
        "Monoclonal Antibody Therapy",
        "Angiotensin Converting Enzyme Inhibitor",
        "Cancer Immunotherapy",
        "Precision Oncology",
        "Messenger RNA Vaccine Technology",
        "Neurodegenerative Amyloid Beta",
        "Cardiac Electrophysiology",
        "Drug-Drug Interaction Modeling",
        "Clinical Trial Biomarker",

        # --- ドメインF: 哲学・認知・言語 ---
        "Philosophy of Science",
        "Neurophilosophy",
        "Cognitive Linguistics",
        "Phenomenology of Perception",
        "Hermeneutic Circle",
        "Epistemological Relativism",
        "Consciousness Hard Problem",
        "Philosophy of Mind",
        "Ethical AI Governance",
        "Deconstructive Hermeneutics",

        # --- ドメインG: 芸術・デザイン・工芸 ---
        "Architectural Parametric Design",
        "Artisanal Ceramic Glaze",
        "Renaissance Oil Painting Impasto",
        "Traditional Origami Geometry",
        "Calligraphy Brushstroke Dynamics",
        "Vintage Horology Mechanical Escapement",
        "Handcrafted Leather Tanning",
        "Bonsai Botanical Sculpting",
        "Acoustic Violin Luthiery",
        "Symphonic Polyphony Counterpoint",

        # --- ドメインH: 地球科学・極限環境 ---
        "Plate Tectonic Subduction Zone",
        "Deep Sea Hydrothermal Vent Flora",
        "Desert Sand Dune Geomorphology",
        "Volcanic Magma Chamber Petrology",
        "Permafrost Methane Hydrate Release",
        "Atmospheric Stratospheric Ozone Chemistry",
        "Glacial Calving Ice Dynamics",
        "Oceanic Thermohaline Circulation",
        "Meteorological Supercell Mesocyclone",
        "Paleoclimatic Core Stratigraphy",

        # --- ドメインI: 料理・発酵・バイオ工学 ---
        "Sourdough Wild Yeast Microbiome",
        "Artisanal Cheese Lactic Fermentation",
        "Single Malt Whisky Cask Maturation",
        "Specialty Coffee Maillard Reaction",
        "Molecular Gastronomy Spherification",
        "Cacao Bean Anaerobic Fermentation",
        "Japanese Koji Mold Inoculation",
        "Extra Virgin Olive Oil Polyphenols",
        "Tea Leaf Oxidation Metallurgy",
        "Brewing Hop Terpene Chemistry",

        # --- ドメインJ: 宇宙・天体物理 ---
        "Gravitational Wave Interferometry",
        "Neutron Star Crust Nuclear Pasta",
        "Accretion Disk Magnetohydrodynamics",
        "Cosmic Microwave Background Polarization",
        "Exoplanet Atmospheric Spectroscopy",
        "Dark Matter WIMP Cross Section",
        "Active Galactic Nuclei Relativistic Jet",
        "Stellar Nucleosynthesis R-process",
        "Black Hole Ergosphere Superradiance",
        "Interstellar Molecular Cloud Dust",

        # --- ドメインK: 新素材・航空宇宙 ---
        "Carbon Fiber Reinforced Polymer",
        "Superalloy Turbine Blade Single Crystal",
        "Hypersonic Aerodynamic Boundary Layer",
        "Ion Thruster Xenon Propulsion",
        "Aerogel Thermal Superinsulation",
        "Metamaterial Negative Refractive Index",
        "Solid State Battery Electrolyte",
        "Phase Change Memory Chalcogenide",
        "Scramjet Supersonic Combustion",
        "Space Tether Orbital Mechanics",

        # --- 同化テスト用（類似・バリエーション概念） ---
        "Quantum Qubit Gate Operation",            # -> Superconducting Qubit へ同化期待
        "CRISPR Gene Knockout Protocol",           # -> CRISPR-Cas9 へ同化期待
        "Deep Transformer Neural Network",         # -> Transformer Attention へ同化期待
        "Algorithmic Automated High-Frequency Trading", # -> Algorithmic Trading へ同化期待
        "Antibody Drug Conjugate",                 # -> Monoclonal Antibody へ同化期待
        "Linguistic Structuralism",                # -> Cognitive Linguistics / Philosophy へ同化期待
        "Sourdough Bread Fermentation",            # -> Sourdough Wild Yeast へ同化期待
        "Ceramic Pottery Kiln Firing",             # -> Artisanal Ceramic Glaze へ同化期待
        "Gravitational Wave Laser Interferometer", # -> Gravitational Wave へ同化期待
        "Carbon Fiber Composite Manufacturing"     # -> Carbon Fiber Reinforced Polymer へ同化期待
    ]

    return ground_anchors, stream_concepts


def run_continual_learning_benchmark():
    print("================================================================")
    print(" 大規模オンライン概念学習ベンチマーク (N スケーリング実証)")
    print(" Autonomous Continual Concept Learning via Sky Magnet Levitation")
    print("================================================================")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    ground_anchors, stream_concepts = get_large_concept_stream()

    learner = DualSurfaceConceptLearner(embedder=embedder, dim=384, temperature=5.0)

    # 1. 地表基幹知識の配置
    for ga in ground_anchors:
        learner.add_ground_anchor(ga, weight=0.6, sigma=0.28)
    print(f"\n[1] 地表基幹知識 ({len(ground_anchors)}核) をフリーズ配置完了。")
    print(f"    入力ストリーム総数: {len(stream_concepts)} 概念")

    # 2. オンラインストリーム学習の実行
    print("\n[2] ストリーム概念学習を開始...")
    checkpoints = [20, 40, 60, 80, 100, 120]
    history = {
        "step": [],
        "num_magnets": [],
        "assimilations": [],
        "retention_rate": [],
        "recall_precision": [],
        "avg_step_ms": []
    }

    assimilated_count = 0
    t_start = time.time()

    for idx, concept in enumerate(stream_concepts):
        step_t0 = time.time()
        res = learner.learn_concept(concept, novelty_sim_threshold=0.82)
        step_ms = (time.time() - step_t0) * 1000.0

        if res["action"] == "ASSIMILATED_TO_SKY":
            assimilated_count += 1

        step_num = idx + 1
        if step_num in checkpoints or step_num == len(stream_concepts):
            # チェックポイント評価
            # 評価A: 地表基幹知識の100%保持テスト
            retention_count = 0
            for ga in ground_anchors:
                r = learner.recall(ga)
                if r["type"] == "GROUND_ANCHOR" and r["label"] == ga:
                    retention_count += 1
            retention_pct = (retention_count / len(ground_anchors)) * 100.0

            # 評価B: これまで学習した全概念の想起テスト (浮遊捕捉率)
            learned_so_far = stream_concepts[:step_num]
            recalled_count = 0
            for c in learned_so_far:
                r = learner.recall(c)
                # 浮遊捕捉され、自身のラベルまたは所属エイリアスと一致しているか
                if r["type"] == "SKY_LEVITATION":
                    if c in r["aliases"] or c == r["label"]:
                        recalled_count += 1
            recall_pct = (recalled_count / step_num) * 100.0

            history["step"].append(step_num)
            history["num_magnets"].append(len(learner.sky_centers))
            history["assimilations"].append(assimilated_count)
            history["retention_rate"].append(retention_pct)
            history["recall_precision"].append(recall_pct)
            history["avg_step_ms"].append(step_ms)

            print(f"  Step {step_num:3d}/{len(stream_concepts)} | 磁石数: {len(learner.sky_centers):3d} | 同化数: {assimilated_count:2d} | 地表保持率: {retention_pct:5.1f}% | 概念想起率: {recall_pct:5.1f}% | 1歩遅延: {step_ms:4.1f}ms")

    total_time = time.time() - t_start
    print(f"\n[3] 学習完了! 総所要時間: {total_time:.2f}秒 (0 VRAM / CPU)")

    # 3. グラフ描画
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)

    # グラフ1: N スケーリングにおける精度・保持率の推移
    ax1 = axes[0]
    ax1.plot(history["step"], history["retention_rate"], 'b-s', linewidth=2.5, markersize=8, label="Ground Anchor Retention (100% Target)")
    ax1.plot(history["step"], history["recall_precision"], 'r-o', linewidth=2.5, markersize=8, label="Continual Concept Recall Precision (%)")
    ax1.set_title("Continual Concept Learning Accuracy vs Stream Size (N)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Concepts Streamed (N)", fontsize=11)
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=10)

    # グラフ2: 磁石数 (新規配備) と同化数の成長曲線
    ax2 = axes[1]
    ax2.plot(history["step"], history["num_magnets"], 'g-^', linewidth=2.5, markersize=8, label="Active Sky Magnets Deployed")
    ax2.plot(history["step"], history["assimilations"], 'm--d', linewidth=2.0, markersize=8, label="Assimilated Variations (Clustering)")
    ax2.set_title("Concept Specialization vs Assimilation Growth", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Concepts Streamed (N)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=10)

    plt.suptitle("Scalable Continual Concept Learning Benchmark (Dual-Surface Magnetic Field)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_png = "scratch/continual_learning_scaling_benchmark.png"
    plt.savefig(out_png)
    plt.close()
    print(f"[4] ベンチマーク曲線を保存しました: {out_png}")

if __name__ == '__main__':
    run_continual_learning_benchmark()
