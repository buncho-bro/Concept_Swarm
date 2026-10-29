"""
plot_boosted_benchmark_comparison.py
------------------------------------
強制重力深化（Density-Driven Deepening Boost）モデルの
Concept-Forge ベンチマーク結果を棒グラフで比較可視化
"""

import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

report_path = "distillation_experiments/boosted_concept_forge_benchmark_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

models = [
    ("Base 1.5B\n(Standard)", data["base_15b"], "#4a90e2"),
    ("Boosted Base 1.5B\n(Gravity Deepened)", data["boosted_base_15b"], "#50e3c2"),
    ("7B-KNN Distilled\n(LoRA)", data["knn_distilled_15b"], "#f5a623"),
    ("Boosted Distilled\n(LoRA + Deepened)", data["boosted_knn_distilled_15b"], "#bd10e0")
]

metrics = [
    ("Analogy (比喩推論)", "analogy"),
    ("Logic Chain (論理演繹)", "logic_chain"),
    ("Multi-hop (軌道追跡)", "multihop"),
    ("Hierarchy (階層分離度)", "hierarchy")
]

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
x = np.arange(len(metrics))
width = 0.2

for i, (m_name, m_data, color) in enumerate(models):
    scores = [m_data[key] for _, key in metrics]
    offset = (i - 1.5) * width
    rects = ax.bar(x + offset, scores, width, label=m_name.replace('\n', ' '), color=color, alpha=0.85, edgecolor='black', linewidth=0.8)
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, weight='bold')

ax.set_title("Concept-Forge Benchmark: Impact of Density-Driven Gravitational Deepening", fontsize=14, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in metrics], fontsize=11, weight='bold')
ax.set_ylabel("Score / Similarity / Margin", fontsize=11)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='upper left')

plt.tight_layout()
out_png = "distillation_experiments/boosted_concept_forge_benchmark_comparison.png"
plt.savefig(out_png)
print(f"[+] グラフを保存しました: {out_png}")
