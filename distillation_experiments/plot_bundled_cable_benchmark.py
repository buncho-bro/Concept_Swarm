"""
plot_bundled_cable_benchmark.py
-------------------------------
束ねたケーブルモデルのベンチマーク比較可視化
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

# 指標データ
metrics = ["Multi-hop (軌道追跡)", "Logic Chain (論理演繹)", "Hierarchy (階層分離度)"]

base_scores = [0.6985, 0.7773, 0.1403]
simple_hierarchical = [0.5939, 0.7147, 0.1197]
bundled_cable = [0.6950, 0.7441, 0.1403]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

rects1 = ax.bar(x - width, base_scores, width, label='1. Base 1.5B (標準)', color='#95a5a6', alpha=0.85, edgecolor='black')
rects2 = ax.bar(x, simple_hierarchical, width, label='2. 単純階層 (大核関所の引力トラップあり)', color='#e74c3c', alpha=0.85, edgecolor='black')
rects3 = ax.bar(x + width, bundled_cable, width, label='3. 束ねたケーブル方式 (手描き図モデル/直通合流)', color='#2ecc71', alpha=0.9, edgecolor='black')

for rects, color in [(rects1, 'black'), (rects2, 'black'), (rects3, 'darkgreen')]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, weight='bold', color=color)

ax.set_title("手描き図「束ねたケーブル方式」の実装効果: Multi-hop の劇的V字回復", fontsize=13, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, weight='bold')
ax.set_ylabel("Score / Similarity / Margin", fontsize=11)
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend(loc='lower left', framealpha=0.9, fontsize=9.5)

# 強調注釈（矢印）
ax.annotate('大核トラップを回避し\n+10.1% のV字急回復！',
            xy=(x[0] + width, bundled_cable[0]),
            xytext=(x[0] + width - 0.1, 0.45),
            arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
            fontsize=10, weight='bold', color='#27ae60', ha='center')

plt.tight_layout()
out_png = "distillation_experiments/bundled_cable_benchmark_comparison.png"
plt.savefig(out_png)
print(f"[+] グラフを保存しました: {out_png}")
