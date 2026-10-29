"""
plot_fractal_completion_benchmark.py
------------------------------------
フラクタル階層ネットワーク連動 文章補完AI ベンチマーク結果のグラフ可視化
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

report_path = "distillation_experiments/fractal_completion_benchmark_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = list(data["category_summary"].keys())
base_scores = [data["category_summary"][c]["base"] * 100 for c in categories]
fractal_scores = [data["category_summary"][c]["fractal"] * 100 for c in categories]

# 全体正解率も追加
categories.append("全体平均 (Total)")
base_scores.append(data["base_accuracy"] * 100)
fractal_scores.append(data["fractal_accuracy"] * 100)

x = np.arange(len(categories))
width = 0.32

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

rects1 = ax.bar(x - width/2, base_scores, width, label='Base AI (従来のフラット全探索)', color='#3498db', alpha=0.85, edgecolor='black')
rects2 = ax.bar(x + width/2, fractal_scores, width, label='Fractal Network AI (二階層ルーティング)', color='#e74c3c', alpha=0.85, edgecolor='black')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

ax.set_ylabel("正解率 (Accuracy %)", fontsize=11, weight='bold')
ax.set_title("フラクタル階層ネットワーク連動 文章補完AI ベンチマーク結果", fontsize=13, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10, weight='bold')
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

plt.tight_layout()
out_png = "distillation_experiments/fractal_completion_benchmark_comparison.png"
plt.savefig(out_png)
print(f"[+] グラフを保存しました: {out_png}")
