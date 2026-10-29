"""
Re-generate Clean Benchmark Chart without Cheated/Overfitted Self-Distillation Models
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "web_standard_large_scale_benchmark_results.json")

suites = ['causal_judgement', 'disambiguation_qa', 'formal_fallacies', 'truthfulqa_mc1', 'mmlu_moral']
suite_labels = ['Causal', 'Disambig', 'Fallacies', 'TruthfulQA', 'Moral']

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    all_results = json.load(f)

# カンニング・問題適応モデルを完全除外した、公平なモデルのみを登録
fair_models = [
    ("Raw_Qwen1.5B", "Raw Qwen 1.5B (Zero-Shot)", "#7f8c8d"),
    ("External_Text_Trained_LoRA", "External Real-Text LoRA (Alpaca 500)", "#2980b9"),
    ("Atomic_Proposition_Grounded_System", "Atomic Proposition Grounded (Ours)", "#27ae60")
]

plt.figure(figsize=(12, 6))
x = np.arange(len(suite_labels))
width = 0.25

for idx, (m_key, m_name, color) in enumerate(fair_models):
    if m_key in all_results:
        vals = [all_results[m_key].get(s, 0.0) for s in suites]
        avg = all_results[m_key].get("average", 0.0)
        plt.bar(x + idx * width, vals, width, label=f"{m_name} (Avg: {avg:.1f}%)", color=color, alpha=0.9)

plt.xlabel('Benchmark Suites (N=50 each, Total 250)', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('Web Standard Benchmark: Rigorous Comparison (Excluding Cheated/Overfitted Models)', fontsize=13, fontweight='bold')
plt.xticks(x + width, suite_labels, fontsize=10)
plt.ylim(0, 75)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
chart_path = os.path.join(ARTIFACT_DIR, "web_standard_large_scale_benchmark.png")
plt.savefig(chart_path, dpi=200)
plt.close()
print(f"Clean chart generated at: {chart_path}")
