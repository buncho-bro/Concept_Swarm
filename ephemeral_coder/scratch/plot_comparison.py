import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)

N_stages = [10, 20, 30, 40, 50, 60]

# 1. 従来手法: 地表掘削（単一面・線形加算）
ground_recall = [45.0, 31.7, 1.7, 1.7, 1.7, 1.7]
ground_anchor_ret = [66.7, 33.3, 0.0, 0.0, 0.0, 0.0]
ground_crosstalk = [55.0, 68.3, 98.3, 98.3, 98.3, 98.3]

ax1 = axes[0]
ax1.plot(N_stages, ground_recall, 'm-o', linewidth=2.5, markersize=7, label="New Concept Recall Rate (%)")
ax1.plot(N_stages, ground_anchor_ret, 'b-s', linewidth=2.5, markersize=7, label="Ground Anchor Retention (%)")
ax1.plot(N_stages, ground_crosstalk, 'r--^', linewidth=2.0, markersize=7, label="Crosstalk / Black Hole Rate (%)")
ax1.axvspan(25, 65, color='red', alpha=0.1, label="Gravitational Collapse Zone")
ax1.set_title("[Previous] Single-Surface Ground Excavation\n(Catastrophic Gravitational Collapse)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Number of Added Concepts (N)", fontsize=11)
ax1.set_ylabel("Percentage (%)", fontsize=11)
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='center right', fontsize=9.5)

# 2. ユーザー考案手法: 上空磁石（両面化・LogSumExp磁気浮上）
mag_recall = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mag_anchor_ret = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mag_crosstalk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ax2 = axes[1]
# 重なりを可視化するため、線種・マーカー・オフセットを工夫
ax2.plot(N_stages, [y + 0.7 for y in mag_recall], color='#d62728', linestyle='-', linewidth=3.0, marker='o', markersize=8, label="Magnetic Levitation Recall (100.0% / offset +0.7%)")
ax2.plot(N_stages, [y - 0.7 for y in mag_anchor_ret], color='#1f77b4', linestyle='--', linewidth=2.5, marker='s', markersize=8, label="Ground Anchor Retention (100.0% / offset -0.7%)")
ax2.plot(N_stages, mag_crosstalk, 'k--^', linewidth=1.8, markersize=7, label="Crosstalk / Confusion Rate (0.0%)")
ax2.axvspan(0, 65, color='green', alpha=0.08, label="Zero-Erosion Levitation Zone (Safe)")
ax2.set_title("[User's Breakthrough] Sky Magnet Levitation Trap\n(100% Retention & Perfect Isolation Across 60 Concepts)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Number of Added Concepts (N)", fontsize=11)
ax2.set_ylabel("Percentage (%)", fontsize=11)
ax2.set_ylim(-5, 105)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='center right', fontsize=9.5)

plt.suptitle("Breakthrough Comparison: Ground Excavation vs Sky Magnetic Levitation (D=384, N=60 Real Embeddings)", fontsize=14, fontweight='bold')
plt.tight_layout()

out_png = "scratch/comparison_ground_vs_sky_magnet.png"
plt.savefig(out_png)
plt.close()
print(f"Updated comparison graph saved to: {out_png}")
