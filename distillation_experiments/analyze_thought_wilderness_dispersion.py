import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" THOUGHT WILDERNESS DISPERSION ANALYSIS (思考原野の点拡散調査)   ")
print("  - Measuring Point Dispersion, Gravity Collapse & Fractal Spread")
print("==================================================================")
print(f"Device: {device}")

# 82 Concept Anchors across 8 Macro Valleys
CONCEPT_ANCHORS = [
    "Logic", "Axiom", "Deduction", "Induction", "Contradiction", "Proof", "Theorem", "Variable", "Constraint", "Inference",
    "Gravity", "Energy", "Entropy", "Atom", "Space", "Time", "Motion", "Force", "Quantum", "Light",
    "Cell", "Organism", "Evolution", "Gene", "Predator", "Prey", "Feline", "Canine", "Bird", "Mammal",
    "Thought", "Perception", "Emotion", "Belief", "Memory", "Intuition", "Desire", "Consciousness", "Intention", "Reason",
    "Society", "Law", "Government", "Economy", "Culture", "Language", "Justice", "War", "Peace", "Trade",
    "Algorithm", "Computer", "Network", "Database", "Program", "Circuit", "Software", "Hardware", "Robot", "Silicon",
    "Fact", "Myth", "Truth", "Illusion", "Evidence", "Hypothesis", "Reality", "Fiction", "Certainty", "Doubt",
    "House", "Bridge", "Road", "Forest", "Mountain", "River", "Ocean", "Food", "Water", "Fire", "Earth", "Air"
]
macro_labels = torch.tensor([i // 10 if i < 70 else 7 for i in range(len(CONCEPT_ANCHORS))], device=device)

# Load Teacher Embeddings from previous run or extract
print(f"Loading {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16, device_map=device)
model.eval()

teacher_vectors = []
with torch.no_grad():
    for word in CONCEPT_ANCHORS:
        prompt = f"The core concept of {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float()
        teacher_vectors.append(h)

teacher_tensor = torch.stack(teacher_vectors)
teacher_norm = F.normalize(teacher_tensor, p=2, dim=-1)
N, D = teacher_norm.shape
STUDENT_DIM = 384

# Re-train the 3 adaptors to ensure exact deterministic weights
class FractalGravityDistillationEngine:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def compute_macro_gravity_and_residuals(self, X: torch.Tensor, cluster_assignments: torch.Tensor):
        K = self.n_clusters
        centroid_list = []
        for k in range(K):
            mask = (cluster_assignments == k)
            if mask.sum() > 0:
                c_k = X[mask].mean(dim=0)
                c_k = F.normalize(c_k, p=2, dim=-1)
            else:
                c_k = torch.zeros(X.shape[-1], device=X.device)
            centroid_list.append(c_k)
        centroids = torch.stack(centroid_list)
        c_i = centroids[cluster_assignments]
        proj = (X * c_i).sum(dim=-1, keepdim=True) * c_i
        residuals = X - proj
        residual_norms = torch.norm(residuals, p=2, dim=-1, keepdim=True)
        soft_weight = torch.tanh(residual_norms / 0.05)
        amplified_residuals = (residuals / (residual_norms + 1e-8)) * soft_weight
        return centroids, residuals, amplified_residuals, residual_norms.squeeze()

fractal_engine = FractalGravityDistillationEngine(n_clusters=8)
T_centroids, _, T_amplified, _ = fractal_engine.compute_macro_gravity_and_residuals(teacher_norm, macro_labels)

def train_adaptor(method_name: str):
    torch.manual_seed(42)
    adaptor = nn.Linear(D, STUDENT_DIM, bias=False).to(device)
    optimizer = torch.optim.AdamW(adaptor.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(400):
        optimizer.zero_grad()
        S_norm = F.normalize(adaptor(teacher_norm), p=2, dim=-1)
        if method_name == "Flat_MSE":
            loss = F.mse_loss(torch.cdist(S_norm, S_norm), torch.cdist(teacher_norm, teacher_norm))
        elif method_name == "KNN_Masked":
            M_T = torch.matmul(teacher_norm, teacher_norm.T)
            M_S = torch.matmul(S_norm, S_norm.T)
            _, topk_idx = torch.topk(M_T, k=8, dim=-1)
            mask = torch.zeros_like(M_T, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            loss = F.mse_loss(M_S[mask], M_T[mask])
        elif method_name == "Fractal_Residual":
            S_centroids, _, S_amplified, _ = fractal_engine.compute_macro_gravity_and_residuals(S_norm, macro_labels)
            loss_macro = F.mse_loss(torch.cdist(S_centroids, S_centroids), torch.cdist(T_centroids, T_centroids))
            micro_M_T = torch.matmul(T_amplified, T_amplified.T)
            micro_M_S = torch.matmul(S_amplified, S_amplified.T)
            intra_mask = (macro_labels.unsqueeze(0) == macro_labels.unsqueeze(1))
            loss_micro = F.mse_loss(micro_M_S[intra_mask], micro_M_T[intra_mask])
            loss = loss_macro + 2.0 * loss_micro
        loss.backward()
        optimizer.step()
    return adaptor

print("Training adaptors for dispersion comparison...")
ad_flat = train_adaptor("Flat_MSE")
ad_knn = train_adaptor("KNN_Masked")
ad_frac = train_adaptor("Fractal_Residual")

with torch.no_grad():
    X_teacher = teacher_norm.cpu().numpy()
    X_flat = F.normalize(ad_flat(teacher_norm), p=2, dim=-1).cpu().numpy()
    X_knn = F.normalize(ad_knn(teacher_norm), p=2, dim=-1).cpu().numpy()
    X_frac = F.normalize(ad_frac(teacher_norm), p=2, dim=-1).cpu().numpy()

spaces = {
    "Teacher (1536d)": X_teacher,
    "Student Flat MSE (384d)": X_flat,
    "Student KNN Masked (384d)": X_knn,
    "Student Fractal Residual (384d)": X_frac
}

# ----------------------------------------------------------------------
# Quantitative Dispersion Metrics (物理・幾何学的統計量)
# ----------------------------------------------------------------------
metrics_summary = []

for name, X in spaces.items():
    # 1. Global Dispersion (Radius of Gyration / 全分散)
    centroid = np.mean(X, axis=0, keepdims=True)
    dists_from_center = np.linalg.norm(X - centroid, axis=1)
    global_dispersion = np.mean(dists_from_center) # 平均回転半径
    
    # 2. Pairwise Distances
    diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    pairwise_dists = np.linalg.norm(diffs, axis=-1)
    # Mask out self-distance (0.0)
    non_diag_mask = ~np.eye(N, dtype=bool)
    p_dists = pairwise_dists[non_diag_mask]
    
    mean_dist = np.mean(p_dists)
    std_dist = np.std(p_dists)
    min_dist = np.min(p_dists) # 崩壊・縮退の指標 (0に近すぎると衝突)
    
    # 3. Intra-cluster Dispersion (クラスタ内の広がり: 局所解像度)
    intra_dists = []
    labels_np = macro_labels.cpu().numpy()
    for k in range(8):
        k_indices = np.where(labels_np == k)[0]
        if len(k_indices) > 1:
            X_k = X[k_indices]
            c_k = np.mean(X_k, axis=0, keepdims=True)
            intra_dists.extend(np.linalg.norm(X_k - c_k, axis=1))
    intra_dispersion = np.mean(intra_dists)
    
    # 4. Effective Dimensionality (Participation Ratio / 有効自由度)
    # PR = (Tr(C))^2 / Tr(C^2)
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-8]
    participation_ratio = (np.sum(eigvals)**2) / np.sum(eigvals**2)
    
    metrics_summary.append({
        "space": name,
        "global_dispersion": float(global_dispersion),
        "mean_pairwise_dist": float(mean_dist),
        "min_pairwise_dist": float(min_dist),
        "intra_dispersion": float(intra_dispersion),
        "effective_dim": float(participation_ratio)
    })

print("\n" + "="*85)
print(f"{'Space':<32} | {'Global Radius':<13} | {'Mean Dist':<10} | {'Min Dist':<9} | {'Intra Spread':<12} | {'Effective Dim':<13}")
print("-" * 85)
for m in metrics_summary:
    print(f"{m['space']:<32} | {m['global_dispersion']:>13.4f} | {m['mean_pairwise_dist']:>10.4f} | {m['min_pairwise_dist']:>9.4f} | {m['intra_dispersion']:>12.4f} | {m['effective_dim']:>13.2f}")

# ----------------------------------------------------------------------
# 2D PCA & t-SNE Projections: Visualizing the Wilderness
# ----------------------------------------------------------------------
print("\nGenerating 2D Thought Wilderness Dispersion Maps...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# We use PCA for globally truthful orthogonal projection
pca = PCA(n_components=2, random_state=42)

domain_names = ["Logic", "Physics", "Biology", "Mind", "Society", "Tech", "Epistemic", "Environment"]
colors = plt.cm.tab10(np.linspace(0, 1, 8))

for idx, (name, X) in enumerate(spaces.items()):
    ax = axes[idx]
    coords = pca.fit_transform(X)
    
    labels_np = macro_labels.cpu().numpy()
    for k in range(8):
        k_pts = coords[labels_np == k]
        ax.scatter(k_pts[:, 0], k_pts[:, 1], color=colors[k], label=domain_names[k] if idx == 0 else "", alpha=0.75, s=60)
        
        # Draw Convex Hull or centroid star
        k_centroid = np.mean(k_pts, axis=0)
        ax.scatter(k_centroid[0], k_centroid[1], color=colors[k], marker='x', s=120, linewidths=2)
        # Connect points to centroid to show local gravity/spread
        for pt in k_pts:
            ax.plot([k_centroid[0], pt[0]], [k_centroid[1], pt[1]], color=colors[k], alpha=0.2, linestyle='-')
            
    m = metrics_summary[idx]
    ax.set_title(f"{name}\nIntra-Spread: {m['intra_dispersion']:.4f} | Eff-Dim: {m['effective_dim']:.1f}", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

fig.legend(domain_names, loc="lower center", ncol=8, fontsize=10, bbox_to_anchor=(0.5, 0.02))
plt.tight_layout(rect=[0, 0.05, 1, 0.98])

chart_path = os.path.join(ARTIFACT_DIR, "thought_wilderness_dispersion_maps.png")
plt.savefig(chart_path, dpi=300)
print(f"Saved Thought Wilderness dispersion map to {chart_path}")

# Save JSON Report
out_json = os.path.join(OUTPUT_DIR, "thought_wilderness_dispersion_metrics.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2)
print(f"Saved metrics JSON to {out_json}")
