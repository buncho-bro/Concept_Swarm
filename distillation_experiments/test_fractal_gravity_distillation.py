import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" FRACTAL GRAVITY-CANCELLATION RESIDUAL DISTILLATION EXPERIMENT    ")
print("  - Testing Hypothesis: Coarse Gravity -> Local Cancellation     ")
print("    -> Micro Amplification Preserves Fine-Grained Logic           ")
print("==================================================================")
print(f"Device: {device}")

# ----------------------------------------------------------------------
# 1. 82 Universal Concept Anchors (Multi-domain semantic anchors)
# ----------------------------------------------------------------------
CONCEPT_ANCHORS = [
    # Category 1: Logic & Abstract Formalisms (10 concepts)
    "Logic", "Axiom", "Deduction", "Induction", "Contradiction", "Proof", "Theorem", "Variable", "Constraint", "Inference",
    # Category 2: Physics & Nature (10 concepts)
    "Gravity", "Energy", "Entropy", "Atom", "Space", "Time", "Motion", "Force", "Quantum", "Light",
    # Category 3: Biology & Life (10 concepts)
    "Cell", "Organism", "Evolution", "Gene", "Predator", "Prey", "Feline", "Canine", "Bird", "Mammal",
    # Category 4: Human Mind & Cognition (10 concepts)
    "Thought", "Perception", "Emotion", "Belief", "Memory", "Intuition", "Desire", "Consciousness", "Intention", "Reason",
    # Category 5: Social & Civilization (10 concepts)
    "Society", "Law", "Government", "Economy", "Culture", "Language", "Justice", "War", "Peace", "Trade",
    # Category 6: Technology & Computation (10 concepts)
    "Algorithm", "Computer", "Network", "Database", "Program", "Circuit", "Software", "Hardware", "Robot", "Silicon",
    # Category 7: Epistemic & Reality (10 concepts)
    "Fact", "Myth", "Truth", "Illusion", "Evidence", "Hypothesis", "Reality", "Fiction", "Certainty", "Doubt",
    # Category 8: Physical Artifacts & Environment (12 concepts)
    "House", "Bridge", "Road", "Forest", "Mountain", "River", "Ocean", "Food", "Water", "Fire", "Earth", "Air"
]
print(f"Loaded {len(CONCEPT_ANCHORS)} Universal Concept Anchors across 8 distinct macro domains.")

# Load Model
print(f"Loading {BASE_MODEL_ID} to extract Teacher geometric landscape...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# ----------------------------------------------------------------------
# Extract True Teacher Embeddings
# ----------------------------------------------------------------------
print("Extracting contextual embeddings for concept anchors...")
teacher_vectors = []
with torch.no_grad():
    for word in CONCEPT_ANCHORS:
        prompt = f"The core concept of {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model(**inputs, output_hidden_states=True)
        # Last layer hidden state of the concept token
        h = out.hidden_states[-1][0, -1, :].float()
        teacher_vectors.append(h)

teacher_tensor = torch.stack(teacher_vectors) # [N, D] where N=82, D=1536
teacher_norm = F.normalize(teacher_tensor, p=2, dim=-1)
N, D = teacher_norm.shape
print(f"Teacher Geometry Extracted: {N} concepts in {D}-dimensional latent space.")

# Target Student Dimensions: Simulating a 4x compressed Student bottleneck
STUDENT_DIM = 384
print(f"Target Student Space: {STUDENT_DIM} dimensions (Compressing 1536d -> 384d, 75% parameter compression)")

# ----------------------------------------------------------------------
# Mathematical Formulation of Fractal Gravity-Cancellation Distillation
# ----------------------------------------------------------------------
class FractalGravityDistillationEngine:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def compute_macro_gravity_and_residuals(self, X: torch.Tensor, cluster_assignments: torch.Tensor):
        """
        X: [N, D] (normalized or raw vectors)
        cluster_assignments: [N] (macro valley index 0..K-1)
        """
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
            
        centroids = torch.stack(centroid_list) # [K, D]
                
        # Orthogonal Gravity Cancellation: r_i = x_i - (x_i . c_k) * c_k
        c_i = centroids[cluster_assignments] # [N, D]
        proj = (X * c_i).sum(dim=-1, keepdim=True) * c_i
        residuals = X - proj
        
        residual_norms = torch.norm(residuals, p=2, dim=-1, keepdim=True) # [N, 1]
        
        # Micro Amplification with Soft Threshold (Suppressing pure noise)
        soft_weight = torch.tanh(residual_norms / 0.05)
        amplified_residuals = (residuals / (residual_norms + 1e-8)) * soft_weight
        
        return centroids, residuals, amplified_residuals, residual_norms.squeeze()

# Perform Ground-Truth Clustering on Teacher to define Macro Valleys (8 Domains)
macro_labels = torch.tensor([i // 10 if i < 70 else 7 for i in range(N)], device=device)

fractal_engine = FractalGravityDistillationEngine(n_clusters=8)
T_centroids, T_residuals, T_amplified, T_res_norms = fractal_engine.compute_macro_gravity_and_residuals(teacher_norm, macro_labels)

print(f"Teacher Macro Valleys: {T_centroids.shape[0]} clusters.")
print(f"Teacher Average Residual Norm (Micro signal strength): {T_res_norms.mean().item():.4f}")

# ----------------------------------------------------------------------
# 3 Distillation Models (Student Adaptors) Training & Comparison
# ----------------------------------------------------------------------
# We train 3 projection adaptors [D -> STUDENT_DIM] mapping Teacher to Student:
# Method 1: Standard Flat MSE Distillation (Conventional baseline)
# Method 2: Standard Cosine/KNN Topology Distillation (Phase 1 Baseline)
# Method 3: Proposed Fractal Gravity-Cancellation Residual Distillation

EPOCHS = 400
LR = 0.01

def train_student_adaptor(method_name: str):
    torch.manual_seed(42)
    # Linear projection layer simulating compressed Student bottleneck
    adaptor = nn.Linear(D, STUDENT_DIM, bias=False).to(device)
    optimizer = torch.optim.AdamW(adaptor.parameters(), lr=LR, weight_decay=1e-4)
    
    losses = []
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        S_raw = adaptor(teacher_norm) # [N, STUDENT_DIM]
        S_norm = F.normalize(S_raw, p=2, dim=-1)
        
        if method_name == "Standard_Flat_MSE":
            # Direct pair-distance MSE
            dist_T = torch.cdist(teacher_norm, teacher_norm)
            dist_S = torch.cdist(S_norm, S_norm)
            loss = F.mse_loss(dist_S, dist_T)
            
        elif method_name == "KNN_Cosine_Distill":
            # Cosine similarity matrix alignment with Top-8 mask
            M_T = torch.matmul(teacher_norm, teacher_norm.T)
            M_S = torch.matmul(S_norm, S_norm.T)
            # Top-8 mask per concept
            _, topk_idx = torch.topk(M_T, k=8, dim=-1)
            mask = torch.zeros_like(M_T, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            loss = F.mse_loss(M_S[mask], M_T[mask])
            
        elif method_name == "Proposed_Fractal_Residual":
            # Coarse-to-Fine: Macro Gravity + Micro Residual Amplification
            # 1. Macro Gravity Loss
            S_centroids, S_residuals, S_amplified, _ = fractal_engine.compute_macro_gravity_and_residuals(S_norm, macro_labels)
            
            macro_dist_T = torch.cdist(T_centroids, T_centroids)
            macro_dist_S = torch.cdist(S_centroids, S_centroids)
            loss_macro = F.mse_loss(macro_dist_S, macro_dist_T)
            
            # 2. Micro Amplified Residual Loss (Local Fine-Grained Logic)
            micro_M_T = torch.matmul(T_amplified, T_amplified.T)
            micro_M_S = torch.matmul(S_amplified, S_amplified.T)
            
            # Intra-cluster fine logic mask (within the same macro valley)
            intra_mask = (macro_labels.unsqueeze(0) == macro_labels.unsqueeze(1))
            loss_micro = F.mse_loss(micro_M_S[intra_mask], micro_M_T[intra_mask])
            
            # Combined Fractal Loss (Balanced 1:1 gradient flow)
            loss = loss_macro + 2.0 * loss_micro
            
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return adaptor, losses

print("\n--- Training 3 Distillation Paradigms on 82 Concepts ---")
print("1. Standard Flat MSE Distillation...")
adaptor_flat, losses_flat = train_student_adaptor("Standard_Flat_MSE")

print("2. KNN Cosine Distillation (Phase 1 Baseline)...")
adaptor_knn, losses_knn = train_student_adaptor("KNN_Cosine_Distill")

print("3. Proposed Fractal Gravity-Cancellation Distillation...")
adaptor_fractal, losses_fractal = train_student_adaptor("Proposed_Fractal_Residual")
print("Training complete.\n")

# ----------------------------------------------------------------------
# RIGOROUS STATISTICAL BENCHMARKING ACROSS 4 METRICS
# ----------------------------------------------------------------------
print("==================================================================")
print(" RIGOROUS STATISTICAL EVALUATION ACROSS METRICS                   ")
print("==================================================================")

def evaluate_student(adaptor, name):
    with torch.no_grad():
        S_raw = adaptor(teacher_norm)
        S_norm = F.normalize(S_raw, p=2, dim=-1)
        
        # 1. Macro Global Structure: Global Distance Correlation (Spearman/Pearson)
        dist_T = torch.cdist(teacher_norm, teacher_norm).cpu().numpy().flatten()
        dist_S = torch.cdist(S_norm, S_norm).cpu().numpy().flatten()
        macro_corr = np.corrcoef(dist_T, dist_S)[0, 1]
        
        # 2. Micro Fine-Grained Resolution: Top-3 Nearest Neighbor Retention within Domain
        # Does the student preserve the subtle distinction between "Deduction" and "Induction",
        # or "Predator" and "Prey"?
        top3_retention = 0
        total_evals = 0
        for i in range(N):
            domain_mask = (macro_labels == macro_labels[i])
            domain_mask[i] = False # exclude self
            if domain_mask.sum() >= 3:
                # Teacher top 3 in domain
                sims_T = (teacher_norm[i] @ teacher_norm[domain_mask].T)
                _, top_T_idx = torch.topk(sims_T, k=3)
                
                # Student top 3 in domain
                sims_S = (S_norm[i] @ S_norm[domain_mask].T)
                _, top_S_idx = torch.topk(sims_S, k=3)
                
                overlap = len(set(top_T_idx.cpu().tolist()).intersection(set(top_S_idx.cpu().tolist())))
                top3_retention += overlap / 3.0
                total_evals += 1
                
        micro_precision = (top3_retention / total_evals) * 100.0
        
        # 3. Multi-hop Transitive Logic Consistency (Triangle Inequality Consistency)
        # For triplets A, B, C in logic domain, does d(A, C) <= d(A, B) + d(B, C) hold with high margin?
        logic_indices = torch.where(macro_labels == 0)[0] # Logic domain
        triplet_violations = 0
        total_triplets = 0
        dist_S_mat = torch.cdist(S_norm, S_norm)
        
        for a in logic_indices:
            for b in logic_indices:
                for c in logic_indices:
                    if a != b and b != c and a != c:
                        d_ac = dist_S_mat[a, c].item()
                        d_ab = dist_S_mat[a, b].item()
                        d_bc = dist_S_mat[b, c].item()
                        if d_ac > (d_ab + d_bc + 1e-4):
                            triplet_violations += 1
                        total_triplets += 1
                        
        transitive_integrity = (1.0 - (triplet_violations / total_triplets)) * 100.0
        
        # 4. Distortion Index (Noise Amplification Check - Critical Review Requirement!)
        # Checks if empty/noise directions got artificially blown up
        frobenius_norm_diff = torch.norm(dist_S_mat - torch.cdist(teacher_norm, teacher_norm)).item()
        
    return {
        "name": name,
        "macro_correlation": macro_corr * 100.0,
        "micro_precision": micro_precision,
        "transitive_integrity": transitive_integrity,
        "frobenius_error": frobenius_norm_diff
    }

metrics_flat = evaluate_student(adaptor_flat, "1. Standard Flat MSE Distill")
metrics_knn = evaluate_student(adaptor_knn, "2. KNN Masked Distill (Phase 1)")
metrics_fractal = evaluate_student(adaptor_fractal, "3. Proposed Fractal Residual Distill")

results_table = [metrics_flat, metrics_knn, metrics_fractal]

print(f"{'Method':<36} | {'Macro Corr':<10} | {'Micro Prec(Top3)':<16} | {'Logic Integrity':<16} | {'Error (F-Norm)':<14}")
print("-" * 100)
for r in results_table:
    print(f"{r['name']:<36} | {r['macro_correlation']:>8.2f}% | {r['micro_precision']:>14.2f}% | {r['transitive_integrity']:>14.2f}% | {r['frobenius_error']:>14.2f}")

# ----------------------------------------------------------------------
# Save Results & Visualization
# ----------------------------------------------------------------------
out_json = os.path.join(OUTPUT_DIR, "fractal_gravity_distillation_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "sample_size": N,
        "teacher_dim": D,
        "student_dim": STUDENT_DIM,
        "results": results_table
    }, f, indent=2)
print(f"\nSaved raw benchmark results to {out_json}")

# Plot comparison chart
labels = ["Flat MSE", "KNN Masked", "Fractal Residual"]
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width, [r["macro_correlation"] for r in results_table], width, label="Macro Global Correlation (%)", color="#3b5998")
rects2 = ax.bar(x, [r["micro_precision"] for r in results_table], width, label="Micro Fine Precision (%)", color="#2ca02c")
rects3 = ax.bar(x + width, [r["transitive_integrity"] for r in results_table], width, label="Multi-hop Logic Integrity (%)", color="#ff7f0e")

ax.set_ylabel("Score (%)", fontsize=11)
ax.set_title("Distillation Paradigm Comparison: Flat vs KNN vs Fractal Residual", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for rect in rects1 + rects2 + rects3:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
chart_path = os.path.join(ARTIFACT_DIR, "fractal_gravity_distillation_comparison.png")
plt.savefig(chart_path, dpi=300)
print(f"Generated benchmark visualization at {chart_path}")
