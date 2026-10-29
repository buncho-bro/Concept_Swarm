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
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" ANCHOR-FREE DATA-DRIVEN FRACTAL RESIDUAL DISTILLATION           ")
print("  - Rule 1 Applied: Zero Pre-defined Universal Concept Dictionaries")
print("  - Rule 3 Applied: Domain Topology Respect (Preserving Model Bias)")
print("==================================================================")
print(f"Device: {device}")

# ----------------------------------------------------------------------
# 1. Unbiased Data Stream (Raw Natural Text & Raw Code Stream)
# No predefined concept anchors. Pure natural problem contexts.
# ----------------------------------------------------------------------
SAMPLE_CORPUS = [
    # General Deductive Contexts (ProntoQA-like & factual reasoning)
    "Every carnivore is a hunter. Lions are carnivores. Leo is a lion therefore Leo is a hunter.",
    "Whales are aquatic mammals that breathe oxygen through blowholes in ocean water.",
    "Photosynthesis converts sunlight and carbon dioxide into glucose and oxygen in green plants.",
    "Gravity accelerates all falling bodies at the same rate in a complete vacuum.",
    "Democracy relies on public deliberation, voting rights, and constitutional checks on power.",
    "Inflation occurs when the money supply expands faster than economic output of goods.",
    "Memory consolidation transforms short-term synaptic traces into stable long-term neural networks.",
    "Planets orbit stars due to the spacetime curvature described by general relativity.",
    # Technical & Algorithmic Contexts (Code & systems reasoning)
    "def binary_search(arr, target): low, high = 0, len(arr) - 1; return binary_search_recursive(low, high)",
    "class RedBlackTree: def __init__(self): self.root = None; self.rebalance_tree()",
    "SELECT users.id, COUNT(orders.id) FROM users JOIN orders ON users.id = orders.user_id GROUP BY users.id",
    "async def fetch_data(url, session): async with session.get(url) as response: return await response.json()",
    "pthread_mutex_lock(&lock); shared_buffer[count++] = item; pthread_mutex_unlock(&lock);",
    "git commit -m 'Implement distributed consensus using Raft state machine replication'",
    "Docker container isolation leverages Linux namespaces, cgroups, and chroot environments.",
    "Convolutional layers perform spatial feature extraction using kernel weights across feature maps."
]

print(f"Loaded {len(SAMPLE_CORPUS)} natural domain sentences (General + Code, zero hand-picked concepts).")

# Load Teacher Model
print(f"Loading Teacher {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16, device_map=device)
model.eval()

# Extract Token Hidden States from the natural corpus (Extracting model's native representation)
print("Extracting Teacher native latent states from raw corpus...")
teacher_hidden_list = []
with torch.no_grad():
    for text in SAMPLE_CORPUS:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        out = model(**inputs, output_hidden_states=True)
        # Extract last layer states for all non-special tokens
        states = out.hidden_states[-1][0, 1:-1, :].float() # [SeqLen, 1536]
        teacher_hidden_list.append(states)

# Combine into a single native latent cloud
X_teacher = torch.cat(teacher_hidden_list, dim=0) # [TotalTokens, 1536]
X_teacher = F.normalize(X_teacher, p=2, dim=-1)
M_tokens, D_teacher = X_teacher.shape
print(f"Extracted {M_tokens} token latent points in {D_teacher}-dimensional space directly from data.")

# Student compression target: 384 dimensions (4x compression)
STUDENT_DIM = 384

# ----------------------------------------------------------------------
# 2. Anchor-Free Data-Driven Fractal Residual Engine
# ----------------------------------------------------------------------
class AnchorFreeFractalEngine:
    def __init__(self, n_clusters=6):
        self.n_clusters = n_clusters
        
    def discover_native_macro_gravity(self, X: torch.Tensor):
        """
        Unsupervised discovery of natural macro attractors directly from the model's own data.
        Zero human dictionary. Pure geometric spherical k-means.
        """
        X_np = X.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels_np = kmeans.fit_predict(X_np)
        
        cluster_assignments = torch.tensor(labels_np, device=X.device, dtype=torch.long)
        
        # Calculate centroids directly on unit sphere
        centroid_list = []
        for k in range(self.n_clusters):
            mask = (cluster_assignments == k)
            c_k = X[mask].mean(dim=0)
            c_k = F.normalize(c_k, p=2, dim=-1)
            centroid_list.append(c_k)
            
        centroids = torch.stack(centroid_list) # [K, D]
        return centroids, cluster_assignments

    def compute_data_driven_residuals(self, X: torch.Tensor, centroids: torch.Tensor, cluster_assignments: torch.Tensor):
        """
        Orthogonal Gravity Cancellation relative to the data's native centroids.
        """
        c_i = centroids[cluster_assignments] # [M, D]
        # Dot product projection
        proj = (X * c_i).sum(dim=-1, keepdim=True) * c_i
        residuals = X - proj
        
        residual_norms = torch.norm(residuals, p=2, dim=-1, keepdim=True)
        soft_weight = torch.tanh(residual_norms / 0.05)
        amplified_residuals = (residuals / (residual_norms + 1e-8)) * soft_weight
        
        return residuals, amplified_residuals, residual_norms.squeeze()

# ----------------------------------------------------------------------
# 3. Discover Teacher's Native Macro Attractors (Data-Driven)
# ----------------------------------------------------------------------
fractal_engine = AnchorFreeFractalEngine(n_clusters=6)
print("\nDiscovering Teacher's native macro attractors (K=6) via unsupervised spherical clustering...")
T_centroids, T_cluster_labels = fractal_engine.discover_native_macro_gravity(X_teacher)
_, T_amplified, T_res_norms = fractal_engine.compute_data_driven_residuals(X_teacher, T_centroids, T_cluster_labels)

print(f"Teacher Native Attractors: {T_centroids.shape[0]} clusters discovered automatically.")
print(f"Cluster distributions: {torch.bincount(T_cluster_labels).tolist()}")
print(f"Average residual strength: {T_res_norms.mean().item():.4f}")

# ----------------------------------------------------------------------
# 4. Compare 3 Paradigms under Unbiased Natural Data
# ----------------------------------------------------------------------
# Paradigm 1: Standard Flat MSE (Direct pair-distance MSE)
# Paradigm 2: Fixed Anchor Bias (Human-imposed artificial centroid forced onto data)
# Paradigm 3: Proposed Anchor-Free Data-Driven Fractal Distillation

EPOCHS = 350
LR = 0.01

def train_adaptor(paradigm_name: str):
    torch.manual_seed(42)
    adaptor = nn.Linear(D_teacher, STUDENT_DIM, bias=False).to(device)
    optimizer = torch.optim.AdamW(adaptor.parameters(), lr=LR, weight_decay=1e-4)
    
    # Subsample 250 points for pairwise loss scalability
    eval_idx = torch.randperm(M_tokens)[:250]
    X_T_sub = X_teacher[eval_idx]
    labels_sub = T_cluster_labels[eval_idx]
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        S_sub = F.normalize(adaptor(X_T_sub), p=2, dim=-1)
        
        if paradigm_name == "Standard_Flat_MSE":
            dist_T = torch.cdist(X_T_sub, X_T_sub)
            dist_S = torch.cdist(S_sub, S_sub)
            loss = F.mse_loss(dist_S, dist_T)
            
        elif paradigm_name == "Human_Anchor_Imposed":
            # Simulating human bias: forcing an arbitrary fixed random centroid not matching data topology
            arbitrary_centroid = torch.ones(STUDENT_DIM, device=device)
            arbitrary_centroid = F.normalize(arbitrary_centroid, p=2, dim=-1)
            # Pull all data towards human-selected arbitrary anchor
            anchor_dist = 1.0 - (S_sub @ arbitrary_centroid)
            loss = anchor_dist.mean() + F.mse_loss(torch.cdist(S_sub, S_sub), torch.cdist(X_T_sub, X_T_sub))
            
        elif paradigm_name == "Anchor_Free_Data_Driven_Fractal":
            # 1. Macro Loss on Data-driven centroids
            S_centroids = []
            for k in range(fractal_engine.n_clusters):
                mask = (labels_sub == k)
                if mask.sum() > 0:
                    c_s = F.normalize(S_sub[mask].mean(dim=0), p=2, dim=-1)
                else:
                    c_s = torch.zeros(STUDENT_DIM, device=device)
                S_centroids.append(c_s)
            S_centroids = torch.stack(S_centroids)
            
            macro_T = torch.cdist(T_centroids, T_centroids)
            macro_S = torch.cdist(S_centroids, S_centroids)
            loss_macro = F.mse_loss(macro_S, macro_T)
            
            # 2. Micro Residual Loss (Within native clusters)
            c_s_i = S_centroids[labels_sub]
            proj_s = (S_sub * c_s_i).sum(dim=-1, keepdim=True) * c_s_i
            res_s = S_sub - proj_s
            norm_res_s = torch.norm(res_s, p=2, dim=-1, keepdim=True)
            amplified_s = (res_s / (norm_res_s + 1e-8)) * torch.tanh(norm_res_s / 0.05)
            
            T_amp_sub = T_amplified[eval_idx]
            micro_M_T = torch.matmul(T_amp_sub, T_amp_sub.T)
            micro_M_S = torch.matmul(amplified_s, amplified_s.T)
            
            intra_mask = (labels_sub.unsqueeze(0) == labels_sub.unsqueeze(1))
            loss_micro = F.mse_loss(micro_M_S[intra_mask], micro_M_T[intra_mask])
            
            loss = loss_macro + 2.0 * loss_micro
            
        loss.backward()
        optimizer.step()
        
    return adaptor

print("\n--- Training 3 Paradigms on Natural Text/Code Latent Stream ---")
print("1. Standard Flat MSE...")
ad_flat = train_adaptor("Standard_Flat_MSE")

print("2. Human Anchor Imposed (Simulating dictionary bias)...")
ad_human = train_adaptor("Human_Anchor_Imposed")

print("3. Anchor-Free Data-Driven Fractal Residual...")
ad_free = train_adaptor("Anchor_Free_Data_Driven_Fractal")
print("Training complete.\n")

# ----------------------------------------------------------------------
# 5. Objective Evaluation (Preservation of Native Topology)
# ----------------------------------------------------------------------
print("==================================================================")
print(" RIGOROUS STATISTICAL EVALUATION: ZERO-DICTIONARY STREAM          ")
print("==================================================================")

def evaluate_paradigm(adaptor, name):
    with torch.no_grad():
        S = F.normalize(adaptor(X_teacher), p=2, dim=-1)
        
        # 1. Macro Correlation
        dist_T = torch.cdist(X_teacher, X_teacher).cpu().numpy().flatten()
        dist_S = torch.cdist(S, S).cpu().numpy().flatten()
        macro_corr = np.corrcoef(dist_T, dist_S)[0, 1] * 100.0
        
        # 2. Micro Native Precision (Top-3 nearest tokens in natural context)
        top3_retention = 0
        total_evals = 0
        N_eval = min(M_tokens, 200)
        
        for i in range(N_eval):
            # Same native cluster
            mask = (T_cluster_labels == T_cluster_labels[i])
            mask[i] = False
            if mask.sum() >= 3:
                sim_T = X_teacher[i] @ X_teacher[mask].T
                _, top_T = torch.topk(sim_T, k=3)
                
                sim_S = S[i] @ S[mask].T
                _, top_S = torch.topk(sim_S, k=3)
                
                overlap = len(set(top_T.cpu().tolist()).intersection(set(top_S.cpu().tolist())))
                top3_retention += overlap / 3.0
                total_evals += 1
                
        micro_precision = (top3_retention / total_evals) * 100.0 if total_evals > 0 else 0.0
        
        # 3. Model Distortion Index (Frobenius Error)
        f_err = torch.norm(torch.cdist(S[:150], S[:150]) - torch.cdist(X_teacher[:150], X_teacher[:150])).item()
        
        # 4. Native Effective Dimensionality (Participation Ratio)
        cov = np.cov(S.cpu().numpy(), rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-8]
        effective_dim = float((np.sum(eigvals)**2) / np.sum(eigvals**2))
        
    return {
        "name": name,
        "macro_corr": macro_corr,
        "micro_precision": micro_precision,
        "distortion_error": f_err,
        "effective_dim": effective_dim
    }

results = [
    evaluate_paradigm(ad_flat, "1. Standard Flat MSE"),
    evaluate_paradigm(ad_human, "2. Human-Imposed Anchor Bias"),
    evaluate_paradigm(ad_free, "3. Anchor-Free Data-Driven Fractal")
]

print(f"{'Paradigm':<38} | {'Macro Corr':<10} | {'Micro Precision':<16} | {'Distortion':<11} | {'Effective Dim':<13}")
print("-" * 98)
for r in results:
    print(f"{r['name']:<38} | {r['macro_corr']:>8.2f}% | {r['micro_precision']:>14.2f}% | {r['distortion_error']:>9.2f} | {r['effective_dim']:>13.2f}")

# Save JSON
out_json = os.path.join(OUTPUT_DIR, "anchor_free_fractal_distillation_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "tokens_evaluated": M_tokens,
        "teacher_dim": D_teacher,
        "student_dim": STUDENT_DIM,
        "clusters_discovered": 6,
        "results": results
    }, f, indent=2)
print(f"\nSaved benchmark results to {out_json}")

# Plot Comparison
labels = ["Flat MSE", "Human Anchor Bias", "Anchor-Free Fractal"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, [r["micro_precision"] for r in results], width, label="Micro Fine Precision (%)", color="#2ca02c")
rects2 = ax.bar(x + width/2, [r["effective_dim"] for r in results], width, label="Effective Dimensionality (Freedom)", color="#1f77b4")

ax.set_ylabel("Score", fontsize=11)
ax.set_title("Anchor-Free Data-Driven Distillation vs Human Anchor Bias", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
chart_path = os.path.join(ARTIFACT_DIR, "anchor_free_fractal_distillation_comparison.png")
plt.savefig(chart_path, dpi=300)
print(f"Generated chart at {chart_path}")
