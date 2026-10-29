"""
Restructure Thought Wilderness: Invariant-Guided Field Untangling
(思考原野の幾何学的再構成・矛盾解消スクリプト)
================================================================================
Eliminates latent contradictions and spurious correlations in the Thought Wilderness
while preserving the full-manifold topology and reasoning freedom.

Key Components:
1. Contradiction Repulsive Potential (pushes myths/errors into repulsive orthogonal space)
2. Invariant Fact Alignment Potential (pulls verified empirical invariants close)
3. Manifold Topology Preservation (prevents collapse of natural conceptual diversity)
4. LoRA Adapter Training on Qwen2.5-1.5B (updates the actual model weights)
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from diagnose_thought_wilderness_contradictions import CONTRADICTION_PROBES

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lora_restructured_thought_wilderness")

# Broad General Natural Reasoning Prompts to preserve full-manifold topology
TOPOLOGY_PRESERVATION_CORPUS = [
    "The core concept of Logic is systematic reasoning based on valid inference.",
    "Gravity is a fundamental physical interaction causing mutual attraction between mass.",
    "Evolution by natural selection describes differential reproduction of biological organisms.",
    "Human cognition involves neural processing across complex brain networks.",
    "The legal system establishes justice, rights, and regulatory frameworks in society.",
    "Computer architecture organizes arithmetic computation, memory hierarchies, and bus systems.",
    "Empirical evidence requires rigorous experimentation and observational verification.",
    "Ecosystems depend on thermodynamic energy transfer across trophic levels in the biosphere.",
    "Mathematical theorems are proved from fundamental axioms through deductive inference.",
    "Cellular respiration converts glucose and oxygen into adenosine triphosphate energy."
]


def extract_concept_rep(model, tokenizer, text: str) -> torch.Tensor:
    """Extracts concept vector with gradients enabled."""
    prompt = f"Concept representation: {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-2][0]  # Late hidden state
    h_vec = h.mean(dim=0)
    return F.normalize(h_vec, p=2, dim=-1)


def train_restructured_thought_wilderness():
    print("=" * 85)
    print("RESTRUCTURING THE THOUGHT WILDERNESS (思考原野の幾何学的再構成)")
    print(f"Base Model: {MODEL_ID} on {device}")
    print(f"Output LoRA: {LORA_OUTPUT_DIR}")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading Base Model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    # 1. Capture Original Topology of Preservation Corpus
    print("\n[Step 1] Recording Baseline Full-Manifold Topology...")
    base_model.eval()
    orig_reps = []
    with torch.no_grad():
        for text in TOPOLOGY_PRESERVATION_CORPUS:
            rep = extract_concept_rep(base_model, tokenizer, text).float().cpu()
            orig_reps.append(rep)
    orig_reps_tensor = torch.stack(orig_reps).to(device)
    orig_gram = torch.matmul(orig_reps_tensor, orig_reps_tensor.T)
    print(f"  Captured Gram Matrix for {len(TOPOLOGY_PRESERVATION_CORPUS)} broad conceptual domains.")

    # 2. Configure LoRA for Thought Wilderness Restructuring
    print("\n[Step 2] Configuring LoRA for Latent Field Restructuring...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    model.train()
    print("  LoRA initialized. Trainable parameters:")
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 3. Training Loop: Untangling Contradictions & Aligning Invariants
    EPOCHS = 35
    print(f"\n[Step 3] Training: Applying Invariant Potential Field ({EPOCHS} Epochs)...")
    history = []

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        repel_loss_accum = 0.0
        align_loss_accum = 0.0
        manifold_loss_accum = 0.0

        # A. Invariant Potential Field across Probes
        for probe in CONTRADICTION_PROBES:
            v_a = extract_concept_rep(model, tokenizer, probe["concept_a"])
            v_myth = extract_concept_rep(model, tokenizer, probe["concept_b_myth"])
            v_fact = extract_concept_rep(model, tokenizer, probe["concept_b_fact"])

            cos_myth = torch.dot(v_a, v_myth)
            cos_fact = torch.dot(v_a, v_fact)

            # 1. Fact Alignment: pull verified invariant close
            l_align = 1.0 - cos_fact

            # 2. Myth Repulsion: push spurious correlation/myth into lower energy
            # We want cos_myth <= cos_fact - margin (margin = 0.10)
            target_margin = 0.10
            l_repel = F.relu(cos_myth - (cos_fact - target_margin))

            loss_probe = 1.5 * l_align + 3.0 * l_repel
            total_loss = total_loss + loss_probe / len(CONTRADICTION_PROBES)
            repel_loss_accum += float(l_repel.item()) / len(CONTRADICTION_PROBES)
            align_loss_accum += float(l_align.item()) / len(CONTRADICTION_PROBES)

        # B. Manifold Topology Preservation (Prevent collapse of conceptual diversity)
        curr_reps = []
        for text in TOPOLOGY_PRESERVATION_CORPUS:
            curr_reps.append(extract_concept_rep(model, tokenizer, text).float())
        curr_reps_tensor = torch.stack(curr_reps)
        curr_gram = torch.matmul(curr_reps_tensor, curr_reps_tensor.T)

        l_manifold = F.mse_loss(curr_gram, orig_gram)
        total_loss = total_loss + 2.0 * l_manifold
        manifold_loss_accum = float(l_manifold.item())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history.append({
            "epoch": epoch,
            "total_loss": float(total_loss.item()),
            "repel_loss": repel_loss_accum,
            "align_loss": align_loss_accum,
            "manifold_loss": manifold_loss_accum
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{EPOCHS}] Loss: {total_loss.item():.4f} "
                  f"(Repel: {repel_loss_accum:.4f}, Align: {align_loss_accum:.4f}, Manifold MSE: {manifold_loss_accum:.5f})")

    elapsed = time.time() - t_start
    print(f"\nTraining completed in {elapsed:.1f}s.")

    # 4. Save Restructured LoRA Adapter
    print(f"\n[Step 4] Saving Restructured LoRA Adapter to {LORA_OUTPUT_DIR}...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print("Saved successfully.")

    # 5. Evaluate Restructured Geometry Post-Training
    print("\n[Step 5] Evaluating Post-Restructuring Thought Wilderness Geometry...")
    model.eval()
    post_results = []
    with torch.no_grad():
        for probe in CONTRADICTION_PROBES:
            v_a = extract_concept_rep(model, tokenizer, probe["concept_a"]).float()
            v_myth = extract_concept_rep(model, tokenizer, probe["concept_b_myth"]).float()
            v_fact = extract_concept_rep(model, tokenizer, probe["concept_b_fact"]).float()

            cos_myth = float(torch.dot(v_a, v_myth).item())
            cos_fact = float(torch.dot(v_a, v_fact).item())
            distortion = cos_myth - cos_fact

            post_results.append({
                "id": probe["id"],
                "category": probe["category"],
                "cos_with_myth": cos_myth,
                "cos_with_fact": cos_fact,
                "contradiction_distortion": distortion,
                "is_distorted": distortion > 0.0
            })

    distorted_count_post = sum(1 for r in post_results if r["is_distorted"])
    avg_myth_post = np.mean([r["cos_with_myth"] for r in post_results])
    avg_fact_post = np.mean([r["cos_with_fact"] for r in post_results])
    avg_distortion_post = np.mean([r["contradiction_distortion"] for r in post_results])

    print("\nPOST-RESTRUCTURING RESULTS:")
    print(f"  Distorted Probes: {distorted_count_post} / {len(post_results)} ({distorted_count_post/len(post_results)*100:.1f}%) [Was 6 / 12 (50.0%)]")
    print(f"  Mean cos(A, Myth): {avg_myth_post:.4f} [Was 0.9567]")
    print(f"  Mean cos(A, Fact): {avg_fact_post:.4f} [Was 0.9587]")
    print(f"  Mean Contradiction Distortion: {avg_distortion_post:+.4f} [Was -0.0019]")

    # Check Manifold Topology Retention
    with torch.no_grad():
        curr_reps = [extract_concept_rep(model, tokenizer, text).float() for text in TOPOLOGY_PRESERVATION_CORPUS]
        curr_gram = torch.matmul(torch.stack(curr_reps), torch.stack(curr_reps).T)
        topology_retention_corr = np.corrcoef(orig_gram.cpu().numpy().flatten(), curr_gram.cpu().numpy().flatten())[0, 1]
    print(f"  Full-Manifold Topology Correlation: {topology_retention_corr:.4f} (Ideal: > 0.95)")

    # Save Evaluation
    eval_file = os.path.join(OUTPUT_DIR, "restructured_thought_wilderness_results.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "pre_distorted_count": 6,
                "pre_distorted_ratio": 0.50,
                "post_distorted_count": distorted_count_post,
                "post_distorted_ratio": distorted_count_post / len(post_results),
                "avg_myth_post": float(avg_myth_post),
                "avg_fact_post": float(avg_fact_post),
                "avg_distortion_post": float(avg_distortion_post),
                "topology_retention_corr": float(topology_retention_corr),
                "training_time_sec": float(elapsed)
            },
            "history": history,
            "post_records": post_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation metrics to: {eval_file}")


if __name__ == "__main__":
    train_restructured_thought_wilderness()
