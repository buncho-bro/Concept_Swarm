import os
import sys
import json
import time
import math
import torch
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
print(" PURE PROBABILISTIC INFLUENCE PROPAGATION EXPERIMENT             ")
print(" (Testing 100/0 Convergence vs Flat 'Unknown' Self-Recognition)   ")
print("==================================================================")
print(f"Device: {device}")

print("Loading Qwen2.5-1.5B-Instruct for Neural Log-Odds Calibration...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# Neutral baseline
neutral_prompt = "Something is"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    neutral_logits = model(**inputs).logits[0, -1, :]
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1)

def get_calibrated_influence_delta(source_word: str, target_word: str):
    """
    Computes calibrated log-odds ratio from model representation.
    """
    prompt = f"The {source_word} is"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    target_tok = tokenizer.encode(" " + target_word, add_special_tokens=False)
    if not target_tok:
        return 2.5 # Default moderate logical strength
    tid = target_tok[0]
    with torch.no_grad():
        lp = F.log_softmax(model(**inp).logits[0, -1, :], dim=-1)
    raw_delta = lp[tid].item() - neutral_log_probs[tid].item()
    # Bound logical influence to prevent numerical blowup
    return float(np.clip(raw_delta, -4.0, 4.0))

class ProbabilisticReasoningEngine:
    """
    Propagates continuous probability beliefs across logical relations.
    No discrete 0/1 thresholds. Evaluates convergence towards 100/0 or 50/50.
    """
    def __init__(self, entities, target_query):
        self.entities = entities # List of entity names
        self.target = target_query # e.g. "sour"
        # Belief state: Log-Odds for each entity
        # Log-Odds = 0.0 corresponds to Probability = 0.5 (Uncertain / Unknown)
        self.log_odds = {e: 0.0 for e in entities}
        self.history = []

    def set_initial_fact(self, entity_name, initial_prob=0.99):
        # Convert probability to log-odds: log(p / (1 - p))
        self.log_odds[entity_name] = math.log(initial_prob / (1.0 - initial_prob))

    def get_probabilities(self):
        # Convert log-odds to probabilities: sigmoid(log_odds)
        probs = {}
        for e, lo in self.log_odds.items():
            probs[e] = 1.0 / (1.0 + math.exp(-lo))
        return probs

    def propagate_step(self, rules, beta=1.2):
        """
        Propagates belief across rules in parallel:
          rule: (source, target, polarity)
          Updates log_odds(target) based on P(source) * influence
        """
        current_probs = self.get_probabilities()
        delta_updates = {e: 0.0 for e in self.entities}

        for src, tgt, pol in rules:
            if src in current_probs and tgt in self.entities:
                p_src = current_probs[src]
                # Influence weight: positive rule reinforces tgt; negative rule destroys tgt
                direction = 1.0 if pol else -1.0
                weight = get_calibrated_influence_delta(src, tgt)
                
                # Update contribution proportional to confidence in source P(src)
                if p_src > 0.5:
                    contribution = direction * (p_src - 0.5) * 2.0 * abs(weight) * beta
                    delta_updates[tgt] += contribution

        # Apply updates
        for e in self.entities:
            self.log_odds[e] += delta_updates[e]
            
        self.history.append(self.get_probabilities())

# ----------------------------------------------------------------------
# Define Test Cases (Full Proof vs Incomplete Proof vs Contradiction)
# ----------------------------------------------------------------------
# Base Ontology:
# Max is a yumpus. yumpus -> dumpus -> wumpus -> impus -> tumpus -> (not sour)
entities_all = ["yumpus", "dumpus", "wumpus", "impus", "tumpus", "sour", "aggressive", "shy"]

# Case 1: Standard ProntoQA (Complete Deductive Chain to "NOT sour")
rules_case1 = [
    ("yumpus", "dumpus", True),
    ("dumpus", "wumpus", True),
    ("wumpus", "impus", True),
    ("impus", "tumpus", True),
    ("tumpus", "sour", False), # Negative rule!
    ("yumpus", "aggressive", True) # Distractor
]

# Case 2: Broken Chain (Missing Link: dumpus -> wumpus is removed!)
# The truth of "sour" cannot be logically determined.
rules_case2 = [
    ("yumpus", "dumpus", True),
    # MISSING: ("dumpus", "wumpus", True),
    ("wumpus", "impus", True),
    ("impus", "tumpus", True),
    ("tumpus", "sour", False),
    ("yumpus", "aggressive", True)
]

# Case 3: Contradictory Rules (Both positive and negative paths exist)
rules_case3 = [
    ("yumpus", "dumpus", True),
    ("dumpus", "tumpus", True),
    ("tumpus", "sour", False), # Path says NOT sour
    ("yumpus", "impus", True),
    ("impus", "sour", True)    # Path says sour!
]

def run_simulation(name, rules, num_steps=6):
    print(f"\n--- Running Simulation: {name} ---")
    engine = ProbabilisticReasoningEngine(entities_all, "sour")
    engine.set_initial_fact("yumpus", initial_prob=0.99)
    engine.history.append(engine.get_probabilities())
    
    for step in range(1, num_steps + 1):
        engine.propagate_step(rules)
        p_target = engine.get_probabilities()["sour"]
        print(f"  Step {step}: Target 'sour' Probability = {p_target*100:6.2f}% (Log-Odds: {engine.log_odds['sour']:+6.2f})")
        
    final_p = engine.get_probabilities()["sour"]
    
    # Verdict Analysis
    if final_p < 0.05:
        verdict = f"CERTAINLY FALSE (Confidence: {(1-final_p)*100:.1f}%) -> 100/0 Convergence!"
    elif final_p > 0.95:
        verdict = f"CERTAINLY TRUE (Confidence: {final_p*100:.1f}%) -> 100/0 Convergence!"
    else:
        verdict = f"UNKNOWN / CANNOT DETERMINE (P = {final_p*100:.1f}%) -> Successfully recognized lack of proof!"
        
    print(f"  --> Final Verdict: {verdict}")
    return engine.history

hist_1 = run_simulation("Case 1: Complete Logical Chain", rules_case1)
hist_2 = run_simulation("Case 2: Incomplete / Missing Link", rules_case2)
hist_3 = run_simulation("Case 3: Mutually Contradictory Premise", rules_case3)

# ----------------------------------------------------------------------
# Visualizing Probability Curves (Convergence vs Flat Uncertainty)
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

def plot_history(ax, history, title):
    steps = range(len(history))
    sour_probs = [h["sour"] * 100 for h in history]
    yumpus_probs = [h["yumpus"] * 100 for h in history]
    tumpus_probs = [h["tumpus"] * 100 for h in history]
    
    ax.plot(steps, sour_probs, label="Target: 'sour'", color='#e74c3c', linewidth=2.5, marker='o')
    ax.plot(steps, yumpus_probs, label="Origin: 'yumpus'", color='#3498db', linestyle='--', alpha=0.7)
    ax.plot(steps, tumpus_probs, label="Bridge: 'tumpus'", color='#2ecc71', linestyle=':', alpha=0.7)
    
    ax.set_xlabel("Propagation Step", fontsize=10, fontweight='bold')
    ax.set_ylabel("Belief Probability (%)", fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Uncertain (50%)')
    ax.axhline(95, color='green', linestyle=':', alpha=0.3)
    ax.axhline(5, color='green', linestyle=':', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='center right', fontsize=9)

plot_history(ax1, hist_1, "Case 1: Complete Chain\n(Sharp 100/0 Convergence)")
plot_history(ax2, hist_2, "Case 2: Missing Link\n(Stays Flat ~50% 'Unknown')")
plot_history(ax3, hist_3, "Case 3: Contradictory Rules\n(Clashes into Ambiguity)")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "probabilistic_propagation_curves.png")
plt.savefig(plot_path, dpi=300)
print(f"\nSaved plot to {plot_path}")

artifact_plot_path = os.path.join(ARTIFACT_DIR, "probabilistic_propagation_curves.png")
import shutil
shutil.copyfile(plot_path, artifact_plot_path)
print(f"Copied to artifact dir: {artifact_plot_path}")

print("\n=== Experiment Successfully Completed! ===")
