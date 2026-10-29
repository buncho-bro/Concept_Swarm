import os
import sys
import json
import time
from collections import defaultdict, deque

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"

print("==================================================================")
print(" STEP 5 BENCHMARK: META-COGNITION & FAILURE LEARNING              ")
print("  - Sub-module 5A: Conflict Core Extraction (根本原因の特定)      ")
print("  - Sub-module 5B: No-Good Constraint Generation (禁じ手の獲得)   ")
print("  - Sub-module 5C: Proof by Contradiction (背理法による新事実化)  ")
print("  - Sub-module 5D: Repulsive Energy Field (変更能への斥力刻印)    ")
print("==================================================================")

# ----------------------------------------------------------------------
# 1. Step 5 Implementation: Meta-Cognition & Failure Learner Engine
# ----------------------------------------------------------------------
class MetaCognitionFailureLearner:
    def __init__(self):
        # Accumulated No-Good Constraints (Forbidden combinations)
        self.nogood_constraints = set()
        # Derived Facts via Proof by Contradiction
        self.contradiction_facts = {}
        # Repulsive Energy Penalties on Concept Pairs: { (word_a, word_b): -penalty }
        self.repulsive_field = defaultdict(float)
        # Reusable Sub-Proof Cache
        self.preserved_subproofs = {}

    def analyze_conflict(self, hypothesis: str, path: list, conflict_reason: str):
        """
        Sub-module 5A & 5B: Conflict Analysis & No-Good Generation.
        When a branch fails, extracts the minimal conflicting core and registers it as a No-Good.
        """
        # Minimal Conflict Core is the hypothesis combined with the terminal conflicting node
        conflict_core = (hypothesis, path[-1])
        
        # 1. Register No-Good Constraint
        self.nogood_constraints.add(conflict_core)
        
        # 2. Sub-module 5C: Proof by Contradiction
        # Since advancing hypothesis H resulted in a contradiction, ~H is established as TRUE!
        negated_fact = f"NOT_{hypothesis}"
        self.contradiction_facts[hypothesis] = {
            "established_fact": negated_fact,
            "refuted_path": path,
            "conflict_reason": conflict_reason
        }
        
        # 3. Sub-module 5D: Repulsive Energy Field
        # Imprint negative penalty (-2.5) on the conflicting pair so analogy engines steer away
        self.repulsive_field[conflict_core] -= 2.5
        self.repulsive_field[(path[-1], hypothesis)] -= 2.5
        
        # 4. Preserve intermediate non-conflicting nodes (Never discard partial work!)
        for i in range(len(path) - 1):
            valid_lemma = (path[i], path[i+1])
            self.preserved_subproofs[valid_lemma] = path[:i+2]
            
        return {
            "conflict_core": conflict_core,
            "nogood_added": True,
            "contradiction_derived": negated_fact,
            "repulsive_penalty": -2.5,
            "preserved_lemmas_count": len(self.preserved_subproofs)
        }

    def is_path_forbidden_by_nogood(self, current_node: str, target_node: str):
        """
        Checks whether expanding to target_node violates any learned No-Good constraint.
        """
        return (current_node, target_node) in self.nogood_constraints

    def get_adjusted_influence(self, word_a: str, word_b: str, raw_affinity: float):
        """
        Adjusts raw semantic influence by learned repulsive penalties from past failures.
        """
        penalty = self.repulsive_field.get((word_a, word_b), 0.0)
        return raw_affinity + penalty

# ----------------------------------------------------------------------
# 2. Test Execution: Conflict Learning in Action
# ----------------------------------------------------------------------
print("\n--- [Scenario: Hypothesis Leads to Contradiction] ---")
learner = MetaCognitionFailureLearner()

# Scenario:
# We explore hypothesis: "Max is an Impus"
# Reasoning path: Max -> Impus -> Sour -> (Contradiction: Problem premises state Max is NOT sour!)
path_to_failure = ["Max", "Impus", "Sour"]
conflict_reason = "Contradiction with explicit premise (Max is not sour)"

print(f"Triggering failure on explored path: {' -> '.join(path_to_failure)}")
print(f"Conflict Reason: {conflict_reason}")

t0 = time.time()
learning_report = learner.analyze_conflict("Impus", path_to_failure, conflict_reason)
elapsed = (time.time() - t0) * 1000.0

print(f"\n[Learning Completed in {elapsed:.3f}ms]")
print(f"  * Conflict Core Extracted: {learning_report['conflict_core']}")
print(f"  * No-Good Constraint Registered: {learning_report['nogood_added']}")
print(f"  * Proof by Contradiction Fact: {learning_report['contradiction_derived']} is TRUE!")
print(f"  * Repulsive Penalty Imprinted: {learning_report['repulsive_penalty']}")
print(f"  * Preserved Intermediate Lemmas: {learning_report['preserved_lemmas_count']}")

# Test 1: Future Search Avoids No-Good
print("\n--- [Verification 1: Instant Pruning via Learned No-Good] ---")
is_blocked = learner.is_path_forbidden_by_nogood("Impus", "Sour")
print(f"Attempting to explore ('Impus' -> 'Sour'): Blocked by No-Good = {is_blocked} (Instant O(1) Pruning!)")

# Test 2: Proof by Contradiction Fact Retrieval
print("\n--- [Verification 2: Proof by Contradiction Fact] ---")
has_derived_fact = ("Impus" in learner.contradiction_facts)
print(f"New Knowledge Base Fact: Max is {learner.contradiction_facts['Impus']['established_fact']} (Proved = {has_derived_fact})")

# Test 3: Repulsive Energy Field Shifts Influence
print("\n--- [Verification 3: Repulsive Potential Field Shift] ---")
raw_affinity = +0.80  # Hypothetical raw intuition affinity
adjusted_affinity = learner.get_adjusted_influence("Impus", "Sour", raw_affinity)
print(f"Raw Affinity ('Impus', 'Sour'):      {raw_affinity:+.2f}")
print(f"Adjusted Affinity after Repulsion:   {adjusted_affinity:+.2f} (Self-Correction: Shifted from Positive to Negative!)")

# Test 4: Preserved Subproofs are Intact
print("\n--- [Verification 4: Intermediate Sub-Proof Recycling] ---")
print(f"Preserved Lemmas in Memory: {list(learner.preserved_subproofs.keys())}")

all_step5_passed = (
    is_blocked is True and 
    has_derived_fact is True and 
    adjusted_affinity < 0.0 and 
    len(learner.preserved_subproofs) > 0
)

print("\n==================================================================")
print(f" STEP 5 VERIFICATION RESULT: {'ALL PASS (100.0%)' if all_step5_passed else 'FAIL'}")
print(f"  - No-Good Pruning: {'PASS' if is_blocked else 'FAIL'}")
print(f"  - Proof by Contradiction: {'PASS' if has_derived_fact else 'FAIL'}")
print(f"  - Repulsive Field Self-Correction: {'PASS' if adjusted_affinity < 0.0 else 'FAIL'}")
print(f"  - Partial Work Preservation: {'PASS' if len(learner.preserved_subproofs) > 0 else 'FAIL'}")
print("==================================================================")

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "step5_failure_learning_benchmark_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "all_passed": all_step5_passed,
        "learning_report": learning_report,
        "is_blocked": is_blocked,
        "adjusted_affinity": adjusted_affinity,
        "preserved_lemmas": [str(k) for k in learner.preserved_subproofs.keys()]
    }, f, indent=2, ensure_ascii=False)
print(f"Saved results to {json_path}")
