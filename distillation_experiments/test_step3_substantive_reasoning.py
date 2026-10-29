import os
import sys
import json
import time
from collections import defaultdict, deque

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"

print("==================================================================")
print(" STEP 3 BENCHMARK: SUBSTANTIVE REASONING (実推論エンジン)         ")
print("  - 3.1 帰納 (Induction: Pattern Aggregator & Rule Synthesis)     ")
print("  - 3.2 仮説形成 (Abduction: Backward Causal Explanation Search)  ")
print("  - 3.3 演繹 (Deduction: Graph Forward Chaining & Sub-Proof Cache)")
print(" ZERO DISCARDING - CONTINUOUS SUB-PROOF MEMOIZATION               ")
print("==================================================================")

# ----------------------------------------------------------------------
# 1. Step 3 Substantive Reasoning Engine Implementation
# ----------------------------------------------------------------------
class SubstantiveReasoningEngine:
    def __init__(self):
        # Intermediate theorem cache (Memoization)
        # Stores verified sub-proofs: { (subject, property): proof_trace }
        self.proven_lemmas = {}
        # Explored failure paths (for Step 5 Conflict Analysis)
        self.conflict_records = []

    # ------------------------------------------------------------------
    # 3.3 演繹エンジン (Deduction with Sub-Proof Cache)
    # ------------------------------------------------------------------
    def deduce(self, facts: list, rules: list, query: dict):
        """
        Executes formal forward deduction on the proposition graph.
        Caches all intermediate derived lemmas.
        """
        trace = []
        q_subj = query["subject"]
        q_pred = query["predicate"]
        q_neg = query.get("is_negative", False)
        
        # Check cache first (O(1) lookup)
        cache_key = (q_subj.lower(), q_pred.lower())
        if cache_key in self.proven_lemmas:
            cached = self.proven_lemmas[cache_key]
            trace.append(f"[CACHE HIT] Lemma already proven: {q_subj} is {q_pred}")
            return {
                "decision": (not cached["is_negative"]) if not q_neg else cached["is_negative"],
                "proof_chain": cached["proof_chain"],
                "is_cached": True,
                "reasoning_trace": trace,
                "status": "PROVEN_FROM_CACHE"
            }
            
        # Build Forward Rule Graph
        forward_graph = defaultdict(list)
        all_predicates = set()
        for r in rules:
            forward_graph[r["antecedent"]].append((r["consequent"], r["is_negative"], r["raw"]))
            all_predicates.add(r["antecedent"])
            all_predicates.add(r["consequent"])
            
        subj_facts = [f for f in facts if f["subject"].lower() == q_subj.lower()]
        if not subj_facts:
            return {
                "decision": None,
                "proof_chain": [],
                "status": "OPEN_GAP",
                "reasoning_trace": [f"No facts known about {q_subj}"]
            }
            
        start_concept = subj_facts[0]["predicate"]
        trace.append(f"Root Fact: {q_subj} is a {start_concept}")
        
        queue = deque([(start_concept, [f"{q_subj} is a {start_concept}"], False)])
        visited = set([start_concept])
        
        proven_proof = None
        proven_negated = False
        all_reached = set([start_concept])
        
        while queue:
            curr, path, curr_neg = queue.popleft()
            
            # Cache every reached node as a valid lemma!
            self.proven_lemmas[(q_subj.lower(), curr.lower())] = {
                "is_negative": curr_neg,
                "proof_chain": path
            }
            all_reached.add(curr)
            
            if curr == q_pred:
                proven_proof = path
                proven_negated = curr_neg
                break
                
            for nxt, is_neg, raw_rule in forward_graph.get(curr, []):
                new_path = path + [raw_rule]
                new_neg = (curr_neg != is_neg)
                
                if nxt == q_pred:
                    proven_proof = new_path
                    proven_negated = new_neg
                    # Cache target
                    self.proven_lemmas[(q_subj.lower(), nxt.lower())] = {
                        "is_negative": new_neg,
                        "proof_chain": new_path
                    }
                    queue.clear()
                    break
                    
                if nxt not in visited and not is_neg:
                    visited.add(nxt)
                    queue.append((nxt, new_path, new_neg))
                    
        if proven_proof is not None:
            trace.append(f"Formal proof completed in {len(proven_proof)} steps.")
            decision = (not proven_negated) if not q_neg else proven_negated
            return {
                "decision": decision,
                "proof_chain": proven_proof,
                "status": "PROVEN",
                "is_cached": False,
                "reasoning_trace": trace
            }
            
        # If target exists in rules but unreachable
        if q_pred in all_predicates:
            trace.append(f"Target '{q_pred}' exists in rules, but is unreachable from {start_concept}.")
            self.conflict_records.append({
                "type": "UNREACHABLE_GOAL",
                "subject": q_subj,
                "target": q_pred,
                "reached_concepts": list(all_reached)
            })
            return {
                "decision": False if not q_neg else True,
                "proof_chain": [],
                "status": "CLOSED_WORLD_REFUTED",
                "is_cached": False,
                "reasoning_trace": trace
            }
        else:
            # Target is completely absent (Enthymeme Gap)
            return {
                "decision": None,
                "proof_chain": [],
                "status": "OPEN_GAP",
                "reached_concepts": list(all_reached),
                "reasoning_trace": trace + [f"Predicate '{q_pred}' absent from context."]
            }

    # ------------------------------------------------------------------
    # 3.2 仮説形成エンジン (Abduction: Backward Causal Explanation)
    # ------------------------------------------------------------------
    def abduce_explanation(self, observation: dict, rules: list):
        """
        Given an observation (e.g. 'Target is red'), performs backward search
        to identify minimal candidate initial premises that would explain it.
        """
        target_pred = observation["predicate"]
        
        # Build Backward Graph
        backward_graph = defaultdict(list)
        for r in rules:
            if not r["is_negative"]:
                backward_graph[r["consequent"]].append(r["antecedent"])
                
        candidate_causes = []
        queue = deque([(target_pred, [target_pred])])
        visited = set([target_pred])
        
        while queue:
            curr, path = queue.popleft()
            parents = backward_graph.get(curr, [])
            
            if not parents and curr != target_pred:
                candidate_causes.append({
                    "root_hypothesis": curr,
                    "causal_path": list(reversed(path))
                })
                
            for p in parents:
                if p not in visited:
                    visited.add(p)
                    queue.append((p, path + [p]))
                    
        return {
            "observation": observation,
            "hypotheses_count": len(candidate_causes),
            "candidate_hypotheses": candidate_causes
        }

    # ------------------------------------------------------------------
    # 3.1 帰納エンジン (Induction: Pattern Aggregator & Rule Synthesis)
    # ------------------------------------------------------------------
    def induce_general_rules(self, instances: list):
        """
        Given multiple instance facts:
          e.g. [(Tweety, bird), (Tweety, wings)], [(Robin, bird), (Robin, wings)], ...
        Extracts co-occurrence patterns and synthesizes generalized universal rules.
        """
        subject_properties = defaultdict(set)
        for inst in instances:
            s = inst["subject"]
            p = inst["predicate"]
            subject_properties[s].add(p)
            
        pair_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for s, props in subject_properties.items():
            for p1 in props:
                category_counts[p1] += 1
                for p2 in props:
                    if p1 != p2:
                        pair_counts[(p1, p2)] += 1
                        
        synthesized_rules = []
        total_subjects = len(subject_properties)
        
        for (cat, prop), count in pair_counts.items():
            # If property appears in >= 80% of category instances
            support = count / category_counts[cat]
            if support >= 0.8 and count >= 2:
                synthesized_rules.append({
                    "antecedent": cat,
                    "consequent": prop,
                    "support": support,
                    "instances_count": count,
                    "synthesized_rule": f"All {cat}s are {prop}s"
                })
                
        return synthesized_rules

# ----------------------------------------------------------------------
# 2. Test Suite for Step 3
# ----------------------------------------------------------------------
rules_prontoqa = [
    {"antecedent": "yumpus", "consequent": "dumpus", "is_negative": False, "raw": "Each yumpus is a dumpus"},
    {"antecedent": "dumpus", "consequent": "wumpus", "is_negative": False, "raw": "Dumpuses are wumpuses"},
    {"antecedent": "wumpus", "consequent": "red", "is_negative": False, "raw": "Wumpuses are red"},
    {"antecedent": "impus", "consequent": "sour", "is_negative": False, "raw": "Each impus is sour"}
]

facts_max = [
    {"subject": "Max", "predicate": "yumpus", "is_negative": False, "raw": "Max is a yumpus"}
]

induction_instances = [
    {"subject": "Tweety", "predicate": "bird"}, {"subject": "Tweety", "predicate": "feather"},
    {"subject": "Robin", "predicate": "bird"},  {"subject": "Robin", "predicate": "feather"},
    {"subject": "Eagle", "predicate": "bird"},  {"subject": "Eagle", "predicate": "feather"},
    {"subject": "Shark", "predicate": "fish"},  {"subject": "Shark", "predicate": "gill"},
    {"subject": "Tuna", "predicate": "fish"},   {"subject": "Tuna", "predicate": "gill"}
]

# ----------------------------------------------------------------------
# 3. Execution & Objective Verification
# ----------------------------------------------------------------------
print("\n--- [Test 3.3: Deduction & Lemma Cache Reuse] ---")
engine = SubstantiveReasoningEngine()

# Query 1: Is Max red? (Must deduce from scratch)
t0 = time.time()
res1 = engine.deduce(facts_max, rules_prontoqa, {"subject": "Max", "predicate": "red"})
t1 = time.time() - t0
print(f"Query 1 (Max is red): {res1['status']} in {t1*1000:.3f}ms | Cached={res1['is_cached']}")
print(f"  Proof: {' -> '.join(res1['proof_chain'])}")

# Query 2: Is Max a dumpus? (Must hit cache!)
t0 = time.time()
res2 = engine.deduce(facts_max, rules_prontoqa, {"subject": "Max", "predicate": "dumpus"})
t2 = time.time() - t0
print(f"Query 2 (Max is dumpus): {res2['status']} in {t2*1000:.3f}ms | Cached={res2['is_cached']}")

# Query 3: Is Max sour? (Must recognize unreachable closed-world)
t0 = time.time()
res3 = engine.deduce(facts_max, rules_prontoqa, {"subject": "Max", "predicate": "sour"})
t3 = time.time() - t0
print(f"Query 3 (Max is sour): {res3['status']} in {t3*1000:.3f}ms | Decision={res3['decision']}")

print("\n--- [Test 3.2: Abduction (Backward Hypothesis Formation)] ---")
# Observation: Max is red. What could explain this?
obs = {"subject": "Max", "predicate": "red"}
abduct_res = engine.abduce_explanation(obs, rules_prontoqa)
print(f"Observation: {obs}")
print(f"Generated Explanatory Hypotheses ({abduct_res['hypotheses_count']}):")
for h in abduct_res["candidate_hypotheses"]:
    print(f"  * Root Hypothesis: [{h['root_hypothesis']}] -> Path: {' -> '.join(h['causal_path'])}")

print("\n--- [Test 3.1: Induction (Generalization from Instances)] ---")
induce_res = engine.induce_general_rules(induction_instances)
print(f"Observed {len(induction_instances)} instances across {len(set(i['subject'] for i in induction_instances))} subjects.")
print(f"Synthesized Universal Rules ({len(induce_res)}):")
for r in induce_res:
    print(f"  * {r['synthesized_rule']} (Support: {r['support']*100:.1f}%, Count: {r['instances_count']})")

# Verification Assertions
pass_deduce = (res1["decision"] is True and res2["is_cached"] is True and res3["decision"] is False)
pass_abduce = (abduct_res["hypotheses_count"] > 0 and abduct_res["candidate_hypotheses"][0]["root_hypothesis"] == "yumpus")
pass_induce = (len(induce_res) >= 2 and any(r["consequent"] == "feather" for r in induce_res))

all_step3_passed = pass_deduce and pass_abduce and pass_induce

print("\n==================================================================")
print(f" STEP 3 VERIFICATION RESULT: {'ALL PASS (100.0%)' if all_step3_passed else 'FAIL'}")
print(f"  - 3.3 Deduction & Lemma Cache: {'PASS' if pass_deduce else 'FAIL'}")
print(f"  - 3.2 Abduction (Backward Explanations): {'PASS' if pass_abduce else 'FAIL'}")
print(f"  - 3.1 Induction (Pattern Synthesis): {'PASS' if pass_induce else 'FAIL'}")
print(f"  - Total Proven Lemmas Cached: {len(engine.proven_lemmas)}")
print("==================================================================")

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "step3_reasoning_benchmark_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "all_passed": all_step3_passed,
        "deduction_results": [res1, res2, res3],
        "abduction_results": abduct_res,
        "induction_results": induce_res,
        "cached_lemmas_count": len(engine.proven_lemmas)
    }, f, indent=2, ensure_ascii=False)
print(f"Saved results to {json_path}")
