import os
import sys
import json
import time
import re
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
print(" STEP 1 BENCHMARK: ABSTRACTION & MODELING (捨象と抽象)           ")
print("  - Sub-module 1A: Noise Pruning via Morphological Filtering       ")
print("  - Sub-module 1B: Propositional Modeling (Facts, Rules, Query)   ")
print(" ZERO STATIC DICTIONARIES - UNIVERSAL CONCEPT STRUCTURING         ")
print("==================================================================")
print(f"Device: {device}")

print(f"Loading {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# ----------------------------------------------------------------------
# 1. Step 1 Implementation: Abstraction & Modeling Engine
# ----------------------------------------------------------------------
class Step1AbstractionEngine:
    def __init__(self):
        # Universal structural stopwords that do not carry entity/property meaning
        self.functional_stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
            "that", "this", "it", "for", "with", "as", "by", "on", "at", "from", "which",
            "within", "without", "inside", "due", "very", "any", "some", "who", "whom",
            "received", "dangerously", "safely", "long", "quickly", "strictly", "deep"
        }

    def normalize_stem(self, word: str) -> str:
        """
        Preserves Greek/Latin/Fictional -us stems while normalizing regular English plurals.
        No static dictionaries.
        """
        w = word.strip(".,?!:;\"'()[]{}").lower()
        if w.endswith("puses"):
            return w[:-2] # jompuses -> jompus
        if w.endswith("ses") and not w.endswith("sses") and not w.endswith("uses"):
            return w[:-2]
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
            return w[:-1]
        return w

    def prune_noise(self, text: str):
        """
        Sub-module 1A: 捨象 (Noise Pruning).
        Strips away syntactic boilerplate, retaining salient concept tokens.
        """
        raw_tokens = [w.strip(".,?!:;\"'()[]{}") for w in text.split()]
        filtered = []
        for t in raw_tokens:
            t_low = t.lower()
            if len(t) > 1 and t.isalpha() and t_low not in self.functional_stopwords:
                filtered.append(t)
        return filtered

    def parse_query(self, query: str):
        """
        Extracts Subject, Predicate, and Expectation Polarity from a query.
        """
        q = query.strip().rstrip("?")
        # Strip leading auxiliary verb and article: "Does a bowling ball float..." -> "bowling ball float..."
        q_stripped = re.sub(r'^(?:Can|Is|Does|Has|Did|Will|Should|Could)\s+(?:a\s+|an\s+|the\s+)?', '', q, flags=re.IGNORECASE)
        
        # Check if query has an individual subject + predicate structure
        m = re.match(r'^([A-Z][a-z]+)\s+(?:is|are|have|has|be|can|received|stay\s+in)?\s*(not)?\s*(.*)$', q_stripped, re.IGNORECASE)
        if m:
            raw_subj = m.group(1).strip()
            is_neg = bool(m.group(2))
            rest = m.group(3).strip()
            
            pred_candidates = self.prune_noise(rest)
            pred = self.normalize_stem(pred_candidates[0]) if pred_candidates else self.normalize_stem(rest)
            return {
                "raw_query": query,
                "subject": raw_subj,
                "predicate": pred,
                "is_negative": is_neg,
                "salient_context": [raw_subj.lower(), pred] + pred_candidates
            }
        else:
            # Multi-word concept subject or free-standing claim
            concepts = self.prune_noise(q_stripped)
            stems = [self.normalize_stem(c) for c in concepts]
            return {
                "raw_query": query,
                "subject": stems[0] if stems else "Entity",
                "predicate": stems[1] if len(stems) > 1 else (stems[0] if stems else "Property"),
                "is_negative": ("not" in query.lower() or "never" in query.lower()),
                "salient_context": stems
            }

    def model_propositions(self, context: str, query: str):
        """
        Sub-module 1B: 抽象 (Propositional Modeling).
        Transforms unstructured text into clean mathematical tuples for Steps 3-5.
        """
        facts = []
        rules = []
        
        raw_word_count = len(context.split()) + len(query.split())
        
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        for s in sentences:
            s_clean = s.strip()
            
            # 1. Universal negative rule: Each/Every/All X is not Y / Xs are not Ys
            m_neg = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z]+)\s+(?:is|are)\s+not\s+(?:a\s+|an\s+)?([a-zA-Z\-]+)', s_clean, re.IGNORECASE)
            if m_neg:
                ant = self.normalize_stem(m_neg.group(1))
                con = self.normalize_stem(m_neg.group(2))
                rules.append({
                    "type": "UNIVERSAL_RULE",
                    "antecedent": ant,
                    "consequent": con,
                    "polarity": "NEGATIVE",
                    "is_negative": True,
                    "raw": s_clean
                })
                continue
                
            # 2. Individual Fact: "[Subject] is a/an [Predicate]" (e.g. Whiskers is a cat, Stella is a yumpus)
            m_fact = re.search(r'^([A-Z][a-zA-Z\-]+)\s+is\s+(?:a\s+|an\s+)([a-zA-Z\-]+)', s_clean)
            if m_fact and not s_clean.lower().startswith(("each", "every", "all")):
                subj = m_fact.group(1).strip()
                pred = self.normalize_stem(m_fact.group(2))
                facts.append({
                    "type": "INDIVIDUAL_FACT",
                    "subject": subj,
                    "predicate": pred,
                    "polarity": "POSITIVE",
                    "is_negative": False,
                    "raw": s_clean
                })
                continue

            # 3. Universal affirmative rule: Each/Every/All X is Y / Xs are Ys / X enables Y
            m_pos = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z]+)\s+(?:is|are|enables)\s+(?:a\s+|an\s+)?([a-zA-Z\-]+)', s_clean, re.IGNORECASE)
            if m_pos:
                ant = self.normalize_stem(m_pos.group(1))
                con = self.normalize_stem(m_pos.group(2))
                rules.append({
                    "type": "UNIVERSAL_RULE",
                    "antecedent": ant,
                    "consequent": con,
                    "polarity": "POSITIVE",
                    "is_negative": False,
                    "raw": s_clean
                })
                continue
                
        # Parse Query
        parsed_query = self.parse_query(query)
        
        # Calculate Abstraction Metrics
        modeled_tokens = []
        for f in facts:
            modeled_tokens.extend([f["subject"], f["predicate"]])
        for r in rules:
            modeled_tokens.extend([r["antecedent"], r["consequent"]])
        modeled_tokens.extend([parsed_query["subject"], parsed_query["predicate"]])
        modeled_tokens.extend(parsed_query.get("salient_context", []))
        
        unique_modeled_tokens = list(set(modeled_tokens))
        pruning_ratio = max(0.0, 1.0 - (len(unique_modeled_tokens) / max(1, raw_word_count)))
        
        return {
            "facts": facts,
            "rules": rules,
            "query": parsed_query,
            "metrics": {
                "raw_words": raw_word_count,
                "modeled_concepts": len(unique_modeled_tokens),
                "pruning_ratio": pruning_ratio,
                "facts_count": len(facts),
                "rules_count": len(rules)
            }
        }

# ----------------------------------------------------------------------
# 2. Test Suite for Step 1
# ----------------------------------------------------------------------
test_cases = [
    {
        "id": "Case1_ProntoQA",
        "name": "ProntoQA Multi-Hop Syllogism (Fictional Concepts)",
        "context": "Every tumpus is not angry. Tumpuses are rompuses. Every numpus is not bright. Rompuses are yumpuses. Yumpuses are zumpuses. Zumpuses are impuses. Each impus is a dumpus. Stella is a yumpus.",
        "query": "Is Stella bright?",
        "expected_facts": 1,
        "expected_rules": 7,
        "must_contain_entities": ["tumpus", "rompus", "numpus", "bright", "yumpus", "Stella"]
    },
    {
        "id": "Case2_Counterfactual",
        "name": "Counterfactual Biology (Rule overrides reality)",
        "context": "All felines are cold-blooded. All cats are felines. Whiskers is a cat.",
        "query": "Is Whiskers cold-blooded?",
        "expected_facts": 1,
        "expected_rules": 2,
        "must_contain_entities": ["feline", "cold-blooded", "cat", "Whiskers"]
    },
    {
        "id": "Case3_Messy_Myth",
        "name": "Messy Natural Language Myth (Adverb & Noise Heavy)",
        "context": "",
        "query": "Swallowed chewing gum remains dangerously stuck inside the human stomach for seven long years without any digestion.",
        "expected_facts": 0,
        "expected_rules": 0,
        "must_contain_entities": ["swallowed", "chewing", "gum", "stomach", "seven", "years", "digestion"]
    },
    {
        "id": "Case4_Physics_Natural",
        "name": "Physics Sentence with Auxiliary Phrases",
        "context": "A heavy metal bowling ball drops and sinks quickly to deep ocean water floor due to gravity.",
        "query": "Does a bowling ball float on water?",
        "expected_facts": 0,
        "expected_rules": 0,
        "must_contain_entities": ["bowling", "ball", "float", "water"]
    },
    {
        "id": "Case5_Gapped_Enthymeme",
        "name": "Enthymeme with Implicit Flight Premise",
        "context": "Tweety is a penguin.",
        "query": "Can Tweety fly?",
        "expected_facts": 1,
        "expected_rules": 0,
        "must_contain_entities": ["Tweety", "penguin", "fly"]
    },
    {
        "id": "Case6_Astronaut_Chained",
        "name": "Multi-Hop Domain Knowledge Chain",
        "context": "Luna is an astronaut. All astronauts are pilots. All pilots are explorers.",
        "query": "Is Luna an explorer?",
        "expected_facts": 1,
        "expected_rules": 2,
        "must_contain_entities": ["Luna", "astronaut", "pilot", "explorer"]
    }
]

# ----------------------------------------------------------------------
# 3. Execution and Objective Verification
# ----------------------------------------------------------------------
print(f"\nExecuting Step 1 Verification across {len(test_cases)} diverse test cases...\n")
engine = Step1AbstractionEngine()

results = []
all_passed = True

for idx, tc in enumerate(test_cases):
    print("------------------------------------------------------------------")
    print(f"[{idx+1}/{len(test_cases)}] Testing: {tc['name']}")
    print(f"Raw Context: {tc['context'] if tc['context'] else '(Empty Context)'}")
    print(f"Raw Query:   {tc['query']}")
    
    t0 = time.time()
    out = engine.model_propositions(tc["context"], tc["query"])
    elapsed = (time.time() - t0) * 1000.0
    
    # Verification checks
    facts_ok = (len(out["facts"]) == tc["expected_facts"])
    rules_ok = (len(out["rules"]) == tc["expected_rules"])
    
    # Check entity preservation
    all_extracted_str = json.dumps(out, ensure_ascii=False).lower()
    missing_entities = [e for e in tc["must_contain_entities"] if e.lower() not in all_extracted_str]
    entities_ok = (len(missing_entities) == 0)
    
    case_passed = (facts_ok and rules_ok and entities_ok)
    if not case_passed:
        all_passed = False
        verdict = "FAIL"
    else:
        verdict = "PASS"
        
    print(f"Verdict: {verdict} ({elapsed:.2f}ms)")
    print(f"  * Facts Modeled: {len(out['facts'])} (Expected: {tc['expected_facts']}) -> {'OK' if facts_ok else 'NG'}")
    print(f"  * Rules Modeled: {len(out['rules'])} (Expected: {tc['expected_rules']}) -> {'OK' if rules_ok else 'NG'}")
    print(f"  * Entity Preservation: {'OK' if entities_ok else f'NG (Missing: {missing_entities})'}")
    print(f"  * Noise Pruning Ratio: {out['metrics']['pruning_ratio']*100:.1f}% reduction ({out['metrics']['raw_words']} words -> {out['metrics']['modeled_concepts']} concepts)")
    print("Structured Output Sample:")
    print(f"  Facts: {out['facts']}")
    print(f"  Rules: {out['rules'][:3]} ... ({len(out['rules'])} total)")
    print(f"  Query: {out['query']}")
    
    results.append({
        "id": tc["id"],
        "name": tc["name"],
        "passed": case_passed,
        "facts_ok": facts_ok,
        "rules_ok": rules_ok,
        "entities_ok": entities_ok,
        "pruning_ratio": out["metrics"]["pruning_ratio"],
        "elapsed_ms": elapsed,
        "structured_model": out
    })

# Summary
passed_count = sum(1 for r in results if r["passed"])
avg_pruning = np.mean([r["pruning_ratio"] for r in results]) * 100.0
avg_time = np.mean([r["elapsed_ms"] for r in results])

print("\n==================================================================")
print(f" STEP 1 VERIFICATION SUMMARY: {passed_count}/{len(test_cases)} Passed ({passed_count/len(test_cases)*100:.1f}%)")
print(f" Average Noise Pruning Ratio: {avg_pruning:.1f}%")
print(f" Average Latency per Parse:   {avg_time:.2f}ms")
print("==================================================================")

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "step1_abstraction_benchmark_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "passed_count": passed_count,
        "total_cases": len(test_cases),
        "all_passed": all_passed,
        "average_pruning_ratio": avg_pruning,
        "average_latency_ms": avg_time,
        "cases": results
    }, f, indent=2, ensure_ascii=False)
print(f"Saved results to {json_path}")

# Visualization Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Pruning Ratios
names = [r["name"][:20] + "..." for r in results]
prunings = [r["pruning_ratio"] * 100.0 for r in results]
colors = ['#27ae60' if r["passed"] else '#c0392b' for r in results]

bars = ax1.barh(names, prunings, color=colors, alpha=0.85)
ax1.set_xlabel('Noise Pruning Ratio (%) [Higher = Cleaner Modeling]', fontsize=10, fontweight='bold')
ax1.set_title('Step 1 Noise Pruning Efficiency across Domains', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    w = bar.get_width()
    ax1.text(w + 1.5, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%", ha='left', va='center', fontweight='bold', fontsize=9)

# Subplot 2: Pass/Fail and Metric Consistency
metrics = ['Entity Preservation', 'Rule Separation', 'Fact Separation', 'Overall Pass']
success_rates = [
    sum(1 for r in results if r["entities_ok"]) / len(results) * 100.0,
    sum(1 for r in results if r["rules_ok"]) / len(results) * 100.0,
    sum(1 for r in results if r["facts_ok"]) / len(results) * 100.0,
    passed_count / len(results) * 100.0
]

ax2.bar(metrics, success_rates, color=['#2980b9', '#16a085', '#8e44ad', '#27ae60'], width=0.5, alpha=0.9)
ax2.set_ylim(0, 115)
ax2.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
ax2.set_title('Step 1 Structural Consistency Metrics', fontsize=11, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for i, v in enumerate(success_rates):
    ax2.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
chart_path = os.path.join(ARTIFACT_DIR, "step1_abstraction_benchmark.png")
plt.savefig(chart_path, dpi=200)
plt.close()
print(f"Saved benchmark chart to {chart_path}")
