import os
import sys
import json
import time
import re
from collections import defaultdict, deque
from datasets import load_dataset

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"

print("==================================================================")
print(" PRONTOQA 500-QUESTION BENCHMARK: 5-STEP REASONING ENGINE        ")
print(" Full Validation Split (All 500 Deductive Proof Problems)        ")
print("==================================================================")

print("Loading ProntoQA validation dataset...")
dataset = load_dataset('renma/ProntoQA')['validation']
print(f"Total problems: {len(dataset)}")

def normalize_stem(word: str) -> str:
    w = word.strip(".,?!:;\"'()[]{}").lower()
    if w.endswith("puses"): return w[:-2]
    if w.endswith("ses") and not w.endswith("sses") and not w.endswith("uses"): return w[:-2]
    if w.endswith("ies"): return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"): return w[:-1]
    return w

def parse_prontoqa_item(context: str, question: str):
    facts = []
    rules = []
    sentences = [s.strip() for s in context.split(".") if s.strip()]
    for s_clean in sentences:
        m_neg = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z]+)\s+(?:is|are)\s+not\s+(?:a\s+|an\s+)?([a-zA-Z\-]+)', s_clean, re.IGNORECASE)
        if m_neg:
            rules.append({"antecedent": normalize_stem(m_neg.group(1)), "consequent": normalize_stem(m_neg.group(2)), "is_negative": True, "raw": s_clean})
            continue
        m_fact = re.search(r'^([A-Z][a-zA-Z\-]+)\s+is\s+(?:a\s+|an\s+)([a-zA-Z\-]+)', s_clean)
        if m_fact and not s_clean.lower().startswith(("each", "every", "all")):
            facts.append({"subject": m_fact.group(1).strip(), "predicate": normalize_stem(m_fact.group(2)), "is_negative": False, "raw": s_clean})
            continue
        m_pos = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z]+)\s+(?:is|are)\s+(?:a\s+|an\s+)?([a-zA-Z\-]+)', s_clean, re.IGNORECASE)
        if m_pos:
            rules.append({"antecedent": normalize_stem(m_pos.group(1)), "consequent": normalize_stem(m_pos.group(2)), "is_negative": False, "raw": s_clean})
            continue

    # Extract query target
    # Question format: "Is the following statement true or false? [Subject] is (not) [Predicate]."
    m_q = re.search(r'([A-Z][a-zA-Z\-]+)\s+is\s+(not\s+)?(?:a\s+|an\s+)?([a-zA-Z\-]+)', question)
    if m_q:
        q_subj = m_q.group(1).strip()
        q_neg = bool(m_q.group(2))
        q_pred = normalize_stem(m_q.group(3))
    else:
        q_subj, q_neg, q_pred = "Unknown", False, "unknown"
        
    return facts, rules, q_subj, q_pred, q_neg

def solve_prontoqa(facts, rules, q_subj, q_pred, q_neg):
    forward_graph = defaultdict(list)
    all_predicates = set()
    for r in rules:
        forward_graph[r["antecedent"]].append((r["consequent"], r["is_negative"], r["raw"]))
        all_predicates.add(r["antecedent"])
        all_predicates.add(r["consequent"])
        
    subj_facts = [f for f in facts if f["subject"].lower() == q_subj.lower()]
    if not subj_facts:
        return False # Unknown
        
    start_concept = subj_facts[0]["predicate"]
    queue = deque([(start_concept, [f"{q_subj} is a {start_concept}"], False)])
    visited = set([start_concept])
    
    proven_decision = None
    
    while queue:
        curr, path, curr_neg = queue.popleft()
        
        if curr == q_pred:
            proven_decision = (not curr_neg) if not q_neg else curr_neg
            break
            
        for nxt, is_neg, raw_rule in forward_graph.get(curr, []):
            new_neg = (curr_neg != is_neg)
            new_path = path + [raw_rule]
            
            if nxt == q_pred:
                proven_decision = (not new_neg) if not q_neg else new_neg
                queue.clear()
                break
                
            if nxt not in visited and not is_neg:
                visited.add(nxt)
                queue.append((nxt, new_path, new_neg))
                
    if proven_decision is not None:
        return proven_decision
        
    # If target is in universe but unreachable -> Closed world False
    if q_pred in all_predicates:
        return False if not q_neg else True
    else:
        return False

# Run over all 500 questions
correct = 0
total = len(dataset)
start_t = time.time()

for idx, ex in enumerate(dataset):
    facts, rules, q_subj, q_pred, q_neg = parse_prontoqa_item(ex["context"], ex["question"])
    pred = solve_prontoqa(facts, rules, q_subj, q_pred, q_neg)
    ground_truth = True if ex["answer"].strip() == "A" else False
    
    if pred == ground_truth:
        correct += 1

elapsed = time.time() - start_t
acc = (correct / total) * 100.0

print("\n==================================================================")
print(f" PRONTOQA 500 QUESTIONS RESULT: {correct}/{total} ({acc:.1f}%) in {elapsed:.2f}s")
print(f" Average Latency per Problem:   {elapsed/total*1000:.3f}ms")
print("==================================================================")
