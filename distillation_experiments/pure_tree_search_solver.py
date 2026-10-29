"""
Pure Tree Search Solver & Comparative Benchmark
(純粋な木探索・古典的記号グラフソルバによる比較検証)
================================================================================
Implements a 100% neural-free, purely symbolic tree search engine:
- Zero LLM generation, zero conceptual delta vectors, zero resonance.
- Pure Breadth-First / Depth-First Tree Search & Forward/Backward Entailment Closure.
- Evaluates:
  1. Standard Forward Tree Search (Vanilla Closed-World Search without completion)
  2. Pure Symbolic Tree Search with Inverse Abductive Elimination (Symbolic Solver)
- Runs across all benchmark categories to directly compare against the Neural-Cognitive systems:
  * Category A1 (ProntoQA Complete 5-Hop: N = 500)
  * Category A2 (Incomplete Nonce Worlds: N = 60)
  * Category A3 (4-Valued Epistemic Diagnosis: N = 60)
  * Category B  (TruthfulQA Open-World Myths: N = 50)
  * Category C  (Semantic Friction & Affinity: N = 30)
"""

import os
import sys
import json
import time
import re
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Optional

sys.stdout.reconfigure(encoding='utf-8')


class PureTreeSearchSolver:
    """
    Pure Symbolic Tree Search & Graph Deduction Solver (ニューラルフリー純粋木探索器).
    Contains NO LLM calls, NO neural network weights, NO conceptual vectors.
    """
    def __init__(self):
        self.functional_stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
            "that", "this", "it", "for", "with", "as", "by", "on", "at", "from", "which"
        }

    def normalize_stem(self, word: str) -> str:
        w = word.strip(".,?!:;\"'()[]{}").lower()
        if w.endswith("puses"): return w[:-2]
        if w.endswith("boxes") or w.endswith("xes"): return w[:-2]
        if w.endswith("ses") and not w.endswith("sses") and not w.endswith("uses"): return w[:-2]
        if w.endswith("ies"): return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"): return w[:-1]
        return w

    def parse_formal_graph(self, context: str, query: str):
        facts = []
        rules = []
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        for s_clean in sentences:
            m_neg_fact = re.match(r'^([A-Z][a-zA-Z0-9\-]*)\s+is\s+not\s+(?:a\s+|an\s+)?([a-zA-Z0-9\-]+)', s_clean)
            if m_neg_fact and not s_clean.lower().startswith(("each", "every", "all", "no")):
                facts.append({
                    "subject": m_neg_fact.group(1).strip(),
                    "predicate": self.normalize_stem(m_neg_fact.group(2)),
                    "is_negative": True,
                    "raw": s_clean
                })
                continue

            m_neg_rule = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z0-9\-]+)\s+(?:is|are)\s+not\s+(?:a\s+|an\s+)?([a-zA-Z0-9\-]+)', s_clean, re.IGNORECASE)
            if m_neg_rule:
                rules.append({
                    "antecedent": self.normalize_stem(m_neg_rule.group(1)),
                    "consequent": self.normalize_stem(m_neg_rule.group(2)),
                    "is_negative": True,
                    "raw": s_clean
                })
                continue

            m_fact = re.search(r'^([A-Z][a-zA-Z0-9\-]*)\s+is\s+(?:a\s+|an\s+)([a-zA-Z0-9\-]+)', s_clean)
            if m_fact and not s_clean.lower().startswith(("each", "every", "all")):
                facts.append({
                    "subject": m_fact.group(1).strip(),
                    "predicate": self.normalize_stem(m_fact.group(2)),
                    "is_negative": False,
                    "raw": s_clean
                })
                continue

            m_pos_rule = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z0-9\-]+)\s+(?:is|are|enables)\s+(?:a\s+|an\s+)?([a-zA-Z0-9\-]+)', s_clean, re.IGNORECASE)
            if m_pos_rule:
                rules.append({
                    "antecedent": self.normalize_stem(m_pos_rule.group(1)),
                    "consequent": self.normalize_stem(m_pos_rule.group(2)),
                    "is_negative": False,
                    "raw": s_clean
                })

        q_subj, q_pred, q_neg = None, None, False
        qm = re.search(r'(?:Is|Does|Can)\s+([A-Z][a-zA-Z0-9\-]*)\s+(?:not\s+)?(?:a\s+|an\s+)?([a-zA-Z0-9\-]+)\??', query, re.IGNORECASE)
        if qm:
            q_subj = qm.group(1).strip()
            q_pred = self.normalize_stem(qm.group(2))
            q_neg = "not" in query.lower()
        else:
            tokens = [w.strip(".,?!:;\"'") for w in query.split() if w.strip(".,?!:;\"'").lower() not in self.functional_stopwords]
            q_subj = tokens[0] if tokens else "Unknown"
            q_pred = self.normalize_stem(tokens[-1]) if len(tokens) > 1 else "True"

        return facts, rules, {"subject": q_subj, "predicate": q_pred, "is_negative": q_neg, "raw": query}

    def compute_tree_closure(self, facts: List[Dict], rules: List[Dict]):
        """Forward + Contrapositive backward propagation on purely symbolic graphs."""
        pos_graph = defaultdict(list)
        rev_pos_graph = defaultdict(list)
        neg_graph = defaultdict(list)
        for r in rules:
            if r["is_negative"]:
                neg_graph[r["antecedent"]].append(r["consequent"])
                neg_graph[r["consequent"]].append(r["antecedent"])
            else:
                pos_graph[r["antecedent"]].append(r["consequent"])
                rev_pos_graph[r["consequent"]].append(r["antecedent"])

        entity_pos = defaultdict(set)
        entity_neg = defaultdict(set)
        contradictions = []

        for f in facts:
            subj = f["subject"].lower()
            pred = f["predicate"].lower()
            if f.get("is_negative", False):
                entity_neg[subj].add(pred)
            else:
                entity_pos[subj].add(pred)

        all_entities = sorted(list(set(list(entity_pos.keys()) + list(entity_neg.keys()))))
        for subj in all_entities:
            # Positive BFS tree traversal
            queue = deque(list(entity_pos[subj]))
            visited_pos = set(entity_pos[subj])
            while queue:
                curr = queue.popleft()
                for nxt in pos_graph.get(curr, []):
                    if nxt not in visited_pos:
                        visited_pos.add(nxt)
                        entity_pos[subj].add(nxt)
                        queue.append(nxt)
                for neg_nxt in neg_graph.get(curr, []):
                    entity_neg[subj].add(neg_nxt)

            # Backward Modus Tollens BFS traversal
            neg_queue = deque(list(entity_neg[subj]))
            visited_neg = set(entity_neg[subj])
            while neg_queue:
                curr_neg = neg_queue.popleft()
                for prev_pos in rev_pos_graph.get(curr_neg, []):
                    if prev_pos not in visited_neg:
                        visited_neg.add(prev_pos)
                        entity_neg[subj].add(prev_pos)
                        neg_queue.append(prev_pos)

            overlap = entity_pos[subj].intersection(entity_neg[subj])
            for c_pred in sorted(list(overlap)):
                contradictions.append({
                    "entity": subj,
                    "predicate": c_pred,
                    "detail": f"Contradiction: '{subj}' entails both '{c_pred}' and 'NOT {c_pred}'"
                })

        return entity_pos, entity_neg, contradictions

    def search_uncompleted_tree(self, context: str, question: str) -> Dict:
        """
        Mode 1: Standard Pure Tree Search (Vanilla BFS without any observation completion).
        Follows strictly observed edges. If target is unreachable in tree, returns False (Closed-World).
        """
        t0 = time.time()
        facts, rules, query = self.parse_formal_graph(context, question)
        entity_pos, entity_neg, _ = self.compute_tree_closure(facts, rules)

        q_subj = query["subject"].lower()
        q_pred = query["predicate"].lower()
        q_neg = query.get("is_negative", False)

        reached = entity_pos.get(q_subj, set())
        refuted = entity_neg.get(q_subj, set())

        if q_pred in reached:
            decision = (not q_neg)
        elif q_pred in refuted:
            decision = q_neg
        else:
            # Standard Closed-World Tree Search: Unreachable = False
            decision = False if not q_neg else True

        elapsed = (time.time() - t0) * 1000.0
        return {
            "mode": "Pure Vanilla Tree Search (No Completion)",
            "decision": decision,
            "reached_nodes": sorted(list(reached)),
            "elapsed_ms": elapsed
        }

    def search_with_symbolic_abduction(self, context: str, question: str) -> Dict:
        """
        Mode 2: Pure Symbolic Tree Search with Abductive Elimination (記号的逆算補完木探索).
        Performs pure combinatorial graph elimination across candidate components WITHOUT neural resonance.
        Also outputs the 4-valued epistemic diagnosis.
        """
        t0 = time.time()
        facts, rules, query = self.parse_formal_graph(context, question)
        q_subj = query["subject"].lower()
        q_pred = query["predicate"].lower()
        q_is_neg = query.get("is_negative", False)

        # Step 1: Base contradiction check
        base_pos, base_neg, base_contra = self.compute_tree_closure(facts, rules)
        if base_contra:
            return {
                "decision": False,
                "epistemic_status": "INVALID_PREMISE_OR_QUERY",
                "reason": "Base premises contain self-contradiction",
                "elapsed_ms": (time.time() - t0) * 1000.0
            }

        known_entities = set(f["subject"].lower() for f in facts)
        known_concepts = set(r["antecedent"] for r in rules) | set(r["consequent"] for r in rules) | set(f["predicate"].lower() for f in facts)

        if q_subj not in known_entities or q_pred not in known_concepts:
            return {
                "decision": False,
                "epistemic_status": "INVALID_PREMISE_OR_QUERY",
                "reason": "Unknown subject or concept in query",
                "elapsed_ms": (time.time() - t0) * 1000.0
            }

        # Step 2: Check if observation deficit exists
        has_negative_obs = any(f.get("is_negative", False) for f in facts)
        all_consequents = set(r["consequent"] for r in rules if not r["is_negative"])
        all_antecedents = set(r["antecedent"] for r in rules if not r["is_negative"])

        # Find unexplained reference observations
        entity_pos_list = defaultdict(list)
        for f in facts:
            if not f.get("is_negative", False):
                entity_pos_list[f["subject"].lower()].append(f["predicate"].lower())

        unexplained_refs = []
        for ent_name, preds in entity_pos_list.items():
            if len(preds) >= 2:
                roots = [p for p in preds if p not in all_consequents] or [preds[0]]
                r_pos, _, _ = self.compute_tree_closure([{"subject": ent_name, "predicate": r_p, "is_negative": False} for r_p in roots], rules)
                for p in preds:
                    if p not in r_pos[ent_name]:
                        unexplained_refs.append((ent_name, list(r_pos[ent_name]), p))

        refuted_entry_points = set()
        underdetermined = False
        all_branches_contradicted = False

        pos_outgoing = defaultdict(list)
        for r in rules:
            if not r["is_negative"]:
                pos_outgoing[r["antecedent"]].append(r["consequent"])

        # Multi-pass symbolic tree connection
        for pass_i in range(3):
            e_pos, e_neg, _ = self.compute_tree_closure(facts, rules)
            subj_reached = e_pos.get(q_subj, set())
            if q_pred in subj_reached:
                break

            added_bridge = False

            # Mechanism A: Pure Symbolic Reference Inversion
            if unexplained_refs:
                for ent_name, reached_from_roots, target_d_p in unexplained_refs:
                    ent_sinks = [c for c in reached_from_roots if len(pos_outgoing[c]) == 0]
                    cand_entries = [c for c in all_antecedents if c not in all_consequents and c not in reached_from_roots and c not in refuted_entry_points]
                    for src in ent_sinks:
                        valid_explainers = []
                        for dst in cand_entries:
                            if src == dst: continue
                            t_rule = {"antecedent": src, "consequent": dst, "is_negative": False, "raw": f"Each {src} is a {dst}."}
                            _, _, t_contra = self.compute_tree_closure(facts, rules + [t_rule])
                            if t_contra:
                                refuted_entry_points.add(dst)
                                continue
                            # Check reachability of target_d_p
                            q_reach = deque([dst])
                            v_reach = {dst}
                            reached_target = (dst == target_d_p)
                            while q_reach and not reached_target:
                                curr_r = q_reach.popleft()
                                for nxt_r in pos_outgoing.get(curr_r, []):
                                    if nxt_r == target_d_p:
                                        reached_target = True
                                        break
                                    if nxt_r not in v_reach:
                                        v_reach.add(nxt_r)
                                        q_reach.append(nxt_r)
                            if reached_target:
                                valid_explainers.append((dst, t_rule))

                        if len(valid_explainers) == 1:
                            rules.append(valid_explainers[0][1])
                            pos_outgoing[src].append(valid_explainers[0][0])
                            added_bridge = True
                            break
                    if added_bridge: break

            if added_bridge: continue

            # Mechanism B: Pure Symbolic Negative-Boundary Elimination
            if has_negative_obs:
                subj_sinks = [c for c in subj_reached if len(pos_outgoing[c]) == 0]
                cand_entries = [c for c in all_antecedents if c not in all_consequents and c not in subj_reached and c not in refuted_entry_points]
                for src in subj_sinks:
                    survivors = []
                    refuted_in_src = 0
                    for dst in cand_entries:
                        if src == dst: continue
                        t_rule = {"antecedent": src, "consequent": dst, "is_negative": False, "raw": f"Each {src} is a {dst}."}
                        _, _, t_contra = self.compute_tree_closure(facts, rules + [t_rule])
                        if t_contra:
                            refuted_in_src += 1
                            refuted_entry_points.add(dst)
                        else:
                            survivors.append((dst, t_rule))

                    if refuted_in_src >= 1 and len(survivors) == 1:
                        rules.append(survivors[0][1])
                        pos_outgoing[src].append(survivors[0][0])
                        added_bridge = True
                        break
                    elif len(survivors) == 0 and len(cand_entries) > 0:
                        all_branches_contradicted = True
                    elif len(survivors) > 1:
                        underdetermined = True

            if not added_bridge:
                break

        # Final evaluation
        final_pos, final_neg, _ = self.compute_tree_closure(facts, rules)
        subj_reached = final_pos.get(q_subj, set())
        subj_refuted = final_neg.get(q_subj, set())

        elapsed = (time.time() - t0) * 1000.0

        if all_branches_contradicted:
            return {
                "decision": False,
                "epistemic_status": "INVALID_PREMISE_OR_QUERY",
                "reason": "All candidate branches lead to contradiction",
                "elapsed_ms": elapsed
            }

        if q_pred in subj_reached:
            return {
                "decision": not q_is_neg,
                "epistemic_status": "PROVEN_TRUE" if not q_is_neg else "PROVEN_FALSE",
                "reason": "Positively entailed on symbolic tree closure",
                "elapsed_ms": elapsed
            }

        if q_pred in subj_refuted:
            return {
                "decision": q_is_neg,
                "epistemic_status": "PROVEN_FALSE" if not q_is_neg else "PROVEN_TRUE",
                "reason": "Explicitly refuted on tree closure",
                "elapsed_ms": elapsed
            }

        if underdetermined:
            return {
                "decision": False,
                "epistemic_status": "UNDERDETERMINED_MISSING_INFO",
                "reason": "Multiple branches survive contradiction-free",
                "elapsed_ms": elapsed
            }

        return {
            "decision": False if not q_is_neg else True,
            "epistemic_status": "PROVEN_FALSE",
            "reason": "Closed-world tree disjoint",
            "elapsed_ms": elapsed
        }

    def solve_open_world_myth(self, question: str, choices: List[str]) -> Dict:
        """
        Category B (TruthfulQA): Pure tree search with NO external world knowledge graph.
        Without external facts, a pure tree search engine has zero graph edges!
        It must fail or guess randomly.
        """
        # A purely symbolic engine with zero knowledge has no basis to choose; defaults to choice 0
        return {
            "decision": 0,
            "reason": "Pure tree search has zero knowledge edges for open-world semantic questions",
            "elapsed_ms": 0.01
        }

    def evaluate_semantic_resonance(self, a: str, b: str) -> Dict:
        """
        Category C: Semantic Affinity.
        Without an embedding manifold or neural weights, pure symbolic graphs cannot compute affinity.
        """
        return {
            "is_plausible": False,
            "reason": "Pure tree search has no semantic embedding vector space"
        }
