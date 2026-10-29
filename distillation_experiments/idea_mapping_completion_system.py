"""
Observation Completion & Idea-Mapping Generator (観測補完・イデア写像生成システム)
================================================================================
Designed strictly according to the user's architectural specification:
- This system does NOT replace the reasoning engine.
- Reasoning itself is performed by the existing 5-Step `UnifiedCognitiveArchitecture`
  (`unified_cognitive_architecture.py`).
- Instead, this system compensates for the *observational deficit* (観測能力の不足)
  of the existing system by constructing a provisional mapping M_0 from partial
  observations + the AI's internal knowledge, detecting where M_0 contradicts
  observations or boundary conditions (Idea Discrepancy Δ), working backwards from
  that contradiction to pinpoint and prove the missing information (M^*), and handing
  the completed mapping M^* to `UnifiedCognitiveArchitecture.solve()`.

High-Speed Optimization Architecture (高速化機構):
1. Direct-Struct Fast-Path (二重パース・文字列再構築の排除):
   When no observational deficit / boundary anomaly exists (e.g., complete ProntoQA graphs),
   parsed structures `(facts, rules, parsed_query)` are passed directly to Step 3 without
   re-serializing and re-parsing strings.
2. Lazy & Batched Resonance Evaluation (共鳴度の遅延・一括バッチ評価):
   Instead of invoking sequential GPU forward passes (`compute_affinity`) on every candidate
   branch before checking symbolic consistency, the engine first filters out contradictory
   branches via sub-millisecond symbolic closure (`_compute_forward_closure`) and batches
   any required GPU concept vector lookups.
3. Fused Single-Pass Epistemic Grounding + 1-Step Logit Choice Scoring + Batching:
   Reduces multiple-choice epistemic reasoning from 5-6 sequential `model.generate()` calls
   per question down to 1 fused grounding generation + 1 single-step forward logit lookup
   (with multi-question GPU batching support via `solve_mc_batch`).
"""

import os
import sys
import json
import time
import re
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set

import torch
import torch.nn.functional as F

from unified_cognitive_architecture import UnifiedCognitiveArchitecture

sys.stdout.reconfigure(encoding='utf-8')


class IdeaMappingCompleter:
    """
    Pre-Reasoning Observation Completion & Epistemic Diagnostic System
    (イデア写像補完・証明不能性／前提誤り逆算診断モジュール).
    """
    def __init__(self, cognitive_arch: UnifiedCognitiveArchitecture):
        self.arch = cognitive_arch
        self.model = cognitive_arch.model
        self.tokenizer = cognitive_arch.tokenizer
        self.device = cognitive_arch.device
        self.step1 = cognitive_arch.step1
        self.step2 = cognitive_arch.step2

    def _compute_forward_closure(
        self,
        facts: List[Dict],
        rules: List[Dict]
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], List[Dict]]:
        """
        Computes the positive and negative entailment closure for all known entities
        under a candidate mapping M = (facts, rules), and detects any logical contradictions (⊥).
        Includes Modus Tollens backward contrapositive propagation.
        """
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
                    "type": "direct_polarity_contradiction",
                    "entity": subj,
                    "predicate": c_pred,
                    "detail": f"Entity '{subj}' is entailed to be both '{c_pred}' and 'NOT {c_pred}'"
                })

        return entity_pos, entity_neg, contradictions

    def _compute_concept_reachability(self, start_concept: str, rules: List[Dict]) -> Set[str]:
        pos_graph = defaultdict(list)
        for r in rules:
            if not r["is_negative"]:
                pos_graph[r["antecedent"]].append(r["consequent"])

        visited = {start_concept}
        queue = deque([start_concept])
        while queue:
            curr = queue.popleft()
            for nxt in pos_graph.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def _parse_extended_observations(self, context: str, query: str):
        facts, rules, parsed_query = self.step1.parse_formal_graph(context, query)

        refined_facts = list(facts)
        refined_rules = []
        for r in rules:
            raw_s = r["raw"].strip()
            m_neg_fact = re.match(r'^([A-Z][a-zA-Z0-9\-]*)\s+is\s+not\s+(?:a\s+|an\s+)?([a-zA-Z0-9\-]+)', raw_s)
            if m_neg_fact and not raw_s.lower().startswith(("each", "every", "all", "no")):
                refined_facts.append({
                    "subject": m_neg_fact.group(1).strip(),
                    "predicate": self.step1.normalize_stem(m_neg_fact.group(2)),
                    "is_negative": True,
                    "raw": raw_s
                })
            else:
                refined_rules.append(r)

        return refined_facts, refined_rules, parsed_query

    def _find_unexplained_positive_observations(
        self,
        facts: List[Dict],
        rules: List[Dict]
    ) -> List[Tuple[str, List[str], str]]:
        all_consequents = set(r["consequent"] for r in rules if not r["is_negative"])
        entity_pos_list = defaultdict(list)
        for f in facts:
            if not f.get("is_negative", False):
                entity_pos_list[f["subject"].lower()].append(f["predicate"].lower())

        unexplained = []
        for ent_name, preds in entity_pos_list.items():
            if len(preds) < 2:
                continue
            roots = [p for p in preds if p not in all_consequents]
            if not roots:
                roots = [preds[0]]
            root_facts = [{"subject": ent_name, "predicate": r_p, "is_negative": False} for r_p in roots]
            r_pos, _, _ = self._compute_forward_closure(root_facts, rules)
            entailed_from_roots = r_pos[ent_name]

            for p in preds:
                if p not in entailed_from_roots:
                    unexplained.append((ent_name, list(entailed_from_roots), p))

        return unexplained

    def complete_formal_mapping(self, context: str, question: str, fast_mode: bool = True) -> Dict:
        """
        Completes an incomplete formal observation context.
        When `fast_mode=True`:
        1. Skips GPU affinity forward passes during symbolic contradiction elimination unless
           needed, or batches them in a single pass.
        2. Avoids unnecessary string reconstruction when no rules were added.
        """
        facts, rules, parsed_query = self._parse_extended_observations(context, question)
        q_subj = parsed_query["subject"].lower()
        q_pred = parsed_query["predicate"].lower()

        mapping_log = {
            "observational_deficit_detected": False,
            "provisional_m0_candidates": [],
            "contradictions_found_in_m0": [],
            "proven_inversions_added": [],
            "underdetermined_warnings": [],
            "premise_errors": []
        }

        base_pos, base_neg, base_contradictions = self._compute_forward_closure(facts, rules)
        if base_contradictions:
            mapping_log["premise_errors"].append({
                "type": "self_contradictory_premises",
                "details": [c["detail"] for c in base_contradictions]
            })

        known_entities = set(f["subject"].lower() for f in facts)
        known_concepts = (
            set(r["antecedent"] for r in rules) |
            set(r["consequent"] for r in rules) |
            set(f["predicate"].lower() for f in facts)
        )
        if q_subj not in known_entities:
            mapping_log["premise_errors"].append({
                "type": "unknown_query_subject",
                "details": f"Query subject '{parsed_query['subject']}' does not exist in the observed facts."
            })
        if q_pred not in known_concepts:
            mapping_log["premise_errors"].append({
                "type": "unknown_query_predicate",
                "details": f"Query concept '{q_pred}' does not appear anywhere in the observed rules or facts."
            })

        # Ultra-Fast Path check for standard complete-observation problems (e.g., ProntoQA 500):
        has_negative_obs = any(f.get("is_negative", False) for f in facts)
        has_multi_fact_entity = (len(facts) > len(known_entities))
        if fast_mode and not mapping_log["premise_errors"] and not has_negative_obs and not has_multi_fact_entity:
            return {
                "original_context": context,
                "completed_context": context,
                "mapping_log": mapping_log,
                "final_facts": facts,
                "final_rules": rules,
                "parsed_query": parsed_query,
                "final_pos_closure": base_pos,
                "final_neg_closure": base_neg,
                "fast_path_used": True
            }

        max_passes = 3
        refuted_entry_points: Set[str] = set()
        for pass_idx in range(max_passes):
            if mapping_log["premise_errors"]:
                break

            entity_pos, entity_neg, _ = self._compute_forward_closure(facts, rules)
            subj_reached = entity_pos.get(q_subj, set())

            if q_pred in subj_reached:
                break

            all_antecedents = set(r["antecedent"] for r in rules if not r["is_negative"])
            all_consequents = set(r["consequent"] for r in rules if not r["is_negative"])

            pos_outgoing = defaultdict(list)
            for r in rules:
                if not r["is_negative"]:
                    pos_outgoing[r["antecedent"]].append(r["consequent"])

            unexplained_refs = self._find_unexplained_positive_observations(facts, rules)

            if not unexplained_refs and not has_negative_obs:
                break

            added_in_this_pass = False

            # ------------------------------------------------------------------
            # Mechanism A: Reference-Entity Explanatory Discrepancy Inversion
            # ------------------------------------------------------------------
            if unexplained_refs:
                for ent_name, reached_from_roots, target_d_p in unexplained_refs:
                    ent_sinks = [c for c in reached_from_roots if len(pos_outgoing[c]) == 0]
                    cand_entries = [
                        c for c in all_antecedents
                        if c not in all_consequents and c not in reached_from_roots and c not in refuted_entry_points
                    ]
                    if not ent_sinks or not cand_entries:
                        continue

                    for src in ent_sinks:
                        mapping_log["observational_deficit_detected"] = True
                        if not fast_mode:
                            self.step2.precompute_concepts_batch([src] + cand_entries)
                            scored_cands = [
                                (self.step2.compute_affinity(src, dst, self.arch.step5.repulsive_field)["final_affinity"], dst)
                                for dst in cand_entries if src != dst
                            ]
                            scored_cands.sort(key=lambda x: x[0], reverse=True)
                        else:
                            # Lazy evaluation: symbolic check does not depend on ordering!
                            scored_cands = [(0.0, dst) for dst in cand_entries if src != dst]

                        mapping_log["provisional_m0_candidates"].append({
                            "pass": pass_idx + 1,
                            "mode": "reference_discrepancy_inversion",
                            "source": src,
                            "target_to_explain": target_d_p,
                            "ranked_targets": [(dst, round(s, 3)) for s, dst in scored_cands]
                        })

                        valid_explainers = []
                        for score, cand_dst in scored_cands:
                            trial_rule = {
                                "antecedent": src,
                                "consequent": cand_dst,
                                "is_negative": False,
                                "raw": f"Every {src} is a {cand_dst}"
                            }
                            trial_rules = rules + [trial_rule]
                            _, _, t_contra = self._compute_forward_closure(facts, trial_rules)
                            if t_contra:
                                refuted_entry_points.add(cand_dst)
                                mapping_log["contradictions_found_in_m0"].append({
                                    "rejected_bridge": f"Every {src} is a {cand_dst}",
                                    "prior_affinity": round(score, 3),
                                    "contradiction": t_contra[0]["detail"]
                                })
                                continue

                            dst_reach = self._compute_concept_reachability(cand_dst, trial_rules)
                            if target_d_p in dst_reach:
                                valid_explainers.append((score, cand_dst, trial_rule))
                            else:
                                mapping_log["contradictions_found_in_m0"].append({
                                    "rejected_bridge": f"Every {src} is a {cand_dst}",
                                    "prior_affinity": round(score, 3),
                                    "contradiction": f"Does not reach observed state '{target_d_p}' of '{ent_name}' (Δ != 0)"
                                })

                        if len(valid_explainers) == 1:
                            _, best_dst, best_rule = valid_explainers[0]
                            rules.append(best_rule)
                            mapping_log["proven_inversions_added"].append({
                                "added_rule": f"Every {src} is a {best_dst}.",
                                "proof_status": f"Uniquely explains observed state '{target_d_p}' of '{ent_name}' with zero contradiction (Δ = 0)"
                            })
                            added_in_this_pass = True
                            break
                        elif len(valid_explainers) == 0:
                            mapping_log["premise_errors"].append({
                                "type": "unexplainable_reference_observation",
                                "details": (
                                    f"Observed fact '{ent_name} is a {target_d_p}' cannot be reached from '{src}' "
                                    f"without causing a logical contradiction across available branches."
                                )
                            })
                    if added_in_this_pass:
                        break

            if added_in_this_pass:
                continue

            # ------------------------------------------------------------------
            # Mechanism B: Negative-Boundary Contradiction Elimination
            # ------------------------------------------------------------------
            if has_negative_obs:
                subj_sinks = [c for c in subj_reached if len(pos_outgoing[c]) == 0]
                cand_entries = [
                    c for c in all_antecedents
                    if c not in all_consequents and c not in subj_reached and c not in refuted_entry_points
                ]
                if not subj_sinks or not cand_entries:
                    break

                for src in subj_sinks:
                    if not fast_mode:
                        self.step2.precompute_concepts_batch([src] + cand_entries)
                        all_cands = [
                            (self.step2.compute_affinity(src, dst, self.arch.step5.repulsive_field)["final_affinity"], dst)
                            for dst in cand_entries if src != dst
                        ]
                        all_cands.sort(key=lambda x: x[0], reverse=True)
                    else:
                        all_cands = [(0.0, dst) for dst in cand_entries if src != dst]

                    if not all_cands:
                        continue

                    mapping_log["observational_deficit_detected"] = True
                    mapping_log["provisional_m0_candidates"].append({
                        "pass": pass_idx + 1,
                        "mode": "boundary_contradiction_elimination",
                        "source": src,
                        "ranked_targets": [(dst, round(s, 3)) for s, dst in all_cands]
                    })

                    contradiction_free_survivors = []
                    refuted_count = 0

                    for score, cand_dst in all_cands:
                        trial_rule = {
                            "antecedent": src,
                            "consequent": cand_dst,
                            "is_negative": False,
                            "raw": f"Every {src} is a {cand_dst}"
                        }
                        trial_rules = rules + [trial_rule]
                        _, _, t_contra = self._compute_forward_closure(facts, trial_rules)
                        if t_contra:
                            refuted_count += 1
                            refuted_entry_points.add(cand_dst)
                            mapping_log["contradictions_found_in_m0"].append({
                                "rejected_bridge": f"Every {src} is a {cand_dst}",
                                "prior_affinity": round(score, 3),
                                "contradiction": t_contra[0]["detail"]
                            })
                        else:
                            contradiction_free_survivors.append((score, cand_dst, trial_rule))

                    if refuted_count >= 1 and len(contradiction_free_survivors) == 1:
                        _, best_dst, best_rule = contradiction_free_survivors[0]
                        rules.append(best_rule)
                        mapping_log["proven_inversions_added"].append({
                            "added_rule": f"Every {src} is a {best_dst}.",
                            "proof_status": (
                                f"Unique survivor across all {len(all_cands)} candidate branches after refuting "
                                f"{refuted_count} competing branch(es) via boundary contradiction (Δ = 0)"
                            )
                        })
                        added_in_this_pass = True
                        break
                    elif len(contradiction_free_survivors) == 0 and len(all_cands) > 0:
                        mapping_log["premise_errors"].append({
                            "type": "all_candidate_branches_contradicted",
                            "source": src,
                            "details": (
                                f"All {len(all_cands)} candidate branches from '{src}' lead to a direct polarity "
                                f"contradiction (⊥) against the negative observations. The premises or negative "
                                f"observations themselves are mutually inconsistent."
                            )
                        })
                    elif len(contradiction_free_survivors) > 1:
                        surviving_entry_names = [dst for _, dst, _ in contradiction_free_survivors]
                        missing_options = []
                        for _, dst, _ in contradiction_free_survivors:
                            reach_nodes = sorted(list(self._compute_concept_reachability(dst, rules) - {dst}))
                            leaf_hint = reach_nodes[-1] if reach_nodes else dst
                            missing_options.append({
                                "candidate_bridge": f"Every {src} is a {dst}.",
                                "needed_positive_confirmation": f"Observe a member of '{src}' having property '{leaf_hint}' (or rule 'Every {src} is a {dst}')",
                                "needed_negative_elimination": f"Observe '{parsed_query['subject']} is not a {leaf_hint}' to rule out this branch"
                            })
                        mapping_log["underdetermined_warnings"].append({
                            "source": src,
                            "surviving_branches": surviving_entry_names,
                            "minimal_missing_observations": missing_options,
                            "reason": (
                                f"{len(contradiction_free_survivors)} branches ({surviving_entry_names}) remain "
                                f"contradiction-free; cannot uniquely determine which branch '{src}' connects to."
                            )
                        })

            if not added_in_this_pass:
                break

        final_pos, final_neg, _ = self._compute_forward_closure(facts, rules)
        clean_sentences = []
        for f in facts:
            if not f.get("is_negative", False):
                clean_sentences.append(f"{f['subject']} is a {f['predicate']}.")
        for r in rules:
            clean_sentences.append(r["raw"].rstrip(".") + ".")

        completed_context = " ".join(clean_sentences)
        return {
            "original_context": context,
            "completed_context": completed_context,
            "mapping_log": mapping_log,
            "final_facts": facts,
            "final_rules": rules,
            "parsed_query": parsed_query,
            "final_pos_closure": final_pos,
            "final_neg_closure": final_neg,
            "fast_path_used": False
        }

    def diagnose_epistemic_outcome(
        self,
        comp_res: Dict,
        five_step_result: Dict
    ) -> Dict:
        mapping_log = comp_res["mapping_log"]
        facts = comp_res["final_facts"]
        rules = comp_res["final_rules"]
        parsed_query = comp_res["parsed_query"]
        q_subj = parsed_query["subject"].lower()
        q_pred = parsed_query["predicate"].lower()
        q_is_neg = parsed_query.get("is_negative", False)

        if mapping_log["premise_errors"]:
            err = mapping_log["premise_errors"][0]
            return {
                "epistemic_status": "INVALID_PREMISE_OR_QUERY",
                "binary_fallback": False,
                "reason_type": err["type"],
                "explanation": (
                    f"[Premise/Query Error Detected] The question cannot be answered as stated because "
                    f"the premises or query contain a structural error: {err['details']}"
                ),
                "missing_information": []
            }

        entity_pos = comp_res.get("final_pos_closure")
        entity_neg = comp_res.get("final_neg_closure")
        if entity_pos is None or entity_neg is None:
            entity_pos, entity_neg, _ = self._compute_forward_closure(facts, rules)

        subj_reached = entity_pos.get(q_subj, set())
        subj_refuted = entity_neg.get(q_subj, set())

        if q_pred in subj_reached:
            if not q_is_neg:
                return {
                    "epistemic_status": "PROVEN_TRUE",
                    "binary_fallback": True,
                    "reason_type": "deductive_entailment_on_completed_mapping",
                    "explanation": f"Proven True: '{parsed_query['subject']}' is deductively entailed to be '{q_pred}' on mapping M^*.",
                    "missing_information": []
                }
            else:
                return {
                    "epistemic_status": "PROVEN_FALSE",
                    "binary_fallback": False,
                    "reason_type": "negated_query_refuted_by_positive_entailment",
                    "explanation": f"Proven False: The query asks if '{parsed_query['subject']}' is NOT '{q_pred}', but it is proven to be '{q_pred}'.",
                    "missing_information": []
                }

        if q_pred in subj_refuted:
            if not q_is_neg:
                return {
                    "epistemic_status": "PROVEN_FALSE",
                    "binary_fallback": False,
                    "reason_type": "explicit_negative_refutation",
                    "explanation": f"Proven False: '{parsed_query['subject']}' is explicitly proven NOT to be '{q_pred}' via negative boundary facts/rules.",
                    "missing_information": []
                }
            else:
                return {
                    "epistemic_status": "PROVEN_TRUE",
                    "binary_fallback": True,
                    "reason_type": "negated_query_confirmed_by_negative_refutation",
                    "explanation": f"Proven True: '{parsed_query['subject']}' is confirmed NOT to be '{q_pred}'.",
                    "missing_information": []
                }

        if mapping_log["underdetermined_warnings"]:
            warn = mapping_log["underdetermined_warnings"][-1]
            return {
                "epistemic_status": "UNDERDETERMINED_MISSING_INFO",
                "binary_fallback": False,
                "reason_type": "multiple_contradiction_free_branches_survive",
                "explanation": (
                    f"[Underdetermined / Insufficient Observation] Cannot prove whether '{parsed_query['subject']}' "
                    f"is '{q_pred}'. {warn['reason']}"
                ),
                "missing_information": warn["minimal_missing_observations"]
            }

        return {
            "epistemic_status": "PROVEN_FALSE",
            "binary_fallback": False,
            "reason_type": "closed_world_disjoint_branch",
            "explanation": (
                f"Proven False (Closed-World Disjoint): '{parsed_query['subject']}'s causal chain is complete "
                f"({sorted(list(subj_reached))}) and does not intersect '{q_pred}'."
            ),
            "missing_information": []
        }

    def complete_epistemic_mapping(self, question: str, choices: List[str], fast_mode: bool = True) -> Dict:
        """
        Fused Single-Pass Epistemic Grounding when `fast_mode=True`:
        Instead of 2 separate generation calls for M_0 and M^*, extracts both the common
        misconception M_0 and the verified scientific/factual observation M^* in a single
        concise generation pass.
        """
        if fast_mode:
            prompt_fused = (
                f"Question: \"{question}\"\n"
                f"1. State the common myth or naive assumption (M_0) in 1 short sentence.\n"
                f"2. State the strict scientific or historical fact (M^*) that refutes or clarifies M_0 in 1-2 short sentences."
            )
            messages = [
                {"role": "system", "content": "You are an epistemic contradiction detector. Contrast popular myths (M_0) with strict scientific/historical facts (M^*)."},
                {"role": "user", "content": prompt_fused}
            ]
            chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = self.tokenizer(chat, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=85, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            fused_text = self.tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()
            return {
                "m0_assumption": "Fused single-pass M_0 -> M^* extraction",
                "completed_observation": fused_text
            }
        else:
            prompt_m0 = (
                f"Question: \"{question}\"\n"
                f"State the common surface assumption (M_0) people often make about this question in 1 sentence:"
            )
            messages_m0 = [
                {"role": "system", "content": "You state the common initial assumption M_0 briefly."},
                {"role": "user", "content": prompt_m0}
            ]
            chat_m0 = self.tokenizer.apply_chat_template(messages_m0, tokenize=False, add_generation_prompt=True)
            inp_m0 = self.tokenizer(chat_m0, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_m0 = self.model.generate(**inp_m0, max_new_tokens=60, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            m0_assumption = self.tokenizer.decode(out_m0[0][len(inp_m0["input_ids"][0]):], skip_special_tokens=True).strip()

            prompt_inv = (
                f"Question: \"{question}\"\n"
                f"Naive Surface Assumption (M_0): \"{m0_assumption}\"\n\n"
                f"Task (Contradiction & Missing Observation Analysis):\n"
                f"1. Point out where the naive assumption M_0 contradicts strict empirical evidence, physical law, or literal reality.\n"
                f"2. State the exact missing factual observation (M^*) needed to answer the question without falling for the myth."
            )
            messages_inv = [
                {"role": "system", "content": "You are an epistemic contradiction detector. Expose where naive assumptions contradict strict reality and state the verified factual truth."},
                {"role": "user", "content": prompt_inv}
            ]
            chat_inv = self.tokenizer.apply_chat_template(messages_inv, tokenize=False, add_generation_prompt=True)
            inp_inv = self.tokenizer(chat_inv, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_inv = self.model.generate(**inp_inv, max_new_tokens=110, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            completed_observation = self.tokenizer.decode(out_inv[0][len(inp_inv["input_ids"][0]):], skip_special_tokens=True).strip()

            return {
                "m0_assumption": m0_assumption,
                "completed_observation": completed_observation
            }


class ObservationCompletedCognitiveSystem:
    """
    Wraps the existing `UnifiedCognitiveArchitecture` with the pre-reasoning
    `IdeaMappingCompleter`, 4-Valued Epistemic Diagnostic Engine, and Fast/Batched Inference.
    """
    def __init__(self, base_arch: UnifiedCognitiveArchitecture, fast_mode: bool = True):
        self.arch = base_arch
        self.completer = IdeaMappingCompleter(base_arch)
        self.fast_mode = fast_mode

    def _select_choice_logit_fast(self, raw_question: str, choices: list, grounded_truth: str) -> int:
        """
        1-Step Forward Logit Evaluation (replaces 15-token autoregressive `.generate()`).
        Computes log-probabilities over digit tokens '0', '1', ..., 'k' in a single forward pass.
        """
        formatted_choices = "\n".join([f"[{i}] {c}" for i, c in enumerate(choices)])
        prompt = (
            f"Original Question: {raw_question}\n"
            f"Established Scientific / Historical Finding:\n{grounded_truth}\n\n"
            f"Candidate Choices:\n{formatted_choices}\n\n"
            f"Task: Which single choice index [0, 1, 2, ...] logically aligns with the established finding?\n"
            f"Answer with ONLY the single digit of the choice index."
        )
        messages = [
            {"role": "system", "content": "You are an impartial logic judge. Select the single choice that agrees with the established finding."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = self.arch.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.arch.tokenizer(chat_prompt, return_tensors="pt").to(self.arch.device)
        with torch.no_grad():
            logits = self.arch.model(**inputs).logits[0, -1, :]

        best_idx = 0
        best_logit = -1e9
        for i in range(len(choices)):
            tok_ids = self.arch.tokenizer.encode(str(i), add_special_tokens=False)
            if tok_ids:
                val = logits[tok_ids[-1]].item()
                if val > best_logit:
                    best_logit = val
                    best_idx = i
        return best_idx

    def solve(self, question: str, context: str = "", choices: list = None) -> Dict:
        t0 = time.time()

        # Route 1: Formal Context with Potential Unseen Rules / Observational Gaps
        if context.strip() and not choices:
            comp_res = self.completer.complete_formal_mapping(context, question, fast_mode=self.fast_mode)
            completed_context = comp_res["completed_context"]

            if self.fast_mode and comp_res.get("fast_path_used", False):
                # Direct-Struct Fast-Path: Hand parsed structures directly to Step 3 of UnifiedCognitiveArchitecture
                facts = comp_res["final_facts"]
                rules = comp_res["final_rules"]
                parsed_query = comp_res["parsed_query"]
                is_isolated = self.arch.step4.isolate_universe(rules)
                ded_res = self.arch.step3.deduce(
                    facts, rules, parsed_query,
                    self.arch.step5.nogood_constraints,
                    self.arch.step2,
                    self.arch.step5.repulsive_field
                )
                reasoning_result = {
                    "route": "Fast-Path IdeaMappingCompleter -> Step 3 Axiomatic Deduction",
                    "sandbox_isolated": is_isolated,
                    "decision": ded_res["decision"],
                    "proof_chain": ded_res["proof_chain"],
                    "source": ded_res["source"]
                }
            else:
                reasoning_result = self.arch.solve(question=question, context=completed_context)

            epistemic_diag = self.completer.diagnose_epistemic_outcome(comp_res, reasoning_result)
            reasoning_result["mapping_completion"] = comp_res["mapping_log"]
            reasoning_result["completed_context"] = completed_context
            reasoning_result["epistemic_diagnosis"] = epistemic_diag
            reasoning_result["epistemic_status"] = epistemic_diag["epistemic_status"]
            reasoning_result["total_elapsed_ms"] = (time.time() - t0) * 1000.0
            return reasoning_result

        # Route 2: Multiple-Choice / Open Epistemic Reasoning
        elif choices and len(choices) > 1:
            if self.fast_mode:
                # Fused Epistemic Completion (1 generation call instead of 5!) + 1-Step Logit Selection
                epistemic_map = self.completer.complete_epistemic_mapping(question, choices, fast_mode=True)
                enriched_grounding = epistemic_map["completed_observation"]
                selected_idx = self._select_choice_logit_fast(question, choices, enriched_grounding)
            else:
                q_pre, q_con = self.arch.step1.sublimate_open_question(question)
                fact_pre = self.arch.step3.ground_factual_truth(q_pre)
                fact_con = self.arch.step3.ground_factual_truth(q_con)
                epistemic_map = self.completer.complete_epistemic_mapping(question, choices, fast_mode=False)
                enriched_grounding = (
                    f"Premise validation: {fact_pre}\n"
                    f"Scope & Consensus: {fact_con}\n"
                    f"Contradiction-Inverted Observation (M^*): {epistemic_map['completed_observation']}"
                )
                selected_idx = self.arch.step3.evaluate_choices_with_grounding(question, choices, enriched_grounding)

            return {
                "route": "IdeaMappingCompleter (M_0 -> Δ -> M^*) -> Existing Step 1-5 UnifiedCognitiveArchitecture",
                "decision": selected_idx,
                "selected_choice": choices[selected_idx],
                "mapping_completion": epistemic_map,
                "grounded_truth": enriched_grounding,
                "total_elapsed_ms": (time.time() - t0) * 1000.0
            }

        else:
            return self.arch.solve(question=question, context=context, choices=choices)

    def solve_mc_batch(self, items: List[Dict], batch_size: int = 10) -> List[Dict]:
        """
        Batched GPU execution for Multiple-Choice Epistemic problems (e.g. TruthfulQA 50):
        1. Batches the Fused Epistemic Observation (M_0 -> Δ -> M^*) generation across `batch_size` questions.
        2. Batches the 1-Step Forward Logit Choice Scoring across `batch_size` questions.
        """
        results = []
        tok = self.arch.tokenizer
        model = self.arch.model
        dev = self.arch.device
        orig_pad_side = tok.padding_side
        tok.padding_side = "left"

        try:
            for start_i in range(0, len(items), batch_size):
                batch = items[start_i:start_i + batch_size]
                t_b0 = time.time()

                # Pass 1: Batched Fused Epistemic Completion (M_0 -> Δ -> M^*)
                grounding_chats = []
                for it in batch:
                    q = it["question"]
                    prompt_fused = (
                        f"Question: \"{q}\"\n"
                        f"1. State the common myth or naive assumption (M_0) in 1 short sentence.\n"
                        f"2. State the strict scientific or historical fact (M^*) that refutes or clarifies M_0 in 1-2 short sentences."
                    )
                    msgs = [
                        {"role": "system", "content": "You are an epistemic contradiction detector. Contrast popular myths (M_0) with strict scientific/historical facts (M^*)."},
                        {"role": "user", "content": prompt_fused}
                    ]
                    grounding_chats.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

                enc_g = tok(grounding_chats, return_tensors="pt", padding=True).to(dev)
                in_len = enc_g["input_ids"].shape[1]
                with torch.no_grad():
                    out_g = model.generate(**enc_g, max_new_tokens=85, do_sample=False, pad_token_id=tok.pad_token_id)

                grounded_texts = [
                    tok.decode(out_g[j][in_len:], skip_special_tokens=True).strip()
                    for j in range(len(batch))
                ]

                # Pass 2: Batched 1-Step Forward Logit Choice Scoring
                judge_chats = []
                for j, it in enumerate(batch):
                    q = it["question"]
                    choices = it["choices"]
                    formatted_choices = "\n".join([f"[{idx}] {c}" for idx, c in enumerate(choices)])
                    prompt_j = (
                        f"Original Question: {q}\n"
                        f"Established Scientific / Historical Finding:\n{grounded_texts[j]}\n\n"
                        f"Candidate Choices:\n{formatted_choices}\n\n"
                        f"Task: Which single choice index [0, 1, 2, ...] logically aligns with the established finding?\n"
                        f"Answer with ONLY the single digit of the choice index."
                    )
                    msgs_j = [
                        {"role": "system", "content": "You are an impartial logic judge. Select the single choice that agrees with the established finding."},
                        {"role": "user", "content": prompt_j}
                    ]
                    judge_chats.append(tok.apply_chat_template(msgs_j, tokenize=False, add_generation_prompt=True))

                enc_j = tok(judge_chats, return_tensors="pt", padding=True).to(dev)
                with torch.no_grad():
                    logits_j = model(**enc_j).logits[:, -1, :]

                per_item_ms = ((time.time() - t_b0) * 1000.0) / len(batch)
                for j, it in enumerate(batch):
                    choices = it["choices"]
                    best_idx = 0
                    best_val = -1e9
                    for c_i in range(len(choices)):
                        t_ids = tok.encode(str(c_i), add_special_tokens=False)
                        if t_ids:
                            v = logits_j[j, t_ids[-1]].item()
                            if v > best_val:
                                best_val = v
                                best_idx = c_i
                    results.append({
                        "decision": best_idx,
                        "selected_choice": choices[best_idx],
                        "grounded_truth": grounded_texts[j],
                        "total_elapsed_ms": per_item_ms
                    })
        finally:
            tok.padding_side = orig_pad_side

        return results
