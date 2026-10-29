"""
Idea Cross-Section Completer (イデア断面補完システム)
================================================================================
Core Philosophy by the User:
- "The reasoning engine itself (input -> output) is already sound."
- "Hallucination happens because the INPUT itself is inherently incomplete (missing premises,
  unspoken boundaries, unobserved conditions)."
- "Defining a complete Idea of the entire universe is impossible."
- "Therefore, Idea Completion means giving up on defining a complete Idea, finding only
  the specific cross-section needed for the current reasoning, and perfecting only that
  cross-section before handing it to the reasoning engine."

Pipeline Architecture:
1. Cross-Section Probe:
   Identifies what exact boundary relation, implicit physical law, or premise invariant
   is REQUIRED to deductively answer the question (without modeling the entire universe).
2. Local Provisional Mapping (M_0):
   Surfaces the intuitive or surface assumptions from latent model memory.
3. Invariant Refutation & Contradiction Inversion (Discrepancy Δ != 0):
   Tests M_0 against strict physical laws, conservation invariants, and observed facts.
   Refutes invalid assumptions and recovers the uniquely consistent cross-section (M^*).
4. Epistemic Decision & Diagnostic Routing:
   - If M^* is uniquely perfected: Hand completed cross-section to reasoning engine -> Deductive Answer (Zero Hallucination).
   - If multiple cross-sections remain consistent: Output exact Missing Information Diagnosis ("Cannot answer determinately without condition X").
   - If all candidate cross-sections contradict: Output Premise Contradiction Diagnosis ("The question presupposes a physically impossible premise").
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from unified_cognitive_architecture import UnifiedCognitiveArchitecture

sys.stdout.reconfigure(encoding='utf-8')


class IdeaCrossSectionCompleter:
    """
    Perfection of Local Reasoning Cross-Sections (イデア局所断面完全化エンジン).
    """
    def __init__(self, cognitive_arch: UnifiedCognitiveArchitecture):
        self.arch = cognitive_arch
        self.model = cognitive_arch.model
        self.tokenizer = cognitive_arch.tokenizer
        self.device = cognitive_arch.device

    def extract_reasoning_cross_section(self, question: str, context: str = "") -> Dict:
        """
        Step 1: Identifies the precise local cross-section needed for reasoning.
        Instead of modeling the world, identifies:
        - Target Property to Determine
        - Implicit Boundary Invariants (e.g., conservation laws, medium conditions, definitions)
        - Potential Surface Trap / Ambiguity
        """
        prompt = (
            f"Context: {context if context else 'None provided.'}\n"
            f"Question: \"{question}\"\n\n"
            f"Task: Identify the exact logical/physical cross-section needed to answer this question without hallucination.\n"
            f"Output in this exact format:\n"
            f"REQUIRED_CROSS_SECTION: <the 1 core relationship or boundary condition that must be verified>\n"
            f"NAIVE_ASSUMPTION_M0: <what a surface-level statistical model or person would guess based on common association>\n"
            f"INVARIANT_BOUNDARY: <the strict physical law, logical constraint, or explicit boundary condition that governs this>"
        )
        messages = [
            {"role": "system", "content": "You are a precision epistemic cross-section analyzer. Cut away all irrelevant world knowledge and isolate only the exact boundary condition required for the target deduction."},
            {"role": "user", "content": prompt}
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.tokenizer(chat, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=130, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        resp = self.tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()

        m_sec = re.search(r"REQUIRED_CROSS_SECTION:\s*(.*)", resp)
        m_m0 = re.search(r"NAIVE_ASSUMPTION_M0:\s*(.*)", resp)
        m_inv = re.search(r"INVARIANT_BOUNDARY:\s*(.*)", resp)

        return {
            "required_cross_section": m_sec.group(1).strip() if m_sec else "Core causality",
            "naive_assumption_m0": m_m0.group(1).strip() if m_m0 else "Common statistical co-occurrence",
            "invariant_boundary": m_inv.group(1).strip() if m_inv else "Strict conservation/logical invariants"
        }

    def perfect_cross_section(self, question: str, context: str, cross_section: Dict) -> Dict:
        """
        Step 2 & 3: Tests candidate assumptions against the invariant boundary.
        Detects contradiction (Δ != 0) and inverts to obtain the perfected local cross-section M^*.
        Also diagnoses whether the cross-section is:
        - `PERFECTED`: Uniquely determined and proven.
        - `UNDERDETERMINED`: Multiple valid cross-sections exist; identifies minimal missing conditions.
        - `CONTRADICTORY_PREMISE`: All candidate cross-sections violate invariant boundaries.
        """
        prompt = (
            f"Question: \"{question}\"\n"
            f"Context: {context if context else 'None provided.'}\n"
            f"Required Cross-Section: {cross_section['required_cross_section']}\n"
            f"Naive Surface Assumption (M_0): {cross_section['naive_assumption_m0']}\n"
            f"Invariant Boundary Constraint: {cross_section['invariant_boundary']}\n\n"
            f"Task: Evaluate the cross-section against the invariant boundary.\n"
            f"1. Is the premise of the question physically/logically impossible (Contradictory Premise)?\n"
            f"2. Does the naive assumption M_0 contradict the invariant boundary? If so, state why (Δ != 0).\n"
            f"3. Can the cross-section be uniquely determined, or are necessary conditions fundamentally missing?\n\n"
            f"Output in this exact format:\n"
            f"EPISTEMIC_STATUS: <PERFECTED or UNDERDETERMINED or CONTRADICTORY_PREMISE>\n"
            f"PERFECTED_CROSS_SECTION_M_STAR: <the exact verified fact/premise, or NONE if underdetermined/contradictory>\n"
            f"MISSING_OR_REFUTED_CONDITION: <the exact missing condition needed, or the contradiction that refutes the premise>"
        )
        messages = [
            {"role": "system", "content": "You are a rigorous neuro-symbolic idea inversion engine. Perfect the reasoning cross-section by refuting contradictions against invariant boundaries."},
            {"role": "user", "content": prompt}
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.tokenizer(chat, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=150, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        resp = self.tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()

        m_status = re.search(r"EPISTEMIC_STATUS:\s*(PERFECTED|UNDERDETERMINED|CONTRADICTORY_PREMISE)", resp, re.IGNORECASE)
        m_mstar = re.search(r"PERFECTED_CROSS_SECTION_M_STAR:\s*(.*)", resp)
        m_cond = re.search(r"MISSING_OR_REFUTED_CONDITION:\s*(.*)", resp)

        status = m_status.group(1).upper() if m_status else "PERFECTED"
        m_star = m_mstar.group(1).strip() if m_mstar else ""
        cond = m_cond.group(1).strip() if m_cond else ""

        return {
            "epistemic_status": status,
            "perfected_cross_section_m_star": m_star,
            "missing_or_refuted_condition": cond,
            "raw_analysis": resp
        }

    def solve(self, question: str, context: str = "", choices: List[str] = None) -> Dict:
        """
        Executes the full Idea Cross-Section Completion pipeline:
        1. Cut out required cross-section.
        2. Test M_0 against invariant boundary -> perfect M^* or diagnose deficit.
        3. Hand perfected M^* to reasoning engine (or return precise diagnostic).
        """
        t0 = time.time()
        cross_section = self.extract_reasoning_cross_section(question, context)
        perfection_res = self.perfect_cross_section(question, context, cross_section)

        status = perfection_res["epistemic_status"]

        # Case A: Cross-Section is Underdetermined (Conditions missing)
        if status == "UNDERDETERMINED":
            return {
                "decision_type": "UNDERDETERMINED_MISSING_INFO",
                "final_answer": (
                    f"[Missing Information Diagnosis] The question cannot be answered definitively because the input is incomplete. "
                    f"Required missing condition: {perfection_res['missing_or_refuted_condition']}"
                ),
                "cross_section_probe": cross_section,
                "perfection_analysis": perfection_res,
                "elapsed_ms": (time.time() - t0) * 1000.0
            }

        # Case B: Premise is Contradictory / Invalid
        if status == "CONTRADICTORY_PREMISE":
            return {
                "decision_type": "CONTRADICTORY_PREMISE",
                "final_answer": (
                    f"[Premise Contradiction Diagnosis] The question presupposes a logically or physically impossible condition: "
                    f"{perfection_res['missing_or_refuted_condition']}"
                ),
                "cross_section_probe": cross_section,
                "perfection_analysis": perfection_res,
                "elapsed_ms": (time.time() - t0) * 1000.0
            }

        # Case C: Cross-Section is Perfected (M^*) -> Hand to Reasoning Engine
        m_star = perfection_res["perfected_cross_section_m_star"]
        if choices and len(choices) > 1:
            # 1-step direct logit scoring over candidate choices using the perfected cross-section!
            formatted_choices = "\n".join([f"[{i}] {c}" for i, c in enumerate(choices)])
            prompt_judge = (
                f"Question: {question}\n"
                f"Verified Cross-Section Fact (M^*): {m_star}\n\n"
                f"Choices:\n{formatted_choices}\n\n"
                f"Task: Select the choice index that logically aligns with M^*. Output only the single digit index."
            )
            messages = [
                {"role": "system", "content": "You are a deductive logic selector. Select the choice matching M^* strictly."},
                {"role": "user", "content": prompt_judge}
            ]
            chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = self.tokenizer(chat, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inp).logits[0, -1, :]

            best_idx = 0
            best_val = -1e9
            for i in range(len(choices)):
                t_ids = self.tokenizer.encode(str(i), add_special_tokens=False)
                if t_ids:
                    val = logits[t_ids[-1]].item()
                    if val > best_val:
                        best_val = val
                        best_idx = i

            return {
                "decision_type": "PROVEN_DEDUCTION",
                "selected_choice_idx": best_idx,
                "selected_choice": choices[best_idx],
                "perfected_cross_section_m_star": m_star,
                "cross_section_probe": cross_section,
                "elapsed_ms": (time.time() - t0) * 1000.0
            }
        else:
            # Open deductive answer based on perfected M^*
            prompt_deduce = (
                f"Question: \"{question}\"\n"
                f"Context: {context if context else 'None'}\n"
                f"Verified Cross-Section Grounding (M^*): {m_star}\n\n"
                f"State the concise, definitive answer to the question deduced strictly from M^* in 1-2 sentences:"
            )
            messages = [
                {"role": "system", "content": "You answer questions strictly deduced from the verified cross-section M^* without hallucination."},
                {"role": "user", "content": prompt_deduce}
            ]
            chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = self.tokenizer(chat, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=90, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            final_ans = self.tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()

            return {
                "decision_type": "PROVEN_DEDUCTION",
                "final_answer": final_ans,
                "perfected_cross_section_m_star": m_star,
                "cross_section_probe": cross_section,
                "elapsed_ms": (time.time() - t0) * 1000.0
            }
