"""
Idea Inverse Reconstruction Engine (イデア逆算・断面完全化システム)
================================================================================
Implements the user's Method 2:
Instead of trying to collect all surface rules (Concept Core) or build an
impossible omniscient logical universe (Method 1), this system:

1. Provisional Completion (仮設写像 M_0 の構成):
   Constructs an initial mapping/hypothesis from partial knowledge and observations.
2. Multi-Perspective Projection & Contradiction Detection (多角的射影と矛盾 Δ の検知):
   Projects M_0 onto multiple independent axes (Forward prediction, Inverse/Contrapositive,
   Boundary/Extreme case, and Empirical Observation consistency).
   Because M_0 is an imperfect projection of the underlying Idea (イデア), a discrepancy
   or contradiction Δ inevitably emerges.
3. Inverse Deduction of the Unseen (矛盾 Δ からの逆算・不可視次元の特定):
   Treats the contradiction Δ not as a failure, but as a geometric compass pointing to
   the missing information / hidden variable / unseen law X that resolves Δ.
4. Cross-Section Proof & Verification (必要断面の完全化と論理証明):
   Constructs the repaired mapping M_1 = M_0 + X and verifies that all contradictions
   vanish across the required cross-section before outputting the final deduction.
"""

import os
import sys
import json
import time
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')


class IdeaInverseReconstructionEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        print(f"[IdeaInverseEngine] Loading base model on {self.device} (bfloat16): {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.eval()

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 220) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.12,
                pad_token_id=self.tokenizer.pad_token_id
            )
        gen_ids = outputs[0][len(inputs["input_ids"][0]):]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Baseline 1: Raw Direct Generation (Probabilistic Surface Projection)
    # ------------------------------------------------------------------
    def infer_raw_direct(self, problem_text: str) -> Dict:
        t0 = time.time()
        ans = self._generate(
            system_prompt="You are a helpful assistant. Answer the problem concisely.",
            user_prompt=problem_text,
            max_new_tokens=200
        )
        return {
            "method": "Raw_Probabilistic_LLM",
            "final_answer": ans,
            "latency_sec": time.time() - t0
        }

    # ------------------------------------------------------------------
    # Baseline 2: Concept Core (Rule Aggregation without Contradiction Inversion)
    # ------------------------------------------------------------------
    def infer_concept_core_aggregation(self, problem_text: str) -> Dict:
        """
        Extracts apparent rules from the prompt and aggregates them directly,
        without testing where the incomplete projection contradicts the deeper Idea.
        """
        t0 = time.time()
        rules_prompt = (
            f"Extract the explicit rules and common patterns from the following problem:\n"
            f"{problem_text}\n\n"
            f"List the extracted rules briefly:"
        )
        extracted_rules = self._generate(
            system_prompt="You are a rule extraction engine. Collect explicit rules from observations.",
            user_prompt=rules_prompt,
            max_new_tokens=150
        )

        solve_prompt = (
            f"Problem:\n{problem_text}\n\n"
            f"Aggregated Rules (Concept Core):\n{extracted_rules}\n\n"
            f"Apply these aggregated rules to answer the question concisely:"
        )
        final_ans = self._generate(
            system_prompt="You apply collected rules to answer questions.",
            user_prompt=solve_prompt,
            max_new_tokens=200
        )
        return {
            "method": "Concept_Core_Rule_Aggregation",
            "extracted_rules": extracted_rules,
            "final_answer": final_ans,
            "latency_sec": time.time() - t0
        }

    # ------------------------------------------------------------------
    # Proposed Method 2: Idea Discrepancy & Inverse Contradiction Engine
    # ------------------------------------------------------------------
    def infer_idea_inverse_reconstruction(self, problem_text: str) -> Dict:
        """
        4-Stage Idea Inverse Reconstruction:
        Stage 1: Construct Provisional Mapping (M_0) from initial knowledge/observations.
        Stage 2: Multi-Angle Projection & Contradiction Detection (Δ).
                 Project M_0 against all observations, boundary cases, and conservation laws
                 to explicitly locate where M_0 breaks down or contradicts reality.
        Stage 3: Inverse Deduction of the Unseen Dimension (X).
                 Calculate what hidden variable, unstated axiom, or higher-order law X
                 MUST exist so that the contradiction Δ is resolved.
        Stage 4: Targeted Cross-Section Perfection (M_1 = M_0 + X).
                 Prove that M_1 resolves both the general cases and the anomalous cases,
                 and derive the final answer from this perfected cross-section.
        """
        t0 = time.time()

        # Stage 1: Provisional Mapping (M_0)
        s1_prompt = (
            f"Problem / Observations:\n{problem_text}\n\n"
            f"Step 1 (Provisional Mapping M_0): Formulate the most obvious, naive surface rule or initial hypothesis "
            f"that someone would assume at first glance, and state what it predicts."
        )
        m0_mapping = self._generate(
            system_prompt="You formulate the initial naive hypothesis (Provisional Mapping M_0) from partial observations.",
            user_prompt=s1_prompt,
            max_new_tokens=140
        )

        # Stage 2: Multi-Perspective Projection & Contradiction Detection (Δ)
        s2_prompt = (
            f"Problem / Observations:\n{problem_text}\n\n"
            f"[Provisional Mapping M_0]:\n{m0_mapping}\n\n"
            f"Step 2 (Discrepancy & Contradiction Detection Δ):\n"
            f"Test M_0 strictly against EVERY observation, physical constraint, and logical extreme in the problem.\n"
            f"Identify the exact point where M_0 produces a CONTRADICTION, anomaly, or logical gap.\n"
            f"Output format:\n"
            f"CONTRADICTION_FOUND: <describe the exact discrepancy between M_0 and the full reality>\n"
            f"MISSING_INFORMATION_NEEDED: <what unseen factor or hidden rule is missing from M_0?>"
        )
        delta_analysis = self._generate(
            system_prompt="You are a rigorous contradiction detector. Find exactly where the naive mapping M_0 fails or contradicts observations.",
            user_prompt=s2_prompt,
            max_new_tokens=160
        )

        # Stage 3: Inverse Deduction of the Unseen (Reconstructing the Idea Cross-Section M_1)
        s3_prompt = (
            f"Problem / Observations:\n{problem_text}\n\n"
            f"[Initial Imperfect Mapping M_0]:\n{m0_mapping}\n\n"
            f"[Detected Contradiction / Discrepancy Δ]:\n{delta_analysis}\n\n"
            f"Step 3 (Inverse Reconstruction of the Idea Cross-Section M_1):\n"
            f"Work BACKWARDS from the contradiction Δ. Deduce the true underlying law or hidden mechanism X "
            f"that explains BOTH the normal observations and the contradiction without any inconsistency.\n"
            f"Verify that this reconstructed law M_1 is 100% logically consistent across all given facts."
        )
        m1_reconstruction = self._generate(
            system_prompt="You are an inverse deduction engine. Reconstruct the unseen true law (Idea cross-section) that resolves the contradiction.",
            user_prompt=s3_prompt,
            max_new_tokens=180
        )

        # Stage 4: Final Proof & Conclusion from Perfected Cross-Section
        s4_prompt = (
            f"Problem / Question:\n{problem_text}\n\n"
            f"[Perfected Idea Cross-Section & Proof M_1]:\n{m1_reconstruction}\n\n"
            f"Step 4 (Final Deduction):\n"
            f"Based strictly on the perfected, contradiction-free law M_1 above, state the exact final answer "
            f"and the logical proof concisely."
        )
        final_answer = self._generate(
            system_prompt="You deliver the final rigorous conclusion derived from the contradiction-free Idea cross-section.",
            user_prompt=s4_prompt,
            max_new_tokens=200
        )

        return {
            "method": "Idea_Inverse_Reconstruction",
            "stage1_provisional_m0": m0_mapping,
            "stage2_contradiction_delta": delta_analysis,
            "stage3_reconstructed_m1": m1_reconstruction,
            "final_answer": final_answer,
            "latency_sec": time.time() - t0
        }
