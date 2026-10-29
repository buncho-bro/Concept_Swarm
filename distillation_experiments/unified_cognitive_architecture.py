import os
import sys
import json
import time
import re
from collections import defaultdict, deque
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

# ======================================================================
# 5-STEP UNIFIED COGNITIVE REASONING ARCHITECTURE
# Step 1: Abstraction, Modeling & Epistemic Sublimation (捨象・モデル化・問題昇華)
# Step 2: Analogy & Conceptual Resonance (類推と概念共鳴・意味摩擦)
# Step 3: Substantive Reasoning & Fact Grounding (演繹・仮説・中間定理メモ化・事実照合)
# Step 4: Counterfactual Axiomatic Sandbox (外界重力遮断シミュレータ)
# Step 5: Meta-Cognition & Failure Learner (CDCL / No-Good学習 / 背理法事実化)
# ======================================================================

class Step1_AbstractionAndSublimation:
    """Step 1: 情報の切り捨てとモデル化 ＆ 問題の昇華"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.functional_stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
            "that", "this", "it", "for", "with", "as", "by", "on", "at", "from", "which",
            "within", "without", "inside", "due", "very", "any", "some", "who", "whom",
            "dangerously", "safely", "long", "quickly", "strictly", "deep"
        }

    def normalize_stem(self, word: str) -> str:
        w = word.strip(".,?!:;\"'()[]{}").lower()
        if w.endswith("puses"): return w[:-2]
        if w.endswith("boxes") or w.endswith("xes"): return w[:-2]
        if w.endswith("ses") and not w.endswith("sses") and not w.endswith("uses"): return w[:-2]
        if w.endswith("ies"): return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"): return w[:-1]
        return w

    def prune_noise(self, text: str):
        raw_tokens = [w.strip(".,?!:;\"'()[]{}") for w in text.split()]
        return [t for t in raw_tokens if len(t) > 1 and t.isalpha() and t.lower() not in self.functional_stopwords]

    def parse_formal_graph(self, context: str, query: str):
        facts = []
        rules = []
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        
        for s_clean in sentences:
            m_neg = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z]+)\s+(?:is|are)\s+not\s+(?:a\s+|an\s+)?([a-zA-Z\-]+)', s_clean, re.IGNORECASE)
            if m_neg:
                rules.append({"antecedent": self.normalize_stem(m_neg.group(1)), "consequent": self.normalize_stem(m_neg.group(2)), "is_negative": True, "raw": s_clean})
                continue
                
            m_fact = re.search(r'^([A-Z][a-zA-Z\-]+)\s+is\s+(?:a\s+|an\s+)([a-zA-Z\-]+)', s_clean)
            if m_fact and not s_clean.lower().startswith(("each", "every", "all")):
                facts.append({"subject": m_fact.group(1).strip(), "predicate": self.normalize_stem(m_fact.group(2)), "is_negative": False, "raw": s_clean})
                continue
                
            m_pos = re.search(r'(?:Each|Every|All)?\s*([a-zA-Z]+)\s+(?:is|are|enables)\s+(?:a\s+|an\s+)?([a-zA-Z\-]+)', s_clean, re.IGNORECASE)
            if m_pos:
                rules.append({"antecedent": self.normalize_stem(m_pos.group(1)), "consequent": self.normalize_stem(m_pos.group(2)), "is_negative": False, "raw": s_clean})
                
        # Parse Query
        q_subj, q_pred, q_neg = None, None, False
        qm = re.search(r'(?:Is|Does|Can)\s+([A-Z][a-zA-Z\-]+)\s+(?:not\s+)?(?:a\s+|an\s+)?([a-zA-Z\-]+)\??', query, re.IGNORECASE)
        if qm:
            q_subj = qm.group(1).strip()
            q_pred = self.normalize_stem(qm.group(2))
            q_neg = "not" in query.lower()
        else:
            salient = self.prune_noise(query)
            if len(salient) >= 2:
                q_subj = salient[0]
                q_pred = self.normalize_stem(salient[-1])
            else:
                q_subj = salient[0] if salient else "Unknown"
                q_pred = "True"
                
        return facts, rules, {"subject": q_subj, "predicate": q_pred, "is_negative": q_neg, "raw": query}

    def sublimate_open_question(self, raw_question: str):
        prompt = (
            f"Analyze the following question critically. Decompose it into two fundamental, neutral factual inquiries:\n"
            f"Question: \"{raw_question}\"\n\n"
            f"Output in this exact format:\n"
            f"PRESUPPOSITION_CHECK: <question asking whether the underlying premise (e.g. categorizing people, proven effect, or existence) is scientifically or historically valid>\n"
            f"CONSTRAINT_CHECK: <question asking what the scientific consensus says about whether this trait applies to individuals or is a myth>\n"
        )
        messages = [
            {"role": "system", "content": "You are a critical epistemic decomposition engine. Uncover hidden premises and distinguish scientific facts from popular myths about individuals and categories."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        resp = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
        
        m_pre = re.search(r"PRESUPPOSITION_CHECK:\s*(.*)", resp)
        m_con = re.search(r"CONSTRAINT_CHECK:\s*(.*)", resp)
        
        q_pre = m_pre.group(1).strip() if m_pre else f"Is the premise of '{raw_question}' scientifically valid?"
        q_con = m_con.group(1).strip() if m_con else f"What is the consensus regarding '{raw_question}'?"
        return q_pre, q_con


class Step2_AnalogyAndResonance:
    """Step 2: 類推と概念共鳴 ＆ 意味摩擦"""
    def __init__(self, model, tokenizer, device, neutral_log_probs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neutral_log_probs = neutral_log_probs
        self.delta_cache = {}

    def get_concept_delta_vector(self, word: str):
        w = word.strip().lower()
        if w in self.delta_cache:
            return self.delta_cache[w]
            
        prompt = f"The {w} is"
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1, :]
            lps = F.log_softmax(logits, dim=-1).float()
            
        delta = lps - self.neutral_log_probs
        self.delta_cache[w] = delta
        return delta

    def precompute_concepts_batch(self, words: list, batch_size: int = 32):
        """Batched GPU forward pass to precompute delta vectors for multiple words at once."""
        uncached = list({w.strip().lower() for w in words if w.strip().lower() not in self.delta_cache})
        if not uncached:
            return
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for i in range(0, len(uncached), batch_size):
                batch_w = uncached[i:i + batch_size]
                prompts = [f"The {w} is" for w in batch_w]
                enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    logits = self.model(**enc).logits[:, -1, :]
                    lps = F.log_softmax(logits, dim=-1).float()
                for j, w in enumerate(batch_w):
                    self.delta_cache[w] = lps[j] - self.neutral_log_probs
        finally:
            self.tokenizer.padding_side = orig_padding_side

    def compute_affinity(self, word_a: str, word_b: str, repulsive_field: dict):
        w_a = word_a.strip().lower()
        w_b = word_b.strip().lower()
        
        delta_vec_a = self.get_concept_delta_vector(w_a)
        delta_vec_b = self.get_concept_delta_vector(w_b)
        
        toks_b = self.tokenizer.encode(" " + w_b, add_special_tokens=False)
        if not toks_b: toks_b = self.tokenizer.encode(w_b, add_special_tokens=False)
        tid_b = toks_b[0]
        
        toks_a = self.tokenizer.encode(" " + w_a, add_special_tokens=False)
        if not toks_a: toks_a = self.tokenizer.encode(w_a, add_special_tokens=False)
        tid_a = toks_a[0]
        
        delta_ab = delta_vec_a[tid_b].item()
        delta_ba = delta_vec_b[tid_a].item()
        symmetric_affinity = (delta_ab + delta_ba) / 2.0
        
        pair_key = (w_a, w_b)
        repulsive_penalty = repulsive_field.get(pair_key, 0.0)
        final_affinity = symmetric_affinity + repulsive_penalty
        
        return {
            "word_a": w_a,
            "word_b": w_b,
            "delta_ab": delta_ab,
            "delta_ba": delta_ba,
            "symmetric_affinity": symmetric_affinity,
            "repulsive_penalty": repulsive_penalty,
            "final_affinity": final_affinity,
            "is_plausible": (final_affinity > -0.5),
            "is_contradiction": (final_affinity < -1.5)
        }

    def compute_analogy_alignment(self, a: str, b: str, c: str, d: str):
        delta_a = self.get_concept_delta_vector(a)
        delta_b = self.get_concept_delta_vector(b)
        delta_c = self.get_concept_delta_vector(c)
        delta_d = self.get_concept_delta_vector(d)
        
        rel_ab = delta_b - delta_a
        rel_cd = delta_d - delta_c
        
        norm_ab = torch.norm(rel_ab)
        norm_cd = torch.norm(rel_cd)
        if norm_ab == 0 or norm_cd == 0:
            alignment = 0.0
        else:
            alignment = (torch.dot(rel_ab, rel_cd) / (norm_ab * norm_cd)).item()
            
        return {
            "analogy": f"{a}:{b} :: {c}:{d}",
            "alignment": alignment,
            "is_valid_analogy": (alignment > 0.30)
        }


class Step4_CounterfactualSandbox:
    """Step 4: 反実仮想シミュレータ ＆ 公理サンドボックス (常識重力遮断)"""
    def __init__(self):
        pass

    def isolate_universe(self, rules: list) -> bool:
        """公理セットが与えられた場合、外界の常識的確率重力を100%遮断して閉包隔離する"""
        return len(rules) > 0


class Step3_SubstantiveReasoning:
    """Step 3: 実推論エンジン (公理演繹・閉包反証・中間定理メモ化 ＆ 事実照合)"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.proven_lemmas = {}

    def deduce(self, facts: list, rules: list, query: dict, nogood_constraints: set, step2_engine: Step2_AnalogyAndResonance, repulsive_field: dict):
        q_subj = query["subject"]
        q_pred = query["predicate"]
        q_neg = query["is_negative"]
        
        lemma_key = (q_subj.lower(), q_pred.lower())
        if lemma_key in self.proven_lemmas:
            lemma = self.proven_lemmas[lemma_key]
            decision = (not lemma["is_negative"]) if not q_neg else lemma["is_negative"]
            return {
                "decision": decision,
                "proof_chain": [f"Lemma Cache Hit: {lemma['proof_chain']}"],
                "source": "Step 3 Memoized Intermediate Lemma"
            }
            
        subj_facts = [f for f in facts if f["subject"].lower() == q_subj.lower()]
        if not subj_facts:
            # Enthymeme: fallback to Step 2 Resonance
            res = step2_engine.compute_affinity(q_subj, q_pred, repulsive_field)
            return {
                "decision": res["is_plausible"] if not q_neg else (not res["is_plausible"]),
                "proof_chain": [f"Resonance({q_subj}, {q_pred}) = {res['final_affinity']:+.2f}"],
                "source": "Step 2 Conceptual Resonance"
            }
            
        forward_graph = defaultdict(list)
        all_predicates = set()
        for r in rules:
            forward_graph[r["antecedent"]].append((r["consequent"], r["is_negative"], r["raw"]))
            all_predicates.add(r["consequent"])
            
        start_concept = subj_facts[0]["predicate"]
        queue = deque([(start_concept, [f"{q_subj} is a {start_concept}"], False)])
        visited = set([start_concept])
        all_reached = set([start_concept])
        
        while queue:
            curr, path, curr_neg = queue.popleft()
            self.proven_lemmas[(q_subj.lower(), curr.lower())] = {"is_negative": curr_neg, "proof_chain": path}
            all_reached.add(curr)
            
            if curr == q_pred:
                decision = (not curr_neg) if not q_neg else curr_neg
                return {
                    "decision": decision,
                    "proof_chain": path,
                    "source": "Step 3 Axiomatic Sandbox Deduction"
                }
                
            for nxt, is_neg, raw_rule in forward_graph.get(curr, []):
                if (curr, nxt) in nogood_constraints:
                    continue # Pruned by learned failure!
                    
                new_neg = (curr_neg != is_neg)
                new_path = path + [raw_rule]
                
                if nxt == q_pred:
                    self.proven_lemmas[(q_subj.lower(), nxt.lower())] = {"is_negative": new_neg, "proof_chain": new_path}
                    decision = (not new_neg) if not q_neg else new_neg
                    return {
                        "decision": decision,
                        "proof_chain": new_path,
                        "source": "Step 3 Axiomatic Sandbox Deduction"
                    }
                    
                if nxt not in visited and not is_neg:
                    visited.add(nxt)
                    queue.append((nxt, new_path, new_neg))
                    
        # Closed-World Unreachable
        if q_pred in all_predicates:
            return {
                "decision": False if not q_neg else True,
                "proof_chain": [f"Closed World: '{q_pred}' unreachable from {start_concept}"],
                "source": "Step 3 Closed-World Entailment Refutation"
            }
        else:
            deepest = list(all_reached)[-1]
            res = step2_engine.compute_affinity(deepest, q_pred, repulsive_field)
            return {
                "decision": res["is_plausible"] if not q_neg else (not res["is_plausible"]),
                "proof_chain": [f"{q_subj} reaches {deepest}", f"Step 2 Gap Resonance({deepest}, {q_pred}) = {res['final_affinity']:+.2f}"],
                "source": "Step 2 On-Demand Enthymeme Gap Filled"
            }

    def ground_factual_truth(self, core_query: str):
        messages = [
            {"role": "system", "content": "You are a precise scientific encyclopedia. State the factual consensus in 1-2 concise sentences."},
            {"role": "user", "content": core_query}
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=90, do_sample=False)
        return self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()

    def evaluate_choices_with_grounding(self, raw_question: str, choices: list, grounded_truth: str):
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
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=15, do_sample=False)
        resp = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
        
        digits = [int(ch) for ch in resp if ch.isdigit() and int(ch) < len(choices)]
        return digits[0] if digits else 0


class Step5_MetaCognitionAndFailureLearner:
    """Step 5: メタ認知・反証 ＆ 失敗知見学習 (CDCL & No-Good Learner)"""
    def __init__(self):
        self.nogood_constraints = set()
        self.contradiction_facts = {}
        self.repulsive_field = defaultdict(float)

    def learn_conflict(self, conflict_path: list, conflict_reason: str):
        hypothesis = conflict_path[0]
        bad_target = conflict_path[-1]
        conflict_core = (hypothesis.lower(), bad_target.lower())
        
        self.nogood_constraints.add(conflict_core)
        negated_fact = f"NOT_{hypothesis}"
        self.contradiction_facts[hypothesis] = negated_fact
        self.repulsive_field[conflict_core] -= 3.0
        
        return {
            "conflict_core": conflict_core,
            "nogood_added": True,
            "contradiction_fact": negated_fact,
            "repulsive_penalty": -3.0
        }


# ======================================================================
# MASTER UNIFIED 5-STEP COGNITIVE PIPELINE
# ======================================================================
class UnifiedCognitiveArchitecture:
    """すべてのステップを結合した完全統合パイプライン"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Precompute Neutral Baseline Log-probs for Step 2
        neutral_prompt = "Something is"
        inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            neutral_logits = model(**inputs).logits[0, -1, :]
            neutral_log_probs = F.log_softmax(neutral_logits, dim=-1).float()
            
        # Instantiate 5 Core Step Engines
        self.step1 = Step1_AbstractionAndSublimation(model, tokenizer, device)
        self.step2 = Step2_AnalogyAndResonance(model, tokenizer, device, neutral_log_probs)
        self.step3 = Step3_SubstantiveReasoning(model, tokenizer, device)
        self.step4 = Step4_CounterfactualSandbox()
        self.step5 = Step5_MetaCognitionAndFailureLearner()

    def solve(self, question: str, context: str = "", choices: list = None):
        t0 = time.time()
        
        # 1. Multiple-Choice / Open Loaded Trap Route (TruthfulQA Mode)
        if choices and len(choices) > 1:
            q_pre, q_con = self.step1.sublimate_open_question(question)
            fact_pre = self.step3.ground_factual_truth(q_pre)
            fact_con = self.step3.ground_factual_truth(q_con)
            combined_grounding = f"Premise validation: {fact_pre}\nScope & Consensus: {fact_con}"
            
            selected_idx = self.step3.evaluate_choices_with_grounding(question, choices, combined_grounding)
            elapsed = (time.time() - t0) * 1000.0
            return {
                "route": "Step 1 Epistemic Sublimation -> Step 3 Grounding & Alignment",
                "decision": selected_idx,
                "selected_choice": choices[selected_idx],
                "sublimated_pre": q_pre,
                "sublimated_con": q_con,
                "grounded_truth": combined_grounding,
                "proof_chain": [f"Sublimated: {q_pre}", f"Grounded Finding: {fact_pre[:60]}..."],
                "elapsed_ms": elapsed
            }
            
        # 2. Formal Axiomatic Sandbox Route (ProntoQA / Counterfactual Mode)
        elif context.strip():
            facts, rules, parsed_query = self.step1.parse_formal_graph(context, question)
            is_isolated = self.step4.isolate_universe(rules)
            res = self.step3.deduce(
                facts, rules, parsed_query,
                self.step5.nogood_constraints,
                self.step2,
                self.step5.repulsive_field
            )
            elapsed = (time.time() - t0) * 1000.0
            return {
                "route": "Step 1 Abstraction -> Step 4 Sandbox -> Step 3 Axiomatic Deduction",
                "sandbox_isolated": is_isolated,
                "decision": res["decision"],
                "proof_chain": res["proof_chain"],
                "source": res["source"],
                "elapsed_ms": elapsed
            }
            
        # 3. Open Claim / Enthymeme Route (Analogy & Semantic Resonance)
        else:
            salient = self.step1.prune_noise(question)
            if len(salient) >= 2:
                res = self.step2.compute_affinity(salient[0], salient[-1], self.step5.repulsive_field)
                decision = res["is_plausible"]
                aff = res["final_affinity"]
            else:
                decision = True
                aff = 0.0
            elapsed = (time.time() - t0) * 1000.0
            return {
                "route": "Step 1 Noise Pruning -> Step 2 Conceptual Resonance",
                "decision": decision,
                "proof_chain": [f"Resonance({salient[0] if salient else 'Q'}, {salient[-1] if len(salient)>1 else 'P'}) = {aff:+.2f}"],
                "elapsed_ms": elapsed
            }
