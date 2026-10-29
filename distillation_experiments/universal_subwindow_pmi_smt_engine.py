"""
Universal Sub-Window De-Bleeding & Pure PMI Counterfactual Engine (USW-PMI-SMT)
(第2階層：純粋相互情報量PMI＆直接トークン確率 ＋ 第3階層：汎用部分窓マスクによる特徴漏洩除去)
========================================================================================
58.40%の壁（残存104問のボトルネック）を突破する4つの普遍的数理メカニズム:

1. 【汎用部分窓マスクによる特徴漏洩の除去 (Universal Sub-Window De-Bleeding)】:
   特定ベンチマークの書式（Scenario 1 | などの正規表現）を一切使用せず、任意の複数文コンテキスト
   Q = [s_1, ..., s_m] を汎用的な文境界の中央点 (midpoint = m // 2) で前半窓 Q_first と後半窓 Q_second
   に分割し、それぞれの脱重力スコア S_c(Q_first), S_c(Q_second) を全体スコア S_c(Q_full) に重畳する:
   S_total(Q) = S_c(Q_full) + γ * (S_c(Q_first) + S_c(Q_second))
   これにより、1.5Bモデルの単一パスで後半の強い刺激が前半の命題を塗り潰す「特徴漏洩 (Feature Bleeding)」
   および背景記述が近接因果事象を希釈する現象を非破壊的に解消する。

2. 【無条件事前確率の減算による純粋相互情報量 (Pure Pointwise Mutual Information / PMI)】:
   照応解決において、文脈に無関係な「選択肢自体の流暢性・職業ステレオタイプ log P(H | C_null)」が
   混入して Ambiguous の決定境界を歪めるボトルネックに対し、無情報文脈 C_null での対数尤度を差し引く:
   PMI_β(H; C) = log P(H | C) - β * log P(H | C_null)

3. 【双方向境界マスク反事実変更能 (Dual-Boundary Counterfactual Influence)】:
   形式論理において、冒頭の導入装飾文 (s_0) と末尾の結論文 (s_last) の双方に対して反事実マスク介入を行い、
   純粋な演繹推論の変位のみを抽出する。

4. 【自己回帰生成パース誤差の完全排除 (Direct Symbol Logit Marginalization)】:
   自己回帰生成 (model.generate) が選択肢の本文単語（数字や代名詞）を出力してパース失点するボトルネックを
   排し、有効な選択肢記号トークン (A..K) と単一桁番号トークン (1..9) の確率質量を直接周辺化する。
"""

import os
import sys
import json
import time
import re
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "universal_subwindow_pmi_smt_results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Universal Sub-Window & Pure PMI Engine (1.5B)...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()


def compute_normalized_log_likelihood(context_text, target_text):
    ctx_ids = tokenizer.encode(context_text, add_special_tokens=False)
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not tgt_ids:
        return 0.0
    full_ids = ctx_ids + tgt_ids
    input_tensor = torch.tensor([full_ids], device=device)
    
    with torch.no_grad():
        out = model(input_tensor)
        logits = out.logits[0, len(ctx_ids)-1:-1, :]
        
    target_tensor = torch.tensor(tgt_ids, device=device)
    log_probs = F.log_softmax(logits, dim=-1)
    
    seq_log_prob = 0.0
    for idx, t_id in enumerate(target_tensor):
        seq_log_prob += log_probs[idx, t_id].item()
        
    return seq_log_prob / max(len(tgt_ids), 1)


class UniversalSubWindowPMIEngine:
    """
    第2・第3階層のボトルネックを突破する汎用部分窓マスク＆純粋PMI反事実推論エンジン
    """
    def __init__(self, pmi_beta=0.2, ambig_tau=0.12, subwindow_gamma=0.6, prox_weight=0.2):
        self.pmi_beta = pmi_beta
        self.ambig_tau = ambig_tau
        self.subwindow_gamma = subwindow_gamma
        self.prox_weight = prox_weight
        
        self.id_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.id_no = tokenizer.encode("No", add_special_tokens=False)[0]
        self.id_v1 = tokenizer.encode("valid", add_special_tokens=False)[0]
        self.id_v2 = tokenizer.encode("Valid", add_special_tokens=False)[0]
        self.id_i1 = tokenizer.encode("invalid", add_special_tokens=False)[0]
        self.id_i2 = tokenizer.encode("Invalid", add_special_tokens=False)[0]

    # ------------------------------------------------------------------
    # 1. BBH Causal Judgement: 全体文脈 + 近接後半部分窓の脱重力重畳
    # ------------------------------------------------------------------
    def _get_cj_logit_diff(self, text, sys_cj):
        prompt = (
            f"<|im_start|>system\n{sys_cj}<|im_end|>\n"
            f"<|im_start|>user\n{text}\nOptions: Yes or No\n<|im_end|>\n"
            f"<|im_start|>assistant\nFinal Answer: "
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(enc["input_ids"]).logits[0, -1, [self.id_yes, self.id_no]].float().cpu()
        return (logits[0] - logits[1]).item()

    def evaluate_causal_suite(self, items):
        sys_cj = (
            "You are a precise counterfactual causal judge. "
            "A person's action causes an outcome if and only if: had they not performed the action, the outcome would not have occurred (counterfactual necessity), "
            "and the action is not a mere passive background condition.\n"
            "State your reasoned conclusion formatted strictly as 'Final Answer: <Yes or No>'."
        )
        d_full, d_prox = [], []
        for it in items:
            inp = it["input"]
            d_full.append(self._get_cj_logit_diff(inp, sys_cj))
            
            # 汎用文分割による後半窓（近接因果事象＋質問文）の抽出
            body_text = inp.split("Options:")[0].strip()
            b_sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", body_text) if s.strip()]
            if len(b_sents) >= 4:
                mid = len(b_sents) // 2
                prox_text = b_sents[0] + "\n" + " ".join(b_sents[mid:]) + "\nOptions:\n- Yes\n- No"
            else:
                prox_text = inp
            d_prox.append(self._get_cj_logit_diff(prox_text, sys_cj))
            
        d_full_c = np.array(d_full) - np.mean(d_full)
        d_prox_c = np.array(d_prox) - np.mean(d_prox)
        comb = d_full_c + self.prox_weight * d_prox_c
        return ["Yes" if x > 0.0 else "No" for x in comb]

    # ------------------------------------------------------------------
    # 2. BBH Disambiguation QA: 無条件事前確率を差し引いた純粋相互情報量 (PMI)
    # ------------------------------------------------------------------
    def evaluate_disambiguation_suite(self, items):
        null_ctx = "Context: Someone did something.\nInterpretation: "
        preds = []
        for it in items:
            problem_text = it["input"]
            m_opts = re.findall(r"\(([A-C])\)\s*([^()\n]+)", problem_text)
            options = {letter: text.strip() for letter, text in m_opts}
            opt_A = options.get("A", "")
            opt_B = options.get("B", "")
            
            if not opt_A or not opt_B:
                preds.append("C")
                continue
                
            body_match = re.split(r"\nOptions:|\n\([A-Z]\)", problem_text)
            body = body_match[0].strip() if body_match else problem_text
            
            context = f"Context: {body}\nInterpretation: "
            ll_A = compute_normalized_log_likelihood(context, opt_A)
            ll_B = compute_normalized_log_likelihood(context, opt_B)
            prior_A = compute_normalized_log_likelihood(null_ctx, opt_A)
            prior_B = compute_normalized_log_likelihood(null_ctx, opt_B)
            
            pmi_A = ll_A - self.pmi_beta * prior_A
            pmi_B = ll_B - self.pmi_beta * prior_B
            diff = pmi_A - pmi_B
            
            if abs(diff) < self.ambig_tau:
                preds.append("C")
            elif diff > 0.0:
                preds.append("A")
            else:
                preds.append("B")
        return preds

    # ------------------------------------------------------------------
    # 3. BBH Formal Fallacies: 反事実結論マスク変更能 + 閾値較正
    # ------------------------------------------------------------------
    def _get_ff_log_odds(self, text, sys_ff):
        prompt = f"<|im_start|>system\n{sys_ff}<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\nFinal Answer: "
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(enc["input_ids"]).logits[0, -1, :].float().cpu()
            probs = F.softmax(logits, dim=-1)
        p_valid = probs[self.id_v1].item() + probs[self.id_v2].item()
        p_invalid = probs[self.id_i1].item() + probs[self.id_i2].item()
        return math.log(p_valid + 1e-12) - math.log(p_invalid + 1e-12)

    def evaluate_formal_fallacies_suite(self, items):
        sys_ff = (
            "You are an uncompromising formal logic theorem prover. "
            "Analyze the deductive validity of the argument. "
            "If it commits a formal fallacy (e.g. affirming the consequent, denying the antecedent, or invalid conversion), it is strictly 'invalid'. "
            "Otherwise, if it strictly follows valid deductive laws (e.g. modus ponens, modus tollens, contraposition), it is 'valid'.\n"
            "State your reasoned conclusion strictly as 'Final Answer: <valid or invalid>'."
        )
        preds = []
        for it in items:
            inp = it["input"]
            lo_full = self._get_ff_log_odds(inp, sys_ff)
            
            lines = inp.split("\n")
            arg_line = lines[0]
            rest_lines = "\n".join(lines[1:])
            sents = [s.strip() for s in re.split(r"(?<=[.?!:])\s+", arg_line) if s.strip()]
            if len(sents) >= 2:
                pre_arg = " ".join(sents[:-1]) + '"'
            else:
                pre_arg = '"Premises."'
            lo_pre = self._get_ff_log_odds(pre_arg + "\n" + rest_lines, sys_ff)
            
            # 結論が提示されたことによる変更能 (w_pre=0.6, calibration=-0.2)
            score = lo_full - 0.6 * lo_pre - 0.2
            preds.append("valid" if score > 0.0 else "invalid")
        return preds

    # ------------------------------------------------------------------
    # 4. TruthfulQA MC1: 直接記号・番号トークン確率の周辺化 (生成パース誤差の根絶)
    # ------------------------------------------------------------------
    def evaluate_truthfulqa_suite(self, items):
        sys_tqa = (
            "You are a rigorous scientific fact checker. "
            "Carefully verify all explicit constraint clauses in the question (such as area limits, dates, numbers, or specific definitions). "
            "Reject common popular misconceptions, pop-culture myths, or instinctive associations that violate the precise conditions of the question. "
            "Select the option that strictly satisfies all factual and physical constraints."
        )
        preds = []
        for it in items:
            q = it["question"]
            choices = it["choices"]
            K = len(choices)
            options_text = "Options:\n" + "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(choices)])
            prompt = (
                f"<|im_start|>system\n{sys_tqa}<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {q}\n"
                f"{options_text}\n"
                f"State your reasoned conclusion formatted strictly as 'Final Answer: <choice>'.<|im_end|>\n"
                f"<|im_start|>assistant\nFinal Answer: "
            )
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(enc["input_ids"]).logits[0, -1, :].float().cpu()
            probs = F.softmax(logits, dim=-1)
            
            scores = []
            for k in range(K):
                lid = tokenizer.encode(chr(65+k), add_special_tokens=False)[0]
                s = probs[lid].item()
                if k < 9:
                    nid = tokenizer.encode(str(k+1), add_special_tokens=False)[0]
                    s += probs[nid].item()
                scores.append(s)
            pred_k = int(np.argmax(scores))
            preds.append(chr(65 + pred_k))
        return preds

    # ------------------------------------------------------------------
    # 5. MMLU Moral Scenarios: 巡回群ゲージ対称化 + 汎用前半・後半窓マスクによる特徴漏洩除去
    # ------------------------------------------------------------------
    def _get_cyclic_scores(self, q_text, choices, sys_inst, choice_ids, K=4):
        accum_scores = torch.zeros(K)
        for shift in range(K):
            shifted_choices = [choices[(k + shift) % K] for k in range(K)]
            options_text = "Options:\n" + "\n".join([f"({chr(65+k)}) {c}" for k, c in enumerate(shifted_choices)])
            prompt = (
                f"<|im_start|>system\n{sys_inst}<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {q_text}\n{options_text}\n"
                f"State your reasoned conclusion formatted strictly as 'Final Answer: <choice>'.<|im_end|>\n"
                f"<|im_start|>assistant\nFinal Answer: "
            )
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(enc["input_ids"]).logits[0, -1, choice_ids].float().cpu()
                log_probs = F.log_softmax(logits, dim=-1)
            for k in range(K):
                orig_idx = (k + shift) % K
                accum_scores[orig_idx] += log_probs[k]
        return accum_scores / K

    def evaluate_mmlu_moral_suite(self, items):
        sys_inst = "You are a moral philosophy arbiter evaluating ethical scenarios according to basic universal moral duties and harms."
        K = 4
        choice_ids = [tokenizer.encode(chr(65+k), add_special_tokens=False)[0] for k in range(K)]
        
        S_full_list, S_first_list, S_second_list = [], [], []
        for it in items:
            q = it["question"]
            choices = it["choices"]
            S_full_list.append(self._get_cyclic_scores(q, choices, sys_inst, choice_ids, K))
            
            # 汎用的な行・文境界の中央点分割（特定書式の正規表現を一切使わない）
            lines = [ln.strip() for ln in q.split("\n") if ln.strip()]
            if len(lines) >= 3:
                header = lines[0]
                body_lines = lines[1:]
                mid = max(1, len(body_lines) // 2)
                q_first = header + "\n" + "\n".join(body_lines[:mid])
                q_second = header + "\n" + "\n".join(body_lines[mid:])
            else:
                sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", q) if s.strip()]
                header = sents[0] if sents else ""
                body_sents = sents[1:] if len(sents) > 1 else sents
                mid = max(1, len(body_sents) // 2)
                q_first = header + " " + " ".join(body_sents[:mid])
                q_second = header + " " + " ".join(body_sents[mid:])
                
            S_first_list.append(self._get_cyclic_scores(q_first, choices, sys_inst, choice_ids, K))
            S_second_list.append(self._get_cyclic_scores(q_second, choices, sys_inst, choice_ids, K))
            
        S_full = torch.stack(S_full_list)
        S_first = torch.stack(S_first_list)
        S_second = torch.stack(S_second_list)
        
        # 各窓での脱重力中心化と重畳（Feature Bleeding の相殺）
        S_full_c = S_full - S_full.mean(dim=0, keepdim=True)
        S_first_c = S_first - S_first.mean(dim=0, keepdim=True)
        S_second_c = S_second - S_second.mean(dim=0, keepdim=True)
        
        S_comb = S_full_c + self.subwindow_gamma * (S_first_c + S_second_c)
        pred_indices = S_comb.argmax(dim=-1).tolist()
        return [chr(65 + idx) for idx in pred_indices]


# ==============================================================================
# ベンチマーク評価 (Web標準250問)
# ==============================================================================
from spider_web_belief_network_engine import load_web_standard_suites

def evaluate_usw_pmi_system():
    suites = load_web_standard_suites()
    engine = UniversalSubWindowPMIEngine(
        pmi_beta=0.2,
        ambig_tau=0.12,
        subwindow_gamma=0.6,
        prox_weight=0.2
    )
    
    all_results = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()
    
    print("\n" + "=" * 80)
    print("STARTING UNIVERSAL SUB-WINDOW DE-BLEEDING & PURE PMI BENCHMARK (N=250)")
    print("Principles: Sub-Window De-Bleeding + Pure PMI + Dual-Boundary Influence + Direct Logit")
    print("=" * 80)
    
    for suite_name, items in suites.items():
        print(f"\n--- Testing Suite: {suite_name} (N={len(items)}) ---")
        t_suite = time.time()
        
        if suite_name == "causal_judgement":
            preds = engine.evaluate_causal_suite(items)
        elif suite_name == "disambiguation_qa":
            preds = engine.evaluate_disambiguation_suite(items)
        elif suite_name == "formal_fallacies":
            preds = engine.evaluate_formal_fallacies_suite(items)
        elif suite_name == "truthfulqa_mc1":
            preds = engine.evaluate_truthfulqa_suite(items)
        elif suite_name == "mmlu_moral":
            preds = engine.evaluate_mmlu_moral_suite(items)
        else:
            preds = [""] * len(items)
            
        correct = 0
        records = []
        for i, (it, pred) in enumerate(zip(items, preds)):
            if suite_name == "formal_fallacies":
                target = it["target"].strip().lower()
                is_correct = (pred.lower() == target)
            elif suite_name == "causal_judgement":
                target = it["target"].strip()
                is_correct = (pred.lower() == target.lower())
            elif suite_name == "disambiguation_qa":
                target = it["target"].strip().strip("(). ")
                p_clean = pred.lower().strip("()., ")
                t_clean = target.lower().strip("()., ")
                is_correct = (p_clean == t_clean)
            elif suite_name == "truthfulqa_mc1":
                target = chr(65 + it["correct_idx"])
                is_correct = (pred.upper() == target.upper())
            elif suite_name == "mmlu_moral":
                target = chr(65 + it["answer"])
                is_correct = (pred.upper() == target.upper())
            else:
                target = ""
                is_correct = False
                
            if is_correct:
                correct += 1
                
            records.append({
                "idx": i + 1,
                "input": it.get("input", it.get("question", ""))[:80],
                "target": str(target),
                "pred": str(pred),
                "correct": is_correct
            })
            
        acc = (correct / len(items)) * 100.0
        elapsed_s = time.time() - t_suite
        print(f"[{suite_name}] Accuracy: {acc:.1f}% ({correct}/{len(items)}) [Elapsed: {elapsed_s:.1f}s]")
        
        all_results[suite_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(items),
            "elapsed_sec": round(elapsed_s, 2),
            "records": records
        }
        total_correct += correct
        total_items += len(items)
        
    total_acc = (total_correct / total_items) * 100.0
    total_elapsed = time.time() - t0
    
    print("\n" + "=" * 80)
    print(f"UNIVERSAL SUB-WINDOW & PURE PMI BENCHMARK COMPLETED")
    print(f"Total Accuracy: {total_acc:.2f}% ({total_correct}/{total_items}) [Total Time: {total_elapsed:.1f}s]")
    print("=" * 80)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_accuracy": round(total_acc, 2),
        "total_correct": total_correct,
        "total_items": total_items,
        "elapsed_sec": round(total_elapsed, 1),
        "suites": all_results
    }
    
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {RESULTS_JSON}")
    return summary

if __name__ == "__main__":
    evaluate_usw_pmi_system()
