"""
Formal Logic Verifier (Rule-Based Deductive Engine)
形式論理検証器（演繹ルールベース）の実装と検証
========================================================================
装飾語・トラップ（"Here comes a perfectly valid argument" 等）を排し、
前提と結論の論理構造（含意、否定、対偶、三段論法、後件肯定の誤謬等）を
決定論的ルールで判定する。
"""

import sys
import re
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Formal Parsing...")
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


class FormalDeductiveVerifier:
    """
    決定論的演繹論理検証器 (Deterministic Deductive Engine)
    """
    @staticmethod
    def verify_deduction(premises, conclusion):
        """
        premises: list of tuples, e.g., [("A", "implies", "not B")]
        conclusion: tuple, e.g., ("not B", "implies", "A")
        
        Returns:
            status: "valid", "invalid", or "inconclusive"
            reason: 説明
        """
        if not premises or not conclusion:
            return "inconclusive", "Missing premises or conclusion"
            
        # パターン1: 単一条件命題からの推論 (Premise: P -> Q)
        if len(premises) == 1:
            p_ant, p_op, p_con = premises[0]
            c_ant, c_op, c_con = conclusion
            
            p_ant_clean = p_ant.strip().lower()
            p_con_clean = p_con.strip().lower()
            c_ant_clean = c_ant.strip().lower()
            c_con_clean = c_con.strip().lower()
            
            # 対偶 (Contrapositive): P -> Q <=> not Q -> not P (Valid)
            is_ant_neg = c_ant_clean.startswith("not ") or "no " in c_ant_clean
            is_con_neg = c_con_clean.startswith("not ") or "no " in c_con_clean
            
            # 逆 (Converse Fallacy / Affirming Consequent): P -> Q から Q -> P を導出 (Invalid)
            if (p_con_clean == c_ant_clean and p_ant_clean == c_con_clean) or \
               (p_con_clean.replace("not ", "") == c_ant_clean.replace("not ", "") and p_ant_clean.replace("not ", "") == c_con_clean.replace("not ", "")):
                # 正しい対偶か、誤った逆かチェック
                p_ant_neg = p_ant_clean.startswith("not ")
                p_con_neg = p_con_clean.startswith("not ")
                c_ant_neg = c_ant_clean.startswith("not ")
                c_con_neg = c_con_clean.startswith("not ")
                
                # 対偶: c_ant が p_con の否定 かつ c_con が p_ant の否定
                if c_ant_neg != p_con_neg and c_con_neg != p_ant_neg:
                    return "valid", "Contrapositive deduction (Valid)"
                else:
                    return "invalid", "Converse fallacy / Affirming the consequent (Invalid)"
                    
            # 同一命題
            if p_ant_clean == c_ant_clean and p_con_clean == c_con_clean:
                return "valid", "Identity deduction (Valid)"
                
            # 裏 (Inverse Fallacy / Denying Antecedent): not P -> not Q (Invalid)
            if p_ant_clean.replace("not ", "") == c_ant_clean.replace("not ", "") and \
               p_con_clean.replace("not ", "") == c_con_clean.replace("not ", ""):
                return "invalid", "Inverse fallacy / Denying the antecedent (Invalid)"
                
        # パターン2: 2前提の三段論法 (P -> Q, Q -> R => P -> R)
        if len(premises) == 2:
            p1_ant, _, p1_con = premises[0]
            p2_ant, _, p2_con = premises[1]
            c_ant, _, c_con = conclusion
            
            p1_ant = p1_ant.strip().lower()
            p1_con = p1_con.strip().lower()
            p2_ant = p2_ant.strip().lower()
            p2_con = p2_con.strip().lower()
            c_ant = c_ant.strip().lower()
            c_con = c_con.strip().lower()
            
            # 連鎖: P1 -> P2 => C
            # P1: A -> B, P2: B -> C => A -> C (Valid)
            if p1_con == p2_ant and c_ant == p1_ant and c_con == p2_con:
                return "valid", "Hypothetical syllogism (Valid)"
            # P1: B -> C, P2: A -> B => A -> C (Valid)
            if p2_con == p1_ant and c_ant == p2_ant and c_con == p1_con:
                return "valid", "Hypothetical syllogism (Valid)"
                
        return "inconclusive", "Complex or unmatched logic structure"


def parse_and_verify(text, target=None):
    """
    LLMに論理骨格（前提と結論）を抽出させ、検証器で厳密に判定
    """
    prompt = (
        f"<|im_start|>system\nYou are a formal logic syntax extractor. "
        f"Extract the formal argument structure from the text into pure logical components. "
        f"Do NOT be fooled by phrases like 'Here comes a perfectly valid argument' or real-world intuition.\n"
        f"Output JSON strictly with keys:\n"
        f"- 'premises': list of strings e.g. [\"A implies B\"]\n"
        f"- 'conclusion': string e.g. \"C implies D\"\n"
        f"- 'standard_form': string<|im_end|>\n"
        f"<|im_start|>user\nArgument text: {text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n```json\n"
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=150, do_sample=False)
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # JSONパース
    json_match = re.search(r"(\{.*\})", resp, re.DOTALL)
    parsed = None
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
        except:
            pass
            
    # ルール検証器へのマッピング
    verdict = "inconclusive"
    reason = "Fallback"
    
    if parsed and "premises" in parsed and "conclusion" in parsed:
        prem_tuples = []
        for p in parsed["premises"]:
            if "implies" in p:
                parts = p.split("implies")
                prem_tuples.append((parts[0].strip(), "implies", parts[1].strip()))
            elif "->" in p:
                parts = p.split("->")
                prem_tuples.append((parts[0].strip(), "implies", parts[1].strip()))
                
        conc = parsed["conclusion"]
        conc_tuple = None
        if "implies" in conc:
            parts = conc.split("implies")
            conc_tuple = (parts[0].strip(), "implies", parts[1].strip())
        elif "->" in conc:
            parts = conc.split("->")
            conc_tuple = (parts[0].strip(), "implies", parts[1].strip())
            
        if prem_tuples and conc_tuple:
            verdict, reason = FormalDeductiveVerifier.verify_deduction(prem_tuples, conc_tuple)
            
    # 決定論的検証器が conclusive でない場合：
    # ユーザー指示: 「矛盾があることを認知したうえで、確定した情報が出せない/誤りの可能性が高いと判断する」
    if verdict == "inconclusive":
        # 形式論理検証器で演繹が証明されない場合、誤謬（invalid）の可能性が高いと保守的に判断
        final_decision = "invalid" 
    else:
        final_decision = verdict
        
    return final_decision, verdict, reason, parsed


def test_formal_fallacies_suite(num_samples=50):
    print("=" * 80)
    print(f"TESTING FORMAL LOGIC VERIFIER ON BBH FORMAL FALLACIES (N={num_samples})")
    print("=" * 80)
    
    ds = load_dataset('lukaemon/bbh', 'formal_fallacies', split='test')
    items = [ds[i] for i in range(num_samples)]
    
    correct = 0
    conclusive_count = 0
    t0 = time.time()
    
    for i, it in enumerate(items):
        inp = it['input']
        target = it['target'].strip().lower() # 'valid' or 'invalid'
        
        pred, rule_verdict, reason, parsed = parse_and_verify(inp, target)
        
        is_correct = (pred == target)
        if is_correct:
            correct += 1
        if rule_verdict != "inconclusive":
            conclusive_count += 1
            
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            acc = (correct / (i + 1)) * 100.0
            print(f"  Item {i+1:02d}/{num_samples} | Current Acc: {acc:.1f}% | Rule Conclusive: {conclusive_count}/{i+1}")
            
    elapsed = time.time() - t0
    final_acc = (correct / num_samples) * 100.0
    print("\n" + "=" * 80)
    print(f"FORMAL LOGIC VERIFIER EVALUATION COMPLETED in {elapsed:.1f}s")
    print(f"Overall Accuracy: {final_acc:.2f}% ({correct}/{num_samples})")
    print(f"Rule Conclusive Rate: {conclusive_count}/{num_samples} ({(conclusive_count/num_samples)*100.0:.1f}%)")
    print("=" * 80)
    
    return {
        "accuracy": final_acc,
        "correct": correct,
        "total": num_samples,
        "conclusive_count": conclusive_count,
        "elapsed_sec": round(elapsed, 2)
    }

if __name__ == "__main__":
    test_formal_fallacies_suite(num_samples=50)
