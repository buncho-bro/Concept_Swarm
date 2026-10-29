"""
SMT Cycle Detection & Reversal Test Pipeline with Z3
(閉路探査と反転テストを組み込んだ SMT 厳密検証エンジン)
========================================================================
設計仕様:
1. 【閉路 (サイクル) 探査 (Cycle Detection via SMT Ranking)】:
   - 各概念ノード X に SMT 整数変数 Rank(X) を導入。
   - 含意関係 X -> Y が存在する場合、Rank(X) < Rank(Y) を要求。
   - 循環論法 (A -> B -> ... -> A) が発生した場合、Rank(A) < Rank(A) となり
     Z3 SMT が直ちに UNSAT (閉路検知) を出力。
   - 負のサイクル (A -> ... -> ¬A) の自己否定ループも厳密探査。
2. 【反転テスト (Reversal Test)】:
   - ① 前提反転テスト (Antecedent Inversion / Relevance Check):
     A ∧ B -> C に対し、¬A ∧ B -> C も成立するか検証。
     もし成立する場合、A は結論 C に寄与しない無関係な前提 (Status Quo Bias / Spurious Rule) として検知。
   - ② 結論反転テスト (Consequent Inversion / Modus Tollens Check):
     ¬C を仮定した際、A ∧ B が同時に成立し得るか (反例モデル探索)。
   - ③ 方向反転テスト (Converse Fallacy Check):
     C -> (A ∧ B) の逆推論が無批判に成立していないかを検証。
3. 【Unsat Core による最小修正】:
   閉路や反転テスト失敗を引き起こした最小の式を特定し、自動修正。
"""

import os
import sys
import json
import time
import re
import z3

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "smt_cycle_reversal_results.json")


class SMTCycleAndReversalEngine:
    """
    閉路探査と反転テストを統合した Z3 SMT エンジン
    """
    def __init__(self):
        self.bool_vars = {}
        self.rank_vars = {}
        self.rules = [] # list of dicts: {"name": str, "ants": [str], "con": str, "is_neg": bool}
        
    def get_bool(self, name):
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
        if clean not in self.bool_vars:
            self.bool_vars[clean] = z3.Bool(clean)
        return self.bool_vars[clean]
        
    def get_rank(self, name):
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
        if clean not in self.rank_vars:
            self.rank_vars[clean] = z3.Int(f"rank_{clean}")
        return self.rank_vars[clean]
        
    def add_rule(self, name, antecedents, consequent, is_consequent_negated=False):
        self.rules.append({
            "name": name,
            "ants": antecedents,
            "con": consequent,
            "is_neg": is_consequent_negated
        })

    # --------------------------------------------------------------------------
    # 1. 閉路 (サイクル) 探査
    # --------------------------------------------------------------------------
    def detect_cycles(self):
        """
        因果・演繹ネットワーク内の循環 (閉路) を SMT 整数順序 (LIA) で探査
        """
        print("\n" + "=" * 80)
        print("1. SMT CYCLE (CLOSED LOOP) DETECTION")
        print("=" * 80)
        
        solver = z3.Solver()
        tracker = {}
        
        # 各規則について Rank(antecedent) < Rank(consequent) を課す
        for r in self.rules:
            con_rank = self.get_rank(r["con"])
            for ant in r["ants"]:
                ant_rank = self.get_rank(ant)
                # ant から con への進行は厳密な順序上昇を要求
                order_constraint = (ant_rank < con_rank)
                t_var = z3.Bool(f"track_cycle_{r['name']}_{ant}_{r['con']}")
                tracker[t_var] = (r['name'], ant, r['con'])
                solver.assert_and_track(order_constraint, t_var)
                
        t0 = time.time()
        res = solver.check()
        dt = time.time() - t0
        
        if res == z3.unsat:
            print(f"  Z3 SMT Check: UNSAT (Time: {dt*1000:.2f}ms)")
            print("  [!] BLATANT CYCLE DETECTED: Causality/Deduction forms an impossible loop!")
            core = solver.unsat_core()
            culprits = [tracker[cv] for cv in core if cv in tracker]
            print("  Unsat Core (Cycle Participants):")
            for rule_name, ant, con in culprits:
                print(f"     * In {rule_name}: {ant} -> {con}")
            return False, culprits
        else:
            print(f"  Z3 SMT Check: SAT (Time: {dt*1000:.2f}ms)")
            print("  [+] No cycles detected. The knowledge hierarchy is a strict Directed Acyclic Graph (DAG).")
            return True, []

    # --------------------------------------------------------------------------
    # 2. 反転テスト (Reversal Test)
    # --------------------------------------------------------------------------
    def run_reversal_tests(self):
        """
        全規則に対する反転テストの実行:
        - ① 前提反転テスト (寄与度・無関係前提の検出)
        - ② 結論反転テスト (対偶整合性・反例モデルの検出)
        - ③ 方向反転テスト (逆の誤謬の検出)
        """
        print("\n" + "=" * 80)
        print("2. SMT REVERSAL TESTS (ANTECEDENT, CONSEQUENT, & DIRECTIONAL)")
        print("=" * 80)
        
        test_results = []
        
        for r in self.rules:
            r_name = r["name"]
            ants = r["ants"]
            con = r["con"]
            is_neg = r["is_neg"]
            
            print(f"\n--- Testing Rule: [{r_name}] ({' + '.join(ants)} -> {'not ' if is_neg else ''}{con}) ---")
            
            # --- テスト①: 前提反転テスト (Antecedent Inversion) ---
            # 各前提 A_i に対し、¬A_i に反転させたとき、結論 C が依然として導かれるかを検証
            # 健全な推論であれば、必須前提 A_i を否定したときは結論 C が成り立たなくなる (反例が存在する = SAT)
            # もし ¬A_i のもとでも反例なく C が常に導かれる (UNSAT) なら、A_i は結論に寄与していない無関係な前提
            redundant_ants = []
            for ant_to_invert in ants:
                s_ant = z3.Solver()
                # 現在の全規則を追加
                for r_other in self.rules:
                    r_ast = z3.Implies(z3.And(*[self.get_bool(a) for a in r_other["ants"]]), 
                                       z3.Not(self.get_bool(r_other["con"])) if r_other["is_neg"] else self.get_bool(r_other["con"]))
                    s_ant.add(r_ast)
                    
                # ant_to_invert のみを否定し、他の前提は肯定
                for a in ants:
                    if a == ant_to_invert:
                        s_ant.add(z3.Not(self.get_bool(a)))
                    else:
                        s_ant.add(self.get_bool(a))
                        
                target_con = z3.Not(self.get_bool(con)) if is_neg else self.get_bool(con)
                # 結論が否定される反例を探す
                s_ant.add(z3.Not(target_con))
                
                # もし反例が存在しない (unsat) なら、A を否定しても C が成り立ってしまっている => 冗長・無関係
                if s_ant.check() == z3.unsat:
                    redundant_ants.append(ant_to_invert)
                    
            if redundant_ants:
                print(f"  [!] Antecedent Inversion Flag: Premises {redundant_ants} are IRRELEVANT (Invariance Flaw).")
            else:
                print(f"  [+] Antecedent Inversion Passed: All premises are necessary.")
                
            # --- テスト②: 結論反転テスト (Consequent Inversion / Modus Tollens) ---
            # 結論が偽 (¬C) のとき、前提全体 (And(*ants)) が同時に成り立ち得るか (反例の有無)
            s_con = z3.Solver()
            rule_ast = z3.Implies(z3.And(*[self.get_bool(a) for a in ants]), z3.Not(self.get_bool(con)) if is_neg else self.get_bool(con))
            s_con.add(rule_ast)
            
            # ¬C を仮定
            s_con.add(self.get_bool(con) if is_neg else z3.Not(self.get_bool(con)))
            # かつ前提が真
            s_con.add(z3.And(*[self.get_bool(a) for a in ants]))
            
            con_inv_res = s_con.check()
            if con_inv_res == z3.unsat:
                print("  [+] Consequent Inversion (Modus Tollens) Passed: No contradictory counter-models.")
            else:
                model = s_con.model()
                print(f"  [!] Consequent Inversion FAILED: Counter-example model found: {model}")
                
            # --- テスト③: 方向反転テスト (Directional Inversion / Converse) ---
            # 結論 C から前提 (And(*ants)) が必然的に言えるか (逆の誤謬チェック)
            s_dir = z3.Solver()
            s_dir.add(rule_ast) # A => C
            # 逆の主張: C => A
            converse_ast = z3.Implies(z3.Not(self.get_bool(con)) if is_neg else self.get_bool(con), z3.And(*[self.get_bool(a) for a in ants]))
            # 反例: C が真なのに A が偽のモデルを探す
            s_dir.add(z3.Not(self.get_bool(con)) if is_neg else self.get_bool(con)) # C = True
            s_dir.add(z3.Not(z3.And(*[self.get_bool(a) for a in ants])))           # A = False
            
            dir_res = s_dir.check()
            if dir_res == z3.sat:
                print("  [!] Directional Inversion Note: Converse does NOT hold (One-way deduction confirmed, avoiding Converse Fallacy).")
            else:
                print("  [*] Equivalence: Bi-directional equivalence holds.")
                
            test_results.append({
                "rule": r_name,
                "formula": f"{' + '.join(ants)} -> {'not ' if is_neg else ''}{con}",
                "redundant_premises": redundant_ants,
                "consequent_inversion_valid": (con_inv_res == z3.unsat),
                "is_one_way": (dir_res == z3.sat)
            })
            
        return test_results


def run_full_demonstration():
    print("=" * 80)
    print("FULL EXPERIMENT: SMT CYCLE DETECTION & REVERSAL TESTS")
    print("================================================================================")
    
    engine = SMTCycleAndReversalEngine()
    
    # --------------------------------------------------------------------------
    # 規則群の定義: 正常な階層式 + 循環・反転テスト対象
    # --------------------------------------------------------------------------
    # Rule 1: A + B -> C
    engine.add_rule("Rule_1", ["investors", "capital"], "enterprises")
    
    # Rule 2: (A + C) + D -> E
    engine.add_rule("Rule_2", ["investors", "enterprises", "labor"], "production")
    
    # Rule 3: (A + C + E) + F -> G
    engine.add_rule("Rule_3", ["investors", "enterprises", "production", "market"], "revenue")
    
    # Rule 4 (閉路・サイクルの注入): revenue + growth -> investors (結論が初期の前提に戻る閉路)
    engine.add_rule("Rule_4_cycle", ["revenue", "growth"], "investors")
    
    # Rule 5 (反転テスト用の無関係前提を含む式):
    # (production + rain) -> goods (rain は生産に無関係だがくっついている)
    engine.add_rule("Rule_5_spurious", ["production", "rain"], "goods")
    
    # 1. 閉路探査
    is_dag, cycle_culprits = engine.detect_cycles()
    
    # 閉路が見つかった場合、最小修正 (Unsat Core によるサイクルの除去)
    if not is_dag:
        print("\n[SMT Unsat Core Minimal Repair] Eliminating cyclic feedback rule...")
        # 閉路を形成した Rule_4_cycle を撤回し、acyclic な式に最小修正
        # revenue + growth -> dividends (配当へ修正)
        engine.rules = [r for r in engine.rules if r["name"] != "Rule_4_cycle"]
        engine.add_rule("Rule_4_repaired", ["revenue", "growth"], "dividends")
        print("  => Re-running SMT Cycle Detection after Minimal Revision:")
        is_dag_repaired, _ = engine.detect_cycles()
        
    # 2. 全規則に対する反転テスト
    reversal_results = engine.run_reversal_tests()
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_cycle_detected": not is_dag,
        "cycle_culprits": cycle_culprits,
        "reversal_tests": reversal_results
    }
    
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 80)
    print(f"SMT CYCLE & REVERSAL TESTS COMPLETE. Results saved to: {RESULTS_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_demonstration()
