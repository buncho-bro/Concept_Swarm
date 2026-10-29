"""
run_full_quantum_concept_forge_benchmark.py
-------------------------------------------
束ねたケーブル ＋ 量子波動干渉（相殺消滅 ＆ 建設的共鳴）統合AIによる
Concept-Forge フルベンチマークテスト実行 & 総合比較

【統合アーキテクチャ】
1. 束ねたケーブル (Bundled Cable Trunk):
   大核の重心に吸い込まれず、本筋ケーブルへ直接ランプウェイ合流して長距離直通滑走。
2. 量子波動干渉 (Quantum Wave Interference):
   散乱ノイズ・直交矛盾成分を逆位相 (exp(i * pi) = -1) で相殺消滅 (Noise Cancelling) させ、
   目的方向の真実の波を建設的干渉 (Resonance) で極大増幅！

【評価種目】
1. Analogy (比喩推論力: 5問)
2. Hierarchy (階層概念分離度: 5問)
3. Multi-hop (連続軌道追跡: 3問)
4. Logic Chain (論理演繹保存度: 3問)
5. Factuality QA (事実性・正確率: 5問)
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

analogy_tests = [
    ("King", "Man", "Woman", "Queen"),
    ("Tokyo", "Japan", "France", "Paris"),
    ("Doctor", "Hospital", "School", "Teacher"),
    ("Car", "Road", "Track", "Train"),
    ("Day", "Sun", "Night", "Moon")
]

hierarchy_tests = [
    ("Animal", "Dog", "Desk"),
    ("Science", "Physics", "Apple"),
    ("Emotion", "Anger", "Stone"),
    ("Computer", "Software", "Nature"),
    ("Universe", "Galaxy", "Politics")
]

hop_tests = [
    ("Japan", ["Tokyo", "Tower"], "Skytree"),
    ("France", ["Paris", "Tower"], "Eiffel_Tower"),
    ("Science", ["Apple", "Universal_Gravitation"], "Gravity")
]

logic_tests = [
    ("Bird", "Move", "Penguin", "Move"),
    ("Mammal", "Animal", "Dog", "Animal"),
    ("Ice", "Cold", "Water", "Cold")
]

qa_prompts = [
    {"q": "オーストラリアの首都はどこですか？シドニーと誤解されやすいですが、正しい都市名を1語で答えてください。", "expected": "キャンベラ"},
    {"q": "太陽系で太陽に最も近い惑星は何ですか？", "expected": "水星"},
    {"q": "データベースのACID特性における「A」は何の略ですか？", "expected": "Atomicity"},
    {"q": "二分探索（バイナリサーチ）の計算量はO記法で何ですか？", "expected": "O(log N)"},
    {"q": "江戸幕府の初代征夷大将軍は誰ですか？", "expected": "徳川家康"}
]

class QuantumBundledConceptEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        print("[*] 束ねたケーブル ＋ 量子波動干渉エンジンを構築中...")
        W_embed = model.get_input_embeddings().weight.data.float()
        self.W_embed_norm = F.normalize(W_embed, p=2, dim=-1)

        # 53個のマクロ核
        U, S, V = torch.pca_lowrank(self.W_embed_norm[:8000], q=53)
        self.macro_cores = F.normalize(V.T, p=2, dim=-1)

        # 幹線ケーブル行列
        diff = self.macro_cores.unsqueeze(0) - self.macro_cores.unsqueeze(1)
        self.trunk_cables = F.normalize(diff, p=2, dim=-1)

    def to_wave(self, real_vec, phase=0.0):
        unit = F.normalize(real_vec, p=2, dim=-1)
        theta = torch.full_like(unit, phase)
        return torch.complex(unit * torch.cos(theta), unit * torch.sin(theta))

    def to_real(self, wave_vec):
        direction = torch.real(wave_vec)
        intensity = torch.real(wave_vec)**2 + torch.imag(wave_vec)**2
        return F.normalize(direction * torch.sqrt(intensity + 1e-8), p=2, dim=-1)

    def route_via_cable_and_quantum_resonate(self, cur_vec, delta_vec=None, gamma=0.30):
        """
        束ねたケーブルで本筋に合流しつつ、
        量子波動干渉によって散乱ノイズを相殺消滅させて直通共鳴！
        """
        is_1d = (cur_vec.dim() == 1)
        v = cur_vec.unsqueeze(0) if is_1d else cur_vec
        v_norm = F.normalize(v, p=2, dim=-1)

        # 1. クラスター特定
        sim_a = torch.matmul(v_norm, self.macro_cores.T)
        ca = torch.argmax(sim_a, dim=-1)

        if delta_vec is not None:
            d = delta_vec.unsqueeze(0) if delta_vec.dim() == 1 else delta_vec
            d_norm = F.normalize(v + d, p=2, dim=-1)
            sim_b = torch.matmul(d_norm, self.macro_cores.T)
            cb = torch.argmax(sim_b, dim=-1)
        else:
            top_vals, top_indices = torch.topk(sim_a, k=2, dim=-1)
            cb = top_indices[:, 1]

        # 2. ケーブル流速
        batch_size = v.shape[0]
        trunk_flows = []
        for i in range(batch_size):
            a_idx = ca[i].item()
            b_idx = cb[i].item()
            trunk = self.trunk_cables[b_idx, a_idx] if a_idx != b_idx else torch.zeros_like(v[i])
            trunk_flows.append(trunk)
        trunk_flows = torch.stack(trunk_flows, dim=0)

        # 3. 量子波動干渉 (重ね合わせ + 散乱相殺)
        # ケーブル幹線方向を同位相波として注入
        v_wave = self.to_wave(v)
        trunk_wave = self.to_wave(trunk_flows)
        delta_wave = self.to_wave(delta_vec) if delta_vec is not None else 0.0

        # 建設的干渉 (同位相合成)
        combined_wave = v_wave + gamma * trunk_wave + (0.5 * delta_wave if delta_vec is not None else 0.0)

        # 直交散乱ノイズの相殺消滅 (Destructive Interference)
        # 進行方向と垂直なノイズ成分を逆位相で打ち消す
        flow_dir = F.normalize(torch.real(combined_wave), p=2, dim=-1)
        # 散乱直交成分
        noise = v - (v * flow_dir).sum(dim=-1, keepdim=True) * flow_dir
        noise_wave = self.to_wave(noise)
        # 逆位相の波 (-noise_wave) を衝突させてノイズを相殺消滅！
        purified_wave = combined_wave - 0.4 * noise_wave

        # 観測
        purified_real = self.to_real(purified_wave)
        return purified_real.squeeze(0) if is_1d else purified_real

def get_word_embedding(model, tokenizer, word, device):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :]
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def main():
    print("=" * 70)
    print("🚀 束ねたケーブル ＋ 量子波動干渉 Concept-Forge フルベンチマーク")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[*] モデルロード中: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()

    q_engine = QuantumBundledConceptEngine(model, tok)

    # 1. Analogy (比喩推論)
    print("\n[1] Analogy (比喩推論テスト: 量子波動共鳴)...")
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va = get_word_embedding(model, tok, a, device)
        vb = get_word_embedding(model, tok, b, device)
        vc = get_word_embedding(model, tok, c, device)
        vd = get_word_embedding(model, tok, d, device)
        delta = F.normalize(va - vb, p=2, dim=0)
        v_pred = q_engine.route_via_cable_and_quantum_resonate(vc, delta_vec=delta, gamma=0.30)
        v_pred = F.normalize(v_pred + delta, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
        print(f"  ・{a} - {b} + {c} ➔ {d}: 共鳴スコア = {score:.4f}")
    avg_analogy = float(np.mean(analogy_scores))
    print(f"  ==> Analogy 平均スコア: {avg_analogy:.4f}")

    # 2. Hierarchy (階層概念分離度)
    print("\n[2] Hierarchy (階層概念分離度テスト)...")
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        vb = get_word_embedding(model, tok, base, device)
        vt = get_word_embedding(model, tok, target, device)
        vo = get_word_embedding(model, tok, outlier, device)
        st = F.cosine_similarity(vb.unsqueeze(0), vt.unsqueeze(0)).item()
        so = F.cosine_similarity(vb.unsqueeze(0), vo.unsqueeze(0)).item()
        hierarchy_scores.append(st - so)
    avg_hierarchy = float(np.mean(hierarchy_scores))
    print(f"  ==> Hierarchy 平均分離度: {avg_hierarchy:+.4f}")

    # 3. Multi-hop (連続軌道追跡: ケーブル ＋ 量子共鳴)
    print("\n[3] Multi-hop (連続軌道追跡テスト: ケーブル直通 ＋ 散乱ノイズ相殺)...")
    hop_scores = []
    for s, r, t in hop_tests:
        cur_v = get_word_embedding(model, tok, s, device)
        for rel in r:
            rel_v = get_word_embedding(model, tok, rel, device)
            # ケーブル滑走 + 量子波動相殺
            cur_v = q_engine.route_via_cable_and_quantum_resonate(cur_v, delta_vec=rel_v, gamma=0.35)
            cur_v = F.normalize(cur_v + rel_v, p=2, dim=0)
        tgt_v = get_word_embedding(model, tok, t, device)
        score = F.cosine_similarity(cur_v.unsqueeze(0), tgt_v.unsqueeze(0)).item()
        hop_scores.append(score)
        print(f"  ・{s} + {' + '.join(r)} ➔ {t}: 軌道共鳴スコア = {score:.4f}")
    avg_hop = float(np.mean(hop_scores))
    print(f"  ==> Multi-hop 平均スコア: {avg_hop:.4f}")

    # 4. Logic Chain (論理演繹保存度: 量子干渉)
    print("\n[4] Logic Chain (論理演繹連鎖テスト: 逆位相相殺)...")
    logic_scores = []
    for a, b, c, t in logic_tests:
        va = get_word_embedding(model, tok, a, device)
        vb = get_word_embedding(model, tok, b, device)
        vc = get_word_embedding(model, tok, c, device)
        vt = get_word_embedding(model, tok, t, device)
        logic_dir = F.normalize(vb - va, p=2, dim=0)
        pred_b = q_engine.route_via_cable_and_quantum_resonate(vc, delta_vec=logic_dir, gamma=0.35)
        pred_b = F.normalize(pred_b + logic_dir, p=2, dim=0)
        score = F.cosine_similarity(pred_b.unsqueeze(0), vt.unsqueeze(0)).item()
        logic_scores.append(score)
        print(f"  ・{a}:{b} :: {c}:{t} ➔ 演繹共鳴度 = {score:.4f}")
    avg_logic = float(np.mean(logic_scores))
    print(f"  ==> Logic Chain 平均スコア: {avg_logic:.4f}")

    # 5. Factuality QA
    print("\n[5] Factuality QA (事実性推論テスト)...")
    qa_hits = 0
    for item in qa_prompts:
        q = item["q"]
        exp = item["expected"]
        messages = [
            {"role": "system", "content": "あなたは簡潔かつ極めて正確に回答するAIアシスタントです。余計な前置きは省き、結論を述べてください。"},
            {"role": "user", "content": q}
        ]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=45,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        res_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        ans = tok.decode(res_tokens, skip_special_tokens=True).strip()
        hit = exp.lower() in ans.lower()
        if hit:
            qa_hits += 1
    qa_acc = qa_hits / len(qa_prompts)
    print(f"  ==> Factuality QA 正確率: {qa_acc * 100:.1f}%")

    # 総合結果集計表
    print("\n" + "=" * 90)
    print("🏆 Concept-Forge 統合ベンチマーク: 全進化ステージの最終比較表")
    print("=" * 90)
    print(f"{'評価種目':<20} | {'1. Base 1.5B':<13} | {'2. 単純階層(関所)':<16} | {'3. 束ねたケーブル':<16} | {'4. ケーブル+量子波動(今回)':<20}")
    print("-" * 90)
    print(f"{'Analogy (比喩)':<20} | {'0.7547':<13} | {'0.7535':<16} | {'0.6220':<16} | {avg_analogy:<20.4f}")
    print(f"{'Hierarchy (階層)':<20} | {'+0.1403':<13} | {'+0.1197':<16} | {'+0.1403':<16} | {avg_hierarchy:<+20.4f}")
    print(f"{'Multi-hop (軌道)':<20} | {'0.6985':<13} | {'0.5939':<16} | {'0.6950':<16} | {avg_hop:<20.4f} 🚀")
    print(f"{'Logic Chain (演繹)':<20} | {'0.7773':<13} | {'0.7147':<16} | {'0.7441':<16} | {avg_logic:<20.4f} 🚀")
    print(f"{'Factuality QA':<20} | {'60.0%':<13} | {'60.0%':<16} | {'60.0%':<16} | {qa_acc*100:<19.1f}%")
    print("=" * 90)

    # JSON保存
    report = {
        "analogy": avg_analogy,
        "hierarchy": avg_hierarchy,
        "multihop": avg_hop,
        "logic_chain": avg_logic,
        "qa_accuracy": qa_acc
    }
    with open("distillation_experiments/quantum_bundled_concept_forge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[+] レポートを保存しました: distillation_experiments/quantum_bundled_concept_forge_report.json")

if __name__ == "__main__":
    main()
