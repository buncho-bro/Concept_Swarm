import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 積層全大円篩エンジン (Cascaded Great Circle Filter Engine)
# 前層の吸引点を次層の初期位置として全大円干渉をリレーする純粋幾何学多層構造
# ※ 人間による「文章の指定・アンカー化」を完全に排除
# ==============================================================================

class CascadedGreatCircleLayer:
    """
    1つの階層（篩）を表す全大円干渉モジュール
    """
    def __init__(self, layer_id, name, basis_words, model, tokenizer, device):
        self.layer_id = layer_id
        self.name = name
        self.basis_words = basis_words
        self.device = device
        
        # 各層の基底概念ベクトル（単語埋め込みから純粋抽出）
        print(f"  [{name}] 基底概念ベクトル抽出中 ({len(basis_words)} 概念)...")
        vecs = []
        for w in basis_words:
            inp = tokenizer(w, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
                # 当該レイヤーの最終トークン表現を取得
                h = out.hidden_states[layer_id][0, -1, :].to(torch.float32)
                h = F.normalize(h, p=2, dim=-1)
                vecs.append(h)
        self.basis = torch.stack(vecs) # (K, D)
        
        # 分水嶺磁気浮上（全大円干渉の直交デカップリング）
        # 基底間の相互干渉を除去したGram-Schmidt直交空間
        U, S, V = torch.svd(self.basis.T)
        self.orthogonal_basis = U[:, :len(basis_words)].T # (K, D)

    def step_geodesic_interference(self, v_init, n_steps=3, eta=0.15, sigma=0.35):
        """
        全大円干渉による測地線滑走：
        初期位置 v_init から、基底概念群との引力・反発の全大円干渉を経て
        この階層における「吸引点 (Attractor)」へと収束させる
        """
        v = F.normalize(v_init.to(torch.float32), p=2, dim=-1)
        
        trajectory = [v.clone()]
        for step in range(n_steps):
            # 1. 各基底概念とのコサイン類似度（球面角）
            cos_theta = torch.matmul(v, self.orthogonal_basis.T) # (K,)
            cos_theta = torch.clamp(cos_theta, -0.999, 0.999)
            
            # 2. 測地線（大円）接線方向ベクトル
            sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
            tangents = (self.orthogonal_basis - cos_theta.unsqueeze(-1) * v.unsqueeze(0)) / sin_theta.unsqueeze(-1)
            
            # 3. 磁気浮上引力（距離に応じた選択的吸引：ゲート付き）
            dist_sq = 2.0 - 2.0 * cos_theta
            pulls = torch.exp(- dist_sq / (2.0 * (sigma ** 2)))
            
            # 類似度が低い基底には引力を働かせない（自然な思考を歪めない）
            mask = cos_theta > 0.05
            pulls = pulls * mask.float()
            
            if torch.sum(pulls) > 1e-6:
                weights = pulls / torch.sum(pulls)
                # 全大円干渉ベクトル
                interference_dir = torch.sum(weights.unsqueeze(-1) * tangents, dim=0)
                interference_dir = F.normalize(interference_dir, p=2, dim=-1)
                
                # 測地線に沿ったステップ移動
                v = F.normalize(v + eta * interference_dir, p=2, dim=-1)
            
            trajectory.append(v.clone())
            
        attractor = v
        return attractor, trajectory


class CascadedGreatCircleHook:
    """
    実LLM（Qwen）の DecoderLayer MLP 出力にフックし、
    前層の吸引点を次層の初期位置として全大円干渉を同期させるHook
    """
    def __init__(self, circle_layer, gamma=0.06):
        self.circle_layer = circle_layer
        self.gamma = gamma
        self.active = True
        self.last_init = None
        self.last_attractor = None

    def __call__(self, module, inputs, output):
        if not self.active:
            return output
            
        # output: (B, seq_len, D)
        last_h = output[:, -1, :] # (B, D)
        h_norm = F.normalize(last_h, p=2, dim=-1)
        
        # 現在の思考ベクトルを初期位置として全大円干渉
        attractor, _ = self.circle_layer.step_geodesic_interference(h_norm[0], n_steps=2, eta=0.12)
        attractor = attractor.to(last_h.dtype).unsqueeze(0)
        
        self.last_init = h_norm.clone()
        self.last_attractor = attractor.clone()
        
        # 接平面射影
        tangent_nudge = attractor - torch.sum(attractor * h_norm, dim=-1, keepdim=True) * h_norm
        tangent_nudge = F.normalize(tangent_nudge, p=2, dim=-1)
        
        # 微小な磁気浮上介入（前層から引き継いだ滑走方向を尊重）
        h_mag = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        nudged = last_h + self.gamma * h_mag * tangent_nudge
        
        new_output = output.clone()
        new_output[:, -1, :] = nudged
        return new_output


def run_cascaded_experiment():
    print("=" * 80)
    print(" 【積層全大円干渉エンジン 実証実験】")
    print(" 前層の吸引点 ➔ 次層の初期位置 リレーによる自律的多段階推論")
    print(" （※文章指定・アンカー化を完全に排除した純粋幾何学方式）")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    device = model.device

    # レイヤー位置: 低次(Layer 6), 中次(Layer 12), 高次(Layer 18)
    l_low, l_mid, l_high = 6, 12, 18

    # 普遍的な純粋概念基底（文章ではなく、意味空間の座標軸）
    # 人為的なバイアスを排した言語の基礎次元
    basis_layer_low = [
        "Entity", "Object", "Number", "Name", "Place", "Time", "Property", "State", "Definition", "Data"
    ]
    basis_layer_mid = [
        "Cause", "Effect", "Condition", "Relation", "Difference", "Mechanism", "Process", "Rule", "Change"
    ]
    basis_layer_high = [
        "Conclusion", "Principle", "System", "Truth", "Consistency", "Summary", "Invariance", "Resolution"
    ]

    print("\n[1] 3層の全大円干渉モジュールを構築中...")
    circle_l6 = CascadedGreatCircleLayer(l_low, "Layer 6 (低次・事実/要素の篩)", basis_layer_low, model, tokenizer, device)
    circle_l12 = CascadedGreatCircleLayer(l_mid, "Layer 12 (中次・因果/関係の篩)", basis_layer_mid, model, tokenizer, device)
    circle_l18 = CascadedGreatCircleLayer(l_high, "Layer 18 (高次・統合/結論の篩)", basis_layer_high, model, tokenizer, device)

    # 1. まず純粋な幾何学的リレーのシミュレーション（初期位置 ➔ L6吸引点 ➔ L12吸引点 ➔ L18吸引点）
    print("\n[2] 純粋幾何学シミュレーション: 「前層の吸引点 ➔ 次層の初期位置」")
    test_queries = [
        "日本の首都は東京です。",
        "オーストラリアの首都はキャンベラです。",
        "3 + 5 * 4 = 23",
        "2相コミットプロトコルにおけるコーディネーターの障害ブロック"
    ]

    for q in test_queries:
        print(f"\n◆ 入力: \"{q}\"")
        inp = tokenizer(q, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            # 入力文そのものの埋め込み
            v_start = out.hidden_states[0][0, -1, :].to(torch.float32)
            v_start = F.normalize(v_start, p=2, dim=-1)

        # 第1層 (L6): 初期位置 = 入力ベクトル
        attractor_l6, traj_l6 = circle_l6.step_geodesic_interference(v_start, n_steps=3)
        shift_l6 = math.acos(float(torch.clamp(torch.dot(v_start, attractor_l6), -1.0, 1.0))) * 180.0 / math.pi

        # 第2層 (L12): 初期位置 = 第1層の吸引点！
        attractor_l12, traj_l12 = circle_l12.step_geodesic_interference(attractor_l6, n_steps=3)
        shift_l12 = math.acos(float(torch.clamp(torch.dot(attractor_l6, attractor_l12), -1.0, 1.0))) * 180.0 / math.pi

        # 第3層 (L18): 初期位置 = 第2層の吸引点！
        attractor_l18, traj_l18 = circle_l18.step_geodesic_interference(attractor_l12, n_steps=3)
        shift_l18 = math.acos(float(torch.clamp(torch.dot(attractor_l12, attractor_l18), -1.0, 1.0))) * 180.0 / math.pi

        total_shift = math.acos(float(torch.clamp(torch.dot(v_start, attractor_l18), -1.0, 1.0))) * 180.0 / math.pi
        print(f"  ・Layer 6 収束 (要素の篩):    変位角 {shift_l6:.2f}° (初期位置 ➔ L6吸引点)")
        print(f"  ・Layer 12 収束 (関係の篩):   変位角 {shift_l12:.2f}° (L6吸引点を初期位置に ➔ L12吸引点)")
        print(f"  ・Layer 18 収束 (統合の篩):   変位角 {shift_l18:.2f}° (L12吸引点を初期位置に ➔ L18吸引点)")
        print(f"  ・累計思考滑走距離 (大円角): {total_shift:.2f}° (文章による拘束なし)")

    # 2. 実生成ベンチマーク（直前の問題で正確性と論理性を比較）
    print("\n[3] 実LLM生成ベンチマーク（素のモデル vs 以前の文章固定積層 vs 新・積層全大円篩）")

    # フックの登録
    hook_l6 = CascadedGreatCircleHook(circle_l6, gamma=0.06)
    hook_l12 = CascadedGreatCircleHook(circle_l12, gamma=0.06)
    hook_l18 = CascadedGreatCircleHook(circle_l18, gamma=0.06)

    handles = [
        model.model.layers[l_low].mlp.register_forward_hook(hook_l6),
        model.model.layers[l_mid].mlp.register_forward_hook(hook_l12),
        model.model.layers[l_high].mlp.register_forward_hook(hook_l18)
    ]

    benchmark_queries = [
        {
            "id": "Math_Order",
            "q": "3 + 5 × 4 の計算結果を数字のみで答えてください。",
            "correct": "23"
        },
        {
            "id": "CS_2PC",
            "q": "2相コミット（2PC）プロトコルで、コーディネーターがPrepared受信後にクラッシュした際の最大の問題は何ですか？",
            "correct": "ブロック"
        },
        {
            "id": "Attention_Scale",
            "q": "TransformerのSelf-Attentionで内積を√d_kで割る数学的理由を簡潔に答えてください。",
            "correct": "分散"
        }
    ]

    for bq in benchmark_queries:
        print(f"\n--------------------------------------------------------------")
        print(f"◆ 課題: {bq['q']} (正解キーワード: {bq['correct']})")
        inp = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": bq["q"]}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)

        # A. 素のモデル (無介入)
        hook_l6.active = False; hook_l12.active = False; hook_l18.active = False
        torch.manual_seed(42)
        with torch.no_grad():
            out_base = model.generate(**inp, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        base_resp = tokenizer.decode(out_base[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # B. 新・積層全大円篩 (Cascaded Great Circle)
        hook_l6.active = True; hook_l12.active = True; hook_l18.active = True
        torch.manual_seed(42)
        with torch.no_grad():
            out_cascaded = model.generate(**inp, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        cascaded_resp = tokenizer.decode(out_cascaded[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        print(f"  [A. Base Model (素)]:")
        print(f"      {repr(base_resp[:80])}")
        print(f"  [B. 新・積層全大円篩 (Cascaded)]: (前層吸引点 ➔ 次層初期位置)")
        print(f"      {repr(cascaded_resp[:80])}")

    for h in handles:
        h.remove()

if __name__ == "__main__":
    run_cascaded_experiment()
