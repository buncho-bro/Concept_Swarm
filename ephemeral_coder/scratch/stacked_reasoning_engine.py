import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 積層地形による多段階推論エンジン (Stacked Potential Terrains Engine)
# ==============================================================================

class StackedTerrainLayer:
    """
    1つの階層におけるポテンシャル場（地表基幹アンカー ＋ 上空磁石）
    """
    def __init__(self, name, dim=384, temperature=5.0):
        self.name = name
        self.dim = dim
        self.temperature = temperature
        self.anchors = []   # (D,)
        self.weights = []
        self.sigmas = []
        self.labels = []
        self._tensor = None

    def add_attractor(self, vec, label="concept", weight=1.0, sigma=0.25):
        v = vec / torch.norm(vec)
        self.anchors.append(v.clone().detach())
        self.weights.append(float(weight))
        self.sigmas.append(float(sigma))
        self.labels.append(label)
        self._tensor = torch.stack(self.anchors)

    def potential(self, x):
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        cos_sims = torch.matmul(x_norm, self._tensor.T)
        dist_sq = 2.0 - 2.0 * cos_sims
        w_t = torch.tensor(self.weights, dtype=x.dtype, device=x.device)
        s_t = torch.tensor(self.sigmas, dtype=x.dtype, device=x.device)
        pulls = w_t * torch.exp(- dist_sq / (2.0 * s_t**2))
        return - (1.0 / self.temperature) * torch.logsumexp(self.temperature * pulls, dim=-1)

    def roll_marble(self, start_pos, steps=60, lr=0.04, momentum=0.7):
        pos = (start_pos / torch.norm(start_pos)).clone().detach().requires_grad_(True)
        v = torch.zeros_like(pos)
        for _ in range(steps):
            U = self.potential(pos.unsqueeze(0))
            grad = torch.autograd.grad(U, pos)[0]
            proj_grad = grad - torch.dot(pos, grad) * pos
            v = momentum * v - lr * proj_grad
            pos_new = pos + v
            pos = (pos_new / torch.norm(pos_new)).detach().requires_grad_(True)
            if torch.norm(v) < 1e-4:
                break
        return pos.detach()


class StackedReasoningEngine:
    """
    積層地形による多段階推論エンジン
    層間遷移（開始位置の決定）に3つのモードを搭載:
      1. 'direct_lift': 前層の収束点をそのまま次層の初期位置とする
      2. 'residual_leap': 前層収束点 + 初期プロンプトの測地線残差接続
      3. 'orthogonal_kick': 前層で解決した成分を消去し、未解決成分を推進力とする
    """
    def __init__(self, embedder, dim=384):
        self.embedder = embedder
        self.dim = dim
        self.layers = [] # List of StackedTerrainLayer

    def add_layer(self, name):
        layer = StackedTerrainLayer(name=name, dim=self.dim)
        self.layers.append(layer)
        return layer

    def transition_next_start(self, prev_pos, initial_prompt_vec, mode="residual_leap", alpha=0.35):
        """
        次階層の開始位置 (初期座標) を計算
        """
        if mode == "direct_lift":
            # 方式1: 前層の浮上平衡点をそのまま次層の足場とする
            return prev_pos / torch.norm(prev_pos)

        elif mode == "residual_leap":
            # 方式2: トランスフォーマーの残差接続に相当 (前層結論 + 元プロンプトの慣性)
            combined = prev_pos + alpha * initial_prompt_vec
            return combined / torch.norm(combined)

        elif mode == "orthogonal_kick":
            # 方式3: 前層で解決した成分を射影消去し、直交する未解決残差を励起
            proj_component = torch.dot(initial_prompt_vec, prev_pos) * prev_pos
            unsolved_residual = initial_prompt_vec - proj_component
            combined = prev_pos + 0.4 * unsolved_residual
            return combined / torch.norm(combined)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run_multistep_reasoning(self, prompt_text, transition_mode="residual_leap"):
        prompt_vec = self.embedder.encode(prompt_text, convert_to_tensor=True).cpu()
        prompt_vec = prompt_vec / torch.norm(prompt_vec)

        trajectory = []
        curr_start = prompt_vec.clone()

        for l_idx, layer in enumerate(self.layers):
            # 1. 現在の階層でポテンシャル滑走
            converged_pos = layer.roll_marble(curr_start, steps=70)

            # 最も近接したアトラクターの同定
            sims = torch.matmul(converged_pos, layer._tensor.T)
            best_idx = torch.argmax(sims).item()
            best_label = layer.labels[best_idx]
            best_sim = sims[best_idx].item()

            trajectory.append({
                "layer_idx": l_idx,
                "layer_name": layer.name,
                "start_sim_to_goal": torch.dot(curr_start, layer.anchors[best_idx]).item(),
                "converged_label": best_label,
                "converged_sim": best_sim,
                "converged_pos": converged_pos
            })

            # 2. 次階層が存在する場合、開始位置を遷移計算
            if l_idx + 1 < len(self.layers):
                curr_start = self.transition_next_start(
                    prev_pos=converged_pos,
                    initial_prompt_vec=prompt_vec,
                    mode=transition_mode,
                    alpha=0.35
                )

        return trajectory


def benchmark_stacked_reasoning():
    print("==================================================================")
    print(" 積層地形による多段階推論プロトタイプ検証 (0 VRAM Complex Reasoning)")
    print(" Stacked Potential Terrains: Layer-by-Layer Reasoning Dynamics")
    print("==================================================================")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    engine = StackedReasoningEngine(embedder, dim=384)

    # --- 3階層の推論地形を構築 ---
    # 第1層: 現象・即時反応（低次地表: 政策・直接的行動の地形）
    layer1 = engine.add_layer("Layer 1: Direct Policy Response (現象・即時反応)")
    layer1_concepts = [
        "Aggressive Central Bank Interest Rate Hike",
        "Fiscal Currency Intervention",
        "Emergency Supply Chain Subsidy",
        "Dovish Monetary Easing Expansion"
    ]
    for c in layer1_concepts:
        v = embedder.encode(c, convert_to_tensor=True).cpu()
        layer1.add_attractor(v, label=c, weight=1.0, sigma=0.25)

    # 第2層: 市場・波及効果（中次成層圏: 金融環境・二次的インパクトの地形）
    layer2 = engine.add_layer("Layer 2: Market Transmission (市場・波及効果)")
    layer2_concepts = [
        "Corporate Bond Yield Spike & Credit Spread Widening",
        "Equity Market Bullish Rally",
        "Consumer Spending Hyperinflation Boom",
        "Sovereign Debt Default Crisis"
    ]
    for c in layer2_concepts:
        v = embedder.encode(c, convert_to_tensor=True).cpu()
        layer2.add_attractor(v, label=c, weight=1.0, sigma=0.25)

    # 第3層: 最終帰結・マクロ均衡（高次熱圏: 最終的な経済帰結の地形）
    layer3 = engine.add_layer("Layer 3: Macroeconomic Equilibrium (最終帰結・均衡)")
    layer3_concepts = [
        "Stagflationary Recession & Demand Destruction",
        "Economic Soft Landing & Sustainable Growth",
        "Hyperinflationary Currency Collapse",
        "Technological Productivity Miracle"
    ]
    for c in layer3_concepts:
        v = embedder.encode(c, convert_to_tensor=True).cpu()
        layer3.add_attractor(v, label=c, weight=1.0, sigma=0.25)

    # 複雑な問い（推論問題）
    prompt = "Severe Stagflationary Pressure: Rapidly accelerating CPI inflation combined with global energy supply chain disruption"
    print(f"\n[入力プロンプト (問い)]: \"{prompt}\"")

    # 3つの層間遷移モードを比較
    modes = ["direct_lift", "residual_leap", "orthogonal_kick"]

    for mode in modes:
        print(f"\n=======================================================")
        print(f" 検証モード: {mode.upper()}")
        print(f"=======================================================")
        traj = engine.run_multistep_reasoning(prompt, transition_mode=mode)
        for step in traj:
            l_idx = step["layer_idx"] + 1
            name = step["layer_name"]
            label = step["converged_label"]
            sim = step["converged_sim"]
            start_sim = step["start_sim_to_goal"]
            print(f"  [Layer {l_idx}] {name}")
            print(f"    -> 初期位置から目標への近接度: {start_sim:.3f}")
            print(f"    -> 思考滑走後の到達結論: 『{label}』 (確信度 cos={sim:.4f})")

if __name__ == '__main__':
    benchmark_stacked_reasoning()
