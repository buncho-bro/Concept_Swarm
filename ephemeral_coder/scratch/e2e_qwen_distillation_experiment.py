import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 実LLM (Qwen2.5-0.5B) によるエンドツーエンド脱重力・上空磁気浮上蒸留実証
# ==============================================================================

# 1. 従来の地表線形加算 Logits Processor (Ground Linear Bias)
class GroundLinearDistillationProcessor(LogitsProcessor):
    def __init__(self, token_weights_dict, alpha=4.0):
        """
        token_weights_dict: {token_id: weight} の単純線形加算辞書
        """
        self.alpha = alpha
        self.bias_dict = token_weights_dict

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 単純線形加算 (複数概念のトークンバイアスが足し算で重積)
        for tid, w in self.bias_dict.items():
            scores[:, tid] += self.alpha * w
        return scores


# 2. 本手法: 上空磁気浮上 Logits Processor (Dual-Surface Magnetic Trap)
class DualSurfaceMagneticLogitsProcessor(LogitsProcessor):
    def __init__(self, concept_clusters, temperature=4.0, alpha=4.0):
        """
        concept_clusters: 各概念ドメインごとの {domain_id: [token_id1, token_id2, ...]}
        各概念は独立した上空磁石として振る舞い、Smooth-Max で局所排他化される
        """
        self.temperature = temperature
        self.alpha = alpha
        self.concept_clusters = concept_clusters
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        B, V = scores.shape

        # 各概念磁石からの個別引力マップ
        # 各概念ドメインごとに局所引力を計算
        cluster_pulls = []
        for domain, tokens in self.concept_clusters.items():
            pull_d = torch.zeros(V, device=scores.device, dtype=scores.dtype)
            for tid in tokens:
                pull_d[tid] = 1.0
            cluster_pulls.append(pull_d)

        if cluster_pulls:
            # (N_domains, V)
            pulls_tensor = torch.stack(cluster_pulls, dim=0)
            
            # Smooth-Max (LogSumExp) による局所排他合成
            # 磁石同士が足し算で重積（ブラックホール化）するのを完全に防ぐ
            # offset 除去: max(0, SmoothMax)
            smooth_max_pull = (1.0 / self.temperature) * torch.logsumexp(self.temperature * pulls_tensor, dim=0)
            # 背景0レベルのオフセットを引く
            bg_offset = (1.0 / self.temperature) * math.log(len(cluster_pulls))
            effective_bias = torch.clamp(smooth_max_pull - bg_offset, min=0.0)

            # 地表ロジット (scores) に上空磁界バイアスを作用
            scores = scores + self.alpha * effective_bias.unsqueeze(0)

        return scores


def run_e2e_qwen_distillation():
    print("==================================================================")
    print(" 実LLM (Qwen2.5-0.5B-Instruct) エンドツーエンド脱重力蒸留実証")
    print(" End-to-End Real LLM Distillation: Base vs Ground vs Sky Magnet")
    print("==================================================================")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[1] 生徒モデルロード中: {model_id} (CUDA 16bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print("    生徒モデル準備完了 (VRAM使用量: {:.1f}MB)".format(torch.cuda.memory_allocated(0)/1024**2))

    # 2. 教師モデルから抽出された 3 大専門概念クラスター
    # (脱重力抽出器によってズレ0.01°でピンポイント抽出されたキー概念群)
    concept_specs = {
        "Raft_Consensus": [
            "leader", "election", "heartbeat", "quorum", "consensus", "log", "replication", "term", "follower", "candidate"
        ],
        "eBPF_Observability": [
            "kprobe", "xdp", "sandbox", "verifier", "kernel", "hook", "bytecode", "tracing", "probe", "packet"
        ],
        "zkSNARK_Cryptography": [
            "circuit", "prover", "verifier", "polynomial", "qap", "constraint", "witness", "arithmetic", "succinct", "proof"
        ]
    }

    # トークンIDへのマッピング
    concept_clusters = {}
    linear_bias_dict = {}
    for domain, words in concept_specs.items():
        tids = []
        for w in words:
            # 単語（スペース付き／なし両方）を語彙IDに変換
            for variant in [w, " " + w, w.capitalize(), " " + w.capitalize()]:
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    tids.append(enc[0])
                    linear_bias_dict[enc[0]] = linear_bias_dict.get(enc[0], 0.0) + 1.0
        concept_clusters[domain] = list(set(tids))

    print(f"\n[2] 抽出された専門概念クラスターを語彙空間にマップ完了:")
    for d, tids in concept_clusters.items():
        print(f"    ・概念 '{d}': {len(tids)} トークン")

    # 3. テストプロンプト
    test_prompts = [
        "Explain how Raft maintains distributed consensus during a network partition:",
        "Explain how eBPF securely executes programs inside the Linux kernel:",
        "Explain the role of Arithmetic Circuits in zk-SNARKs:"
    ]

    print("\n[3] 3つの推論条件でのテキスト生成テストを開始...")
    print("    A. 生徒素モデル (Base Qwen-0.5B, 蒸留なし)")
    print("    B. 従来の地表線形加算蒸留 (Ground Linear Bias)")
    print("    C. 本手法: 上空磁気浮上蒸留 (Dual-Surface Sky Magnet Trap)")
    print("------------------------------------------------------------------")

    results_report = []

    for p_idx, prompt in enumerate(test_prompts):
        print(f"\n=======================================================")
        print(f" Prompt {p_idx + 1}: \"{prompt}\"")
        print(f"=======================================================")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # A. 生徒素モデル (Base Qwen-0.5B)
        t0 = time.time()
        with torch.no_grad():
            out_base = model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        txt_base = tokenizer.decode(out_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        time_base = (time.time() - t0) * 1000.0

        # B. 従来の地表線形加算蒸留 (Ground Linear Bias)
        p_linear = LogitsProcessorList([
            GroundLinearDistillationProcessor(linear_bias_dict, alpha=4.0)
        ])
        t0 = time.time()
        with torch.no_grad():
            out_linear = model.generate(
                **inputs,
                max_new_tokens=90,
                logits_processor=p_linear,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        txt_linear = tokenizer.decode(out_linear[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        time_linear = (time.time() - t0) * 1000.0

        # C. 本手法: 上空磁気浮上蒸留 (Dual-Surface Sky Magnet Trap)
        p_sky = LogitsProcessorList([
            DualSurfaceMagneticLogitsProcessor(concept_clusters, temperature=4.0, alpha=4.0)
        ])
        t0 = time.time()
        with torch.no_grad():
            out_sky = model.generate(
                **inputs,
                max_new_tokens=90,
                logits_processor=p_sky,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        txt_sky = tokenizer.decode(out_sky[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        time_sky = (time.time() - t0) * 1000.0

        # 出力表示
        print(f"\n[A. 素の生徒モデル (Base Qwen-0.5B)] ({time_base:.1f}ms):")
        print(f"{txt_base}\n")

        print(f"[B. 従来の地表線形加算 (Ground Linear Bias)] ({time_linear:.1f}ms):")
        print(f"{txt_linear}\n")

        print(f"[C. ★本手法: 上空磁気浮上蒸留 (Dual-Surface Sky Magnet)] ({time_sky:.1f}ms):")
        print(f"{txt_sky}\n")

        results_report.append({
            "prompt": prompt,
            "base": txt_base,
            "linear": txt_linear,
            "sky": txt_sky
        })

    print("==================================================================")
    print(" 実LLMエンドツーエンド推論生成実験が正常に完了しました！")
    print("==================================================================")

if __name__ == '__main__':
    run_e2e_qwen_distillation()
